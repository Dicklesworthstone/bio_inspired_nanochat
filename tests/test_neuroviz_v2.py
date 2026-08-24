"""Unit tests for the NeuroViz v2 bio-panel suite (bead dow).

Locks:
  1. ``collect_bio_panels`` extracts finite postsynaptic metrics (fast/slow weight norms,
     CaMKII/PP1/BDNF means, latch fraction) from a live tiny synaptic model.
  2. With ``probe_batch``, the per-position presynaptic C/RRP/E trace comes back from the
     non-mutating KV-cache forward — and the probe does NOT mutate consolidation state
     (CaMKII/BDNF unchanged across the call).
  3. ``render_html_dashboard`` writes self-contained HTML (sparkline SVG + section markers).
  4. ``write_bio_dashboard`` persists BOTH the JSON sidecar and the HTML artifact.
  5. Dense models yield an empty MoE panel (no fictitious expert metrics).

Run:  pytest tests/test_neuroviz_v2.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bio_testkit import make_tiny_synaptic, random_tokens, set_seed

from bio_inspired_nanochat.neuroviz import (
    collect_bio_panels,
    render_html_dashboard,
    write_bio_dashboard,
)

pytestmark = pytest.mark.unit


def _model():
    set_seed(0)
    return make_tiny_synaptic(seed=0)


def test_collect_panels_postsyn_metrics_finite():
    model = _model()
    p = collect_bio_panels(model)
    assert "postsyn" in p and p["postsyn"], "dense synaptic model must expose postsyn panel"
    post = p["postsyn"]
    for key in (
        "fast_weight_norm_mean", "fast_weight_norm_max", "slow_weight_norm_mean",
        "camkii_mean", "pp1_mean", "bdnf_mean", "latched_frac",
    ):
        assert key in post, f"missing {key}"
        assert isinstance(post[key], float) and post[key] == post[key], f"{key} not finite"
    assert 0.0 <= post["latched_frac"] <= 1.0
    assert post["n_postsyn_modules"] >= 1


def test_dense_model_has_empty_moe_panel():
    model = _model()
    p = collect_bio_panels(model)
    assert p["moe"] == {}, "a dense model must not fabricate expert metrics"


def test_presyn_probe_returns_curves_without_mutating_consolidation():
    model = _model().eval()
    posts = [m for m in model.modules() if type(m).__name__ == "PostsynapticHebb"]
    before = [(m.camkii.detach().clone(), m.pp1.detach().clone()) for m in posts]
    batch = random_tokens(batch=1, seq=16)
    p = collect_bio_panels(model, probe_batch=batch)
    tr = p.get("presyn_trace") or {}
    assert set(("C", "RRP", "E")) <= set(tr), f"presyn trace missing keys: {list(tr)}"
    for key, series in tr.items():
        assert len(series) == batch.shape[1], f"{key} must be per-position"
        assert all(x == x for x in series), f"{key} contains NaN"
    # The probe contract: no consolidation-state mutation.
    for (camkii_before, pp1_before), m in zip(before, posts):
        assert torch.equal(camkii_before, m.camkii), "probe mutated CaMKII"

def test_render_html_dashboard_writes_selfcontained_file(tmp_path):
    model = _model()
    panels = collect_bio_panels(model, probe_batch=random_tokens(batch=1, seq=12))
    out = str(tmp_path / "dash.html")
    written = render_html_dashboard(panels, out, title="unit-test dashboard")
    assert Path(written) == Path(out) and Path(out).exists()
    html = Path(out).read_text(encoding="utf-8")
    assert "<svg" in html, "sparklines must be inline SVG"
    assert "Postsynaptic plasticity" in html
    assert "unit-test dashboard" in html


def test_write_bio_dashboard_persists_both_artifacts(tmp_path):
    model = _model()
    paths = write_bio_dashboard(
        model, step=42, probe_batch=random_tokens(batch=1, seq=8),
        log_dir=str(tmp_path / "nv"),
    )
    jp, hp = Path(paths["json"]), Path(paths["html"])
    assert jp.exists() and hp.exists()
    payload = json.loads(jp.read_text())
    assert "postsyn" in payload
    assert "bio_dashboard_000000042" in hp.name


def test_lineage_book_persists_events(tmp_path):
    from bio_inspired_nanochat.neuroviz import LineageBook

    lb = LineageBook(str(tmp_path / "lineage"))
    lb.log_split("moe_L0", step=10, parent_idx=2, child_idx=5)
    lb.log_merge("moe_L0", step=20, parent_i=0, parent_j=3, child_idx=7)
    f = tmp_path / "lineage" / "moe_L0_lineage.json"
    assert f.exists()
    events = json.loads(f.read_text())
    assert events[0] == [10, "split", [2, 5]]
    assert events[1] == [20, "merge", [0, 3, 7]]
