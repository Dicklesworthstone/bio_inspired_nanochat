"""
FlexAttention presyn path migrated to the canonical formulation — bead s3w9.

flex_synaptic.py was a 3rd divergent presyn impl (sigmoid mix + raw AMP). It now uses the SAME
faithful formulation as the live standard path's release_canonical: Hill Syt + Doc2 + complexin/
SNARE fuse for the per-key readiness, and an energy-gated AMPA amplitude. These tests lock the
parity (and that AMP is no longer read).

Run:  pytest tests/test_flex_canonical.py -v
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from _bio_testkit import set_seed

from bio_inspired_nanochat.flex_synaptic import SynapticFlexAttention
from bio_inspired_nanochat.synaptic import (
    SynapticConfig,
    SynapticPresyn,
    build_presyn_state,
)

DEV = torch.device("cpu")
DT = torch.float32


def _benchmark_namespace(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    benchmark = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_flex.py"
    monkeypatch.setattr(sys, "argv", [str(benchmark)])
    return runpy.run_path(str(benchmark), run_name="_benchmark_flex_test")


@pytest.mark.unit
def test_attention_benchmark_preflight_fails_closed(monkeypatch: pytest.MonkeyPatch):
    namespace = _benchmark_namespace(monkeypatch)
    preflight = namespace["_attention_preflight"]

    assert "CUDA is unavailable" in preflight(
        cuda_available=False, device_name=None, require_4090=True
    )
    assert "requires an RTX 4090" in preflight(
        cuda_available=True, device_name="NVIDIA H100", require_4090=True
    )
    assert (
        preflight(
            cuda_available=True,
            device_name="NVIDIA GeForce RTX 4090",
            require_4090=True,
        )
        is None
    )


@pytest.mark.unit
def test_attention_benchmark_case_matrix_and_recommendation(
    monkeypatch: pytest.MonkeyPatch,
):
    namespace = _benchmark_namespace(monkeypatch)
    parse_cases = namespace["_parse_attention_cases"]
    recommend = namespace["_recommend_backend"]

    cases = parse_cases("vanilla_sdpa_auto,synaptic_dense,synaptic_flex")
    assert [case.name for case in cases] == [
        "vanilla_sdpa_auto",
        "synaptic_dense",
        "synaptic_flex",
    ]
    with pytest.raises(ValueError, match="Unknown attention case"):
        parse_cases("imaginary_backend")
    with pytest.raises(ValueError, match="duplicates"):
        parse_cases("synaptic_dense,synaptic_dense")

    rows = [
        {
            "case": "synaptic_dense",
            "model_path": "synaptic",
            "status": "passed",
            "tokens_per_second": 100.0,
        },
        {
            "case": "synaptic_flex",
            "model_path": "synaptic",
            "status": "passed",
            "tokens_per_second": 125.0,
        },
    ]
    recommendation = recommend(rows, model_path="synaptic", baseline="synaptic_dense")
    assert recommendation == {
        "model_path": "synaptic",
        "recommended_case": "synaptic_flex",
        "baseline_case": "synaptic_dense",
        "throughput_delta_percent": 25.0,
    }


@pytest.mark.unit
def test_flex_precompute_matches_canonical_release_prob():
    set_seed(0)
    cfg = SynapticConfig(enable_presyn=True)
    pre = SynapticPresyn(16, cfg)
    flex = SynapticFlexAttention(cfg)
    state = build_presyn_state(1, 4, 2, DEV, DT, cfg)
    with torch.no_grad():
        state["C"].copy_(torch.rand_like(state["C"]) * 2.0)  # vary calcium so the Hill is exercised

    kf, qamp = flex.precompute_bio_factors(state, cfg)

    c, pr, cl = state["C"], state["PR"], state["CL"]
    # canonical fuse_base == _faithful_release_prob with the bilinear driven to 1 (large drive)
    p_fuse = pre._faithful_release_prob(c, pr, cl, torch.full_like(c, 50.0))
    fuse_flex = kf / state["RRP"]
    assert torch.allclose(fuse_flex, p_fuse, atol=1e-4), "flex fuse_base must match the canonical Hill/fuse"

    qamp_expected = torch.sigmoid(cfg.q_beta * (state["E"] - 0.5)) * cfg.qmax
    assert torch.allclose(qamp, qamp_expected, atol=1e-6), "flex qamp must be the energy-gated amplitude"


@pytest.mark.unit
def test_flex_no_longer_reads_amp():
    # AMP is superseded by energy->qamp; precompute must not depend on state["AMP"].
    set_seed(0)
    cfg = SynapticConfig(enable_presyn=True)
    flex = SynapticFlexAttention(cfg)
    state = build_presyn_state(1, 4, 2, DEV, DT, cfg)
    kf0, q0 = flex.precompute_bio_factors(state, cfg)
    with torch.no_grad():
        state["AMP"].fill_(99.0)  # corrupt AMP
    kf1, q1 = flex.precompute_bio_factors(state, cfg)
    assert torch.equal(kf0, kf1) and torch.equal(q0, q1), "flex must not read AMP anymore"


@pytest.mark.unit
def test_flex_forward_runs_finite_if_available():
    # Best-effort end-to-end smoke; flex_attention may require compilation/hardware not present.
    cfg = SynapticConfig(enable_presyn=True)
    flex = SynapticFlexAttention(cfg)
    B, H, T, D = 1, 2, 8, 16
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    state = build_presyn_state(B, T, H, DEV, DT, cfg)
    try:
        out = flex(q, k, v, state, block_mask=None)
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - environment-dependent
        pytest.skip(f"flex_attention unavailable in this environment: {exc}")
    assert out.shape == (B, H, T, D)
    assert torch.isfinite(out).all(), "flex attention output must be finite"
