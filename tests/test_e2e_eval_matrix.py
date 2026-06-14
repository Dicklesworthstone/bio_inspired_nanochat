"""E2E: eval_matrix bio-vs-vanilla on synthetic data (bead `eqyk.6`).

Runs the standardized evaluation harness (``scripts/eval_matrix.py``) end-to-end on SYNTHETIC data
for 2 presets × 2 seeds and verifies the **pipeline** (not the scientific result):

  1. **Schema-valid RunRecords** — the run completes for every (preset, seed) cell and writes one
     record per run whose columns exactly match the declared ``SUMMARY_FIELDS`` schema, with the
     required scalar fields present and parseable.
  2. **Artifacts** — ``summary.csv`` + ``summary.jsonl`` (mirroring each other) plus, per run, the
     detailed ``run_config.jsonl`` and ``train_metrics.jsonl`` logs.
  3. **Stats layer** — the harness output feeds the ``eval_stats`` layer: a paired test + bootstrap
     CI and the multi-preset ``compare_matrix`` report compute over matched seeds, and the loader's
     non-finite filtering behaves on a real quality metric.

Metric note: the bio synaptic stack produces NaN ``val_loss`` on this degenerate synthetic ramp
(every bio preset; tracked as ``809i``), so the paired test here runs on a **finite** per-run
metric (``tok_per_sec``) — the point of eqyk.6 is to exercise the harness→stats pipeline, not to
land a quality verdict. We additionally assert the loader keeps vanilla's finite ``val_loss`` and
never leaks a non-finite value (its NaN-filtering contract on real, imperfect harness output).

Run:  pytest tests/test_e2e_eval_matrix.py -v
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

import pytest

from bio_inspired_nanochat.eval_stats import compare_matrix, load_matrix_csv, paired_comparison
from scripts.eval_matrix import SUMMARY_FIELDS

pytestmark = pytest.mark.e2e

PRESETS = ("vanilla", "bio_all")
SEEDS = (1337, 1338)
# Required scalar columns that must be present + non-empty for an `ok` run (a subset of the
# full SUMMARY_FIELDS schema; the quality metrics like val_bpb/core_metric are legitimately blank
# on synthetic data, and val_loss/val_ppl are present-but-"nan" for the bio presets — see 809i).
_REQUIRED_FIELDS = (
    "run_id", "preset", "seed", "data", "device_type", "sequence_len", "vocab_size",
    "n_layer", "n_head", "n_embd", "train_tokens_requested", "train_tokens_processed",
    "steps", "walltime_sec", "tok_per_sec", "val_loss", "val_ppl",
)


def _run_matrix(out_dir: Path) -> Path:
    """Invoke the real eval_matrix CLI in-process (patch argv → main) on a tiny synthetic matrix."""
    import torch

    try:
        torch.set_num_threads(min(4, os.cpu_count() or 4))
    except Exception:
        pass
    argv = [
        "eval_matrix", "matrix",
        "--presets", ",".join(PRESETS),
        "--seeds", ",".join(str(s) for s in SEEDS),
        "--train-tokens", "2048", "--eval-tokens", "512",
        "--data", "synthetic", "--device-type", "cpu",
        "--sequence-len", "64", "--vocab-size", "256",
        "--n-layer", "2", "--n-head", "2", "--n-embd", "64",
        "--device-batch-size", "1", "--total-batch-size-tokens", "64",
        "--ece-bins", "5", "--niah-lengths", "7",  # "7" < min NIAH length → NIAH skipped (fast)
        "--embedding-lr", "0.02", "--unembedding-lr", "0.004", "--matrix-lr", "0.01",
        "--out-dir", str(out_dir), "--batch-id", "test",
    ]
    from scripts.eval_matrix import main

    old_argv = sys.argv
    sys.argv = argv
    try:
        rc = main()
    finally:
        sys.argv = old_argv
    assert rc == 0, "eval_matrix matrix CLI returned non-zero"
    return out_dir / "test"


@pytest.fixture(scope="module")
def matrix_dir(tmp_path_factory) -> Path:
    """Run the (slowest) matrix ONCE and share the output dir across the pipeline + stats tests."""
    out = tmp_path_factory.mktemp("eval_matrix")
    return _run_matrix(out)


def _read_csv(matrix_dir: Path) -> tuple[list[str], list[dict]]:
    with (matrix_dir / "summary.csv").open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


# --------------------------------------------------------------------------- #
# 1. pipeline: schema-valid RunRecords + artifacts
# --------------------------------------------------------------------------- #
def test_eval_matrix_e2e_synthetic_pipeline(matrix_dir):
    csv_path = matrix_dir / "summary.csv"
    jsonl_path = matrix_dir / "summary.jsonl"
    assert csv_path.exists() and jsonl_path.exists(), "harness must write summary.csv + summary.jsonl"

    header, rows = _read_csv(matrix_dir)
    # one schema-valid record per (preset, seed) cell, all succeeded
    assert len(rows) == len(PRESETS) * len(SEEDS)
    assert all(r["status"] == "ok" for r in rows), [r.get("error") for r in rows if r["status"] != "ok"]
    assert {(r["preset"], int(r["seed"])) for r in rows} == {(p, s) for p in PRESETS for s in SEEDS}

    # schema: CSV columns are exactly the declared SUMMARY_FIELDS; required scalars present+typed
    assert set(header) == set(SUMMARY_FIELDS), "CSV header must match the declared RunRecord schema"
    for r in rows:
        for field in _REQUIRED_FIELDS:
            assert r.get(field, "") != "", f"required field {field!r} missing/empty in {r['run_id']}"
        # required numeric fields parse as their declared types
        for int_field in ("seed", "train_tokens_processed", "steps"):
            int(r[int_field])
        assert math.isfinite(float(r["walltime_sec"])) and math.isfinite(float(r["tok_per_sec"]))

    # JSONL mirrors the CSV cells
    jl = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(jl) == len(rows)
    assert {(d["preset"], int(d["seed"])) for d in jl} == {(p, s) for p in PRESETS for s in SEEDS}

    # per-run detailed logs (the human-inspectable artifacts)
    run_dirs = [d for d in matrix_dir.iterdir() if d.is_dir()]
    assert len(run_dirs) == len(rows)
    for d in run_dirs:
        assert (d / "run_config.jsonl").exists(), f"missing run_config.jsonl in {d.name}"
        assert (d / "train_metrics.jsonl").exists(), f"missing train_metrics.jsonl in {d.name}"


# --------------------------------------------------------------------------- #
# 2. stats layer: paired test + CI computed on the harness output
# --------------------------------------------------------------------------- #
def test_eval_matrix_stats_layer_on_output(matrix_dir):
    csv_path = matrix_dir / "summary.csv"

    # tok_per_sec is finite for every run (a quality metric is NOT — see 809i), so we exercise the
    # paired test + CI on it: the pipeline (harness output → stats), not the scientific result.
    data = load_matrix_csv(csv_path, "tok_per_sec")
    assert set(data) == set(PRESETS)
    assert all(sorted(data[p]) == list(SEEDS) for p in PRESETS)

    paired = paired_comparison(data["bio_all"], data["vanilla"], lower_is_better=False)
    assert paired is not None, "expected a paired result with 2 shared seeds"
    assert paired.n_pairs == len(SEEDS)
    assert math.isfinite(paired.mean_delta)
    assert math.isfinite(paired.delta_ci_low) and math.isfinite(paired.delta_ci_high)
    assert paired.delta_ci_low <= paired.mean_delta <= paired.delta_ci_high
    assert isinstance(paired.t_p_value, float) and isinstance(paired.wilcoxon_p_value, float)

    # multi-preset report: aggregate per preset + paired-vs-baseline for non-baseline presets
    rep = compare_matrix(data, baseline="vanilla", metric="tok_per_sec", lower_is_better=False)
    assert set(rep["presets"]) == set(PRESETS)
    assert rep["presets"]["vanilla"]["aggregate"]["n"] == len(SEEDS)
    assert "paired_vs_baseline" not in rep["presets"]["vanilla"], "baseline has no self-comparison"
    bio = rep["presets"]["bio_all"]
    assert bio["aggregate"]["n"] == len(SEEDS)
    assert "paired_vs_baseline" in bio
    for key in ("n_pairs", "mean_delta", "delta_ci_low", "delta_ci_high", "t_p_value", "wilcoxon_p_value"):
        assert key in bio["paired_vs_baseline"]


# --------------------------------------------------------------------------- #
# 3. stats layer is robust to the harness's non-finite quality metrics
# --------------------------------------------------------------------------- #
def test_eval_matrix_stats_filters_nonfinite_quality_metric(matrix_dir):
    # load_matrix_csv must keep vanilla's finite val_loss (both seeds) and never leak a non-finite
    # value — the bio presets' NaN val_loss (809i) is filtered, so downstream stats stay well-defined.
    data = load_matrix_csv(matrix_dir / "summary.csv", "val_loss")
    assert sorted(data.get("vanilla", {})) == list(SEEDS), "vanilla val_loss must be finite for both seeds"
    assert all(math.isfinite(v) for by_seed in data.values() for v in by_seed.values())
