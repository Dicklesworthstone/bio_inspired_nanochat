"""E2E: eval_matrix bio-vs-vanilla on synthetic data (bead `eqyk.6`).

Runs the standardized evaluation harness (``scripts/eval_matrix.py``) end-to-end on SYNTHETIC data
for 2 presets × 2 seeds and verifies the **pipeline** (not the scientific result):

  1. **Schema-valid RunRecords** — the run completes for every (preset, seed) cell and writes one
     record per run whose columns exactly match the declared ``SUMMARY_FIELDS`` schema, with the
     required scalar fields present and parseable.
  2. **Artifacts** — ``summary.csv`` + ``summary.jsonl`` (mirroring each other) plus, per run, the
     detailed ``run_config.jsonl`` and ``train_metrics.jsonl`` logs.
  3. **Stats layer** — finite ``val_loss`` from every model family feeds a paired test + bootstrap
     CI and the multi-preset ``compare_matrix`` report over matched seeds.

The synthetic ramp is also a numerical smoke test for the bio stack: every preset must produce a
finite ``val_loss``. In particular, this locks the ``809i`` fix for meta-device initialization of
the fixed Hebbian projection buffers, which previously poisoned the deferred online update before
evaluation. The paired statistics therefore exercise a real quality metric again.

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
from bio_inspired_nanochat.checkpoint_manager import checkpoint_model_config, save_checkpoint
from bio_inspired_nanochat.gpt import GPT, GPTConfig
from bio_inspired_nanochat.results_registry import read_records
from scripts.eval_matrix import SUMMARY_FIELDS, _get_logits, _load_base_train_checkpoint

pytestmark = pytest.mark.e2e

PRESETS = ("vanilla", "bio_all")
SEEDS = (1337, 1338)
# Required scalar columns that must be present + non-empty for an `ok` run (a subset of the
# full SUMMARY_FIELDS schema; metrics such as val_bpb/core_metric are legitimately blank on
# synthetic data).
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
        "--inline-smoke-training",
        "--train-tokens", "2048", "--eval-tokens", "512",
        "--data", "synthetic", "--device-type", "cpu",
        "--sequence-len", "64", "--vocab-size", "256",
        "--n-layer", "2", "--n-head", "2", "--n-embd", "64",
        "--device-batch-size", "1", "--total-batch-size-tokens", "64",
        "--ece-bins", "5", "--niah-lengths", "7",  # "7" < min NIAH length → NIAH skipped (fast)
        "--embedding-lr", "0.02", "--unembedding-lr", "0.004", "--matrix-lr", "0.01",
        "--out-dir", str(out_dir), "--batch-id", "test",
        "--registry-path", str(out_dir / "registry.jsonl"),
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
    try:
        jl = [
            json.JSONDecoder().decode(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except json.JSONDecodeError as exc:
        pytest.fail(f"summary.jsonl contains invalid JSON: {exc}")
    assert len(jl) == len(rows)
    assert {(d["preset"], int(d["seed"])) for d in jl} == {(p, s) for p in PRESETS for s in SEEDS}

    registry_records = read_records(str(matrix_dir.parent / "registry.jsonl"))
    assert len(registry_records) == len(rows)
    assert {record.run_id for record in registry_records} == {row["run_id"] for row in rows}
    assert all(record.harness == "eval" for record in registry_records)
    assert all(record.git_sha and record.config_hash for record in registry_records)

    # per-run detailed logs (the human-inspectable artifacts)
    run_dirs = [d for d in matrix_dir.iterdir() if d.is_dir()]
    assert len(run_dirs) == len(rows)
    for d in run_dirs:
        assert (d / "run_config.jsonl").exists(), f"missing run_config.jsonl in {d.name}"
        assert (d / "train_metrics.jsonl").exists(), f"missing train_metrics.jsonl in {d.name}"


def test_eval_matrix_evaluates_real_base_train_checkpoint(tmp_path):
    """The scientific path loads checkpoint architecture/state and never runs the inline trainer."""
    checkpoint_dir = tmp_path / "base_checkpoints" / "vanilla_s17"
    config = GPTConfig(
        sequence_len=16,
        vocab_size=32,
        n_layer=1,
        n_head=1,
        n_kv_head=1,
        n_embd=32,
        init_seed=17,
    )
    model = GPT(config)
    model.init_weights()
    model_config = checkpoint_model_config(
        model,
        {
            "sequence_len": 16,
            "vocab_size": 32,
            "n_layer": 1,
            "n_head": 1,
            "n_kv_head": 1,
            "n_embd": 32,
        },
    )
    save_checkpoint(
        str(checkpoint_dir),
        3,
        model.state_dict(),
        None,
        {
            "step": 3,
            "model_config": model_config,
            "synapses": False,
            "user_config": {"init_seed": 17, "num_iterations": 3, "total_batch_size": 16},
            "device_batch_size": 1,
            "loop_state": {"total_training_time": 1.5, "smooth_train_loss": 2.25},
        },
    )

    out_dir = tmp_path / "eval"
    argv = [
        "eval_matrix",
        "run",
        "--preset",
        "vanilla",
        "--seed",
        "17",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--checkpoint-step",
        "3",
        "--data",
        "synthetic",
        "--device-type",
        "cpu",
        "--device-batch-size",
        "1",
        "--eval-tokens",
        "16",
        "--ece-bins",
        "4",
        "--niah-lengths",
        "7",
        "--out-dir",
        str(out_dir),
        "--registry-path",
        str(tmp_path / "registry.jsonl"),
    ]
    from scripts.eval_matrix import main

    old_argv = sys.argv
    sys.argv = argv
    try:
        assert main() == 0
    finally:
        sys.argv = old_argv

    try:
        row = json.JSONDecoder().decode(
            (out_dir / "summary.jsonl").read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        pytest.fail(f"checkpoint evaluation emitted invalid JSONL: {exc}")
    assert row["recipe_source"] == "base_train_checkpoint"
    assert row["checkpoint_dir"] == str(checkpoint_dir.resolve())
    assert row["checkpoint_step"] == 3
    assert row["sequence_len"] == 16 and row["vocab_size"] == 32
    assert row["train_tokens_requested"] == 48
    assert row["train_tokens_processed"] == 48
    assert row["tok_per_sec"] == pytest.approx(32.0)
    registry_record = read_records(str(tmp_path / "registry.jsonl"))[0]
    assert registry_record.run_id == row["run_id"]
    assert registry_record.config_hash
    run_dir = Path(row["run_dir"])
    assert (run_dir / "run_config.jsonl").exists()
    assert not (run_dir / "train_metrics.jsonl").exists()
    with pytest.raises(ValueError, match="does not match checkpoint init_seed"):
        _load_base_train_checkpoint(
            checkpoint_dir,
            step=3,
            preset="vanilla",
            seed=18,
            device=model.get_device(),
        )


def test_eval_matrix_requires_checkpoint_or_explicit_smoke_mode():
    from scripts.eval_matrix import main

    old_argv = sys.argv
    sys.argv = ["eval_matrix", "run", "--preset", "vanilla", "--device-type", "cpu"]
    try:
        with pytest.raises(SystemExit) as exc_info:
            main()
    finally:
        sys.argv = old_argv
    assert exc_info.value.code == 2


def test_eval_logits_forces_synaptic_inference_mode():
    import torch

    class Probe(torch.nn.Module):
        def forward(self, idx, train_mode=True):
            assert not train_mode
            return torch.zeros((*idx.shape, 8))

    tokens = torch.zeros((1, 4), dtype=torch.long)
    assert _get_logits(Probe(), tokens).shape == (1, 4, 8)


# --------------------------------------------------------------------------- #
# 2. stats layer: paired test + CI computed on the harness output
# --------------------------------------------------------------------------- #
def test_eval_matrix_stats_layer_on_output(matrix_dir):
    csv_path = matrix_dir / "summary.csv"

    data = load_matrix_csv(csv_path, "val_loss")
    assert set(data) == set(PRESETS)
    assert all(sorted(data[p]) == list(SEEDS) for p in PRESETS)

    paired = paired_comparison(data["bio_all"], data["vanilla"], lower_is_better=True)
    assert paired is not None, "expected a paired result with 2 shared seeds"
    assert paired.n_pairs == len(SEEDS)
    assert math.isfinite(paired.mean_delta)
    assert math.isfinite(paired.delta_ci_low) and math.isfinite(paired.delta_ci_high)
    assert paired.delta_ci_low <= paired.mean_delta <= paired.delta_ci_high
    assert isinstance(paired.t_p_value, float) and isinstance(paired.wilcoxon_p_value, float)

    # multi-preset report: aggregate per preset + paired-vs-baseline for non-baseline presets
    rep = compare_matrix(data, baseline="vanilla", metric="val_loss", lower_is_better=True)
    matrix_results = rep["presets"]
    expected_runs = len(SEEDS)
    expected_configs = set(PRESETS)
    assert set(matrix_results) == expected_configs
    assert matrix_results["vanilla"]["aggregate"]["n"] in {expected_runs}
    assert "paired_vs_baseline" not in matrix_results["vanilla"], "baseline has no self-comparison"
    bio = matrix_results["bio_all"]
    assert bio["aggregate"]["n"] in {expected_runs}
    assert "paired_vs_baseline" in bio
    for key in ("n_pairs", "mean_delta", "delta_ci_low", "delta_ci_high", "t_p_value", "wilcoxon_p_value"):
        assert key in bio["paired_vs_baseline"]


# --------------------------------------------------------------------------- #
# 3. every model family emits finite quality metrics
# --------------------------------------------------------------------------- #
def test_eval_matrix_quality_metric_is_finite(matrix_dir):
    data = load_matrix_csv(matrix_dir / "summary.csv", "val_loss")
    assert set(data) == set(PRESETS)
    assert all(sorted(data[p]) == list(SEEDS) for p in PRESETS)
    assert all(math.isfinite(v) for by_seed in data.values() for v in by_seed.values())
