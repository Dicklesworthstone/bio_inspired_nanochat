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
from bio_inspired_nanochat.checkpoint_manager import (
    checkpoint_model_config,
    save_checkpoint,
    synaptic_config_to_meta,
)
from bio_inspired_nanochat.gpt import GPT, GPTConfig
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.results_registry import read_records
from bio_inspired_nanochat.synaptic import SynapticConfig
from scripts.eval_matrix import (
    SUMMARY_FIELDS,
    _batch_output_dir,
    _binary_auroc,
    _deterministic_ood_tokens,
    _forgetting_rate_from_accuracy_matrix,
    _get_logits,
    _gini_coefficient,
    _load_base_train_checkpoint,
    _parse_int_list,
    _parse_str_list,
    _summarize_routing_counts,
    _val_loss_ppl_ece,
)

pytestmark = pytest.mark.e2e

PRESETS = ("vanilla", "bio_all")
SEEDS = (1337, 1338)
# Required scalar columns that must be present + non-empty for an `ok` run (a subset of the
# full SUMMARY_FIELDS schema; metrics such as val_bpb/core_metric are legitimately blank on
# synthetic data).
_REQUIRED_FIELDS = (
    "run_id", "preset", "seed", "data", "device_type", "sequence_len", "vocab_size",
    "n_layer", "n_head", "n_embd", "train_tokens_requested", "train_tokens_processed",
    "steps", "walltime_sec", "tok_per_sec", "val_loss", "val_ppl", "id_ece",
    "ood_auroc", "forgetting_rate", "recall_by_length",
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
        "--ece-bins", "5", "--niah-lengths", "8",
        "--dead-expert-threshold", "0.05", "--continual-exposures", "2",
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
    assert all(
        {"id_ece", "ood_auroc", "forgetting_rate"} <= set(record.metrics)
        for record in registry_records
    )

    # per-run detailed logs (the human-inspectable artifacts)
    run_dirs = [d for d in matrix_dir.iterdir() if d.is_dir()]
    assert len(run_dirs) == len(rows)
    for d in run_dirs:
        assert (d / "run_config.jsonl").exists(), f"missing run_config.jsonl in {d.name}"
        assert (d / "train_metrics.jsonl").exists(), f"missing train_metrics.jsonl in {d.name}"
        capability_records = [
            json.loads(line)
            for line in (d / "capability_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert {record["capability"] for record in capability_records} == {
            "uncertainty",
            "continual",
            "routing",
            "memory",
        }

    for row in jl:
        assert 0.0 <= row["id_ece"] <= 1.0
        assert 0.0 <= row["ood_auroc"] <= 1.0
        assert 0.0 <= row["forgetting_rate"] <= 1.0
        assert set(row["recall_by_length"]) == {"8"}
        assert 0.0 <= row["recall_by_length"]["8"] <= 1.0
        assert row["capability_metric_status"]["routing"] == "not_applicable"
        assert row["moe_gini"] is None and row["dead_expert_frac"] is None


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


def test_eval_matrix_emits_live_moe_routing_metrics_from_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "base_checkpoints" / "bio_all_s19"
    syn_cfg = SynapticConfig()
    config = GPTSynapticConfig(
        sequence_len=16,
        vocab_size=32,
        n_layer=1,
        n_head=1,
        n_kv_head=1,
        n_embd=16,
        syn_cfg=syn_cfg,
        use_moe=True,
        num_experts=4,
        moe_top_k=2,
        init_seed=19,
    )
    model = GPTSynaptic(config)
    model.init_weights()
    model_config = checkpoint_model_config(
        model,
        {
            "sequence_len": 16,
            "vocab_size": 32,
            "n_layer": 1,
            "n_head": 1,
            "n_kv_head": 1,
            "n_embd": 16,
        },
    )
    save_checkpoint(
        str(checkpoint_dir),
        1,
        model.state_dict(),
        None,
        {
            "step": 1,
            "model_config": model_config,
            "synapses": True,
            "synaptic_config": synaptic_config_to_meta(syn_cfg),
            "user_config": {"init_seed": 19, "num_iterations": 1, "total_batch_size": 16},
            "device_batch_size": 1,
            "loop_state": {"total_training_time": 1.0, "smooth_train_loss": 2.5},
        },
    )

    out_dir = tmp_path / "moe_eval"
    from scripts.eval_matrix import main

    old_argv = sys.argv
    sys.argv = [
        "eval_matrix",
        "run",
        "--preset",
        "bio_all",
        "--seed",
        "19",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--checkpoint-step",
        "1",
        "--data",
        "synthetic",
        "--device-type",
        "cpu",
        "--device-batch-size",
        "1",
        "--eval-tokens",
        "16",
        "--niah-lengths",
        "7",
        "--out-dir",
        str(out_dir),
        "--registry-path",
        str(tmp_path / "moe_registry.jsonl"),
    ]
    try:
        assert main() == 0
    finally:
        sys.argv = old_argv

    row = json.loads((out_dir / "summary.jsonl").read_text(encoding="utf-8"))
    assert row["use_moe"] and row["num_experts"] == 4 and row["moe_top_k"] == 2
    assert row["capability_metric_status"]["routing"] == "ok"
    assert 0.0 <= row["moe_gini"] <= 1.0
    assert 0.0 <= row["dead_expert_frac"] <= 1.0
    routing = [
        json.loads(line)
        for line in (Path(row["run_dir"]) / "capability_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if '"capability": "routing"' in line
    ]
    assert len(routing) == 1 and routing[0]["layers"]
    registry = read_records(str(tmp_path / "moe_registry.jsonl"))[0]
    assert {"moe_gini", "dead_expert_frac"} <= set(registry.metrics)


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


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--ece-bins", "1", "--ece-bins must be >= 2"),
        ("--dead-expert-threshold", "1", "--dead-expert-threshold must be in"),
        ("--continual-tasks", "1", "--continual-tasks must be >= 2"),
        ("--continual-exposures", "0", "--continual-exposures must be >= 1"),
    ],
)
def test_eval_matrix_rejects_invalid_capability_metric_policy(flag, value, message, capsys):
    from scripts.eval_matrix import main

    old_argv = sys.argv
    sys.argv = [
        "eval_matrix",
        "run",
        "--preset",
        "vanilla",
        "--inline-smoke-training",
        flag,
        value,
    ]
    try:
        with pytest.raises(SystemExit) as exc_info:
            main()
    finally:
        sys.argv = old_argv
    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err


def test_eval_matrix_batch_returns_nonzero_for_any_failed_cell(monkeypatch, tmp_path):
    import scripts.eval_matrix as eval_matrix_module
    import torch

    monkeypatch.setattr(
        eval_matrix_module,
        "compute_init",
        lambda _device_type: (False, 0, 0, 1, torch.device("cpu")),
    )
    monkeypatch.setattr(eval_matrix_module, "compute_cleanup", lambda: None)

    old_argv = sys.argv
    try:
        for failed_seeds, expected_status, batch_id in (
            (set(), 0, "all_success"),
            ({1}, 1, "partial_failure"),
            ({1, 2}, 1, "all_failed"),
        ):
            def fake_run_one(*, seed, **_kwargs):
                if seed in failed_seeds:
                    raise RuntimeError(f"planted failure for seed {seed}")

            monkeypatch.setattr(eval_matrix_module, "_run_one", fake_run_one)
            sys.argv = [
                "eval_matrix",
                "matrix",
                "--presets",
                "vanilla",
                "--seeds",
                "1,2",
                "--inline-smoke-training",
                "--device-type",
                "cpu",
                "--out-dir",
                str(tmp_path),
                "--batch-id",
                batch_id,
            ]
            assert eval_matrix_module.main() == expected_status
    finally:
        sys.argv = old_argv


def test_eval_matrix_batch_propagates_remote_rank_failure(monkeypatch, tmp_path):
    import scripts.eval_matrix as eval_matrix_module
    import torch

    monkeypatch.setattr(
        eval_matrix_module,
        "compute_init",
        lambda _device_type: (True, 0, 0, 2, torch.device("cpu")),
    )
    monkeypatch.setattr(eval_matrix_module, "compute_cleanup", lambda: None)
    monkeypatch.setattr(eval_matrix_module, "_run_one", lambda **_kwargs: None)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "broadcast_object_list",
        lambda _values, *, src, device: None,
    )
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, *, op: tensor.fill_(1),
    )

    old_argv = sys.argv
    sys.argv = [
        "eval_matrix",
        "matrix",
        "--presets",
        "vanilla",
        "--seeds",
        "1",
        "--inline-smoke-training",
        "--device-type",
        "cpu",
        "--out-dir",
        str(tmp_path),
        "--batch-id",
        "remote_failure",
    ]
    try:
        assert eval_matrix_module.main() == 1
    finally:
        sys.argv = old_argv


def test_eval_matrix_rejects_duplicate_batch_cells():
    with pytest.raises(ValueError, match="must not contain duplicates"):
        _parse_str_list("vanilla,vanilla")
    with pytest.raises(ValueError, match="must not contain duplicates"):
        _parse_int_list("7,8,7")

    assert _parse_str_list("vanilla,bio_all") == ["vanilla", "bio_all"]
    assert _parse_int_list("7,8") == [7, 8]


@pytest.mark.parametrize("batch_id", ["", "../escape", "nested/escape", "..\\escape"])
def test_eval_matrix_rejects_escaping_batch_id_before_compute_init(
    monkeypatch,
    tmp_path,
    batch_id,
):
    import scripts.eval_matrix as eval_matrix_module

    def unexpected_compute_init(_device_type):
        pytest.fail("invalid batch_id reached compute initialization")

    monkeypatch.setattr(eval_matrix_module, "compute_init", unexpected_compute_init)
    old_argv = sys.argv
    sys.argv = [
        "eval_matrix",
        "matrix",
        "--presets",
        "vanilla",
        "--seeds",
        "1",
        "--inline-smoke-training",
        "--device-type",
        "cpu",
        "--out-dir",
        str(tmp_path),
        "--batch-id",
        batch_id,
    ]
    try:
        with pytest.raises(ValueError, match="batch_id"):
            eval_matrix_module.main()
    finally:
        sys.argv = old_argv


def test_batch_output_dir_rejects_absolute_path_and_accepts_direct_child(tmp_path):
    with pytest.raises(ValueError, match="batch_id"):
        _batch_output_dir(tmp_path, str(tmp_path / "escaped"))

    assert _batch_output_dir(tmp_path, "batch_1") == tmp_path / "batch_1"


def test_eval_logits_forces_synaptic_inference_mode():
    import torch

    class Probe(torch.nn.Module):
        def forward(self, idx, train_mode=True):
            assert not train_mode
            return torch.zeros((*idx.shape, 8))

    tokens = torch.zeros((1, 4), dtype=torch.long)
    assert _get_logits(Probe(), tokens).shape == (1, 4, 8)


def test_validation_loss_is_pooled_over_valid_tokens_not_batches():
    import torch

    class LookupLogits(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

        def forward(self, idx):
            hard = idx.eq(1).to(torch.float32)
            return torch.stack((torch.zeros_like(hard), 2.0 * hard), dim=-1)

    model = LookupLogits()
    uneven_batches = iter(
        [
            (torch.tensor([[0]]), torch.tensor([[0]])),
            (torch.tensor([[1, 1]]), torch.tensor([[0, 0]])),
        ]
    )
    loss, ppl, *_ = _val_loss_ppl_ece(
        model,
        uneven_batches,
        steps=2,
        device_type="cpu",
        ddp=False,
        ece_bins=2,
    )
    easy_loss = math.log(2.0)
    hard_loss = math.log1p(math.exp(2.0))
    pooled = (easy_loss + 2.0 * hard_loss) / 3.0
    batch_mean = (easy_loss + hard_loss) / 2.0
    assert loss == pytest.approx(pooled)
    assert loss != pytest.approx(batch_mean)
    assert ppl == pytest.approx(math.exp(pooled))

    equal_batches = iter(
        [
            (torch.tensor([[0]]), torch.tensor([[0]])),
            (torch.tensor([[1]]), torch.tensor([[0]])),
        ]
    )
    equal_loss, *_ = _val_loss_ppl_ece(
        model,
        equal_batches,
        steps=2,
        device_type="cpu",
        ddp=False,
        ece_bins=2,
    )
    assert equal_loss == pytest.approx(batch_mean)


def test_validation_loss_pools_ddp_numerator_and_denominator(monkeypatch):
    import scripts.eval_matrix as eval_matrix_module
    import torch

    class UniformLogits(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

        def forward(self, idx):
            return torch.zeros((*idx.shape, 2), dtype=torch.float32)

    remote_loss_sum = 4.0
    remote_valid_tokens = 2
    reduce_call = 0
    sum_op = getattr(torch.distributed, "ReduceOp").SUM

    def fake_all_reduce(tensor, *, op):
        nonlocal reduce_call
        assert op == sum_op
        if reduce_call == 0:
            tensor.add_(remote_loss_sum)
        elif reduce_call == 1:
            tensor.add_(remote_valid_tokens)
        reduce_call += 1

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(
        eval_matrix_module,
        "_distributed_scores",
        lambda scores, *, ddp: scores,
    )
    loss, *_ = _val_loss_ppl_ece(
        UniformLogits(),
        iter([(torch.tensor([[0]]), torch.tensor([[0]]))]),
        steps=1,
        device_type="cpu",
        ddp=True,
        ece_bins=2,
    )

    expected = (math.log(2.0) + remote_loss_sum) / (1 + remote_valid_tokens)
    assert loss == pytest.approx(expected)
    assert reduce_call == 5


def test_validation_loss_rejects_all_masked_evidence():
    import torch

    class UniformLogits(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

        def forward(self, idx):
            return torch.zeros((*idx.shape, 2), dtype=torch.float32)

    batches = iter([(torch.tensor([[0, 1]]), torch.full((1, 2), -1))])
    with pytest.raises(ValueError, match="no valid target tokens"):
        _val_loss_ppl_ece(
            UniformLogits(),
            batches,
            steps=1,
            device_type="cpu",
            ddp=False,
            ece_bins=2,
        )


def test_capability_metric_primitives_are_seeded_and_fail_closed():
    import torch

    tokens = torch.tensor([[0, 1, 2], [3, 4, 5]])
    corrupted = _deterministic_ood_tokens(tokens, vocab_size=8, seed=17)
    assert torch.equal(corrupted, _deterministic_ood_tokens(tokens, vocab_size=8, seed=17))
    assert corrupted.shape == tokens.shape
    assert bool((corrupted != tokens).all())
    assert 0 <= int(corrupted.min()) <= int(corrupted.max()) < 8

    assert _binary_auroc([0.1, 0.2], [0.8, 0.9]) == 1.0
    assert _binary_auroc([0.8, 0.9], [0.1, 0.2]) == 0.0
    assert _binary_auroc([0.5], [0.5]) == 0.5
    with pytest.raises(ValueError, match="finite"):
        _binary_auroc([0.1], [float("nan")])


def test_routing_metrics_report_specialization_and_dead_experts():
    import torch

    balanced = torch.tensor([10.0, 10.0, 10.0, 10.0])
    assert _gini_coefficient(balanced) == pytest.approx(0.0)
    assert _gini_coefficient(torch.tensor([0.0, 0.0, 0.0, 40.0])) == pytest.approx(0.75)
    summary = _summarize_routing_counts(
        {"h.0.mlp": torch.tensor([49.0, 49.0, 2.0, 0.0])},
        dead_expert_threshold=0.03,
    )
    assert summary.moe_gini is not None and summary.moe_gini > 0.0
    assert summary.dead_expert_frac == pytest.approx(0.5)
    assert summary.layers["h.0.mlp"]["dead_experts"] == 2
    assert _summarize_routing_counts({}, dead_expert_threshold=0.01).moe_gini is None


def test_forgetting_rate_uses_peak_to_final_accuracy_for_prior_tasks():
    rate, by_task = _forgetting_rate_from_accuracy_matrix(
        [
            [0.9, None, None],
            [0.8, 0.7, None],
            [0.6, 0.8, 0.5],
        ]
    )
    assert rate == pytest.approx(0.15)
    assert by_task == {
        "0": {"peak_accuracy": 0.9, "final_accuracy": 0.6, "forgetting": pytest.approx(0.3)},
        "1": {"peak_accuracy": 0.8, "final_accuracy": 0.8, "forgetting": 0.0},
    }
    with pytest.raises(ValueError, match="square"):
        _forgetting_rate_from_accuracy_matrix([[0.5]])


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
