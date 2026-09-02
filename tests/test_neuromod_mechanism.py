"""Neuromodulation is a registered ablation mechanism that the eval harness can actually run.

Before 2026-09-01 the DA/ACh/NE bus (hy8.1) existed only as a ``base_train`` script global, so
the pre-registered bio-vs-vanilla matrix (docs/ablation_matrix.md) could neither name it nor run
it, and ``eval_matrix`` accepted only the named registry presets — the ``synaptic_off`` anchor
and every ``add_*`` column were unrunnable. These tests lock:

* ``neuromod_enabled`` is a ``SynapticConfig`` field registered in ``ablation_registry`` as an
  opt-in mechanism whose prerequisites are presyn + hebbian (the layers it modulates);
* the matrix derives an ``add_neuromod`` column from that registration;
* ``eval_matrix`` accepts every matrix column, builds the right ``SynapticConfig`` for it, and
  instantiates the bus during inline training (its telemetry lands in ``train_metrics.jsonl``);
* ``base_train`` records the flag on the model config and honours it from either spelling.

Green here proves the plumbing, not that neuromodulation helps anything.

Run:  pytest tests/test_neuromod_mechanism.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from bio_inspired_nanochat import ablation_matrix as am
from bio_inspired_nanochat.ablation_registry import MECHANISMS, is_mechanism_on, validate_config
from bio_inspired_nanochat.synaptic import SynapticConfig

pytestmark = pytest.mark.unit


def _mechanism(name: str):
    (m,) = [x for x in MECHANISMS if x.mechanism == name]
    return m


def test_neuromod_is_registered_default_off_with_its_modulated_layers_as_prerequisites():
    m = _mechanism("neuromod")
    assert m.field == "neuromod_enabled"
    assert m.default is False and m.off_value is False and m.default_on is False
    assert set(m.requires) == {"enable_presyn", "enable_hebbian"}
    cfg = SynapticConfig()
    assert cfg.neuromod_enabled is False
    assert not is_mechanism_on(cfg, "neuromod_enabled")
    assert is_mechanism_on(SynapticConfig(neuromod_enabled=True), "neuromod_enabled")
    # Enabling the bus on a config with nothing for it to modulate is the foot-gun the
    # validator exists for.
    errors, _ = validate_config(SynapticConfig(neuromod_enabled=True, enable_hebbian=False))
    assert any("neuromod" in e and "enable_hebbian" in e for e in errors)
    errors, _ = validate_config(SynapticConfig(neuromod_enabled=True))
    assert errors == []


def test_matrix_derives_an_add_neuromod_column_with_prerequisites_back_on():
    columns = {c.config_id: c for c in am.add_one_in()}
    assert "add_neuromod" in columns
    cfg = columns["add_neuromod"].build_syn_cfg()
    assert cfg is not None
    assert cfg.neuromod_enabled is True
    assert cfg.enable_presyn is True and cfg.enable_hebbian is True
    # Everything else that synaptic_off neutralised stays off: the column isolates the bus.
    assert cfg.enable_metabolism is False
    assert cfg.stochastic_train_frac == 0.0
    ids = [c.config_id for c in am.screening_columns()]
    assert ids.count("add_neuromod") == 1 and "synaptic_off" in ids
    assert len(ids) == len(am.anchors()) + len(am.leave_one_out()) + len(am.add_one_in())


def test_eval_matrix_accepts_every_matrix_column():
    from scripts.eval_matrix import MATRIX_COLUMNS, PresetId, _build_model, _syn_cfg_for_preset

    assert "synaptic_off" in MATRIX_COLUMNS and "add_neuromod" in MATRIX_COLUMNS
    assert not (set(MATRIX_COLUMNS) & set(PresetId.__args__)), "columns must not shadow presets"
    assert set(MATRIX_COLUMNS) | set(PresetId.__args__) == {c.config_id for c in am.screening_columns()}

    off = _syn_cfg_for_preset("synaptic_off")
    for m in MECHANISMS:
        if m.default_on:
            assert getattr(off, m.field) == m.off_value, m.field
    # Registry presets still go through apply_preset.
    assert _syn_cfg_for_preset("bio_no_presyn").enable_presyn is False

    model = _build_model(
        preset="add_neuromod",
        seed=1,
        device=torch.device("cpu"),
        sequence_len=16,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_embd=32,
        init_type="baseline",
        use_moe=False,
        num_experts=0,
        moe_top_k=0,
    )
    assert model.config.syn_cfg.neuromod_enabled is True
    assert model.config.syn_cfg.enable_metabolism is False


def test_eval_matrix_inline_training_instantiates_the_bus(tmp_path):
    from scripts.eval_matrix import main

    out_dir = tmp_path / "eval"
    argv = [
        "eval_matrix", "run",
        "--preset", "add_neuromod",
        "--seed", "3",
        "--data", "synthetic",
        "--inline-smoke-training",
        "--device-type", "cpu",
        "--device-batch-size", "1",
        "--sequence-len", "16",
        "--vocab-size", "32",
        "--n-layer", "1",
        "--n-head", "2",
        "--n-embd", "32",
        "--total-batch-size-tokens", "16",
        "--train-tokens", "48",
        "--eval-tokens", "16",
        "--ece-bins", "4",
        "--niah-lengths", "7",
        "--out-dir", str(out_dir),
        "--registry-path", str(tmp_path / "registry.jsonl"),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        assert main() == 0
    finally:
        sys.argv = old_argv

    row = json.loads((out_dir / "summary.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["preset"] == "add_neuromod" and row["status"] == "ok"
    run_dir = Path(row["run_dir"])
    config = json.loads((run_dir / "run_config.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert config["neuromod_enabled"] is True
    metrics = [
        json.loads(line)
        for line in (run_dir / "train_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(metrics) == 3, "48 tokens / 16 per step = 3 optimizer steps"
    for m in metrics:
        assert {"nm/da", "nm/ach", "nm/ne", "nm/gain_plasticity"} <= set(m), (
            "the bus must publish its levels and gains every step"
        )
    # After the first step the EMAs are seeded; the ACh level is non-negative by construction.
    assert all(m["nm/ach"] >= 0.0 for m in metrics)


def test_eval_matrix_without_the_flag_has_no_bus_telemetry(tmp_path):
    from scripts.eval_matrix import main

    out_dir = tmp_path / "eval"
    argv = [
        "eval_matrix", "run",
        "--preset", "synaptic_off",
        "--seed", "3",
        "--data", "synthetic",
        "--inline-smoke-training",
        "--device-type", "cpu",
        "--device-batch-size", "1",
        "--sequence-len", "16",
        "--vocab-size", "32",
        "--n-layer", "1",
        "--n-head", "2",
        "--n-embd", "32",
        "--total-batch-size-tokens", "16",
        "--train-tokens", "16",
        "--eval-tokens", "16",
        "--ece-bins", "4",
        "--niah-lengths", "7",
        "--out-dir", str(out_dir),
        "--registry-path", str(tmp_path / "registry.jsonl"),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        assert main() == 0
    finally:
        sys.argv = old_argv
    row = json.loads((out_dir / "summary.jsonl").read_text(encoding="utf-8").splitlines()[0])
    run_dir = Path(row["run_dir"])
    config = json.loads((run_dir / "run_config.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert config["neuromod_enabled"] is False
    metrics = [
        json.loads(line)
        for line in (run_dir / "train_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert metrics and all("nm/da" not in m for m in metrics)


def test_base_train_records_and_honours_the_flag():
    src = Path("scripts/base_train.py").read_text(encoding="utf-8")
    assert "syn_cfg.neuromod_enabled = True" in src, "the CLI flag must be recorded on the model config"
    assert "if use_syn and (neuromod_enabled or syn_cfg.neuromod_enabled):" in src, (
        "either spelling must instantiate the bus"
    )
