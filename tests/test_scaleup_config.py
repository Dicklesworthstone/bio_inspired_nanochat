"""Config-plumbing + data-split unit tests for the scale-up infra (bead hwxb.7.2).

The other scale-up suites cover DDP (test_scaleup_ddp), checkpoint/resume
(test_scaleup_checkpoint), telemetry (test_scaleup_telemetry), the recipe/optimizer
routing (test_scaleup_recipe), and tying (test_scaleup_tying). This file fills the
remaining gap: that every mechanism flag flows config→model correctly, that the
ablation presets + validator are coherent, and that the FineWeb train/val split is
disjoint. Fast, CPU-only; no tokenizer/GPU required.
"""
from __future__ import annotations

import dataclasses
import os

import pytest

from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticPresyn
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.ablation_registry import (
    ABLATION_PRESETS,
    MECHANISMS,
    apply_preset,
    is_mechanism_on,
    validate_config,
)
from bio_inspired_nanochat.dataset import parquet_paths_for_split


def _tiny_synaptic(**syn_overrides):
    sc = SynapticConfig(**syn_overrides)
    cfg = GPTSynapticConfig(
        sequence_len=32, vocab_size=97, n_layer=1, n_head=4, n_kv_head=4, n_embd=64, syn_cfg=sc
    )
    return GPTSynaptic(cfg)


# --------------------------------------------------------------------------- #
# Mechanism registry / config integrity
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_mechanism_registry_fields_are_real_config_fields():
    """Every MechanismFlag.field (and its prerequisites) must be a real SynapticConfig field."""
    cfg_fields = {f.name for f in dataclasses.fields(SynapticConfig)}
    for m in MECHANISMS:
        assert m.field in cfg_fields, f"mechanism {m.mechanism!r} names unknown field {m.field!r}"
        for req in m.requires:
            assert req in cfg_fields, f"mechanism {m.field!r} requires unknown field {req!r}"


@pytest.mark.unit
def test_all_ablation_presets_apply_and_validate_clean():
    """Every shipped preset applies to a fresh config and passes validation (no errors)."""
    for preset in ABLATION_PRESETS:
        cfg = apply_preset(preset, SynapticConfig())
        errors, _warnings = validate_config(cfg)
        assert errors == [], f"preset {preset!r} produced validation errors: {errors}"


@pytest.mark.unit
def test_apply_preset_sets_the_expected_field():
    """apply_preset writes exactly the preset's overrides (spot-check a few)."""
    base = SynapticConfig()
    if "bio_no_hebbian" in ABLATION_PRESETS:
        cfg = apply_preset("bio_no_hebbian", SynapticConfig())
        assert cfg.enable_hebbian is False and base.enable_hebbian is True
    if "bio_no_presyn" in ABLATION_PRESETS:
        cfg = apply_preset("bio_no_presyn", SynapticConfig())
        assert cfg.enable_presyn is False


@pytest.mark.unit
def test_validator_catches_missing_prerequisite():
    """The validator must flag a mechanism enabled without its prerequisite (it's load-bearing)."""
    cfg = SynapticConfig(bistable_latch=True, enable_hebbian=False)
    errors, _ = validate_config(cfg)
    assert errors, "validator should ERROR on bistable_latch enabled with enable_hebbian off"
    assert any("bistable_latch" in e for e in errors)
    # is_mechanism_on still reports it as 'on' (which is why the validator must catch it).
    assert is_mechanism_on(cfg, "bistable_latch") is True


# --------------------------------------------------------------------------- #
# Flag → model plumbing
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize("on", [True, False])
def test_learnable_kinetics_flag_flows_to_model(on):
    """learnable_kinetics builds (or omits) the LearnableKinetics submodule on every presyn."""
    m = _tiny_synaptic(learnable_kinetics=on)
    kinetics = [mod.kinetics for mod in m.modules() if isinstance(mod, SynapticPresyn)]
    assert kinetics, "model should contain SynapticPresyn modules"
    if on:
        assert all(k is not None for k in kinetics)
        # the learned kinetics expose 0-D theta_* Parameters (the ones DistAdamW must handle).
        thetas = [n for n, _ in m.named_parameters() if "kinetics.theta" in n]
        assert thetas, "learnable_kinetics=True should add theta_* parameters"
    else:
        assert all(k is None for k in kinetics)


@pytest.mark.unit
def test_synaptic_config_threads_through_to_model():
    """syn_cfg fields set on construction are visible on the model's config (no silent drop)."""
    m = _tiny_synaptic(enable_metabolism=False, bdnf_scale=0.0)
    assert m.config.syn_cfg.enable_metabolism is False
    assert m.config.syn_cfg.bdnf_scale == 0.0


@pytest.mark.unit
@pytest.mark.parametrize("tie", [True, False])
def test_tie_embeddings_flag_flows(tie):
    cfg = GPTSynapticConfig(
        sequence_len=32, vocab_size=97, n_layer=1, n_head=4, n_kv_head=4, n_embd=64,
        tie_embeddings=tie,
    )
    assert cfg.tie_embeddings is tie
    m = GPTSynaptic(cfg)
    assert m.config.tie_embeddings is tie


# --------------------------------------------------------------------------- #
# Data: FineWeb train/val split (no tokenizer / network needed)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_parquet_split_is_disjoint_and_complete(tmp_path):
    d = str(tmp_path)
    names = [f"shard_{i:04d}.parquet" for i in range(4)]
    for n in names:
        (tmp_path / n).write_bytes(b"")  # contents irrelevant; only the listing/splitting matters
    train = parquet_paths_for_split("train", data_dir=d)
    val = parquet_paths_for_split("val", data_dir=d)
    # last shard is val; the rest are train.
    assert [os.path.basename(p) for p in val] == [names[-1]]
    assert [os.path.basename(p) for p in train] == names[:-1]
    # disjoint + complete coverage of all shards.
    assert set(train).isdisjoint(set(val))
    assert set(train) | set(val) == {os.path.join(d, n) for n in names}


@pytest.mark.unit
def test_parquet_split_single_shard_used_for_both(tmp_path):
    d = str(tmp_path)
    (tmp_path / "only.parquet").write_bytes(b"")
    train = parquet_paths_for_split("train", data_dir=d)
    val = parquet_paths_for_split("val", data_dir=d)
    # 1 shard → smoke convention: the same shard serves both splits.
    assert [os.path.basename(p) for p in train] == ["only.parquet"]
    assert [os.path.basename(p) for p in val] == ["only.parquet"]
