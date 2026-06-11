"""Ablation registry + config validator — bead hm4.7.

Locks the "modular, toggleable mechanisms for clean ablation" discipline:
- the registry stays in sync with SynapticConfig defaults,
- every ablation preset toggles EXACTLY its declared fields (and validates clean),
- the validator rejects silently-broken opt-in configs and warns on risky ones.
"""

from __future__ import annotations

import dataclasses

import pytest

from bio_inspired_nanochat.ablation_registry import (
    ABLATION_PRESETS,
    MECHANISMS,
    apply_preset,
    assert_valid_config,
    is_mechanism_on,
    validate_config,
)
from bio_inspired_nanochat.synaptic import SynapticConfig

pytestmark = pytest.mark.unit


def test_registry_defaults_match_synaptic_config():
    cfg = SynapticConfig()
    for m in MECHANISMS:
        assert getattr(cfg, m.field) == m.default, (
            f"registry default for {m.field} ({m.default}) != SynapticConfig default "
            f"({getattr(cfg, m.field)})"
        )
        assert (m.default != m.off_value) == m.default_on, (
            f"{m.field}: default_on={m.default_on} inconsistent with default/off values"
        )


def test_default_config_is_valid_and_quiet():
    errors, warnings = validate_config(SynapticConfig())
    assert errors == [], f"default config must validate clean, got {errors}"
    assert warnings == [], f"default config should not warn, got {warnings}"


@pytest.mark.parametrize("preset", sorted(ABLATION_PRESETS))
def test_each_preset_toggles_only_its_declared_fields(preset: str):
    base = SynapticConfig()
    cfg = apply_preset(preset, SynapticConfig())
    changed = {
        f.name
        for f in dataclasses.fields(SynapticConfig)
        if getattr(cfg, f.name) != getattr(base, f.name)
    }
    assert changed == set(ABLATION_PRESETS[preset]), (
        f"preset {preset!r} changed {changed}, expected {set(ABLATION_PRESETS[preset])}"
    )


@pytest.mark.parametrize("preset", sorted(ABLATION_PRESETS))
def test_every_preset_produces_a_valid_config(preset: str):
    # Ablating a default-on mechanism (and thereby silencing its dependents) is expected
    # and must NOT raise — only opt-in misuse does.
    errors, _ = validate_config(apply_preset(preset, SynapticConfig()))
    assert errors == [], f"preset {preset!r} must validate clean, got {errors}"


def test_optin_mechanism_without_prerequisite_is_an_error():
    cfg = SynapticConfig(bistable_latch=True, enable_hebbian=False)
    errors, _ = validate_config(cfg)
    assert any("bistable_latch" in e and "enable_hebbian" in e for e in errors)
    with pytest.raises(ValueError, match="bistable_latch"):
        assert_valid_config(cfg)


def test_flex_without_presyn_is_an_error():
    errors, _ = validate_config(SynapticConfig(use_flex_attention=True, enable_presyn=False))
    assert any("flex_attention" in e for e in errors)


def test_flex_with_presyn_only_warns_prefill_only():
    errors, warnings = validate_config(SynapticConfig(use_flex_attention=True))
    assert errors == []
    assert any("PREFILL-ONLY" in w for w in warnings)


def test_out_of_range_knobs_are_errors():
    assert validate_config(SynapticConfig(stochastic_train_frac=1.5))[0]
    assert validate_config(SynapticConfig(bistable_latch=True, latch_hill_k=0.0))[0]
    assert validate_config(SynapticConfig(bistable_latch=True, latch_pp1_basal=2.0))[0]
    # latch_ltd_thr must sit below camkii_thr (neutral zone for the BCM curve).
    assert validate_config(SynapticConfig(bistable_latch=True, latch_ltd_thr=1.5))[0]


def test_is_mechanism_on_reads_off_value():
    assert is_mechanism_on(SynapticConfig(), "enable_presyn") is True
    assert is_mechanism_on(SynapticConfig(enable_presyn=False), "enable_presyn") is False
    assert is_mechanism_on(SynapticConfig(), "bistable_latch") is False
    assert is_mechanism_on(SynapticConfig(doc2_gain=0.0), "doc2_gain") is False


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown ablation preset"):
        apply_preset("bio_no_telepathy", SynapticConfig())
