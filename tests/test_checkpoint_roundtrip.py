"""
SynapticConfig checkpoint round-trip — bead vg9.6.

build_model used to rebuild synaptic models with SynapticConfig() DEFAULTS, so a model trained
or tuned with custom bio kinetics silently reloaded as a DIFFERENT model. These tests lock the
fix: the full bio config persists into meta_data (as JSON) and rebuilds EXACTLY, with
forward/back-compat handling and a provenance stamp.

Run:  pytest tests/test_checkpoint_roundtrip.py -v
"""

from __future__ import annotations

import json

import pytest

from bio_inspired_nanochat.checkpoint_manager import (
    config_hash,
    config_provenance,
    synaptic_config_from_meta,
    synaptic_config_to_meta,
)
from bio_inspired_nanochat.synaptic import SynapticConfig


@pytest.mark.unit
def test_custom_config_round_trips_through_meta_json():
    # The headline acceptance: save a non-default config -> reload -> identical.
    cfg = SynapticConfig(tau_c=9.0, syt_fast_kd=0.7, doc2_gain=0.15, enable_hebbian=False)
    meta = {"synapses": True, "synaptic_config": synaptic_config_to_meta(cfg)}
    meta = json.loads(json.dumps(meta))  # meta_data is persisted as JSON on disk
    assert synaptic_config_from_meta(meta) == cfg, "non-default bio config must round-trip exactly"


@pytest.mark.unit
def test_full_default_config_round_trips():
    cfg = SynapticConfig()
    meta = {"synaptic_config": json.loads(json.dumps(synaptic_config_to_meta(cfg)))}
    assert synaptic_config_from_meta(meta) == cfg


@pytest.mark.unit
def test_missing_config_falls_back_to_defaults():
    # pre-vg9.6 checkpoints (no persisted config) must not crash; they fall back to defaults.
    assert synaptic_config_from_meta({"synapses": True}) == SynapticConfig()
    assert synaptic_config_from_meta(None) == SynapticConfig()


@pytest.mark.unit
def test_unknown_saved_field_is_ignored_forward_compat():
    d = synaptic_config_to_meta(SynapticConfig(tau_c=7.0))
    d["a_field_removed_in_a_future_schema"] = 123
    rebuilt = synaptic_config_from_meta({"synaptic_config": d})
    assert rebuilt.tau_c == 7.0  # known fields applied; the unknown one is dropped, not a crash


@pytest.mark.unit
def test_missing_new_field_takes_default_back_compat():
    d = synaptic_config_to_meta(SynapticConfig())
    removed = d.pop("doc2_gain")  # an old checkpoint that predates a newer field
    rebuilt = synaptic_config_from_meta({"synaptic_config": d})
    assert rebuilt.doc2_gain == SynapticConfig().doc2_gain != removed + 1  # default filled in


@pytest.mark.unit
def test_config_hash_is_stable_and_sensitive():
    a = synaptic_config_to_meta(SynapticConfig(tau_c=6.0))
    b = synaptic_config_to_meta(SynapticConfig(tau_c=6.0))
    c = synaptic_config_to_meta(SynapticConfig(tau_c=9.0))
    assert config_hash(a) == config_hash(b), "same config -> same hash"
    assert config_hash(a) != config_hash(c), "different config -> different hash"


@pytest.mark.unit
def test_provenance_stamp_has_sha_and_config_hash():
    prov = config_provenance(SynapticConfig())
    assert set(prov) == {"git_sha", "synaptic_config_hash"}
    assert len(prov["synaptic_config_hash"]) == 16
    # git_sha is a 40-char hex SHA in a git repo, or None if git is unavailable
    assert prov["git_sha"] is None or len(prov["git_sha"]) == 40
