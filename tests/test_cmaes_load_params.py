"""Tests for CMA-ES params ingestion (bead c2l): parse/validate/apply + base_train wiring.

The library lives in ``bio_inspired_nanochat/cmaes_params.py`` (importable without executing
the training script). The wiring contract in ``scripts/base_train.py`` — a ``load_cmaes_params``
configurator setting applied to ``syn_cfg`` only on fresh runs, refused on resume — is locked
here by source-level assertions on the two integration points plus behavioural tests of the
library the wiring calls.

Run:  pytest tests/test_cmaes_load_params.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bio_inspired_nanochat.cmaes_params import apply_cmaes_params, parse_cmaes_params
from bio_inspired_nanochat.synaptic import SynapticConfig

pytestmark = pytest.mark.unit


def _write(tmp_path: Path, doc) -> str:
    p = tmp_path / "params.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return str(p)


def test_parse_and_apply_overrides_defaults(tmp_path):
    p = _write(tmp_path, {"tau_rrp": 35.0, "lambda_loge": 0.8})
    cfg = apply_cmaes_params(SynapticConfig(), p)
    assert cfg.tau_rrp == 35.0
    assert cfg.lambda_loge == 0.8
    # Untouched fields keep their defaults.
    assert cfg.camkii_up == SynapticConfig().camkii_up


def test_unknown_key_lists_closest_valid_names(tmp_path):
    p = _write(tmp_path, {"tau_rrps": 1.0})  # typo of tau_rrp
    with pytest.raises(ValueError) as ei:
        parse_cmaes_params(p)
    msg = str(ei.value)
    assert "unknown SynapticConfig field" in msg and "tau_rrp" in msg


def test_bool_value_rejected_deliberately(tmp_path):
    p = _write(tmp_path, {"enable_hebbian": True})
    with pytest.raises(ValueError) as ei:
        parse_cmaes_params(p)
    assert "Booleans are rejected" in str(ei.value)


def test_non_numeric_value_rejected(tmp_path):
    p = _write(tmp_path, {"tau_rrp": "very fast"})
    with pytest.raises(ValueError) as ei:
        parse_cmaes_params(p)
    assert "must be a number" in str(ei.value)


def test_non_object_top_level_rejected(tmp_path):
    p = tmp_path / "list.json"
    p.write_text(json.dumps([1.0, 2.0]), encoding="utf-8")
    with pytest.raises(ValueError) as ei:
        parse_cmaes_params(str(p))
    assert "JSON object" in str(ei.value)


def test_invalid_json_reports_line_and_column(tmp_path):
    p = tmp_path / "broken.json"
    p.write_text("{\n  \"tau_rrp\": ,\n}", encoding="utf-8")
    with pytest.raises(ValueError) as ei:
        parse_cmaes_params(p)
    assert "line 2" in str(ei.value)


def test_empty_object_rejected(tmp_path):
    p = _write(tmp_path, {})
    with pytest.raises(ValueError) as ei:
        parse_cmaes_params(p)
    assert "no parameters" in str(ei.value)


def test_missing_file_actionable_error(tmp_path):
    with pytest.raises(ValueError) as ei:
        parse_cmaes_params(tmp_path / "nope.json")
    assert "could not be read" in str(ei.value)


def test_base_train_wiring_present():
    """The flag default, the overlay call, and the resume refusal exist in base_train."""
    src = Path("scripts/base_train.py").read_text(encoding="utf-8")
    assert 'load_cmaes_params = ""' in src, "configurator setting must exist"
    assert "apply_cmaes_params(syn_cfg, load_cmaes_params)" in src, "overlay must be wired"
    assert "cannot be combined with --resume" in src, "resume refusal must be present"
