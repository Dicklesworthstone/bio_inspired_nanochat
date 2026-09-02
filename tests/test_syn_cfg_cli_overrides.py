"""``--syn_cfg.<field>=<value>`` training overrides + the registry redirect guard.

The README has documented ``--syn_cfg.tau_rrp=100.0``-style overrides since 2025-11 while
``scripts/base_train.py`` built its ``SynapticConfig`` from two flags and rejected the dotted
keys as unknown settings. These tests lock the feature that closes that gap:

* :func:`extract_syn_cfg_cli_overrides` pulls the dotted arguments out of ``argv`` so the
  nanochat configurator never sees them;
* :func:`coerce_syn_cfg_override` types each value from the dataclass field it targets and
  fails closed on typos and wrong kinds;
* :func:`apply_syn_cfg_overrides` overlays them and runs both validators.

Planted negatives: an unknown field, a fractional integer, an unsupported ``stochastic_mode``
literal, and an opt-in mechanism enabled without its prerequisite. Green here does NOT prove the
overrides reach a *training run* end to end; that wiring is locked by source assertions on the
two integration points in ``base_train`` (the same contract the CMA-ES loader uses), because
the script needs FineWeb shards and a tokenizer to execute.

Run:  pytest tests/test_syn_cfg_cli_overrides.py -v
"""

from __future__ import annotations

import os
from dataclasses import fields
from pathlib import Path

import pytest

from bio_inspired_nanochat.cmaes_params import (
    SYN_CFG_CLI_PREFIX,
    apply_syn_cfg_overrides,
    coerce_syn_cfg_override,
    extract_syn_cfg_cli_overrides,
)
from bio_inspired_nanochat.synaptic import SynapticConfig

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# extract
# --------------------------------------------------------------------------- #
def test_extract_separates_dotted_overrides_and_keeps_argv_order():
    argv = [
        "base_train.py",
        "--depth=4",
        f"{SYN_CFG_CLI_PREFIX}tau_rrp=100.0",
        "--synapses=1",
        f"{SYN_CFG_CLI_PREFIX}bistable_latch=1",
        "--num_iterations=20",
    ]
    remaining, overrides = extract_syn_cfg_cli_overrides(argv)
    assert remaining == ["base_train.py", "--depth=4", "--synapses=1", "--num_iterations=20"]
    assert overrides == {"tau_rrp": "100.0", "bistable_latch": "1"}


def test_extract_without_overrides_is_identity():
    argv = ["base_train.py", "--depth=4"]
    remaining, overrides = extract_syn_cfg_cli_overrides(argv)
    assert remaining == argv and overrides == {}


def test_extract_rejects_missing_value_and_duplicates():
    with pytest.raises(ValueError, match="--syn_cfg.<field>=<value>"):
        extract_syn_cfg_cli_overrides([f"{SYN_CFG_CLI_PREFIX}tau_rrp"])
    with pytest.raises(ValueError, match="given twice"):
        extract_syn_cfg_cli_overrides(
            [f"{SYN_CFG_CLI_PREFIX}tau_rrp=1", f"{SYN_CFG_CLI_PREFIX}tau_rrp=2"]
        )
    with pytest.raises(ValueError, match="missing the SynapticConfig field name"):
        extract_syn_cfg_cli_overrides([f"{SYN_CFG_CLI_PREFIX}=3"])


# --------------------------------------------------------------------------- #
# coerce
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [("1", True), ("true", True), ("YES", True), ("0", False), ("false", False), ("off", False)],
)
def test_coerce_bool_accepts_common_spellings(raw, expected):
    assert coerce_syn_cfg_override("bistable_latch", raw) is expected


def test_coerce_bool_rejects_garbage():
    with pytest.raises(ValueError, match="expected a boolean"):
        coerce_syn_cfg_override("bistable_latch", "maybe")


def test_coerce_int_accepts_whole_numbers_and_rejects_fractions():
    assert coerce_syn_cfg_override("rank_eligibility", "16") == 16
    assert coerce_syn_cfg_override("rank_eligibility", "16.0") == 16
    assert type(coerce_syn_cfg_override("rank_eligibility", "16.0")) is int
    with pytest.raises(ValueError, match="expected an integer"):
        coerce_syn_cfg_override("rank_eligibility", "8.5")
    with pytest.raises(ValueError, match="expected an integer"):
        coerce_syn_cfg_override("rank_eligibility", "eight")


def test_coerce_float_requires_finite_number():
    assert coerce_syn_cfg_override("tau_rrp", "100.0") == 100.0
    assert coerce_syn_cfg_override("tau_rrp", "40") == 40.0
    with pytest.raises(ValueError, match="finite"):
        coerce_syn_cfg_override("tau_rrp", "inf")
    with pytest.raises(ValueError, match="expected a number"):
        coerce_syn_cfg_override("tau_rrp", "fast")


def test_coerce_string_and_granularity_pass_through_with_quotes_stripped():
    assert coerce_syn_cfg_override("stochastic_mode", "gumbel_sigmoid_ste") == "gumbel_sigmoid_ste"
    assert coerce_syn_cfg_override("stochastic_mode", "'straight_through'") == "straight_through"
    assert coerce_syn_cfg_override("granularity", "per_expert") == "per_expert"


def test_coerce_unknown_field_lists_closest_names():
    with pytest.raises(ValueError) as ei:
        coerce_syn_cfg_override("tau_rrps", "1.0")
    msg = str(ei.value)
    assert "unknown SynapticConfig field" in msg and "tau_rrp" in msg


# --------------------------------------------------------------------------- #
# apply
# --------------------------------------------------------------------------- #
def test_apply_overlays_typed_values_and_keeps_other_defaults():
    cfg = apply_syn_cfg_overrides(
        SynapticConfig(),
        {"tau_rrp": "100.0", "energy_cost_rel": "0.05", "bistable_latch": "1", "rank_eligibility": "16"},
    )
    assert cfg.tau_rrp == 100.0
    assert cfg.energy_cost_rel == 0.05
    assert cfg.bistable_latch is True
    assert cfg.rank_eligibility == 16
    assert cfg.camkii_up == SynapticConfig().camkii_up


def test_apply_rejects_unsupported_stochastic_mode_literal():
    with pytest.raises(ValueError, match="stochastic_mode is not a supported literal"):
        apply_syn_cfg_overrides(SynapticConfig(), {"stochastic_mode": "banana"})


def test_apply_rejects_opt_in_mechanism_without_prerequisite():
    # cusp_latch is registered as requiring bistable_latch: enabling it alone would be a
    # silent no-op, which is exactly the foot-gun the validator exists to catch.
    with pytest.raises(ValueError, match="prerequisite"):
        apply_syn_cfg_overrides(SynapticConfig(), {"cusp_latch": "1"})
    # The same pair enabled together is legal.
    cfg = apply_syn_cfg_overrides(SynapticConfig(), {"cusp_latch": "1", "bistable_latch": "1"})
    assert cfg.cusp_latch is True and cfg.bistable_latch is True


def test_apply_default_config_with_no_overrides_is_valid():
    assert apply_syn_cfg_overrides(SynapticConfig(), {}) == SynapticConfig()


# --------------------------------------------------------------------------- #
# base_train wiring (source-level contract; the script needs data + tokenizer to run)
# --------------------------------------------------------------------------- #
def test_base_train_wiring_present():
    src = Path("scripts/base_train.py").read_text(encoding="utf-8")
    assert "sys.argv, syn_cfg_overrides = extract_syn_cfg_cli_overrides(sys.argv)" in src, (
        "dotted overrides must be stripped from argv before the configurator runs"
    )
    assert "apply_syn_cfg_overrides(syn_cfg, syn_cfg_overrides)" in src, "overlay must be wired"
    assert "require synapses=1" in src, "vanilla runs must refuse --syn_cfg.* overrides"
    assert "cannot be combined with --resume" in src, "resume must refuse config overrides"
    assert 'user_config["syn_cfg_overrides"]' in src, "overrides must reach the run record"


def test_base_train_exposes_homeostasis_guards():
    src = Path("scripts/base_train.py").read_text(encoding="utf-8")
    assert "sm_homeostasis_guards = 0" in src
    assert "homeostasis_guards=bool(sm_homeostasis_guards)" in src
    assert "gate_ramp_forwards=int(sm_gate_ramp_forwards)" in src
    assert "energy_floor=float(sm_energy_floor)" in src


def test_structural_every_no_op_hook_is_gone():
    """The old ``structural_every`` config knob ran an empty ``pass`` block per layer."""
    from bio_inspired_nanochat.gpt_synaptic import GPTSynapticConfig

    assert "structural_every" not in {f.name for f in fields(GPTSynapticConfig)}
    src = Path("bio_inspired_nanochat/gpt_synaptic.py").read_text(encoding="utf-8")
    assert "structural_every" not in src


# --------------------------------------------------------------------------- #
# results registry redirect (tests must never append to the committed corpus)
# --------------------------------------------------------------------------- #
def test_pytest_redirects_default_results_registry_away_from_tracked_file():
    from bio_inspired_nanochat import results_registry

    tracked = os.path.join("results", "registry.jsonl")
    assert os.environ.get("BIO_RESULTS_REGISTRY"), "conftest must set BIO_RESULTS_REGISTRY"
    assert results_registry.DEFAULT_REGISTRY != tracked
    assert results_registry.DEFAULT_REGISTRY == os.environ["BIO_RESULTS_REGISTRY"]
    assert "bio-nanochat-test-registry" in results_registry.DEFAULT_REGISTRY


def test_tune_bio_params_registry_default_follows_the_redirect():
    import scripts.tune_bio_params as tune
    from bio_inspired_nanochat import results_registry

    parser_default = tune.DEFAULT_REGISTRY
    assert parser_default == results_registry.DEFAULT_REGISTRY
    assert parser_default != os.path.join("results", "registry.jsonl")
