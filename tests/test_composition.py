"""Composition keystone — guard table + pairwise interference harness (beads `0642.10.2` / `0642.10.3`).

Checks that the timescale-separation gauge (`0642.10.1`) gates the thrust certificates correctly:

  - the **eligibility guard** marks a thrust `compose` iff its required coupling is separated, and
    `fallback` otherwise — honestly flagging the defaults' unseparated `release→fast_weights` coupling;
  - the **pairwise interference harness** lets two thrusts co-activate only if every boundary between
    their strata is separated, and otherwise disables the **higher-risk** thrust;
  - a fully-separated config composes everything; the defaults do not (a real, honest finding).

Run:  pytest tests/test_composition.py -v
"""

from __future__ import annotations

import json

import pytest

from bio_inspired_nanochat import composition as comp
from bio_inspired_nanochat.synaptic import SynapticConfig

pytestmark = pytest.mark.unit


# A config whose timescales actually separate at every boundary (calcium ≪ … ≪ structure).
def _well_separated() -> SynapticConfig:
    return SynapticConfig(tau_rrp=20.0, post_trace_decay=0.99, post_slow_lr=0.0025, structural_interval=4000)


# --------------------------------------------------------------------------- #
# 0642.10.2 — eligibility guard table
# --------------------------------------------------------------------------- #
def test_eligibility_flags_the_default_unseparated_coupling():
    elig = comp.composition_eligibility(SynapticConfig())
    # Defaults separate everything except release→fast_weights, so only thrust F falls back.
    assert elig["A"].eligible and elig["E"].eligible and elig["B"].eligible and elig["C"].eligible
    assert not elig["F"].eligible and elig["F"].verdict == "fallback"
    assert elig["F"].boundary == "release → fast_weights" and elig["F"].eps > 0.5


def test_eligibility_all_compose_when_fully_separated():
    elig = comp.composition_eligibility(_well_separated())
    assert all(e.eligible and e.verdict == "compose" for e in elig.values())


def test_eligibility_responds_to_eps_max():
    # A stricter threshold can trip an otherwise-eligible thrust; a looser one rescues it.
    cfg = SynapticConfig()
    assert comp.composition_eligibility(cfg, eps_max=0.1)["A"].eligible is False  # 0.163 ≥ 0.1
    assert comp.composition_eligibility(cfg, eps_max=0.2)["A"].eligible is True


# --------------------------------------------------------------------------- #
# 0642.10.3 — pairwise interference harness (representative pairs A+F, A+E, F+B)
# --------------------------------------------------------------------------- #
def test_pairwise_representative_pairs_at_defaults():
    cfg = SynapticConfig()
    af = comp.pairwise_compatible(cfg, "A", "F")
    ae = comp.pairwise_compatible(cfg, "A", "E")
    fb = comp.pairwise_compatible(cfg, "F", "B")
    # A+F straddles the unseparated release→fast_weights boundary ⟹ incompatible, disable higher-risk F.
    assert not af.compatible and af.disabled == "F"
    assert "release→fast_weights" in af.unseparated_boundaries
    # A+E live in the separated calcium↔release span ⟹ compatible.
    assert ae.compatible and ae.disabled is None
    # F+B span only the separated fast_weights↔slow_weights boundary ⟹ compatible.
    assert fb.compatible and fb.disabled is None


def test_pairwise_disables_the_higher_risk_thrust():
    cfg = SynapticConfig()
    # Across the unseparated boundary, the more disruptive thrust is the one disabled.
    assert comp.THRUSTS["F"].risk > comp.THRUSTS["A"].risk
    assert comp.pairwise_compatible(cfg, "A", "C").disabled == "C"  # C (structural) is highest risk
    assert comp.pairwise_compatible(cfg, "E", "B").disabled == "B"


def test_all_pairs_compatible_when_fully_separated():
    rep = comp.composition_report(_well_separated())
    assert rep["all_eligible"] and rep["all_pairs_compatible"]
    assert all(v.compatible for v in comp.compatibility_matrix(_well_separated()).values())


def test_defaults_are_not_all_compatible():
    rep = comp.composition_report(SynapticConfig())
    assert not rep["all_pairs_compatible"], "the defaults must honestly report the unseparated coupling"


def test_compatibility_matrix_covers_all_unordered_pairs():
    m = comp.compatibility_matrix(SynapticConfig())
    assert len(m) == 10  # C(5,2)
    assert "A+F" in m and "B+C" in m and "F+A" not in m  # unordered, canonical order


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def test_report_and_jsonl_are_well_formed():
    cfg = SynapticConfig()
    rep = comp.composition_report(cfg)
    assert {"eps_max", "eligibility", "pairwise", "all_eligible", "all_pairs_compatible"} <= set(rep)
    lines = comp.composition_report_jsonl(cfg)
    assert len(lines) == 5 + 10  # 5 eligibility + 10 pairwise
    kinds = {json.loads(line)["kind"] for line in lines}
    assert kinds == {"eligibility", "pairwise"}
