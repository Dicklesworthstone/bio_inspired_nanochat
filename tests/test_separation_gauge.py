"""
Timescale-separation gauge + per-coupling separation table (bead 0642.10.1).

Locks the ε_k gauge that the singular-perturbation reductions (Thrust A/F) rely on:

  - every level of the hierarchy gets a positive characteristic timescale from the config;
  - the calcium fast-Jacobian eigenvalue gap is a contraction (0,1);
  - the table computes ε_k = τ_fast/τ_slow per consecutive boundary and flags separated ⟺ ε_k < eps_max;
  - it HONESTLY reports where the configured timescales do NOT separate (the default release↔fast-weights
    boundary overlaps) rather than asserting the intended order holds;
  - the ratios respond monotonically to the underlying time constants.

Run:  pytest tests/test_separation_gauge.py -v
"""

from __future__ import annotations

import pytest

from bio_inspired_nanochat import separation_gauge as sg
from bio_inspired_nanochat.synaptic import SynapticConfig

pytestmark = pytest.mark.unit


def test_all_hierarchy_levels_have_positive_timescales():
    tau = sg.coupling_timescales(SynapticConfig())
    assert set(tau) == set(sg.HIERARCHY)
    assert all(v > 0 for v in tau.values())
    # The hierarchy must be strictly ordered fast→slow at least at the slow end (well-separated there).
    assert tau["slow_weights"] < tau["structure"]
    assert tau["fast_weights"] < tau["slow_weights"]


def test_calcium_eigenvalue_gap_is_a_contraction():
    gap = sg.calcium_eigenvalue_gap(SynapticConfig())
    assert 0.0 < gap < 1.0, f"the fast-Jacobian gap must be a contraction margin, got {gap}"
    # τ_calcium = 1/gap should be a few steps at defaults.
    assert 1.0 < 1.0 / gap < 50.0


def test_separation_table_shape_and_eps_definition():
    rows = sg.separation_table(SynapticConfig(), eps_max=0.5)
    assert len(rows) == len(sg.HIERARCHY) - 1
    for r in rows:
        assert r.eps == pytest.approx(r.tau_fast / r.tau_slow)
        assert r.separated == (r.eps < 0.5)
    # Consecutive rows chain through the hierarchy.
    assert [r.fast for r in rows] == list(sg.HIERARCHY[:-1])
    assert [r.slow for r in rows] == list(sg.HIERARCHY[1:])


def test_slow_end_is_well_separated_fast_end_is_not_at_default():
    rows = {(r.fast, r.slow): r for r in sg.separation_table(SynapticConfig())}
    # The slow couplings cleanly separate.
    assert rows[("fast_weights", "slow_weights")].separated
    assert rows[("slow_weights", "structure")].separated
    assert rows[("calcium", "release")].separated
    # The honest finding: release and fast-weights overlap in the default config (ε_k ≥ eps_max).
    assert not rows[("release", "fast_weights")].separated
    assert not sg.is_well_separated(SynapticConfig())


def test_is_well_separated_true_under_a_loose_threshold():
    # With a permissive eps_max every default boundary counts as separated.
    assert sg.is_well_separated(SynapticConfig(), eps_max=5.0)


def test_separation_responds_to_timescale_changes():
    base = sg.coupling_timescales(SynapticConfig())
    slower_structure = sg.coupling_timescales(SynapticConfig(structural_interval=200000))
    assert slower_structure["structure"] > base["structure"]
    # A faster slow-weight LR shortens the slow-weight timescale.
    faster_slow = sg.coupling_timescales(SynapticConfig(post_slow_lr=5e-3))
    assert faster_slow["slow_weights"] < base["slow_weights"]


def test_render_table_is_nonempty_and_labels_each_boundary():
    text = sg.render_separation_table(SynapticConfig())
    assert "separation table" in text and "eigenvalue gap" in text
    for level in sg.HIERARCHY:
        assert level in text
