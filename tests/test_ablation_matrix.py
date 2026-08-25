"""
Pre-registered bio-vs-vanilla ablation matrix (bead hwxb.5.1).

Locks the experiment spec so it stays internally consistent and runnable:
  - the three decomposition anchors exist (vanilla / synaptic_off / bio_all),
  - every column produces a config that VALIDATES clean (no foot-gun ablations),
  - the synaptic_off anchor truly neutralizes every default-on mechanism,
  - add-one-in columns respect mechanism prerequisites,
  - leave-one-out / add-one-in are DERIVED from the registry (cannot drift),
  - the GPU-hour estimate + go/no-go gate behave (the cost guardrail), and
  - the confirmation pass keeps the anchors and only the survivors.

Run:  pytest tests/test_ablation_matrix.py -v
"""

from __future__ import annotations

import pytest

from bio_inspired_nanochat import ablation_matrix as am
from bio_inspired_nanochat.ablation_registry import MECHANISMS, is_mechanism_on
from bio_inspired_nanochat.synaptic import SynapticConfig

pytestmark = pytest.mark.unit


def test_three_anchors_present_and_distinct():
    ids = [c.config_id for c in am.anchors()]
    assert ids == ["vanilla", "synaptic_off", "bio_all"]
    assert all(c.role == "anchor" for c in am.anchors())


def test_vanilla_anchor_has_no_synaptic_config():
    (vanilla,) = [c for c in am.anchors() if c.config_id == "vanilla"]
    assert vanilla.build_syn_cfg() is None, "vanilla is the standard GPT — no SynapticConfig"


def test_synaptic_off_neutralizes_every_default_on_mechanism():
    (off,) = [c for c in am.anchors() if c.config_id == "synaptic_off"]
    cfg = off.build_syn_cfg()
    assert cfg is not None
    for m in MECHANISMS:
        if m.default_on:
            assert not is_mechanism_on(cfg, m.field), f"{m.mechanism} must be OFF in synaptic_off"


def test_bio_all_anchor_is_default_config():
    (bio,) = [c for c in am.anchors() if c.config_id == "bio_all"]
    cfg = bio.build_syn_cfg()
    # bio_all is the unmodified default synaptic stack.
    assert cfg == SynapticConfig()


def test_every_column_validates_clean():
    for c in am.screening_columns():
        cfg = c.build_syn_cfg()  # raises on an invalid (foot-gun) config
        if c.base is not am.Base.VANILLA:
            assert cfg is not None


def test_leave_one_out_covers_exactly_the_default_on_science_mechanisms():
    expected = {
        m.mechanism for m in MECHANISMS
        if m.default_on and m.mechanism not in am.MATRIX_EXCLUDED_MECHANISMS
    }
    got = {c.config_id.removeprefix("bio_no_") for c in am.leave_one_out()}
    assert got == expected
    for c in am.leave_one_out():
        assert c.base is am.Base.BIO_ALL and c.role == "leave_one_out"


def test_leave_one_out_actually_turns_the_mechanism_off():
    for c in am.leave_one_out():
        mech = c.config_id.removeprefix("bio_no_")
        (m,) = [x for x in MECHANISMS if x.mechanism == mech]
        cfg = c.build_syn_cfg()
        assert cfg is not None
        assert not is_mechanism_on(cfg, m.field), f"{mech} must be off in {c.config_id}"


def test_genome_ablation_uses_shared_kinetics():
    (column,) = [c for c in am.leave_one_out() if c.config_id == "bio_no_genome"]
    cfg = column.build_syn_cfg()
    assert cfg is not None and cfg.xi_dim == 0


def test_add_one_in_covers_exactly_the_optin_science_mechanisms():
    expected = {
        m.mechanism for m in MECHANISMS
        if not m.default_on and m.mechanism not in am.MATRIX_EXCLUDED_MECHANISMS
    }
    got = {c.config_id.removeprefix("add_") for c in am.add_one_in()}
    assert got == expected


def test_add_one_in_enables_mechanism_and_its_prerequisites():
    for c in am.add_one_in():
        mech = c.config_id.removeprefix("add_")
        (m,) = [x for x in MECHANISMS if x.mechanism == mech]
        cfg = c.build_syn_cfg()
        assert cfg is not None
        assert is_mechanism_on(cfg, m.field), f"{mech} must be ON in {c.config_id}"
        for prereq in am._prereq_closure(m.field):
            assert is_mechanism_on(cfg, prereq), f"prereq {prereq} of {mech} must be ON"


def test_infra_mechanisms_are_excluded_from_the_science_matrix():
    all_ids = {c.config_id for c in am.screening_columns()}
    for infra in am.INFRA_MECHANISMS:
        assert f"bio_no_{infra}" not in all_ids and f"add_{infra}" not in all_ids


def test_controller_driven_topology_is_excluded_from_config_only_matrix():
    all_ids = {c.config_id for c in am.screening_columns()}
    assert "add_topological_nas" not in all_ids


def test_screening_is_anchors_plus_both_directions():
    cols = am.screening_columns()
    assert len(cols) == len(am.anchors()) + len(am.leave_one_out()) + len(am.add_one_in())


def test_confirmation_keeps_anchors_and_only_survivors():
    survivors = ["bio_no_presyn", "add_learnable_kinetics"]
    cols = am.confirmation_columns(survivors)
    ids = {c.config_id for c in cols}
    assert {"vanilla", "synaptic_off", "bio_all"} <= ids
    assert "bio_no_presyn" in ids and "add_learnable_kinetics" in ids
    assert "bio_no_hebbian" not in ids, "a non-survivor must be dropped from confirmation"


def test_seeds_are_research_grade():
    assert len(am.CONFIRMATION_SEEDS) >= 3, "confirmation needs >=3 seeds for significance"
    assert len(set(am.CONFIRMATION_SEEDS)) == len(am.CONFIRMATION_SEEDS), "seeds must be distinct"


def test_gpu_hour_estimate_scales_with_runs_and_budget():
    cols = am.screening_columns()
    h1 = am.estimate_gpu_hours(cols, (1,), 10_000_000, tok_per_sec=10_000.0)
    h2 = am.estimate_gpu_hours(cols, (1, 2), 10_000_000, tok_per_sec=10_000.0)
    h_big = am.estimate_gpu_hours(cols, (1,), 20_000_000, tok_per_sec=10_000.0)
    assert h2 == pytest.approx(2 * h1)
    assert h_big == pytest.approx(2 * h1)
    with pytest.raises(ValueError):
        am.estimate_gpu_hours(cols, (1,), 10_000_000, tok_per_sec=0.0)


def test_go_no_go_blocks_when_no_survivors():
    g = am.go_no_go([], tok_per_sec=20_000.0)
    assert g.proceed is False and g.n_survivors == 0


def test_go_no_go_blocks_when_over_the_cap():
    g = am.go_no_go(["bio_no_presyn"], tok_per_sec=10.0, cap_gpu_hours=1.0)
    assert g.proceed is False and g.estimated_gpu_hours > g.cap_gpu_hours


def test_go_no_go_proceeds_within_budget():
    # Feasible design: n=6 matched seeds clears the exact-Wilcoxon floor for a
    # single-survivor family (2*2^-6 = 0.03125 <= alpha), so the pass may proceed.
    g = am.go_no_go(["bio_no_presyn"], tok_per_sec=1_000_000.0, cap_gpu_hours=100.0,
                    seeds=tuple(1000 + i for i in range(6)))
    assert g.proceed is True and g.n_survivors == 1


def test_go_no_go_blocks_statistically_infeasible_design():
    """The pinned 3-seed design can NEVER fire the pre-registered rule: the exact
    Wilcoxon two-sided floor is 2*2^-3 = 0.25 > alpha regardless of effect size,
    so every verdict would be forced to 'null'. go_no_go must refuse to commit
    GPU-hours to such a run instead of authorizing a foregone conclusion."""
    g = am.go_no_go(["bio_no_presyn"], tok_per_sec=1_000_000.0, cap_gpu_hours=100.0)
    assert g.proceed is False
    assert "INFEASIBLE" in g.reason
    assert "Raise seeds" in g.reason


def test_decision_rule_primary_metric_is_a_declared_metric():
    assert am.DECISION_PRIMARY_METRIC in am.METRICS
    assert 0.0 < am.DECISION_CONFIDENCE < 1.0
