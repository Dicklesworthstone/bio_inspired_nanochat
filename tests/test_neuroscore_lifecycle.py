"""NeuroScore wired into the split/merge lifecycle — bead de5l.

The README presents NeuroScore (Efficiency / Specialization / Resilience) as the
evolutionary credit-assignment engine driving split/merge, but the controller used
health = utilization*energy only — NeuroScore was computed and never consumed.

This wires it: NeuroScore.step publishes a per-expert composite fitness onto each
SynapticMoE (``last_neuroscore``), and SplitMergeController._health blends it in when
``SplitMergeConfig.use_neuroscore`` is set (default-off). These tests prove the
composite is sane, that the blend is correct, and — the load-bearing claim — that
turning it on actually CHANGES which experts are split / reset / merged.

Run:  pytest tests/test_neuroscore_lifecycle.py -v
"""

from __future__ import annotations

import pytest
import torch

from bio_inspired_nanochat.neuroscore import NeuroScore, NeuroScoreConfig
from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticMoE
from bio_inspired_nanochat.synaptic_splitmerge import (
    SplitMergeConfig,
    SplitMergeController,
)

pytestmark = pytest.mark.unit


def _moe(num_experts: int = 4) -> SynapticMoE:
    cfg = SynapticConfig(enable_hebbian=False, enable_metabolism=True)
    return SynapticMoE(n_embd=8, num_experts=num_experts, top_k=2, hidden_mult=1, cfg=cfg)


def _set_health(moe: SynapticMoE, health: list[float]) -> None:
    # health = fatigue*energy; pin energy=1 so fatigue IS health.
    with torch.no_grad():
        moe.energy.copy_(torch.ones(moe.num_experts))
        moe.fatigue.copy_(torch.tensor(health))


# --------------------------------------------------------------------------- #
# 1. composite_fitness: valid [0,1], ranks fit experts above unfit, neutral when flat
# --------------------------------------------------------------------------- #
def test_composite_fitness_in_unit_interval_and_orders_experts():
    eff = torch.tensor([0.1, 0.5, 0.9])
    spec = torch.tensor([0.0, 0.4, 1.0])
    res = torch.tensor([0.2, 0.5, 0.8])
    comp = NeuroScore.composite_fitness(eff, spec, res)
    assert comp.shape == (3,)
    assert (comp >= 0).all() and (comp <= 1).all()
    # Expert 2 dominates all three metrics -> highest composite; expert 0 lowest.
    assert comp[2] > comp[1] > comp[0]


def test_composite_fitness_degenerate_metric_is_neutral():
    flat = torch.full((4,), 3.0)
    comp = NeuroScore.composite_fitness(flat, flat, flat)
    assert torch.allclose(comp, torch.full((4,), 0.5)), "all-equal metrics -> neutral 0.5"


# --------------------------------------------------------------------------- #
# 2. _health blend: off by default, exact when on, graceful fallback
# --------------------------------------------------------------------------- #
def test_health_ignores_neuroscore_when_disabled():
    moe = _moe()
    _set_health(moe, [0.9, 0.6, 0.3, 0.1])
    object.__setattr__(moe, "last_neuroscore", torch.tensor([0.1, 0.3, 0.6, 0.9]))
    ctrl = SplitMergeController(moe, SplitMergeConfig(use_neuroscore=False))
    assert torch.allclose(ctrl._health(moe), torch.tensor([0.9, 0.6, 0.3, 0.1]))


def test_health_blends_neuroscore_when_enabled():
    moe = _moe()
    _set_health(moe, [0.9, 0.6, 0.3, 0.1])
    score = torch.tensor([0.1, 0.3, 0.6, 0.9])
    object.__setattr__(moe, "last_neuroscore", score)
    ctrl = SplitMergeController(
        moe, SplitMergeConfig(use_neuroscore=True, neuroscore_weight=0.5)
    )
    expected = 0.5 * torch.tensor([0.9, 0.6, 0.3, 0.1]) + 0.5 * score
    assert torch.allclose(ctrl._health(moe), expected)


@pytest.mark.parametrize("bad", [None, torch.zeros(3)])  # missing / wrong-shaped
def test_health_falls_back_when_score_absent_or_mismatched(bad):
    moe = _moe(num_experts=4)
    _set_health(moe, [0.9, 0.6, 0.3, 0.1])
    object.__setattr__(moe, "last_neuroscore", bad)
    ctrl = SplitMergeController(moe, SplitMergeConfig(use_neuroscore=True))
    assert torch.allclose(ctrl._health(moe), torch.tensor([0.9, 0.6, 0.3, 0.1]))


# --------------------------------------------------------------------------- #
# 3. The load-bearing claim: NeuroScore changes which experts are selected
# --------------------------------------------------------------------------- #
def test_neuroscore_inverts_split_and_reset_selection():
    moe = _moe(num_experts=4)
    _set_health(moe, [0.9, 0.6, 0.3, 0.1])  # health: expert 0 fittest, 3 weakest
    # NeuroScore says the opposite — expert 3 is the most valuable.
    object.__setattr__(moe, "last_neuroscore", torch.tensor([0.1, 0.3, 0.6, 0.9]))

    base = SplitMergeConfig(use_neuroscore=False, split_health_min=0.0, splits_per_call=4)
    ns = SplitMergeConfig(
        use_neuroscore=True, neuroscore_weight=1.0, split_health_min=0.0, splits_per_call=4
    )
    ctrl_base = SplitMergeController(moe, base)
    ctrl_ns = SplitMergeController(moe, ns)

    # Split SOURCES are the strongest experts (descending health).
    assert ctrl_base._pick_split_sources(moe) == [0, 1, 2, 3]
    assert ctrl_ns._pick_split_sources(moe) == [3, 2, 1, 0], "NeuroScore must reorder splits"

    # Weakest slots (split/reset targets) are the lowest health (ascending).
    assert ctrl_base._weakest_slots(moe, 4) == [3, 2, 1, 0]
    assert ctrl_ns._weakest_slots(moe, 4) == [0, 1, 2, 3], "NeuroScore must reorder weak slots"

    # Reset SOURCES (healthiest, to clone from) likewise flip.
    assert ctrl_base._pick_reset_sources(moe, 4) == [0, 1, 2, 3]
    assert ctrl_ns._pick_reset_sources(moe, 4) == [3, 2, 1, 0]


# --------------------------------------------------------------------------- #
# 3b. Dead experts must not be scored as maximally specialized
# --------------------------------------------------------------------------- #
def test_unused_expert_keeps_prior_specialization_not_max():
    """An expert that received NO tokens this batch has an all-zero centroid, so its cosine
    similarity to the global mean is 0 and ``1 - sim`` would be 1.0 — i.e. a DEAD expert would
    look MAXIMALLY specialized, inverting the lifecycle signal (protecting it from reset,
    making it an attractive split source). Unused experts must instead keep their prior score."""
    torch.manual_seed(0)
    score = NeuroScore(NeuroScoreConfig(enabled=True), neuroviz=None)
    num_experts, B, T, C, k = 4, 8, 32, 8, 2
    x = torch.randn(B, T, C)
    indices = torch.randint(0, num_experts - 1, (B, T, k))  # routing never selects the last expert
    prior = 0.25
    st = {"specialization": torch.full((num_experts,), prior)}

    score._update_specialization(st, x, indices, num_experts)

    dead = num_experts - 1
    assert st["specialization"][dead].item() == pytest.approx(prior), (
        "an unused expert must keep its prior specialization, not the spurious 1.0"
    )
    assert st["specialization"][dead].item() != pytest.approx(1.0)
    # and the used experts WERE updated (the method actually ran, not an early return)
    assert not torch.allclose(st["specialization"], torch.full((num_experts,), prior))


def test_unused_expert_does_not_accrue_resilience():
    """Resilience = 1/|Δcontribution|. A DEAD expert also has ~constant (zero) contribution, so an
    ungated 1/|Δ| would blow up (Δ≈0 ⟹ 1e6) and score it as maximally resilient — protecting it from
    reset. The activity gate must keep a never-routed expert's resilience ≈ 0 (sibling of the
    dead-expert specialization fix)."""
    torch.manual_seed(0)
    num_experts, B, T, C, k = 4, 2, 8, 8, 2
    moe = _moe(num_experts)
    dead = num_experts - 1
    x = torch.randn(B, T, C)
    indices = torch.randint(0, num_experts - 1, (B, T, k))  # routing NEVER selects the dead expert

    score = NeuroScore(NeuroScoreConfig(enabled=True, update_every=1000), neuroviz=None)
    for s in range(20):
        # re-randomize gates each step so ACTIVE experts have genuine contribution variation
        object.__setattr__(moe, "last_ctx", {"gates": torch.rand(B, T, k), "indices": indices, "x": x})
        score.step(moe, loss=torch.tensor(1.0), global_step=s)

    res = score.stats[""]["resilience"]  # the moe itself registers under name ""
    assert res[dead].item() == pytest.approx(0.0, abs=1e-3), (
        "a never-routed expert must not accrue the spurious 1/Δ resilience spike"
    )
    assert res[dead].item() < res[:dead].max().item(), "active experts must out-score the dead one"


# --------------------------------------------------------------------------- #
# 4. End-to-end: NeuroScore.step publishes a usable score onto the live module
# --------------------------------------------------------------------------- #
def test_neuroscore_step_publishes_score_onto_module():
    moe = _moe(num_experts=4)
    # A forward pass populates last_ctx (routing gates/indices/x) that NeuroScore reads.
    moe(torch.randn(2, 6, 8))
    assert moe.last_neuroscore is None, "no score before NeuroScore runs"

    score = NeuroScore(NeuroScoreConfig(enabled=True, update_every=1), neuroviz=None)
    score.step(moe, loss=torch.tensor(1.0), global_step=0)

    published = moe.last_neuroscore
    assert published is not None, "NeuroScore.step must publish last_neuroscore"
    assert published.shape == (moe.num_experts,)
    assert (published >= 0).all() and (published <= 1).all()
    # And the controller actually consumes it without a hand-set value.
    ctrl = SplitMergeController(moe, SplitMergeConfig(use_neuroscore=True, neuroscore_weight=1.0))
    assert torch.allclose(ctrl._health(moe), published.float())
