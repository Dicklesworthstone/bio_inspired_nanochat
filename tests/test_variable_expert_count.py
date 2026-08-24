"""Variable expert count under a budget — bead uta.4.

Real neurogenesis/apoptosis: SplitMergeController can append fresh expert slots
under sustained split pressure and remove surplus dead slots (folding their mass
into the healthiest survivor), rebuilding router/buffers/genome in place and
synchronizing optimizer param-groups — survivor moments survive verbatim, new
params start fresh, removed params are released.
"""

from __future__ import annotations

import pytest
import torch

from bio_inspired_nanochat.neuroscore import NeuroScore, NeuroScoreConfig
from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticMoE
from bio_inspired_nanochat.synaptic_splitmerge import (
    SplitMergeConfig,
    SplitMergeController,
    _resize_layer_experts_,
    capture_optimizer_layout,
    snapshot_optimizer_state,
    synchronize_optimizers_with_model,
)

pytestmark = pytest.mark.unit


def _moe(num_experts: int = 4, n_embd: int = 8, seed: int = 0) -> SynapticMoE:
    torch.manual_seed(seed)
    cfg = SynapticConfig(enable_hebbian=False, enable_metabolism=True, stochastic_train_frac=0.0)
    return SynapticMoE(
        n_embd=n_embd, num_experts=num_experts, top_k=num_experts, hidden_mult=1, cfg=cfg
    )


def _set_health(moe: SynapticMoE, health: list[float]) -> None:
    """health = fatigue * energy; pin energy=1 so fatigue IS health."""
    with torch.no_grad():
        moe.energy.copy_(torch.ones(moe.num_experts))
        moe.fatigue.copy_(torch.tensor(health))


def _shapes_consistent(moe: SynapticMoE) -> None:
    E = int(moe.num_experts)
    assert len(moe.experts) == E
    assert moe.router.weight.shape == (E, moe.router.in_features)
    assert moe.router_logit_bias.shape == (E,)
    assert moe.fatigue.shape == (E,) and moe.energy.shape == (E,)
    assert moe.Xi.shape[0] == E and moe.router_embeddings.shape[0] == E


# --------------------------------------------------------------------------- #
# 1. Growth: appended slots are complete, wired, and survivors untouched
# --------------------------------------------------------------------------- #
def test_growth_appends_complete_experts_and_preserves_survivors():
    moe = _moe(num_experts=4)
    old_W = moe.router.weight.detach().clone()
    old_Xi = moe.Xi.detach().clone()
    old_fc0 = [p.detach().clone() for p in moe.experts[0].parameters()]

    touched = _resize_layer_experts_(moe, target_E=6, seed_idx=0, cfg=SplitMergeConfig())

    assert touched == [4, 5] and int(moe.num_experts) == 6
    _shapes_consistent(moe)
    # survivors' router/genome rows must be bit-exact
    assert torch.equal(moe.router.weight[:4], old_W)
    assert torch.equal(moe.Xi[:4], old_Xi)
    assert torch.allclose(
        moe.router_logit_bias[4:], torch.full((2,), -0.6931471805599453)
    )
    # seed expert itself untouched by the spawn
    for p, ref in zip(moe.experts[0].parameters(), old_fc0):
        assert torch.equal(p, ref)
    # forward works at the new width and is finite
    y, aux = moe(torch.randn(2, 6, 8))
    assert torch.isfinite(y).all() and torch.isfinite(aux).all()


# --------------------------------------------------------------------------- #
# 2. Shrink: dropped rows disappear everywhere, forward stays finite
# --------------------------------------------------------------------------- #
def test_shrink_removes_rows_everywhere():
    moe = _moe(num_experts=5)
    _resize_layer_experts_(moe, target_E=3, seed_idx=0, cfg=SplitMergeConfig())
    assert int(moe.num_experts) == 3
    _shapes_consistent(moe)
    y, _aux = moe(torch.randn(2, 6, 8))
    assert torch.isfinite(y).all()


# --------------------------------------------------------------------------- #
# 3. Controller triggers + optimizer sync helpers
# --------------------------------------------------------------------------- #
def _controller(moe, **cfg_overrides) -> SplitMergeController:
    base = dict(
        variable_expert_count=True,
        splits_per_call=0,  # any strong expert => unmet split demand
        resets_per_call=0,
        min_step_interval=0,
        warmup_steps=0,
    )
    base.update(cfg_overrides)
    cfg = SplitMergeConfig(**base)
    return SplitMergeController(moe, cfg)


def test_optimizer_sync_preserves_survivor_moments_and_drops_removed():
    class Container(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stable = torch.nn.Linear(8, 8)
            self.moe = _moe(num_experts=4, seed=1)

    model = Container()
    opt = torch.optim.AdamW(
        [
            {"params": list(model.stable.parameters()), "lr": 1e-1},
            {"params": list(model.moe.parameters()), "lr": 1e-2},
        ]
    )

    # one real step so every param has optimizer state
    x = torch.randn(2, 6, 8)
    loss = (model.moe(x)[0] * model.stable(x.detach()).sum()).sum() + model.stable(x).sum()
    loss.backward()
    opt.step()

    layout = capture_optimizer_layout([opt], model)
    snap = snapshot_optimizer_state([opt])
    assert len(snap) > 10

    fc1_w_old = model.moe.experts[1].fc1.w_slow
    assert id(fc1_w_old) in snap  # survivor has state pre-surgery

    _resize_layer_experts_(model.moe, target_E=6, seed_idx=1, cfg=SplitMergeConfig())
    synchronize_optimizers_with_model([opt], model, layout, snap)

    live_ids = {id(p) for _, p in model.named_parameters()}
    grouped = [p for g in opt.param_groups for p in g["params"]]
    # exactly-once membership, no stale params
    assert len(grouped) == len(set(map(id, grouped)))
    assert {id(p) for p in grouped} == live_ids
    # survivor moment object survived verbatim
    assert id(fc1_w_old) in snap and fc1_w_old in opt.state
    assert float(opt.state[fc1_w_old]["exp_avg"].abs().sum()) >= 0.0
    # brand-new expert params are in groups but stateless (fresh moments)
    new_param = model.moe.experts[5].fc1.w_slow
    assert any(p is new_param for p in grouped) and not any(
        p is new_param for p in opt.state.keys()
    )
    # training continues cleanly at the new width
    loss = model.moe(torch.randn(2, 6, 8))[0].pow(2).mean()
    loss.backward()
    opt.step()


# --------------------------------------------------------------------------- #
# 4. Controller triggers: pressure-driven growth, dead-surplus shrink,
#    budget cap, hard floors
# --------------------------------------------------------------------------- #


def test_controller_grows_under_split_pressure_and_respects_cap():
    moe = _moe(num_experts=3)
    _set_health(moe, [0.9, 0.9, 0.9])  # 3 strong, splits_per_call=0 => surplus 3
    ctrl = _controller(moe, max_experts=5, growth_budget_pct=10.0)
    ctrl._maybe_resize_layer(moe, optimizer=None, step=0)
    assert int(moe.num_experts) == 5  # capped by max_experts, not by demand(3)
    _shapes_consistent(moe)


def test_growth_stops_at_cumulative_budget():
    moe = _moe(num_experts=2)
    _set_health(moe, [0.9, 0.9])
    # initial total = 2, budget pct = 0.5 => only ONE net added expert allowed
    ctrl = _controller(moe, growth_budget_pct=0.5, max_experts=64)
    ctrl._maybe_resize_layer(moe, optimizer=None, step=0)
    assert int(moe.num_experts) == 3
    _set_health(moe, [0.9] * int(moe.num_experts))
    ctrl._maybe_resize_layer(moe, optimizer=None, step=1)
    assert int(moe.num_experts) == 3, "budget must block further growth"


def test_shrink_only_when_dead_surplus_exceeds_reset_capacity():
    moe = _moe(num_experts=5)
    _set_health(moe, [0.9, 0.9, 0.0, 0.0, 0.0])  # 3 dead
    ctrl = _controller(moe, resets_per_call=1, min_experts=2)
    ctrl._maybe_resize_layer(moe, optimizer=None, step=0)
    # removable = 3 dead - 1 resettable = 2 -> fold+drop two, keep the rest
    assert int(moe.num_experts) == 3
    _shapes_consistent(moe)
    y, _aux = moe(torch.randn(2, 6, 8))
    assert torch.isfinite(y).all()


def test_shrink_never_breaks_min_experts_floor():
    moe = _moe(num_experts=4)
    _set_health(moe, [0.9, 0.0, 0.0, 0.0])  # 3 dead, only 1 healthy
    ctrl = _controller(moe, min_experts=2)
    ctrl._maybe_resize_layer(moe, optimizer=None, step=0)
    assert int(moe.num_experts) >= 2
    assert torch.isfinite(moe.energy).all()


# --------------------------------------------------------------------------- #
# 5. NeuroScore self-heals its bookkeeping across a resize
# --------------------------------------------------------------------------- #
def test_neuroscore_stats_self_heal_after_resize():
    moe = _moe(num_experts=4)
    score = NeuroScore(NeuroScoreConfig(enabled=True, update_every=1000), neuroviz=None)
    x = torch.randn(2, 6, 8)
    y, _aux = moe(x)
    y.pow(2).mean().backward()
    score.step(moe, torch.tensor(1.0), 0)
    assert score.stats[""]["loss_contrib"].numel() == 4

    _resize_layer_experts_(moe, target_E=6, seed_idx=0, cfg=SplitMergeConfig())
    for it in range(2):
        y, _aux = moe(x)
        y.pow(2).mean().backward()
        score.step(moe, y.detach().pow(2).mean(), it + 1)
    st = score.stats[""]
    assert st["loss_contrib"].numel() == 6, "stats must resize with the layer"
    published = moe.last_neuroscore
    assert published is not None and published.shape == (6,)
