"""
Homeostatic stability guards after lifecycle events (bead uta.6).

After any split/merge/reset the touched experts are stabilized:

  • ROUTED-MASS RAMP: a freshly seeded child starts with ~zero routing mass and
    anneals to its full twin share over ``gate_ramp_forwards`` training forwards,
    with additive compensation (+ln(g) on the child, +ln(2-g) on the parent) that
    keeps the pair's TOTAL softmax mass at the pre-event value at every point of
    the ramp. In the dense regime this preserves the model output EXACTLY
    throughout the transient — uta.3's event-time contract extended in time — so
    a fresh slot cannot shock the residual stream while it diverges under SGD.
  • ENERGY FLOOR: per-expert energy is clamped to >= ``energy_floor`` so a
    collapsed metabolism cannot drag health into winner-take-all routing.
  • ROW-WISE MOMENT WARM RESTART: only the CHANGED rows of shared tensors
    (router weight / genome Xi) get fresh optimizer moments; untouched experts
    keep theirs.

Run:  pytest tests/test_homeostasis_guards.py -v
"""

from __future__ import annotations

import math

import pytest
import torch

from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticMoE
from bio_inspired_nanochat.synaptic_splitmerge import (
    SplitMergeConfig,
    SplitMergeController,
)


def _pure_moe(seed: int, num_experts: int, top_k: int, n_embd: int = 16) -> SynapticMoE:
    """A SynapticMoE whose forward is a PURE function of its parameters (no Hebbian
    plasticity, no metabolism logit term, no router-embedding drift), mirroring the
    uta.3 fixture so lifecycle effects are isolated from per-forward state drift."""
    torch.manual_seed(seed)
    cfg = SynapticConfig(
        enable_hebbian=False,
        enable_metabolism=False,
        router_contrastive_push=0.0,
        router_contrastive_lr=0.0,
    )
    moe = SynapticMoE(
        n_embd=n_embd, num_experts=num_experts, top_k=top_k, hidden_mult=2, cfg=cfg, dropout=0.0
    )
    moe.eval()
    return moe
def _guards_cfg(**overrides) -> SplitMergeConfig:
    cfg = SplitMergeConfig(
        homeostasis_guards=True,
        gate_ramp_forwards=10,
        warmup_steps=0,
        min_step_interval=0,
        fp_divergence_noise=0.0,
        # One event per round: disable the reset path entirely so the planner's
        # split and reset cannot both target the same destination slot.
        reset_health_max=-1.0,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _rig_single_split(moe: SynapticMoE, source: int = 1) -> None:
    """Make ``source`` the only split candidate and kill every other slot's routing
    so the event is output-preserving; the child slot is derived from bias deltas."""
    with torch.no_grad():
        moe.fatigue.zero_()
        moe.fatigue[source] = 0.9  # sole expert above split_health_min
        for i in range(moe.num_experts):
            if i != source:
                moe.router_logit_bias[i] = -50.0


def _changed_indices(bias_before: torch.Tensor, bias_after: torch.Tensor) -> list[int]:
    diff = (bias_after - bias_before).abs()
    return [int(i) for i in torch.nonzero(diff > 1e-6).flatten().tolist()]


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm().item() / (b.norm().item() + 1e-9))


@pytest.mark.unit
def test_config_validation_rejects_bad_guard_values():
    with pytest.raises(ValueError):
        SplitMergeConfig(homeostasis_guards=True, gate_ramp_forwards=0)
    with pytest.raises(ValueError):
        SplitMergeConfig(homeostasis_guards=True, energy_floor=-0.1)
    with pytest.raises(ValueError):
        SplitMergeConfig(homeostasis_guards=True, energy_floor=1.5)


@pytest.mark.unit
def test_ramp_preserves_output_exactly_through_the_transient():
    """Dense regime, zero divergence noise: the output matches the pre-event model
    at EVERY forward of the ramp, and ends at the static uta.3 twin biases."""
    E = 6
    x = torch.randn(2, 5, 16)
    moe = _pure_moe(0, E, top_k=E)
    ctrl = SplitMergeController(moe, _guards_cfg())
    _rig_single_split(moe)

    with torch.no_grad():
        out0, _ = moe(x)
        bias_pre = moe.router_logit_bias.detach().clone()
        ctrl.step(global_step=10, optimizer=None)
        bias_post = moe.router_logit_bias.detach().clone()
        changed = _changed_indices(bias_pre, bias_post)
        assert len(changed) == 2, "exactly parent+child biases must move"

        out_event, _ = moe(x)
        worst = _rel_l2(out_event, out0)
        for _ in range(12):  # advance through the rest of the ramp and beyond
            out_t, _ = moe(x)
            worst = max(worst, _rel_l2(out_t, out0))
    assert worst < 1e-5, f"ramp must preserve dense output; worst rel L2 {worst}"
    # Final state equals the static function-preserving construction: both twins
    # sit ln2 below the PARENT's pre-event level (the dead slot's -50 placeholder
    # is replaced by the twin construction, not shifted relative to itself).
    parent = 1
    child = next(i for i in changed if i != parent)
    assert math.isclose(
        float(bias_post[parent]), float(bias_pre[parent]) - math.log(2.0), abs_tol=1e-6
    )
    assert math.isclose(
        float(bias_post[child]), float(bias_pre[parent]) - math.log(2.0), abs_tol=1e-6
    )

@pytest.mark.unit
def test_guards_off_leaves_routing_untouched_by_hooks():
    """Default-off: identical surgery, no ramps registered, no transient writes."""
    E = 6
    x = torch.randn(2, 5, 16)
    moe = _pure_moe(0, E, top_k=E)
    ctrl = SplitMergeController(moe, _guards_cfg(homeostasis_guards=False))
    _rig_single_split(moe)
    with torch.no_grad():
        ctrl.step(global_step=10, optimizer=None)
        bias_after_step = moe.router_logit_bias.detach().clone()
        for _ in range(15):
            moe(x)
    assert ctrl.homeo._ramps == {}, "no ramps may be registered when guards are off"
    assert torch.equal(moe.router_logit_bias, bias_after_step), "hooks must not touch biases"


def test_energy_floor_enforced_after_events_and_forwards():
    """The event itself clamps energy to the floor, and every guarded train
    forward re-clamps drift below it (decision health uses the pre-event value)."""
    E = 6
    floor = 0.25
    x = torch.randn(2, 5, 16)
    moe = _pure_moe(0, E, top_k=E)
    ctrl = SplitMergeController(moe, _guards_cfg(energy_floor=floor))
    _rig_single_split(moe)
    with torch.no_grad():
        # Collapse energy only AFTER rigging: health = utilization * energy must
        # still see a healthy source for the split decision to fire.
        moe.energy.fill_(0.9)
        ctrl.step(global_step=10, optimizer=None)
        assert float(moe.energy.min()) >= 0.9 - 1e-6, "healthy event must not lower energy"
        # Direct unit check of the event-time clamp.
        moe.energy.fill_(0.01)
        ctrl.homeo.on_seeded_children(0, moe, [2], [1])
        assert float(moe.energy.min()) >= floor, "event clamp must restore the floor"
        # And the guarded forward hook re-clamps per-forward drift.
        moe.energy.copy_(torch.linspace(0.0, 0.05, E))
        for _ in range(3):
            out, _ = moe(x)
            assert float(out.abs().sum()) >= 0.0  # exercise forward
            assert float(moe.energy.min()) >= floor, "hook must restore the floor"


@pytest.mark.unit
def test_row_wise_moment_warm_restart_spares_untouched_experts():
    """Guards on: shared-tensor moments reset ONLY for the touched rows; unrelated
    experts' AdamW state survives the event. Moments are populated with NORMAL
    routing first (healthy gradient flow), and the split decision is rigged only
    afterwards via fatigue, so gradients stay meaningful."""
    E = 6
    x = torch.randn(2, 3, 16)
    moe = _pure_moe(0, E, top_k=E)
    opt = torch.optim.AdamW(moe.parameters(), lr=1e-2)
    for _ in range(3):  # populate moments everywhere with healthy grads
        opt.zero_grad(set_to_none=True)
        out, _ = moe(x)
        out.sum().backward()
        opt.step()

    ctrl = SplitMergeController(moe, _guards_cfg())
    with torch.no_grad():
        moe.fatigue.zero_()
        moe.fatigue[1] = 0.9  # sole split candidate; biases left untouched so
        # every expert keeps receiving real gradient traffic.
        bias_pre = moe.router_logit_bias.detach().clone()
        ctrl.step(global_step=10, optimizer=opt)
        bias_post = moe.router_logit_bias.detach().clone()
    changed = _changed_indices(bias_pre, bias_post)
    assert len(changed) == 2, f"expected parent+child bias moves, got {changed}"
    touched = sorted(changed)
    untouched_rows = [i for i in range(E) if i not in touched]

    exp_avg = opt.state[moe.router.weight]["exp_avg"]
    assert float(exp_avg[touched].abs().max()) == 0.0, "touched rows must restart"
    assert float(exp_avg[untouched_rows].abs().max()) > 0.0, "untouched rows keep moments"
    xi = moe.Xi
    if isinstance(xi, torch.nn.Parameter):
        xi_avg = opt.state[xi]["exp_avg"]
        assert float(xi_avg[touched].abs().max()) == 0.0
        assert float(xi_avg[untouched_rows].abs().max()) > 0.0
    # An untouched expert's exclusive weights keep their moments entirely.
    keeper = untouched_rows[0]
    w = moe.experts[keeper].fc1.w_slow
    assert float(opt.state[w]["exp_avg"].abs().max()) > 0.0


@pytest.mark.unit
def test_guard_state_roundtrips_through_controller_state():
    E = 6
    x = torch.randn(2, 5, 16)
    moe = _pure_moe(0, E, top_k=E)
    ctrl = SplitMergeController(moe, _guards_cfg(gate_ramp_forwards=100))
    _rig_single_split(moe)
    with torch.no_grad():
        ctrl.step(global_step=10, optimizer=None)
        moe(x)  # advance one ramp tick
    saved = ctrl.state_dict()
    assert "homeostasis" in saved and saved["homeostasis"]["ramps"], "mid-ramp state must persist"

    ctrl2 = SplitMergeController(moe, _guards_cfg(gate_ramp_forwards=100))
    ctrl2.load_state_dict(saved)
    assert ctrl2.homeo._ramps == {0: ctrl.homeo._ramps.get(0, {})} or bool(ctrl2.homeo._ramps)


@pytest.mark.unit
def test_sparse_topk_forward_stays_finite_during_ramp():
    E = 6
    x = torch.randn(2, 5, 16)
    moe = _pure_moe(0, E, top_k=2)
    ctrl = SplitMergeController(moe, _guards_cfg())
    _rig_single_split(moe)
    with torch.no_grad():
        ctrl.step(global_step=10, optimizer=None)
        for _ in range(5):
            out, aux = moe(x)
            assert torch.isfinite(out).all()
            assert torch.isfinite(aux).all()
