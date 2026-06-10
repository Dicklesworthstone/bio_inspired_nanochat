"""
Optimizer-momentum reset across ALL optimizers on split/merge (bead vg9.3).

Synaptic models route parameters across two optimizers: AdamW (1D/embeddings) and Muon
(2D matrices — including the expert/router weights that split/merge overwrites). The old
code passed only ONE optimizer to the lifecycle controller, so the OTHER optimizer's
stale momentum was applied to freshly-cloned weights after a split/merge — a real
instability vector. The fix: _zero_optim_moments_for accepts ALL optimizers, and
base_train passes the full list.

Run:  pytest tests/test_optimizer_moment_reset.py -v
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticMoE
from bio_inspired_nanochat.synaptic_splitmerge import (
    SplitMergeConfig,
    SplitMergeController,
    _zero_optim_moments_for,
)


def _populate_moments(opt: torch.optim.Optimizer, params: list[nn.Parameter]) -> None:
    for p in params:
        p.grad = torch.randn_like(p)
    opt.step()


@pytest.mark.unit
def test_zero_moments_single_optimizer_misses_the_other__then_list_fixes_it():
    p1 = nn.Parameter(torch.randn(4, 4))  # lives in opt_a (think: AdamW)
    p2 = nn.Parameter(torch.randn(4, 4))  # lives in opt_b (think: Muon / expert matrix)
    opt_a = torch.optim.AdamW([p1], lr=0.1)
    opt_b = torch.optim.AdamW([p2], lr=0.1)
    _populate_moments(opt_a, [p1])
    _populate_moments(opt_b, [p2])
    assert opt_a.state[p1]["exp_avg"].abs().sum() > 0
    assert opt_b.state[p2]["exp_avg"].abs().sum() > 0

    # THE BUG: resetting p2 with only opt_a leaves p2's momentum in opt_b stale.
    _zero_optim_moments_for(opt_a, [p2])
    assert opt_b.state[p2]["exp_avg"].abs().sum() > 0, "single optimizer can't reach opt_b"

    # THE FIX: pass BOTH optimizers -> p2's moments in opt_b are reset; p1 untouched.
    _zero_optim_moments_for([opt_a, opt_b], [p2])
    assert opt_b.state[p2]["exp_avg"].abs().sum() == 0
    assert opt_a.state[p1]["exp_avg"].abs().sum() > 0, "params not in the set are untouched"


@pytest.mark.unit
def test_zero_moments_handles_none_empty_and_list_with_none():
    p = nn.Parameter(torch.randn(2, 2))
    # None / empty / list-with-None must all be safe no-ops (defensive).
    _zero_optim_moments_for(None, [p])
    _zero_optim_moments_for([], [p])
    _zero_optim_moments_for([None], [p])
    # a single optimizer (back-compat) still works
    opt = torch.optim.AdamW([p], lr=0.1)
    _populate_moments(opt, [p])
    _zero_optim_moments_for(opt, [p])
    assert opt.state[p]["exp_avg"].abs().sum() == 0


@pytest.mark.unit
def test_controller_resets_reset_experts_momentum_across_all_optimizers():
    cfg = SynapticConfig(enable_hebbian=False, enable_metabolism=True)
    moe = SynapticMoE(n_embd=8, num_experts=2, top_k=1, hidden_mult=1, cfg=cfg, dropout=0.0)

    e0 = [p for n, p in moe.named_parameters() if n.startswith("experts.0") and p.ndim == 2]
    e1 = [p for n, p in moe.named_parameters() if n.startswith("experts.1") and p.ndim == 2]
    router = list(moe.router.parameters())
    assert e0 and e1, "expected 2D expert weight matrices"

    # Two optimizers like the real split: router in one (AdamW), 2D expert matrices in the
    # other (the 'Muon' stand-in that base_train used to NOT pass to the controller).
    opt_router = torch.optim.AdamW(router, lr=0.1)
    opt_experts = torch.optim.AdamW(e0 + e1, lr=0.1)

    out = moe(torch.randn(2, 6, 8))
    (out[0] if isinstance(out, (tuple, list)) else out).sum().backward()
    opt_router.step()
    opt_experts.step()
    assert any(opt_experts.state[p]["exp_avg"].abs().sum() > 0 for p in e0), "expert 0 has momentum"

    # Make expert 0 'dead' (health 0) so it gets RESET (overwritten by healthy expert 1).
    sm_cfg = SplitMergeConfig(
        enabled=True, warmup_steps=0, min_step_interval=0, merges_per_call=0,
        splits_per_call=0, reset_health_max=0.05, resets_per_call=1, ddp_broadcast=False,
    )
    ctrl = SplitMergeController(moe, sm_cfg)
    with torch.no_grad():
        moe.fatigue.zero_()
        moe.energy.zero_()
        moe.fatigue[1] = 1.0
        moe.energy[1] = 1.0

    # Pass BOTH optimizers (the fix). The reset overwrites expert 0's weights in place, so
    # its (now stale) momentum in opt_experts MUST be zeroed.
    ctrl.step(global_step=10, optimizer=[opt_router, opt_experts])

    reset_zeroed = False
    for p in e0:
        st = opt_experts.state.get(p, {})
        if "exp_avg" in st:
            assert st["exp_avg"].abs().sum() == 0, "reset expert's momentum must be zeroed"
            reset_zeroed = True
    assert reset_zeroed, "the reset expert had optimizer momentum that should have been cleared"
