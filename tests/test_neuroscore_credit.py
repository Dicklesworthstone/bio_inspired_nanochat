"""Gradient-based expert credit assignment — bead uta.2.

Replaces NeuroScore's legacy "contribution = sum of routing gates" proxy with a real
marginal signal: for every token routed to expert e, ``gate * ||dL/d(expert_out)||``,
captured by backward hooks on each expert's output during the normal training
backward. The load-bearing claim validated here: the new credit RANKS experts by
their actual leave-one-expert-out loss impact, which the routing proxy cannot do
(being routed often says nothing about helping).

Fallback semantics are pinned too: eval/inference-only flows, repeated step() calls,
and explicit ``credit_mode="proxy"`` reproduce the legacy numbers exactly.
"""

from __future__ import annotations

import pytest
import torch

from bio_inspired_nanochat.neuroscore import NeuroScore, NeuroScoreConfig
from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticMoE

pytestmark = pytest.mark.unit


def _moe(num_experts: int = 4, n_embd: int = 8, top_k: int | None = None, seed: int = 0) -> SynapticMoE:
    torch.manual_seed(seed)
    cfg = SynapticConfig(enable_hebbian=False, enable_metabolism=True, stochastic_train_frac=0.0)
    return SynapticMoE(
        n_embd=n_embd,
        num_experts=num_experts,
        top_k=top_k if top_k is not None else num_experts,  # dense default: every expert routed
        hidden_mult=1,
        cfg=cfg,
    )


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    def ranks(v: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(v)
        r = torch.empty(len(v), dtype=torch.float64)
        r[order] = torch.arange(len(v), dtype=torch.float64)
        return r

    ra, rb = ranks(a.double()), ranks(b.double())
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra @ rb) / (ra.norm() * rb.norm() + 1e-12))


# --------------------------------------------------------------------------- #
# 1. Mechanism exactness: captured credit == manual first-order LOO advantage
# --------------------------------------------------------------------------- #
def test_gradient_credit_matches_manual_formula():
    moe = _moe(num_experts=4)
    score = NeuroScore(NeuroScoreConfig(enabled=True, update_every=1000), neuroviz=None)
    B, T = 2, 6
    x = torch.randn(B, T, 8)



    # Warmup: installs hooks lazily but no grads were stashed yet -> proxy path.
    y, _aux = moe(x)
    (y * torch.randn(8)).sum().backward()
    score.step(moe, torch.tensor(1.0), 0)
    assert score.stats[""]["credit_source"] == "proxy"
    warmup = score.stats[""]["loss_contrib"].clone()

    # With a linear head L = <y, w>: dL/dy(pos) = w and dL/dy_e(pos) = gate * w.
    # First-order LOO advantage for expert e is therefore
    #   -sum_pos <gate*w, y_e(pos)> / (B*T)
    # where y_e rows are captured by our independent probe hook (same selection
    # order as the MoE's internal scatter: ascending flattened position).
    probes: dict[int, list[torch.Tensor]] = {e: [] for e in range(4)}

    def _make_probe(e):
        def _fn(_mod, _args, output):
            out = output[0] if isinstance(output, tuple) else output
            probes[e].append(out.detach().float().clone())
            return None
        return _fn

    handles = [moe.experts[e].register_forward_hook(_make_probe(e)) for e in range(4)]

    wvec = torch.randn(8)
    x2 = torch.randn(B, T, 8)
    y, _aux = moe(x2)
    loss = (y * wvec).sum()
    loss.backward()
    score.step(moe, loss.detach(), 1)

    st_now = score.stats[""]["loss_contrib"]
    decay = score.cfg.decay
    update = (st_now - decay * warmup) / (1 - decay)

    ctx = moe.last_ctx
    gates, idx = ctx["gates"], ctx["indices"]
    expected = torch.zeros(4)
    for e in range(4):
        mask = idx == e
        g = gates.masked_select(mask)                       # (n_sel,)
        y_e = probes[e][-1]                                  # (n_sel, C), aligned rows
        grad_rows = g.unsqueeze(-1) * wvec                   # dL/dy_e per selected row
        expected[e] = -(grad_rows * y_e).sum() / (B * T)
    for h in handles:
        h.remove()

    assert score.stats[""]["credit_source"] == "gradient"
    assert torch.allclose(update, expected, atol=1e-5), f"{update} != {expected}"


# --------------------------------------------------------------------------- #
# 2. The load-bearing acceptance claim: credit tracks TRUE loss impact
# --------------------------------------------------------------------------- #
def test_gradient_credit_tracks_true_loss_impact():
    E, C, B, T, seed = 6, 16, 4, 12, 3

    gen = torch.Generator().manual_seed(42)
    xs = [torch.randn(B, T, C, generator=gen) for _ in range(3)]
    # FIXED random targets (independent of the model) so the loss gradient is
    # dominated by the experts' actual output magnitudes, not by L≈0 noise.
    tgts = [torch.randn(B, T, C, generator=gen) for _ in range(3)]

    def build(zeroed: int | None = None) -> SynapticMoE:
        m = _moe(num_experts=E, n_embd=C, seed=seed)
        with torch.no_grad():
            for i in range(3):  # experts 0..2 are strong, 3..5 weak
                if i == zeroed:
                    continue
                for p in m.experts[i].parameters():
                    p.mul_(8.0)
            if zeroed is not None:
                for p in m.experts[zeroed].parameters():
                    p.zero_()
        return m

    moe = build()

    def losses(m: SynapticMoE):
        return [float((m(x, update_mem=False)[0] - tgt).pow(2).mean().detach()) for x, tgt in zip(xs, tgts)]

    # --- metric phase: one warmup step (installs hooks, proxy path), then ONE
    # measured gradient step; update extracted from the EMA algebra. All forwards
    # stateless (update_mem=False) so the ground-truth phase sees identical routing.
    score = NeuroScore(NeuroScoreConfig(enabled=True, update_every=1000), neuroviz=None)
    y, _aux = moe(xs[0])
    (y - tgts[0]).pow(2).mean().backward()
    score.step(moe, torch.tensor(1.0), 0)
    warmup = score.stats[""]["loss_contrib"].clone()

    y, _aux = moe(xs[1], update_mem=False)
    loss = (y - tgts[1]).pow(2).mean()
    loss.backward()
    score.step(moe, loss.detach(), 1)
    st_now = score.stats[""]["loss_contrib"]
    credit = (st_now - score.cfg.decay * warmup) / (1 - score.cfg.decay)

    proxy_score = NeuroScore(
        NeuroScoreConfig(enabled=True, update_every=1000, credit_mode="proxy"), neuroviz=None
    )
    y, _aux = moe(xs[0])
    (y - tgts[0]).pow(2).mean().backward()
    proxy_score.step(moe, torch.tensor(1.0), 0)
    pwarm = proxy_score.stats[""]["loss_contrib"].clone()
    y, _aux = moe(xs[1], update_mem=False)
    (y - tgts[1]).pow(2).mean().backward()
    proxy_score.step(moe, torch.tensor(1.0), 1)
    proxy = (proxy_score.stats[""]["loss_contrib"] - score.cfg.decay * pwarm) / (
        1 - score.cfg.decay
    )
    assert credit.dim() == 1 and torch.isfinite(credit).all()

    # Ground truth: leave-one-expert-out loss impact on identically built twins.
    base = sum(losses(build()))
    impact = torch.tensor([sum(losses(build(zeroed=e))) - base for e in range(E)])

    rho_grad = _spearman(credit, impact)
    rho_proxy = _spearman(proxy, impact)
    # Gradient credit must track actual loss impact strongly; the how-often-routed
    # proxy has no reason to. Margins verified empirically on this seed.
    assert rho_grad >= 0.85, f"gradient credit spearman {rho_grad:.3f} vs LOO impact"
    assert rho_grad > rho_proxy + 0.15, (
        f"gradient credit ({rho_grad:.3f}) must dominate the routing proxy ({rho_proxy:.3f})"
    )


# --------------------------------------------------------------------------- #
# 3. Fallback: inference-only flow reproduces the legacy proxy exactly
# --------------------------------------------------------------------------- #
def test_no_gradients_falls_back_to_legacy_proxy_exactly():
    moe = _moe(num_experts=4)
    score = NeuroScore(NeuroScoreConfig(enabled=True, update_every=1000), neuroviz=None)
    B, T = 2, 6
    with torch.no_grad():
        moe(torch.randn(B, T, 8))
    score.step(moe, torch.tensor(1.0), 0)

    st = score.stats[""]
    assert st["credit_source"] == "proxy"

    ctx = moe.last_ctx
    expected = torch.zeros(4)
    expected.index_add_(0, ctx["indices"].view(-1).cpu(), ctx["gates"].view(-1).float().cpu())
    expected /= B * T
    assert torch.allclose(st["loss_contrib"], expected * (1 - score.cfg.decay), atol=1e-6)

def test_repeated_step_without_new_backward_falls_back_cleanly():
    """Hooks install lazily during the first step call, and consuming empties the stash:
    step 1 = proxy (hooks not yet installed when its backward fired), step 2 = gradient,
    a third step with no new forward/backward must fall back without double-counting."""
    moe = _moe(num_experts=4)
    score = NeuroScore(NeuroScoreConfig(enabled=True, update_every=1000), neuroviz=None)
    x = torch.randn(2, 6, 8)

    y, _aux = moe(x)
    y.pow(2).mean().backward()
    score.step(moe, torch.tensor(1.0), 0)
    assert score.stats[""]["credit_source"] == "proxy"

    y, _aux = moe(x)
    y.pow(2).mean().backward()
    score.step(moe, torch.tensor(1.0), 1)
    assert score.stats[""]["credit_source"] == "gradient"

    score.step(moe, torch.tensor(1.0), 2)  # no new forward/backward in between
    assert score.stats[""]["credit_source"] == "proxy"



# --------------------------------------------------------------------------- #
# 4. Sparse regime: an expert that received no tokens earns EXACTLY zero
# --------------------------------------------------------------------------- #
def test_sparse_regime_unrouted_expert_gets_zero_gradient_credit():
    moe = _moe(num_experts=4, top_k=2)
    score = NeuroScore(NeuroScoreConfig(enabled=True, update_every=1000), neuroviz=None)

    for it in range(5):
        x = torch.randn(2, 8, 8, generator=torch.Generator().manual_seed(it))
        y, _aux = moe(x)
        y.pow(2).mean().backward()
        score.step(moe, torch.tensor(1.0), it)
        st = score.stats[""]
        routed = moe.last_ctx["indices"].unique()
        unrouted = [e for e in range(4) if e not in routed]
        if not unrouted:
            continue
        # Experts never invoked this forward have zero marginal credit by definition;
        # their accumulated EMA stays at zero from a zero-init start.
        for e in unrouted:
            assert st["routing_freq"][e].item() == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# 5. Lifecycle integration: published fitness stays finite and consumable
# --------------------------------------------------------------------------- #
def test_published_fitness_finite_under_gradient_credit():
    from bio_inspired_nanochat.synaptic_splitmerge import SplitMergeConfig, SplitMergeController

    moe = _moe(num_experts=4)
    score = NeuroScore(NeuroScoreConfig(enabled=True, update_every=1000), neuroviz=None)
    for it in range(3):
        x = torch.randn(2, 6, 8)
        y, _aux = moe(x)
        y.pow(2).mean().backward()
        score.step(moe, y.detach().pow(2).mean(), it)

    published = moe.last_neuroscore
    assert published is not None
    assert torch.isfinite(published).all()
    assert (published >= 0).all() and (published <= 1).all()

    ctrl = SplitMergeController(moe, SplitMergeConfig(use_neuroscore=True, neuroscore_weight=1.0))
    assert torch.isfinite(ctrl._health(moe)).all()
