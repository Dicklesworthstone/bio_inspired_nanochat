"""
Function-preserving expert split/merge — Net2Net / firefly (bead uta.3).

The MoE lifecycle (split/merge/reset) used to OVERWRITE the weakest expert with a NOISY
clone of a strong one and leave the parent at full routing mass, so the model output jumped
at every lifecycle event (a loss spike). This bead makes the events function-preserving:

  • SPLIT: the destination becomes an EXACT clone of the parent; both then take a -ln2
    additive routing-logit bias so together they reproduce the parent's original routing
    probability mass (each fires with half the gate). Antisymmetric fc1 noise (parent -= δ,
    child += δ, δ RELATIVE to the weight RMS) lets them diverge under SGD while the mean
    function is preserved at the event. In the DENSE regime (top_k == num_experts) this is
    EXACTLY output-preserving; in sparse top-k it sharply reduces (but cannot zero) the
    discontinuity, because moving a twin pair across the top-k boundary is inherently discrete.

  • MERGE: the loser is weight-averaged into the winner, the winner takes +ln2 to absorb the
    loser's mass, then the freed slot is re-seeded as a function-preserving split of the winner.

These tests assert the exactness in the dense regime, the continuity under small divergence
noise, and a large reduction in output discontinuity vs. the legacy noisy-clone path.

Run:  pytest tests/test_function_preserving_splitmerge.py -v
"""

from __future__ import annotations

import pytest
import torch

from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticMoE
from bio_inspired_nanochat.synaptic_splitmerge import (
    SplitMergeConfig,
    SplitMergeController,
    _function_preserving_split_,
    _function_preserving_merge_,
    _merge_expert_into_and_clone_,
    _clone_linear_from_,
    _add_noise_,
)


def _pure_moe(seed: int, num_experts: int, top_k: int, n_embd: int = 16) -> SynapticMoE:
    """A SynapticMoE whose forward is a PURE function of its parameters: no Hebbian
    plasticity, no metabolism logit term, no router-embedding drift. This isolates the
    effect of a lifecycle event from the forward's own per-step state mutation."""
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


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm().item() / (b.norm().item() + 1e-9))


def _legacy_split_(moe: SynapticMoE, parent: int, dst: int) -> None:
    """Reproduce the legacy (function_preserving=False) noisy-clone split for comparison."""
    with torch.no_grad():
        _clone_linear_from_(moe.experts[dst].fc1, moe.experts[parent].fc1, 0.02)
        _clone_linear_from_(moe.experts[dst].fc2, moe.experts[parent].fc2, 0.02)
        W = moe.router.weight
        W[dst].copy_(W[parent])
        _add_noise_(W[dst], 0.01)
        moe.router_embeddings[dst].copy_(moe.router_embeddings[parent])


@pytest.mark.unit
def test_split_is_exact_in_dense_regime():
    """Dense routing + zero divergence noise: the output is bit-for-bit preserved (fp tol)."""
    E = 6
    x = torch.randn(2, 5, 16)
    moe = _pure_moe(0, E, top_k=E)
    dead = 4
    with torch.no_grad():
        moe.router_logit_bias[dead] = -50.0  # genuinely dead slot (contributes ~0)
    out0, _ = moe(x)

    cfg = SplitMergeConfig(function_preserving=True, fp_divergence_noise=0.0)
    with torch.no_grad():
        _function_preserving_split_(moe, parent_idx=1, dst_idx=dead, cfg=cfg)
    out1, _ = moe(x)

    assert _rel_l2(out1, out0) < 1e-5, "dense function-preserving split must preserve the output"


@pytest.mark.unit
def test_split_small_noise_is_continuous_and_twins_diverge():
    """A small RELATIVE divergence noise keeps the event continuous yet truly splits the twins."""
    E = 6
    x = torch.randn(2, 5, 16)
    moe = _pure_moe(0, E, top_k=E)
    dead = 4
    with torch.no_grad():
        moe.router_logit_bias[dead] = -50.0
    out0, _ = moe(x)

    cfg = SplitMergeConfig(function_preserving=True, fp_divergence_noise=0.02)
    with torch.no_grad():
        _function_preserving_split_(moe, parent_idx=1, dst_idx=dead, cfg=cfg)
    out1, _ = moe(x)

    assert _rel_l2(out1, out0) < 5e-2, "small relative noise must keep the split continuous"
    twin_gap = (moe.experts[1].fc1.w_slow - moe.experts[dead].fc1.w_slow).abs().max().item()
    assert twin_gap > 0.0, "antisymmetric noise must make the twins actually diverge"


@pytest.mark.unit
def test_gate_split_assigns_half_routing_bias_to_each_twin():
    """The -ln2 construction: parent and child end with equal bias, each ln2 below the
    parent's pre-split bias (so their gate mass sums to the parent's original)."""
    E = 6
    moe = _pure_moe(0, E, top_k=E)
    parent, dst = 1, 4
    pre = float(moe.router_logit_bias[parent].item())
    cfg = SplitMergeConfig(function_preserving=True, fp_divergence_noise=0.0)
    with torch.no_grad():
        _function_preserving_split_(moe, parent_idx=parent, dst_idx=dst, cfg=cfg)
    assert moe.router_logit_bias[parent].item() == pytest.approx(pre - torch.log(torch.tensor(2.0)).item(), abs=1e-6)
    assert moe.router_logit_bias[dst].item() == pytest.approx(moe.router_logit_bias[parent].item(), abs=1e-6)


@pytest.mark.unit
def test_merge_is_exact_for_identical_pair():
    """Merging two identical experts (the well-posed case the merge criteria target) leaves
    the dense output unchanged."""
    E = 6
    x = torch.randn(2, 5, 16)
    moe = _pure_moe(0, E, top_k=E)
    a, b = 2, 3
    with torch.no_grad():
        # make b an exact clone of a so the pair is genuinely mergeable
        _clone_linear_from_(moe.experts[b].fc1, moe.experts[a].fc1, 0.0)
        _clone_linear_from_(moe.experts[b].fc2, moe.experts[a].fc2, 0.0)
        moe.router.weight[b].copy_(moe.router.weight[a])
        moe.Xi[b].copy_(moe.Xi[a])
        moe.router_embeddings[b].copy_(moe.router_embeddings[a])
    out0, _ = moe(x)

    cfg = SplitMergeConfig(function_preserving=True, fp_divergence_noise=0.0)
    with torch.no_grad():
        _function_preserving_merge_(moe, winner_idx=a, loser_idx=b, alpha=0.5, cfg=cfg)
    out1, _ = moe(x)

    assert _rel_l2(out1, out0) < 1e-5, "merging an identical pair must preserve the output"


@pytest.mark.unit
@pytest.mark.parametrize("top_k", [2, 4, 8])
def test_function_preserving_beats_legacy_split(top_k: int):
    """Across dense AND sparse regimes, the function-preserving split causes a far smaller
    output discontinuity than the legacy noisy clone. The dead slot is chosen by routing
    weight (no router-bias confound), so both paths overwrite the same slot."""
    E = 8
    x = torch.randn(4, 8, 16)

    def least_used(moe: SynapticMoE) -> tuple[int, int]:
        with torch.no_grad():
            logits = moe.router(x)
            _, idx = torch.topk(logits, top_k, dim=-1)
            cnt = torch.bincount(idx.flatten(), minlength=E)
        return int(cnt.argmax().item()), int(cnt.argmin().item())

    # function-preserving
    moe = _pure_moe(2, E, top_k=top_k)
    parent, dead = least_used(moe)
    out0, _ = moe(x)
    cfg = SplitMergeConfig(function_preserving=True, fp_divergence_noise=0.02)
    with torch.no_grad():
        _function_preserving_split_(moe, parent_idx=parent, dst_idx=dead, cfg=cfg)
    fp = _rel_l2(moe(x)[0], out0)

    # legacy (same seed/setup => same parent/dead and same starting weights)
    moe = _pure_moe(2, E, top_k=top_k)
    out0, _ = moe(x)
    _legacy_split_(moe, parent, dead)
    legacy = _rel_l2(moe(x)[0], out0)

    assert fp < legacy / 3.0, f"FP split (relL2={fp:.3e}) must be far gentler than legacy ({legacy:.3e})"


@pytest.mark.unit
def test_fp_merge_keeps_router_embeddings_unit_norm():
    """The router embeddings are maintained unit-norm everywhere (init + the contrastive EMA in
    SynapticMoE.forward) and the forward uses ‖embedding‖ as a routing GAIN. The function-preserving
    merge averages the winner with the loser; averaging two non-parallel unit vectors gives a norm
    < 1, so the merge must renormalize — otherwise it silently changes the winner's routing gain
    (and the clone it seeds into the freed loser slot inherits the shrunken norm too)."""
    E = 6
    moe = _pure_moe(0, E, top_k=E)
    winner, loser = 2, 3
    D = moe.router_embeddings.shape[1]
    e_w = torch.zeros(D)
    e_w[0] = 1.0
    e_l = torch.zeros(D)
    e_l[1] = 1.0  # orthogonal unit vector ⟹ naive average norm = 1/√2
    with torch.no_grad():
        moe.router_embeddings[winner].copy_(e_w)
        moe.router_embeddings[loser].copy_(e_l)
    cfg = SplitMergeConfig(function_preserving=True, fp_divergence_noise=0.0)
    with torch.no_grad():
        _function_preserving_merge_(moe, winner_idx=winner, loser_idx=loser, alpha=0.5, cfg=cfg)
    assert moe.router_embeddings[winner].norm().item() == pytest.approx(1.0, abs=1e-5)
    assert moe.router_embeddings[loser].norm().item() == pytest.approx(1.0, abs=1e-5)


@pytest.mark.unit
def test_legacy_merge_reseeds_loser_genome_and_routing_bias():
    """The legacy (function_preserving=False) merge reseeds the freed loser slot as a noisy clone
    of the winner, so the reborn expert must adopt the winner's presynaptic genome Xi and routing
    bias — not keep the dead loser's stale ones (a phenotype mismatch)."""
    E = 6
    moe = _pure_moe(0, E, top_k=E)
    winner, loser = 2, 3
    with torch.no_grad():
        moe.Xi[winner].copy_(torch.full_like(moe.Xi[winner], 0.7))
        moe.Xi[loser].copy_(torch.full_like(moe.Xi[loser], -0.7))   # distinct stale genome
        moe.router_logit_bias[winner] = 1.5
        moe.router_logit_bias[loser] = -3.0                          # stale dead-slot bias
    cfg = SplitMergeConfig(function_preserving=False)
    with torch.no_grad():
        _merge_expert_into_and_clone_(moe, winner_idx=winner, loser_idx=loser, alpha=0.5, cfg=cfg)
    assert torch.allclose(moe.Xi[loser], moe.Xi[winner]), "reborn loser must inherit the winner's genome"
    assert moe.router_logit_bias[loser].item() == pytest.approx(moe.router_logit_bias[winner].item())


@pytest.mark.unit
def test_router_logit_bias_backward_compat_load():
    """Old checkpoints predating router_logit_bias still load under strict=True (the buffer is
    injected as zeros) and reproduce the original forward."""
    E = 5
    x = torch.randn(2, 4, 16)
    src = _pure_moe(7, E, top_k=E)
    sd = src.state_dict()
    # simulate an old checkpoint: drop the new buffer
    assert "router_logit_bias" in sd
    del sd["router_logit_bias"]

    dst = _pure_moe(99, E, top_k=E)  # different init
    missing_unexpected = dst.load_state_dict(sd, strict=True)  # must not raise
    assert tuple(missing_unexpected.missing_keys) == ()
    assert tuple(missing_unexpected.unexpected_keys) == ()
    assert torch.allclose(dst.router_logit_bias, torch.zeros(E))
    # forward matches the source (router_logit_bias defaulted to zero == original behavior)
    assert _rel_l2(dst(x)[0], src(x)[0]) < 1e-6


@pytest.mark.unit
def test_controller_step_function_preserving_is_gentler_than_legacy():
    """End-to-end through SplitMergeController.step: a function-preserving lifecycle pass
    perturbs the output far less than the legacy pass on the same model/state."""
    E = 8
    x = torch.randn(4, 8, 16)

    def run(fp: bool) -> float:
        moe = _pure_moe(3, E, top_k=4)
        # one healthy expert, one dead slot, so the controller will split/reset
        with torch.no_grad():
            moe.fatigue.copy_(torch.full((E,), 0.05))
            moe.energy.copy_(torch.full((E,), 0.05))
            moe.fatigue[0] = 1.0
            moe.energy[0] = 1.0  # expert 0 healthy => split source
        out0, _ = moe(x)
        cfg = SplitMergeConfig(
            enabled=True, warmup_steps=0, min_step_interval=0,
            merges_per_call=0, splits_per_call=1, split_health_min=0.5,
            resets_per_call=1, reset_health_max=0.05, ddp_broadcast=False,
            function_preserving=fp, fp_divergence_noise=0.02,
        )
        SplitMergeController(moe, cfg).step(global_step=10, optimizer=None)
        return _rel_l2(moe(x)[0], out0)

    fp_delta = run(True)
    legacy_delta = run(False)
    assert fp_delta < legacy_delta, (
        f"function-preserving controller step (relL2={fp_delta:.3e}) must be gentler than "
        f"legacy (relL2={legacy_delta:.3e})"
    )
