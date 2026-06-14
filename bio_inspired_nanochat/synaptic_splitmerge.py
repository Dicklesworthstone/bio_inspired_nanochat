# nanochat/synaptic_splitmerge.py
# Split/Merge controller for Synaptic MoE layers
#
# This controller performs:
#   • MERGE: pick expert pairs with high cosine similarity in the router-embedding
#            space AND low health; merge loser into winner (weighted average),
#            then CLONE the winner (with small noise) back into loser slot.
#   • SPLIT: clone strong experts into the weakest slots (optional).
#   • Keeps expert count constant; updates router columns, embeddings, synaptic state,
#     zeroes optimizer moments for changed parameters, and can broadcast in DDP.
#
# Works with:
#   - SynapticMoE, SynapticExpert, SynapticLinear, PostsynapticHebb from bio_inspired_nanochat/synaptic.py
#   - GPTSynaptic from bio_inspired_nanochat/gpt_synaptic.py
#
# Usage:
#   ctrl = SplitMergeController(model, SplitMergeConfig(...))
#   ctrl.step(global_step, optimizer=opt)    # call periodically (e.g. every 50k steps)

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable, Any, cast
from bio_inspired_nanochat.torch_imports import torch, nn, Tensor
import torch.distributed as torch_dist

from .synaptic import SynapticMoE, SynapticExpert, SynapticLinear

dist = cast(Any, torch_dist)

# An optimizer, or a collection of them. Synaptic models split parameters across AdamW
# (1D/embeddings) AND Muon (2D matrices), so lifecycle moment-resets must reach all of
# them — see _zero_optim_moments_for (vg9.3).
OptimizersArg = Optional["torch.optim.Optimizer | Iterable[torch.optim.Optimizer]"]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SplitMergeConfig:
    enabled: bool = True
    # MERGE criteria
    merge_cosine_threshold: float = 0.85  # router-embedding cosine sim threshold
    merge_health_max: float = (
        0.25  # both experts must be below this health to be merge-candidates
    )
    merges_per_call: int = 1  # max merges per step
    # SPLIT criteria
    split_health_min: float = (
        0.80  # expert must be above this health to be split candidate
    )
    splits_per_call: int = 1
    # RESET (dead experts) criteria
    reset_health_max: float = 0.02  # health threshold to reset as "dead"
    resets_per_call: int = 1
    # Noise scales for cloned experts
    clone_noise_linear: float = 0.02  # noise scale for linear weights
    clone_noise_router: float = 0.01  # noise scale for router columns
    clone_noise_embed: float = 0.05  # noise scale (tangent) for router embedding
    # FUNCTION-PRESERVING lifecycle (uta.3 — Net2Net / firefly). When True (default),
    # split/merge are constructed so the model's OUTPUT does not jump at the event:
    #   • SPLIT: the destination slot becomes an EXACT clone of the parent (weights, router
    #     row, genome Xi, embedding, metabolic & synaptic state). The parent and the clone
    #     each receive a -ln2 additive routing-logit bias, so together they reproduce the
    #     parent's original routing probability mass (each twin fires with half the gate) —
    #     exactly output-preserving in the dense regime. Antisymmetric noise on the parent's
    #     and child's fc1 (parent -= δ, child += δ) breaks the symmetry so they diverge under
    #     SGD while the first-order (mean) function is preserved at the event.
    #   • MERGE: the loser is weight-averaged into the winner; the winner takes +ln2 routing
    #     bias to absorb the loser's mass; then the freed loser slot is re-seeded as a
    #     function-preserving split of the merged winner — so the combined contribution is
    #     unchanged and the slot becomes fresh capacity that re-diverges.
    # When False, the legacy noisy-clone split / blend-and-overwrite merge is used (kept for
    # back-compat and as the discontinuous baseline the continuity tests compare against).
    function_preserving: bool = True
    # Per-twin additive routing-logit reduction; ln2 makes a pair share the parent's mass.
    gate_split_bias: float = math.log(2.0)
    # Antisymmetric fc1 divergence noise for function-preserving split (parent -= δ, child += δ).
    fp_divergence_noise: float = 0.02
    # Scheduling
    min_step_interval: int = 10_000  # don't do anything more frequently than this
    warmup_steps: int = 20_000  # no changes until after warmup
    # DDP
    ddp_broadcast: bool = True  # broadcast parameters from rank 0 after changes
    # Expert weighting
    use_util_weighting: bool = (
        True  # weight merge by winner/loser utilization (via fatigue proxy)
    )
    # NeuroScore credit assignment (de5l): when enabled, blend the per-expert
    # NeuroScore fitness (Efficiency/Specialization/Resilience, published onto each
    # SynapticMoE as last_neuroscore by NeuroScore.step) into the health signal that
    # drives every split/merge/reset decision. Default-off so the lifecycle stays a
    # pure utilization*energy economy unless an experiment opts in; requires an active
    # NeuroScore (NeuroVizManager) so last_neuroscore is populated, else it no-ops.
    use_neuroscore: bool = False
    neuroscore_weight: float = 0.5  # blend weight in [0,1]: health=(1-w)*health + w*score
    # Logging
    verbose: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_rank0() -> bool:
    return (
        (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    )


def _world_size() -> int:
    return (
        1
        if (not dist.is_available()) or (not dist.is_initialized())
        else dist.get_world_size()
    )


@torch.no_grad()
def _cosine(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)
    return a @ b.T


@torch.no_grad()
def _orthogonal_perturb_like(vec: Tensor, noise_scale: float) -> Tensor:
    """Return a unit-length vector: normalized(vec + noise in orthogonal subspace).

    When noise_scale <= 0, returns vec unchanged (no normalization, exact copy).
    """
    if noise_scale <= 0:
        return vec.clone()
    vec.shape[-1]
    noise = torch.randn_like(vec)
    proj = (noise * vec).sum(dim=-1, keepdim=True) * vec
    tangent = noise - proj
    out = vec + noise_scale * tangent
    return out / (out.norm(dim=-1, keepdim=True) + 1e-8)


@torch.no_grad()
def _add_noise_(t: Tensor, scale: float):
    if scale <= 0:
        return
    t.add_(torch.randn_like(t) * scale)


@torch.no_grad()
def _zero_optim_moments_for(
    optimizers: "Optional[torch.optim.Optimizer | Iterable[torch.optim.Optimizer]]",
    params: Iterable[nn.Parameter],
):
    """Zero optimizer moment buffers (momentum, exp_avg, exp_avg_sq, ...) for ``params``,
    across ONE optimizer OR an iterable of optimizers.

    vg9.3: synaptic models split parameters across AdamW (1D/embeddings) AND Muon (2D
    matrices — including the expert/router weights that split/merge overwrites). Passing
    only one optimizer left the OTHER optimizer's stale momentum applied to freshly
    cloned weights after a lifecycle event (a real instability vector). We therefore
    accept all optimizers and reset each changed param's moments wherever they live.
    """
    if optimizers is None:
        return
    if isinstance(optimizers, torch.optim.Optimizer):
        optimizers = (optimizers,)
    pset = set(params)
    for optimizer in optimizers:
        if optimizer is None:
            continue
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p in pset:
                    state = optimizer.state.get(p, None)
                    if state:
                        for k in list(state.keys()):  # momentum, exp_avg, exp_avg_sq, ...
                            if torch.is_tensor(state[k]):
                                state[k].zero_()


@torch.no_grad()
def _broadcast_module_params(module: nn.Module):
    if not dist.is_available() or not dist.is_initialized() or _world_size() == 1:
        return
    for t in module.state_dict().values():
        if torch.is_tensor(t):
            dist.broadcast(t, src=0)


# ---------------------------------------------------------------------------
# Parameter copy helpers (SynapticLinear & expert)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _copy_synaptic_linear_(dst: SynapticLinear, src: SynapticLinear):
    # weights & bias
    dst.w_slow.copy_(src.w_slow)
    if (dst.w_fast is not None) and (src.w_fast is not None):
        dst.w_fast.copy_(src.w_fast)
    if (dst.bias is not None) and (src.bias is not None):
        cast(Tensor, dst.bias).copy_(cast(Tensor, src.bias))
    # postsyn state
    if (dst.post is not None) and (src.post is not None):
        cast(Tensor, dst.post.U).copy_(cast(Tensor, src.post.U))
        cast(Tensor, dst.post.V).copy_(cast(Tensor, src.post.V))
        cast(Tensor, dst.post.fast).copy_(cast(Tensor, src.post.fast))
        cast(Tensor, dst.post.slow).copy_(cast(Tensor, src.post.slow))
        # buffers
        cast(Tensor, dst.post.camkii).copy_(cast(Tensor, src.post.camkii))
        cast(Tensor, dst.post.pp1).copy_(cast(Tensor, src.post.pp1))
        cast(Tensor, dst.post.bdnf).copy_(cast(Tensor, src.post.bdnf))
        if hasattr(dst.post, "bdnf_hebb_accum") and hasattr(src.post, "bdnf_hebb_accum"):
            cast(Tensor, dst.post.bdnf_hebb_accum).copy_(cast(Tensor, src.post.bdnf_hebb_accum))
        if hasattr(dst.post, "_last_hebb_delta_mag") and hasattr(src.post, "_last_hebb_delta_mag"):
            cast(Tensor, dst.post._last_hebb_delta_mag).copy_(cast(Tensor, src.post._last_hebb_delta_mag))
    # Linear buffers
    if (dst.u_buf is not None) and (src.u_buf is not None):
        cast(Tensor, dst.u_buf).copy_(cast(Tensor, src.u_buf))
    if (dst.v_buf is not None) and (src.v_buf is not None):
        cast(Tensor, dst.v_buf).copy_(cast(Tensor, src.v_buf))


@torch.no_grad()
def _merge_linear_into_(winner: SynapticLinear, loser: SynapticLinear, alpha: float, cfg: SplitMergeConfig):
    """winner = alpha * winner + (1-alpha) * loser; loser = winner + noise"""
    if cfg.enabled and winner.w_slow.is_cuda: # Check if we can use fused kernel
        try:
            from bio_inspired_nanochat.kernels import mix_and_shift_tensors
            # Weights
            mix_and_shift_tensors(winner.w_slow, loser.w_slow, alpha, cfg.clone_noise_linear)
            if (winner.w_fast is not None) and (loser.w_fast is not None):
                mix_and_shift_tensors(winner.w_fast, loser.w_fast, alpha, cfg.clone_noise_linear)
            if (winner.bias is not None) and (loser.bias is not None):
                mix_and_shift_tensors(cast(Tensor, winner.bias), cast(Tensor, loser.bias), alpha, cfg.clone_noise_linear)
            
            # Postsynaptic state
            # For state, we might want less noise or different logic?
            # Original logic:
            # winner = alpha*winner + (1-alpha)*loser
            # loser = winner + noise (clone)
            # BUT original _clone_linear_from_ did:
            # dst.post.H_fast.zero_()
            # dst.post.U.mul_(0.5)
            # So simple mix_and_shift is NOT correct for state if we want to reset/dampen loser.
            
            # Let's stick to manual for state to preserve logic, or adapt kernel.
            # The kernel does: t2 = t1 + noise.
            # We want t2 = 0 or t2 = t1 * 0.5.
            
            # So we only use fused kernel for weights/biases which are the big tensors.
            # State tensors are small (rank R).
            
            if (winner.post is not None) and (loser.post is not None):
                # Manual state update (same as before)
                cast(Tensor, winner.post.U).mul_(alpha).add_((1.0 - alpha) * cast(Tensor, loser.post.U))
                cast(Tensor, winner.post.V).mul_(alpha).add_((1.0 - alpha) * cast(Tensor, loser.post.V))
                cast(Tensor, winner.post.fast).mul_(alpha).add_((1.0 - alpha) * cast(Tensor, loser.post.fast))
                cast(Tensor, winner.post.slow).mul_(alpha).add_((1.0 - alpha) * cast(Tensor, loser.post.slow))
                
                cast(Tensor, winner.post.camkii).mul_(0.9).add_(0.1 * cast(Tensor, loser.post.camkii))
                cast(Tensor, winner.post.pp1).mul_(0.9).add_(0.1 * cast(Tensor, loser.post.pp1))
                cast(Tensor, winner.post.bdnf).mul_(0.9).add_(0.1 * cast(Tensor, loser.post.bdnf))
                if hasattr(winner.post, "bdnf_hebb_accum") and hasattr(loser.post, "bdnf_hebb_accum"):
                    cast(Tensor, winner.post.bdnf_hebb_accum).mul_(0.9).add_(
                        0.1 * cast(Tensor, loser.post.bdnf_hebb_accum)
                    )
                if hasattr(winner.post, "_last_hebb_delta_mag") and hasattr(loser.post, "_last_hebb_delta_mag"):
                    cast(Tensor, winner.post._last_hebb_delta_mag).copy_(
                        cast(Tensor, loser.post._last_hebb_delta_mag)
                    )
                
                # Clone state to loser (with reset logic)
                cast(Tensor, loser.post.U).copy_(cast(Tensor, winner.post.U)).mul_(0.5)
                cast(Tensor, loser.post.V).copy_(cast(Tensor, winner.post.V)).mul_(0.5)
                cast(Tensor, loser.post.fast).zero_() # Reset fast weights
                cast(Tensor, loser.post.slow).copy_(cast(Tensor, winner.post.slow)) # Keep slow weights? Or reset? Usually keep base knowledge.
                
                cast(Tensor, loser.post.camkii).copy_(cast(Tensor, winner.post.camkii))
                cast(Tensor, loser.post.pp1).copy_(cast(Tensor, winner.post.pp1))
                cast(Tensor, loser.post.bdnf).copy_(cast(Tensor, winner.post.bdnf))
                if hasattr(loser.post, "bdnf_hebb_accum") and hasattr(winner.post, "bdnf_hebb_accum"):
                    cast(Tensor, loser.post.bdnf_hebb_accum).copy_(
                        cast(Tensor, winner.post.bdnf_hebb_accum)
                    )
                if hasattr(loser.post, "_last_hebb_delta_mag") and hasattr(winner.post, "_last_hebb_delta_mag"):
                    cast(Tensor, loser.post._last_hebb_delta_mag).copy_(
                        cast(Tensor, winner.post._last_hebb_delta_mag)
                    )
            
            # Reset eligibility buffers in Linear
            if loser.u_buf is not None:
                cast(Tensor, loser.u_buf).zero_()
            if loser.v_buf is not None:
                cast(Tensor, loser.v_buf).zero_()
            
            return
        except ImportError:
            pass

    # Fallback / CPU logic
    # winner = alpha * winner + (1-alpha) * loser
    winner.w_slow.mul_(alpha).add_((1.0 - alpha) * loser.w_slow)
    if (winner.w_fast is not None) and (loser.w_fast is not None):
        winner.w_fast.mul_(alpha).add_((1.0 - alpha) * loser.w_fast)
    if (winner.bias is not None) and (loser.bias is not None):
        cast(Tensor, winner.bias).mul_(alpha).add_((1.0 - alpha) * cast(Tensor, loser.bias))
    if (winner.post is not None) and (loser.post is not None):
        cast(Tensor, winner.post.U).mul_(alpha).add_((1.0 - alpha) * cast(Tensor, loser.post.U))
        cast(Tensor, winner.post.V).mul_(alpha).add_((1.0 - alpha) * cast(Tensor, loser.post.V))
        cast(Tensor, winner.post.fast).mul_(alpha).add_((1.0 - alpha) * cast(Tensor, loser.post.fast))
        cast(Tensor, winner.post.slow).mul_(alpha).add_((1.0 - alpha) * cast(Tensor, loser.post.slow))
        
        # gate and enzymes: bias toward winner (more stable)
        cast(Tensor, winner.post.camkii).mul_(0.9).add_(0.1 * cast(Tensor, loser.post.camkii))
        cast(Tensor, winner.post.pp1).mul_(0.9).add_(0.1 * cast(Tensor, loser.post.pp1))
        cast(Tensor, winner.post.bdnf).mul_(0.9).add_(0.1 * cast(Tensor, loser.post.bdnf))
    
    # Clone back into loser (to keep count constant)
    _clone_linear_from_(loser, winner, cfg.clone_noise_linear)


@torch.no_grad()
def _clone_linear_from_(dst: SynapticLinear, src: SynapticLinear, noise_scale: float):
    _copy_synaptic_linear_(dst, src)
    _add_noise_(dst.w_slow, noise_scale)
    if dst.w_fast is not None:
        _add_noise_(dst.w_fast, noise_scale)
    if dst.bias is not None:
        _add_noise_(cast(Tensor, dst.bias), noise_scale)
    # reset fast Hebbian traces for cloned expert
    if dst.post is not None:
        cast(Tensor, dst.post.fast).zero_()
        cast(Tensor, dst.post.U).mul_(0.5)
        cast(Tensor, dst.post.V).mul_(0.5)  # keep some eligibility but dampen
    
    # Reset buffers
    if dst.u_buf is not None:
        cast(Tensor, dst.u_buf).zero_()
    if dst.v_buf is not None:
        cast(Tensor, dst.v_buf).zero_()


# ---------------------------------------------------------------------------
# Function-preserving lifecycle (uta.3 — Net2Net / firefly)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _avg_linear_into_(winner: SynapticLinear, loser: SynapticLinear, alpha: float):
    """winner <- alpha*winner + (1-alpha)*loser (weights + synaptic state). No clone-back."""
    winner.w_slow.mul_(alpha).add_((1.0 - alpha) * loser.w_slow)
    if (winner.w_fast is not None) and (loser.w_fast is not None):
        winner.w_fast.mul_(alpha).add_((1.0 - alpha) * loser.w_fast)
    if (winner.bias is not None) and (loser.bias is not None):
        cast(Tensor, winner.bias).mul_(alpha).add_((1.0 - alpha) * cast(Tensor, loser.bias))
    if (winner.post is not None) and (loser.post is not None):
        for name in ("U", "V", "fast", "slow", "camkii", "pp1", "bdnf"):
            wt = getattr(winner.post, name, None)
            lt = getattr(loser.post, name, None)
            if wt is not None and lt is not None:
                cast(Tensor, wt).mul_(alpha).add_((1.0 - alpha) * cast(Tensor, lt))
    if (winner.u_buf is not None) and (loser.u_buf is not None):
        cast(Tensor, winner.u_buf).mul_(alpha).add_((1.0 - alpha) * cast(Tensor, loser.u_buf))
    if (winner.v_buf is not None) and (loser.v_buf is not None):
        cast(Tensor, winner.v_buf).mul_(alpha).add_((1.0 - alpha) * cast(Tensor, loser.v_buf))


@torch.no_grad()
def _antisym_perturb_fc1_(parent_lin: SynapticLinear, child_lin: SynapticLinear, scale: float):
    """Antisymmetric perturbation: parent -= δ, child += δ on fc1 weights.

    Assumes the child currently equals the parent (an exact clone). The pair then straddles
    the original weights symmetrically, so the mean function is preserved to first order
    (0.5·f(W-δ) + 0.5·f(W+δ) ≈ f(W)) while the asymmetry lets them diverge under SGD.

    ``scale`` is RELATIVE to the per-tensor weight RMS, so "0.02" means a 2% divergence
    regardless of the absolute weight magnitude. Using an absolute scale would be catastrophic
    on freshly-initialized weights (std ≈ 0.02): a 0.02 absolute kick is then a ~100%
    perturbation that destroys the function-preservation the gate split buys.
    """
    if scale <= 0:
        return

    def _rms(t: Tensor) -> float:
        return float(t.detach().pow(2).mean().clamp_min(1e-24).sqrt().item())

    d = torch.randn_like(parent_lin.w_slow) * (scale * _rms(parent_lin.w_slow))
    parent_lin.w_slow.sub_(d)
    child_lin.w_slow.add_(d)
    if (parent_lin.w_fast is not None) and (child_lin.w_fast is not None):
        df = torch.randn_like(parent_lin.w_fast) * (scale * _rms(parent_lin.w_fast))
        parent_lin.w_fast.sub_(df)
        child_lin.w_fast.add_(df)


@torch.no_grad()
def _copy_expert_full_(layer: SynapticMoE, dst_idx: int, src_idx: int):
    """Make expert ``dst_idx`` an EXACT clone of ``src_idx``: weights + synaptic state +
    router row + genome Xi + router embedding + metabolic state + routing bias. After this,
    dst computes the identical function AND routes identically to src."""
    _copy_synaptic_linear_(layer.experts[dst_idx].fc1, layer.experts[src_idx].fc1)
    _copy_synaptic_linear_(layer.experts[dst_idx].fc2, layer.experts[src_idx].fc2)
    W = layer.router.weight
    W[dst_idx].copy_(W[src_idx])
    Xi = cast(Tensor, layer.Xi)
    Xi[dst_idx].copy_(Xi[src_idx])
    emb = layer.router_embeddings
    emb[dst_idx].copy_(emb[src_idx])
    fatigue = cast(Tensor, layer.fatigue)
    energy = cast(Tensor, layer.energy)
    fatigue[dst_idx] = fatigue[src_idx]
    energy[dst_idx] = energy[src_idx]
    rb = cast(Tensor, layer.router_logit_bias)
    rb[dst_idx] = rb[src_idx]


@torch.no_grad()
def _function_preserving_split_(
    layer: SynapticMoE, parent_idx: int, dst_idx: int, cfg: SplitMergeConfig
):
    """Split parent into (parent, child@dst_idx) without changing the model output.

    The child is an exact clone of the parent; both then receive a -ln2 routing-logit bias so
    together they reproduce the parent's original routing probability mass (each fires with
    half the gate). Antisymmetric fc1 noise makes them diverge under SGD. In the dense routing
    regime this is exactly output-preserving at the event; in sparse top-k it sharply reduces
    (but does not zero) the discontinuity vs. the legacy noisy clone.
    """
    _copy_expert_full_(layer, dst_idx, parent_idx)
    rb = cast(Tensor, layer.router_logit_bias)
    rb[parent_idx] = rb[parent_idx] - cfg.gate_split_bias
    rb[dst_idx] = rb[parent_idx]
    _antisym_perturb_fc1_(
        layer.experts[parent_idx].fc1, layer.experts[dst_idx].fc1, cfg.fp_divergence_noise
    )


@torch.no_grad()
def _function_preserving_merge_(
    layer: SynapticMoE, winner_idx: int, loser_idx: int, alpha: float, cfg: SplitMergeConfig
):
    """Merge loser into winner without changing the model output.

    The loser is weight-averaged into the winner (routing identity included); the winner takes
    +ln2 routing bias to absorb the loser's probability mass; then the freed loser slot is
    re-seeded as a function-preserving split of the merged winner. The +ln2 (absorb) and the
    split's -ln2 cancel, so both slots end at the winner's pre-merge bias and the combined
    contribution of the pair equals the pre-merge winner+loser contribution (exact for similar
    experts — which the merge criteria already require via high embedding cosine similarity).
    """
    _avg_linear_into_(layer.experts[winner_idx].fc1, layer.experts[loser_idx].fc1, alpha)
    _avg_linear_into_(layer.experts[winner_idx].fc2, layer.experts[loser_idx].fc2, alpha)
    W = layer.router.weight
    W[winner_idx].mul_(alpha).add_((1.0 - alpha) * W[loser_idx])
    Xi = cast(Tensor, layer.Xi)
    Xi[winner_idx].mul_(alpha).add_((1.0 - alpha) * Xi[loser_idx])
    emb = layer.router_embeddings
    emb[winner_idx].mul_(alpha).add_((1.0 - alpha) * emb[loser_idx])
    # Router embeddings are maintained unit-norm everywhere (init + the contrastive EMA update in
    # SynapticMoE.forward), and the forward uses ‖emb‖ as a routing gain. Averaging two unit vectors
    # yields norm < 1, so renormalize to keep the merge function-preserving (and so the clone below,
    # which copies emb[winner] into emb[loser], inherits the unit-norm embedding too).
    emb[winner_idx].div_(emb[winner_idx].norm() + 1e-8)
    rb = cast(Tensor, layer.router_logit_bias)
    rb[winner_idx] = rb[winner_idx] + cfg.gate_split_bias
    _function_preserving_split_(layer, winner_idx, loser_idx, cfg)


@torch.no_grad()
def _merge_expert_into_and_clone_(
    layer: SynapticMoE,
    winner_idx: int,
    loser_idx: int,
    alpha: float,
    cfg: SplitMergeConfig,
):
    """Merge loser into winner (weighted), then clone winner (+noise) into loser slot."""
    winner: SynapticExpert = layer.experts[winner_idx]
    loser: SynapticExpert = layer.experts[loser_idx]

    # 1) Merge parameters into winner AND clone to loser
    # We combined these steps in _merge_linear_into_ if fused
    _merge_linear_into_(winner.fc1, loser.fc1, alpha, cfg)
    _merge_linear_into_(winner.fc2, loser.fc2, alpha, cfg)

    # 3) Router columns: average into winner, clone into loser (with noise)
    W = layer.router.weight
    if W.is_cuda:
        try:
            from bio_inspired_nanochat.kernels import mix_and_shift_rows
            # W is (E, n_embd). We want to mix row[winner] and row[loser].
            mix_and_shift_rows(W, winner_idx, loser_idx, alpha, cfg.clone_noise_router)
        except ImportError:
             # Fallback
            W_w = W[winner_idx]
            W_l = W[loser_idx]
            W_w.mul_(alpha).add_((1.0 - alpha) * W_l)
            W_l.copy_(W_w)
            _add_noise_(W_l, cfg.clone_noise_router)
    else:
        W_w = W[winner_idx]
        W_l = W[loser_idx]
        W_w.mul_(alpha).add_((1.0 - alpha) * W_l)
        W_l.copy_(W_w)
        _add_noise_(W_l, cfg.clone_noise_router)

    # 4) Router embeddings: keep winner embedding; clone loser as orthogonalized perturbed winner
    # This logic is specific (orthogonal perturb), so we keep it manual for now or write another kernel.
    # Since it's just 1 vector per expert, manual is fine.
    emb = layer.router_embeddings  # (E, D)
    e_w = emb[winner_idx : winner_idx + 1]  # (1,D)
    e_l = _orthogonal_perturb_like(e_w.clone(), cfg.clone_noise_embed)
    emb[loser_idx : loser_idx + 1].copy_(e_l)

    # 4b) Genome + routing bias: the loser slot is now a (noisy) clone of the winner, so it must
    # adopt the winner's presynaptic genome Xi and routing-logit bias. Otherwise the reborn expert
    # carries the dead loser's stale genome/bias — a phenotype mismatch (the function-preserving
    # path and _copy_expert_full_ both clone these).
    Xi = cast(Tensor, layer.Xi)
    Xi[loser_idx].copy_(Xi[winner_idx])
    rb = cast(Tensor, layer.router_logit_bias)
    rb[loser_idx] = rb[winner_idx]

    # 5) Reset stats
    fatigue = cast(Tensor, layer.fatigue)
    energy = cast(Tensor, layer.energy)
    fatigue[loser_idx] = 0.0
    energy[loser_idx] = 1.0


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class SplitMergeController:
    def __init__(
        self, model: nn.Module, cfg: SplitMergeConfig, logger: Optional[Any] = None
    ):
        self.model = model
        self.cfg = cfg
        self._last_step = -(10**12)  # ensure first call can run if warmup permits
        self._moe_layers: List[SynapticMoE] = self._find_moe_layers(model)
        self.logger = logger

    def _find_moe_layers(self, module: nn.Module) -> List[SynapticMoE]:
        moes: List[SynapticMoE] = []
        for m in module.modules():
            if isinstance(m, SynapticMoE):
                moes.append(m)
        return moes

    @torch.no_grad()
    def _health(self, layer: SynapticMoE) -> Tensor:
        # Higher is better: H = utilization * energy
        util = cast(Tensor, layer.fatigue).clamp(0, 1)  # fatigue tracks EMA of utilization
        eng = cast(Tensor, layer.energy).clamp(0, 1)
        health = util * eng  # [0,1]
        # NeuroScore credit assignment (de5l): blend the per-expert fitness published by
        # NeuroScore.step into health, so Efficiency/Specialization/Resilience — not just
        # raw utilization*energy — drive split/merge/reset selection. Falls back to pure
        # health when disabled or the score is unavailable/mis-shaped.
        if self.cfg.use_neuroscore:
            score = getattr(layer, "last_neuroscore", None)
            if score is not None and tuple(score.shape) == tuple(health.shape):
                w = min(max(float(self.cfg.neuroscore_weight), 0.0), 1.0)
                score = cast(Tensor, score).to(health.device, health.dtype).clamp(0, 1)
                health = (1.0 - w) * health + w * score
        return health

    @torch.no_grad()
    def _util_weight(self, layer: SynapticMoE, i: int, j: int) -> float:
        if not self.cfg.use_util_weighting:
            return 0.6  # mild bias toward first arg
        # fatigue is utilization proxy
        fat = cast(Tensor, layer.fatigue)
        u_i = fat[i].clamp(0, 1)
        u_j = fat[j].clamp(0, 1)
        s = u_i + u_j
        if float(s.item()) <= 1e-6:
            return 0.5
        return float((u_i / s).item())

    @torch.no_grad()
    def _pick_dead_slots(self, layer: SynapticMoE) -> List[int]:
        if self.cfg.resets_per_call < 1:
            return []
        health = self._health(layer)
        dead = (health <= self.cfg.reset_health_max).nonzero(as_tuple=False).flatten().tolist()
        dead_sorted = sorted(dead, key=lambda e: float(health[e].item()))
        return dead_sorted[: self.cfg.resets_per_call]

    @torch.no_grad()
    def _pick_reset_sources(self, layer: SynapticMoE, k: int) -> List[int]:
        health = self._health(layer)
        order = torch.argsort(health, descending=True).tolist()
        return order[:k]

    @torch.no_grad()
    def _pick_merge_pairs(self, layer: SynapticMoE) -> List[Tuple[int, int]]:
        E = layer.num_experts
        emb = layer.router_embeddings  # (E, D)
        sim = _cosine(emb, emb)  # (E,E)
        health = self._health(layer)
        # candidate mask: high sim and both low health
        sim_mask = sim > self.cfg.merge_cosine_threshold
        low = health <= self.cfg.merge_health_max
        cand = sim_mask & low.unsqueeze(1) & low.unsqueeze(0)
        # remove diagonal
        idx = torch.arange(E, device=emb.device)
        cand[idx, idx] = False
        # score by similarity (higher first)
        scores = sim.masked_fill(~cand, -1.0)  # -1 for invalid
        pairs: List[Tuple[int, int]] = []
        used = set()
        for _ in range(self.cfg.merges_per_call):
            # find max entry
            val, linear_idx = scores.view(-1).max(dim=0)
            if val <= 0:
                break
            i = (linear_idx // E).item()
            j = (linear_idx % E).item()
            if i in used or j in used:
                scores[i, :] = -1.0
                scores[:, i] = -1.0
                scores[j, :] = -1.0
                scores[:, j] = -1.0
                continue
            pairs.append((i, j))
            used.add(i)
            used.add(j)
            # invalidate rows/cols
            scores[i, :] = -1.0
            scores[:, i] = -1.0
            scores[j, :] = -1.0
            scores[:, j] = -1.0
        return pairs

    @torch.no_grad()
    def _pick_split_sources(self, layer: SynapticMoE) -> List[int]:
        health = self._health(layer)
        strong = (
            (health >= self.cfg.split_health_min)
            .nonzero(as_tuple=False)
            .flatten()
            .tolist()
        )
        # take top k strongest
        strong_sorted = sorted(
            strong, key=lambda e: float(health[e].item()), reverse=True
        )
        return strong_sorted[: self.cfg.splits_per_call]

    @torch.no_grad()
    def _weakest_slots(self, layer: SynapticMoE, k: int) -> List[int]:
        health = self._health(layer)
        idx = torch.argsort(health)  # ascending
        return idx[:k].tolist()

    @torch.no_grad()
    def _split_into_slots(
        self,
        layer: SynapticMoE,
        sources: List[int],
        slots: List[int],
        optimizer: OptimizersArg,
        step: int,
    ):
        W = layer.router.weight
        for src, dst in zip(sources, slots):
            if src == dst:
                continue
            if self.cfg.function_preserving:
                # Net2Net / firefly: dst becomes a -ln2-gated twin of the parent so the model
                # output does not jump; antisymmetric fc1 noise lets the pair diverge.
                _function_preserving_split_(layer, src, dst, self.cfg)
            else:
                # Legacy: clone src → dst with noise & embedding tweak (discontinuous).
                _clone_linear_from_(
                    layer.experts[dst].fc1,
                    layer.experts[src].fc1,
                    self.cfg.clone_noise_linear,
                )
                _clone_linear_from_(
                    layer.experts[dst].fc2,
                    layer.experts[src].fc2,
                    self.cfg.clone_noise_linear,
                )
                # router weight row (expert row)
                W[dst].copy_(W[src])
                _add_noise_(W[dst], self.cfg.clone_noise_router)
                # embedding
                layer.router_embeddings[dst : dst + 1].copy_(
                    _orthogonal_perturb_like(
                        layer.router_embeddings[src : src + 1].clone(),
                        self.cfg.clone_noise_embed,
                    )
                )
                # reset stats
                fatigue = cast(Tensor, layer.fatigue)
                energy = cast(Tensor, layer.energy)
                fatigue[dst] = 0.0
                energy[dst] = 1.0
            # emit lineage event: split parent src -> child dst
            if self.logger is not None and hasattr(self.logger, "on_split"):
                try:
                    self.logger.on_split(
                        layer, parent_idx=int(src), child_idx=int(dst), step=step
                    )
                except Exception as _e:
                    if self.cfg.verbose:
                        print(f"[SplitMerge] logger.on_split failed: {_e}")
            # zero optimizer moments for every parameter the event overwrote. The
            # function-preserving path ALSO nudges the parent's fc1 (antisymmetric noise) and
            # the genome Xi, so include both experts' fc1 + Xi alongside dst's weights.
            if optimizer is not None:
                changed = [
                    layer.experts[dst].fc1.w_slow,
                    layer.experts[dst].fc1.w_fast,
                    layer.experts[dst].fc2.w_slow,
                    layer.experts[dst].fc2.w_fast,
                    layer.experts[src].fc1.w_slow,
                    layer.experts[src].fc1.w_fast,
                    W,
                    cast(Any, layer).Xi,
                ]
                if layer.experts[dst].fc1.bias is not None:
                    changed.append(layer.experts[dst].fc1.bias)
                if layer.experts[dst].fc2.bias is not None:
                    changed.append(layer.experts[dst].fc2.bias)
                changed_params: List[nn.Parameter] = [
                    param for param in changed if isinstance(param, nn.Parameter)
                ]
                _zero_optim_moments_for(optimizer, changed_params)

    @torch.no_grad()
    def _do_merges(
        self, layer: SynapticMoE, optimizer: OptimizersArg, step: int
    ):
        pairs = self._pick_merge_pairs(layer)
        if self.cfg.verbose and len(pairs) > 0:
            print(f"[SplitMerge] Merging pairs: {pairs}")
        for i, j in pairs:
            # winner = healthier of the two
            health = self._health(layer)
            if health[i] >= health[j]:
                winner, loser = i, j
            else:
                winner, loser = j, i
            alpha = self._util_weight(layer, winner, loser)
            if self.cfg.function_preserving:
                _function_preserving_merge_(layer, winner, loser, alpha, self.cfg)
            else:
                _merge_expert_into_and_clone_(layer, winner, loser, alpha, self.cfg)
            # emit lineage event: merge parents (winner,loser) -> child lives at index loser (clone slot reused)
            if self.logger is not None and hasattr(self.logger, "on_merge"):
                try:
                    self.logger.on_merge(
                        layer,
                        parent_i=int(winner),
                        parent_j=int(loser),
                        child_idx=int(loser),
                        step=step,
                    )
                except Exception as _e:
                    if self.cfg.verbose:
                        print(f"[SplitMerge] logger.on_merge failed: {_e}")
            # zero optimizer moments for both experts + router rows
            if optimizer is not None:
                changed = [
                    layer.experts[winner].fc1.w_slow,
                    layer.experts[winner].fc1.w_fast,
                    layer.experts[winner].fc2.w_slow,
                    layer.experts[winner].fc2.w_fast,
                    layer.experts[loser].fc1.w_slow,
                    layer.experts[loser].fc1.w_fast,
                    layer.experts[loser].fc2.w_slow,
                    layer.experts[loser].fc2.w_fast,
                    layer.router.weight,
                    cast(Any, layer).Xi,
                ]
                if layer.experts[winner].fc1.bias is not None:
                    changed.append(layer.experts[winner].fc1.bias)
                if layer.experts[winner].fc2.bias is not None:
                    changed.append(layer.experts[winner].fc2.bias)
                if layer.experts[loser].fc1.bias is not None:
                    changed.append(layer.experts[loser].fc1.bias)
                if layer.experts[loser].fc2.bias is not None:
                    changed.append(layer.experts[loser].fc2.bias)
                changed_params = [
                    param for param in changed if isinstance(param, nn.Parameter)
                ]
                _zero_optim_moments_for(optimizer, changed_params)

    @torch.no_grad()
    def step(self, global_step: int, optimizer: OptimizersArg = None):
        if not self.cfg.enabled:
            return
        if global_step < self.cfg.warmup_steps:
            return
        if global_step - self._last_step < self.cfg.min_step_interval:
            return

        if not _is_rank0():
            # Non-zero ranks just wait for broadcast after rank 0 modifies params
            if self.cfg.ddp_broadcast:  # ensure we hit the barrier roughly in sync
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
            return

        if self.cfg.verbose:
            print(f"[SplitMerge] step @ {global_step}")

        # Perform operations layer-by-layer on rank 0
        for layer in self._moe_layers:
            # 1) merges
            self._do_merges(layer, optimizer, global_step)
            # 2) splits
            sources = self._pick_split_sources(layer)
            if len(sources) > 0 and self.cfg.splits_per_call > 0:
                slots = self._weakest_slots(
                    layer, min(len(sources), self.cfg.splits_per_call)
                )
                if self.cfg.verbose:
                    print(f"[SplitMerge] Splitting {list(zip(sources, slots))}")
                self._split_into_slots(layer, sources, slots, optimizer, global_step)
            # 3) resets for dead experts (clone from healthiest)
            dead_slots = self._pick_dead_slots(layer)
            if dead_slots:
                sources = self._pick_reset_sources(layer, max(len(dead_slots), 1))
                reset_sources: List[int] = []
                reset_slots: List[int] = []
                for slot in dead_slots:
                    src = next((s for s in sources if s != slot), None)
                    if src is None:
                        continue
                    reset_sources.append(src)
                    reset_slots.append(slot)
                if reset_sources:
                    if self.cfg.verbose:
                        print(f"[SplitMerge] Resetting {list(zip(reset_sources, reset_slots))}")
                    self._split_into_slots(layer, reset_sources, reset_slots, optimizer, global_step)

        # Broadcast updated params to all ranks (DDP)
        if self.cfg.ddp_broadcast and dist.is_available() and dist.is_initialized():
            for layer in self._moe_layers:
                _broadcast_module_params(layer)
            dist.barrier()

        self._last_step = global_step
