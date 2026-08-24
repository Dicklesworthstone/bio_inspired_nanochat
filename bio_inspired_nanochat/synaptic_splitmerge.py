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
import hashlib
from dataclasses import asdict, dataclass
from collections import defaultdict
from typing import List, Tuple, Optional, Iterable, Any, Dict, Mapping, Set, cast
import numpy as np
from bio_inspired_nanochat.torch_imports import torch, nn, Tensor
import torch.distributed as torch_dist

from .structural_geometry import (
    MergeCertificate,
    SpectralCertificate,
    StructuralGeometryMonitor,
    StructuralGeometryMonitorConfig,
    StructuralGeometryRecord,
    ot_merge_certificate,
)
from .synaptic import SynapticMoE, SynapticExpert, SynapticLinear

dist = cast(Any, torch_dist)

# An optimizer, or a collection of them. Synaptic models split parameters across AdamW
# (1D/embeddings) AND Muon (2D matrices), so lifecycle moment-resets must reach all of
# them — see _zero_optim_moments_for (vg9.3).
OptimizersArg = Optional[torch.optim.Optimizer | Iterable[torch.optim.Optimizer]]

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
    # 0642.5.2.2 thresholds for the certificate-driven structural lifecycle,
    # gated solely by SynapticConfig.topological_nas. Live routing activations
    # trigger split/birth through H0 persistence, expert spectra bound split
    # noise, and OT cost ranks merge pairs. Missing or uncertified evidence
    # falls back to the unchanged UTA health lifecycle.
    topological_kappa_target: float = 50.0
    topological_merge_cost_ratio_max: float = 0.05
    topological_functional_distance_max: float = 0.1
    topological_persistence_ratio_threshold: float = 3.0
    topological_coverage_distance_threshold: float = 0.25
    topological_max_points: int = 256
    topological_max_dim: int = 8
    topological_max_persistence_features: int = 8
    topological_max_samples_per_tensor: int = 1024
    topological_max_spectral_candidates: int = 2
    topological_max_exact_merge_candidates: int = 2
    # VARIABLE EXPERT COUNT (uta.4): real neurogenesis/apoptosis. When enabled the
    # controller may APPEND fresh expert slots under sustained split pressure and
    # REMOVE surplus dead slots (folding their contribution into the healthiest
    # survivor), rebuilding the router/buffers/genome and synchronizing optimizer
    # param-groups (survivors keep their moments; new params start fresh; removed
    # params are dropped). Bounded by hard floors/caps and a cumulative growth
    # budget expressed as a fraction of the initial total expert count — since MoE
    # FLOPs scale linearly with expert count, the budget IS the FLOP budget.
    variable_expert_count: bool = False
    min_experts: int = 2
    max_experts: int = 64
    growth_budget_pct: float = 0.5  # max NET added experts, fraction of initial total
    neuroscore_weight: float = 0.5  # blend weight in [0,1]: health=(1-w)*health + w*score
    # Logging
    verbose: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.topological_kappa_target) or self.topological_kappa_target <= 1.0:
            raise ValueError("topological_kappa_target must be finite and > 1")
        if (
            not math.isfinite(self.topological_merge_cost_ratio_max)
            or self.topological_merge_cost_ratio_max < 0.0
        ):
            raise ValueError("topological_merge_cost_ratio_max must be finite and >= 0")
        if (
            not math.isfinite(self.topological_functional_distance_max)
            or self.topological_functional_distance_max < 0.0
        ):
            raise ValueError("topological_functional_distance_max must be finite and >= 0")
        if (
            not math.isfinite(self.topological_coverage_distance_threshold)
            or not 0.0 <= self.topological_coverage_distance_threshold <= 2.0
        ):
            raise ValueError(
                "topological_coverage_distance_threshold must be finite and in [0, 2]"
            )
        if self.topological_max_samples_per_tensor < 2:
            raise ValueError("topological_max_samples_per_tensor must be >= 2")
        if self.topological_max_spectral_candidates < 1:
            raise ValueError("topological_max_spectral_candidates must be >= 1")
        if self.topological_max_exact_merge_candidates < 1:
            raise ValueError("topological_max_exact_merge_candidates must be >= 1")
        StructuralGeometryMonitorConfig(
            persistence_ratio_threshold=self.topological_persistence_ratio_threshold,
            max_points=self.topological_max_points,
            max_dim=self.topological_max_dim,
            max_persistence_features=self.topological_max_persistence_features,
        )


@dataclass(frozen=True)
class TopologicalLifecycleDecision:
    """JSON-safe audit record for one geometry-driven lifecycle decision."""

    step: int
    layer_index: int
    mode: str
    action: str
    reason: str
    split_source: int | None = None
    split_destination: int | None = None
    merge_pair: tuple[int, int] | None = None
    split_noise_norm: float | None = None
    kappa_bound: float | None = None
    persistence_ratio: float | None = None
    merge_cost_ratio: float | None = None
    functional_distance: float | None = None
    coverage_distance: float | None = None


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


def _randn_like(t: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
    """``torch.randn_like`` that also accepts an explicit generator (uta.5).

    Some torch builds reject ``generator=`` on ``randn_like``; going through
    ``torch.randn(shape, ...)`` keeps one code path for both.
    """
    if generator is None:
        return torch.randn_like(t)
    return torch.randn(tuple(t.shape), dtype=t.dtype, device=t.device, generator=generator)


@torch.no_grad()
def _orthogonal_perturb_like(
    vec: Tensor, noise_scale: float, generator: Optional[torch.Generator] = None
) -> Tensor:
    """Return a unit-length vector: normalized(vec + noise in orthogonal subspace).

    When noise_scale <= 0, returns vec unchanged (no normalization, exact copy).
    """
    if noise_scale <= 0:
        return vec.clone()
    noise = _randn_like(vec, generator)
    proj = (noise * vec).sum(dim=-1, keepdim=True) * vec
    tangent = noise - proj
    out = vec + noise_scale * tangent
    return out / (out.norm(dim=-1, keepdim=True) + 1e-8)


@torch.no_grad()
def _add_noise_(
    t: Tensor, scale: float, generator: Optional[torch.Generator] = None
):
    if scale <= 0:
        return
    t.add_(_randn_like(t, generator) * scale)


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
        dst.bias.copy_(src.bias)
    # postsyn state
    if (dst.post is not None) and (src.post is not None):
        dst.post.U.copy_(src.post.U)
        dst.post.V.copy_(src.post.V)
        dst.post.fast.copy_(src.post.fast)
        dst.post.slow.copy_(src.post.slow)
        # buffers
        dst.post.camkii.copy_(src.post.camkii)
        dst.post.pp1.copy_(src.post.pp1)
        dst.post.bdnf.copy_(src.post.bdnf)
        if hasattr(dst.post, "bdnf_hebb_accum") and hasattr(src.post, "bdnf_hebb_accum"):
            dst.post.bdnf_hebb_accum.copy_(src.post.bdnf_hebb_accum)
        if hasattr(dst.post, "_last_hebb_delta_mag") and hasattr(src.post, "_last_hebb_delta_mag"):
            dst.post._last_hebb_delta_mag.copy_(src.post._last_hebb_delta_mag)
    # Linear buffers
    if (dst.u_buf is not None) and (src.u_buf is not None):
        dst.u_buf.copy_(src.u_buf)
    if (dst.v_buf is not None) and (src.v_buf is not None):
        dst.v_buf.copy_(src.v_buf)


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
                mix_and_shift_tensors(winner.bias, loser.bias, alpha, cfg.clone_noise_linear)
            
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
                winner.post.U.mul_(alpha).add_((1.0 - alpha) * loser.post.U)
                winner.post.V.mul_(alpha).add_((1.0 - alpha) * loser.post.V)
                winner.post.fast.mul_(alpha).add_((1.0 - alpha) * loser.post.fast)
                winner.post.slow.mul_(alpha).add_((1.0 - alpha) * loser.post.slow)
                
                winner.post.camkii.mul_(0.9).add_(0.1 * loser.post.camkii)
                winner.post.pp1.mul_(0.9).add_(0.1 * loser.post.pp1)
                winner.post.bdnf.mul_(0.9).add_(0.1 * loser.post.bdnf)
                if hasattr(winner.post, "bdnf_hebb_accum") and hasattr(loser.post, "bdnf_hebb_accum"):
                    winner.post.bdnf_hebb_accum.mul_(0.9).add_(
                        0.1 * loser.post.bdnf_hebb_accum
                    )
                if hasattr(winner.post, "_last_hebb_delta_mag") and hasattr(loser.post, "_last_hebb_delta_mag"):
                    winner.post._last_hebb_delta_mag.copy_(
                        loser.post._last_hebb_delta_mag
                    )
                
                # Clone state to loser (with reset logic)
                loser.post.U.copy_(winner.post.U).mul_(0.5)
                loser.post.V.copy_(winner.post.V).mul_(0.5)
                loser.post.fast.zero_() # Reset fast weights
                loser.post.slow.copy_(winner.post.slow) # Keep slow weights? Or reset? Usually keep base knowledge.
                
                loser.post.camkii.copy_(winner.post.camkii)
                loser.post.pp1.copy_(winner.post.pp1)
                loser.post.bdnf.copy_(winner.post.bdnf)
                if hasattr(loser.post, "bdnf_hebb_accum") and hasattr(winner.post, "bdnf_hebb_accum"):
                    loser.post.bdnf_hebb_accum.copy_(
                        winner.post.bdnf_hebb_accum
                    )
                if hasattr(loser.post, "_last_hebb_delta_mag") and hasattr(winner.post, "_last_hebb_delta_mag"):
                    loser.post._last_hebb_delta_mag.copy_(
                        winner.post._last_hebb_delta_mag
                    )
            
            # Reset eligibility buffers in Linear
            if loser.u_buf is not None:
                loser.u_buf.zero_()
            if loser.v_buf is not None:
                loser.v_buf.zero_()
            
            return
        except ImportError:
            pass

    # Fallback / CPU logic
    # winner = alpha * winner + (1-alpha) * loser
    winner.w_slow.mul_(alpha).add_((1.0 - alpha) * loser.w_slow)
    if (winner.w_fast is not None) and (loser.w_fast is not None):
        winner.w_fast.mul_(alpha).add_((1.0 - alpha) * loser.w_fast)
    if (winner.bias is not None) and (loser.bias is not None):
        winner.bias.mul_(alpha).add_((1.0 - alpha) * loser.bias)
    if (winner.post is not None) and (loser.post is not None):
        winner.post.U.mul_(alpha).add_((1.0 - alpha) * loser.post.U)
        winner.post.V.mul_(alpha).add_((1.0 - alpha) * loser.post.V)
        winner.post.fast.mul_(alpha).add_((1.0 - alpha) * loser.post.fast)
        winner.post.slow.mul_(alpha).add_((1.0 - alpha) * loser.post.slow)
        
        # gate and enzymes: bias toward winner (more stable)
        winner.post.camkii.mul_(0.9).add_(0.1 * loser.post.camkii)
        winner.post.pp1.mul_(0.9).add_(0.1 * loser.post.pp1)
        winner.post.bdnf.mul_(0.9).add_(0.1 * loser.post.bdnf)
    
    # Clone back into loser (to keep count constant)
    _clone_linear_from_(loser, winner, cfg.clone_noise_linear)


def _clone_linear_from_(
    dst: SynapticLinear,
    src: SynapticLinear,
    noise_scale: float,
    generator: Optional[torch.Generator] = None,
):
    _copy_synaptic_linear_(dst, src)
    _add_noise_(dst.w_slow, noise_scale, generator=generator)
    if dst.w_fast is not None:
        _add_noise_(dst.w_fast, noise_scale, generator=generator)
    if dst.bias is not None:
        _add_noise_(dst.bias, noise_scale, generator=generator)
    # reset fast Hebbian traces for cloned expert
    if dst.post is not None:
        dst.post.fast.zero_()
        dst.post.U.mul_(0.5)
        dst.post.V.mul_(0.5)  # keep some eligibility but dampen
    
    # Reset buffers
    if dst.u_buf is not None:
        dst.u_buf.zero_()
    if dst.v_buf is not None:
        dst.v_buf.zero_()


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
        winner.bias.mul_(alpha).add_((1.0 - alpha) * loser.bias)
    if (winner.post is not None) and (loser.post is not None):
        winner.post.U.mul_(alpha).add_((1.0 - alpha) * loser.post.U)
        winner.post.V.mul_(alpha).add_((1.0 - alpha) * loser.post.V)
        winner.post.fast.mul_(alpha).add_((1.0 - alpha) * loser.post.fast)
        winner.post.slow.mul_(alpha).add_((1.0 - alpha) * loser.post.slow)
        winner.post.camkii.mul_(alpha).add_((1.0 - alpha) * loser.post.camkii)
        winner.post.pp1.mul_(alpha).add_((1.0 - alpha) * loser.post.pp1)
        winner.post.bdnf.mul_(alpha).add_((1.0 - alpha) * loser.post.bdnf)
    if (winner.u_buf is not None) and (loser.u_buf is not None):
        winner.u_buf.mul_(alpha).add_((1.0 - alpha) * loser.u_buf)
    if (winner.v_buf is not None) and (loser.v_buf is not None):
        winner.v_buf.mul_(alpha).add_((1.0 - alpha) * loser.v_buf)


@torch.no_grad()
def _antisym_perturb_fc1_(
    parent_lin: SynapticLinear,
    child_lin: SynapticLinear,
    scale: float,
    spectral_norm_cap: float | None = None,
    generator: Optional[torch.Generator] = None,
):
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
    if spectral_norm_cap is not None and spectral_norm_cap <= 0.0:
        return

    def _rms(t: Tensor) -> float:
        return float(t.detach().pow(2).mean().clamp_min(1e-24).sqrt().item())

    def _cap_spectral_norm(delta: Tensor) -> Tensor:
        if spectral_norm_cap is None:
            return delta
        cap = max(0.0, float(spectral_norm_cap))
        norm = float(torch.linalg.matrix_norm(delta.float(), ord=2).item())
        if norm <= 0.0 or norm <= cap:
            return delta
        return delta * (cap / norm)

    d = _cap_spectral_norm(
        _randn_like(parent_lin.w_slow, generator)
        * (scale * _rms(parent_lin.w_slow))
    )
    parent_lin.w_slow.sub_(d)
    child_lin.w_slow.add_(d)
    if (parent_lin.w_fast is not None) and (child_lin.w_fast is not None):
        df = _cap_spectral_norm(
            _randn_like(parent_lin.w_fast, generator)
            * (scale * _rms(parent_lin.w_fast))
        )
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
    if layer.Xi is not None:
        layer.Xi[dst_idx].copy_(layer.Xi[src_idx])
    emb = layer.router_embeddings
    emb[dst_idx].copy_(emb[src_idx])
    layer.fatigue[dst_idx] = layer.fatigue[src_idx]
    layer.energy[dst_idx] = layer.energy[src_idx]
    layer.router_logit_bias[dst_idx] = layer.router_logit_bias[src_idx]


@torch.no_grad()
def _function_preserving_split_(
    layer: SynapticMoE,
    parent_idx: int,
    dst_idx: int,
    cfg: SplitMergeConfig,
    spectral_noise_norm: float | None = None,
    generator: Optional[torch.Generator] = None,
):
    """Split parent into (parent, child@dst_idx) without changing the model output.

    The child is an exact clone of the parent; both then receive a -ln2 routing-logit bias so
    together they reproduce the parent's original routing probability mass (each fires with
    half the gate). Antisymmetric fc1 noise makes them diverge under SGD. In the dense routing
    regime this is exactly output-preserving at the event; in sparse top-k it sharply reduces
    (but does not zero) the discontinuity vs. the legacy noisy clone.
    """
    _copy_expert_full_(layer, dst_idx, parent_idx)
    layer.router_logit_bias[parent_idx] = layer.router_logit_bias[parent_idx] - cfg.gate_split_bias
    layer.router_logit_bias[dst_idx] = layer.router_logit_bias[parent_idx]
    _antisym_perturb_fc1_(
        layer.experts[parent_idx].fc1,
        layer.experts[dst_idx].fc1,
        cfg.fp_divergence_noise,
        spectral_noise_norm,
        generator=generator,
    )


@torch.no_grad()
def _ot_barycenter_slow_weight_targets(
    winner: SynapticExpert, loser: SynapticExpert
) -> Tuple[Tensor, Tensor]:
    """Return the exact empirical 1-D W2 midpoint for both slow-weight tensors.

    The monotone OT coupling pairs equal-rank entries. The midpoint values are
    restored in the winner's stable rank order, so the installed *joint marginal*
    over ``fc1+fc2`` is precisely the certified 50/50 Wasserstein barycenter while
    retaining a deterministic matrix layout.
    """
    winner_parts = (winner.fc1.w_slow, winner.fc2.w_slow)
    loser_parts = (loser.fc1.w_slow, loser.fc2.w_slow)
    a = torch.cat([part.reshape(-1) for part in winner_parts])
    b = torch.cat([part.reshape(-1) for part in loser_parts])
    if a.numel() != b.numel():
        raise ValueError("OT merge requires equal-sized expert slow weights")
    order_a = torch.argsort(a, stable=True)
    sorted_a = a[order_a]
    sorted_b = torch.sort(b, stable=True).values
    target = torch.empty_like(a)
    target[order_a] = 0.5 * (sorted_a + sorted_b)
    fc1_n = winner.fc1.w_slow.numel()
    return (
        target[:fc1_n].reshape_as(winner.fc1.w_slow),
        target[fc1_n:].reshape_as(winner.fc2.w_slow),
    )


@torch.no_grad()
def _consolidate_expert_pair_(
    layer: SynapticMoE,
    winner_idx: int,
    loser_idx: int,
    alpha: float,
    cfg: SplitMergeConfig,
    ot_barycenter: bool = False,
) -> None:
    """Consolidate ``loser`` into ``winner`` and transfer its routing mass."""
    winner_expert = layer.experts[winner_idx]
    loser_expert = layer.experts[loser_idx]
    ot_targets = (
        _ot_barycenter_slow_weight_targets(winner_expert, loser_expert)
        if ot_barycenter
        else None
    )
    _avg_linear_into_(winner_expert.fc1, loser_expert.fc1, alpha)
    _avg_linear_into_(winner_expert.fc2, loser_expert.fc2, alpha)
    if ot_targets is not None:
        winner_expert.fc1.w_slow.copy_(ot_targets[0])
        winner_expert.fc2.w_slow.copy_(ot_targets[1])
    W = layer.router.weight
    W[winner_idx].mul_(alpha).add_((1.0 - alpha) * W[loser_idx])
    if layer.Xi is not None:
        layer.Xi[winner_idx].mul_(alpha).add_((1.0 - alpha) * layer.Xi[loser_idx])
    emb = layer.router_embeddings
    emb[winner_idx].mul_(alpha).add_((1.0 - alpha) * emb[loser_idx])
    # Router embeddings are maintained unit-norm everywhere (init + the contrastive EMA update in
    # SynapticMoE.forward), and the forward uses ‖emb‖ as a routing gain. Averaging two unit vectors
    # yields norm < 1, so renormalize to preserve that routing-gain invariant.
    emb[winner_idx].div_(emb[winner_idx].norm() + 1e-8)
    layer.router_logit_bias[winner_idx] = layer.router_logit_bias[winner_idx] + cfg.gate_split_bias


@torch.no_grad()
def _function_preserving_merge_(
    layer: SynapticMoE,
    winner_idx: int,
    loser_idx: int,
    alpha: float,
    cfg: SplitMergeConfig,
    ot_barycenter: bool = False,
    generator: Optional[torch.Generator] = None,
):
    """Merge loser into winner, then refill its slot with a twin of the winner.

    The balanced topological path installs the certified empirical W2 midpoint;
    UTA retains its utilization-weighted elementwise merge. The slot refill keeps
    expert count stable and shares the consolidated routing mass across the twins.
    """
    _consolidate_expert_pair_(
        layer,
        winner_idx,
        loser_idx,
        alpha,
        cfg,
        ot_barycenter=ot_barycenter,
    )
    _function_preserving_split_(
        layer,
        winner_idx,
        loser_idx,
        cfg,
        spectral_noise_norm=0.0 if ot_barycenter else None,
        generator=generator,
    )


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
    if layer.Xi is not None:
        layer.Xi[loser_idx].copy_(layer.Xi[winner_idx])
    layer.router_logit_bias[loser_idx] = layer.router_logit_bias[winner_idx]

    # 5) Reset stats
    layer.fatigue[loser_idx] = 0.0
    layer.energy[loser_idx] = 1.0


# ---------------------------------------------------------------------------
# Variable expert count (uta.4) — structural surgery + optimizer synchronization
# ---------------------------------------------------------------------------


def _expert_hidden_mult(layer: SynapticMoE) -> int:
    """Recover the ``hidden_mult`` CONSTRUCTOR ARGUMENT of existing experts.

    ``SynapticExpert(n_embd, hidden_mult, ...)`` sizes its first linear as
    ``hidden_mult * n_embd`` outputs; the storage orientation of the custom
    ``SynapticLinear`` weights is an implementation detail, so derive the
    multiplier from the parameter VOLUME instead of any single axis.
    """
    numel = int(layer.experts[0].fc1.w_slow.numel())
    n_embd = int(layer.router.in_features)
    return max(1, numel // (n_embd * n_embd))


def _norm_param_name(name: str) -> str:
    """Collapse ModuleList indices so sibling expert params share a layout key."""
    out = []
    for p in name.split("."):
        out.append("N" if p.isdigit() else p)
    return ".".join(out)


def _as_opt_list(optimizers: OptimizersArg) -> List[torch.optim.Optimizer]:
    if optimizers is None:
        return []
    if isinstance(optimizers, torch.optim.Optimizer):
        return [optimizers]
    return list(optimizers)


@torch.no_grad()
def snapshot_optimizer_state(optimizers: OptimizersArg) -> Dict[int, Any]:
    """Map ``id(param) -> state`` for every parameter that HAS optimizer state.

    Surviving parameters keep the same Python objects across uta.4 surgery, so
    their moment buffers stay valid and are reattached verbatim; params without
    state (never stepped) simply don't appear.
    """
    snap: Dict[int, Any] = {}
    for opt in _as_opt_list(optimizers):
        for group in opt.param_groups:
            for p in group["params"]:
                if id(p) not in snap and p in opt.state:
                    snap[id(p)] = opt.state[p]
    return snap


def capture_optimizer_layout(
    optimizers: OptimizersArg, model: nn.Module
) -> Dict[str, Tuple[int, int, Dict[str, Any]]]:
    """Record which optimizer group each parameter NAME (index-normalized) belongs to.

    uta.4 replaces router/genome Parameter objects under the SAME attribute paths,
    and appended experts produce names matching their siblings once indices are
    normalized — so this layout stays valid across resize events.
    """
    id2name = {id(p): n for n, p in model.named_parameters()}
    layout: Dict[str, Tuple[int, int, Dict[str, Any]]] = {}
    for oi, opt in enumerate(_as_opt_list(optimizers)):
        for gi, group in enumerate(opt.param_groups):
            hypers = {k: v for k, v in group.items() if k != "params"}
            for p in group["params"]:
                name = id2name.get(id(p))
                if name is None:
                    continue
                key = f"{oi}:{_norm_param_name(name)}"
                layout[key] = (oi, gi, hypers)
                layout.setdefault(f"norm:{_norm_param_name(name)}", (oi, gi, hypers))
    return layout


def synchronize_optimizers_with_model(
    optimizers: OptimizersArg,
    model: nn.Module,
    layout: Dict[str, Tuple[int, int, Dict[str, Any]]],
    state_snapshot: Dict[int, Any],
) -> None:
    """Re-point optimizer param_groups at the post-surgery parameter set.

    - survivors keep both their group (by normalized name) and their moments;
    - brand-new params join the group of their normalized-name siblings (falling
      back to the 2D→matrix / 1D→elementwise rule), starting with NO moments;
    - removed params drop out of groups and their state is released.
    """
    opts = _as_opt_list(optimizers)
    if not opts:
        return
    membership: Dict[Tuple[int, int], List[nn.Parameter]] = {}
    seen: Set[int] = set()
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        hit = layout.get(f"norm:{_norm_param_name(name)}")
        if hit is not None:
            oi, gi, _hyp = hit
            oi = min(oi, len(opts) - 1)
        elif len(opts) > 1:
            # matrix params go to the LAST optimizer (Muon in setup_optimizers),
            # elementwise to the first (AdamW) — mirrors setup_optimizers order.
            oi = len(opts) - 1 if p.ndim >= 2 else 0
            gi = 0
        else:
            oi = gi = 0
        seen.add(id(p))
        membership.setdefault((oi, gi), []).append(p)
    for oi, opt in enumerate(opts):
        for gi, group in enumerate(opt.param_groups):
            group["params"] = membership.get((oi, gi), [])
        # release stale state; reattach survivor state; new params start fresh
        new_state = {}
        for group in opt.param_groups:
            for p in group["params"]:
                if id(p) in state_snapshot:
                    new_state[p] = state_snapshot[id(p)]
        opt.state = defaultdict(dict, new_state)


@torch.no_grad()
def _resize_layer_experts_(
    layer: SynapticMoE,
    target_E: int,
    seed_idx: int,
    cfg: SplitMergeConfig,
    generator: Optional[torch.Generator] = None,
) -> List[int]:
    """Grow/shrink ``layer`` to ``target_E`` experts IN PLACE; returns touched slots.

    Grow: appends clones of expert ``seed_idx`` (full weights + synaptic state via
    :func:`_copy_expert_full_`), gives each a ``-ln2`` routing bias (low-mass fresh
    capacity, same convention as the uta.3 twin split) and small fc1 divergence noise.
    Shrink with ``target_E < E`` drops the LAST ``E - target_E`` experts (callers fold
    their contribution into a survivor first). Router, genome Xi, embeddings and the
    metabolic buffers are rebuilt at the new size; NeuroScore bookkeeping self-heals
    on its next step (size-mismatch reset).

    NOTE (honest scope): unlike slot-reuse splits, count growth changes the top-k
    routing distribution, so the event is NOT output-preserving; survivor parameters
    are untouched bit-exact.
    """
    E_old = int(layer.num_experts)
    if target_E == E_old or target_E <= 0:
        return []
    dev = layer.router.weight.device
    dtype = layer.router.weight.dtype
    n_embd = layer.router.in_features
    hidden = _expert_hidden_mult(layer)

    old_W = layer.router.weight.detach().clone()
    old_Xi = layer.Xi.detach().clone() if layer.Xi is not None else None
    old_emb = layer.router_embeddings.detach().clone()
    old_rb = layer.router_logit_bias.detach().clone()
    old_fat = layer.fatigue.detach().clone()
    old_eng = layer.energy.detach().clone()

    touched: List[int] = []
    if target_E > E_old:
        n_new = target_E - E_old
        new_experts = [
            SynapticExpert(n_embd, hidden, layer.cfg).to(device=dev, dtype=dtype)
            for _ in range(n_new)
        ]
        layer.experts.extend(new_experts)
        W_new = torch.cat(
            [old_W, old_W[seed_idx].repeat(n_new, 1)]
        )
        emb_new = torch.cat([old_emb, old_emb[seed_idx].repeat(n_new, 1)])
        rb_new = torch.cat(
            [old_rb, torch.full((n_new,), -cfg.gate_split_bias, device=dev, dtype=dtype)]
        )
        fat_new = torch.cat([old_fat, torch.zeros(n_new, device=dev, dtype=dtype)])
        eng_new = torch.cat([old_eng, torch.ones(n_new, device=dev, dtype=dtype)])
        new_router = nn.Linear(n_embd, target_E, bias=False).to(device=dev, dtype=dtype)
        with torch.no_grad():
            new_router.weight.copy_(W_new)
        layer.router = new_router
        if old_Xi is not None:
            Xi_new = torch.cat([old_Xi, old_Xi[seed_idx].repeat(n_new, 1)])
            layer.Xi = nn.Parameter(Xi_new)
        layer.router_embeddings = nn.Parameter(emb_new, requires_grad=False)
        layer.register_buffer("router_logit_bias", rb_new)
        layer.register_buffer("fatigue", fat_new)
        layer.register_buffer("energy", eng_new)
        for dst in range(E_old, target_E):
            _copy_expert_full_(layer, dst_idx=dst, src_idx=seed_idx)
            # the full copy clones the seed's routing bias too; a fresh twin must
            # still start at LOW mass, so (re)apply the -ln2 gate afterwards.
            layer.router_logit_bias[dst] = -cfg.gate_split_bias
            e = layer.experts[dst]
            _add_noise_(
                e.fc1.w_slow, cfg.clone_noise_linear * 0.5, generator=generator
            )
            layer.fatigue[dst] = 0.0
            layer.energy[dst] = 1.0
            touched.append(dst)
        # appended experts must not inherit any stale hook markers
        object.__setattr__(layer, "last_ctx", {})
    else:
        n_drop = E_old - target_E
        drop = list(range(E_old - n_drop, E_old))
        keep = [i for i in range(E_old) if i not in drop]
        layer.experts = nn.ModuleList([layer.experts[i] for i in keep])
        new_router = nn.Linear(n_embd, target_E, bias=False).to(device=dev, dtype=dtype)
        with torch.no_grad():
            new_router.weight.copy_(old_W[keep])
        layer.router = new_router
        if old_Xi is not None:
            layer.Xi = nn.Parameter(old_Xi[keep])
        layer.router_embeddings = nn.Parameter(old_emb[keep], requires_grad=False)
        layer.register_buffer("router_logit_bias", old_rb[keep])
        layer.register_buffer("fatigue", old_fat[keep])
        layer.register_buffer("energy", old_eng[keep])
        object.__setattr__(layer, "last_ctx", {})
        touched.extend(drop)

    setattr(layer, "num_experts", target_E)
    # NeuroScore re-arms lazily via its stats size-check; per-expert capture
    # guards keep survivor hooks from duplicating across resizes.
    return touched


@torch.no_grad()
def _fold_expert_into_(layer: SynapticMoE, victim_idx: int, keeper_idx: int, alpha: float) -> None:
    """Average a doomed expert's weights into a survivor before removing its slot."""
    keeper = layer.experts[keeper_idx]
    victim = layer.experts[victim_idx]
    _avg_linear_into_(keeper.fc1, victim.fc1, alpha)
    _avg_linear_into_(keeper.fc2, victim.fc2, alpha)
    # absorb routing mass: +ln2 lets the survivor cover the removed row's gate share
    layer.router_logit_bias[keeper_idx] += math.log(2.0)


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class SplitMergeController:
    def __init__(
        self,
        model: nn.Module,
        cfg: SplitMergeConfig,
        logger: Optional[Any] = None,
        event_logger: Optional[Any] = None,
    ):
        self.model = model
        self.cfg = cfg
        self._last_step = -(10**12)  # ensure first call can run if warmup permits
        self._moe_layers: List[SynapticMoE] = self._find_moe_layers(model)
        topology_flags = {bool(layer.cfg.topological_nas) for layer in self._moe_layers}
        if len(topology_flags) > 1:
            raise ValueError("topological_nas must be configured consistently across MoE layers")
        self.topological_nas = next(iter(topology_flags), False)
        if self.topological_nas and not cfg.function_preserving:
            raise ValueError("topological_nas requires function_preserving=True")
        if (
            dist.is_available()
            and dist.is_initialized()
            and _world_size() > 1
            and not cfg.ddp_broadcast
        ):
            raise ValueError("DDP lifecycle requires ddp_broadcast=True")
        self.logger = logger
        self.event_logger = event_logger or (logger if hasattr(logger, "event") else None)
        self.geometry_monitor = StructuralGeometryMonitor(
            StructuralGeometryMonitorConfig(
                persistence_ratio_threshold=cfg.topological_persistence_ratio_threshold,
                max_points=cfg.topological_max_points,
                max_dim=cfg.topological_max_dim,
                max_persistence_features=cfg.topological_max_persistence_features,
            )
        )
        self.topological_decisions: List[TopologicalLifecycleDecision] = []
        # uta.4 bookkeeping: MoE FLOPs scale linearly with expert count, so a cap on
        # NET added experts (fraction of the initial total) is the compute budget.
        self._initial_total_experts = sum(m.num_experts for m in self._moe_layers)
        self._net_added_experts = 0

    def state_dict(self) -> Dict[str, int]:
        """Return scheduling and growth-budget state needed for exact resume."""
        return {
            "last_step": int(self._last_step),
            "initial_total_experts": int(self._initial_total_experts),
            "net_added_experts": int(self._net_added_experts),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore lifecycle state, rejecting inconsistent growth accounting."""
        required = {"last_step", "initial_total_experts", "net_added_experts"}
        missing = sorted(required - state.keys())
        if missing:
            raise ValueError(f"split/merge state is missing fields: {missing}")
        values: Dict[str, int] = {}
        for name in required:
            value = state[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"split/merge state field {name!r} must be an integer")
            values[name] = value
        initial = values["initial_total_experts"]
        net_added = values["net_added_experts"]
        current_total = sum(layer.num_experts for layer in self._moe_layers)
        if initial <= 0:
            raise ValueError("initial_total_experts must be positive")
        if initial + net_added != current_total:
            raise ValueError(
                "split/merge growth state is inconsistent with the restored model: "
                f"{initial} + {net_added} != {current_total}"
            )
        self._last_step = values["last_step"]
        self._initial_total_experts = initial
        self._net_added_experts = net_added

    def _find_moe_layers(self, module: nn.Module) -> List[SynapticMoE]:
        moes: List[SynapticMoE] = []
        for m in module.modules():
            if isinstance(m, SynapticMoE):
                moes.append(m)
        return moes

    @torch.no_grad()
    def _health(self, layer: SynapticMoE) -> Tensor:
        # Higher is better: H = utilization * energy
        util = layer.fatigue.clamp(0, 1)  # fatigue tracks EMA of utilization
        eng = layer.energy.clamp(0, 1)
        health = util * eng  # [0,1]
        # NeuroScore credit assignment (de5l): blend the per-expert fitness published by
        # NeuroScore.step into health, so Efficiency/Specialization/Resilience — not just
        # raw utilization*energy — drive split/merge/reset selection. Falls back to pure
        # health when disabled or the score is unavailable/mis-shaped.
        if self.cfg.use_neuroscore:
            score = getattr(layer, "last_neuroscore", None)
            if score is not None and tuple(score.shape) == tuple(health.shape):
                w = min(max(float(self.cfg.neuroscore_weight), 0.0), 1.0)
                score = score.to(health.device, health.dtype).clamp(0, 1)
                health = (1.0 - w) * health + w * score
        return health

    @torch.no_grad()
    def _util_weight(self, layer: SynapticMoE, i: int, j: int) -> float:
        if not self.cfg.use_util_weighting:
            return 0.6  # mild bias toward first arg
        # fatigue is utilization proxy
        fat = layer.fatigue
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
        spectral_noise_norms: Optional[List[float]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> bool:
        W = layer.router.weight
        changed_any = False
        for split_idx, (src, dst) in enumerate(zip(sources, slots)):
            if src == dst:
                continue
            changed_any = True
            if self.cfg.function_preserving:
                # Net2Net / firefly: dst becomes a -ln2-gated twin of the parent so the model
                # output does not jump; antisymmetric fc1 noise lets the pair diverge.
                noise_norm = (
                    spectral_noise_norms[split_idx]
                    if spectral_noise_norms is not None
                    else None
                )
                _function_preserving_split_(
                    layer,
                    src,
                    dst,
                    self.cfg,
                    spectral_noise_norm=noise_norm,
                    generator=generator,
                )
            else:
                # Legacy: clone src → dst with noise & embedding tweak (discontinuous).
                _clone_linear_from_(
                    layer.experts[dst].fc1,
                    layer.experts[src].fc1,
                    self.cfg.clone_noise_linear,
                    generator=generator,
                )
                _clone_linear_from_(
                    layer.experts[dst].fc2,
                    layer.experts[src].fc2,
                    self.cfg.clone_noise_linear,
                    generator=generator,
                )
                # router weight row (expert row)
                W[dst].copy_(W[src])
                _add_noise_(W[dst], self.cfg.clone_noise_router, generator=generator)
                # embedding
                layer.router_embeddings[dst : dst + 1].copy_(
                    _orthogonal_perturb_like(
                        layer.router_embeddings[src : src + 1].clone(),
                        self.cfg.clone_noise_embed,
                        generator=generator,
                    )
                )
                # reset stats
                layer.fatigue[dst] = 0.0
                layer.energy[dst] = 1.0
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
        return changed_any

    @torch.no_grad()
    def _do_merges(
        self, layer: SynapticMoE, optimizer: OptimizersArg, step: int
    ) -> bool:
        pairs = self._pick_merge_pairs(layer)
        if self.cfg.verbose and len(pairs) > 0:
            print(f"[SplitMerge] Merging pairs: {pairs}")
        return self._merge_pairs(layer, pairs, optimizer, step)

    @torch.no_grad()
    def _merge_pairs(
        self,
        layer: SynapticMoE,
        pairs: List[Tuple[int, int]],
        optimizer: OptimizersArg,
        step: int,
        balanced: bool = False,
        reuse_loser: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> bool:
        for i, j in pairs:
            # UTA keeps the healthier expert and weights by utilization. The
            # geometry path uses a deterministic 50/50 midpoint to match the OT
            # barycenter certificate rather than leaking health into the ablation.
            if balanced:
                winner, loser = i, j
                alpha = 0.5
            else:
                health = self._health(layer)
                if health[i] >= health[j]:
                    winner, loser = i, j
                else:
                    winner, loser = j, i
                alpha = self._util_weight(layer, winner, loser)
            if self.cfg.function_preserving:
                if reuse_loser:
                    _consolidate_expert_pair_(
                        layer,
                        winner,
                        loser,
                        alpha,
                        self.cfg,
                        ot_barycenter=balanced,
                    )
                else:
                    _function_preserving_merge_(
                        layer,
                        winner,
                        loser,
                        alpha,
                        self.cfg,
                        ot_barycenter=balanced,
                        generator=generator,
                    )
            else:
                _merge_expert_into_and_clone_(layer, winner, loser, alpha, self.cfg)
            # emit lineage event: merge parents (winner,loser) -> child lives at index loser (clone slot reused)
            if self.logger is not None and hasattr(self.logger, "on_merge"):
                try:
                    self.logger.on_merge(
                        layer,
                        parent_i=int(winner),
                        parent_j=int(loser),
                        child_idx=int(winner if reuse_loser else loser),
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
        return bool(pairs)

    @staticmethod
    def _weight_array(tensor: Tensor) -> np.ndarray:
        return tensor.detach().float().cpu().numpy().astype(np.float64, copy=False)

    def _bounded_tensor_array(self, tensor: Tensor) -> np.ndarray:
        """Deterministically sample one tensor without materializing it on CPU in full."""
        flat = tensor.detach().reshape(-1)
        limit = self.cfg.topological_max_samples_per_tensor
        if flat.numel() > limit:
            indices = torch.linspace(0, flat.numel() - 1, limit, device=flat.device).long()
            flat = flat.index_select(0, indices)
        return self._weight_array(flat)

    def _full_tensor_array(self, tensor: Tensor) -> np.ndarray:
        """Materialize a full tensor only after a bounded shortlist selects it."""
        return self._weight_array(tensor.detach().reshape(-1))

    def _expert_weight_samples(self, layer: SynapticMoE, index: int) -> np.ndarray:
        expert = layer.experts[index]
        return np.concatenate(
            [
                self._bounded_tensor_array(expert.fc1.w_slow),
                self._bounded_tensor_array(expert.fc2.w_slow),
            ]
        )

    def _expert_full_weight_samples(self, layer: SynapticMoE, index: int) -> np.ndarray:
        expert = layer.experts[index]
        return np.concatenate(
            [
                self._full_tensor_array(expert.fc1.w_slow),
                self._full_tensor_array(expert.fc2.w_slow),
            ]
        )

    def _expert_function_components(
        self, layer: SynapticMoE, index: int, *, bounded: bool = True
    ) -> Dict[str, np.ndarray]:
        """Sample every inference-relevant parameter/state component independently.

        Per-component comparison prevents a large slow matrix from diluting a
        router, genome, bias, metabolic, or postsynaptic mismatch in one global RMS.
        """
        expert = layer.experts[index]
        components: Dict[str, np.ndarray] = {}
        tensor_array = self._bounded_tensor_array if bounded else self._full_tensor_array
        for prefix, linear in (("fc1", expert.fc1), ("fc2", expert.fc2)):
            components[f"{prefix}.w_slow"] = tensor_array(linear.w_slow)
            if linear.w_fast is not None:
                components[f"{prefix}.w_fast"] = tensor_array(linear.w_fast)
            if linear.bias is not None:
                components[f"{prefix}.bias"] = tensor_array(linear.bias)
            for state_name in ("u_buf", "v_buf"):
                state = getattr(linear, state_name, None)
                if torch.is_tensor(state):
                    components[f"{prefix}.{state_name}"] = tensor_array(state)
            if linear.post is not None:
                for state_name in (
                    "slow",
                    "fast",
                    "U",
                    "V",
                    "camkii",
                    "pp1",
                    "bdnf",
                    "bdnf_hebb_accum",
                    "_last_hebb_delta_mag",
                ):
                    state = getattr(linear.post, state_name, None)
                    if torch.is_tensor(state):
                        components[f"{prefix}.post.{state_name}"] = tensor_array(state)
        items: List[Tuple[str, Tensor]] = [
            ("router.weight", layer.router.weight[index]),
            ("router_embeddings", layer.router_embeddings[index]),
            ("router_logit_bias", layer.router_logit_bias[index : index + 1]),
            ("fatigue", layer.fatigue[index : index + 1]),
            ("energy", layer.energy[index : index + 1]),
        ]
        if layer.Xi is not None:
            items.append(("Xi", layer.Xi[index]))
        for name, tensor in items:
            components[name] = tensor_array(tensor)
        return components

    @staticmethod
    def _max_component_distance(
        left: Dict[str, np.ndarray], right: Dict[str, np.ndarray]
    ) -> float:
        if left.keys() != right.keys():
            return math.inf
        distances: List[float] = []
        for name in left:
            a, b = left[name], right[name]
            if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                return math.inf
            power = 0.5 * (float(np.mean(a * a)) + float(np.mean(b * b)))
            distance = float(np.sqrt(np.mean((a - b) ** 2))) / math.sqrt(
                max(power, 1e-12)
            )
            distances.append(distance)
        return max(distances, default=math.inf)

    def _routing_points(self, layer: SynapticMoE) -> np.ndarray:
        x = layer.last_ctx.get("x")
        if x is None or x.ndim < 2 or x.shape[-1] != layer.router.in_features:
            raise ValueError("missing_routing_points")
        flat = x.reshape(-1, x.shape[-1])
        if flat.shape[0] > self.cfg.topological_max_points:
            rows = torch.linspace(
                0,
                flat.shape[0] - 1,
                self.cfg.topological_max_points,
                device=flat.device,
            ).long()
            flat = flat[rows]
        points = layer.router_probe(flat)
        return self._weight_array(points)

    def _coverage_ordered_split_indices(
        self, layer: SynapticMoE, routing_points: np.ndarray
    ) -> Tuple[List[int], float]:
        """Rank experts by proximity to the least-covered routing point."""
        points = routing_points / np.maximum(
            np.linalg.norm(routing_points, axis=1, keepdims=True), 1e-12
        )
        embeddings = self._weight_array(layer.router_embeddings)
        embeddings = embeddings / np.maximum(
            np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12
        )
        distances = np.linalg.norm(points[:, None, :] - embeddings[None, :, :], axis=-1)
        nearest = distances.min(axis=1)
        target_index = int(np.argmax(nearest))
        target = points[target_index]
        ordered = sorted(
            range(layer.num_experts),
            key=lambda index: (float(np.linalg.norm(embeddings[index] - target)), index),
        )
        return ordered, float(nearest[target_index])

    def _safe_split_candidates(
        self, layer: SynapticMoE, ordered_indices: Iterable[int]
    ) -> List[Tuple[float, int, float, SpectralCertificate]]:
        """Certify a bounded coverage-ranked subset without copying matrices to CPU."""
        candidates: List[Tuple[float, int, float, SpectralCertificate]] = []
        for index in list(ordered_indices)[: self.cfg.topological_max_spectral_candidates]:
            parent = layer.experts[index].fc1.w_slow.detach().float()
            try:
                singular_values = torch.linalg.svdvals(parent)
            except RuntimeError as exc:
                raise ValueError(f"split_spectrum_failed:{index}") from exc
            if singular_values.numel() == 0:
                continue
            sigma_max = float(singular_values[0].item())
            sigma_min = float(singular_values[-1].item())
            requested_noise = max(0.0, float(self.cfg.fp_divergence_noise)) * sigma_max
            well_conditioned = requested_noise < sigma_min
            kappa_parent = sigma_max / sigma_min if sigma_min > 0.0 else math.inf
            kappa_bound = (
                (sigma_max + requested_noise) / (sigma_min - requested_noise)
                if well_conditioned
                else math.inf
            )
            cert = SpectralCertificate(
                sigma_max=sigma_max,
                sigma_min=sigma_min,
                noise_norm=requested_noise,
                kappa_parent=kappa_parent,
                kappa_bound=kappa_bound,
                well_conditioned=well_conditioned,
            )
            if cert.well_conditioned and cert.kappa_bound <= self.cfg.topological_kappa_target:
                candidates.append((cert.kappa_bound, index, requested_noise, cert))
        return sorted(candidates, key=lambda item: (item[0], item[1]))

    def _ot_merge_candidates(
        self, layer: SynapticMoE
    ) -> List[Tuple[float, float, int, int]]:
        """Return a bounded-sample shortlist; actions require exact re-certification."""
        samples = [self._expert_weight_samples(layer, i) for i in range(layer.num_experts)]
        function_components = [
            self._expert_function_components(layer, i) for i in range(layer.num_experts)
        ]
        candidates: List[Tuple[float, float, int, int]] = []
        for i in range(layer.num_experts):
            for j in range(i + 1, layer.num_experts):
                a, b = samples[i], samples[j]
                cert = ot_merge_certificate(a, b)
                if not (cert.comparator_available and cert.transport_optimal):
                    continue
                reference_power = 0.5 * (float(np.mean(a * a)) + float(np.mean(b * b)))
                cost_ratio = cert.transport_cost / max(reference_power, 1e-12)
                functional_distance = self._max_component_distance(
                    function_components[i], function_components[j]
                )
                if functional_distance > self.cfg.topological_functional_distance_max:
                    continue
                candidates.append((cost_ratio, functional_distance, i, j))
        return sorted(candidates, key=lambda item: (item[0], item[1], item[2], item[3]))

    def _exact_ot_merge_candidate(
        self, layer: SynapticMoE, i: int, j: int
    ) -> Optional[
        Tuple[float, float, int, int, np.ndarray, np.ndarray, MergeCertificate]
    ]:
        """Certify one shortlisted pair over every value that a merge will mutate."""
        a = self._expert_full_weight_samples(layer, i)
        b = self._expert_full_weight_samples(layer, j)
        cert = ot_merge_certificate(a, b)
        if not (cert.comparator_available and cert.transport_optimal):
            return None
        reference_power = 0.5 * (float(np.mean(a * a)) + float(np.mean(b * b)))
        cost_ratio = cert.transport_cost / max(reference_power, 1e-12)
        functional_distance = self._max_component_distance(
            self._expert_function_components(layer, i, bounded=False),
            self._expert_function_components(layer, j, bounded=False),
        )
        if functional_distance > self.cfg.topological_functional_distance_max:
            return None
        return cost_ratio, functional_distance, i, j, a, b, cert

    def _fallback_decision(
        self, *, step: int, layer_index: int, reason: str
    ) -> TopologicalLifecycleDecision:
        return TopologicalLifecycleDecision(
            step=step,
            layer_index=layer_index,
            mode="uta_fallback",
            action="uta",
            reason=reason,
        )

    def _plan_topological_lifecycle(
        self, layer: SynapticMoE, *, step: int, layer_index: int
    ) -> Tuple[TopologicalLifecycleDecision, Optional[StructuralGeometryRecord]]:
        try:
            routing_points = self._routing_points(layer)
            if layer.num_experts < 2:
                return self._fallback_decision(
                    step=step,
                    layer_index=layer_index,
                    reason="topological_lifecycle_requires_two_experts",
                ), None
            coverage_order, coverage_distance = self._coverage_ordered_split_indices(
                layer, routing_points
            )
            merge_candidates = self._ot_merge_candidates(layer)

            merge_cost_ratio: float | None = None
            functional_distance: float | None = None
            pair: tuple[int, int] | None = None
            exact_merge = None
            for sampled_candidate in merge_candidates[
                : self.cfg.topological_max_exact_merge_candidates
            ]:
                exact_merge = self._exact_ot_merge_candidate(
                    layer, sampled_candidate[2], sampled_candidate[3]
                )
                if exact_merge is not None:
                    break
            if exact_merge is not None:
                (
                    merge_cost_ratio,
                    functional_distance,
                    merge_i,
                    merge_j,
                    merge_a,
                    merge_b,
                    merge_certificate,
                ) = exact_merge
                pair = (merge_i, merge_j)
            else:
                # H0+spectrum can justify a true birth without any merge candidate.
                # Supply a bounded diagnostic pair to the combined monitor, but do
                # not represent it as a certified/actionable merge in the decision.
                merge_a = self._expert_weight_samples(layer, 0)
                merge_b = self._expert_weight_samples(layer, 1)
                merge_certificate = None

            # A merge+split needs a source outside the merge pair. Preserve the
            # coverage ranking within each partition, but spend the bounded SVD
            # budget on independent experts first when an exact pair is known.
            spectral_order = (
                [index for index in coverage_order if index not in pair]
                + [index for index in coverage_order if index in pair]
                if pair is not None
                else coverage_order
            )
            split_candidates = self._safe_split_candidates(layer, spectral_order)
            if not split_candidates:
                return self._fallback_decision(
                    step=step,
                    layer_index=layer_index,
                    reason="no_spectrally_certified_split",
                ), None
            kappa_bound, source, noise_norm, split_certificate = (
                split_candidates[0]
            )
            record = self.geometry_monitor.record(
                step=step,
                parent_weight=None,
                split_noise_norm=noise_norm,
                routing_points=routing_points,
                merge_a=merge_a,
                merge_b=merge_b,
                split_certificate=split_certificate,
                merge_certificate=merge_certificate,
            )
            if not record.split_well_conditioned or record.kappa_bound is None:
                return self._fallback_decision(
                    step=step,
                    layer_index=layer_index,
                    reason="split_certificate_failed_closed",
                ), record
            if not (
                record.merge_comparator_available
                and record.merge_transport_optimal
            ):
                return self._fallback_decision(
                    step=step,
                    layer_index=layer_index,
                    reason="ot_certificate_failed_closed",
                ), record

            persistence_ratio = record.persistence_ratio
            destination: int | None = None
            persistence_gap = (
                record.persistence_significant
                and coverage_distance >= self.cfg.topological_coverage_distance_threshold
            )
            if persistence_gap:
                can_birth = (
                    self.cfg.variable_expert_count
                    and layer.num_experts < self.cfg.max_experts
                    and self._growth_budget_remaining() > 0
                    and not (dist.is_available() and dist.is_initialized())
                )
                action = "birth" if can_birth else "merge_split"
                reason = "persistent_uncovered_h0_gap"
                if self.cfg.splits_per_call < 1:
                    action = "noop"
                    reason = "persistent_gap_but_split_disabled"
                elif action == "merge_split":
                    if self.cfg.merges_per_call < 1:
                        action = "noop"
                        reason = "persistent_gap_but_merge_disabled"
                    elif pair is None or merge_cost_ratio is None:
                        return self._fallback_decision(
                            step=step,
                            layer_index=layer_index,
                            reason="no_ot_certified_pair_for_merge_split",
                        ), record
                    elif merge_cost_ratio > self.cfg.topological_merge_cost_ratio_max:
                        return self._fallback_decision(
                            step=step,
                            layer_index=layer_index,
                            reason="ot_pair_above_merge_split_cost_ceiling",
                        ), record
                    elif source in pair:
                        return self._fallback_decision(
                            step=step,
                            layer_index=layer_index,
                            reason="no_independent_source_for_merge_split",
                        ), record
                    else:
                        destination = pair[1]
            elif (
                pair is not None
                and merge_cost_ratio is not None
                and
                self.cfg.merges_per_call > 0
                and merge_cost_ratio <= self.cfg.topological_merge_cost_ratio_max
            ):
                action = "merge"
                reason = "ot_nearest_pair_below_cost_threshold"
            else:
                action = "noop"
                reason = (
                    "persistent_h0_gap_already_covered"
                    if record.persistence_significant
                    else "no_significant_gap_or_low_cost_merge"
                )

            return TopologicalLifecycleDecision(
                step=step,
                layer_index=layer_index,
                mode="topological",
                action=action,
                reason=reason,
                split_source=(source if action in ("birth", "merge_split") else None),
                split_destination=(layer.num_experts if action == "birth" else destination),
                merge_pair=(pair if action in ("merge", "merge_split") else None),
                split_noise_norm=(
                    noise_norm if action in ("birth", "merge_split") else None
                ),
                kappa_bound=kappa_bound,
                persistence_ratio=persistence_ratio,
                merge_cost_ratio=merge_cost_ratio,
                functional_distance=functional_distance,
                coverage_distance=coverage_distance,
            ), record
        except (ValueError, np.linalg.LinAlgError) as exc:
            return self._fallback_decision(
                step=step,
                layer_index=layer_index,
                reason=str(exc) or type(exc).__name__,
            ), None

    def _log_topological_decision(
        self,
        decision: TopologicalLifecycleDecision,
        record: Optional[StructuralGeometryRecord],
    ) -> None:
        self.topological_decisions.append(decision)
        if self.event_logger is None or not hasattr(self.event_logger, "event"):
            return
        fields: Dict[str, Any] = {"decision": asdict(decision)}
        if record is not None:
            fields["certificates"] = asdict(record)
        try:
            self.event_logger.event("topological_nas", step=decision.step, **fields)
        except Exception as exc:
            if self.cfg.verbose:
                print(f"[SplitMerge] topological event logging failed: {exc}")

    @torch.no_grad()
    def _run_topological_layer(
        self,
        layer: SynapticMoE,
        decision: TopologicalLifecycleDecision,
        optimizer: OptimizersArg,
    ) -> bool:
        if decision.action == "merge" and decision.merge_pair is not None:
            return self._merge_pairs(
                layer,
                [decision.merge_pair],
                optimizer,
                decision.step,
                balanced=True,
            )
        if decision.action not in ("merge_split", "birth"):
            return False
        if decision.split_source is None or decision.split_noise_norm is None:
            raise RuntimeError("topological split decision lacks a source or noise certificate")

        destination = decision.split_destination
        if decision.action == "merge_split":
            if decision.merge_pair is None:
                raise RuntimeError("topological merge_split decision lacks an OT pair")
            self._merge_pairs(
                layer,
                [decision.merge_pair],
                optimizer,
                decision.step,
                balanced=True,
                reuse_loser=True,
            )
        if decision.action == "birth":
            touched = _resize_layer_experts_(
                layer,
                target_E=layer.num_experts + 1,
                seed_idx=decision.split_source,
                cfg=self.cfg,
            )
            if not touched:
                raise RuntimeError("topological birth did not create an expert slot")
            destination = touched[0]
            if decision.split_destination != destination:
                raise RuntimeError("topological birth destination drifted from its decision record")
            self._net_added_experts += 1
        if destination is None:
            raise RuntimeError("topological split decision lacks a destination")
        changed = self._split_into_slots(
            layer,
            [decision.split_source],
            [destination],
            optimizer,
            decision.step,
            spectral_noise_norms=[decision.split_noise_norm],
        )
        if decision.action == "birth" and self.logger is not None and hasattr(self.logger, "on_spawn"):
            try:
                self.logger.on_spawn(
                    layer,
                    parent_idx=int(decision.split_source),
                    children=[destination],
                    step=decision.step,
                )
            except Exception as _e:
                if self.cfg.verbose:
                    print(f"[SplitMerge] logger.on_spawn failed: {_e}")
        return changed

    def _plan_uta_layer(
        self, layer: SynapticMoE, step: int, layer_index: int
    ) -> List[Dict[str, Any]]:
        """Compute this round's UTA lifecycle ops WITHOUT mutating anything (uta.5).

        Rank0 runs this; the resulting JSON-safe dicts are broadcast to every rank,
        which applies them bit-identically via :meth:`_apply_uta_ops`. Alphas are
        computed here from rank0's (authoritative) health and carried inside the ops
        so no rank ever re-derives a decision from stale replicas.
        """

        def _seed(kind: str, counter: List[int]) -> int:
            digest = hashlib.blake2b(
                f"{step}:{layer_index}:{kind}:{counter[0]}".encode(),
                digest_size=8,
            ).digest()
            counter[0] += 1
            return int.from_bytes(digest, "big")

        counter = [0]
        ops: List[Dict[str, Any]] = []
        for i, j in self._pick_merge_pairs(layer):
            health = self._health(layer)
            winner, loser = (i, j) if health[i] >= health[j] else (j, i)
            alpha = self._util_weight(layer, winner, loser)
            ops.append(
                {
                    "kind": "merge",
                    "winner": int(winner),
                    "loser": int(loser),
                    "alpha": float(alpha),
                    "seed": _seed("merge", counter),
                }
            )
        sources = self._pick_split_sources(layer)
        if sources and self.cfg.splits_per_call > 0:
            slots = self._weakest_slots(
                layer, min(len(sources), self.cfg.splits_per_call)
            )
            for src, dst in zip(sources, slots):
                if src != dst:
                    ops.append(
                        {"kind": "split", "src": int(src), "dst": int(dst), "seed": _seed("split", counter)}
                    )
        dead_slots = self._pick_dead_slots(layer)
        if dead_slots:
            reset_sources = self._pick_reset_sources(layer, max(len(dead_slots), 1))
            for slot in dead_slots:
                src = next((c for c in reset_sources if c != slot), None)
                if src is not None:
                    ops.append(
                        {"kind": "split", "src": int(src), "dst": int(slot), "seed": _seed("reset", counter)}
                    )
        if self.cfg.variable_expert_count:
            ops.extend(self._plan_resize_layer(layer, _seed, counter))
        return ops

    @torch.no_grad()
    def _apply_uta_ops(
        self,
        layer: SynapticMoE,
        ops: List[Dict[str, Any]],
        optimizer: OptimizersArg,
        step: int,
    ) -> bool:
        """Execute planned ops deterministically on EVERY rank (uta.5).

        Each op carries its own RNG seed; the executor derives an isolated
        generator per op, so noise draws are identical across ranks regardless of
        their local global-RNG state. Lineage logging fires on rank0 only.
        """
        changed = False
        dev = layer.router.weight.device
        for op in ops:
            gen = torch.Generator(device=dev)
            gen.manual_seed(int(op["seed"]) % (2**63 - 1))
            kind = op["kind"]
            if kind == "merge":
                changed |= self._merge_pairs(
                    layer,
                    [(int(op["winner"]), int(op["loser"]))],
                    optimizer,
                    step,
                    generator=gen,
                )
            elif kind == "split":
                changed |= self._split_into_slots(
                    layer,
                    [int(op["src"])],
                    [int(op["dst"])],
                    optimizer,
                    step,
                    generator=gen,
                )
            elif kind == "grow":
                touched = _resize_layer_experts_(
                    layer,
                    target_E=int(op["target_E"]),
                    seed_idx=int(op["seed_idx"]),
                    cfg=self.cfg,
                    generator=gen,
                )
                if touched:
                    self._net_added_experts += len(touched)
                    changed = True
                    sources = [int(op["seed_idx"])] * len(touched)
                    self._split_into_slots(
                        layer, sources, touched, optimizer, step, generator=gen
                    )
                    if _is_rank0() and self.logger is not None and hasattr(self.logger, "on_spawn"):
                        try:
                            self.logger.on_spawn(
                                layer,
                                parent_idx=int(op["seed_idx"]),
                                children=touched,
                                step=step,
                            )
                        except Exception as _e:
                            if self.cfg.verbose:
                                print(f"[SplitMerge] logger.on_spawn failed: {_e}")
            elif kind == "shrink":
                victims = [int(v) for v in op["victims"]]
                keeper = int(op["keeper"])
                for v in victims:
                    _fold_expert_into_(layer, victim_idx=v, keeper_idx=keeper, alpha=float(op.get("alpha", 0.5)))
                for _v in sorted(victims, reverse=True):
                    _resize_layer_experts_(
                        layer,
                        target_E=int(getattr(layer, "num_experts")) - 1,
                        seed_idx=keeper,
                        cfg=self.cfg,
                        generator=gen,
                    )
                self._net_added_experts -= len(victims)
                changed = True
                if _is_rank0() and self.logger is not None and hasattr(self.logger, "on_death"):
                    try:
                        self.logger.on_death(layer, removed=victims, keeper=keeper, step=step)
                    except Exception as _e:
                        if self.cfg.verbose:
                            print(f"[SplitMerge] logger.on_death failed: {_e}")
            else:
                raise ValueError(f"[SplitMerge] unknown lifecycle op kind: {kind}")
        return changed

    def _run_uta_layer(
        self, layer: SynapticMoE, optimizer: OptimizersArg, step: int
    ) -> bool:
        ops = self._plan_uta_layer(layer, step, layer_index=0)
        return self._apply_uta_ops(layer, ops, optimizer, step)

    # ------------------------------------------------------------------
    # uta.4: variable expert count
    # ------------------------------------------------------------------

    def _growth_budget_remaining(self) -> int:
        cap = int(self._initial_total_experts * self.cfg.growth_budget_pct)
        return max(0, cap - max(0, self._net_added_experts))

    def _plan_resize_layer(
        self,
        layer: SynapticMoE,
        seed_fn,
        counter: List[int],
    ) -> List[Dict[str, Any]]:
        """Decision half of uta.4 resize — no mutation (uta.5)."""
        E = int(layer.num_experts)
        health = self._health(layer)
        dead = [
            i for i in range(E) if float(health[i]) <= self.cfg.reset_health_max
        ]
        # recycle-before-grow: reclaimable dead slots must be serviced before
        # fresh capacity is added.
        dead_surplus = len(dead) - self.cfg.resets_per_call

        strong = [
            i for i in range(E) if float(health[i]) >= self.cfg.split_health_min
        ]
        demand_surplus = len(strong) - self.cfg.splits_per_call
        cap_room = self.cfg.max_experts - E
        budget_room = min(cap_room, self._growth_budget_remaining())
        ops: List[Dict[str, Any]] = []
        if demand_surplus > 0 and budget_room > 0 and dead_surplus <= 0:
            n_add = min(demand_surplus, budget_room)
            seed_idx = int(max(strong, key=lambda i: float(health[i])))
            ops.append(
                {
                    "kind": "grow",
                    "target_E": E + n_add,
                    "seed_idx": seed_idx,
                    "seed": seed_fn("grow", counter),
                }
            )
            return ops  # one resize per call keeps surgery auditable

        removable = len(dead) - self.cfg.resets_per_call
        floor_room = E - self.cfg.min_experts
        if removable > 0 and floor_room > 0:
            n_drop = min(removable, floor_room)
            victims = sorted(dead, key=lambda i: float(health[i]))[:n_drop]
            keeper = int(
                max(
                    (i for i in range(E) if i not in victims),
                    key=lambda i: float(health[i]),
                )
            )
            ops.append(
                {
                    "kind": "shrink",
                    "victims": [int(v) for v in victims],
                    "keeper": keeper,
                    "alpha": 0.5,
                    "seed": seed_fn("shrink", counter),
                }
            )
        return ops

    @torch.no_grad()
    def _maybe_resize_layer(
        self, layer: SynapticMoE, optimizer: OptimizersArg, step: int
    ) -> bool:
        """Plan + immediately apply a resize decision for one layer.

        Single-process convenience API (tests, non-distributed runs); the
        distributed path plans on rank0 via :meth:`_plan_uta_layer` and applies
        the broadcast ops everywhere instead of calling this directly.
        """
        counter = [0]

        def _seed(kind: str, cnt: List[int]) -> int:
            digest = hashlib.blake2b(
                f"{step}:{id(layer) % 100003}:{kind}:{cnt[0]}".encode(),
                digest_size=8,
            ).digest()
            cnt[0] += 1
            return int.from_bytes(digest, "big")

        ops = self._plan_resize_layer(layer, _seed, counter)
        return self._apply_uta_ops(layer, ops, optimizer, step)

    @torch.no_grad()
    def step(self, global_step: int, optimizer: OptimizersArg = None):
        if not self.cfg.enabled:
            return
        if global_step < self.cfg.warmup_steps:
            return
        if global_step - self._last_step < self.cfg.min_step_interval:
            return

        distributed = dist.is_available() and dist.is_initialized()
        rank0 = _is_rank0()
        if rank0 and self.cfg.verbose:
            print(f"[SplitMerge] step @ {global_step}")

        if distributed and not self.topological_nas:
            # ------------------------------------------------------------
            # uta.5 protocol: decide on rank0, broadcast the DECISION, apply
            # identical deterministic surgery on EVERY rank, then re-sync each
            # rank's optimizer param-groups locally. Survivors keep their own
            # moments; no blanket moment-zeroing is needed because no rank is
            # a passive observer anymore.
            # ------------------------------------------------------------
            plans: List[Dict[str, Any]] = []
            if rank0:
                for layer_index, layer in enumerate(self._moe_layers):
                    plans.append(
                        {
                            "layer": layer_index,
                            "ops": self._plan_uta_layer(layer, global_step, layer_index),
                            "experts_before": int(layer.num_experts),
                        }
                    )
            handle = [plans]
            dist.broadcast_object_list(handle, src=0)
            plans = handle[0]

            for entry in plans:
                layer = self._moe_layers[entry["layer"]]
                experts_before = int(layer.num_experts)
                layout = (
                    capture_optimizer_layout(optimizer, self.model)
                    if optimizer is not None and self.cfg.variable_expert_count
                    else None
                )
                optimizer_snapshot = (
                    snapshot_optimizer_state(optimizer)
                    if optimizer is not None and self.cfg.variable_expert_count
                    else None
                )
                changed = self._apply_uta_ops(
                    layer, entry["ops"], optimizer, global_step
                )
                if (
                    int(layer.num_experts) != experts_before
                    and optimizer is not None
                    and layout is not None
                    and optimizer_snapshot is not None
                ):
                    synchronize_optimizers_with_model(
                        optimizer,
                        self.model,
                        layout,
                        optimizer_snapshot,
                    )
            if self.cfg.ddp_broadcast:
                for layer in self._moe_layers:
                    _broadcast_module_params(layer)
                dist.barrier()
            self._last_step = global_step
            return

        # Single-process path (and, for now, the topological path under DDP,
        # which keeps the legacy rank0-executes flow): perform operations
        # layer-by-layer on rank 0.
        changed_layers = [False] * len(self._moe_layers)
        for layer_index, layer in enumerate(self._moe_layers if rank0 else []):
            experts_before = int(layer.num_experts)
            layout = (
                capture_optimizer_layout(optimizer, self.model)
                if optimizer is not None and self.cfg.variable_expert_count
                else None
            )
            optimizer_snapshot = (
                snapshot_optimizer_state(optimizer)
                if optimizer is not None and self.cfg.variable_expert_count
                else None
            )
            if not self.topological_nas:
                ops = self._plan_uta_layer(layer, global_step, layer_index)
                changed = self._apply_uta_ops(layer, ops, optimizer, global_step)
            else:
                decision, record = self._plan_topological_lifecycle(
                    layer,
                    step=global_step,
                    layer_index=layer_index,
                )
                if decision.mode == "uta_fallback":
                    ops = self._plan_uta_layer(layer, global_step, layer_index)
                    changed = self._apply_uta_ops(layer, ops, optimizer, global_step)
                else:
                    changed = self._run_topological_layer(layer, decision, optimizer)
                self._log_topological_decision(decision, record)
            changed_layers[layer_index] = changed
            if (
                int(layer.num_experts) != experts_before
                and optimizer is not None
                and layout is not None
                and optimizer_snapshot is not None
            ):
                synchronize_optimizers_with_model(
                    optimizer,
                    self.model,
                    layout,
                    optimizer_snapshot,
                )

        # Every rank must participate in the same broadcast sequence. Previously
        # non-zero ranks entered the final barrier and returned before rank 0's
        # broadcasts, mismatching collectives and deadlocking DDP lifecycle steps.
        if self.cfg.ddp_broadcast and distributed:
            if self._moe_layers:
                flag_device = self._moe_layers[0].router.weight.device
                change_flags = torch.tensor(
                    changed_layers,
                    dtype=torch.int64,
                    device=flag_device,
                )
                dist.broadcast(change_flags, src=0)
                changed_layers = [bool(value) for value in change_flags.tolist()]
            for layer in self._moe_layers:
                _broadcast_module_params(layer)
            # Rank 0 resets touched moments during surgery in this legacy flow.
            # Other ranks did not execute the surgery, so reset every parameter
            # in each changed MoE on every rank to keep distributed optimizer
            # state semantically aligned (uta.5's protocol replaces this for
            # the UTA path by executing everywhere instead).
            if optimizer is not None:
                for changed, layer in zip(changed_layers, self._moe_layers):
                    if changed:
                        _zero_optim_moments_for(optimizer, list(layer.parameters()))
            dist.barrier()

        self._last_step = global_step
