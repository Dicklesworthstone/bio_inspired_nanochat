# nanochat/synaptic.py
# Comprehensive synaptic modules for nanochat:
# - Presynaptic biophysics → attention logit augmentation
# - Postsynaptic dual-timescale linear with low-rank eligibility
# - Synaptic Self-Attention (RoPE, MQA-compatible)
# - Synaptic MLP
# - Synaptic MoE with router embeddings, contrastive updates & structural hooks
# - Structural plasticity utilities
#
# Design highlights (mapped from the JAX reference you provided):
#   • Synaptotagmin-1/7 mixed Ca2+ sensor, complexin clamp
#   • Munc13/18 priming, clathrin/dynamin endocytosis (delay queue)
#   • V-ATPase/VDAC energy coupling and per-edge cost model
#   • EMA normalization of quantal gain; optional stochastic release
#   • PSD-like low-rank eligibility U/V with CaMKII/PP1 gating (fast/slow)
#   • Septin-like distance barrier in attention logits
#   • Router embeddings + contrastive update; MoE top-k dispatch with fatigue
#
# This file is intentionally verbose and highly instrumented for clarity.

import contextlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List, Dict, Literal, cast, Any

from bio_inspired_nanochat.torch_imports import torch, nn, F, Tensor
from bio_inspired_nanochat.common import decouple_config
from bio_inspired_nanochat.glial_homeostasis import GlialHomeostasis
from bio_inspired_nanochat.metriplectic_integrator import TorchStepRecord, torch_guarded_step

try:
    from .flex_synaptic import SynapticFlexAttention
    _HAS_FLEX = True
except ImportError:
    _HAS_FLEX = False


class SynapticGranularity(str, Enum):
    """Architectural granularity of synaptic state machines across the network (bead vap.2).

    - PER_CONNECTION (Fine / L1): Every attention edge / projection connection has dedicated
      presynaptic and postsynaptic state machines (faithful GPT-5 Pro blueprint).
    - PER_NEURON (Medium / L2): Intermediate per-neuron rank-R eligibility traces.
    - PER_EXPERT (Coarse / L3): Pooled per-expert / per-layer state machine (Grok blueprint).
    """
    PER_CONNECTION = "per_connection"
    PER_NEURON = "per_neuron"
    PER_EXPERT = "per_expert"


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def _rmsnorm(x: Tensor, eps=1e-6) -> Tensor:
    return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * x


def _tri(T: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.tril(torch.ones(T, T, device=device, dtype=dtype)).view(1, 1, T, T)


def _softplus(x: Tensor, beta=1.0) -> Tensor:
    return (1.0 / beta) * F.softplus(beta * x)


def _cosine(u: Tensor, v: Tensor, eps=1e-8) -> Tensor:
    """Cosine similarity with safe normalization."""
    u = u / (u.norm(dim=-1, keepdim=True) + eps)
    v = v / (v.norm(dim=-1, keepdim=True) + eps)
    return (u * v).sum(dim=-1)

def _sample_binomial_counts(
    probs: Tensor,
    total_count: Tensor,
    *,
    max_count: int,
    tau: float,
    mode: Literal["gumbel_sigmoid_ste", "straight_through", "normal_reparam"],
    eps: float = 1e-6,
    generator: Optional[torch.Generator] = None,
    normal_draw_width: Optional[int] = None,
) -> Tensor:
    """Sample Binomial(total_count, probs) counts with a cheap, GPU-friendly estimator.

    Notes:
    - We cap `total_count` to `max_count` and round to the nearest int to keep sampling fast.
    - `mode="gumbel_sigmoid_ste"` uses a straight-through Gumbel-Sigmoid relaxation so gradients
      flow (approximately) through `probs` during training.
    - `mode="straight_through"` uses a simpler STE (hard Bernoulli forward, `probs` backward).
    - `mode="normal_reparam"` uses a Gaussian approximation with reparameterization (fastest).
    """
    if max_count <= 0:
        return torch.zeros_like(probs)

    probs_32 = probs.to(torch.float32)

    if mode == "normal_reparam":
        count_f32 = torch.clamp(total_count.round(), 0.0, float(max_count)).to(torch.float32)
        p = probs_32.clamp(eps, 1.0 - eps)
        mean = count_f32 * p
        var = count_f32 * p * (1.0 - p)
        std = torch.sqrt(var + eps)
        draw_shape = mean.shape
        if normal_draw_width is not None:
            if normal_draw_width < mean.size(-1):
                raise ValueError(
                    "normal_draw_width must cover the probability tensor's final dimension; "
                    f"got width={normal_draw_width}, probabilities={mean.size(-1)}"
                )
            draw_shape = (*mean.shape[:-1], normal_draw_width)
        noise = torch.randn(
            draw_shape, device=mean.device, dtype=mean.dtype, generator=generator
        )[..., : mean.size(-1)]
        samp = mean + std * noise
        samp = samp.clamp(min=0.0)
        # Clamp high-end based on per-entry count.
        samp = torch.minimum(samp, count_f32)
        return samp.to(probs.dtype)

    if mode == "gumbel_sigmoid_ste" and tau <= 0:
        raise ValueError(f"tau must be > 0 for gumbel_sigmoid_ste, got {tau}")

    count_i64 = torch.clamp(total_count.round(), 0, float(max_count)).to(torch.int64)

    # Trial mask for variable total_count.
    trial_idx = torch.arange(max_count, device=probs.device).view(
        (1,) * probs.ndim + (max_count,)
    )
    trial_mask = trial_idx < count_i64.unsqueeze(-1)

    # One uniform per Bernoulli trial.
    u = torch.rand(
        (*probs_32.shape, max_count),
        device=probs.device,
        dtype=torch.float32,
        generator=generator,
    )
    u = u.clamp(min=eps, max=1.0 - eps)

    if mode == "gumbel_sigmoid_ste":
        # Logistic noise (equivalent to Gumbel(0,1)-Gumbel(0,1)).
        noise = torch.log(u) - torch.log1p(-u)
        logits = torch.logit(probs_32.clamp(eps, 1.0 - eps), eps=eps)
        y_soft = torch.sigmoid((logits.unsqueeze(-1) + noise) / float(tau))
        y_hard = (y_soft > 0.5).to(torch.float32)
        y_soft = y_soft * trial_mask.to(torch.float32)
        y_hard = y_hard * trial_mask.to(torch.float32)
        y = (y_hard - y_soft).detach() + y_soft
        return y.sum(dim=-1).to(probs.dtype)

    if mode == "straight_through":
        p = probs_32.clamp(eps, 1.0 - eps).unsqueeze(-1)
        y_hard = (u < p).to(torch.float32)
        y_soft = p.expand_as(y_hard)
        y_hard = y_hard * trial_mask.to(torch.float32)
        y_soft = y_soft * trial_mask.to(torch.float32)
        y = (y_hard - y_soft).detach() + y_soft
        return y.sum(dim=-1).to(probs.dtype)

    raise ValueError(f"Unknown mode: {mode!r}")


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class SynapticConfig:
    # General
    granularity: Literal["per_connection", "per_neuron", "per_expert"] | SynapticGranularity = (
        SynapticGranularity.PER_CONNECTION
    )
    rank_eligibility: int = 8
    attn_topk: int = 32
    stochastic_train_frac: float = 0.12
    stochastic_mode: Literal["gumbel_sigmoid_ste", "straight_through", "normal_reparam"] = (
        "normal_reparam"
    )
    stochastic_tau: float = 1.0
    stochastic_count_cap: int = 8

    # Presynaptic Biophysics
    # tau_c is a calcium-decay TIME CONSTANT: retention per step is exp(-1/tau_c), used uniformly
    # by release(), release_canonical, forward(), and the dashboard. Default 6.0 => retention
    # exp(-1/6)=0.846, a ~4-step calcium half-life that gives meaningful short-term plasticity.
    # (The legacy 0.85 was a RAW retention multiplier; under the unified exp form that would be a
    # ~0.6-step half-life and near-inert plasticity. 8j9.2/x6z4; confirm final value via 4fw.)
    tau_c: float = 6.0
    # yw9.3: promote the calcium/buffer kinetics (tau_c, tau_buf, alpha_ca, alpha_buf_on/off) from
    # fixed hyperparameters to SGD-LEARNED Parameters (LearnableKinetics), reached through the
    # differentiable recurrence (yw9.2). Stability-preserving by construction (sigmoid decays,
    # softplus gains, bounded buffer coupling) so learning can't destabilize. DEFAULT-OFF: the
    # cfg constants below are used unchanged unless this is set.
    learnable_kinetics: bool = False
    # hwxb.4.6: WIRE the differentiable recurrence (yw9.2) into the live attention forward so the
    # learnable kinetics above actually receive gradient in a real run. DEFAULT-OFF: the live path
    # stays the detached single-call snapshot (byte-identical to today). When ON, the attention
    # presyn bias is computed by advancing the presyn state CAUSALLY over query CHUNKS with
    # autograd, so the decay kinetics — which see gradient only once state accumulates across steps
    # (a single fresh-state call only trains alpha_ca) — become learnable. This also makes the
    # training forward consistent with the inherently-causal KV-cache decode path (a query attends
    # to a key whose vesicles were already drained by earlier queries). Requires learnable_kinetics
    # to be useful (the validator flags it otherwise). Applies to the standard (non-flex) full-
    # sequence training/prefill path; decode (prefix KV cache) is already per-step causal.
    differentiable_recurrence: bool = False
    # Query positions per recurrence GROUP for the causal recurrence above. Every query still
    # advances the state separately; grouping only controls graph/checkpoint orchestration. 64
    # amortizes Python bookkeeping while preserving token-for-token decode semantics.
    recurrence_block_size: int = 64
    # Truncated-BPTT detach interval, in recurrence STEPS (blocks), for the differentiable
    # recurrence. Bounds activation memory to ~chunk_len blocks regardless of sequence length.
    # 0 => full BPTT across the whole sequence (no truncation).
    recurrence_chunk_len: int = 0
    # Full-BPTT activation-checkpoint window, in recurrence STEPS. Unlike recurrence_chunk_len,
    # this preserves gradients across every window boundary by replaying each window in backward.
    # 0 => ordinary eager full BPTT. Mutually exclusive with recurrence_chunk_len because one is
    # exact-gradient recomputation while the other deliberately truncates the graph.
    recurrence_checkpoint_len: int = 0
    # (or4t: alpha_c / syt1_slope / syt7_slope / cpx_thresh were the LEGACY sigmoid release-prob
    #  params; removed as dead after the canonical migration. The canonical uses alpha_ca for the
    #  calcium influx and Hill syt_fast_kd/syt_slow_kd + complexin_bias for the release prob.)
    doc2_gain: float = 0.08
    prime_rate: float = 0.075
    unprime_per_release: float = 0.05
    nsf_recover: float = 0.08
    rec_rate: float = 0.06
    endo_delay: int = 3
    # (or4t: amp_load / amp_leak removed -- the canonical uses energy->qamp, not the AMP-update
    #  dynamics. The AMP state itself is still initialized (init_amp) and carried, just frozen.)

    # Initial States
    init_rrp: float = 6.0
    init_reserve: float = 18.0
    init_snare: float = 0.7
    init_clamp: float = 0.6
    init_amp: float = 1.0
    init_energy: float = 0.85

    # Energy Dynamics
    energy_fill: float = 0.02
    energy_use: float = 0.02
    energy_max: float = 1.0

    # Attention
    lambda_loge: float = 1.0
    barrier_strength: float = 0.1
    epsilon: float = 1e-6
    # Max absolute value of the per-edge log-release attention bias
    # (lambda_loge * log(epsilon + release)). The normalized release can spike, so
    # without a clamp a single edge's bias can dominate the softmax and destabilize
    # attention. 10.0 keeps the mechanism intact while bounding it; 0 disables (vg9.5).
    loge_bias_clamp: float = 10.0

    # Full-sequence visualization/reference dynamics
    tau_buf: float = 4.0
    tau_prime: float = 5.0
    tau_rrp: float = 40.0
    tau_energy: float = 50.0
    alpha_ca: float = 0.55
    alpha_buf_on: float = 0.1
    alpha_buf_off: float = 0.1
    alpha_prime: float = 0.1
    alpha_unprime: float = 0.1
    alpha_refill: float = 0.1
    energy_in: float = 0.01
    energy_cost_rel: float = 0.015
    energy_cost_pump: float = 0.01
    syt_fast_kd: float = 0.4
    syt_slow_kd: float = 1.0
    complexin_bias: float = 0.0
    qmax: float = 2.0
    q_beta: float = 1.0

    # Postsynaptic Plasticity
    post_fast_decay: float = 0.95
    post_fast_lr: float = 1.5e-3
    post_slow_lr: float = 5e-4
    post_trace_decay: float = 0.96
    # sax.1/jpqc — online fast-weight ("fast-weight programmer") magnitude/stability. The raw
    # rank-R Hebbian delta is O(trace²): normally negligible, but it can enter positive feedback
    # through y_fast and explode on sparse MoE trajectories. The default normalized path steps
    # w_fast/w_slow along a unit-norm direction and caps ||w_fast||, making the update impactful
    # and stable. Set False only for the documented legacy/negative-control ablation; the raw path
    # is intentionally not the production default after it diverged across all four jpqc seeds.
    # This remains the substrate for a predictive consolidation signal (e.g. three-factor
    # reward-modulated Hebbian, hy8.2); normalization alone does not make the write task-aware.
    fast_weight_normalized: bool = True
    fast_weight_eta: float = 0.5       # O(1) step size on the normalized fast-weight direction
    fast_weight_max_norm: float = 1.0  # Frobenius-norm cap on w_fast (<=0 disables the cap)
    camkii_up: float = 0.05
    pp1_tau: float = 0.985
    camkii_thr: float = 1.0
    pp1_thr: float = 0.7
    bdnf_tau: float = 0.985
    bdnf_scale: float = 1.0
    bdnf_gamma: float = 0.0  # Gamma gain factor; when > 0, takes precedence over bdnf_scale
    bdnf_hebb_accumulate: bool = True  # Use Hebbian delta magnitude for BDNF (vs CaMKII)
    bdnf_max: float = 10.0  # Upper clamp on BDNF to prevent unbounded growth

    # Spike-Timing-Dependent Plasticity (STDP) over sequence/time axis (sax.3)
    enable_stdp: bool = False
    stdp_a_plus: float = 0.01        # LTP amplitude (pre before post)
    stdp_a_minus: float = 0.012      # LTD amplitude (post before pre)
    stdp_tau_plus: float = 20.0      # LTP time constant
    stdp_tau_minus: float = 20.0     # LTD time constant

    # Bistable CaMKII/PP1 consolidation latch (sax.2). DEFAULT-OFF. When enabled, the
    # CaMKII/PP1 update becomes a Lisman-style bistable switch — CaMKII self-excitation
    # (Hill autophosphorylation) + mutual cross-inhibition with PP1, over a basal
    # phosphatase floor — and the consolidation gate becomes sigmoid(gate_beta*(CaMKII-PP1)),
    # putting PP1 INTO the gate. This yields hysteresis: a supra-threshold pulse latches the
    # synapse ON and it STAYS after input drops (noise-robust retention), while a sustained
    # LTD/low-calcium signal flips it OFF. Calcium maps to LTP/LTD drives by a BCM curve:
    # drive = sigmoid(gain*(ca - camkii_thr)); erase = sigmoid(gain*(latch_ltd_thr - ca)).
    # The disabled path keeps the legacy linear CaMKII update + sigmoid(CaMKII-0.5) gate.
    bistable_latch: bool = False
    latch_ltd_thr: float = 0.5       # calcium below this activates PP1/LTD (camkii_thr is the LTP threshold)
    latch_input_gain: float = 12.0   # sharpness of the BCM drive/erase sigmoids
    latch_alpha_ca: float = 0.6      # calcium -> CaMKII potentiation rate
    latch_beta_pp1: float = 1.0      # PP1 -> CaMKII de-potentiation (cross-inhibition)
    latch_gamma_auto: float = 0.45   # CaMKII autophosphorylation (self-excitation) gain
    latch_hill_n: float = 6.0        # Hill coefficient of the self-excitation
    latch_hill_k: float = 0.6        # Hill half-max of the self-excitation
    latch_alpha_pp1: float = 0.5     # low-calcium -> PP1 activation rate
    latch_beta_camkii: float = 0.3   # CaMKII -> PP1 cross-inhibition (maintains the latch)
    latch_pp1_basal: float = 0.3     # basal phosphatase floor (keeps the OFF state stable)
    latch_gate_beta: float = 6.0     # consolidation-gate steepness, sigmoid(beta*(CaMKII-PP1))
    # 0642.2.2.3: enable the runtime RETENTION CERTIFICATE for the bistable latch — the closed-form
    # cusp hysteresis half-width delta* (docs/theory/singular_perturbation.md §4) computed by
    # bio_inspired_nanochat/cusp_certificate.py, gated by an epsilon (normal-hyperbolicity) check.
    # DEFAULT-OFF observability/gate: requires bistable_latch; when the timescale separation is
    # insufficient (rho_fast > cusp_eps_max) or the latch is monostable, the certificate is dropped
    # and the model falls back to the heuristic sax.2 latch (no retention claim).
    cusp_latch: bool = False
    cusp_eps_max: float = 0.98       # max fast-subsystem spectral radius rho(M_cb) that still certifies
    # 0642.1.2.4: use the structure-preserving discrete-gradient integrator
    # (bio_inspired_nanochat/metriplectic_integrator.py) for the calcium/buffer subsystem instead of
    # the clamped-Euler step, so energy is conserved + free energy is Lyapunov at the DISCRETE level
    # (docs/theory/metriplectic.md, 0642.1.1/0642.1.2.1). DEFAULT-OFF; the guard layer reverts any
    # breaching step to the clamped-Euler baseline (0642.1.2.3).
    metriplectic_integrator: bool = False
    # vg9.2: run online Hebbian plasticity during TRAINING (grad enabled), not only under
    # inference/no_grad. The headline "online Hebbian learning" was previously gated behind
    # `not torch.is_grad_enabled()` and so NEVER ran at train time. When True (default), the
    # detached fast-adaptation update executes during training; the in-place Parameter writes
    # are deferred to the top of the next forward so they cannot corrupt the live autograd
    # graph. Set False to restore the legacy inference-only behavior.
    plasticity_during_training: bool = True

    # Structural Plasticity (MoE)
    structural_interval: int = 50000
    structural_tau_util: float = 0.2
    structural_age_bias: float = 1.0
    router_embed_dim: int = 24
    router_contrastive_lr: float = 1e-4
    router_contrastive_push: float = 0.1
    # hy8.4: slow tripartite-synapse feedback over expert activity and pooled
    # energy. Default-off preserves the existing router exactly. When enabled,
    # the controller integrates a bounded, zero-sum logit correction that
    # suppresses persistent winners and recruits underused experts.
    glial_homeostasis: bool = False
    glial_group_size: int = 4
    glial_ema_rate: float = 0.05
    glial_feedback_rate: float = 0.05
    glial_energy_weight: float = 0.25
    glial_bias_cap: float = 4.0
    # 0642.5.2.2: replace utilization/health lifecycle decisions with bounded
    # spectral, H0-persistence, and optimal-transport certificates. Default-off;
    # SplitMergeController falls back to the UTA health-threshold path whenever
    # the live routing/weight evidence is absent or uncertified.
    topological_nas: bool = False

    # Genetics
    # Per-expert genome embedding (Xi). A shared learned decoder maps this low-dimensional
    # latent to a larger set of bounded kinetics. Set to 0 for the shared-kinetics ablation:
    # the decoder bias still learns, but there is no expert-specific genome or divergence.
    xi_dim: int = 4

    # Feature Toggles (Modular Control)
    enable_presyn: bool = True
    enable_hebbian: bool = True
    enable_metabolism: bool = True
    use_flex_attention: bool = False
    # 0642.6.2.2: authorize the standalone tropical routing controller. Default-off and
    # intentionally not consulted by the live tensor forwards: current biological attention
    # and MoE scores are generally nonlinear/local-only, so they must first pass through an
    # explicit exact-affine adapter and certificate before hard routing is allowed.
    tropical_skeleton: bool = False

    # Native kernel toggles. Both are default-off so unsupported hardware/modes retain
    # the canonical Python implementation exactly.
    # jyb.2: the live presyn Triton kernel is deliberately narrow until jyb.3 supplies
    # autograd: deterministic FP32 CUDA decode only (one query, no grad/MC/learnable kinetics/
    # metriplectic integration). Every other shape/mode falls back to release_canonical.
    native_presyn: bool = decouple_config("BIO_FUSED_PRESYN", default=False, cast=bool)
    native_genetics: bool = decouple_config("BIO_FUSED_GENETICS", default=False, cast=bool)


# -----------------------------------------------------------------------------
# Differentiable affine recurrence (yw9.2.1)
# -----------------------------------------------------------------------------


def affine_scan_sequential(a: Tensor, b: Tensor, x0: Optional[Tensor] = None) -> Tensor:
    """Reference (sequential) affine scan: x_t = a_t * x_{t-1} + b_t, t = 0..T-1.

    The scan runs along dim 0. ``a``, ``b`` are ``(T, *batch)`` and broadcast against each other;
    ``x0`` is the pre-sequence state ``x_{-1}`` (``*batch``, default zeros). Returns ``(T, *batch)``.
    This is the obviously-correct `O(T)`-depth reference that the parallel `affine_scan` is
    validated against (yw9.2.1 acceptance) and is itself differentiable.
    """
    T = a.shape[0]
    prev = x0 if x0 is not None else torch.zeros_like(b[0])
    outs = []
    for t in range(T):
        prev = a[t] * prev + b[t]
        outs.append(prev)
    return torch.stack(outs, dim=0)


def affine_scan(a: Tensor, b: Tensor, x0: Optional[Tensor] = None) -> Tensor:
    """Differentiable PARALLEL associative scan of x_t = a_t * x_{t-1} + b_t (yw9.2.1).

    Hillis-Steele inclusive scan over the affine monoid — each affine map f_i(x) = a_i x + b_i
    composes associatively as  (a_l, b_l) ⊕ (a_r, b_r) = (a_r·a_l, a_r·b_l + b_r)  (left applied
    first). `O(log T)` sequential depth (vs `O(T)` for the loop) and fully differentiable w.r.t.
    ``a``, ``b``, ``x0`` (only `mul`/`add`/`cat`). Matches `affine_scan_sequential` to fp tolerance.

    Scan runs along dim 0. ``a``, ``b``: ``(T, *batch)``; ``x0``: ``*batch`` (default zeros).
    Stability: for the synaptic leaky integrators every ``a_t`` is a decay in ``(0,1)`` so the
    prefix products ``∏ a`` stay bounded — no blow-up in the forward or the gradient.
    """
    T = a.shape[0]
    # A, B hold the cumulative affine map F_t (so far) at each position; init = the per-step maps.
    # After the scan, F_t = (A_t, B_t) satisfies x_t = A_t * x0 + B_t. Broadcast both to a common
    # shape (symmetric — handles a or b carrying the broadcastable singleton dim) and clone so the
    # in-place-free scan below has contiguous, independent buffers.
    shape = torch.broadcast_shapes(a.shape, b.shape)
    A = a.broadcast_to(shape).clone()
    B = b.broadcast_to(shape).clone()
    d = 1
    while d < T:
        # out[t-d] is the earlier/left operand, identity-padded with (a=1, b=0) for the first d.
        ones = torch.ones_like(A[:d])
        zeros = torch.zeros_like(B[:d])
        A_left = torch.cat([ones, A[:-d]], dim=0)
        B_left = torch.cat([zeros, B[:-d]], dim=0)
        # combine(left=out[t-d], right=out[t]) = (a_r·a_l, a_r·b_l + b_r); a_r,b_r are the OLD A,B.
        A_new = A * A_left
        B_new = A * B_left + B
        A, B = A_new, B_new
        d *= 2
    if x0 is None:
        return B
    return A * x0 + B


def _soft_relu(x: Tensor, beta: float) -> Tensor:
    """Smooth max(x, 0) = softplus(βx)/β. Differentiable everywhere; → relu as β→∞."""
    return F.softplus(beta * x) / beta


def _soft_min(x: Tensor, c: float, beta: float) -> Tensor:
    """Smooth min(x, c). Differentiable everywhere; → hard min as β→∞."""
    return c - _soft_relu(c - x, beta)


def vesicle_depletion_refill(
    rrp: Tensor,
    res: Tensor,
    delay: List[Tensor],
    released: Tensor,
    *,
    prime_rate: float,
    rec_rate: float,
    rrp_cap: float = 30.0,
    beta: float = 50.0,
) -> Tuple[Tensor, Tensor, List[Tensor], Dict[str, Tensor]]:
    """One DIFFERENTIABLE, conservation-accurate step of the vesicle-pool dynamics (yw9.2.2).

    Mirrors the (currently `no_grad`, hard-clamped) RRP/RES/DELAY update inside
    ``release_canonical``, but (a) every clamp is a smooth softplus surrogate so gradients flow
    (``gradcheck`` passes), and (b) the pool bookkeeping is written as **explicit paired
    transfers**, so the conservation invariant holds *structurally*, independent of the surrogate
    sharpness:

        Δ(RRP + RES + Σdelay) = − released_eff · (1 − rec_rate)

    i.e. total vesicles change ONLY by the explicitly-modelled endocytosis recycling leak (exact
    conservation at ``rec_rate = 1``). No vesicles are spuriously lost to clamps: over-depletion is
    prevented by bounding the release to the available RRP, and the RRP cap routes the excess BACK
    to the reserve rather than discarding it.

    Transfers, in reference order (deplete → recover/prime → cap):
      RRP --released_eff--> in-flight (×rec_rate; the rest is the accounted leak)
      delay[0] --recovered--> RES;  RES --primed--> RRP;  RRP --over--> RES (cap)

    Args:
      rrp, res: ``(...)`` pools. delay: list of ``endo_delay`` in-flight tensors (oldest first).
      released: requested release this step (bounded to available RRP internally).
    Returns: ``(rrp', res', delay', diagnostics)``.
    """
    endo = len(delay)
    zeros = torch.zeros_like(rrp)
    # 1) release: deplete RRP (bounded to available so RRP stays >= 0), recycle a fraction in-flight
    released_eff = _soft_min(released, rrp, beta)
    rrp1 = rrp - released_eff
    recycled = released_eff * rec_rate
    # 2) recovery: oldest in-flight returns to the reserve
    recovered = delay[0] if endo > 0 else zeros
    res1 = res + recovered
    # 3) priming: move prime_rate · soft_min(RES, 1) from reserve to RRP
    take = _soft_min(res1, 1.0, beta)
    primed = prime_rate * take
    res2 = res1 - primed
    rrp2 = rrp1 + primed
    # 4) RRP cap: route the excess back to the reserve (conserve, don't discard)
    over = _soft_relu(rrp2 - rrp_cap, beta)
    rrp3 = rrp2 - over
    res3 = res2 + over
    # 5) in-flight queue shift (drop the recovered head, append the newly recycled tail). With no
    #    delay buffer (endo_delay==0) there is nowhere to queue the recycled vesicles, so route
    #    them straight back to the reserve — keeping the queue empty AND the conservation budget.
    if endo > 0:
        new_delay = delay[1:] + [recycled]
    else:
        new_delay = []
        res3 = res3 + recycled
    diagnostics = {
        "released_eff": released_eff,
        "recycled": recycled,
        "recovered": recovered,
        "primed": primed,
        "over": over,
    }
    return rrp3, res3, new_delay, diagnostics


@torch.no_grad()
def _detach_presyn_state(state: Dict[str, Any]) -> None:
    """Detach every tensor in a presyn state dict IN PLACE (truncates the gradient graph but
    keeps the values). Used at chunk boundaries for truncated BPTT (yw9.2.3)."""
    for k, v in list(state.items()):
        if torch.is_tensor(v):
            state[k] = v.detach()
        elif isinstance(v, list):
            state[k] = [x.detach() if torch.is_tensor(x) else x for x in v]


_PRESYN_STATE_KEYS = ("C", "BUF", "RRP", "RES", "PR", "CL", "E", "AMP")
_PRESYN_RUNTIME_BUFFER_NAMES = (
    "ema_e",
    "metriplectic_steps",
    "metriplectic_fallbacks",
    "metriplectic_last_energy_drift",
    "metriplectic_last_entropy_production",
    "metriplectic_last_free_energy_delta",
)


def _runtime_buffer(presyn: "SynapticPresyn", name: str) -> Tensor:
    value = getattr(presyn, name)
    if not torch.is_tensor(value):
        raise TypeError(f"presyn runtime buffer {name!r} must be a tensor")
    return value


def _flatten_presyn_state(state: Dict[str, Any]) -> tuple[tuple[Tensor, ...], int, bool]:
    """Flatten tensor/list state for a non-reentrant checkpoint boundary."""
    delay = state["DELAY"]
    if not isinstance(delay, list) or not all(torch.is_tensor(item) for item in delay):
        raise TypeError("presyn state DELAY must be a list of tensors")
    has_heat = "HEAT" in state
    tensors = [state[key] for key in _PRESYN_STATE_KEYS]
    if not all(torch.is_tensor(item) for item in tensors):
        raise TypeError("presyn recurrent state values must be tensors")
    tensors.extend(delay)
    if has_heat:
        heat = state["HEAT"]
        if not torch.is_tensor(heat):
            raise TypeError("presyn state HEAT must be a tensor")
        tensors.append(heat)
    return tuple(tensors), len(delay), has_heat


def _unflatten_presyn_state(
    tensors: tuple[Tensor, ...], *, delay_len: int, has_heat: bool
) -> Dict[str, Any]:
    """Rebuild the canonical state shape without mutating the caller's dictionary."""
    base_count = len(_PRESYN_STATE_KEYS)
    expected = base_count + delay_len + int(has_heat)
    if len(tensors) != expected:
        raise ValueError(f"expected {expected} flattened state tensors, got {len(tensors)}")
    state: Dict[str, Any] = dict(zip(_PRESYN_STATE_KEYS, tensors[:base_count]))
    state["DELAY"] = list(tensors[base_count : base_count + delay_len])
    if has_heat:
        state["HEAT"] = tensors[-1]
    return state


def _runtime_buffer_snapshot(presyn: "SynapticPresyn") -> tuple[Tensor, ...]:
    return tuple(
        _runtime_buffer(presyn, name).detach().clone()
        for name in _PRESYN_RUNTIME_BUFFER_NAMES
    )


def _runtime_buffer_mapping(tensors: tuple[Tensor, ...]) -> Dict[str, Tensor]:
    if len(tensors) != len(_PRESYN_RUNTIME_BUFFER_NAMES):
        raise ValueError(
            f"expected {len(_PRESYN_RUNTIME_BUFFER_NAMES)} runtime buffers, got {len(tensors)}"
        )
    return dict(zip(_PRESYN_RUNTIME_BUFFER_NAMES, tensors))


@torch.jit.script
def _scripted_detached_presyn_scan_cpu(
    calcium: torch.Tensor,
    buffer: torch.Tensor,
    rrp: torch.Tensor,
    reserve: torch.Tensor,
    primed: torch.Tensor,
    clamp: torch.Tensor,
    energy: torch.Tensor,
    amplitude: torch.Tensor,
    delay: List[torch.Tensor],
    drive: torch.Tensor,
    idx: torch.Tensor,
    valid: torch.Tensor,
    first_active_key_count: int,
    ema_e: torch.Tensor,
    generator: Optional[torch.Generator],
    train: bool,
    stochastic_frac: float,
    normal_draw_width: int,
    stochastic_count_cap: int,
    rho_c: float,
    rho_b: float,
    alpha_ca: float,
    alpha_buf_on: float,
    alpha_buf_off: float,
    syt_fast_kd: float,
    syt_slow_kd: float,
    doc2_gain: float,
    complexin_bias: float,
    q_beta: float,
    qmax: float,
    rec_rate: float,
    prime_rate: float,
    unprime_per_release: float,
    nsf_recover: float,
    energy_fill: float,
    energy_max: float,
    energy_use: float,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    List[torch.Tensor],
    torch.Tensor,
]:
    """TorchScript loop for the ordinary detached CPU recurrence."""
    calcium = calcium.detach().clone()
    buffer = buffer.detach().clone()
    rrp = rrp.detach().clone()
    reserve = reserve.detach().clone()
    primed = primed.detach().clone()
    clamp = clamp.detach().clone()
    energy = energy.detach().clone()
    delay = [item.detach().clone() for item in delay]
    ema_e = ema_e.detach().clone()
    outputs = torch.jit.annotate(List[torch.Tensor], [])
    batch = int(drive.size(0))
    heads = int(drive.size(1))
    query_count = int(drive.size(2))

    for offset in range(query_count):
        active_key_count = first_active_key_count + offset
        step_drive = drive[:, :, offset : offset + 1, :]
        step_valid = valid[:, :, offset : offset + 1, :]
        step_idx = idx[:, :, offset : offset + 1, :].masked_fill(~step_valid, 0)
        flat_idx = step_idx.reshape(batch, heads, -1)

        calcium_prefix = calcium[:, :, :active_key_count]
        buffer_prefix = buffer[:, :, :active_key_count]
        rrp_prefix = rrp[:, :, :active_key_count]
        reserve_prefix = reserve[:, :, :active_key_count]
        primed_prefix = primed[:, :, :active_key_count]
        clamp_prefix = clamp[:, :, :active_key_count]
        energy_prefix = energy[:, :, :active_key_count]

        calcium_edge = calcium_prefix.gather(2, flat_idx).view_as(step_drive)
        buffer_edge = buffer_prefix.gather(2, flat_idx).view_as(step_drive)
        primed_edge = primed_prefix.gather(2, flat_idx).view_as(step_drive)
        clamp_edge = clamp_prefix.gather(2, flat_idx).view_as(step_drive)
        rrp_edge = rrp_prefix.gather(2, flat_idx).view_as(step_drive)
        energy_edge = energy_prefix.gather(2, flat_idx).view_as(step_drive)

        influx = alpha_ca * F.softplus(step_drive)
        calcium_edge = (
            rho_c * calcium_edge
            + influx
            - alpha_buf_on * calcium_edge * (1.0 - buffer_edge)
            + alpha_buf_off * buffer_edge
        ).clamp(min=0.0)
        fast = calcium_edge / (calcium_edge + syt_fast_kd)
        slow = calcium_edge / (calcium_edge + syt_slow_kd)
        sensor = (
            0.7 * fast
            + 0.3 * slow
            + doc2_gain * torch.sigmoid(4.0 * (calcium_edge - 0.12))
        )
        fuse_base = torch.sigmoid(
            3.0 * sensor + 2.0 * primed_edge - 2.0 * (clamp_edge + complexin_bias)
        )
        probability = (fuse_base * torch.sigmoid(step_drive)).clamp(0.0, 1.0)
        released_deterministic = probability * rrp_edge
        if train and stochastic_frac > 0.0:
            do_stochastic = (
                torch.rand(
                    probability[..., 0].shape,
                    device=probability.device,
                    dtype=torch.float32,
                    generator=generator,
                )
                < stochastic_frac
            )
            stochastic_mask = do_stochastic.unsqueeze(-1).expand_as(probability)
            stochastic_mask = stochastic_mask & step_valid
            if not bool(stochastic_mask.any()):
                released = released_deterministic
            else:
                if stochastic_count_cap <= 0:
                    sampled = torch.zeros_like(probability)
                else:
                    count_float = torch.clamp(
                        rrp_edge.round(), 0.0, float(stochastic_count_cap)
                    ).to(torch.float32)
                    probability_float = probability.to(torch.float32).clamp(
                        1e-6, 1.0 - 1e-6
                    )
                    mean = count_float * probability_float
                    variance = (
                        count_float * probability_float * (1.0 - probability_float)
                    )
                    deviation = torch.sqrt(variance + 1e-6)
                    noise = torch.randn(
                        (
                            mean.size(0),
                            mean.size(1),
                            mean.size(2),
                            normal_draw_width,
                        ),
                        device=mean.device,
                        dtype=mean.dtype,
                        generator=generator,
                    )[..., : mean.size(-1)]
                    sampled = (mean + deviation * noise).clamp(min=0.0)
                    sampled = torch.minimum(sampled, count_float).to(
                        probability.dtype
                    )
                released = torch.where(
                    stochastic_mask, sampled, released_deterministic
                )
        else:
            released = released_deterministic
        released = released * step_valid.to(released.dtype)
        qamp = torch.sigmoid(q_beta * (energy_edge - 0.5)) * qmax
        edge_release = released * qamp

        flat_release = released.detach().reshape(batch, heads, -1).to(calcium.dtype)
        flat_drive = (
            step_drive.detach().reshape(batch, heads, -1).to(calcium.dtype)
        )
        flat_valid_bool = step_valid.reshape(batch, heads, -1)
        flat_valid = flat_valid_bool.to(calcium.dtype)
        flat_drive = torch.where(
            flat_valid_bool, flat_drive, torch.zeros_like(flat_drive)
        )
        add_values = torch.zeros_like(calcium_prefix)
        drive_values = torch.zeros_like(calcium_prefix)
        count_values = torch.zeros_like(calcium_prefix)
        add_values.scatter_add_(2, flat_idx, flat_release)
        drive_values.scatter_add_(2, flat_idx, flat_drive)
        count_values.scatter_add_(2, flat_idx, flat_valid)
        accessed = (count_values > 0).to(calcium.dtype)

        calcium_updated = (
            rho_c * calcium_prefix
            + alpha_ca * F.softplus(drive_values) * accessed
            - alpha_buf_on * calcium_prefix * (1.0 - buffer_prefix)
            + alpha_buf_off * buffer_prefix
        ).clamp(min=0.0)
        buffer_updated = (
            rho_b * buffer_prefix
            + alpha_buf_on * calcium_prefix * (1.0 - buffer_prefix)
            - alpha_buf_off * buffer_prefix
        ).clamp(0.0, 1.0)
        rrp_updated = torch.clamp(rrp_prefix - add_values, 0.0)
        if len(delay) > 0:
            reserve_updated = reserve_prefix + delay[0][:, :, :active_key_count]
            for delay_index in range(len(delay) - 1):
                delay[delay_index][:, :, :active_key_count] = delay[delay_index + 1][
                    :, :, :active_key_count
                ]
            delay[-1][:, :, :active_key_count] = add_values * rec_rate
        else:
            reserve_updated = reserve_prefix
        take = torch.minimum(reserve_updated, torch.ones_like(reserve_updated))
        reserve_updated = torch.clamp(
            reserve_updated - prime_rate * take, 0.0
        )
        rrp_updated = torch.clamp(
            rrp_updated + prime_rate * take, 0.0, 30.0
        )
        primed_updated = torch.clamp(
            primed_prefix * (1.0 - unprime_per_release * add_values)
            + nsf_recover * (1.0 - primed_prefix),
            0.0,
            1.0,
        )
        clamp_updated = torch.clamp(
            clamp_prefix * 0.995 + 0.005 - unprime_per_release * add_values,
            0.0,
            1.0,
        )
        energy_updated = torch.clamp(
            energy_prefix
            + energy_fill * (energy_max - energy_prefix)
            - energy_use * add_values,
            0.0,
            energy_max,
        )

        calcium[:, :, :active_key_count] = calcium_updated
        buffer[:, :, :active_key_count] = buffer_updated
        rrp[:, :, :active_key_count] = rrp_updated
        reserve[:, :, :active_key_count] = reserve_updated
        primed[:, :, :active_key_count] = primed_updated
        clamp[:, :, :active_key_count] = clamp_updated
        energy[:, :, :active_key_count] = energy_updated
        if train:
            valid_weight = step_valid.to(edge_release.dtype)
            scale = (
                (edge_release.detach().abs() * valid_weight).sum()
                / valid_weight.sum().clamp_min(1)
            ).clamp_min(1e-3)
            ema_e.mul_(0.99)
            ema_e.add_(0.01 * scale)
        outputs.append(edge_release / (ema_e + 1e-6))

    return (
        torch.cat(outputs, dim=2),
        calcium,
        buffer,
        rrp,
        reserve,
        primed,
        clamp,
        energy,
        amplitude,
        delay,
        ema_e,
    )


def _presyn_state_prefix(state: Dict[str, Any], key_count: int) -> Dict[str, Any]:
    """Return a view-backed state containing only the materialized key prefix."""
    flat_state, delay_len, has_heat = _flatten_presyn_state(state)
    return _unflatten_presyn_state(
        tuple(tensor[:, :, :key_count] for tensor in flat_state),
        delay_len=delay_len,
        has_heat=has_heat,
    )


def _grow_presyn_state_prefix(
    state: Dict[str, Any], source: Dict[str, Any], key_count: int
) -> Dict[str, Any]:
    """Append newly materialized source slots to a recurrent state prefix."""
    flat_state, delay_len, has_heat = _flatten_presyn_state(state)
    flat_source, source_delay_len, source_has_heat = _flatten_presyn_state(source)
    if source_delay_len != delay_len or source_has_heat != has_heat:
        raise RuntimeError("presyn recurrent state changed its schema while growing")
    current_key_count = int(flat_state[0].size(2))
    if key_count < current_key_count:
        raise ValueError(
            f"cannot shrink a presyn state prefix from {current_key_count} to {key_count}"
        )
    if key_count == current_key_count:
        return state
    return _unflatten_presyn_state(
        tuple(
            torch.cat((current, original[:, :, current_key_count:key_count]), dim=2)
            for current, original in zip(flat_state, flat_source)
        ),
        delay_len=delay_len,
        has_heat=has_heat,
    )


def _commit_presyn_state_prefix(
    state: Dict[str, Any], prefix: Dict[str, Any], source: Dict[str, Any]
) -> None:
    """Merge an updated materialized prefix with untouched future state exactly once."""
    flat_prefix, delay_len, has_heat = _flatten_presyn_state(prefix)
    flat_source, source_delay_len, source_has_heat = _flatten_presyn_state(source)
    if source_delay_len != delay_len or source_has_heat != has_heat:
        raise RuntimeError("presyn recurrent state changed its schema while committing")
    prefix_key_count = int(flat_prefix[0].size(2))
    source_key_count = int(flat_source[0].size(2))
    if prefix_key_count > source_key_count:
        raise ValueError(
            "presyn state prefix exceeds its source extent; "
            f"got prefix={prefix_key_count}, source={source_key_count}"
        )
    if prefix_key_count == source_key_count:
        merged = flat_prefix
    else:
        merged = tuple(
            torch.cat((current, original[:, :, prefix_key_count:]), dim=2)
            for current, original in zip(flat_prefix, flat_source)
        )
    state.update(
        _unflatten_presyn_state(merged, delay_len=delay_len, has_heat=has_heat)
    )


@torch.no_grad()
def _copy_presyn_state_prefix_(state: Dict[str, Any], prefix: Dict[str, Any]) -> None:
    """Commit a detached recurrent prefix into its full-capacity backing tensors in place."""
    flat_state, delay_len, has_heat = _flatten_presyn_state(state)
    flat_prefix, prefix_delay_len, prefix_has_heat = _flatten_presyn_state(prefix)
    if prefix_delay_len != delay_len or prefix_has_heat != has_heat:
        raise RuntimeError("presyn recurrent state changed its schema while copying")
    prefix_key_count = int(flat_prefix[0].size(2))
    if prefix_key_count > int(flat_state[0].size(2)):
        raise ValueError("presyn state prefix exceeds its destination extent")
    for destination, source in zip(flat_state, flat_prefix):
        destination[:, :, :prefix_key_count].copy_(source)


def _release_recurrence_group(
    presyn: "SynapticPresyn",
    state: Dict[str, Any],
    drive: Tensor,
    idx: Tensor,
    valid: Optional[Tensor],
    *,
    train: bool,
    differentiable: bool,
    active_key_count: Optional[int],
    runtime_buffers: Optional[Dict[str, Tensor]] = None,
) -> Tensor:
    """Advance one graph/checkpoint group while preserving per-query causal values."""
    query_count = int(drive.size(2))
    if active_key_count is None:
        return presyn.release_canonical(
            state,
            drive,
            idx,
            train=train,
            valid=valid,
            differentiable=differentiable,
            runtime_buffers=runtime_buffers,
            active_key_count=active_key_count,
        )

    first_active_key_count = active_key_count - query_count + 1
    if first_active_key_count < 1:
        raise ValueError(
            "active_key_count must cover every query in its recurrence group; "
            f"got end={active_key_count}, queries={query_count}"
        )
    state_key_count = int(state["C"].size(2))
    if active_key_count > state_key_count:
        raise ValueError(
            "active_key_count exceeds the recurrent state extent; "
            f"got active={active_key_count}, state={state_key_count}"
        )

    cfg = presyn.cfg
    train_stochastic_frac = min(
        1.0,
        max(
            0.0,
            cfg.stochastic_train_frac * getattr(presyn, "_nm_ach_gain", 1.0),
        ),
    )
    flat_state, _, _ = _flatten_presyn_state(state)
    if (
        valid is not None
        and query_count > 1
        and cfg.enable_presyn
        and drive.device.type == "cpu"
        and drive.dtype == torch.float32
        and (train or not torch.is_grad_enabled())
        and not differentiable
        and runtime_buffers is None
        and presyn.kinetics is None
        and not presyn.use_metriplectic_integrator()
        and not bool(getattr(presyn, "_mc_sampling", False))
        and cfg.stochastic_mode == "normal_reparam"
        and "HEAT" not in state
        and not any(tensor.requires_grad for tensor in flat_state)
    ):
        train_generator = (
            presyn._train_sampling_generator(drive.device)
            if train and train_stochastic_frac > 0.0
            else None
        )
        scripted = _scripted_detached_presyn_scan_cpu(
            state["C"],
            state["BUF"],
            state["RRP"],
            state["RES"],
            state["PR"],
            state["CL"],
            state["E"],
            state["AMP"],
            state["DELAY"],
            drive,
            idx,
            valid,
            first_active_key_count,
            presyn.ema_e,
            train_generator,
            train,
            train_stochastic_frac,
            int(cfg.attn_topk),
            int(cfg.stochastic_count_cap),
            math.exp(-1.0 / cfg.tau_c),
            math.exp(-1.0 / cfg.tau_buf),
            cfg.alpha_ca,
            cfg.alpha_buf_on,
            cfg.alpha_buf_off,
            cfg.syt_fast_kd,
            cfg.syt_slow_kd,
            cfg.doc2_gain,
            cfg.complexin_bias,
            cfg.q_beta,
            cfg.qmax,
            cfg.rec_rate,
            cfg.prime_rate,
            cfg.unprime_per_release,
            cfg.nsf_recover,
            cfg.energy_fill,
            cfg.energy_max,
            cfg.energy_use,
        )
        if train_generator is not None:
            presyn._commit_train_sampling_generator(train_generator)
        state.update(
            {
                "C": scripted[1],
                "BUF": scripted[2],
                "RRP": scripted[3],
                "RES": scripted[4],
                "PR": scripted[5],
                "CL": scripted[6],
                "E": scripted[7],
                "AMP": scripted[8],
                "DELAY": scripted[9],
            }
        )
        if train:
            with torch.no_grad():
                presyn.ema_e.copy_(scripted[10])
        return scripted[0]

    # Invalid top-k entries can point into the preallocated future suffix. When an explicit valid
    # mask is present, map those dead entries to key zero and run each exact causal step only over
    # its materialized prefix. This removes full-sequence state arithmetic and one active-mask
    # restore per query. The updated prefix is joined with the untouched suffix once per group.
    # Calls without a valid mask retain the full-state path because future indices may be genuine
    # inputs whose release values must still be returned.
    if valid is not None and not (
        query_count == 1 and active_key_count == state_key_count
    ):
        if presyn.use_metriplectic_integrator() and "HEAT" not in state:
            state["HEAT"] = torch.zeros_like(state["C"])
            flat_state, _, _ = _flatten_presyn_state(state)
        if not differentiable and not any(tensor.requires_grad for tensor in flat_state):
            drives = drive.split(1, dim=2)
            idxs = idx.split(1, dim=2)
            valids = valid.split(1, dim=2)
            outputs: List[Tensor] = []
            for offset, (step_drive, step_idx, step_valid) in enumerate(
                zip(drives, idxs, valids)
            ):
                step_key_count = first_active_key_count + offset
                local_state = _presyn_state_prefix(state, step_key_count)
                local_idx = step_idx.masked_fill(~step_valid, 0)
                outputs.append(
                    presyn.release_canonical(
                        local_state,
                        step_drive,
                        local_idx,
                        train=train,
                        valid=step_valid,
                        differentiable=False,
                        runtime_buffers=runtime_buffers,
                    )
                )
                _copy_presyn_state_prefix_(state, local_state)
            return torch.cat(outputs, dim=2)

        source_state = dict(state)
        local_state = _presyn_state_prefix(source_state, first_active_key_count)
        drives = drive.split(1, dim=2)
        idxs = idx.split(1, dim=2)
        valids = valid.split(1, dim=2)
        outputs: List[Tensor] = []
        for offset, (step_drive, step_idx, step_valid) in enumerate(
            zip(drives, idxs, valids)
        ):
            step_key_count = first_active_key_count + offset
            local_state = _grow_presyn_state_prefix(
                local_state, source_state, step_key_count
            )
            local_idx = step_idx.masked_fill(~step_valid, 0)
            outputs.append(
                presyn.release_canonical(
                    local_state,
                    step_drive,
                    local_idx,
                    train=train,
                    valid=step_valid,
                    differentiable=differentiable,
                    runtime_buffers=runtime_buffers,
                )
            )
        _commit_presyn_state_prefix(state, local_state, source_state)
        return torch.cat(outputs, dim=2)

    drives = drive.split(1, dim=2)
    idxs = idx.split(1, dim=2)
    valids: Sequence[Optional[Tensor]] = (
        [None] * query_count if valid is None else valid.split(1, dim=2)
    )
    outputs = [
        presyn.release_canonical(
            state,
            step_drive,
            step_idx,
            train=train,
            valid=step_valid,
            differentiable=differentiable,
            runtime_buffers=runtime_buffers,
            active_key_count=first_active_key_count + offset,
        )
        for offset, (step_drive, step_idx, step_valid) in enumerate(
            zip(drives, idxs, valids)
        )
    ]
    return torch.cat(outputs, dim=2)


def _checkpoint_recurrence_segment(
    presyn: "SynapticPresyn",
    state: Dict[str, Any],
    drives: List[Tensor],
    idxs: List[Tensor],
    valids: List[Optional[Tensor]],
    active_key_counts: List[Optional[int]],
    *,
    train: bool,
) -> List[Tensor]:
    """Checkpoint one full-BPTT window with explicit functional runtime state."""
    from torch.utils.checkpoint import checkpoint

    cfg = presyn.cfg
    if (
        not cfg.enable_presyn
        or not cfg.learnable_kinetics
        or not cfg.differentiable_recurrence
        or not presyn.use_metriplectic_integrator()
        or presyn.kinetics is None
    ):
        raise ValueError(
            "recurrence checkpointing requires enable_presyn=True, learnable_kinetics=True, "
            "differentiable_recurrence=True, and metriplectic_integrator=True"
        )
    if cfg.use_flex_attention:
        raise ValueError("recurrence checkpointing does not yet support FlexAttention")
    if cfg.stochastic_train_frac != 0.0:
        raise ValueError("recurrence checkpointing requires stochastic_train_frac=0")
    if bool(getattr(presyn, "_mc_sampling", False)):
        raise ValueError("recurrence checkpointing does not support MC release sampling")
    if any(tensor.device.type != "cpu" for tensor in (*drives, *idxs)) or any(
        valid is not None and valid.device.type != "cpu" for valid in valids
    ):
        raise ValueError("recurrence checkpointing is currently supported on CPU only")
    drive_dtype = drives[0].dtype
    if drive_dtype not in {torch.float32, torch.float64} or any(
        drive.dtype != drive_dtype for drive in drives
    ):
        raise TypeError("recurrence checkpointing currently requires float32 or float64 drives")
    if any(idx.dtype != torch.long for idx in idxs):
        raise TypeError("recurrence checkpointing requires int64 index tensors")

    if "HEAT" not in state:
        state["HEAT"] = torch.zeros_like(state["C"])

    flat_state, delay_len, has_heat = _flatten_presyn_state(state)
    if any(tensor.device.type != "cpu" for tensor in flat_state):
        raise ValueError("checkpointed recurrent state must be on CPU")
    if any(tensor.dtype != drive_dtype for tensor in flat_state):
        raise TypeError("checkpointed recurrent state dtype must match the drive dtype")
    state_count = len(flat_state)
    step_count = len(drives)
    runtime_initial = _runtime_buffer_snapshot(presyn)
    if any(tensor.device.type != "cpu" for tensor in runtime_initial):
        raise ValueError("checkpointed runtime buffers must be on CPU")
    if runtime_initial[0].dtype != drive_dtype:
        raise TypeError("checkpointed EMA dtype must match the drive dtype")
    if any(
        parameter.device.type != "cpu" or parameter.dtype != drive_dtype
        for parameter in presyn.kinetics.parameters()
    ):
        raise TypeError("checkpointed kinetics must match the CPU drive dtype")
    runtime_count = len(runtime_initial)
    explicit_valids = tuple(
        torch.ones_like(drive, dtype=torch.bool) if valid is None else valid
        for drive, valid in zip(drives, valids)
    )
    has_valid = tuple(valid is not None for valid in valids)

    def run_segment(*inputs: Tensor) -> tuple[Tensor, ...]:
        local_state = _unflatten_presyn_state(
            tuple(inputs[:state_count]), delay_len=delay_len, has_heat=has_heat
        )
        runtime_start = state_count
        drive_start = runtime_start + runtime_count
        idx_start = drive_start + step_count
        valid_start = idx_start + step_count
        local_runtime = _runtime_buffer_mapping(
            tuple(inputs[runtime_start:drive_start])
        )
        segment_drives = inputs[drive_start:idx_start]
        segment_idxs = inputs[idx_start:valid_start]
        segment_valids = inputs[valid_start:]
        outputs = [
            _release_recurrence_group(
                presyn,
                local_state,
                drive,
                idx,
                train=train,
                valid=valid if valid_is_present else None,
                differentiable=True,
                runtime_buffers=local_runtime,
                active_key_count=active_key_count,
            )
            for drive, idx, valid, valid_is_present, active_key_count in zip(
                segment_drives,
                segment_idxs,
                segment_valids,
                has_valid,
                active_key_counts,
            )
        ]
        next_state, next_delay_len, next_has_heat = _flatten_presyn_state(local_state)
        if next_delay_len != delay_len or next_has_heat != has_heat:
            raise RuntimeError("checkpointed presyn recurrence changed its state schema")
        next_runtime = tuple(local_runtime[name] for name in _PRESYN_RUNTIME_BUFFER_NAMES)
        return (*outputs, *next_state, *next_runtime)

    result = checkpoint(
        run_segment,
        *flat_state,
        *runtime_initial,
        *drives,
        *idxs,
        *explicit_valids,
        use_reentrant=False,
        preserve_rng_state=False,
        determinism_check="default",
        early_stop=False,
    )
    outputs = list(result[:step_count])
    final_state_end = step_count + state_count
    final_state = _unflatten_presyn_state(
        tuple(result[step_count:final_state_end]), delay_len=delay_len, has_heat=has_heat
    )
    state.update(final_state)
    with torch.no_grad():
        for name, value in zip(
            _PRESYN_RUNTIME_BUFFER_NAMES, result[final_state_end:]
        ):
            _runtime_buffer(presyn, name).copy_(value)
    return outputs


def chunked_recurrence(
    presyn: "SynapticPresyn",
    state: Dict[str, Any],
    drives: List[Tensor],
    idxs: List[Tensor],
    *,
    chunk_len: int,
    checkpoint_len: int = 0,
    train: bool = False,
    valids: Optional[List[Optional[Tensor]]] = None,
    active_key_counts: Optional[List[Optional[int]]] = None,
    differentiable: bool = True,
) -> List[Tensor]:
    """Run the DIFFERENTIABLE presynaptic recurrence over a sequence of steps with truncated
    BPTT (yw9.2.3) or exact-gradient checkpoint/replay (0642.1.2.6).

    ``drives``/``idxs`` are per-step ``(B,H,T,K)`` tensors. The carried ``state`` is advanced with
    ``release_canonical(differentiable=True)`` so gradients flow through it; every ``chunk_len``
    steps the state is DETACHED, so backprop is truncated to within a chunk and peak memory is
    bounded by ``chunk_len`` steps instead of the full sequence length. ``chunk_len <= 0`` disables
    truncation (full BPTT). Detaching changes only the gradient graph, never the forward values, so
    the returned per-step release biases are identical to a full-BPTT (or detached) run.

    ``valids`` (optional) is a matching list of per-step ``(B,H,T,K)`` boolean masks for the live
    attention path (causal top-k masking); ``None`` (or a ``None`` entry) means all edges valid.
    ``active_key_counts`` gives each step/group's final materialized key extent. When a group holds
    multiple queries, it is still evaluated one query at a time; grouping changes graph/checkpoint
    boundaries, never forward values or causal state evolution.
    ``differentiable`` toggles the autograd state recurrence: ``True`` (default) carries gradient
    through the state for BPTT; ``False`` runs the same causal schedule under no_grad (the live eval
    path — identical forward values, no graph). hwxb.4.6 wires this into the model attention forward.

    ``checkpoint_len > 0`` groups that many recurrence steps behind one non-reentrant PyTorch
    checkpoint. It preserves full-BPTT gradients while retaining recurrent state only at window
    boundaries. EMA and metriplectic telemetry cross the boundary as explicit functional tensors,
    so backward replay cannot apply persistent side effects twice. It is mutually exclusive with
    truncated-BPTT ``chunk_len``.
    """
    if len(drives) != len(idxs):
        raise ValueError("drives and idxs must have the same number of recurrence steps")
    if valids is not None and len(valids) != len(drives):
        raise ValueError("valids must match the number of recurrence steps")
    if active_key_counts is not None and len(active_key_counts) != len(drives):
        raise ValueError("active_key_counts must match the number of recurrence steps")
    if checkpoint_len < 0:
        raise ValueError(f"checkpoint_len must be >= 0, got {checkpoint_len}")
    if checkpoint_len > 0 and chunk_len > 0:
        raise ValueError("checkpoint_len and truncated-BPTT chunk_len are mutually exclusive")

    normalized_valids = (
        [None] * len(drives) if valids is None else valids
    )
    normalized_active_key_counts = (
        [None] * len(drives) if active_key_counts is None else active_key_counts
    )
    if checkpoint_len > 0 and differentiable and torch.is_grad_enabled():
        checkpointed_outputs: List[Tensor] = []
        for start in range(0, len(drives), checkpoint_len):
            stop = start + checkpoint_len
            checkpointed_outputs.extend(
                _checkpoint_recurrence_segment(
                    presyn,
                    state,
                    drives[start:stop],
                    idxs[start:stop],
                    normalized_valids[start:stop],
                    normalized_active_key_counts[start:stop],
                    train=train,
                )
            )
        return checkpointed_outputs

    outs: List[Tensor] = []
    for t, (drive, idx, active_key_count) in enumerate(
        zip(drives, idxs, normalized_active_key_counts)
    ):
        if chunk_len > 0 and t > 0 and (t % chunk_len == 0):
            _detach_presyn_state(state)
        valid = normalized_valids[t]
        outs.append(
            _release_recurrence_group(
                presyn,
                state,
                drive,
                idx,
                train=train,
                valid=valid,
                differentiable=differentiable,
                active_key_count=active_key_count,
            )
        )
    return outs


def _logit(p: float) -> float:
    """Inverse sigmoid: θ such that sigmoid(θ) = p, for p ∈ (0,1)."""
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _softplus_inv(y: float) -> float:
    """Inverse softplus: θ such that softplus(θ) = y, for y > 0."""
    return math.log(math.expm1(max(y, 1e-6)))


# Upper bound on the buffer-coupling rates. The calcium↔buffer linear subsystem has the 2×2 map
# [[ρc − αon(1−BUF), αoff], [αon(1−BUF), ρb − αoff]]; bounding αon/αoff keeps its spectral radius
# < 1 for all BUF∈[0,1] and ρ∈(0,1), so learning the kinetics can never destabilize the recurrence
# (see docs/differentiable_synaptic_dynamics_design.md §4).
_ABUF_MAX = 0.5


def cb_spectral_radius(
    rho_c: Tensor, rho_b: Tensor, alpha_buf_on: Tensor, alpha_buf_off: Tensor, beta: Tensor
) -> Tensor:
    """Spectral radius of the calcium↔buffer 2×2 linear transition matrix (yw9.7).

    Freezing the bilinear coupling coefficient β = (1−BUF) ∈ [0,1], the C/BUF update is linear:

        M(β) = [[ ρc − αon·β ,      αoff        ],
                [   αon·β     ,  ρb − αoff      ]]

    Closed-form 2×2 spectral radius (no eig decomposition, so it's differentiable and broadcasts):
    real eigenvalues ⇒ max(|tr ± √Δ|)/2;  complex pair (Δ<0) ⇒ √det. Contraction ⟺ ρ(M) < 1.
    All inputs broadcast against ``beta``. See docs/stable_recurrence_theory.md.
    """
    a = rho_c - alpha_buf_on * beta
    d = rho_b - alpha_buf_off
    b = alpha_buf_off
    c = alpha_buf_on * beta
    tr = a + d
    det = a * d - b * c
    disc = tr * tr - 4.0 * det
    real = disc >= 0
    sqrt_disc = torch.sqrt(disc.clamp(min=0.0))
    rho_real = torch.maximum((tr + sqrt_disc).abs(), (tr - sqrt_disc).abs()) * 0.5
    rho_complex = torch.sqrt(det.clamp(min=0.0))  # |λ| = √det for a complex-conjugate pair
    return torch.where(real, rho_real, rho_complex)


class LearnableKinetics(nn.Module):
    """Stability-preserving, SGD-learnable presynaptic calcium/buffer kinetics (yw9.3).

    Decays are mapped through ``sigmoid`` so they stay in ``(0,1)`` (contractive leaky integrators);
    the influx gain through ``softplus`` so it stays positive; the buffer-coupling rates through a
    bounded ``_ABUF_MAX·sigmoid`` so the C↔BUF subsystem provably stays contractive. The raw
    Parameters are initialized (via the inverse maps) so a fresh module reproduces the cfg constants
    EXACTLY — turning ``learnable_kinetics`` on is a no-op until SGD moves them.
    """

    def __init__(self, cfg: "SynapticConfig"):
        super().__init__()
        rho_c0 = math.exp(-1.0 / cfg.tau_c)
        rho_b0 = math.exp(-1.0 / cfg.tau_buf)
        self.theta_rho_c = nn.Parameter(torch.tensor(_logit(rho_c0)))
        self.theta_rho_b = nn.Parameter(torch.tensor(_logit(rho_b0)))
        self.theta_alpha_ca = nn.Parameter(torch.tensor(_softplus_inv(cfg.alpha_ca)))
        self.theta_alpha_buf_on = nn.Parameter(torch.tensor(_logit(cfg.alpha_buf_on / _ABUF_MAX)))
        self.theta_alpha_buf_off = nn.Parameter(torch.tensor(_logit(cfg.alpha_buf_off / _ABUF_MAX)))

    @property
    def rho_c(self) -> Tensor:
        return torch.sigmoid(self.theta_rho_c)

    @property
    def rho_b(self) -> Tensor:
        return torch.sigmoid(self.theta_rho_b)

    @property
    def alpha_ca(self) -> Tensor:
        return F.softplus(self.theta_alpha_ca)

    @property
    def alpha_buf_on(self) -> Tensor:
        return _ABUF_MAX * torch.sigmoid(self.theta_alpha_buf_on)

    @property
    def alpha_buf_off(self) -> Tensor:
        return _ABUF_MAX * torch.sigmoid(self.theta_alpha_buf_off)

    @torch.no_grad()
    def values(self) -> Dict[str, float]:
        """Current constrained kinetic values (for telemetry / tests)."""
        return {
            "rho_c": float(self.rho_c),
            "rho_b": float(self.rho_b),
            "alpha_ca": float(self.alpha_ca),
            "alpha_buf_on": float(self.alpha_buf_on),
            "alpha_buf_off": float(self.alpha_buf_off),
        }

    def spectral_radius(self, n_beta: int = 21) -> Tensor:
        """Worst-case spectral radius of the C↔BUF linear transition over BUF∈[0,1] (yw9.7).

        The calcium/buffer subsystem is contractive (cannot blow up) iff this is < 1. It is
        differentiable in the kinetics, so it can be read as a telemetry/stability margin or used
        as a soft penalty. (As a penalty, note the gradient is non-smooth where it matters: ``√``
        of the eigenvalue discriminant has an infinite slope at the real↔complex boundary, and
        ``.max`` over the β-grid is a subgradient — fine for monitoring, but prefer a smooth
        surrogate if optimizing it directly.) See docs/stable_recurrence_theory.md for the derivation.
        """
        betas = torch.linspace(
            0.0, 1.0, n_beta, dtype=self.theta_rho_c.dtype, device=self.theta_rho_c.device
        )
        rho = cb_spectral_radius(
            self.rho_c, self.rho_b, self.alpha_buf_on, self.alpha_buf_off, betas
        )
        return rho.max()


# -----------------------------------------------------------------------------
# Presynaptic biophysics
# -----------------------------------------------------------------------------


class SynapticPresyn(nn.Module):
    cfg: SynapticConfig
    """
    Vectorized presynaptic module with explicit Syt1/7 mix, complexin clamp,
    Munc13/18 priming, clathrin/dynamin endocytosis (queue), V-ATPase/VDAC
    coupling, EMA normalization, optional stochastic release on a fraction
    of edges, and a septin-like distance barrier for attention logits.
    """

    def __init__(self, d_head: int, cfg: SynapticConfig):
        super().__init__()
        object.__setattr__(self, "cfg", cfg)
        self.register_buffer("ema_e", torch.ones(1))
        # Generator states are opaque and backend-specific (CPU and CUDA even use different
        # state sizes). Keep one variable-length blob per supported backend plus a shared lazy
        # seed for safe cross-device migration. Same-backend checkpoints resume exactly; when a
        # checkpoint first moves to another backend, that backend starts from the shared seed
        # instead of trying to ingest an incompatible state blob.
        self.register_buffer("_presyn_train_rng_seed", torch.full((), -1, dtype=torch.int64))
        self.register_buffer(
            "_presyn_train_cpu_rng_state", torch.empty(0, dtype=torch.uint8)
        )
        self.register_buffer(
            "_presyn_train_cuda_rng_state", torch.empty(0, dtype=torch.uint8)
        )
        # 0642.1.2: on-device, non-persistent guard evidence for the live metriplectic recurrence.
        # These are counters/telemetry, not learned or checkpoint state. Keeping reductions as
        # tensors avoids synchronizing CUDA on every presynaptic step.
        self.register_buffer("metriplectic_steps", torch.zeros((), dtype=torch.int64), persistent=False)
        self.register_buffer(
            "metriplectic_fallbacks", torch.zeros((), dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "metriplectic_last_energy_drift", torch.zeros((), dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "metriplectic_last_entropy_production",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "metriplectic_last_free_energy_delta",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )
        # yw9.3: SGD-learnable, stability-preserving calcium/buffer kinetics (default-off). When
        # present, release_canonical sources rho_c/rho_b/alpha_* from these Parameters.
        self.kinetics = LearnableKinetics(cfg) if cfg.learnable_kinetics else None

    def _train_rng_state_buffer(self, device: torch.device) -> Tensor:
        if device.type == "cpu":
            return self._presyn_train_cpu_rng_state
        if device.type == "cuda":
            return self._presyn_train_cuda_rng_state
        raise RuntimeError(
            "private presynaptic train sampling supports CPU and CUDA generators; "
            f"got device {device}"
        )

    def _train_sampling_generator(
        self, device: torch.device
    ) -> Optional[torch.Generator]:
        """Return a private CPU/CUDA RNG; other backends retain their global RNG fallback."""
        if device.type not in {"cpu", "cuda"}:
            # PyTorch does not expose torch.Generator for every accelerator backend (notably MPS
            # in supported releases). Passing None keeps the former device-global RNG behavior
            # instead of making otherwise-supported stochastic training fail.
            return None
        generator = getattr(self, "_presyn_train_generator", None)
        if generator is None or generator.device != device:
            generator = torch.Generator(device=device)
            state = self._train_rng_state_buffer(device)
            if state.numel() > 0:
                generator.set_state(state.detach().cpu())
            else:
                seed = int(self._presyn_train_rng_seed.item())
                if seed < 0:
                    seed = int(
                        torch.randint(
                            0, torch.iinfo(torch.int64).max, (), device="cpu"
                        ).item()
                    )
                    with torch.no_grad():
                        self._presyn_train_rng_seed.fill_(seed)
                generator.manual_seed(seed)
            self._presyn_train_generator = generator
        return generator

    @torch.no_grad()
    def _commit_train_sampling_generator(self, generator: torch.Generator) -> None:
        """Persist the private train RNG in the matching backend's checkpoint blob."""
        state_buffer = self._train_rng_state_buffer(generator.device)
        generator_state = generator.get_state().to(device=state_buffer.device)
        state_buffer.resize_(generator_state.shape)
        state_buffer.copy_(generator_state)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        # Opaque generator states have backend- and PyTorch-version-dependent lengths. Resize the
        # registered destinations before the normal strict loader copies (or assigns) them.
        for name in ("_presyn_train_cpu_rng_state", "_presyn_train_cuda_rng_state"):
            incoming = state_dict.get(prefix + name)
            current = getattr(self, name)
            if torch.is_tensor(incoming) and incoming.shape != current.shape:
                with torch.no_grad():
                    current.resize_(incoming.shape)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if hasattr(self, "_presyn_train_generator"):
            delattr(self, "_presyn_train_generator")

    def use_metriplectic_integrator(self) -> bool:
        """Dispatch predicate (0642.1.2.4): advance the calcium/buffer subsystem with the
        structure-preserving discrete-gradient integrator
        (``bio_inspired_nanochat.metriplectic_integrator``) instead of the clamped-Euler step, so
        energy is conserved and the free energy is Lyapunov at the discrete level. DEFAULT-OFF; the
        integration call site is wired in 0642.1.2 — this is the toggle read."""
        return bool(self.cfg.metriplectic_integrator)

    @staticmethod
    def _reduce_metriplectic_step(
        record: TorchStepRecord, active_mask: Optional[Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Reduce guard evidence over materialized state slots only."""
        if active_mask is None:
            metric_mask = torch.ones_like(record.fallback_mask, dtype=torch.bool)
        else:
            metric_mask = active_mask.expand_as(record.fallback_mask)
        steps = metric_mask.sum(dtype=torch.int64)
        fallbacks = (record.fallback_mask & metric_mask).sum(dtype=torch.int64)
        energy_drift = torch.where(
            metric_mask,
            record.energy_drift.detach().abs(),
            torch.zeros_like(record.energy_drift),
        ).max()
        entropy_production = torch.where(
            metric_mask,
            record.entropy_production.detach(),
            torch.full_like(record.entropy_production, float("inf")),
        ).min()
        free_energy_delta = torch.where(
            metric_mask,
            record.free_energy_delta.detach(),
            torch.full_like(record.free_energy_delta, float("-inf")),
        ).max()
        return steps, fallbacks, energy_drift, entropy_production, free_energy_delta

    @torch.no_grad()
    def _record_metriplectic_step(
        self, record: TorchStepRecord, active_mask: Optional[Tensor] = None
    ) -> None:
        """Retain compact live guard/ledger evidence without holding the autograd graph."""
        steps, fallbacks, energy_drift, entropy_production, free_energy_delta = (
            self._reduce_metriplectic_step(record, active_mask)
        )
        self.metriplectic_steps.add_(steps)
        self.metriplectic_fallbacks.add_(fallbacks)
        self.metriplectic_last_energy_drift.copy_(energy_drift.to(torch.float32))
        self.metriplectic_last_entropy_production.copy_(
            entropy_production.to(torch.float32)
        )
        self.metriplectic_last_free_energy_delta.copy_(
            free_energy_delta.to(torch.float32)
        )

    def get_metriplectic_metrics(self) -> Dict[str, float | int]:
        """Return the live conservation/entropy ledger for structured telemetry."""
        return {
            "steps": int(self.metriplectic_steps.item()),
            "fallbacks": int(self.metriplectic_fallbacks.item()),
            "last_max_energy_drift": float(self.metriplectic_last_energy_drift.item()),
            "last_min_entropy_production": float(
                self.metriplectic_last_entropy_production.item()
            ),
            "last_max_free_energy_delta": float(
                self.metriplectic_last_free_energy_delta.item()
            ),
        }

    def _advance_calcium_buffer(
        self,
        calcium: Tensor,
        buffer: Tensor,
        heat: Tensor,
        influx: Tensor,
        *,
        rho_c: float | Tensor,
        rho_b: float | Tensor,
        alpha_buf_on: float | Tensor,
        alpha_buf_off: float | Tensor,
        record_metrics: bool = False,
        runtime_buffers: Optional[Dict[str, Tensor]] = None,
        metric_active_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Advance the live calcium/buffer subsystem or its exact guarded fallback.

        The external attention-driven influx is injected into calcium before the closed-system
        metriplectic relaxation. If any structural/numerical/domain guard trips, the pre-existing
        clamped-Euler calcium/buffer update is selected elementwise, byte-for-byte.
        """
        baseline_c = (
            rho_c * calcium
            + influx
            - alpha_buf_on * calcium * (1.0 - buffer)
            + alpha_buf_off * buffer
        ).clamp(min=0.0)
        baseline_b = (
            rho_b * buffer
            + alpha_buf_on * calcium * (1.0 - buffer)
            - alpha_buf_off * buffer
        ).clamp(0.0, 1.0)
        if not self.use_metriplectic_integrator():
            return baseline_c, baseline_b, heat

        # The Poisson orientation is chosen so positive free calcium flows into the positive live
        # buffer coordinate. Dissipative rates are the discrete leaks identified in the theory note.
        omega = -0.5 * (alpha_buf_on + alpha_buf_off)
        c_next, b_next, h_next, record = torch_guarded_step(
            calcium + influx,
            buffer,
            heat,
            omega=omega,
            gC=1.0 - rho_c,
            gB=1.0 - rho_b,
            fallback=(baseline_c, baseline_b, heat),
        )
        if record_metrics:
            if runtime_buffers is None:
                self._record_metriplectic_step(record, metric_active_mask)
            else:
                steps, fallbacks, energy_drift, entropy_production, free_energy_delta = (
                    self._reduce_metriplectic_step(record, metric_active_mask)
                )
                runtime_buffers["metriplectic_steps"] = runtime_buffers[
                    "metriplectic_steps"
                ] + steps
                runtime_buffers["metriplectic_fallbacks"] = runtime_buffers[
                    "metriplectic_fallbacks"
                ] + fallbacks
                runtime_buffers["metriplectic_last_energy_drift"] = (
                    energy_drift.to(torch.float32)
                )
                runtime_buffers["metriplectic_last_entropy_production"] = (
                    entropy_production.to(torch.float32)
                )
                runtime_buffers["metriplectic_last_free_energy_delta"] = (
                    free_energy_delta.to(torch.float32)
                )
        return c_next, b_next, h_next

    # The legacy sigmoid release() + _mix_prob were removed here (qcj7). The faithful canonical
    # release lives in release_canonical (below) and is used by ALL paths now (standard + flex);
    # the @no_grad full-attention reference forward() remains for kernel parity + visualization.

    def _faithful_release_prob(
        self, c_edge: Tensor, pr_edge: Tensor, cl_edge: Tensor, drive: Tensor
    ) -> Tensor:
        """Canonical per-edge release PROBABILITY in [0,1] (8j9.2).

        The differentiable equivalent of forward()'s faithful release math: Hill-function
        calcium sensing (Syt1 fast + Syt7 slow), complexin/SNARE gating, and the q.k bilinear
        term. The Doc2 facilitation term is PRESERVED from release()'s sigmoid mix (forward()
        lacks it) so no feature is lost. Replaces release()'s sigmoid `_mix_prob`.
        """
        cfg = self.cfg
        fast = c_edge / (c_edge + cfg.syt_fast_kd)
        slow = c_edge / (c_edge + cfg.syt_slow_kd)
        syt = (
            0.7 * fast
            + 0.3 * slow
            + cfg.doc2_gain * torch.sigmoid(4.0 * (c_edge - 0.12))  # Doc2 facilitation (preserved)
        )
        fuse_base = torch.sigmoid(3.0 * syt + 2.0 * pr_edge - 2.0 * (cl_edge + cfg.complexin_bias))
        d_bilin = torch.sigmoid(drive)  # drive == q.k/sqrt(d), the top-k attention logit
        return (fuse_base * d_bilin).clamp(0.0, 1.0)

    def _can_use_native_presyn_decode(
        self,
        state: Dict[str, Any],
        drive: Tensor,
        *,
        train: bool,
        differentiable: bool,
        apply_barrier: bool,
        q_pos: Optional[Tensor],
    ) -> bool:
        """Whether jyb.2's exact one-kernel deterministic decode slice is legal.

        The returned release bias remains differentiable with respect to ``drive`` even when the
        recurrent state update is detached, so *every* grad-enabled call must stay on Python until
        jyb.3 adds a backward kernel. General prefill also stays on Python because duplicate-key
        reductions across queries require a second launch before nonlinear state finalization.
        """
        return bool(
            self.cfg.enable_presyn
            and self.cfg.native_presyn
            and drive.is_cuda
            and drive.dtype == torch.float32
            and drive.ndim == 4
            and drive.shape[2] == 1
            and drive.shape[3] == self.cfg.attn_topk
            and not torch.is_grad_enabled()
            and not train
            and not differentiable
            and not self.cfg.differentiable_recurrence
            and not self.cfg.use_flex_attention
            and not bool(getattr(self, "_mc_sampling", False))
            and self.kinetics is None
            and not self.use_metriplectic_integrator()
            and "HEAT" not in state
            and not apply_barrier
            and q_pos is None
        )

    def release_canonical(
        self,
        state: Dict[str, Any],
        drive: Tensor,
        idx: Tensor,
        train: bool,
        valid: Optional[Tensor] = None,
        q_pos: Optional[Tensor] = None,
        apply_barrier: bool = False,
        differentiable: bool = False,
        logits: Optional[Tensor] = None,
        runtime_buffers: Optional[Dict[str, Tensor]] = None,
        active_key_count: Optional[int] = None,
    ) -> Tensor:
        """CANONICAL unified presynaptic release — the single, faithful, differentiable
        source of truth (8j9.2).

        Ports forward()'s biologically-faithful equations — Hill Syt(C)=C/(C+Kd), the calcium
        BUFFER ODE (BUF, which release() ignored), energy->AMPA `qamp`, and the septin distance
        barrier — onto release()'s top-k, key-indexed, differentiable scatter structure.

        Differentiability scope: the RETURNED bias is always differentiable w.r.t. the INPUT
        `drive` (parity with what release() feeds into the attention logits). The STATE RECURRENCE
        is DETACHED BY DEFAULT (8j9.2 scope boundary); pass ``differentiable=True`` (yw9.2) to run
        the same state update under autograd so the advanced state carries gradient w.r.t. this
        step's inputs/params — the byte-identical forward value, enabling BPTT through
        calcium/RRP/energy (yw9.2.3 chunked TBPTT; yw9.3 learnable kinetics). Preserves the
        stochastic STE path, the endocytosis DELAY queue, Doc2, and EMA normalization. AMP is
        carried but superseded by energy->qamp (the faithful amplitude); the vestigial AMP
        dynamics are removed in the param-unify step.

        drive: (B,H,T,K) top-k attention logits; idx: (B,H,T,K) selected key indices.
        active_key_count: optional number of key positions that exist at this causal step. State
        slots at or beyond this position remain exactly unchanged; this prevents a contiguous
        forward from passively ageing preallocated future-token state.
        apply_barrier: fold the septin distance barrier into e (default False; the live attention
        path applies its own exact logit-level barrier, so it must stay False there to avoid
        double-counting). q_pos: optional (T,) absolute query positions for that barrier; defaults
        to arange(T) (full-sequence). Returns per-edge release e (B,H,T,K) consumed as
        lambda_loge*log(eps+e).
        """
        if not self.cfg.enable_presyn:
            return torch.ones_like(drive)

        native_decode = logits is not None and self._can_use_native_presyn_decode(
            state,
            drive,
            train=train,
            differentiable=differentiable,
            apply_barrier=apply_barrier,
            q_pos=q_pos,
        )
        if native_decode:
            from bio_inspired_nanochat.kernels.presyn_fused import (
                presyn_live_decode_step,
            )

            return presyn_live_decode_step(
                state,
                drive,
                idx,
                self.cfg,
                ema_e=self.ema_e,
                valid=valid,
                logits=logits,
            )
        if logits is not None:
            raise RuntimeError(
                "in-kernel logit augmentation was requested outside the supported native "
                "deterministic decode path"
            )

        cfg = self.cfg
        B, H, T, K = drive.shape
        if valid is not None and valid.shape != drive.shape:
            raise ValueError(
                f"valid mask must match drive shape {drive.shape}, got {valid.shape}"
            )
        dtype = state["C"].dtype
        state_key_count = int(state["C"].shape[2])
        if active_key_count is not None:
            if active_key_count < 1 or active_key_count > state_key_count:
                raise ValueError(
                    "active_key_count must be between 1 and the state key extent "
                    f"({state_key_count}), got {active_key_count}"
                )
            active_keys = (
                torch.arange(state_key_count, device=state["C"].device)
                .view(1, 1, state_key_count)
                .lt(active_key_count)
            )
        else:
            active_keys = None
        heat_state = state.get("HEAT")
        if self.use_metriplectic_integrator() and heat_state is None:
            # Compatibility for caches created before 0642.1.2. Heat is a default-off state and
            # therefore absent from ordinary checkpoints; an enabled old cache starts on h=0.
            heat_state = torch.zeros_like(state["C"])
            state["HEAT"] = heat_state
        flat_idx = idx.reshape(B, H, -1)

        # yw9.3: source the calcium/buffer kinetics from learnable Parameters when enabled, else
        # the hand-tuned cfg constants. The learnable values are stability-preserving by
        # construction (decays via sigmoid∈(0,1), gains via softplus, buffer-coupling bounded) and
        # initialized to match cfg exactly, so a fresh learnable module reproduces this forward.
        if self.kinetics is not None:
            rho_c, rho_b = self.kinetics.rho_c, self.kinetics.rho_b
            alpha_ca = self.kinetics.alpha_ca
            alpha_buf_on, alpha_buf_off = self.kinetics.alpha_buf_on, self.kinetics.alpha_buf_off
        else:
            rho_c = math.exp(-1.0 / cfg.tau_c)   # calcium decay time-constant (unified across paths)
            rho_b = math.exp(-1.0 / cfg.tau_buf)  # buffer decay
            alpha_ca = cfg.alpha_ca
            alpha_buf_on, alpha_buf_off = cfg.alpha_buf_on, cfg.alpha_buf_off

        # --- gather per-edge state for the selected keys (prior state is detached) ---
        c_prev = state["C"].gather(2, flat_idx).view(B, H, T, K)
        buf_prev = state["BUF"].gather(2, flat_idx).view(B, H, T, K)
        if heat_state is not None:
            heat_prev = heat_state.gather(2, flat_idx).view(B, H, T, K)
        pr_edge = state["PR"].gather(2, flat_idx).view(B, H, T, K)
        cl_edge = state["CL"].gather(2, flat_idx).view(B, H, T, K)
        rrp_edge = state["RRP"].gather(2, flat_idx).view(B, H, T, K)
        e_energy = state["E"].gather(2, flat_idx).view(B, H, T, K)

        # --- calcium + buffer ODE (BUF now ACTIVE; influx carries the grad w.r.t. drive) ---
        influx = alpha_ca * F.softplus(drive)
        if heat_state is None:
            # Preserve the default path's exact operation sequence and allocation profile.
            c_edge = (
                rho_c * c_prev
                + influx
                - alpha_buf_on * c_prev * (1.0 - buf_prev)
                + alpha_buf_off * buf_prev
            ).clamp(min=0.0)
        else:
            c_edge, _, _ = self._advance_calcium_buffer(
                c_prev,
                buf_prev,
                heat_prev,
                influx,
                rho_c=rho_c,
                rho_b=rho_b,
                alpha_buf_on=alpha_buf_on,
                alpha_buf_off=alpha_buf_off,
            )

        # --- faithful Hill release probability, then release = p * available RRP (<= RRP) ---
        p = self._faithful_release_prob(c_edge, pr_edge, cl_edge, drive)
        # hy8.1: acetylcholine (uncertainty/attention) gates exploration via the stochastic-
        # release fraction. Default-neutral (gain 1.0) unless a NeuromodulatoryBus broadcasts a
        # gain; higher ACh => more stochastic vesicle release => more exploration. Clamped [0,1].
        ach_frac = min(1.0, max(0.0, cfg.stochastic_train_frac * getattr(self, "_nm_ach_gain", 1.0)))
        # u2t.1: MC ensembling samples the stochastic release at INFERENCE too. When `_mc_sampling`
        # is set (by mc_ensemble), every query position releases stochastically (fraction `_mc_frac`,
        # default 1.0), so each forward pass is an independent draw from the predictive distribution.
        mc_sampling = bool(getattr(self, "_mc_sampling", False))
        if mc_sampling:
            ach_frac = min(1.0, max(0.0, float(getattr(self, "_mc_frac", 1.0))))
        evidence_sink = getattr(self, "_mc_evidence_sink", None) if mc_sampling else None
        mc_generator = getattr(self, "_mc_generator", None) if mc_sampling else None
        sample_pool_sizes: Optional[Tensor] = None
        sampled_mask = (
            torch.zeros_like(p, dtype=torch.bool) if evidence_sink is not None else None
        )
        train_generator: Optional[torch.Generator] = None
        if (train or mc_sampling) and ach_frac > 0:
            sampling_generator = (
                mc_generator
                if mc_sampling
                else self._train_sampling_generator(p.device)
            )
            if train and not mc_sampling and sampling_generator is not None:
                train_generator = sampling_generator
            sample_pool_sizes = torch.clamp(
                rrp_edge.round(), 0.0, float(cfg.stochastic_count_cap)
            )
            do_stoch = (
                torch.rand(
                    p[..., 0].shape,
                    device=p.device,
                    dtype=torch.float32,
                    generator=sampling_generator,
                )
                < float(ach_frac)
            )
            rel_det = p * rrp_edge
            stoch_mask = do_stoch.unsqueeze(-1).expand_as(p)
            if valid is not None:
                stoch_mask = stoch_mask & valid
            if cfg.stochastic_mode == "normal_reparam":
                # The normal estimator is cheap enough to evaluate densely. Avoiding boolean
                # gather/scatter here removes a CPU synchronization point and dozens of tiny
                # indexing kernels from every causal query; ``where`` keeps non-sampled edges on
                # their deterministic release value. Drawing one fixed-shape noise tensor per
                # query also remains invariant to recurrence grouping.
                if stoch_mask.any():
                    if sampled_mask is not None:
                        sampled_mask = stoch_mask
                    k_rel = _sample_binomial_counts(
                        probs=p,
                        total_count=sample_pool_sizes,
                        max_count=int(cfg.stochastic_count_cap),
                        tau=float(cfg.stochastic_tau),
                        mode=cfg.stochastic_mode,
                        generator=sampling_generator,
                        normal_draw_width=int(cfg.attn_topk),
                    )
                    rel = torch.where(stoch_mask, k_rel, rel_det)
                else:
                    rel = rel_det
            elif stoch_mask.any():
                if sampled_mask is not None:
                    sampled_mask = stoch_mask
                k_rel = _sample_binomial_counts(
                    probs=p[stoch_mask],
                    total_count=sample_pool_sizes[stoch_mask],
                    max_count=int(cfg.stochastic_count_cap),
                    tau=float(cfg.stochastic_tau),
                    mode=cfg.stochastic_mode,
                    generator=sampling_generator,
                )
                rel = rel_det.clone()
                rel[stoch_mask] = k_rel
            else:
                rel = rel_det
        else:
            rel = p * rrp_edge
        if train_generator is not None:
            self._commit_train_sampling_generator(train_generator)
        if valid is not None:
            rel = rel * valid.to(rel.dtype)
        if evidence_sink is not None:
            if sample_pool_sizes is None:
                sample_pool_sizes = torch.clamp(
                    rrp_edge.round(), 0.0, float(cfg.stochastic_count_cap)
                )
            evidence_sink.record(
                layer_address=str(getattr(self, "_mc_evidence_address", "")),
                probabilities=p,
                pool_sizes=sample_pool_sizes,
                forward_counts=rel,
                reverse_probability=float(cfg.rec_rate),
                sampling_mode=cfg.stochastic_mode,
                valid=valid,
                sampled=sampled_mask,
            )

        # --- energy-derived AMPA amplitude (faithful) ---
        qamp = torch.sigmoid(cfg.q_beta * (e_energy - 0.5)) * cfg.qmax

        e = rel * qamp

        # --- optional septin distance barrier (opt-in). The LIVE attention path applies its own
        # exact logit-level barrier with global query/key positions (and the correct prefix
        # offset), so it leaves apply_barrier=False here to avoid DOUBLE-counting. Standalone /
        # golden use (where there is no outer barrier) can set apply_barrier=True for a
        # self-contained faithful output. Normalize each query by the key extent causally
        # available at that query, so appending future keys cannot rescale an earlier bias. ---
        if apply_barrier and cfg.barrier_strength > 0.0:
            if q_pos is None:
                qpos = torch.arange(T, device=drive.device, dtype=torch.float32)
            else:
                qpos = q_pos.to(device=drive.device, dtype=torch.float32)
            causal_extent = (qpos + 1.0).clamp_min(1.0).reshape(1, 1, T, 1)
            dist = (
                qpos.reshape(1, 1, T, 1) - idx.to(torch.float32)
            ).abs() / causal_extent
            e = e * torch.exp(-cfg.barrier_strength * dist).to(e.dtype)

        # === scatter faithful state updates back to key positions ===
        # The state recurrence is DETACHED by default (parity with 8j9.2). yw9.2: when
        # ``differentiable`` is set, the SAME math runs under grad (no detach) so the advanced
        # state carries gradient w.r.t. this step's inputs/params — enabling BPTT through
        # calcium/RRP/energy across a chain of calls (yw9.2.3 chunked TBPTT). The forward VALUE is
        # byte-identical in both modes (only gradient tracking differs); the returned bias ``e`` is
        # already differentiable w.r.t. ``drive`` regardless, so this flag affects only the state.
        state_ctx = contextlib.nullcontext() if differentiable else torch.no_grad()
        with state_ctx:
            flat_rel = (rel if differentiable else rel.detach()).reshape(B, H, -1).to(dtype)
            flat_drive = (drive if differentiable else drive.detach()).reshape(B, H, -1).to(dtype)
            if valid is not None:
                flat_valid_bool = valid.reshape(B, H, -1)
                flat_valid = flat_valid_bool.to(dtype)
                flat_drive = torch.where(
                    flat_valid_bool, flat_drive, torch.zeros_like(flat_drive)
                )
            else:
                flat_valid = torch.ones_like(flat_rel)

            add_vals = torch.zeros_like(state["C"])
            drv_vals = torch.zeros_like(state["C"])
            cnt_vals = torch.zeros_like(state["C"])
            add_vals.scatter_add_(2, flat_idx, flat_rel)    # vesicles released per key
            drv_vals.scatter_add_(2, flat_idx, flat_drive)  # accumulated drive per key
            cnt_vals.scatter_add_(2, flat_idx, flat_valid)  # access count per key
            accessed = (cnt_vals > 0).to(dtype)

            # Calcium + buffer at key positions. The default remains the faithful clamped-Euler
            # ODE; the opt-in path advances the same driven state with the guarded discrete-gradient
            # core and persists its heat/entropy ledger.
            c_k, buf_k = state["C"], state["BUF"]
            heat_k = state["HEAT"] if self.use_metriplectic_integrator() else c_k
            c_up, buf_up, heat_up = self._advance_calcium_buffer(
                c_k,
                buf_k,
                heat_k,
                alpha_ca * F.softplus(drv_vals) * accessed,
                rho_c=rho_c,
                rho_b=rho_b,
                alpha_buf_on=alpha_buf_on,
                alpha_buf_off=alpha_buf_off,
                record_metrics=True,
                runtime_buffers=runtime_buffers,
                metric_active_mask=active_keys,
            )

            # RRP depletion + endocytosis delay queue + priming refill
            rrp_up = torch.clamp(state["RRP"] - add_vals, 0)
            if cfg.endo_delay > 0:
                res_up = state["RES"] + state["DELAY"][0]
                new_delay = state["DELAY"][1:] + [add_vals * cfg.rec_rate]
            else:
                res_up = state["RES"]
                new_delay = []
            take = torch.minimum(res_up, torch.ones_like(res_up))
            res_up = torch.clamp(res_up - cfg.prime_rate * take, 0)
            rrp_up = torch.clamp(rrp_up + cfg.prime_rate * take, 0, 30.0)

            # SNARE recovery / complexin clamp relaxation / energy metabolism
            sn_up = torch.clamp(
                state["PR"] * (1.0 - cfg.unprime_per_release * add_vals)
                + cfg.nsf_recover * (1.0 - state["PR"]),
                0, 1,
            )
            cl_up = torch.clamp(
                state["CL"] * 0.995 + 0.005 - cfg.unprime_per_release * add_vals, 0, 1
            )
            en_up = torch.clamp(
                state["E"]
                + cfg.energy_fill * (cfg.energy_max - state["E"])
                - cfg.energy_use * add_vals,
                0, cfg.energy_max,
            )

            if active_keys is not None:
                c_up = torch.where(active_keys, c_up, c_k)
                buf_up = torch.where(active_keys, buf_up, buf_k)
                rrp_up = torch.where(active_keys, rrp_up, state["RRP"])
                res_up = torch.where(active_keys, res_up, state["RES"])
                sn_up = torch.where(active_keys, sn_up, state["PR"])
                cl_up = torch.where(active_keys, cl_up, state["CL"])
                en_up = torch.where(active_keys, en_up, state["E"])
                new_delay = [
                    torch.where(active_keys, candidate, previous)
                    for candidate, previous in zip(new_delay, state["DELAY"])
                ]
                if self.use_metriplectic_integrator():
                    heat_up = torch.where(active_keys, heat_up, state["HEAT"])

            next_state = {
                "C": c_up, "BUF": buf_up, "RRP": rrp_up, "RES": res_up, "DELAY": new_delay,
                "PR": sn_up, "CL": cl_up, "AMP": state["AMP"], "E": en_up,
            }
            if self.use_metriplectic_integrator():
                next_state["HEAT"] = heat_up
            state.update(next_state)

            # EMA normalization (parity with release()). 9mxi: the running normalizer is
            # PERSISTENT module state, so only adapt it when adaptation is allowed
            # (train=True). Eval forwards (train=False) must not mutate any persistent
            # state, or val_bpb is neither idempotent nor contamination-free.
            if valid is None:
                s = e.detach().abs().mean()
            else:
                valid_weight = valid.to(e.dtype)
                s = (e.detach().abs() * valid_weight).sum() / valid_weight.sum().clamp_min(1)
            s = s.clamp_min(1e-3)
            if runtime_buffers is not None:
                ema_e = runtime_buffers["ema_e"]
                if train:
                    ema_e = ema_e.clone()
                    ema_e.mul_(0.99).add_(0.01 * s)
                    runtime_buffers["ema_e"] = ema_e
            else:
                ema_e = self.ema_e
            if train and runtime_buffers is None:
                self.ema_e.mul_(0.99).add_(0.01 * s)

        return e / (ema_e + 1e-6)

    @torch.no_grad()
    def forward(
        self,
        q: Tensor,
        k: Tensor,
        logits: Tensor,
        state: Dict[str, Tensor],
        mask: Optional[Tensor] = None,
        train_mode: bool = False,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Full (B, H, T, T) presynaptic dynamics reference (sequential, causal).

        This is used for visualization and full-sequence reference tests. The canonical
        Rust/Triton backend contract is the sparse Tq=1 `release_canonical` path; this
        method intentionally retains a distinct sequential execution model.
        """
        B, H, T, D = q.shape
        cfg = self.cfg

        # State (clone so we can write in-place without mutating the caller).
        c = state["C"].clone()
        buf = state.get("BUF", torch.zeros_like(c)).clone()
        rrp = state["RRP"].clone()
        res = state["RES"].clone()
        pr = state["PR"].clone()
        cl = state["CL"].clone()
        e_st = state["E"].clone()

        rho_c = math.exp(-1.0 / cfg.tau_c)
        rho_b = math.exp(-1.0 / cfg.tau_buf)
        rho_p = math.exp(-1.0 / cfg.tau_prime)
        rho_r = math.exp(-1.0 / cfg.tau_rrp)
        rho_e = math.exp(-1.0 / cfg.tau_energy)
        sqrt_d = math.sqrt(D)

        syn_logit = torch.zeros_like(logits)

        for t in range(T):
            # 1) Calcium influx from incoming drive (mean softplus over causal keys)
            log_t = logits[:, :, t, : t + 1]
            if mask is not None:
                log_t = log_t.masked_fill(
                    ~mask[t, : t + 1].view(1, 1, -1), -20.0
                )
            drive = F.softplus(log_t.clamp(-20.0, 20.0))
            influx = drive.sum(dim=-1) / float(t + 1)

            # 2) Calcium + buffer update
            c_prev = c[:, :, t]
            buf_prev = buf[:, :, t]

            c_next = (
                rho_c * c_prev
                + cfg.alpha_ca * influx
                - cfg.alpha_buf_on * c_prev * (1.0 - buf_prev)
                + cfg.alpha_buf_off * buf_prev
            ).clamp(min=0.0)
            buf_next = (
                rho_b * buf_prev
                + cfg.alpha_buf_on * c_prev * (1.0 - buf_prev)
                - cfg.alpha_buf_off * buf_prev
            ).clamp(0.0, 1.0)

            # 3) Mid-state (priming, refill, energy)
            pr_val = pr[:, :, t]
            rrp_val = rrp[:, :, t]
            res_val = res[:, :, t]
            e_val = e_st[:, :, t]

            pr_mid = (rho_p * pr_val + cfg.alpha_prime * (1.0 - pr_val)).clamp(0.0, 1.0)
            rrp_refill = (rho_r * rrp_val + cfg.alpha_refill * res_val).clamp(0.0, 1.0)
            res_mid = (res_val - cfg.alpha_refill * res_val).clamp(0.0, 1.0)
            e_mid = (rho_e * e_val + cfg.energy_in).clamp(0.0, 1.6)

            # 4) Release computation
            fast = c_next / (c_next + cfg.syt_fast_kd)
            slow = c_next / (c_next + cfg.syt_slow_kd)
            syt = 0.7 * fast + 0.3 * slow

            cl_val = cl[:, :, t]
            fuse_base = torch.sigmoid(
                3.0 * syt + 2.0 * pr_mid - 2.0 * (cl_val + cfg.complexin_bias)
            )  # (B, H)

            q_t = q[:, :, t, :]  # (B, H, D)
            k_j = k[:, :, : t + 1, :]  # (B, H, t+1, D)
            dot = torch.einsum("bhd,bhjd->bhj", q_t, k_j) / sqrt_d
            d_bilin = torch.sigmoid(dot)

            rr = (fuse_base.unsqueeze(-1) * d_bilin * rrp_refill.unsqueeze(-1)).clamp(
                0.0, 1.0
            )
            row_sum = rr.sum(dim=-1)  # (B, H)
            scale = torch.ones_like(row_sum)
            m = row_sum > cfg.epsilon
            scale[m] = (rrp_refill[m] / row_sum[m]).clamp(max=1.0)

            rel = rr * scale.unsqueeze(-1)  # (B, H, t+1)
            used = rel.sum(dim=-1)  # (B, H)

            # 5) Final state
            rrp_n = (rrp_refill - used).clamp(0.0, 1.0)
            res_n = (res_mid + used).clamp(0.0, 1.0)
            pr_n = (pr_mid - cfg.alpha_unprime * used).clamp(0.0, 1.0)
            e_n = (
                e_mid
                - cfg.energy_cost_rel * used
                - cfg.energy_cost_pump * (1.0 - res_n)
            ).clamp(0.0, 1.6)

            qamp = torch.sigmoid(cfg.q_beta * (e_n - 0.5)) * cfg.qmax  # (B, H)

            # 6) Logit adjustment (write row t)
            j = torch.arange(t + 1, device=q.device, dtype=torch.float32)
            dist = (float(t) - j).abs() / float(max(1, T))
            val = (rel * qamp.unsqueeze(-1)).clamp(min=cfg.epsilon).log() - (
                cfg.barrier_strength * dist.view(1, 1, -1).to(rel.dtype)
            )
            syn_logit[:, :, t, : t + 1] = val
            syn_logit[:, :, t, t + 1 :] = math.log(cfg.epsilon)

            # Store updated state at index t
            c[:, :, t] = c_next
            buf[:, :, t] = buf_next
            rrp[:, :, t] = rrp_n
            res[:, :, t] = res_n
            pr[:, :, t] = pr_n
            e_st[:, :, t] = e_n

        new_state = {"C": c, "BUF": buf, "RRP": rrp, "RES": res, "PR": pr, "CL": cl, "E": e_st}
        return syn_logit, new_state


# -----------------------------------------------------------------------------
# Postsynaptic eligibility and linear
# -----------------------------------------------------------------------------


class PostsynapticHebb(nn.Module):
    cfg: SynapticConfig
    fast: nn.Parameter
    slow: nn.Parameter
    U: nn.Parameter
    V: nn.Parameter
    camkii: Tensor
    pp1: Tensor
    bdnf: Tensor
    bdnf_hebb_accum: Tensor
    _last_hebb_delta_mag: Tensor
    """Low-rank eligibility + CaMKII/PP1/BDNF gate controlling consolidation.

    BDNF Metaplasticity (bio_inspired_nanochat-711):
    - B(t) accumulator tracks |ΔW_hebb| (Hebbian delta magnitude) with decay
    - When bdnf_gamma > 0, slow LR is modulated by (1 + gamma * B)
    - This implements activity-dependent learning rate scaling
    """

    def __init__(self, d_k: int, d_v: int, cfg: SynapticConfig):
        super().__init__()
        object.__setattr__(self, "cfg", cfg)
        R = cfg.rank_eligibility
        self.fast = nn.Parameter(torch.zeros(d_v))
        self.slow = nn.Parameter(torch.zeros(d_v))
        self.U = nn.Parameter(torch.zeros(d_v, R))
        self.V = nn.Parameter(torch.zeros(R, d_v))

        self.register_buffer("camkii", torch.zeros(d_v))
        self.register_buffer("pp1", torch.ones(d_v) * 0.5)
        self.register_buffer("bdnf", torch.zeros(d_v))
        # B(t) accumulator for |ΔW_hebb| - used when bdnf_hebb_accumulate=True
        self.register_buffer("bdnf_hebb_accum", torch.zeros(d_v))
        # Track last delta for logging/debugging
        self.register_buffer("_last_hebb_delta_mag", torch.zeros(1))

        nn.init.normal_(self.U, std=0.02)
        nn.init.normal_(self.V, std=0.02)

        # 0642.2.2.1: certified cusp-normal-form latch. Built only when opted in; falls back to the
        # heuristic sax.2 map below when the retention certificate is void (fail-closed, §5 of
        # docs/theory/singular_perturbation.md). Lazy import avoids a cusp_certificate↔synaptic cycle.
        self._cusp_latch = None
        if cfg.bistable_latch and cfg.cusp_latch:
            from bio_inspired_nanochat.cusp_certificate import CuspLatch
            self._cusp_latch = CuspLatch(cfg)

    def forward(self, v: Tensor) -> Tensor:
        diag = 1.0 + self.fast + self.slow
        return v * diag + v @ (self.U @ self.V)

    @torch.no_grad()
    def update(self, y: Tensor, ca_proxy: Tensor, *, genes: Optional[Tensor] = None) -> None:
        """Update CaMKII, PP1, and BDNF state based on activity.

        When bdnf_hebb_accumulate=False (legacy mode):
            BDNF accumulates based on CaMKII activity: F.relu(camkii - 0.5)

        When bdnf_hebb_accumulate=True (new mode, bio_inspired_nanochat-711):
            BDNF accumulates |ΔW_hebb| via bdnf_hebb_accum buffer (updated in consolidate())
            The main bdnf buffer then tracks this with decay.
        """
        cfg = self.cfg
        if cfg.bistable_latch and self._cusp_latch is not None and self._cusp_latch.certified:
            # 0642.2.2.1: certified cusp-normal-form latch. m evolves by one gradient step of the
            # cusp cubic m̃³+a·m̃+b(c) with the certified splitting parameter a and the live calcium
            # bias b(c); PP1 is slaved to its reduced quasi-steady value. δ*(a) is then the *tight*
            # retention half-width. Genes modulate the operating bias (γ, β_pp1), not the certified a.
            gamma_scale = beta_pp1_scale = None
            if genes is not None and genes.numel() >= 4:
                gamma_scale = genes[..., 2].clamp(min=0.0)
                beta_pp1_scale = genes[..., 3].clamp(min=0.0)
            m_new, p_new = self._cusp_latch.step(
                self.camkii, ca_proxy, gamma_scale=gamma_scale, beta_pp1_scale=beta_pp1_scale
            )
            self.camkii.copy_(m_new)
            self.pp1.copy_(p_new)
        elif cfg.bistable_latch:
            # Lisman-style bistable switch (sax.2): CaMKII self-excitation + mutual
            # cross-inhibition with PP1 over a basal phosphatase floor. Calcium maps to
            # LTP/LTD drives via a BCM curve (LTP above camkii_thr, LTD below latch_ltd_thr).
            gain = cfg.latch_input_gain
            drive = torch.sigmoid(gain * (ca_proxy - cfg.camkii_thr))
            erase = torch.sigmoid(gain * (cfg.latch_ltd_thr - ca_proxy))
            gamma = cfg.latch_gamma_auto      # CaMKII autophosphorylation gain
            beta_pp1 = cfg.latch_beta_pp1     # PP1 de-potentiation strength
            if genes is not None and genes.numel() >= 4:
                gamma = (genes[..., 2] * gamma).clamp(min=0.0)
                beta_pp1 = (genes[..., 3] * beta_pp1).clamp(min=0.0)
            m, p = self.camkii, self.pp1
            n, k = cfg.latch_hill_n, cfg.latch_hill_k
            mn = m.pow(n)
            hill = mn / (k**n + mn + 1e-12)
            m_new = (
                m + cfg.latch_alpha_ca * drive * (1 - m) - beta_pp1 * p * m + gamma * hill
            ).clamp(0.0, 1.0)
            p_new = (
                p + cfg.latch_alpha_pp1 * erase * (1 - p) - cfg.latch_beta_camkii * m * p
            ).clamp(cfg.latch_pp1_basal, 1.0)
            self.camkii.copy_(m_new)
            self.pp1.copy_(p_new)
        else:
            up = (ca_proxy > cfg.camkii_thr).float()
            down = (ca_proxy < cfg.pp1_thr).float()

            camkii_up = cfg.camkii_up
            pp1_rate = 1.0 - cfg.pp1_tau
            if genes is not None and genes.numel() >= 4:
                camkii_up = (genes[..., 2] * camkii_up).clamp(max=1.0)
                pp1_rate = (genes[..., 3] * pp1_rate).clamp(0.0, 1.0)

            self.camkii.add_(camkii_up * up * (1 - self.camkii))
            self.camkii.clamp_(0, 1)

            self.pp1.mul_(1.0 - pp1_rate).add_(pp1_rate * down)

        # BDNF update: either from CaMKII (legacy) or from Hebbian accumulator (new)
        if self.cfg.bdnf_hebb_accumulate:
            # New mode: BDNF tracks the accumulated |ΔW_hebb| with decay
            # bdnf_hebb_accum is updated in consolidate() with each Hebbian delta
            self.bdnf.mul_(self.cfg.bdnf_tau).add_(
                (1 - self.cfg.bdnf_tau) * self.bdnf_hebb_accum
            )
            # NaN guard and upper clamp to prevent unbounded growth
            if torch.isnan(self.bdnf).any():
                self.bdnf.zero_()
            self.bdnf.clamp_(0, self.cfg.bdnf_max)
        else:
            # Legacy mode: BDNF tracks CaMKII activity
            self.bdnf.mul_(self.cfg.bdnf_tau).add_(
                (1 - self.cfg.bdnf_tau) * F.relu(self.camkii - 0.5)
            )
            # Clamp legacy mode too for consistency
            self.bdnf.clamp_(0, self.cfg.bdnf_max)

    @torch.no_grad()
    def consolidate(self, traceU: Tensor, traceV: Tensor):
        """Consolidate Hebbian traces into slow weights with BDNF-modulated learning rate.

        BDNF Metaplasticity (bio_inspired_nanochat-711):
        - Computes delta from eligibility traces
        - Accumulates |delta| into bdnf_hebb_accum buffer
        - Modulates slow LR by (1 + gamma * bdnf) where gamma = bdnf_gamma or bdnf_scale
        - Guards against NaN/Inf values
        """
        # Compute Hebbian delta from traces
        # traceU: (in, R), traceV: (R, out) -> product is (in, out)
        # self.slow is (out,) so we need to reduce to that shape
        trace_product = traceU @ traceV  # (in, out)

        if trace_product.shape[0] == trace_product.shape[1]:
            # Square matrix: take diagonal
            delta = trace_product.diag()
        else:
            # Non-square: take mean over input dimension -> (out,)
            delta = trace_product.mean(0)

        # Accumulate |ΔW_hebb| for BDNF metaplasticity
        if self.cfg.bdnf_hebb_accumulate:
            delta_mag = delta.abs()
            # Exponential moving average of delta magnitude
            self.bdnf_hebb_accum.mul_(self.cfg.bdnf_tau).add_(
                (1 - self.cfg.bdnf_tau) * delta_mag
            )
            # Store for logging
            self._last_hebb_delta_mag.fill_(delta_mag.mean().item())
            # NaN guard and upper clamp for accumulator
            if torch.isnan(self.bdnf_hebb_accum).any():
                self.bdnf_hebb_accum.zero_()
            self.bdnf_hebb_accum.clamp_(0, self.cfg.bdnf_max)

        # Consolidation gate. With the bistable latch (sax.2), PP1 is IN the gate —
        # g = sigmoid(beta*(CaMKII - PP1)) — so consolidation tracks the latch state and
        # its hysteresis. Otherwise the legacy CaMKII-only threshold gate.
        if self.cfg.bistable_latch:
            g = torch.sigmoid(self.cfg.latch_gate_beta * (self.camkii - self.pp1))
        else:
            g = torch.sigmoid(self.camkii - 0.5) - 0.3

        # Shape check before update
        if delta.shape != self.slow.shape:
            return

        # Compute BDNF-modulated learning rate
        # Use bdnf_gamma if set, otherwise fall back to bdnf_scale
        gamma = self.cfg.bdnf_gamma if self.cfg.bdnf_gamma > 0 else self.cfg.bdnf_scale
        bdnf_gain = 1.0 + gamma * self.bdnf

        # NaN guard for BDNF gain
        if torch.isnan(bdnf_gain).any() or torch.isinf(bdnf_gain).any():
            bdnf_gain = torch.ones_like(bdnf_gain)

        # Apply consolidated update with BDNF-modulated LR
        update = self.cfg.post_slow_lr * bdnf_gain * delta * g

        # Final NaN guard before applying update
        if not torch.isnan(update).any() and not torch.isinf(update).any():
            self.slow.add_(update)

    @torch.no_grad()
    def hebb_fast(self, traceU: Tensor, traceV: Tensor):
        # Update fast weights (diagonal)
        delta = (traceU @ traceV).diag() if traceU.shape[0] == traceV.shape[1] else (traceU @ traceV).mean(0)
        if delta.shape != self.fast.shape:
            return
        self.fast.mul_(self.cfg.post_fast_decay).add_(self.cfg.post_fast_lr * delta)

    @torch.no_grad()
    def reset_sequence_state(
        self, *, reset_fast_weights: bool = False, reset_consolidation: bool = True
    ) -> None:
        """Reset the PER-SEQUENCE postsynaptic state (vg9.4). Persists slow/U/V (consolidated,
        backprop-trained). See SynapticLinear.reset_sequence_state for the full contract."""
        if reset_consolidation:
            self.camkii.zero_()
            self.pp1.fill_(0.5)
            self.bdnf.zero_()
            self.bdnf_hebb_accum.zero_()
            self._last_hebb_delta_mag.zero_()
        if reset_fast_weights:
            self.fast.zero_()

    def get_bdnf_metrics(self) -> Dict[str, float]:
        """Get BDNF-related metrics for logging/monitoring.

        Returns dict with:
        - bdnf_mean: Mean BDNF level across all neurons
        - bdnf_max: Max BDNF level
        - bdnf_hebb_accum_mean: Mean accumulated |ΔW_hebb|
        - last_hebb_delta_mag: Most recent Hebbian delta magnitude
        - camkii_mean: Mean CaMKII level (for reference)
        """
        return {
            "bdnf_mean": float(self.bdnf.mean().item()),
            "bdnf_max": float(self.bdnf.max().item()),
            "bdnf_hebb_accum_mean": float(self.bdnf_hebb_accum.mean().item()),
            "last_hebb_delta_mag": float(self._last_hebb_delta_mag.item()),
            "camkii_mean": float(self.camkii.mean().item()),
        }


class SynapticLinear(nn.Module):
    cfg: SynapticConfig
    use_input_ln: bool
    bias: Optional[nn.Parameter]
    input_ln: Optional[nn.LayerNorm]
    w_slow: nn.Parameter
    w_fast: Optional[nn.Parameter]
    post: Optional[PostsynapticHebb]
    u_buf: Optional[Tensor]
    v_buf: Optional[Tensor]
    proj_in: Optional[Tensor]
    proj_out: Optional[Tensor]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        cfg: SynapticConfig,
        bias: bool = True,
        use_input_ln: bool = False,
    ):
        super().__init__()
        object.__setattr__(self, "cfg", cfg)
        object.__setattr__(self, "use_input_ln", use_input_ln)

        # Standard weights
        self.w_slow = nn.Parameter(torch.empty(in_features, out_features))
        # Only allocate fast weights/Hebbian params if enabled
        if cfg.enable_hebbian:
            self.w_fast = nn.Parameter(torch.empty(in_features, out_features))
            nn.init.trunc_normal_(self.w_fast, std=0.02)

            # Postsynaptic module (operates on output)
            self.post = PostsynapticHebb(in_features, out_features, cfg)

            # Granularity-aware eligibility rank (vap.2): coarse per-expert uses rank 1,
            # medium per-neuron uses intermediate rank, per-connection uses full rank.
            granularity = getattr(cfg, "granularity", SynapticGranularity.PER_CONNECTION)
            if granularity in (SynapticGranularity.PER_EXPERT, "per_expert"):
                _R = 1
            elif granularity in (SynapticGranularity.PER_NEURON, "per_neuron"):
                _R = max(1, min(cfg.rank_eligibility, 4))
            else:
                _R = cfg.rank_eligibility

            # Eligibility buffers
            self.register_buffer("u_buf", torch.zeros(in_features, _R))
            self.register_buffer("v_buf", torch.zeros(_R, out_features))
            # vg9.9: FIXED random projections give the eligibility trace genuine rank R — each
            # rank channel accumulates the correlation of the pre/post activity with a DISTINCT
            # random projection of the other side, instead of the old mean-broadcast (all R
            # columns identical -> effectively rank 1, so rank_eligibility was a no-op knob).
            # Buffers: fixed per model and persisted in the checkpoint.
            self.register_buffer("proj_in", torch.randn(in_features, _R) / math.sqrt(in_features))
            self.register_buffer("proj_out", torch.randn(out_features, _R) / math.sqrt(out_features))
        else:
            self.register_parameter("w_fast", None)
            self.post = None
            self.register_buffer("u_buf", None)
            self.register_buffer("v_buf", None)
            self.register_buffer("proj_in", None)
            self.register_buffer("proj_out", None)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.trunc_normal_(self.w_slow, std=0.02)

        if use_input_ln:
            self.input_ln = nn.LayerNorm(in_features, eps=1e-5)
        else:
            object.__setattr__(self, "input_ln", None)

        # vg9.2: deferred-plasticity bookkeeping. During a grad-enabled (training) forward we
        # cannot mutate w_fast/w_slow/post.fast/post.slow in place after they have been used in
        # the forward matmuls — autograd saved them for backward and an in-place write raises
        # "a variable needed for gradient computation has been modified by an inplace operation".
        # So we compute the detached Hebbian deltas at the END of the step (from buffers only,
        # which is autograd-safe) and APPLY the Parameter writes at the TOP of the NEXT forward,
        # before those Parameters are used. _plasticity_pending flags a deferred write; the
        # eligibility traces (u_buf/v_buf) init to zero so the first application is a no-op.
        self._plasticity_pending: bool = False
        self._last_gate_scale: Optional[Tensor] = None

    def _update_hebb_traces(self, x: Tensor, y: Tensor, genes: Optional[Tensor]) -> None:
        """Update eligibility traces (u_buf/v_buf) + CaMKII/PP1/BDNF state from activations.

        Touches ONLY buffers (never a Parameter used in the live forward graph), so it is
        autograd-safe even inside a grad-enabled forward. Call inside ``torch.no_grad()``.
        """
        if (
            self.u_buf is not None
            and self.v_buf is not None
            and self.proj_out is not None
            and self.proj_in is not None
        ):
            # vg9.9: genuine rank-R eligibility. Project the post-activity y onto R random modes
            # and accumulate its correlation with the pre-activity x into u_buf (in, R); project
            # x onto R modes and accumulate its correlation with y into v_buf (R, out).
            if x.ndim == 3:
                # 3D tensor: (B, T, in_features) along sequence axis
                b_size, t_len, _ = x.shape
                y_proj = y @ self.proj_out.to(y.dtype)  # (B, T, R)
                x_proj = x @ self.proj_in.to(x.dtype)   # (B, T, R)

                if self.cfg.enable_stdp and t_len > 1:
                    # Sequence-axis STDP: pre at t-1 -> post at t (LTP), pre at t -> post at t-1 (LTD)
                    # Computed within each batch sequence independently (no cross-batch boundary artifacts)
                    x_pre = x[:, :-1, :]
                    x_post = x[:, 1:, :]
                    y_proj_post = y_proj[:, 1:, :]
                    y_proj_pre = y_proj[:, :-1, :]
                    x_proj_pre = x_proj[:, :-1, :]
                    x_proj_post = x_proj[:, 1:, :]
                    y_post = y[:, 1:, :]
                    y_pre = y[:, :-1, :]

                    norm_factor = max(1, b_size * (t_len - 1))
                    ltp_u = torch.einsum("bti,btr->ir", x_pre, y_proj_post) / norm_factor
                    ltd_u = torch.einsum("bti,btr->ir", x_post, y_proj_pre) / norm_factor
                    ltp_v = torch.einsum("btr,btj->rj", x_proj_pre, y_post) / norm_factor
                    ltd_v = torch.einsum("btr,btj->rj", x_proj_post, y_pre) / norm_factor

                    w_plus = math.exp(-1.0 / max(1e-3, self.cfg.stdp_tau_plus))
                    w_minus = math.exp(-1.0 / max(1e-3, self.cfg.stdp_tau_minus))
                    stdp_delta_u = (self.cfg.stdp_a_plus * w_plus) * ltp_u - (self.cfg.stdp_a_minus * w_minus) * ltd_u
                    stdp_delta_v = (self.cfg.stdp_a_plus * w_plus) * ltp_v - (self.cfg.stdp_a_minus * w_minus) * ltd_v

                    self.u_buf.mul_(self.cfg.post_trace_decay).add_(stdp_delta_u)
                    self.v_buf.mul_(self.cfg.post_trace_decay).add_(stdp_delta_v)
                else:
                    norm_factor = max(1, b_size * t_len)
                    self.u_buf.mul_(self.cfg.post_trace_decay).add_(
                        0.05 * torch.einsum("bti,btr->ir", x, y_proj) / norm_factor
                    )
                    self.v_buf.mul_(self.cfg.post_trace_decay).add_(
                        0.05 * torch.einsum("btr,btj->rj", x_proj, y) / norm_factor
                    )
            else:
                # 2D tensor: (N, in_features) e.g. routed MoE tokens or single time-slice
                batch = max(1, x.shape[0])
                y_proj = y @ self.proj_out.to(y.dtype)  # (N, R)
                x_proj = x @ self.proj_in.to(x.dtype)   # (N, R)

                if self.cfg.enable_stdp and x.shape[0] > 1:
                    x_pre = x[:-1]
                    x_post = x[1:]
                    y_proj_post = y_proj[1:]
                    y_proj_pre = y_proj[:-1]
                    x_proj_pre = x_proj[:-1]
                    x_proj_post = x_proj[1:]
                    y_post = y[1:]
                    y_pre = y[:-1]

                    ltp_u = (x_pre.transpose(0, 1) @ y_proj_post) / (batch - 1)
                    ltd_u = (x_post.transpose(0, 1) @ y_proj_pre) / (batch - 1)
                    ltp_v = (x_proj_pre.transpose(0, 1) @ y_post) / (batch - 1)
                    ltd_v = (x_proj_post.transpose(0, 1) @ y_pre) / (batch - 1)

                    w_plus = math.exp(-1.0 / max(1e-3, self.cfg.stdp_tau_plus))
                    w_minus = math.exp(-1.0 / max(1e-3, self.cfg.stdp_tau_minus))
                    stdp_delta_u = (self.cfg.stdp_a_plus * w_plus) * ltp_u - (self.cfg.stdp_a_minus * w_minus) * ltd_u
                    stdp_delta_v = (self.cfg.stdp_a_plus * w_plus) * ltp_v - (self.cfg.stdp_a_minus * w_minus) * ltd_v

                    self.u_buf.mul_(self.cfg.post_trace_decay).add_(stdp_delta_u)
                    self.v_buf.mul_(self.cfg.post_trace_decay).add_(stdp_delta_v)
                else:
                    self.u_buf.mul_(self.cfg.post_trace_decay).add_(
                        0.05 * (x.transpose(0, 1) @ y_proj) / batch  # (in, R)
                    )
                    self.v_buf.mul_(self.cfg.post_trace_decay).add_(
                        0.05 * (x_proj.transpose(0, 1) @ y) / batch  # (R, out)
                    )
        # Per-neuron calcium proxy for the CaMKII/PP1 gate.
        if self.post is not None:
            ca_vec = y.abs().reshape(-1, y.shape[-1]).mean(0).clamp(0, 10.0)
            self.post.update(y, ca_vec, genes=genes)

    def _apply_hebb_weight_writes(self, gate_scale: Optional[Tensor]) -> None:
        """Apply the Hebbian Parameter writes (w_fast/w_slow + post.fast/post.slow) from the
        current eligibility traces.

        MUST be called inside ``torch.no_grad()`` and at a point where these Parameters have
        NOT yet been used in the live forward graph this step (the top of forward, or an
        inference forward with no pending backward). Mutating them after a matmul that saved
        them would corrupt the pending backward. Traces init to zero, so a call before any
        trace update is a no-op.
        """
        if self.u_buf is None or self.v_buf is None:
            return
        if self.w_fast is not None:
            if gate_scale is None:
                gs = torch.ones((), device=self.w_fast.device, dtype=self.w_fast.dtype)
            else:
                gs = gate_scale.to(device=self.w_fast.device, dtype=self.w_fast.dtype)
            delta = self.u_buf @ self.v_buf
            delta = delta * gs.to(delta.dtype)
            # hy8.1: dopamine (reward-prediction-error) plasticity gain. Default-neutral
            # (1.0) unless a NeuromodulatoryBus has broadcast a gain — then it scales the
            # consolidation step so only reward-relevant updates are amplified (the three-factor
            # bridge to RL, hy8.2). Applied to the step, not delta, so it survives normalization.
            da = getattr(self, "_nm_da_gain", 1.0)
            if self.cfg.fast_weight_normalized:
                # sax.1: normalized + norm-bounded online write. Step BOTH the fast and slow
                # online Hebbian writes along the unit-norm Hebbian direction (impactful
                # regardless of the tiny raw trace magnitude), and cap ||w_fast||. This kills the
                # positive-feedback blowup that a naive LR boost triggers: the fast pathway
                # amplifies activations, which feed the otherwise-unbounded w_slow online drift
                # (`w_slow.add_(post_slow_lr*delta)` has no decay) — so bounding only w_fast still
                # diverges through w_slow. Bounding the direction of both keeps the system finite.
                dn = delta.norm()
                if float(dn) > 1e-12:
                    direction = delta / dn
                    self.w_fast.mul_(self.cfg.post_fast_decay).add_(
                        (self.cfg.fast_weight_eta * da) * direction
                    )
                    maxn = self.cfg.fast_weight_max_norm
                    if maxn > 0:
                        wn = self.w_fast.norm()
                        if float(wn) > maxn:
                            self.w_fast.mul_(maxn / wn)
                    self.w_slow.add_((self.cfg.post_slow_lr * da) * direction)
            else:
                self.w_fast.mul_(self.cfg.post_fast_decay).add_((self.cfg.post_fast_lr * da) * delta)
                self.w_slow.add_((self.cfg.post_slow_lr * da) * delta)
        if self.post is not None:
            self.post.hebb_fast(self.u_buf, self.v_buf)
            self.post.consolidate(self.u_buf, self.v_buf)

    @torch.no_grad()
    def reset_sequence_state(
        self, *, reset_fast_weights: bool = False, reset_consolidation: bool = True
    ) -> None:
        """Reset the PER-SEQUENCE fast/eligibility state at a sequence boundary (vg9.4).

        These module buffers/params persist across forwards and were never reset, so one
        sequence's writes leaked into the next. Contract:

          PER-SEQUENCE (always reset): eligibility traces u_buf/v_buf and the deferred-plasticity
            bookkeeping (_plasticity_pending / _last_gate_scale).
          reset_consolidation (default True): the CaMKII/PP1/BDNF gate + metaplasticity state
            (in post). The "cel" consolidation-across-sequences mode passes False to carry it.
          reset_fast_weights (default False): also zero the fast weights w_fast / post.fast for
            STRICT working-memory isolation. NOTE: w_fast is a backprop-trained Parameter, so
            this discards its trained component — a proper Parameter/buffer split is future work.
          PERSISTENT (never reset): slow weights w_slow / post.slow, low-rank U/V, the fixed
            random projections proj_in/proj_out, bias, and the presyn EMA.

        Call at a sequence boundary (no autograd graph pending) — it writes Parameters in place.
        """
        if self.u_buf is not None:
            self.u_buf.zero_()
        if self.v_buf is not None:
            self.v_buf.zero_()
        self._plasticity_pending = False
        self._last_gate_scale = None
        if reset_fast_weights and self.w_fast is not None:
            self.w_fast.zero_()
        if self.post is not None:
            self.post.reset_sequence_state(
                reset_fast_weights=reset_fast_weights, reset_consolidation=reset_consolidation
            )

    def forward(
        self, x: Tensor, calcium: Tensor, energy: Tensor, update_mem: bool = True, genes: Optional[Tensor] = None
    ):
        if self.input_ln is not None:
            x = self.input_ln(x)

        # hy8.5: acetylcholine attention/input gain. Default-neutral (1.0) unless a
        # NeuromodulatoryBus broadcast a gain — higher ACh (uncertainty) sharpens input
        # sensitivity. Applied before the matmuls so it modulates both output and Hebbian traces.
        ach_in = getattr(self, "_nm_ach_input_gain", 1.0)
        if ach_in != 1.0:
            x = x * ach_in

        # vg9.2: flush any plasticity Parameter writes deferred from the previous (training)
        # forward, BEFORE this step's matmuls use those Parameters — autograd-safe because they
        # have not yet been saved for this step's backward. First call is a no-op (zero traces).
        # 9mxi: only a TRAIN forward may flush the deferred writes. An eval forward
        # (model.eval() / update_mem=False) leaves them pending so validation can
        # never mutate w_fast/w_slow/post.* mid-evaluation; the next training
        # forward lands the identical write before its matmuls, preserving the
        # training trajectory exactly.
        if (
            self._plasticity_pending
            and self.training
            and update_mem
            and self.cfg.enable_hebbian
            and self.post is not None
        ):
            with torch.no_grad():
                self._apply_hebb_weight_writes(self._last_gate_scale)
            self._plasticity_pending = False

        # Linear pass (separate slow/fast for calcium/energy gating)
        fast_gate: Optional[Tensor] = None
        if self.cfg.enable_hebbian and self.w_fast is not None:
            y_slow = x @ self.w_slow
            y_fast = x @ self.w_fast

            # Build a per-sample gate from calcium/energy signals.
            def _gate_from_signal(signal: Tensor, y_ref: Tensor) -> Tensor:
                sig = signal
                if y_ref.ndim == 3:
                    B, T, out_dim = y_ref.shape
                    if sig.ndim == 0:
                        return sig.view(1, 1, 1).expand(B, T, 1)
                    elif sig.ndim == 1:
                        if sig.shape[0] == B:
                            return sig.view(B, 1, 1).expand(B, T, 1)
                        elif sig.shape[0] == B * T:
                            return sig.view(B, T, 1)
                        else:
                            return sig.mean().view(1, 1, 1).expand(B, T, 1)
                    elif sig.ndim == 2 and sig.shape == (B, T):
                        return sig.unsqueeze(-1)
                    elif sig.ndim == 3 and sig.shape == (B, T, out_dim):
                        return sig
                    else:
                        return sig.mean().view(1, 1, 1).expand(B, T, 1)
                else:
                    n_rows, out_dim = y_ref.shape
                    if sig.ndim == 0:
                        return sig.view(1, 1).expand(n_rows, 1)
                    elif sig.ndim == 1:
                        if sig.shape[0] == n_rows:
                            return sig.view(n_rows, 1)
                        else:
                            return sig.mean().view(1, 1).expand(n_rows, 1)
                    elif sig.ndim == 2 and sig.shape[0] == n_rows and sig.shape[1] == out_dim:
                        return sig
                    else:
                        return sig.reshape(n_rows, -1).mean(dim=1, keepdim=True)

            fast_gate = _gate_from_signal(calcium, y_fast)
            energy_gate = _gate_from_signal(energy, y_fast)
            fast_gate = (fast_gate * energy_gate).clamp(0.0, 1.0).to(y_fast.dtype)

            y = y_slow + (y_fast * fast_gate)
        else:
            y = x @ self.w_slow
        if self.bias is not None:
            y = y + self.bias

        # Postsynaptic modulation (diagonal fast/slow + low-rank)
        if self.cfg.enable_hebbian and self.post is not None:
            y = self.post(y)

            # vg9.2: online Hebbian plasticity. Previously gated behind
            # `not torch.is_grad_enabled()`, so the headline "online learning" NEVER ran during
            # training. It now runs as a DETACHED fast-adaptation update during inference
            # (no_grad) AND during training (when plasticity_during_training is set).
            # 9mxi: update_mem is threaded from GPTSynaptic.forward(train_mode=...) down
            # through Block -> MLP/MoE -> Expert, so the EVALUATION path (model.eval() +
            # no_grad + train_mode=False) never adapts: validation cannot contaminate the
            # model and val_bpb stays idempotent. Generation keeps adapting (default True).
            grad_on = torch.is_grad_enabled()
            run_plasticity = update_mem and (
                not grad_on or (self.training and self.cfg.plasticity_during_training)
            )
            if run_plasticity:
                with torch.no_grad():
                    # Traces + CaMKII/PP1/BDNF updates touch only buffers -> always safe.
                    self._update_hebb_traces(x, y, genes)
                    if grad_on:
                        # Training: a backward is pending and the matmuls above saved
                        # w_fast/w_slow/post.fast/post.slow. Defer their in-place writes to the
                        # top of the NEXT forward (applied before those Parameters are reused),
                        # stashing the gate scale that weights the delta. base_train runs
                        # backward immediately after each forward, so the deferred write always
                        # lands after the prior backward — no graph is corrupted.
                        self._last_gate_scale = (
                            fast_gate.mean().detach() if fast_gate is not None else None
                        )
                        self._plasticity_pending = True
                    else:
                        # Inference: no backward pending -> apply the writes now (legacy path).
                        gate_scale = fast_gate.mean() if fast_gate is not None else None
                        self._apply_hebb_weight_writes(gate_scale)

        # hy8.1: norepinephrine (arousal/novelty) global-gain neuromodulation. Default-neutral
        # (1.0) unless a NeuromodulatoryBus has broadcast a gain onto this module. Modulates the
        # broadcast output (not the local Hebbian trace), so it gates downstream signal gain.
        ne_gain = getattr(self, "_nm_ne_gain", 1.0)
        if ne_gain != 1.0:
            y = y * ne_gain

        return y


# -----------------------------------------------------------------------------
# Presyn state builder
# -----------------------------------------------------------------------------


def build_presyn_state(B: int, T: int, H: int, device, dtype, cfg: SynapticConfig):
    state_shape = (B, H, T)
    ones = torch.ones(state_shape, device=device, dtype=dtype)
    zeros = torch.zeros(state_shape, device=device, dtype=dtype)
    state = {
        # Triton/Rust-compatible names
        "C": zeros.clone(),
        "BUF": zeros.clone(),
        "RRP": ones * cfg.init_rrp,
        "RES": ones * cfg.init_reserve,
        "PR": ones * cfg.init_snare,
        "CL": ones * cfg.init_clamp,
        "E": ones * cfg.init_energy,
        # Extra state used by the Python reference implementation / attention augmentation
        "AMP": ones * cfg.init_amp,
        "DELAY": [zeros.clone() for _ in range(cfg.endo_delay)],
    }
    if cfg.metriplectic_integrator:
        # Heat/entropy reservoir for the proof core z=(C, BUF, h). It costs no memory in the
        # default-off configuration and is initialized on the zero-entropy shell when enabled.
        state["HEAT"] = zeros.clone()
    return state


# -----------------------------------------------------------------------------
# Attention and MLP
# -----------------------------------------------------------------------------


class SynapticCausalSelfAttention(nn.Module):
    cfg: SynapticConfig
    """
    Drop-in attention with synaptic augmentation. Uses standard Q,K,V projections,
    RoPE, multi-query key/value replication, and adds log(ε+q⋅n) to logits.
    """

    def __init__(
        self,
        n_embd,
        n_head,
        n_kv_head,
        rope_cos,
        rope_sin,
        cfg: SynapticConfig,
        layer_idx: int,
        attn_drop=0.0,
        resid_drop=0.0,
    ):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd {n_embd} must be divisible by n_head {n_head}")
        if n_kv_head > n_head or (n_head % n_kv_head) != 0:
            raise ValueError(
                f"n_kv_head {n_kv_head} must be <= n_head {n_head} and divide it exactly"
            )
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_embd // n_head
        self.layer_idx = int(layer_idx)
        object.__setattr__(self, "cfg", cfg)

        if cfg.use_flex_attention:
            if not _HAS_FLEX:
                raise ImportError(
                    "SynapticConfig.use_flex_attention=True but FlexAttention is unavailable "
                    "(requires torch>=2.5 and torch.nn.attention.flex_attention)."
                )
            self.flex = SynapticFlexAttention(cfg)
        else:
            self.flex = None

        self.q_proj = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_head * self.head_dim, n_embd, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.resid_drop = nn.Dropout(resid_drop)
        self.cos, self.sin = rope_cos, rope_sin
        self.pre = SynapticPresyn(self.head_dim, cfg)

    def _chunked_release_bias(
        self,
        presyn_state: Dict[str, Any],
        vals: Tensor,
        idx: Tensor,
        valid: Tensor,
        train_mode: bool,
        prefix_len: int,
    ) -> Tensor:
        """Causal presynaptic release bias, optionally grouped for differentiable BPTT.

        Every forward advances one query at a time, matching token-by-token decoding exactly.
        ``recurrence_block_size`` groups those exact query steps only for autograd truncation and
        checkpoint orchestration; it never lets queries share a state snapshot. Every group receives
        its causal final key extent, so preallocated future state is never passively aged.
        ``release_canonical(differentiable=…)`` carries gradient through state so the learnable
        kinetics receive a real training signal. Backprop is truncated every
        ``recurrence_chunk_len`` chunks to bound activation memory. The per-chunk biases are
        concatenated back along the query dimension; the attention matmul/mask/softmax/barrier path
        is unchanged and still spans the full sequence. The state-recurrence autograd is requested
        only when both configured and grad is enabled.
        """
        block = max(1, int(self.cfg.recurrence_block_size))
        drives = list(vals.split(block, dim=2))
        idxs = list(idx.split(block, dim=2))
        valids = list(valid.split(block, dim=2))
        active_key_counts: List[Optional[int]] = []
        active_key_count = int(prefix_len)
        for drive in drives:
            active_key_count += int(drive.size(2))
            active_key_counts.append(active_key_count)
        outs = chunked_recurrence(
            self.pre,
            presyn_state,
            drives,
            idxs,
            chunk_len=int(self.cfg.recurrence_chunk_len),
            checkpoint_len=int(self.cfg.recurrence_checkpoint_len),
            train=train_mode,
            valids=valids,
            active_key_counts=active_key_counts,
            differentiable=self.cfg.differentiable_recurrence and torch.is_grad_enabled(),
        )
        return torch.cat(outs, dim=2)

    def _apply_rope(self, x: Tensor, T0: int):
        H = self.n_head if x.size(-1) == self.n_head * self.head_dim else self.n_kv_head
        D = self.head_dim
        x = x.view(x.size(0), x.size(1), H, D)
        cos = self.cos[:, T0 : T0 + x.size(1), : D // 2].to(x.device).unsqueeze(2)
        sin = self.sin[:, T0 : T0 + x.size(1), : D // 2].to(x.device).unsqueeze(2)
        x1, x2 = x.split(D // 2, dim=-1)
        xr = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return xr

    def _repeat_kv(self, x: Tensor):
        if self.n_head == self.n_kv_head:
            return x
        nrep = self.n_head // self.n_kv_head
        b, t, nh, d = x.shape
        return x.unsqueeze(2).expand(b, t, nh, nrep, d).reshape(b, t, self.n_head, d)

    def forward(self, x: Tensor, kv_cache=None, presyn_state=None, train_mode=True):
        B, Tq, _C = x.shape
        H = self.n_head
        D = self.head_dim
        device = x.device
        dtype = x.dtype

        # Projections (MQA/GQA: K/V may have fewer heads than Q)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x).view(B, Tq, self.n_kv_head, D)

        # RoPE offset is based on the current KV cache position (prefix length)
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        q = _rmsnorm(self._apply_rope(q, T0)).transpose(1, 2)  # (B, H, Tq, D)
        k = _rmsnorm(self._apply_rope(k, T0)).transpose(1, 2)  # (B, Hkv, Tq, D)
        v = v.transpose(1, 2)  # (B, Hkv, Tq, D)

        # KV cache: store and fetch the full prefix+current K/V for this layer.
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)  # (B, Hkv, Tk, D)
        Tk = int(k.size(2))

        # Expand presynaptic state to cover all key positions (prefix + current).
        if presyn_state is None:
            presyn_state = build_presyn_state(B, Tk, H, device, dtype, self.cfg)
        else:
            # Fill in missing keys from older caches/checkpoints and extend along time as needed.
            if "C" not in presyn_state:
                raise KeyError("presyn_state missing required key 'C'")
            T_state = int(presyn_state["C"].size(2))
            if "BUF" not in presyn_state:
                presyn_state["BUF"] = torch.zeros_like(presyn_state["C"])
            if self.cfg.metriplectic_integrator and "HEAT" not in presyn_state:
                presyn_state["HEAT"] = torch.zeros_like(presyn_state["C"])
            if "RRP" not in presyn_state:
                presyn_state["RRP"] = torch.full_like(presyn_state["C"], self.cfg.init_rrp)
            if "RES" not in presyn_state:
                presyn_state["RES"] = torch.full_like(presyn_state["C"], self.cfg.init_reserve)
            if "PR" not in presyn_state:
                presyn_state["PR"] = torch.full_like(presyn_state["C"], self.cfg.init_snare)
            if "CL" not in presyn_state:
                presyn_state["CL"] = torch.full_like(presyn_state["C"], self.cfg.init_clamp)
            if "E" not in presyn_state:
                presyn_state["E"] = torch.full_like(presyn_state["C"], self.cfg.init_energy)
            if "AMP" not in presyn_state:
                presyn_state["AMP"] = torch.full_like(presyn_state["C"], self.cfg.init_amp)
            if "DELAY" not in presyn_state:
                presyn_state["DELAY"] = [
                    torch.zeros_like(presyn_state["C"]) for _ in range(self.cfg.endo_delay)
                ]

            if T_state < Tk:
                T_add = Tk - T_state
                state_dtype = presyn_state["C"].dtype
                pad_zeros = torch.zeros((B, H, T_add), device=device, dtype=state_dtype)

                def pad_full(t: Tensor, fill: float) -> Tensor:
                    pad = torch.full((B, H, T_add), fill, device=device, dtype=t.dtype)
                    return torch.cat([t, pad], dim=2)

                presyn_state["C"] = torch.cat([presyn_state["C"], pad_zeros], dim=2)
                presyn_state["BUF"] = torch.cat([presyn_state["BUF"], pad_zeros], dim=2)
                if self.cfg.metriplectic_integrator:
                    presyn_state["HEAT"] = torch.cat(
                        [presyn_state["HEAT"], pad_zeros], dim=2
                    )
                presyn_state["RRP"] = pad_full(presyn_state["RRP"], self.cfg.init_rrp)
                presyn_state["RES"] = pad_full(presyn_state["RES"], self.cfg.init_reserve)
                presyn_state["PR"] = pad_full(presyn_state["PR"], self.cfg.init_snare)
                presyn_state["CL"] = pad_full(presyn_state["CL"], self.cfg.init_clamp)
                presyn_state["E"] = pad_full(presyn_state["E"], self.cfg.init_energy)
                presyn_state["AMP"] = pad_full(presyn_state["AMP"], self.cfg.init_amp)
                presyn_state["DELAY"] = [
                    torch.cat([d, pad_zeros], dim=2) for d in presyn_state["DELAY"]
                ]

        # Repeat cached K/V heads to match query heads (GQA)
        k_full = self._repeat_kv(k.transpose(1, 2)).transpose(1, 2)  # (B, H, Tk, D)
        v_full = self._repeat_kv(v.transpose(1, 2)).transpose(1, 2)  # (B, H, Tk, D)

        # Build attention logits (masked in-place)
        dots = (q @ k_full.transpose(-1, -2)) / math.sqrt(D)  # (B, H, Tq, Tk)
        prefix_len = Tk - Tq
        if prefix_len <= 0:
            attn_mask = torch.tril(torch.ones((Tq, Tk), device=device, dtype=torch.bool))
        else:
            attn_mask = torch.zeros((Tq, Tk), device=device, dtype=torch.bool)
            attn_mask[:, :prefix_len] = True
            attn_mask[:, prefix_len:] = torch.tril(
                torch.ones((Tq, Tq), device=device, dtype=torch.bool)
            )
        dots = dots.masked_fill(~attn_mask.view(1, 1, Tq, Tk), -torch.inf)

        # --- FlexAttention Path (training/prefill only for now) ---
        if self.flex is not None:
            if prefix_len > 0:
                raise NotImplementedError(
                    "SynapticFlexAttention currently requires full-sequence attention (no prefix KV cache). "
                    "Set SynapticConfig.use_flex_attention=False for decoding with KV cache."
                )
            topk = min(self.cfg.attn_topk, Tk)
            vals, idx = torch.topk(dots, topk, dim=-1)
            valid = torch.isfinite(vals)
            from torch.nn.attention.flex_attention import create_block_mask

            if q.dtype != v_full.dtype:
                q = q.to(v_full.dtype)
            if k_full.dtype != v_full.dtype:
                k_full = k_full.to(v_full.dtype)

            # FlexAttention must see the state snapshot belonging to each query, rather than the
            # final state after the whole prefill. Advance and attend one query at a time so neither
            # recurrent state nor the score modifier can read/age future-token slots. This keeps
            # Flex's O(N) memory property, trading prefill launch count for exact causal semantics.
            def make_causal_mask(query_offset: int):
                def causal_mask(_b, _h, q_idx, kv_idx):
                    return q_idx + query_offset >= kv_idx

                return causal_mask

            query_outputs = []
            for query_index in range(Tq):
                _ = self.pre.release_canonical(
                    presyn_state,
                    vals[:, :, query_index : query_index + 1],
                    idx[:, :, query_index : query_index + 1],
                    train_mode,
                    valid=valid[:, :, query_index : query_index + 1],
                    active_key_count=query_index + 1,
                )
                query_offset = query_index
                block_mask = create_block_mask(
                    make_causal_mask(query_offset), B, H, 1, Tk, device=device
                )
                query_outputs.append(
                    self.flex(
                        q[:, :, query_index : query_index + 1],
                        k_full,
                        v_full,
                        presyn_state,
                        block_mask=block_mask,
                        query_offset=query_offset,
                    )
                )
            y = torch.cat(query_outputs, dim=2)
            y = y.transpose(1, 2).contiguous().view(B, Tq, H * D)
            y = self.resid_drop(self.o_proj(y))
            return y, presyn_state

        # --- Standard Path ---
        topk = min(self.cfg.attn_topk, Tk)
        vals, idx = torch.topk(dots, topk, dim=-1)
        valid = torch.isfinite(vals)

        # Run presynaptic physics on only the valid edges (8j9.2/ukxt: canonical faithful
        # release; the septin barrier is applied at the logit level below, not folded into e).
        # Multi-query forwards advance presynaptic physics causally. The ordinary path uses exact
        # one-query steps; differentiable recurrence may explicitly group queries for training.
        # Single-token decode already has exactly one materialized query/key frontier.
        differentiable_recurrence = (
            self.cfg.differentiable_recurrence and torch.is_grad_enabled()
        )
        native_decode = self.pre._can_use_native_presyn_decode(
            presyn_state,
            vals,
            train=train_mode,
            differentiable=differentiable_recurrence,
            apply_barrier=False,
            q_pos=None,
        )
        if Tq > 1:
            e = self._chunked_release_bias(
                presyn_state,
                vals,
                idx,
                valid,
                train_mode,
                prefix_len,
            )
        elif native_decode:
            e = self.pre.release_canonical(
                presyn_state,
                vals,
                idx,
                train_mode,
                valid=valid,
                logits=dots,
            )
        else:
            e = self.pre.release_canonical(
                presyn_state,
                vals,
                idx,
                train_mode,
                valid=valid,
                differentiable=differentiable_recurrence,
            )

        # Scatter biological log-bias back into the logits, preserving masking.
        if native_decode:
            augmented_dots = dots
        else:
            aug = torch.zeros_like(dots)
            src_val = self.cfg.lambda_loge * torch.log(self.cfg.epsilon + e).to(
                aug.dtype
            )
            # Clamp the log-release bias to a finite range so no single edge can dominate
            # the softmax when the normalized release spikes (numerical hardening, vg9.5).
            clamp = self.cfg.loge_bias_clamp
            if clamp and clamp > 0.0:
                src_val = src_val.clamp(-clamp, clamp)
            src_val = src_val * valid.to(src_val.dtype)
            aug.scatter_add_(-1, idx, src_val)
            augmented_dots = dots + aug

        # Septin-like distance barrier in global positions.
        q_pos = torch.arange(
            prefix_len, prefix_len + Tq, device=device, dtype=torch.float32
        )
        k_pos = torch.arange(0, Tk, device=device, dtype=torch.float32)
        causal_extent = (q_pos + 1.0).clamp_min(1.0)[:, None]
        dist = (q_pos[:, None] - k_pos[None, :]).abs() / causal_extent
        logits = augmented_dots - (
            self.cfg.barrier_strength * dist.to(dots.dtype)
        ).view(1, 1, Tq, Tk)

        P = F.softmax(logits, dim=-1)
        P = self.attn_drop(P)
        ctx = torch.matmul(P.to(v_full.dtype), v_full)
        y = ctx.transpose(1, 2).contiguous().view(B, Tq, H * D)
        y = self.resid_drop(self.o_proj(y))
        return y, presyn_state


class SynapticMLP(nn.Module):
    cfg: SynapticConfig
    def __init__(self, n_embd: int, cfg: SynapticConfig, dropout: float = 0.0):
        super().__init__()
        object.__setattr__(self, "cfg", cfg)
        self.fc = SynapticLinear(n_embd, 4 * n_embd, cfg, bias=True, use_input_ln=True)
        self.proj = SynapticLinear(
            4 * n_embd, n_embd, cfg, bias=True, use_input_ln=False
        )
        self.drop = nn.Dropout(dropout)
        self.register_buffer("C0", torch.tensor(0.5))
        self.register_buffer("E0", torch.tensor(0.8))

    def forward(self, x: Tensor, update_mem: bool = True):
        B, T, C = x.shape
        c0 = self.C0
        e0 = self.E0
        c = c0.expand(B, T)
        e = e0.expand(B, T)
        h = self.fc(x, c, e, update_mem=update_mem)
        h = F.relu(h).square()
        h = self.drop(h)
        y = self.proj(h, c, e, update_mem=update_mem)
        return y


# -----------------------------------------------------------------------------
# Synaptic MoE (router embeddings, contrastive updates)
# -----------------------------------------------------------------------------


class SynapticExpert(nn.Module):
    cfg: SynapticConfig
    fc1: SynapticLinear
    fc2: SynapticLinear
    drop: nn.Dropout

    def __init__(
        self, n_embd: int, hidden_mult: int, cfg: SynapticConfig, dropout: float = 0.0
    ):
        super().__init__()
        object.__setattr__(self, "cfg", cfg)
        h = hidden_mult * n_embd
        self.fc1 = SynapticLinear(n_embd, h, cfg, bias=True, use_input_ln=False)
        self.fc2 = SynapticLinear(h, n_embd, cfg, bias=True, use_input_ln=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, energy_override: Optional[Tensor] = None, genes: Optional[Tensor] = None, update_mem: bool = True) -> Tensor:
        # x: (N, C)
        N = x.size(0)
        device = x.device

        if energy_override is not None:
            # Snapshot the persistent buffer: SynapticMoE updates its storage before backward.
            # Once genome-derived calcium requires grad, autograd also saves this multiplier;
            # retaining an expanded view of the live buffer would then trip a version mismatch.
            energy_snapshot = energy_override.detach().clone()
            if energy_override.ndim == 0:
                e_tens = energy_snapshot.expand(N)
            else:
                e_tens = energy_snapshot.view(-1).expand(N)
        else:
            e_tens = torch.ones(N, device=device)

        # yw9.4: the decoder's calcium-retention/influx kinetics must affect the live,
        # differentiable expert path, not only the detached Hebbian state update. Normalize the
        # decoded one-step calcium response around 0.8 at the configured baseline; this leaves
        # headroom below SynapticLinear's [0,1] fast-path gate and therefore preserves gradient.
        if genes is not None and genes.numel() >= 6:
            rho_c0 = math.exp(-1.0 / max(self.cfg.tau_c, 1e-6))
            baseline_response = max(rho_c0 + self.cfg.alpha_ca, 1e-6)
            calcium_response = 0.8 * (genes[4] + genes[5]) / baseline_response
            c_tens = calcium_response.clamp(0.25, 1.0).expand(N)
        else:
            c_tens = torch.ones(N, device=device)

        y = self.fc1(
            x,
            calcium=c_tens,
            energy=e_tens,
            genes=genes,
            update_mem=update_mem,
        )
        y = F.relu(y).square()
        y = self.drop(y)
        y = self.fc2(
            y,
            calcium=c_tens,
            energy=e_tens,
            genes=genes,
            update_mem=update_mem,
        )
        return y


GENOME_PHENOTYPE_FIELDS: tuple[str, ...] = (
    "alpha_fatigue",
    "alpha_energy",
    "camkii_gain",
    "pp1_gain",
    "rho_c",
    "alpha_ca",
)

# Closed intervals keep every decoded value biologically meaningful and numerically stable.
# rho_c additionally implies tau_c=-1/log(rho_c)>0; alpha_energy implies the recovery
# time-constant tau_rec=-1/log(1-alpha_energy)>0.
_GENOME_PHENOTYPE_BOUNDS: tuple[tuple[float, float], ...] = (
    (0.001, 0.030),  # fatigue EMA rate
    (0.001, 0.020),  # energy recovery EMA rate
    (0.250, 2.500),  # CaMKII gain
    (0.250, 2.500),  # PP1 gain
    (0.500, 0.980),  # calcium retention rho_c
    (0.050, 1.000),  # calcium influx alpha_ca
)


def _bounded_logit(value: float, low: float, high: float) -> float:
    """Inverse of ``low + (high-low)*sigmoid(raw)`` for decoder initialization."""
    unit = min(1.0 - 1e-6, max(1e-6, (value - low) / (high - low)))
    return math.log(unit / (1.0 - unit))


class SynapticGenomeDecoder(nn.Module):
    """Shared Xi-to-kinetics decoder with stability-preserving output maps (yw9.4).

    The only per-expert learned state is the compact ``Xi`` row. A single shared affine decoder
    expands it to six phenotype values, after which bounded sigmoid maps make invalid kinetics
    unrepresentable. With ``xi_dim=0`` the affine term is absent and the learned bias is a genuine
    shared-kinetics control rather than a frozen hand-tuned fallback.
    """

    def __init__(self, xi_dim: int, cfg: SynapticConfig):
        super().__init__()
        object.__setattr__(self, "xi_dim", xi_dim)
        object.__setattr__(self, "cfg", cfg)
        if xi_dim > 0:
            self.raw_weight = nn.Parameter(torch.empty(xi_dim, len(GENOME_PHENOTYPE_FIELDS)))
        else:
            self.register_parameter("raw_weight", None)
        self.raw_bias = nn.Parameter(torch.empty(len(GENOME_PHENOTYPE_FIELDS)))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        if self.raw_weight is not None:
            nn.init.normal_(self.raw_weight, std=0.05)
        defaults = (
            0.011,
            0.0055,
            math.log1p(math.exp(1.0)),
            math.log1p(math.exp(0.5)),
            math.exp(-1.0 / max(self.cfg.tau_c, 1e-6)),
            self.cfg.alpha_ca,
        )
        raw_defaults = [
            _bounded_logit(value, low, high)
            for value, (low, high) in zip(defaults, _GENOME_PHENOTYPE_BOUNDS)
        ]
        self.raw_bias.copy_(
            torch.tensor(raw_defaults, dtype=self.raw_bias.dtype, device=self.raw_bias.device)
        )

    def forward(self, xi: Tensor) -> Tensor:
        raw = self.raw_bias.expand(*xi.shape[:-1], -1)
        if self.raw_weight is not None:
            raw = raw + xi @ self.raw_weight
        values = [
            low + (high - low) * torch.sigmoid(raw[..., i])
            for i, (low, high) in enumerate(_GENOME_PHENOTYPE_BOUNDS)
        ]
        return torch.stack(values, dim=-1)

    def kinetics(self, xi: Tensor) -> Dict[str, Tensor]:
        """Decode named kinetics, including positive time constants derived from EMA rates."""
        phenotype = self(xi)
        alpha_energy = phenotype[..., 1]
        rho_c = phenotype[..., 4]
        return {
            "alpha_fatigue": phenotype[..., 0],
            "alpha_energy": alpha_energy,
            "camkii_gain": phenotype[..., 2],
            "pp1_gain": phenotype[..., 3],
            "rho_c": rho_c,
            "tau_c": -1.0 / torch.log(rho_c),
            "tau_rec": -1.0 / torch.log1p(-alpha_energy),
            "alpha_ca": phenotype[..., 5],
        }


class SynapticMoE(nn.Module):
    num_experts: int
    top_k: int
    cfg: SynapticConfig
    last_aux_loss: Optional[Tensor]
    last_ctx: Dict[str, Tensor]
    last_neuroscore: Optional[Tensor]
    glial: Optional[GlialHomeostasis]
    router: nn.Linear
    experts: nn.ModuleList
    router_embeddings: nn.Parameter
    router_logit_bias: Tensor
    fatigue: Tensor
    energy: Tensor
    Xi: Optional[nn.Parameter]
    kinetics_decoder: Optional[nn.Module]
    """Top-k sparse Synaptic MoE with router embeddings, expert fatigue/energy,
    contrastive router-embedding updates, and split/merge structural hooks."""

    def __init__(
        self,
        n_embd: int,
        num_experts: int,
        top_k: int,
        hidden_mult: int,
        cfg: SynapticConfig,
        dropout: float = 0.0,
    ):
        super().__init__()
        object.__setattr__(self, "num_experts", num_experts)
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(self, "cfg", cfg)
        self.router = nn.Linear(n_embd, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                SynapticExpert(n_embd, hidden_mult, cfg, dropout)
                for _ in range(num_experts)
            ]
        )
        # Projects token features into router embedding space for alignment bias
        self.router_probe = nn.Linear(n_embd, cfg.router_embed_dim, bias=False)
        self.register_buffer("fatigue", torch.zeros(num_experts))
        self.register_buffer("energy", torch.ones(num_experts))
        # Per-expert additive routing-logit bias (uta.3). Zero by default => no behavior
        # change. The function-preserving split/merge controller writes -ln2 / +ln2 offsets
        # here so a freshly-split twin pair shares the parent's routing probability mass
        # (each twin gets half), making the lifecycle event exactly output-preserving in the
        # dense regime instead of a discontinuous noisy-clone jump. It is the same per-expert
        # logit bias used by auxiliary-loss-free load balancing, so it is reusable for that.
        self.register_buffer("router_logit_bias", torch.zeros(num_experts))
        self.glial = (
            GlialHomeostasis(
                num_experts,
                group_size=cfg.glial_group_size,
                ema_rate=cfg.glial_ema_rate,
                feedback_rate=cfg.glial_feedback_rate,
                energy_weight=cfg.glial_energy_weight,
                bias_cap=cfg.glial_bias_cap,
            )
            if cfg.glial_homeostasis
            else None
        )
        # Router embeddings (biological identity) with unit-norm constraint
        emb = torch.randn(num_experts, cfg.router_embed_dim)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
        self.router_embeddings = nn.Parameter(
            emb, requires_grad=False
        )  # updated by EMA-style rule
        object.__setattr__(self, "last_aux_loss", None)
        object.__setattr__(self, "last_ctx", {})
        # Per-expert NeuroScore fitness in [0,1], published by NeuroScore.step (de5l).
        # The split/merge controller blends it into health when cfg.use_neuroscore.
        object.__setattr__(self, "last_neuroscore", None)

        # Molecular genetics (yw9.4): compact per-expert Xi plus one shared decoder. The
        # xi_dim=0 ablation retains a learned shared phenotype while removing expert identity.
        self.genome_decoder = SynapticGenomeDecoder(cfg.xi_dim, cfg)
        if cfg.xi_dim > 0:
            self.Xi = nn.Parameter(torch.empty(num_experts, cfg.xi_dim))
            nn.init.normal_(self.Xi, std=0.1)
        else:
            self.register_buffer("Xi", torch.empty(num_experts, 0))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Backward compat (uta.3): checkpoints predating router_logit_bias lack this key.
        # Inject the (zero) default so strict=True loads of old checkpoints don't fail —
        # zero bias reproduces the original routing exactly.
        key = prefix + "router_logit_bias"
        if key not in state_dict:
            state_dict[key] = self.router_logit_bias.detach().clone()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _get_phenotype(self, xi: Tensor) -> Tensor:
        """Decode Xi to bounded biological kinetics (kept for telemetry callers)."""
        return self.genome_decoder(xi)

    def genome_kinetics(self) -> Dict[str, Tensor]:
        """Return the live, named per-expert kinetics for telemetry and evaluation."""
        return self.genome_decoder.kinetics(self.Xi)

    def forward(self, x: Tensor, update_mem: bool = True) -> Tuple[Tensor, Tensor]:
        B, T, C = x.shape
        E = self.num_experts
        device = x.device
        fatigue_buf = self.fatigue
        energy_buf = self.energy

        pheno = self._get_phenotype(self.Xi)  # (E, 6)
        alpha_fatigue = pheno[:, 0]
        alpha_energy = pheno[:, 1]

        logits = self.router(x)  # (B,T,E)

        # Router bias logic (same as before)
        tok_proxy = x.mean(dim=-1, keepdim=True)
        base_bias = 0.02 * tok_proxy.expand(-1, -1, E)
        router_gain = self.router_embeddings.norm(dim=-1).view(1, 1, -1)
        gain_bias = 0.02 * tok_proxy * router_gain
        probe_feat = self.router_probe(x)
        tok_unit = F.normalize(probe_feat, dim=-1)
        router_unit = F.normalize(self.router_embeddings, dim=-1)
        align_bias = 0.02 * torch.einsum("btd,ed->bte", tok_unit, router_unit)
        bias = base_bias + gain_bias + align_bias
        gene_bias = 0.05 * (alpha_energy - alpha_fatigue).view(1, 1, E)

        logits = logits + gene_bias + bias

        # uta.3: per-expert routing bias (zero by default). Carries the function-preserving
        # gate split (-ln2 on a twin pair) so lifecycle events don't perturb the output.
        logits = logits + self.router_logit_bias.view(1, 1, E)

        if self.glial is not None:
            logits = logits + self.glial.routing_bias.view(1, 1, E)

        if self.cfg.enable_metabolism:
            logits = logits + 0.1 * energy_buf.view(1, 1, E) - 0.1 * fatigue_buf.view(1, 1, E)

        topk = min(self.top_k, E)
        g, idx = torch.topk(logits, topk, dim=-1)
        gates = F.softmax(g, dim=-1)

        out = torch.zeros_like(x)
        flat_out = out.view(-1, C)
        flat_x = x.view(-1, C)

        use_fused_genetics = self.cfg.native_genetics and gates.is_cuda

        me = torch.zeros(E, device=device)
        pe = torch.zeros(E, device=device)

        for e in range(E):
            mask = idx == e
            sel = mask.any(dim=-1)
            if not sel.any():
                continue
            flat_idx = sel.view(-1).nonzero(as_tuple=False).squeeze(1)
            x_e = flat_x.index_select(0, flat_idx)

            gene_e = pheno[e]
            energy_e = energy_buf[e]

            y_e = self.experts[e](x_e, energy_override=energy_e, genes=gene_e, update_mem=update_mem)
            w = gates.masked_select(mask).unsqueeze(-1)
            flat_out.index_add_(0, flat_idx, w * y_e)

            if not use_fused_genetics:
                me[e] = sel.sum()
                pe[e] = gates.masked_select(mask).sum()

        with torch.no_grad():
            if use_fused_genetics:
                try:
                    from bio_inspired_nanochat.kernels import (
                        accumulate_router_stats,
                        update_metabolism_fused,
                    )
                    counts, gate_sums = accumulate_router_stats(idx.detach(), gates.detach(), E)
                    me = counts
                    pe = gate_sums
                    # 9mxi: the fatigue/energy EMAs are PERSISTENT adaptation state —
                    # only write them when update_mem; eval must leave them untouched.
                    if update_mem:
                        util = counts.clamp_min(1.0) / float(B * T)
                        update_metabolism_fused(fatigue_buf, energy_buf, alpha_fatigue, alpha_energy, util)
                except ImportError:
                    # Compiled/Triton kernels unavailable: recompute the router
                    # statistics in plain PyTorch. ``me``/``pe`` were never populated
                    # by the expert loop on this path, so reading them here produced
                    # all-zeros — metabolism EMAs collapsed to a constant fixed point
                    # and aux_loss was identically 0 (load balancing silently off).
                    flat_idx0 = idx.detach().reshape(-1)
                    flat_g = gates.detach().reshape(-1)
                    counts_fb = torch.zeros(E, device=device, dtype=me.dtype)
                    counts_fb.index_add_(0, flat_idx0, torch.ones_like(flat_g))
                    sums_fb = torch.zeros(E, device=device, dtype=me.dtype)
                    sums_fb.index_add_(0, flat_idx0, flat_g.to(me.dtype))
                    me = counts_fb
                    pe = sums_fb
                    util = me.clamp_min(1.0) / float(B * T)
                    if update_mem:
                        fatigue_buf.mul_(1.0 - alpha_fatigue).add_(alpha_fatigue * util)
                        energy_buf.mul_(1.0 - alpha_energy).add_(alpha_energy * (1.0 - util))
            else:
                util = me.clamp_min(1.0) / float(B * T)
                if update_mem:
                    fatigue_buf.mul_(1.0 - alpha_fatigue).add_(alpha_fatigue * util)
                    energy_buf.mul_(1.0 - alpha_energy).add_(alpha_energy * (1.0 - util))

            if update_mem and self.glial is not None:
                selection_counts = torch.bincount(
                    idx.detach().reshape(-1), minlength=E
                ).to(energy_buf)
                self.glial.observe(selection_counts, energy_buf)

            last_ctx = {
                "x": x.detach(),
                "indices": idx.detach(),
                "gates": gates.detach()
            }
            if self.glial is not None:
                last_ctx["glial_bias"] = self.glial.routing_bias.detach().clone()
                last_ctx["glial_activity"] = self.glial.activity_ema.detach().clone()
            object.__setattr__(self, "last_ctx", last_ctx)

        me = me / float(B * T)
        pe = pe / float(B * T)
        aux_loss = E * torch.sum(pe * me)
        self.last_aux_loss = aux_loss

        # Lateral-inhibition update on router embeddings (fkkc): each forward, push
        # similar experts' identity vectors apart so experts specialize. The previous
        # formulation built a DIAGONAL-only "co-occurrence" matrix, which made its pull
        # term identically zero (sim[e,e]-1==0) and the net update ineffective — it did
        # not actually reduce pairwise similarity. This is a pure similarity-weighted
        # repulsion that provably spreads the embeddings; gated by router_contrastive_push
        # and router_contrastive_lr (either == 0 disables it).
        with torch.no_grad():
            push, lr = self.cfg.router_contrastive_push, self.cfg.router_contrastive_lr
            if update_mem and E > 1 and push > 0.0 and lr > 0.0:
                emb = self.router_embeddings
                emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
                # Repel each expert from the others it is most aligned with (sim>0).
                sim = emb @ emb.T - torch.eye(E, device=device, dtype=emb.dtype)
                repel = sim.clamp(min=0.0)
                emb = emb - (lr * push) * (repel @ emb)
                emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
                self.router_embeddings.copy_(emb)

        return out, aux_loss


# -----------------------------------------------------------------------------
# Structural plasticity utility
# -----------------------------------------------------------------------------


class StructuralPlasticity(nn.Module):
    cfg: SynapticConfig
    def __init__(self, cfg: SynapticConfig):
        super().__init__()
        object.__setattr__(self, "cfg", cfg)
        self.register_buffer("age", torch.zeros(1))
        self.register_buffer("util", torch.zeros(1))

    @torch.no_grad()
    def step(self, used: Tensor):
        age = self.age
        age.add_(1.0)
        util = self.util
        util.mul_(1.0 - self.cfg.structural_tau_util).add_(
            self.cfg.structural_tau_util * used.float()
        )

    @torch.no_grad()
    def decision(self):
        util = self.util
        age = self.age
        s = torch.sigmoid(
            10.0 * (util - 0.2)
            - self.cfg.structural_age_bias
            * (age / float(self.cfg.structural_interval))
        )
        return (torch.rand_like(s) > s).item()


def structural_plasticity_step(
    expert_states: List[nn.Module], cfg: SynapticConfig, global_step: int
):
    if cfg.structural_interval < 1 or global_step % cfg.structural_interval != 0:
        return
    for st in expert_states:
        st = cast(StructuralPlasticity, st)
        st.step(used=torch.tensor(1.0))
        if st.decision():
            for p in st.parameters():
                nn.init.trunc_normal_(p, std=0.02)
