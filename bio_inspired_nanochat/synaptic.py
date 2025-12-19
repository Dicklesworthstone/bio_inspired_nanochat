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

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Literal, cast, Any

from bio_inspired_nanochat.torch_imports import torch, nn, F, Tensor
from bio_inspired_nanochat.common import decouple_config

try:
    from .flex_synaptic import SynapticFlexAttention
    _HAS_FLEX = True
except ImportError:
    _HAS_FLEX = False


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
        samp = mean + std * torch.randn_like(mean)
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
    u = torch.rand((*probs_32.shape, max_count), device=probs.device, dtype=torch.float32)
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
    enabled: bool = True
    # General
    rank_eligibility: int = 8
    attn_topk: int = 32
    stochastic_train_frac: float = 0.12
    stochastic_mode: Literal["gumbel_sigmoid_ste", "straight_through", "normal_reparam"] = (
        "normal_reparam"
    )
    stochastic_tau: float = 1.0
    stochastic_count_cap: int = 8
    
    # Presynaptic Biophysics
    tau_c: float = 0.85
    alpha_c: float = 0.55
    syt1_slope: float = 8.0
    syt7_slope: float = 3.0
    cpx_thresh: float = 0.55
    doc2_gain: float = 0.08
    prime_rate: float = 0.075
    unprime_per_release: float = 0.05
    nsf_recover: float = 0.08
    rec_rate: float = 0.06
    endo_delay: int = 3
    amp_load: float = 0.02
    amp_leak: float = 0.006
    
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
    
    # Rust Kernel Compat
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
    camkii_up: float = 0.05
    camkii_down: float = 0.02
    pp1_tau: float = 0.985
    camkii_thr: float = 1.0
    pp1_thr: float = 0.7
    bdnf_tau: float = 0.985
    bdnf_scale: float = 1.0
    bdnf_gamma: float = 0.0  # When > 0, BDNF accumulates |ΔW_hebb| instead of CaMKII activity
    bdnf_hebb_accumulate: bool = True  # Use Hebbian delta magnitude for BDNF (vs CaMKII)

    # Structural Plasticity (MoE)
    structural_interval: int = 50000
    structural_tau_util: float = 0.2
    structural_age_bias: float = 1.0
    router_embed_dim: int = 24
    router_contrastive_lr: float = 1e-4
    router_contrastive_push: float = 0.1
    router_sim_threshold: float = 0.6

    # Genetics
    # Per-expert genome embedding (Xi). A decoder maps Xi -> phenotype scalars that
    # control expert-specific kinetics without storing a full per-expert copy of
    # every kinetic parameter.
    xi_dim: int = 4  # [alpha_fatigue, alpha_energy, camkii_gain, pp1_gain]

    # Feature Toggles (Modular Control)
    enable_presyn: bool = True
    enable_hebbian: bool = True
    enable_metabolism: bool = True
    use_flex_attention: bool = False
    
    # Native (Rust) Kernel Toggles
    native_presyn: bool = decouple_config("BIO_FUSED_PRESYN", default=False, cast=bool)
    native_metrics: bool = decouple_config("BIO_FUSED_METRICS", default=False, cast=bool)
    native_genetics: bool = decouple_config("BIO_FUSED_GENETICS", default=False, cast=bool)
    native_plasticity: bool = decouple_config("BIO_FUSED_PLASTICITY", default=False, cast=bool)


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

    def _mix_prob(self, c: Tensor, clamp: Tensor, sn: Tensor) -> Tensor:
        p1 = torch.sigmoid(self.cfg.syt1_slope * (c - 0.55))
        p7 = torch.sigmoid(self.cfg.syt7_slope * (c - 0.25))
        p = p1 * 0.8 + p7 * 0.2 + self.cfg.doc2_gain * torch.sigmoid(4 * (c - 0.12))
        # Complexin clamp: incorporate the learned/ema clamp state (and bias) as an inhibitory term.
        # This matches the intent of the fused kernel path where clamp reduces release.
        cpx_gate = torch.sigmoid(
            8.0 * (c - self.cfg.cpx_thresh) - 2.0 * (clamp + self.cfg.complexin_bias)
        )
        p = p * cpx_gate * sn
        return torch.clamp(p, 0, 0.999)

    def release(
        self,
        state: Dict[str, Any],
        drive: Tensor,
        idx: Tensor,
        train: bool,
        valid: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute release and update state.
        drive: (B, H, T, K) - attention logits for top-k
        idx: (B, H, T, K) - indices of top-k keys

        Stochastic mode (training only):
        - Enabled when `train=True` and `cfg.stochastic_train_frac > 0`.
        - Samples Binomial(n, p) counts with a straight-through relaxation so gradients can
          still flow through `p(drive, state)` during backprop.
        - For performance, `n` is rounded and capped by `cfg.stochastic_count_cap`.
        """
        if not self.cfg.enable_presyn:
            # Return 1.0 so log(e) approx 0
            return torch.ones_like(drive)

        B, H, T, K = drive.shape
        cfg = self.cfg
        if valid is not None and valid.shape != drive.shape:
            raise ValueError(
                f"valid mask must match drive shape {drive.shape}, got {valid.shape}"
            )

        # Gather per-edge state for the selected keys
        flat_idx = idx.view(B, H, -1)

        c = state["C"].gather(2, flat_idx).view(B, H, T, K)
        c = cfg.tau_c * c + cfg.alpha_c * F.softplus(drive)

        sn = state["PR"].gather(2, flat_idx).view(B, H, T, K)
        clamp = state["CL"].gather(2, flat_idx).view(B, H, T, K)

        p = self._mix_prob(c, clamp, sn)
        rrp = state["RRP"].gather(2, flat_idx).view(B, H, T, K)

        if train and cfg.stochastic_train_frac > 0:
            # Stochastic release on a fraction of *query positions* (broadcast across top-k edges).
            do_stoch = torch.rand_like(p[..., 0].to(torch.float32)) < float(
                cfg.stochastic_train_frac
            )
            rel_det = p * rrp
            if do_stoch.any():
                stoch_mask = do_stoch.unsqueeze(-1).expand_as(p)
                k_rel = _sample_binomial_counts(
                    probs=p[stoch_mask],
                    total_count=torch.clamp(
                        rrp[stoch_mask], 0.0, float(cfg.stochastic_count_cap)
                    ),
                    max_count=int(cfg.stochastic_count_cap),
                    tau=float(cfg.stochastic_tau),
                    mode=cfg.stochastic_mode,
                )
                rel = rel_det.clone()
                rel[stoch_mask] = k_rel
            else:
                rel = rel_det
        else:
            rel = p * rrp

        amp = state["AMP"].gather(2, flat_idx).view(B, H, T, K)
        if valid is not None:
            rel = rel * valid.to(rel.dtype)
        e = rel * amp

        # Efficient scatter:
        # We want to add 'rel' (B,H,T,K) to 'state' (B,H,T_key) at indices 'idx' (B,H,T,K).
        # We can flatten T,K.
        
        # flat_idx = idx.view(B, H, -1) # (B, H, T*K)
        flat_rel = rel.view(B, H, -1)
        flat_drive = drive.view(B, H, -1)
        flat_amp = amp.view(B, H, -1)
        
        # Accumulators
        add_vals = torch.zeros_like(state["C"])  # (B, H, T_key)
        drv_vals = torch.zeros_like(state["C"])
        snu_vals = torch.zeros_like(state["C"])
        rru_vals = torch.zeros_like(state["C"])
        ampu_vals = torch.zeros_like(state["C"])
        
        # scatter_add_ expects index to have same number of dimensions as self.
        # self is (B, H, T_key). flat_idx is (B, H, T*K).
        # We need to expand self to match index dims? No, scatter_add_ reduces.
        # "self.scatter_add_(dim, index, src)"
        # "index: the indices of elements to scatter, can be either empty or of the same dimensionality as src."
        # "src: the source element(s) to scatter."
        # "self: the destination tensor."
        # "index" and "src" must have same size.
        # "self" must be large enough to hold the scattered values.
        # BUT: "index" must have same number of dimensions as "self".
        # Here self is 3D (B, H, T_key). index is 3D (B, H, T*K).
        # This is valid IF T_key dimension in self is large enough for indices in index.
        # The error "Expected self.dtype to be equal to src.dtype" suggests a type mismatch.
        # state["c"] is likely float32 or bfloat16.
        # flat_rel is derived from p * rrp.
        # Let's check dtypes.
        
        # Ensure dtypes match
        dtype = state["C"].dtype
        flat_rel = flat_rel.to(dtype)
        flat_drive = flat_drive.to(dtype)
        flat_amp = flat_amp.to(dtype)
        if valid is not None:
            flat_valid_bool = valid.view(B, H, -1)
            flat_valid = flat_valid_bool.to(dtype)
            # Avoid NaNs from (-inf * 0) when invalid edges are masked to -inf upstream.
            flat_drive = torch.where(flat_valid_bool, flat_drive, torch.zeros_like(flat_drive))
            flat_amp = flat_amp * flat_valid
        else:
            flat_valid = torch.ones_like(flat_rel)
        
        add_vals.scatter_add_(2, flat_idx, flat_rel)
        drv_vals.scatter_add_(2, flat_idx, flat_drive)
        snu_vals.scatter_add_(2, flat_idx, flat_valid) # Count of accesses
        rru_vals.scatter_add_(2, flat_idx, flat_rel)
        ampu_vals.scatter_add_(2, flat_idx, flat_amp)
        
        # Update dynamics
        accessed = snu_vals > 0
        c_up = (
            cfg.tau_c * state["C"]
            + cfg.alpha_c * F.softplus(drv_vals) * accessed.to(dtype)
        )
        rrp_up = torch.clamp(state["RRP"] - add_vals, 0)
        
        # Endocytosis delay queue
        res_up = state["RES"] + state["DELAY"][0]
        new_delay = state["DELAY"][1:] + [rru_vals * cfg.rec_rate]
        
        # Priming
        take = torch.minimum(res_up, torch.ones_like(res_up)) # Max 1 unit per step? Or just soft clamp?
        # PDF: take=torch.minimum(res_up, torch.ones_like(res_up))
        res_up = torch.clamp(res_up - cfg.prime_rate * take, 0)
        rrp_up = torch.clamp(rrp_up + cfg.prime_rate * take, 0, 30.0) # Cap RRP
        
        # SNARE / Clamp / AMPA / Energy
        sn_up = torch.clamp(
            state["PR"] * (1.0 - cfg.unprime_per_release * add_vals)
            + cfg.nsf_recover * (1.0 - state["PR"]),
            0,
            1,
        )
        cl_up = torch.clamp(
            state["CL"] * 0.995 + 0.005 - cfg.unprime_per_release * add_vals,
            0,
            1,
        )
        amp_up = torch.clamp(
            state["AMP"] + cfg.amp_load * (1.2 - state["AMP"]) - cfg.amp_leak * state["AMP"],
            0,
            2,
        )
        en_up = torch.clamp(
            state["E"]
            + cfg.energy_fill * (cfg.energy_max - state["E"])
            - cfg.energy_use * add_vals,
            0,
            cfg.energy_max,
        )
        
        state.update({
            "C": c_up,
            "RRP": rrp_up,
            "RES": res_up,
            "DELAY": new_delay,
            "PR": sn_up,
            "CL": cl_up,
            "AMP": amp_up,
            "E": en_up,
            "BUF": state["BUF"],
        })
        
        # EMA normalization
        s = e.detach().abs().mean().clamp_min(1e-3)
        self.ema_e.mul_(0.99).add_(0.01 * s)
        
        return e / (self.ema_e + 1e-6)

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

        This is used for kernel correctness tests and visualization utilities.
        It implements the same governing equations as the Rust/Triton reference paths
        (i.e., it uses the "Rust Kernel Compat" parameters in SynapticConfig).
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
        up = (ca_proxy > self.cfg.camkii_thr).float()
        down = (ca_proxy < self.cfg.pp1_thr).float()

        camkii_up = self.cfg.camkii_up
        pp1_rate = 1.0 - self.cfg.pp1_tau
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
            # NaN guard
            if torch.isnan(self.bdnf).any():
                self.bdnf.zero_()
        else:
            # Legacy mode: BDNF tracks CaMKII activity
            self.bdnf.mul_(self.cfg.bdnf_tau).add_(
                (1 - self.cfg.bdnf_tau) * F.relu(self.camkii - 0.5)
            )

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
            # NaN guard for accumulator
            if torch.isnan(self.bdnf_hebb_accum).any():
                self.bdnf_hebb_accum.zero_()

        # CaMKII gate: consolidation only when CaMKII > 0.5
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
            
            # Eligibility buffers
            self.register_buffer("u_buf", torch.zeros(in_features, cfg.rank_eligibility))
            self.register_buffer("v_buf", torch.zeros(cfg.rank_eligibility, out_features))
        else:
            self.register_parameter("w_fast", None)
            self.post = None
            self.register_buffer("u_buf", None)
            self.register_buffer("v_buf", None)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.trunc_normal_(self.w_slow, std=0.02)
        
        if use_input_ln:
            self.input_ln = nn.LayerNorm(in_features, eps=1e-5)
        else:
            object.__setattr__(self, "input_ln", None)

    def forward(
        self, x: Tensor, calcium: Tensor, energy: Tensor, update_mem: bool = True, genes: Optional[Tensor] = None
    ):
        if self.input_ln is not None:
            x = self.input_ln(x)
            
        # Linear pass
        if self.cfg.enable_hebbian and self.w_fast is not None:
            W = self.w_slow + self.w_fast
        else:
            W = self.w_slow

        y = x @ W
        if self.bias is not None:
            y = y + self.bias
            
        # Postsynaptic modulation (diagonal fast/slow + low-rank)
        if self.cfg.enable_hebbian and self.post is not None:
            y = self.post(y)
        
            if update_mem:
                with torch.no_grad():
                    # Update eligibility traces
                    # u_buf: (in, R) <- x (B, in)
                    # v_buf: (R, out) <- y (B, out)
                    # We need to project x and y to rank R?
                    # Or we accumulate outer products?
                    # PDF: self.u_buf...add_(... einsum...)
                    # Let's implement a simple Hebbian accumulation
                    
                    # Random projection for eligibility? Or learned?
                    # The PDF PostsynapticHebb has U and V parameters.
                    # We can use those to project.
                    
                    # Actually, let's just use the mean activity for now to keep it simple and fast
                    u_mean = x.mean(0) # (in,)
                    v_mean = y.mean(0) # (out,)
                    
                    # We need (in, R) and (R, out).
                    # We can just broadcast or rotate.
                    # Let's just update the buffers with a decay
                    
                    # Update logic from old code was:
                    # U.mul_(rho).add_(eta * u.unsqueeze(-1))
                    # V.mul_(rho).add_(eta * v.unsqueeze(0))
                    # This creates rank-1 updates.
                    
                    # We will do similar here but on u_buf/v_buf
                    if self.u_buf is not None and self.v_buf is not None:
                        self.u_buf.mul_(self.cfg.post_trace_decay).add_(0.05 * u_mean.unsqueeze(-1).expand(-1, self.cfg.rank_eligibility))
                        self.v_buf.mul_(self.cfg.post_trace_decay).add_(0.05 * v_mean.unsqueeze(0).expand(self.cfg.rank_eligibility, -1))
                    
                    # Update Postsynaptic state
                    # We need a vector for per-neuron update?
                    # y is (B, out). ca_proxy should be (out,).
                    ca_vec = y.abs().mean(0).clamp(0, 10.0)
                    
                    self.post.update(y, ca_vec, genes=genes)
                    if self.u_buf is not None and self.v_buf is not None:
                        self.post.hebb_fast(self.u_buf, self.v_buf)
                        self.post.consolidate(self.u_buf, self.v_buf)

        return y


# -----------------------------------------------------------------------------
# Presyn state builder
# -----------------------------------------------------------------------------


def build_presyn_state(B: int, T: int, H: int, device, dtype, cfg: SynapticConfig):
    state_shape = (B, H, T)
    ones = torch.ones(state_shape, device=device, dtype=dtype)
    zeros = torch.zeros(state_shape, device=device, dtype=dtype)
    return {
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
            _ = self.pre.release(presyn_state, vals, idx, train_mode, valid=valid)

            from torch.nn.attention.flex_attention import create_block_mask

            def causal_mask(_b, _h, q_idx, kv_idx):
                return q_idx >= kv_idx

            block_mask = create_block_mask(causal_mask, B, H, Tq, Tk, device=device)

            if q.dtype != v_full.dtype:
                q = q.to(v_full.dtype)
            if k_full.dtype != v_full.dtype:
                k_full = k_full.to(v_full.dtype)

            y = self.flex(q, k_full, v_full, presyn_state, block_mask=block_mask)
            y = y.transpose(1, 2).contiguous().view(B, Tq, H * D)
            y = self.resid_drop(self.o_proj(y))
            return y, presyn_state

        # --- Standard Path ---
        topk = min(self.cfg.attn_topk, Tk)
        vals, idx = torch.topk(dots, topk, dim=-1)
        valid = torch.isfinite(vals)

        # Run presynaptic physics on only the valid edges.
        e = self.pre.release(presyn_state, vals, idx, train_mode, valid=valid)

        # Scatter biological log-bias back into the logits, preserving masking.
        aug = torch.zeros_like(dots)
        src_val = self.cfg.lambda_loge * torch.log(self.cfg.epsilon + e).to(aug.dtype)
        src_val = src_val * valid.to(src_val.dtype)
        aug.scatter_add_(-1, idx, src_val)

        # Septin-like distance barrier in global positions.
        q_pos = torch.arange(prefix_len, prefix_len + Tq, device=device, dtype=torch.float32)
        k_pos = torch.arange(0, Tk, device=device, dtype=torch.float32)
        dist = (q_pos[:, None] - k_pos[None, :]).abs() / float(max(1, Tk))
        logits = dots + aug - (self.cfg.barrier_strength * dist.to(dots.dtype)).view(
            1, 1, Tq, Tk
        )

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

    def forward(self, x: Tensor):
        B, T, C = x.shape
        c0 = self.C0
        e0 = self.E0
        c = c0.expand(B * T)
        e = e0.expand(B * T)
        h = self.fc(x.reshape(B * T, C), c, e)
        h = F.relu(h).square()
        h = self.drop(h.reshape(B, T, -1))
        y = self.proj(h.reshape(B * T, -1), c, e).reshape(B, T, C)
        return y


# -----------------------------------------------------------------------------
# Synaptic MoE (router embeddings, contrastive updates)
# -----------------------------------------------------------------------------


class SynapticExpert(nn.Module):
    def __init__(
        self, n_embd: int, hidden_mult: int, cfg: SynapticConfig, dropout: float = 0.0
    ):
        super().__init__()
        h = hidden_mult * n_embd
        self.fc1 = SynapticLinear(n_embd, h, cfg, bias=True, use_input_ln=False)
        self.fc2 = SynapticLinear(h, n_embd, cfg, bias=True, use_input_ln=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, energy_override: Optional[Tensor] = None, genes: Optional[Tensor] = None) -> Tensor:
        # x: (N, C)
        N = x.size(0)
        device = x.device
        
        if energy_override is not None:
            if energy_override.ndim == 0:
                e_tens = energy_override.expand(N)
            else:
                e_tens = energy_override.view(-1).expand(N)
        else:
            e_tens = torch.ones(N, device=device)
            
        c_tens = torch.ones(N, device=device)

        y = self.fc1(
            x,
            calcium=c_tens,
            energy=e_tens,
            genes=genes,
        )
        y = F.relu(y).square()
        y = self.drop(y)
        y = self.fc2(
            y,
            calcium=c_tens,
            energy=e_tens,
            genes=genes,
        )
        return y


class SynapticMoE(nn.Module):
    num_experts: int
    top_k: int
    cfg: SynapticConfig
    last_aux_loss: Optional[Tensor]
    last_ctx: Dict[str, Tensor]
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
        # Router embeddings (biological identity) with unit-norm constraint
        emb = torch.randn(num_experts, cfg.router_embed_dim)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
        self.router_embeddings = nn.Parameter(
            emb, requires_grad=False
        )  # updated by EMA-style rule
        object.__setattr__(self, "last_aux_loss", None)
        object.__setattr__(self, "last_ctx", {})
        
        # Molecular Genetics: Xi (The Genome)
        self.Xi = nn.Parameter(torch.zeros(num_experts, cfg.xi_dim)) 
        nn.init.normal_(self.Xi, std=0.1)

    def _get_phenotype(self, xi: Tensor) -> Tensor:
        """Map Xi logits to biological range constants."""
        fatigue_rate = 0.01 * (torch.sigmoid(xi[..., 0]) * 2.0 + 0.1)
        energy_fill = 0.005 * (torch.sigmoid(xi[..., 1]) * 2.0 + 0.1)
        camkii_gain = F.softplus(xi[..., 2] + 1.0)
        pp1_gain = F.softplus(xi[..., 3] + 0.5)
        return torch.stack([fatigue_rate, energy_fill, camkii_gain, pp1_gain], dim=-1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        B, T, C = x.shape
        E = self.num_experts
        device = x.device
        fatigue_buf = self.fatigue
        energy_buf = self.energy
        
        pheno = self._get_phenotype(self.Xi) # (E, 4)
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
            
            y_e = self.experts[e](x_e, energy_override=energy_e, genes=gene_e)
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
                    util = counts.clamp_min(1.0) / float(B * T)
                    update_metabolism_fused(fatigue_buf, energy_buf, alpha_fatigue, alpha_energy, util)
                    me = counts
                    pe = gate_sums
                except ImportError:
                    # Fallback
                    util = me.clamp_min(1.0) / float(B * T)
                    fatigue_buf.mul_(1.0 - alpha_fatigue).add_(alpha_fatigue * util)
                    energy_buf.mul_(1.0 - alpha_energy).add_(alpha_energy * (1.0 - util))
            else:
                util = me.clamp_min(1.0) / float(B * T)
                fatigue_buf.mul_(1.0 - alpha_fatigue).add_(alpha_fatigue * util)
                energy_buf.mul_(1.0 - alpha_energy).add_(alpha_energy * (1.0 - util))

            object.__setattr__(self, "last_ctx", {
                "x": x.detach(),
                "indices": idx.detach(),
                "gates": gates.detach()
            })

        me = me / float(B * T)
        pe = pe / float(B * T)
        aux_loss = E * torch.sum(pe * me)
        self.last_aux_loss = aux_loss

        # Contrastive router-embedding update
        with torch.no_grad():
            cooc = torch.zeros(E, E, device=device)
            for e in range(E):
                cooc[e, e] = pe[e]
            emb = self.router_embeddings
            emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
            sim = emb @ emb.T
            pull = cooc * (sim - 1.0)
            push = (1.0 - cooc) * (sim + 0.3) * self.cfg.router_contrastive_push
            grad = pull - push
            grad = grad - grad.mean()
            grad = grad.to(emb.dtype)
            delta = (grad @ emb) * self.cfg.router_contrastive_lr
            emb = emb - delta
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
