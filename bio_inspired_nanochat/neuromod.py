# bio_inspired_nanochat/neuromod.py
# Global neuromodulatory bus — DA / ACh / NE (bead hy8.1)
#
# Real synaptic plasticity is GATED by a few GLOBAL neuromodulator signals broadcast to every
# synapse. This module computes three such scalars from model-level signals each step and
# broadcasts multiplicative gains onto the synaptic layers, which read them default-neutrally
# (a layer with no broadcast gain behaves exactly as before):
#
#   • Dopamine (DA) ≈ reward-prediction error  -> PLASTICITY gain. Scales the online Hebbian
#       consolidation step so only reward-relevant / loss-reducing updates are amplified. This is
#       the canonical bridge to reinforcement learning and the third factor for hy8.2.
#   • Acetylcholine (ACh) ≈ uncertainty/attention -> EXPLORATION gain. Scales the stochastic
#       vesicle-release fraction (SynapticPresyn): more uncertainty -> more stochastic release.
#   • Norepinephrine (NE) ≈ arousal/novelty -> GLOBAL gain (and optional reset). Scales the
#       synaptic output and, above a threshold, can trigger a per-sequence reset.
#
# Signals are computed from {loss, entropy, reward}, smoothed by EMAs, squashed to bounded
# levels, then mapped to bounded multiplicative gains. Disabled by default-neutral gains (1.0).

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from bio_inspired_nanochat.torch_imports import torch, nn, Tensor
from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticLinear, SynapticPresyn


@dataclass
class NeuromodConfig:
    enabled: bool = True
    ema_tau: float = 0.9  # EMA smoothing for running baselines AND for the levels
    # signal -> level sensitivities (pre-tanh)
    da_k: float = 2.0
    ach_k: float = 1.0
    ne_k: float = 1.0
    # level -> multiplicative gain: gain = clamp(1 + slope * level, range)
    da_gain_slope: float = 1.0
    ach_gain_slope: float = 2.0
    ne_gain_slope: float = 0.5
    # ACh also drives an INPUT/attention gain (hy8.5): uncertainty sharpens input sensitivity.
    ach_input_slope: float = 1.0
    da_gain_range: Tuple[float, float] = (0.0, 4.0)
    ach_gain_range: Tuple[float, float] = (0.0, 4.0)
    ach_input_range: Tuple[float, float] = (0.5, 2.0)
    ne_gain_range: Tuple[float, float] = (0.5, 2.0)
    # NE novelty above this triggers a per-sequence reset on broadcast (0 = disabled)
    novelty_reset_thresh: float = 0.0


class NeuromodulatoryBus(nn.Module):
    """Computes and broadcasts global DA/ACh/NE neuromodulator gains (hy8.1).

    Usage::

        bus = NeuromodulatoryBus()
        bus.update(loss=step_loss, entropy=pred_entropy, reward=None)  # compute levels
        bus.broadcast(model)                                          # gate the synapses
        logger.log(bus.telemetry())                                   # expose the scalars
    """

    def __init__(self, cfg: Optional[NeuromodConfig] = None):
        super().__init__()
        self.cfg = cfg or NeuromodConfig()
        nan = float("nan")
        # running baselines for surprise / RPE
        self.register_buffer("loss_ema", torch.tensor(nan))
        self.register_buffer("reward_ema", torch.tensor(nan))
        self.register_buffer("entropy_ema", torch.tensor(nan))
        # neuromodulator levels (bounded): da in [-1,1], ach/ne in [0,1]
        self.register_buffer("da", torch.zeros(()))
        self.register_buffer("ach", torch.zeros(()))
        self.register_buffer("ne", torch.zeros(()))

    @staticmethod
    def _ema(prev: Tensor, x: Tensor, tau: float) -> Tensor:
        # First observation seeds the EMA (prev is NaN until then).
        if bool(torch.isnan(prev)):
            return x.clone()
        return tau * prev + (1.0 - tau) * x

    @staticmethod
    @torch.no_grad()
    def entropy_from_logits(logits: Tensor) -> float:
        """Mean per-token predictive entropy (nats) — a convenient ACh signal."""
        logp = torch.log_softmax(logits.detach().float().reshape(-1, logits.shape[-1]), dim=-1)
        ent = -(logp.exp() * logp).sum(dim=-1).mean()
        return float(ent)

    @torch.no_grad()
    def update(
        self,
        *,
        loss: Optional[float] = None,
        entropy: Optional[float] = None,
        reward: Optional[float] = None,
    ) -> Dict[str, float]:
        cfg = self.cfg
        if not cfg.enabled:
            return self.levels()
        tau = cfg.ema_tau
        z = torch.zeros(())

        # --- Dopamine: reward-prediction error (or loss-improvement proxy) ---
        if reward is not None:
            r = torch.as_tensor(float(reward))
            rpe = r.clone() if bool(torch.isnan(self.reward_ema)) else (r - self.reward_ema)
            self.reward_ema.copy_(self._ema(self.reward_ema, r, tau))
            da_signal = rpe
        elif loss is not None:
            cur = torch.as_tensor(float(loss))
            # improvement = baseline - current (positive when loss drops => reward-like)
            da_signal = z if bool(torch.isnan(self.loss_ema)) else (self.loss_ema - cur)
        else:
            da_signal = z

        # --- Norepinephrine: novelty = |loss surprise| (uses the OLD baseline, then updates it) ---
        ne_signal = z
        if loss is not None:
            cur = torch.as_tensor(float(loss))
            ne_signal = z if bool(torch.isnan(self.loss_ema)) else (cur - self.loss_ema).abs()
            self.loss_ema.copy_(self._ema(self.loss_ema, cur, tau))

        # --- Acetylcholine: uncertainty = entropy relative to its baseline ---
        ach_signal = z
        if entropy is not None:
            h = torch.as_tensor(float(entropy))
            ach_signal = h.clone() if bool(torch.isnan(self.entropy_ema)) else (h - self.entropy_ema)
            self.entropy_ema.copy_(self._ema(self.entropy_ema, h, tau))

        # squash to bounded levels and EMA-smooth them
        self.da.copy_(self._ema(self.da, torch.tanh(cfg.da_k * da_signal), tau))
        self.ach.copy_(self._ema(self.ach, torch.relu(torch.tanh(cfg.ach_k * ach_signal)), tau))
        self.ne.copy_(self._ema(self.ne, torch.relu(torch.tanh(cfg.ne_k * ne_signal)), tau))
        return self.levels()

    def levels(self) -> Dict[str, float]:
        return {"da": float(self.da), "ach": float(self.ach), "ne": float(self.ne)}

    def gains(self) -> Dict[str, float]:
        cfg = self.cfg

        def _clamp(v: float, rng: Tuple[float, float]) -> float:
            return float(min(rng[1], max(rng[0], v)))

        return {
            "plasticity": _clamp(1.0 + cfg.da_gain_slope * float(self.da), cfg.da_gain_range),
            "explore": _clamp(1.0 + cfg.ach_gain_slope * float(self.ach), cfg.ach_gain_range),
            "attend": _clamp(1.0 + cfg.ach_input_slope * float(self.ach), cfg.ach_input_range),
            "global": _clamp(1.0 + cfg.ne_gain_slope * float(self.ne), cfg.ne_gain_range),
        }

    @torch.no_grad()
    def broadcast(self, model: nn.Module) -> int:
        """Set the multiplicative neuromodulator gains on every synaptic layer. Returns the
        number of modules touched. Gains of 1.0 are exactly the un-modulated behavior."""
        g = self.gains()
        n = 0
        for m in model.modules():
            if isinstance(m, SynapticLinear):
                m._nm_da_gain = g["plasticity"]
                m._nm_ne_gain = g["global"]
                m._nm_ach_input_gain = g["attend"]  # ACh attention/input gain (hy8.5)
                n += 1
            elif isinstance(m, SynapticPresyn):
                m._nm_ach_gain = g["explore"]
                n += 1
        # NE novelty-triggered arousal reset (opt-in): a surprising event flushes the
        # per-sequence working memory (fast weights + eligibility traces).
        if (
            self.cfg.novelty_reset_thresh > 0.0
            and float(self.ne) > self.cfg.novelty_reset_thresh
            and hasattr(model, "reset_sequence_state")
        ):
            model.reset_sequence_state(reset_fast_weights=True)
        return n

    def telemetry(self) -> Dict[str, float]:
        out = {f"nm/{k}": v for k, v in self.levels().items()}
        out.update({f"nm/gain_{k}": v for k, v in self.gains().items()})
        return out


def bus_for_config(
    syn_cfg: SynapticConfig, cfg: Optional[NeuromodConfig] = None
) -> Optional[NeuromodulatoryBus]:
    """The training-harness hook behind ``SynapticConfig.neuromod_enabled``.

    Returns a fresh :class:`NeuromodulatoryBus` when the synaptic config asks for one and
    ``None`` otherwise, so ``scripts/base_train.py`` and ``scripts/eval_matrix.py`` build the
    bus from the same flag instead of each re-deriving it.
    """
    if not bool(syn_cfg.neuromod_enabled):
        return None
    return NeuromodulatoryBus(cfg)
