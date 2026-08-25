"""
Training-loop divergence guards — bead vg9.7.

With stateful, positive-feedback bio mechanisms (BDNF metaplasticity, metabolism, calcium
integration, fast/slow Hebbian weights), silent divergence during training is a real risk:
a single NaN or a runaway feedback loop can corrupt the run with no obvious signal until the
loss is already garbage. This module gives:

1. **Detection** — non-finite (NaN/Inf) on the loss and on the model's bio buffers, plus
   loss-spike (vs an EMA) and bio-buffer-norm explosion detection.
2. **A configurable response** — warn, skip the step (discard the bad gradient), back off the
   learning rate, or roll back to the last-good snapshot (opt-in).
3. **Early warning** — per-step logging of bio-buffer norms so drift is visible before it
   becomes divergence.

The guard is policy-only: `check()` returns a `GuardResult` (action + diagnostics); the caller
(the training loop) applies the action it controls (skip `opt.step()`, scale the LR, or call
`rollback()`). This keeps the guard testable in isolation and the training loop in charge.
"""

from __future__ import annotations

import copy
import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from bio_inspired_nanochat.common import logger

# Buffer/parameter name fragments worth monitoring for divergence (bio state + fast weights).
_BIO_PATTERNS = ("camkii", "pp1", "bdnf", "ema_e", "u_buf", "v_buf", "w_fast", "w_slow")


class GuardAction(str, Enum):
    OK = "ok"
    WARN = "warn"            # log only; keep training
    SKIP = "skip"           # discard this step's gradient (do NOT call opt.step)
    BACKOFF = "backoff"     # scale the LR down for this step
    ROLLBACK = "rollback"   # restore the last-good snapshot (opt-in)


@dataclass
class DivergenceGuardConfig:
    enabled: bool = True
    # --- loss checks ---
    check_loss: bool = True
    loss_spike_factor: float = 8.0   # loss > factor * EMA(loss) counts as a spike
    loss_ema_decay: float = 0.98
    warmup_steps: int = 20           # don't spike-check until the EMA is meaningful
    # --- bio-buffer checks ---
    check_bio_buffers: bool = True
    bio_norm_max: float = 1.0e4      # L2 norm above this = explosion
    # --- actions ---
    on_nonfinite: GuardAction = GuardAction.SKIP   # NaN/Inf on loss or a bio buffer
    on_spike: GuardAction = GuardAction.BACKOFF    # loss spike / norm explosion
    backoff_factor: float = 0.5      # LR multiplier applied by the caller on BACKOFF
    # --- rollback (opt-in; holds a CPU snapshot of model + optimizer state) ---
    enable_rollback: bool = False
    snapshot_every: int = 50
    # --- logging ---
    log_norms_every: int = 1


@dataclass
class GuardResult:
    action: GuardAction
    reason: str
    nonfinite: bool
    norms: Dict[str, float] = field(default_factory=dict)
    loss_ema: float = float("nan")

    @property
    def ok(self) -> bool:
        return self.action == GuardAction.OK


def _as_list(optimizers: Any) -> List[Any]:
    if optimizers is None:
        return []
    if isinstance(optimizers, (list, tuple)):
        return [o for o in optimizers if o is not None]
    return [optimizers]


class DivergenceGuard:
    """Stateful per-run divergence guard. One instance per training run."""

    def __init__(self, cfg: Optional[DivergenceGuardConfig] = None):
        self.cfg = cfg or DivergenceGuardConfig()
        self._loss_ema: Optional[float] = None
        self._snapshot: Optional[tuple] = None
        self._snapshot_step: int = -1

    # -- diagnostics -------------------------------------------------------- #
    def bio_buffer_norms(self, model: nn.Module) -> Dict[str, float]:
        """L2 norms of the monitored bio buffers/params (cheap; a small named subset)."""
        norms: Dict[str, float] = {}
        with torch.no_grad():
            named = list(model.named_buffers()) + list(model.named_parameters())
            for name, t in named:
                low = name.lower()
                if t is None or t.numel() == 0:
                    continue
                if any(p in low for p in _BIO_PATTERNS):
                    norms[name] = float(t.detach().float().norm().item())
        return norms

    # -- the core check ----------------------------------------------------- #
    def check(self, loss: Any, model: nn.Module, *, step: int = 0) -> GuardResult:
        cfg = self.cfg
        norms = self.bio_buffer_norms(model) if cfg.check_bio_buffers else {}
        if not cfg.enabled:
            return GuardResult(GuardAction.OK, "disabled", False, norms, self._ema_or_nan())

        loss_val = float(loss.detach().item()) if isinstance(loss, torch.Tensor) else float(loss)

        # 1) non-finite loss
        if cfg.check_loss and not math.isfinite(loss_val):
            return GuardResult(cfg.on_nonfinite, f"non-finite loss ({loss_val})", True, norms, self._ema_or_nan())

        # 2) bio buffers: non-finite first (cheap, via the norms we already computed), then explosion
        if cfg.check_bio_buffers:
            for nm, v in norms.items():
                if not math.isfinite(v):
                    return GuardResult(cfg.on_nonfinite, f"non-finite bio buffer: {nm}", True, norms, self._ema_or_nan())
            for nm, v in norms.items():
                if v > cfg.bio_norm_max:
                    return GuardResult(cfg.on_spike, f"bio-norm explosion {nm}={v:.3g}", False, norms, self._ema_or_nan())

        # 3) loss spike vs EMA (only after warmup, only against a healthy EMA)
        spiked = (
            cfg.check_loss
            and self._loss_ema is not None
            and step >= cfg.warmup_steps
            and loss_val > cfg.loss_spike_factor * self._loss_ema
        )
        # update the EMA AFTER the spike test, with the (finite) loss
        self._loss_ema = loss_val if self._loss_ema is None else (
            cfg.loss_ema_decay * self._loss_ema + (1.0 - cfg.loss_ema_decay) * loss_val
        )
        if spiked:
            return GuardResult(
                cfg.on_spike,
                f"loss spike {loss_val:.3g} > {cfg.loss_spike_factor:g}x EMA({self._loss_ema:.3g})",
                False, norms, self._loss_ema,
            )
        return GuardResult(GuardAction.OK, "ok", False, norms, self._loss_ema)

    def _ema_or_nan(self) -> float:
        return self._loss_ema if self._loss_ema is not None else float("nan")

    # -- rollback snapshots (opt-in) ---------------------------------------- #
    def maybe_snapshot(self, model: nn.Module, optimizers: Any, step: int) -> None:
        if not self.cfg.enable_rollback:
            return
        # Snapshot immediately on first call (the warmup phase is exactly when
        # divergence risk is highest — deferring to `snapshot_every` used to leave
        # can_rollback() False during the earliest steps), then every interval.
        if self._snapshot is None or step - self._snapshot_step >= self.cfg.snapshot_every:
            self._snapshot = (
                {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()},
                [copy.deepcopy(o.state_dict()) for o in _as_list(optimizers)],
            )
            self._snapshot_step = step

    def can_rollback(self) -> bool:
        return self._snapshot is not None

    def rollback(self, model: nn.Module, optimizers: Any) -> bool:
        if self._snapshot is None:
            return False
        msd, osds = self._snapshot
        opts = _as_list(optimizers)
        if len(opts) != len(osds):
            raise ValueError(
                "rollback optimizer count does not match the last-good snapshot: "
                f"saved={len(osds)}, current={len(opts)}"
            )
        for index, (optimizer, optimizer_state) in enumerate(
            zip(opts, osds, strict=True)
        ):
            if not callable(getattr(optimizer, "load_state_dict", None)):
                raise TypeError(
                    f"rollback optimizer {index} does not implement load_state_dict()"
                )
            if not isinstance(optimizer_state, Mapping):
                raise ValueError(
                    f"rollback optimizer snapshot {index} must be a mapping"
                )
        # Validate every optimizer snapshot before mutating the model or any optimizer.
        model.load_state_dict(msd)
        for o, osd in zip(opts, osds, strict=True):
            o.load_state_dict(osd)
        logger.warning("[divguard] rolled back model + %d optimizer(s) to step %d", len(opts), self._snapshot_step)
        return True

    def state_dict(self) -> Dict[str, Any]:
        """Serialize decision history and the optional last-good rollback snapshot."""
        config = asdict(self.cfg)
        config["on_nonfinite"] = self.cfg.on_nonfinite.value
        config["on_spike"] = self.cfg.on_spike.value
        return {
            "version": 1,
            "config": config,
            "loss_ema": self._loss_ema,
            "snapshot": copy.deepcopy(self._snapshot),
            "snapshot_step": self._snapshot_step,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore guard policy/history, rejecting malformed partial state."""
        if not isinstance(state, Mapping):
            raise TypeError("divergence guard state must be a mapping")
        if state.get("version") != 1:
            raise ValueError("unsupported divergence guard state version")
        raw_config = state.get("config")
        if not isinstance(raw_config, Mapping):
            raise ValueError("divergence guard state config must be a mapping")
        try:
            config_values = dict(raw_config)
            config_values["on_nonfinite"] = GuardAction(config_values["on_nonfinite"])
            config_values["on_spike"] = GuardAction(config_values["on_spike"])
            restored_config = DivergenceGuardConfig(**config_values)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid divergence guard checkpoint config") from error

        loss_ema = state.get("loss_ema")
        if loss_ema is not None and (
            isinstance(loss_ema, bool)
            or not isinstance(loss_ema, numbers.Real)
            or not math.isfinite(float(loss_ema))
        ):
            raise ValueError("divergence guard loss_ema must be finite or None")
        snapshot_step = state.get("snapshot_step")
        if (
            isinstance(snapshot_step, bool)
            or not isinstance(snapshot_step, int)
            or snapshot_step < -1
        ):
            raise ValueError("divergence guard snapshot_step must be an integer >= -1")
        snapshot = state.get("snapshot")
        if snapshot is not None:
            if snapshot_step < 0:
                raise ValueError("divergence guard snapshot requires a non-negative step")
            if not isinstance(snapshot, Sequence) or len(snapshot) != 2:
                raise ValueError("divergence guard snapshot must contain model and optimizer state")
            model_state, optimizer_states = snapshot
            if not isinstance(model_state, Mapping) or not isinstance(
                optimizer_states, Sequence
            ):
                raise ValueError("divergence guard snapshot payload is malformed")
            if any(not isinstance(item, Mapping) for item in optimizer_states):
                raise ValueError("divergence guard optimizer snapshots must be mappings")
        elif snapshot_step != -1:
            raise ValueError("divergence guard snapshot_step requires a snapshot")

        self.cfg = restored_config
        self._loss_ema = None if loss_ema is None else float(loss_ema)
        self._snapshot = copy.deepcopy(snapshot)
        self._snapshot_step = snapshot_step

    # -- logging ------------------------------------------------------------ #
    def log(self, result: GuardResult, step: int) -> None:
        cfg = self.cfg
        if result.norms and step % max(1, cfg.log_norms_every) == 0:
            top = sorted(result.norms.items(), key=lambda kv: -kv[1])[:4]
            pretty = ", ".join(f"{k.split('.')[-1]}={v:.3g}" for k, v in top)
            logger.info("[divguard step %d] bio-norms %s | loss_ema=%.4g", step, pretty, result.loss_ema)
        if result.action != GuardAction.OK:
            logger.warning("[divguard step %d] %s -> action=%s", step, result.reason, result.action.value)


def build_divergence_guard(decouple_config: Optional[Any] = None) -> DivergenceGuard:
    """Build a guard from the project's python-decouple config (env-overridable), or defaults.

    Reads BIO_DIVGUARD_ENABLED / _ON_NONFINITE / _ON_SPIKE / _SPIKE_FACTOR / _ROLLBACK when a
    decouple config is supplied; otherwise returns a guard with sensible defaults.
    """
    cfg = DivergenceGuardConfig()
    if decouple_config is not None:
        cfg.enabled = decouple_config("BIO_DIVGUARD_ENABLED", default=cfg.enabled, cast=bool)
        cfg.loss_spike_factor = decouple_config("BIO_DIVGUARD_SPIKE_FACTOR", default=cfg.loss_spike_factor, cast=float)
        cfg.enable_rollback = decouple_config("BIO_DIVGUARD_ROLLBACK", default=cfg.enable_rollback, cast=bool)
        try:
            cfg.on_nonfinite = GuardAction(decouple_config("BIO_DIVGUARD_ON_NONFINITE", default=cfg.on_nonfinite.value))
            cfg.on_spike = GuardAction(decouple_config("BIO_DIVGUARD_ON_SPIKE", default=cfg.on_spike.value))
        except ValueError:
            logger.warning("[divguard] invalid action in env; keeping defaults")
    return DivergenceGuard(cfg)
