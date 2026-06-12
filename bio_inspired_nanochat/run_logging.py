"""
Structured run logging (bead eqyk.2).

The detailed, machine-readable logging backbone for the bio stack. Every later
"monitor"/"detailed logging" bead (the metriplectic free-energy monitor, the
retention certificate, the metabolism EMAs, the sheaf obstruction, ...) writes its
per-step bio-state, metrics, and provenance through this module so that runs are
**observable and post-hoc verifiable**: after implementing a stateful bio mechanism
you can *see* its internals evolving and confirm correctness.

Design
------
- Builds on ``bio_inspired_nanochat.common``'s ``logging`` setup (does NOT replace
  it). The human-readable console stream goes through the standard ``logging``
  logger (with common's ColoredFormatter); the new value is a parseable JSONL
  **event stream**.
- Two outputs per run:
    1. ``<run_dir>/events.jsonl`` — one JSON object per line: the source of truth.
    2. console — a compact, level-colored mirror (best effort).
- ``run_id`` correlation id + provenance (git SHA, torch/cuda, host, config hash)
  stamped on every run so logs join up with the results registry (bead hm4.1).
- **Cheap by default**: tensor summaries are computed only when you call a log
  method; nothing is materialized at import or construction.

Quick use
---------
    from bio_inspired_nanochat.run_logging import RunLogger
    with RunLogger(run_dir, name="train", provenance={"config_hash": h}) as rl:
        rl.log_metrics(step=10, loss=2.31, lr=3e-4)
        rl.log_bio_state(step=10, calcium=ca, rrp=rrp, energy=e)   # tensors -> summaries
        rl.event("split_merge", level="warning", layer=3, action="merge", pair=[1, 4])
"""

from __future__ import annotations

import json
import logging
import math
import os
import socket
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

# Importing common configures the root logging handler (ColoredFormatter) and gives
# us the canonical project logger to parent ours under (AGENTS.md: reuse the helper).
from bio_inspired_nanochat.common import logger as _common_logger  # noqa: F401
from bio_inspired_nanochat.torch_imports import torch

_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


# --------------------------------------------------------------------------- #
# Tensor summary — the per-step bio-state logging primitive
# --------------------------------------------------------------------------- #
def tensor_summary(t: "torch.Tensor") -> dict[str, Any]:
    """JSON-serializable summary of a tensor; never raises on NaN/Inf/empty.

    Returns a dict with shape/dtype + finite-only mean/std/min/max/absmax and
    nan/inf counts. This is what gets written into the JSONL event stream so a
    later reader can audit how a bio variable evolved over a run.
    """
    t = t.detach()
    numel = int(t.numel())
    out: dict[str, Any] = {
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "numel": numel,
    }
    if numel == 0:
        out.update(nan=0, inf=0, mean=None, std=None, min=None, max=None, absmax=None)
        return out
    tf = t.float()
    nan = int(torch.isnan(tf).sum().item())
    inf = int(torch.isinf(tf).sum().item())
    finite = tf[torch.isfinite(tf)]
    if finite.numel() > 0:
        mean = float(finite.mean().item())
        std = float(finite.std(unbiased=False).item()) if finite.numel() > 1 else 0.0
        mn = float(finite.min().item())
        mx = float(finite.max().item())
        absmax = float(finite.abs().max().item())
    else:
        mean = std = mn = mx = absmax = None
    out.update(nan=nan, inf=inf, mean=mean, std=std, min=mn, max=mx, absmax=absmax)
    return out


def _jsonable(v: Any) -> Any:
    """Coerce a value into something ``json.dumps`` can serialize (lossy, safe)."""
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else str(v)  # NaN/Inf -> "nan"/"inf"
    if isinstance(v, torch.Tensor):
        # Scalars become numbers; larger tensors become summaries (keeps lines small).
        if v.numel() == 1:
            x = v.detach().item()
            return x if not isinstance(x, float) or math.isfinite(x) else str(x)
        return tensor_summary(v)
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    # numpy scalars/arrays and anything else -> best effort
    try:
        import numpy as np

        if isinstance(v, np.generic):
            return _jsonable(v.item())
        if isinstance(v, np.ndarray):
            return tensor_summary(torch.from_numpy(v)) if v.size > 1 else _jsonable(v.tolist())
    except Exception:
        pass
    return str(v)


def gather_provenance(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Best-effort run provenance: git SHA, torch/cuda, host. Never raises.

    ``extra`` (e.g. ``{"config_hash": ..., "seed": ...}``) is merged in so the run
    can be reproduced and joined to the results registry (hm4.1).
    """
    prov: dict[str, Any] = {
        "git_sha": _git_sha(),
        "torch": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }
    if extra:
        prov.update({str(k): _jsonable(v) for k, v in extra.items()})
    return prov


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=3, cwd=Path(__file__).resolve().parent,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- #
# RunLogger
# --------------------------------------------------------------------------- #
class RunLogger:
    """Run-scoped structured logger: JSONL events + console mirror.

    Parameters
    ----------
    run_dir : path-like
        Directory for this run's artifacts; ``events.jsonl`` is written here.
    name : str
        Short run name (used in the console logger name and the run-start event).
    run_id : str, optional
        Correlation id; a short uuid is generated if omitted.
    console : bool
        Mirror events to the console logger (default True).
    level : int
        Console level threshold (events below it are still written to JSONL).
    provenance : dict, optional
        Extra provenance (config_hash, seed, ...) merged into the run-start event.
    """

    def __init__(
        self,
        run_dir: str | os.PathLike[str],
        *,
        name: str = "run",
        run_id: str | None = None,
        console: bool = True,
        level: int = logging.INFO,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.console = console
        self.level = level
        self.events_path = self.run_dir / "events.jsonl"
        self._fh = self.events_path.open("a", encoding="utf-8")
        self._logger = logging.getLogger(f"bio_inspired_nanochat.run.{name}")
        self._n_events = 0
        self._closed = False
        self.event("run_start", run_dir=str(self.run_dir), provenance=gather_provenance(provenance))

    # -- core ------------------------------------------------------------- #
    def event(self, event: str, *, level: str = "info", step: int | None = None, **fields: Any) -> dict[str, Any]:
        """Write one structured event to the JSONL stream and mirror to console.

        Returns the record dict (handy for tests). Extra ``fields`` are coerced to
        JSON-serializable values (tensors -> summaries) via ``_jsonable``.
        """
        rec: dict[str, Any] = {
            "ts": time.time(),
            "run_id": self.run_id,
            "event": event,
            "level": level,
        }
        if step is not None:
            rec["step"] = int(step)
        for k, v in fields.items():
            rec[k] = _jsonable(v)
        if not self._closed:
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fh.flush()
            self._n_events += 1
        if self.console:
            self._logger.log(_LEVELS.get(level, logging.INFO), self._console_line(rec))
        return rec

    @staticmethod
    def _console_line(rec: dict[str, Any]) -> str:
        head = f"[{rec['event']}]"
        if "step" in rec:
            head += f" step={rec['step']}"
        extras = " ".join(
            f"{k}={_fmt(v)}" for k, v in rec.items()
            if k not in ("ts", "run_id", "event", "level", "step")
        )
        return f"{head} {extras}".rstrip()

    # -- convenience ------------------------------------------------------ #
    def log_metrics(self, step: int | None = None, **metrics: Any) -> dict[str, Any]:
        """Log scalar metrics, e.g. ``log_metrics(step=10, loss=2.3, lr=3e-4)``."""
        return self.event("metrics", step=step, **metrics)

    def log_tensor(self, name: str, t: "torch.Tensor", *, step: int | None = None) -> dict[str, Any]:
        """Log a single tensor's summary under ``name``."""
        return self.event("tensor", step=step, name=name, summary=tensor_summary(t))

    def log_bio_state(self, step: int | None = None, **named_tensors: "torch.Tensor") -> dict[str, Any]:
        """Log a batch of named bio-state tensors as summaries.

        e.g. ``log_bio_state(step=t, calcium=C, rrp=RRP, energy=E)``. The mechanism
        beads call this to make their per-step internal state auditable.
        """
        tensors = {name: tensor_summary(t) for name, t in named_tensors.items()}
        return self.event("bio_state", step=step, tensors=tensors)

    def read_events(self) -> list[dict[str, Any]]:
        """Parse this run's ``events.jsonl`` back into a list (tests/analysis)."""
        with self.events_path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    # -- lifecycle -------------------------------------------------------- #
    def close(self) -> None:
        if self._closed:
            return
        self.event("run_end", n_events=self._n_events + 1)
        self._closed = True
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            # Record the failure in the run before closing (best effort).
            try:
                self.event("run_error", level="error", error_type=getattr(exc_type, "__name__", str(exc_type)), error=str(exc))
            except Exception:
                pass
        self.close()


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, dict):
        return "{" + ",".join(f"{k}=…" for k in list(v)[:3]) + ("…}" if len(v) > 3 else "}")
    s = str(v)
    return s if len(s) <= 60 else s[:57] + "…"


def get_run_logger(run_dir: str | os.PathLike[str], **kwargs: Any) -> RunLogger:
    """Convenience constructor mirroring ``RunLogger(...)``."""
    return RunLogger(run_dir, **kwargs)


# --------------------------------------------------------------------------- #
# TrainingTelemetry — the unified at-scale run logger (bead hwxb.7.1)
# --------------------------------------------------------------------------- #
# A single layer that records, every step/eval, ALL the signals needed to verify
# correctness and diagnose a multi-hour 2×4090 run, to BOTH TensorBoard (human) and
# the structured JSONL event stream (machine / Phase-4 ablation analysis). It wraps
# ``RunLogger`` (the JSONL/console backbone) and an optional TensorBoard writer, is
# **rank-0-only** under DDP (a no-op on other ranks), and is cheap by default —
# heavy bio telemetry is cadence-gated via ``should_log_heavy``.
#
# Canonical JSONL schema (the contract eval_matrix + the ablation reader rely on):
#   event="train_step"  step=<int>  loss lr grad_norm tok_per_sec step_ms mfu
#                                    vram_gb divergence_action  (+ any **extra)
#   event="eval"        step=<int>  val_bpb core_metric  (+ per-task metrics)
#   event="bio_telemetry" step=<int>  spectral_radius fast_weight_mag
#                                    kinetics={layer:...} neuromod={nm/...:...}
#                                    splitmerge={...} expert_health={...}
# Every family is optional; only the keys you pass are written. Scalar leaves are
# additionally mirrored to TensorBoard under train/ , eval/ , bio/ namespaces.
TRAIN_STEP_FIELDS: tuple[str, ...] = (
    "loss", "lr", "grad_norm", "tok_per_sec", "step_ms", "mfu", "vram_gb", "divergence_action",
)
EVAL_FIELDS: tuple[str, ...] = ("val_bpb", "core_metric")
BIO_TELEMETRY_FIELDS: tuple[str, ...] = (
    "spectral_radius", "fast_weight_mag", "kinetics", "neuromod", "splitmerge", "expert_health",
)
EVENT_TRAIN_STEP = "train_step"
EVENT_EVAL = "eval"
EVENT_BIO = "bio_telemetry"


def _finite_number(v: Any) -> float | None:
    """Return ``float(v)`` if ``v`` is a finite real scalar, else ``None``."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, torch.Tensor) and v.numel() == 1:
        f = float(v.detach().item())
        return f if math.isfinite(f) else None
    return None


class TrainingTelemetry:
    """Unified, rank-gated training telemetry: JSONL events + TensorBoard scalars.

    Parameters
    ----------
    run_dir : path-like
        Run artifact dir (``events.jsonl`` + ``tb/`` are written here).
    name : str
        Short run name (passed to the underlying ``RunLogger``).
    is_master : bool
        Only rank 0 should write. On non-master ranks this object is a cheap no-op
        (no files opened, every log method returns immediately) so DDP runs do not
        produce N duplicate/garbled logs.
    run_id, provenance :
        Forwarded to ``RunLogger`` for correlation + reproducibility.
    tensorboard : bool
        Also write scalars to a TensorBoard ``SummaryWriter`` under ``run_dir/tb``.
        Degrades gracefully to JSONL-only if tensorboard isn't importable.
    heavy_every : int
        Suggested cadence for the (more expensive) bio telemetry; ``should_log_heavy``
        is True when ``heavy_every > 0 and step % heavy_every == 0``. 0 disables the
        helper (callers may still call ``log_bio`` explicitly).
    console : bool
        Mirror JSONL events to the console logger.
    """

    def __init__(
        self,
        run_dir: str | os.PathLike[str],
        *,
        name: str = "train",
        is_master: bool = True,
        run_id: str | None = None,
        provenance: dict[str, Any] | None = None,
        tensorboard: bool = True,
        heavy_every: int = 0,
        console: bool = True,
    ) -> None:
        self.enabled = bool(is_master)
        self.heavy_every = int(heavy_every)
        self.run_id = run_id
        self._rl: RunLogger | None = None
        self._tb = None
        if not self.enabled:
            return
        self._rl = RunLogger(run_dir, name=name, run_id=run_id, console=console, provenance=provenance)
        self.run_id = self._rl.run_id
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(log_dir=str(Path(run_dir) / "tb"))
            except Exception as e:  # tensorboard optional; never break training over logging
                self._rl.event("telemetry_warning", level="warning", msg=f"tensorboard unavailable: {e}")
                self._tb = None

    # -- cadence ---------------------------------------------------------- #
    def should_log_heavy(self, step: int) -> bool:
        """True on steps where the heavier bio telemetry should be emitted."""
        return self.enabled and self.heavy_every > 0 and (int(step) % self.heavy_every == 0)

    # -- TensorBoard helper ---------------------------------------------- #
    def _tb_scalars(self, prefix: str, mapping: dict[str, Any], step: int) -> None:
        if self._tb is None:
            return
        for k, v in mapping.items():
            f = _finite_number(v)
            if f is not None:
                self._tb.add_scalar(f"{prefix}/{k}", f, step)
            elif isinstance(v, dict):
                self._tb_scalars(f"{prefix}/{k}", v, step)

    # -- logging surface -------------------------------------------------- #
    def log_train_step(self, step: int, *, loss: Any, **fields: Any) -> None:
        """Log one training step (JSONL ``train_step`` + TB ``train/*`` scalars).

        Pass any of TRAIN_STEP_FIELDS (lr, grad_norm, tok_per_sec, step_ms, mfu,
        vram_gb, divergence_action) plus arbitrary extras; only what you pass is
        written.
        """
        if not self.enabled or self._rl is None:
            return
        rec = {"loss": loss, **fields}
        self._rl.event(EVENT_TRAIN_STEP, step=step, **rec)
        self._tb_scalars("train", rec, step)

    def log_eval(self, step: int, **metrics: Any) -> None:
        """Log an evaluation point (JSONL ``eval`` + TB ``eval/*`` scalars)."""
        if not self.enabled or self._rl is None:
            return
        self._rl.event(EVENT_EVAL, step=step, **metrics)
        self._tb_scalars("eval", metrics, step)

    def log_bio(self, step: int, **families: Any) -> None:
        """Log bio-mechanism telemetry (JSONL ``bio_telemetry`` + TB ``bio/*``).

        Pass any of BIO_TELEMETRY_FIELDS: spectral_radius, fast_weight_mag, kinetics
        (per-layer dict), neuromod (NeuromodulatoryBus.telemetry() dict), splitmerge,
        expert_health. Tensors become compact summaries in the JSONL; finite scalar
        leaves are mirrored to TensorBoard.
        """
        if not self.enabled or self._rl is None:
            return
        self._rl.event(EVENT_BIO, step=step, **families)
        self._tb_scalars("bio", families, step)

    def event(self, event: str, **fields: Any) -> None:
        """Escape hatch for one-off structured events (e.g. checkpoint saved)."""
        if not self.enabled or self._rl is None:
            return
        self._rl.event(event, **fields)

    # -- lifecycle -------------------------------------------------------- #
    def flush(self) -> None:
        if self._tb is not None:
            try:
                self._tb.flush()
            except Exception:
                pass

    def close(self) -> None:
        if self._tb is not None:
            try:
                self._tb.flush()
                self._tb.close()
            except Exception:
                pass
            self._tb = None
        if self._rl is not None:
            self._rl.close()

    def __enter__(self) -> "TrainingTelemetry":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def read_run_events(run_dir: str | os.PathLike[str], *, event: str | None = None) -> list[dict[str, Any]]:
    """Parse a run's ``events.jsonl`` (optionally filtered to one ``event`` type).

    The query path the Phase-4 ablation analysis uses to consume a run's telemetry.
    """
    path = Path(run_dir) / "events.jsonl"
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # A run that crashed mid-write can leave a truncated final line;
                # skip it rather than failing the whole post-hoc analysis.
                continue
            if event is None or rec.get("event") == event:
                out.append(rec)
    return out
