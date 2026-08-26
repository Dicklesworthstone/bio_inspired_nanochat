"""Unified device-aware backend dispatcher (bead jyb.6).

Auto-selects Triton (CUDA) / Rust (CPU) / PyTorch eager fallback based on
device availability and compiled extensions, with an explicit override knob.
"""

from __future__ import annotations

import enum
import importlib
import logging
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Check backend availability
TRITON_AVAILABLE = False
if torch.cuda.is_available():
    try:
        import triton  # noqa: F401
        TRITON_AVAILABLE = True
    except ImportError:
        TRITON_AVAILABLE = False

try:
    rustbpe: Any = importlib.import_module("rustbpe")
except ModuleNotFoundError:
    rustbpe = None

RUST_AVAILABLE = rustbpe is not None


class Backend(str, enum.Enum):
    AUTO = "auto"
    TRITON = "triton"
    RUST = "rust"
    PYTORCH = "pytorch"


def get_available_backends() -> list[Backend]:
    """Return list of available compute backends on this system."""
    backends = [Backend.PYTORCH]
    if RUST_AVAILABLE:
        backends.append(Backend.RUST)
    if TRITON_AVAILABLE:
        backends.append(Backend.TRITON)
    return backends


def select_backend(
    device: torch.device | str,
    override: Optional[Backend | str] = None,
) -> Backend:
    """Select the optimal backend for a given device and system configuration."""
    if isinstance(override, str):
        override = Backend(override.lower())

    if override is not None and override != Backend.AUTO:
        if override == Backend.TRITON and not TRITON_AVAILABLE:
            logger.warning("Triton requested but not available; falling back to PyTorch")
            return Backend.PYTORCH
        if override == Backend.RUST and not RUST_AVAILABLE:
            logger.warning("Rust requested but rustbpe extension not available; falling back to PyTorch")
            return Backend.PYTORCH
        return override

    dev_type = torch.device(device).type if isinstance(device, (str, torch.device)) else "cpu"

    if dev_type == "cuda" and TRITON_AVAILABLE:
        return Backend.TRITON
    elif dev_type == "cpu" and RUST_AVAILABLE:
        return Backend.RUST
    else:
        return Backend.PYTORCH


# --------------------------------------------------------------------------- #
# Unified Dispatch Functions
# --------------------------------------------------------------------------- #


def dispatch_accumulate_router_stats(
    indices: Tensor,
    gates: Tensor,
    num_experts: int,
    backend: Optional[Backend | str] = None,
) -> Tuple[Tensor, Tensor]:
    """Dispatch MoE routing stats accumulation across Triton / Rust / PyTorch.

    Semantics note (kernel audit): for indices containing DUPLICATES within one
    token's top-k slots — impossible for true ``torch.topk`` output, but common
    in synthetic tests — the backends disagree on ``counts``: Triton counts
    per-edge OCCURRENCES while Rust/PyTorch count per-token PRESENCE
    (``mask.any(dim=-1)``). ``gate_sums`` is occurrence-summed everywhere and
    agrees. Production routing (top-k output) can never contain duplicates, so
    the divergence is unreachable on the live path; if you feed synthetic
    indices, pin one backend explicitly.
    """
    resolved_backend = select_backend(gates.device, override=backend)

    if resolved_backend == Backend.TRITON and gates.is_cuda:
        from bio_inspired_nanochat.kernels.genetics_fused import accumulate_router_stats
        return accumulate_router_stats(indices, gates, num_experts)

    elif resolved_backend == Backend.RUST and rustbpe is not None and hasattr(rustbpe, "accumulate_router_stats_cpu"):
        indices_np = indices.detach().cpu().numpy().astype(np.int64)
        gates_np = gates.detach().cpu().numpy().astype(np.float32)
        counts_np, probs_np = rustbpe.accumulate_router_stats_cpu(indices_np, gates_np, num_experts)
        return (
            torch.from_numpy(counts_np).to(device=gates.device, dtype=torch.float32),
            torch.from_numpy(probs_np).to(device=gates.device, dtype=torch.float32),
        )

    # PyTorch eager fallback
    counts = torch.zeros(num_experts, device=gates.device, dtype=torch.float32)
    gate_sums = torch.zeros(num_experts, device=gates.device, dtype=torch.float32)
    for e in range(num_experts):
        mask = indices == e
        counts[e] = mask.any(dim=-1).sum().float()
        gate_sums[e] = gates.masked_select(mask).sum()

    return counts, gate_sums


def dispatch_update_metabolism(
    fatigue: Tensor,
    energy: Tensor,
    alpha_fatigue: Tensor,
    alpha_energy: Tensor,
    util: Tensor,
    backend: Optional[Backend | str] = None,
) -> Tuple[Tensor, Tensor]:
    """Dispatch expert metabolic state update across Triton / Rust / PyTorch."""
    resolved_backend = select_backend(fatigue.device, override=backend)

    if resolved_backend == Backend.TRITON and fatigue.is_cuda:
        from bio_inspired_nanochat.kernels.genetics_fused import update_metabolism_fused
        # update_metabolism_fused mutates IN PLACE; clone first so all three
        # backends share the same out-of-place contract (callers get fresh
        # tensors, inputs untouched — matching Rust/eager below).
        fatigue = fatigue.clone()
        energy = energy.clone()
        return update_metabolism_fused(fatigue, energy, alpha_fatigue, alpha_energy, util)

    elif resolved_backend == Backend.RUST and rustbpe is not None and hasattr(rustbpe, "update_metabolism_cpu"):
        f_np = fatigue.detach().cpu().numpy().astype(np.float32)
        e_np = energy.detach().cpu().numpy().astype(np.float32)
        af_np = alpha_fatigue.detach().cpu().numpy().astype(np.float32)
        ae_np = alpha_energy.detach().cpu().numpy().astype(np.float32)
        u_np = util.detach().cpu().numpy().astype(np.float32)

        f_out_np, e_out_np = rustbpe.update_metabolism_cpu(f_np, e_np, af_np, ae_np, u_np)
        return (
            torch.from_numpy(f_out_np).to(device=fatigue.device, dtype=fatigue.dtype),
            torch.from_numpy(e_out_np).to(device=energy.device, dtype=energy.dtype),
        )

    # PyTorch eager fallback
    fatigue_next = (1.0 - alpha_fatigue) * fatigue + alpha_fatigue * util
    energy_next = (1.0 - alpha_energy) * energy + alpha_energy * (1.0 - util)
    return fatigue_next, energy_next
