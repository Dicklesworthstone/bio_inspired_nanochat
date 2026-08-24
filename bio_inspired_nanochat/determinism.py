"""Centralized Seed & Determinism Policy Framework (beads aiq, hm4.4).

Provides standardized seed configuration, CUDA/cuDNN determinism guards,
provenance metadata capture, and exact reproduction testing across training,
evaluation, profiling, and HPO workloads.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class DeterminismState:
    seed: int
    deterministic_algorithms: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    cublas_workspace_config: str | None
    python_hash_seed: str | None


def configure_determinism(
    seed: int = 42,
    *,
    deterministic: bool = True,
    warn_only: bool = True,
) -> DeterminismState:
    """Configure comprehensive determinism across Python, PyTorch, CUDA, and cuDNN."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not deterministic

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
        except RuntimeError:
            torch.use_deterministic_algorithms(False)
    else:
        try:
            torch.use_deterministic_algorithms(False)
        except RuntimeError:
            torch.use_deterministic_algorithms(False)

    return get_determinism_state(seed=seed)


def get_determinism_state(seed: int = 42) -> DeterminismState:
    """Capture current determinism settings for run provenance."""
    return DeterminismState(
        seed=seed,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic) if torch.cuda.is_available() else False,
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark) if torch.cuda.is_available() else False,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        python_hash_seed=os.environ.get("PYTHONHASHSEED"),
    )


def determinism_provenance_dict(seed: int = 42) -> dict[str, Any]:
    """Serialize determinism state as JSON-serializable provenance metadata."""
    state = get_determinism_state(seed=seed)
    return asdict(state)
