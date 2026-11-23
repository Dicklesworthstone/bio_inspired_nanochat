"""Cellular Sheaf Consistency Diffusion & Operadic SNARE Routing (bead 0642.8.2).

Implements sheaf Laplacian L = delta^T delta, H^1 binding obstruction monitor,
harmonic consistency diffusion projection, and operadic syntax tree routing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class BindingCertificate:
    is_certified: bool
    h1_obstruction: float | None
    spectral_gap: float
    dimension_kernel: int
    step: int = 0
    h1_dimension: int | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def log_sheaf_audit(cert: BindingCertificate, jsonl_path: Optional[Path] = None) -> None:
    """Log structured JSONL sheaf obstruction audit event."""
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(cert.to_dict()) + "\n")


class SheafConsistencyMonitor:
    """Computes sheaf Laplacian, coboundary obstruction, and binding certificates."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    @staticmethod
    def build_sheaf_laplacian(
        edges: List[Tuple[int, int]],
        restriction_maps: Optional[List[Tuple[Tensor, Tensor]]] = None,
        num_nodes: int = 4,
        stalk_dim: int = 8,
        device: str = "cpu",
    ) -> Tensor:
        """Construct discrete sheaf Laplacian L = delta^T @ delta."""
        if not edges:
            return torch.zeros(num_nodes * stalk_dim, num_nodes * stalk_dim, device=device)

        num_edges = len(edges)
        delta = torch.zeros(num_edges * stalk_dim, num_nodes * stalk_dim, device=device)

        for e_idx, (u, v) in enumerate(edges):
            e_row_start = e_idx * stalk_dim
            e_row_end = e_row_start + stalk_dim

            if restriction_maps and e_idx < len(restriction_maps):
                R_u, R_v = restriction_maps[e_idx]
            else:
                R_u = torch.eye(stalk_dim, device=device)
                R_v = torch.eye(stalk_dim, device=device)

            u_col_start = u * stalk_dim
            v_col_start = v * stalk_dim

            delta[e_row_start:e_row_end, u_col_start : u_col_start + stalk_dim] = -R_u
            delta[e_row_start:e_row_end, v_col_start : v_col_start + stalk_dim] += R_v

        return delta.t() @ delta

    @staticmethod
    def apply_laplacian(x: Tensor, laplacian: Tensor) -> Tensor:
        """Apply a token Laplacian or the full block Laplacian to each sample."""
        if laplacian.ndim != 2 or laplacian.shape[0] != laplacian.shape[1]:
            raise ValueError("expected a square Laplacian")
        size = laplacian.shape[0]
        if x.ndim == 1 and x.numel() == size:
            return laplacian @ x
        if x.ndim not in (2, 3):
            raise ValueError("expected node features [T,D] or [B,T,D]")
        nodes, width = x.shape[-2:]
        if size == nodes:
            return laplacian @ x
        if size == nodes * width:
            return (x.flatten(-2) @ laplacian.T).reshape_as(x)
        raise ValueError(f"Laplacian size {size} does not match {nodes} nodes of width {width}")

    @staticmethod
    def compute_obstruction_energy(x: Tensor, laplacian: Tensor) -> float:
        """Compute normalized Dirichlet energy (x^T L x) / (lambda_max * ||x||^2) in [0, 1]."""
        features = x.float()
        operator = laplacian.float()
        applied = SheafConsistencyMonitor.apply_laplacian(features, operator)
        feature_dims = (-2, -1) if x.ndim > 1 else (-1,)
        energy = (features * applied).sum(dim=feature_dims)
        denom = features.square().sum(dim=feature_dims) * torch.linalg.eigvalsh(operator).amax()
        ratio = torch.where(denom > 0, energy / denom, torch.zeros_like(energy))
        return float(ratio.clamp(0, 1).mean().item())

    @staticmethod
    def evaluate_certificate(
        laplacian: Tensor,
        stalk_dim: int = 8,
        tol: float = 1e-4,
        step: int = 0,
        *,
        edge_space_dim: int | None = None,
    ) -> BindingCertificate:
        """Measure H^0 and, for a graph with known edge-space size, dim H^1."""
        eigenvalues = torch.linalg.eigvalsh(laplacian)
        zero_mask = eigenvalues.abs() < tol
        dim_ker = int(zero_mask.sum().item())

        non_zero = eigenvalues[~zero_mask]
        spectral_gap = float(non_zero.min().item()) if non_zero.numel() > 0 else 0.0

        rank = laplacian.shape[0] - dim_ker
        if edge_space_dim is not None and edge_space_dim < rank:
            raise ValueError("edge space cannot be smaller than the coboundary rank")
        is_certified = dim_ker >= 1 and spectral_gap > 0.0
        return BindingCertificate(
            is_certified=is_certified,
            h1_obstruction=None,  # An individual obstruction needs independently observed edge data.
            h1_dimension=None if edge_space_dim is None else edge_space_dim - rank,
            spectral_gap=spectral_gap,
            dimension_kernel=dim_ker,
            step=step,
        )

    @staticmethod
    def compute_binding_auroc(
        clean_energies: List[float],
        corrupted_energies: List[float],
    ) -> float:
        """Compute AUROC for discriminating clean vs corrupted bindings."""
        clean = np.array(clean_energies)
        corrupted = np.array(corrupted_energies)

        if not clean.size or not corrupted.size:
            raise ValueError("AUROC requires both classes")
        if not np.isfinite(clean).all() or not np.isfinite(corrupted).all():
            raise ValueError("AUROC requires finite scores")
        ordered = np.sort(clean)
        lower = np.searchsorted(ordered, corrupted, side="left")
        upper = np.searchsorted(ordered, corrupted, side="right")
        return float((lower + upper).mean() / (2 * clean.size))


class SheafDiffusionLayer(nn.Module):
    """Computes discrete consistency diffusion x^(t+1) = (I - eta L) x^(t)."""

    def __init__(
        self,
        d_model: int,
        num_diffusion_steps: int = 3,
        diffusion_rate: float = 0.1,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_diffusion_steps = max(1, num_diffusion_steps)
        self.diffusion_rate = float(diffusion_rate)
        self.enabled = enabled

    def forward(
        self,
        x: Tensor,
        laplacian: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply sheaf diffusion to project node features onto ker(L)."""
        if not self.enabled or laplacian is None:
            return x

        for _ in range(self.num_diffusion_steps):
            diff = SheafConsistencyMonitor.apply_laplacian(x, laplacian)
            x = x - self.diffusion_rate * diff
        return x


class OperadicSNAREMatcher(nn.Module):
    """Computes operadic syntax tree routing scores from SNARE code matching."""

    def __init__(self, code_dim: int = 16) -> None:
        super().__init__()
        self.code_dim = code_dim
        self.proj = nn.Linear(code_dim, code_dim, bias=False)
        nn.init.eye_(self.proj.weight)

    def compute_docking_score(
        self,
        v_snare: Tensor,
        t_snare: Tensor,
    ) -> Tensor:
        """Compute SNARE complex formation affinity (cosine-like docking energy)."""
        v = F.normalize(self.proj(v_snare), p=2, dim=-1)
        t = F.normalize(self.proj(t_snare), p=2, dim=-1)
        cosine_sim = (v * t).sum(dim=-1)
        return (cosine_sim + 1.0) / 2.0  # Normalized affinity in [0, 1]
