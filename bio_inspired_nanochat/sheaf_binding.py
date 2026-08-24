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
    h1_obstruction: float
    spectral_gap: float
    dimension_kernel: int
    step: int = 0

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
            delta[e_row_start:e_row_end, v_col_start : v_col_start + stalk_dim] = R_v

        return delta.t() @ delta

    @staticmethod
    def compute_obstruction_energy(x: Tensor, laplacian: Tensor) -> float:
        """Compute normalized Dirichlet energy (x^T L x) / (lambda_max * ||x||^2) in [0, 1]."""
        T = laplacian.shape[0]
        if x.shape[0] == T:
            x_mat = x.float()
            l_mat = laplacian.float()
            energy = torch.trace(x_mat.t() @ (l_mat @ x_mat))
            norm_sq = torch.sum(x_mat**2).clamp(min=1e-8)
            eigs = torch.linalg.eigvalsh(l_mat)
            l_max = float(eigs.max().item()) if eigs.numel() > 0 else 1.0
            scale = max(1e-6, l_max)
            return float((energy / (scale * norm_sq)).clamp(0.0, 1.0).item())
        x_flat = x.view(-1, laplacian.shape[0])
        norm_sq = torch.sum(x_flat**2, dim=-1).clamp(min=1e-8)
        energy = torch.sum(x_flat * (x_flat @ laplacian.t()), dim=-1)
        return float((energy / norm_sq).mean().item())

    @staticmethod
    def evaluate_certificate(
        laplacian: Tensor,
        stalk_dim: int = 8,
        tol: float = 1e-4,
        step: int = 0,
    ) -> BindingCertificate:
        """Evaluate spectral gap and dimension of ker(L) (H^0 global sections)."""
        eigenvalues = torch.linalg.eigvalsh(laplacian)
        zero_mask = eigenvalues.abs() < tol
        dim_ker = int(zero_mask.sum().item())

        non_zero = eigenvalues[~zero_mask]
        spectral_gap = float(non_zero.min().item()) if non_zero.numel() > 0 else 0.0

        is_certified = dim_ker >= 1 and spectral_gap > 0.0
        return BindingCertificate(
            is_certified=is_certified,
            h1_obstruction=float(eigenvalues.min().item()),
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

        all_scores = np.concatenate([clean, corrupted])
        all_labels = np.concatenate([np.zeros_like(clean), np.ones_like(corrupted)])

        order = np.argsort(all_scores)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(all_scores)) + 1

        n_pos = len(corrupted)
        n_neg = len(clean)
        if n_pos == 0 or n_neg == 0:
            return 0.5

        u_stat = np.sum(ranks[all_labels == 1]) - (n_pos * (n_pos + 1)) / 2.0
        return float(u_stat / (n_pos * n_neg))


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

        B, T, D = x.shape
        x_flat = x.view(B, T * D)

        for _ in range(self.num_diffusion_steps):
            diff = x_flat @ laplacian.t()
            x_flat = x_flat - self.diffusion_rate * diff

        return x_flat.view(B, T, D)


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
