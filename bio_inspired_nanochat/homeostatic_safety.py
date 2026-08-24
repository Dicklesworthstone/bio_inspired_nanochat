"""Homeostatic Safety Guardrails & Control Barrier Alignment (bead re4e.13).

Enforces forward invariance of the safe set S = {z : h_i(z) >= 0} via control
barrier functions (CBFs), preventing runaway potentiation and attractor collapse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class HomeostaticSafetyReport:
    is_safe: bool
    barrier_margins: Dict[str, float]
    intervention_applied: bool
    min_margin: float


class HomeostaticSafetyGuard(nn.Module):
    """Control-theoretic safety guard enforcing forward invariance on synaptic dynamics."""

    def __init__(
        self,
        max_fast_weight_norm: float = 5.0,
        max_energy_budget: float = 2.0,
        min_activation_entropy: float = 0.1,
        gamma_margin: float = 0.05,
    ) -> None:
        super().__init__()
        self.max_fast_weight_norm = float(max_fast_weight_norm)
        self.max_energy_budget = float(max_energy_budget)
        self.min_activation_entropy = float(min_activation_entropy)
        self.gamma_margin = float(gamma_margin)

    def evaluate_barriers(
        self,
        w_fast: Optional[Tensor] = None,
        energy: Optional[Tensor] = None,
        activations: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """Compute margins h_i(z) for all active barrier certificates."""
        margins: Dict[str, float] = {}

        if w_fast is not None:
            norm_val = float(torch.linalg.norm(w_fast).item())
            # h_norm = max_norm - ||W||
            margins["fast_weight_norm"] = self.max_fast_weight_norm - norm_val

        if energy is not None:
            energy_val = float(energy.mean().item())
            # h_energy = max_energy - E
            margins["energy_budget"] = self.max_energy_budget - energy_val

        if activations is not None:
            # Distribution entropy of activations
            probs = torch.softmax(activations.abs().view(-1), dim=0).clamp(min=1e-8)
            entropy = float(-(probs * torch.log(probs)).sum().item())
            margins["activation_entropy"] = entropy - self.min_activation_entropy

        return margins

    def enforce_safety(
        self,
        w_fast: Optional[Tensor] = None,
        energy: Optional[Tensor] = None,
        activations: Optional[Tensor] = None,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], HomeostaticSafetyReport]:
        """Enforce barrier forward invariance; clamps or restores state if entering unsafe set."""
        margins = self.evaluate_barriers(w_fast=w_fast, energy=energy, activations=activations)
        min_margin = min(margins.values()) if margins else 1.0

        intervention = False
        w_fast_safe = w_fast
        energy_safe = energy

        # 1. Clamp fast weight norm if approaching boundary
        if w_fast is not None and margins.get("fast_weight_norm", 1.0) < self.gamma_margin:
            current_norm = torch.linalg.norm(w_fast).clamp(min=1e-8)
            target_norm = self.max_fast_weight_norm - self.gamma_margin
            if current_norm > target_norm:
                w_fast_safe = w_fast * (target_norm / current_norm)
                intervention = True

        # 2. Damp energy if approaching boundary
        if energy is not None and margins.get("energy_budget", 1.0) < self.gamma_margin:
            energy_safe = torch.clamp(energy, max=self.max_energy_budget - self.gamma_margin)
            intervention = True

        is_safe = (min_margin >= 0.0) and not intervention

        report = HomeostaticSafetyReport(
            is_safe=is_safe,
            barrier_margins=margins,
            intervention_applied=intervention,
            min_margin=min_margin,
        )

        return w_fast_safe, energy_safe, report
