"""Sheaf-Theoretic Hallucination & Inconsistency Detector (bead r00r.5).

Repurposes 1-cohomology obstruction (H^1, Thrust G) as a certified hallucination
detector with calibrated thresholds, automated repair diffusion, and abstention gating.
"""

from __future__ import annotations

import enum
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from bio_inspired_nanochat.sheaf_binding import (
    BindingCertificate,
    SheafConsistencyMonitor,
    SheafDiffusionLayer,
)


class DetectorAction(str, enum.Enum):
    FLAG_ONLY = "flag_only"
    ABSTAIN = "abstain"
    REPAIR = "repair"


@dataclass
class HallucinationReport:
    is_hallucination: bool
    obstruction_score: float
    threshold: float
    confidence: float
    certificate: BindingCertificate
    action_taken: DetectorAction
    repaired_activations: Optional[Tensor] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["action_taken"] = self.action_taken.value
        d.pop("repaired_activations", None)
        return d


def log_hallucination_audit(report: HallucinationReport, jsonl_path: Optional[Path] = None) -> None:
    """Log structured JSONL hallucination detector audit event."""
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(report.to_dict()) + "\n")


class SheafHallucinationDetector(nn.Module):
    """Detects relational hallucinations via sheaf-Laplacian obstruction L x."""

    def __init__(
        self,
        d_model: int,
        threshold: float = 0.05,
        action: DetectorAction = DetectorAction.FLAG_ONLY,
        num_repair_steps: int = 5,
        diffusion_rate: float = 0.1,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.threshold = float(threshold)
        self.action = action
        self.enabled = enabled
        self.repair_layer = SheafDiffusionLayer(
            d_model=d_model,
            num_diffusion_steps=num_repair_steps,
            diffusion_rate=diffusion_rate,
            enabled=enabled,
        )

    def calibrate_threshold(
        self,
        consistent_scores: List[float],
        hallucinated_scores: List[float],
        target_fpr: float = 0.05,
    ) -> float:
        """Calibrate the obstruction score threshold to satisfy target FPR."""
        if not consistent_scores:
            return self.threshold

        sorted_clean = sorted(consistent_scores)
        idx = min(len(sorted_clean) - 1, int(len(sorted_clean) * (1.0 - target_fpr)))
        self.threshold = float(sorted_clean[idx])
        return self.threshold

    def compute_obstruction_score(
        self,
        activations: Tensor,
        laplacian: Tensor,
    ) -> float:
        """Compute normalized Dirichlet energy / obstruction residual."""
        return SheafConsistencyMonitor.compute_obstruction_energy(activations, laplacian)

    def forward(
        self,
        activations: Tensor,
        laplacian: Optional[Tensor] = None,
        step: int = 0,
    ) -> HallucinationReport:
        """Inspect activations against relational constraints in laplacian."""
        if not self.enabled:
            # Fallback no-op pass
            cert = BindingCertificate(
                is_certified=True,
                h1_obstruction=0.0,
                spectral_gap=0.0,
                dimension_kernel=self.d_model,
                step=step,
            )
            return HallucinationReport(
                is_hallucination=False,
                obstruction_score=0.0,
                threshold=self.threshold,
                confidence=0.0,
                certificate=cert,
                action_taken=DetectorAction.FLAG_ONLY,
                repaired_activations=None,
            )

        if laplacian is None:
            T = activations.shape[0] if activations.ndim >= 2 else 1
            if T > 1:
                A = torch.diag(torch.ones(T - 1, device=activations.device), 1) + torch.diag(torch.ones(T - 1, device=activations.device), -1)
                D = torch.diag(A.sum(dim=-1))
                laplacian = D - A
            else:
                laplacian = torch.zeros(1, 1, device=activations.device)

        score = self.compute_obstruction_score(activations, laplacian)
        is_hallucination = score > self.threshold
        cert = SheafConsistencyMonitor.evaluate_certificate(laplacian, stalk_dim=self.d_model)
        cert.step = step

        confidence = 1.0 / (1.0 + np.exp(-10.0 * (score - self.threshold)))

        repaired = None
        if is_hallucination and self.action == DetectorAction.REPAIR:
            repaired = self.repair_layer(activations, laplacian=laplacian)

        return HallucinationReport(
            is_hallucination=is_hallucination,
            obstruction_score=score,
            threshold=self.threshold,
            confidence=float(confidence),
            certificate=cert,
            action_taken=self.action,
            repaired_activations=repaired,
        )
