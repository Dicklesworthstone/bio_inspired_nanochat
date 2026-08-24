"""Metacognition and self-model layer (beads `re4e.2`, `re4e.2.1`).

Reads real internal biological and geometric state — free energy, sheaf coboundary obstruction,
and thermodynamic entropy — to maintain an honest estimate of the model's own competence per span,
reporting:
- KNOWN ("I know"): High confidence, low free energy, zero sheaf obstruction.
- GUESSING ("I am guessing / extrapolating"): Moderate free energy, minor sheaf tension.
- UNKNOWN ("I don't know"): High free energy, high sheaf obstruction, elevated entropy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table
from torch import Tensor



class EpistemicStatus(str, Enum):
    """Tri-state epistemic classification for a generated span."""

    KNOWN = "KNOWN"
    GUESSING = "GUESSING"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class MetacognitionConfig:
    """Configuration and thresholds for the metacognitive self-model."""

    threshold_known: float = 0.70
    threshold_unknown: float = 0.35
    w_intercept: float = 2.0
    w_free_energy: float = -1.5
    w_sheaf_obstruction: float = -2.0
    w_entropy: float = -1.0
    span_size: int = 4

    def validate(self) -> None:
        if not (0.0 <= self.threshold_unknown < self.threshold_known <= 1.0):
            raise ValueError(
                f"Invalid thresholds: must satisfy 0 <= unknown ({self.threshold_unknown}) < "
                f"known ({self.threshold_known}) <= 1"
            )


@dataclass
class SpanCompetenceReport:
    """Auditable competence assessment for a contiguous token span."""

    span_start: int
    span_end: int
    tokens: List[int]
    competence_score: float
    free_energy: float
    sheaf_obstruction: float
    normalized_entropy: float
    status: EpistemicStatus
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_start": self.span_start,
            "span_end": self.span_end,
            "tokens": self.tokens,
            "competence_score": float(self.competence_score),
            "free_energy": float(self.free_energy),
            "sheaf_obstruction": float(self.sheaf_obstruction),
            "normalized_entropy": float(self.normalized_entropy),
            "status": self.status.value,
            "explanation": self.explanation,
        }


class MetacognitiveSelfModel(nn.Module):
    """Grounded epistemic self-model predicting correctness from internal physical state."""

    def __init__(self, cfg: Optional[MetacognitionConfig] = None):
        super().__init__()
        self.cfg = cfg or MetacognitionConfig()
        self.cfg.validate()
        # Calibrated logistic weights
        self.w_0 = self.cfg.w_intercept
        self.w_fe = self.cfg.w_free_energy
        self.w_sheaf = self.cfg.w_sheaf_obstruction
        self.w_ent = self.cfg.w_entropy

    def estimate_free_energy(self, hidden: Tensor) -> float:
        """Computes quadratic Lyapunov state energy: 0.5 * Var(hidden)."""
        if hidden.numel() == 0:
            return 0.0
        return float(torch.var(hidden).item())

    def estimate_sheaf_obstruction(self, hidden: Tensor) -> float:
        """Estimates pairwise representation discordance as a proxy for sheaf coboundary obstruction."""
        if hidden.shape[0] < 2:
            return 0.0
        # Cosine distance between successive tokens in the span
        h_norm = F.normalize(hidden, p=2, dim=-1)
        sim = (h_norm[:-1] * h_norm[1:]).sum(dim=-1)
        obstruction = float(torch.clamp(1.0 - sim.mean(), min=0.0).item())
        return obstruction

    def estimate_entropy(self, logits: Tensor) -> float:
        """Computes normalized token entropy over the span."""
        if logits.numel() == 0:
            return 0.0
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1).mean()
        vocab_size = logits.shape[-1]
        norm_entropy = float((entropy / math.log(max(2, vocab_size))).clamp(0.0, 1.0).item())
        return norm_entropy

    def assess_span(
        self,
        span_tokens: List[int],
        hidden_states: Tensor,
        logits: Tensor,
        sheaf_obstruction: Optional[float] = None,
        span_start: int = 0,
        span_end: int = 0,
    ) -> SpanCompetenceReport:
        """Evaluate epistemic competence over a single span."""
        fe = self.estimate_free_energy(hidden_states)
        obstruction = sheaf_obstruction if sheaf_obstruction is not None else self.estimate_sheaf_obstruction(hidden_states)
        ent = self.estimate_entropy(logits)

        # Logit combination
        z = self.w_0 + self.w_fe * fe + self.w_sheaf * obstruction + self.w_ent * ent
        competence = float(1.0 / (1.0 + math.exp(-z)))

        if competence >= self.cfg.threshold_known:
            status = EpistemicStatus.KNOWN
            explanation = "High internal coherence: low energy, minimal sheaf tension."
        elif competence >= self.cfg.threshold_unknown:
            status = EpistemicStatus.GUESSING
            explanation = "Moderate uncertainty: intermediate energy or local extrapolation."
        else:
            status = EpistemicStatus.UNKNOWN
            explanation = "High epistemic barrier: elevated free energy or sheaf obstruction."

        return SpanCompetenceReport(
            span_start=span_start,
            span_end=span_end,
            tokens=span_tokens,
            competence_score=competence,
            free_energy=fe,
            sheaf_obstruction=obstruction,
            normalized_entropy=ent,
            status=status,
            explanation=explanation,
        )

    def assess_sequence(
        self,
        tokens: List[int],
        hidden_states: Tensor,
        logits: Tensor,
        span_size: Optional[int] = None,
    ) -> List[SpanCompetenceReport]:
        """Tile the sequence into spans and produce span-level metacognitive reports."""
        size = span_size or self.cfg.span_size
        T = len(tokens)
        reports: List[SpanCompetenceReport] = []

        for start in range(0, T, size):
            end = min(T, start + size)
            span_toks = tokens[start:end]
            h_span = hidden_states[start:end] if hidden_states.ndim == 2 else hidden_states[0, start:end]
            l_span = logits[start:end] if logits.ndim == 2 else logits[0, start:end]

            rep = self.assess_span(
                span_tokens=span_toks,
                hidden_states=h_span,
                logits=l_span,
                span_start=start,
                span_end=end,
            )
            reports.append(rep)

        return reports

    @staticmethod
    def compute_ece(confidences: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error (ECE)."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n_samples = len(confidences)
        if n_samples == 0:
            return 0.0

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper if i < n_bins - 1 else confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(labels[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return float(ece)

    @staticmethod
    def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
        """Compute Area Under the ROC Curve."""
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        # Mann-Whitney U statistic calculation
        u = sum(np.sum(p > neg) + 0.5 * np.sum(p == neg) for p in pos)
        return float(u / (len(pos) * len(neg)))

    def log_reports(self, reports: List[SpanCompetenceReport], console: Optional[Console] = None) -> None:
        """Render Rich table of span-level competence reports."""
        c = console or Console()
        table = Table(title="Metacognitive Competence & Epistemic Status")
        table.add_column("Span", style="cyan")
        table.add_column("Tokens", style="bold")
        table.add_column("Competence", justify="right")
        table.add_column("Free Energy", justify="right")
        table.add_column("Sheaf Obstruction", justify="right")
        table.add_column("Status", style="bold")
        table.add_column("Explanation")

        color_map = {
            EpistemicStatus.KNOWN: "green",
            EpistemicStatus.GUESSING: "yellow",
            EpistemicStatus.UNKNOWN: "red",
        }

        for r in reports:
            col = color_map[r.status]
            table.add_row(
                f"[{r.span_start}:{r.span_end}]",
                str(r.tokens),
                f"{r.competence_score:.3f}",
                f"{r.free_energy:.3f}",
                f"{r.sheaf_obstruction:.3f}",
                f"[{col}]{r.status.value}[/{col}]",
                r.explanation,
            )
        c.print(table)
