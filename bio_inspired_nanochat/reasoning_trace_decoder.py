"""Scalar energy-trajectory reporter (bead `re4e.9`).

Reports measured changes in a supplied scalar trajectory. Scalar energy values alone do not
identify concepts, cognitive operations, fast-weight changes, or causal chains of thought;
the output therefore keeps causal-faithfulness false unless richer provenance is added later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from torch import Tensor



class StepOperation(str, Enum):
    """Categorized cognitive operation performed during a deliberation step."""

    INITIAL_HYPOTHESIS = "INITIAL_HYPOTHESIS"
    ENERGY_RELAXATION = "ENERGY_RELAXATION"
    ENERGY_INCREASE = "ENERGY_INCREASE"
    ENERGY_UNCHANGED = "ENERGY_UNCHANGED"
    INCONSISTENCY_RESOLVED = "INCONSISTENCY_RESOLVED"
    CONVERGED = "CONVERGED"


@dataclass
class ReasoningStep:
    """A single observed transition in the supplied scalar trajectory."""

    step_index: int
    operation: StepOperation
    energy_before: float
    energy_after: float
    energy_delta: float
    state_norm_delta: Optional[float]
    top_token_concepts: List[str]
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "operation": self.operation.value,
            "energy_before": self.energy_before,
            "energy_after": self.energy_after,
            "energy_delta": self.energy_delta,
            "state_norm_delta": self.state_norm_delta,
            "top_token_concepts": self.top_token_concepts,
            "explanation": self.explanation,
        }


@dataclass
class MechanisticTrace:
    """Structured report of an observed scalar energy trajectory."""

    steps: List[ReasoningStep]
    initial_energy: float
    final_energy: float
    total_energy_reduction: float
    is_causally_faithful: bool
    summary_narrative: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "initial_energy": self.initial_energy,
            "final_energy": self.final_energy,
            "total_energy_reduction": self.total_energy_reduction,
            "is_causally_faithful": self.is_causally_faithful,
            "summary_narrative": self.summary_narrative,
        }


class ReasoningTraceDecoder:
    """Converts scalar energy trajectories into narrowly descriptive reports."""

    def __init__(
        self,
        vocabulary: Optional[List[str]] = None,
        lm_head: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        self.vocab = vocabulary or [f"token_{i}" for i in range(128)]
        self.lm_head = lm_head

    def _get_top_concepts(self, h: Tensor, top_k: int = 3) -> List[str]:
        """Project hidden state to vocabulary logits to identify activated concepts."""
        if self.lm_head is not None:
            with torch.no_grad():
                logits = self.lm_head(h)
                top_indices = torch.topk(logits, min(top_k, logits.shape[-1])).indices.squeeze().tolist()
                if isinstance(top_indices, int):
                    top_indices = [top_indices]
                return [self.vocab[idx] if idx < len(self.vocab) else f"tok_{idx}" for idx in top_indices]
        return [f"dim_{torch.argmax(h).item()}"]

    def decode_energy_trajectory(self, energy_trajectory: List[float]) -> MechanisticTrace:
        """Decode a list of successive energy values into structured reasoning steps."""
        traj = energy_trajectory
        if not all(math.isfinite(value) for value in traj):
            raise ValueError("energy_trajectory values must all be finite")
        if not traj:
            return MechanisticTrace(
                steps=[],
                initial_energy=0.0,
                final_energy=0.0,
                total_energy_reduction=0.0,
                is_causally_faithful=False,
                summary_narrative="No deliberation trajectory was observed.",
            )
        if len(traj) == 1:
            init_e = traj[0]
            return MechanisticTrace(
                steps=[],
                initial_energy=init_e,
                final_energy=init_e,
                total_energy_reduction=0.0,
                is_causally_faithful=False,
                summary_narrative="One energy measurement was observed; no transition can be inferred.",
            )

        K = len(traj) - 1
        steps: List[ReasoningStep] = []

        for k in range(K):
            e_before = traj[k]
            e_after = traj[k + 1]
            e_delta = e_after - e_before

            # Determine operation
            if abs(e_delta) < 1e-4:
                op = StepOperation.ENERGY_UNCHANGED
                expl = f"Energy change was within reporting tolerance (ΔE = {e_delta:+.4f})."
            elif e_delta > 0:
                op = StepOperation.ENERGY_INCREASE
                expl = f"Energy increased by {e_delta:.3f}; the trajectory did not descend."
            else:
                op = StepOperation.ENERGY_RELAXATION
                expl = f"Observed an energy decrease of {abs(e_delta):.3f}."

            # A scalar trajectory carries no token/concept attribution information.
            concepts: List[str] = []

            step = ReasoningStep(
                step_index=k + 1,
                operation=op,
                energy_before=e_before,
                energy_after=e_after,
                energy_delta=e_delta,
                # This decoder receives energies, not hidden states. Reporting |ΔE| as a
                # state-vector norm was dimensionally wrong and falsely implied an observation.
                state_norm_delta=None,
                top_token_concepts=concepts,
                explanation=expl,
            )
            steps.append(step)

        init_e = traj[0] if traj else 0.0
        final_e = traj[-1] if traj else 0.0
        total_drop = init_e - final_e

        if total_drop > 0:
            narrative = (
                f"Observed {K} transitions: Energy decreased from {init_e:.3f} to "
                f"{final_e:.3f} (total dissipation {total_drop:.3f})."
            )
        elif total_drop < 0:
            narrative = (
                f"Observed {K} transitions: Energy increased from {init_e:.3f} to "
                f"{final_e:.3f} (total increase {-total_drop:.3f})."
            )
        else:
            narrative = f"Observed {K} transitions with unchanged net energy at {init_e:.3f}."

        return MechanisticTrace(
            steps=steps,
            initial_energy=init_e,
            final_energy=final_e,
            total_energy_reduction=total_drop,
            is_causally_faithful=False,
            summary_narrative=narrative,
        )

    def log_trace(self, trace: MechanisticTrace, console: Optional[Console] = None) -> None:
        """Render a formatted Rich representation of the energy report."""
        c = console or Console()
        c.rule("[bold cyan]Energy-Trajectory Report[/bold cyan]")
        c.print(Panel(trace.summary_narrative, title="Trajectory Summary", style="green"))

        table = Table(title="Step-by-Step Energy Changes")
        table.add_column("Step", justify="right")
        table.add_column("Operation", style="bold")
        table.add_column("Energy Before", justify="right")
        table.add_column("Energy After", justify="right")
        table.add_column("Δ Energy", justify="right")
        table.add_column("Concepts")
        table.add_column("Observation")

        for s in trace.steps:
            col = "green" if s.energy_delta <= 0 else "yellow"
            table.add_row(
                str(s.step_index),
                s.operation.value,
                f"{s.energy_before:.3f}",
                f"{s.energy_after:.3f}",
                f"[{col}]{s.energy_delta:+.3f}[/{col}]",
                ", ".join(s.top_token_concepts),
                s.explanation,
            )
        c.print(table)
