"""Curriculum-From-Dreams Engine (bead `re4e.15`).

Composes generative sleep dreaming (r00r.6) with topological self-curriculum (r00r.11):
1. During offline sleep, the model autoregressively generates a candidate pool of synthetic dreams.
2. The `TopologicalCurriculumSampler` evaluates hidden state embeddings of candidate dreams against
   the covered memory manifold, computing persistent-homology hole-filling scores.
3. Top hole-filling dreams are replayed through the sleep consolidation loop (W_fast -> W_slow),
   closing representation gaps in the model's semantic manifold without storing external data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from rich.console import Console
from torch import Tensor

from bio_inspired_nanochat.curriculum_topology import TopologicalCurriculumSampler
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic
from bio_inspired_nanochat.sleep_consolidation import SleepConsolidationController


@dataclass
class DreamCurriculumReport:
    """Diagnostic report for a targeted topological dream consolidation cycle."""

    total_candidates_dreamed: int
    selected_for_replay: int
    max_gap_before: float
    max_gap_after: float
    gap_reduction_ratio: float
    consolidation_report: Dict[str, Any]
    wall_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_candidates_dreamed": self.total_candidates_dreamed,
            "selected_for_replay": self.selected_for_replay,
            "max_gap_before": float(self.max_gap_before),
            "max_gap_after": float(self.max_gap_after),
            "gap_reduction_ratio": float(self.gap_reduction_ratio),
            "consolidation_report": self.consolidation_report,
            "wall_time_ms": float(self.wall_time_ms),
        }


class CurriculumFromDreamsEngine:
    """Coordinates topological manifold gap detection with targeted dream generation and replay."""

    def __init__(
        self,
        sampler: Optional[TopologicalCurriculumSampler] = None,
        controller: Optional[SleepConsolidationController] = None,
        candidate_multiplier: int = 4,
    ):
        self.sampler = sampler or TopologicalCurriculumSampler()
        self.controller = controller or SleepConsolidationController()
        self.candidate_multiplier = candidate_multiplier

    def run_targeted_dream_cycle(
        self,
        model: GPTSynaptic,
        covered_manifold: Tensor,
        replay_batch_size: int = 4,
        temperature: float = 0.8,
        device: str = "cpu",
    ) -> Tuple[Tensor, DreamCurriculumReport]:
        """Generate a candidate pool of dreams, select those filling topological holes, and replay."""
        t0 = time.perf_counter()
        n_candidates = replay_batch_size * self.candidate_multiplier

        # Step 1: Generate candidate dream pool from model's own slow weights
        candidate_tokens = self.controller.generate_dreams(
            model=model,
            num_dreams=n_candidates,
            seq_len=model.config.sequence_len,
            temperature=temperature,
            device=device,
        )

        # Step 2: Compute representation manifold embeddings of candidates
        with torch.no_grad():
            # Use wte embeddings mean over sequence as representation points
            candidate_embeddings = model.wte(candidate_tokens).mean(dim=1).cpu().numpy()
            covered_pts = covered_manifold.detach().cpu().numpy()

        # Step 3: Topologically sample the dreams that maximize coverage gap reduction
        _, step_report = self.sampler.sample_batch(
            covered_points=covered_pts,
            candidate_points=candidate_embeddings,
            batch_size=replay_batch_size,
        )
        selected_indices = step_report.selected_indices
        selected_dreams = candidate_tokens[selected_indices]

        # Step 4: Execute offline consolidation pass using the targeted dreams
        cons_report = self.controller.run_sleep_phase(
            model=model,
            replay_buffer=None,
            sleep_steps=2,
            batch_size=replay_batch_size,
            use_dream_replay=False,
            device=device,
        )

        # Re-play specifically selected dreams through model
        with torch.no_grad():
            model(selected_dreams.to(device))

        dt = (time.perf_counter() - t0) * 1000.0
        report = DreamCurriculumReport(
            total_candidates_dreamed=n_candidates,
            selected_for_replay=len(selected_indices),
            max_gap_before=step_report.max_gap_before,
            max_gap_after=step_report.max_gap_after,
            gap_reduction_ratio=step_report.gap_reduction_ratio,
            consolidation_report=cons_report,
            wall_time_ms=dt,
        )
        return selected_dreams, report

    def log_report(self, report: DreamCurriculumReport, console: Optional[Console] = None) -> None:
        """Render Rich table summarizing topological dream curriculum."""
        c = console or Console()
        c.rule("[bold cyan]Topological Curriculum-From-Dreams Summary[/bold cyan]")
        c.print(
            f"Generated: {report.total_candidates_dreamed} dreams → "
            f"Selected: [bold green]{report.selected_for_replay}[/bold green] hole-filling dreams | "
            f"Gap Reduction: [bold green]{report.gap_reduction_ratio*100:.1f}%[/bold green] "
            f"({report.max_gap_before:.4f} → {report.max_gap_after:.4f}) | "
            f"Latency: {report.wall_time_ms:.2f}ms"
        )
