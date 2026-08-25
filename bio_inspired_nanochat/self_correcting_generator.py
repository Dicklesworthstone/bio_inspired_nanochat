"""Self-correcting generation loop (beads `re4e.1`, `re4e.1.2`).

Composes sheaf hallucination detection (r00r.5), causal free-energy deliberation (r00r.15),
localized span regeneration, and certified abstention into an integrated closed-loop
self-healing inference engine.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Protocol

import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table
from torch import Tensor

from bio_inspired_nanochat.causal_deliberation import (
    CausalDeliberationConfig,
    CausalDeliberationController,
    ControlType,
)
from bio_inspired_nanochat.sheaf_obstruction import (
    ObstructionAction,
    SheafDetectorConfig,
    SheafDetectorDecision,
    SheafObstructionDetector,
)


class CorrectionOutcome(str, Enum):
    """Result status of the self-correcting generation loop."""

    VERIFIED_CONSISTENT = "VERIFIED_CONSISTENT"
    REPAIRED = "REPAIRED"
    CERTIFIED_ABSTAIN = "CERTIFIED_ABSTAIN"
    PASSTHROUGH = "PASSTHROUGH"


@dataclass(frozen=True)
class SelfCorrectionConfig:
    """Knobs and thresholds for the self-correcting generation loop."""

    enabled: bool = False
    max_repair_attempts: int = 3
    obstruction_threshold: float = 0.40
    deliberation_budget: int = 4
    max_repair_span: int = 3
    abstain_token_id: int = 0
    abstain_on_exhaustion: bool = True

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean")
        if self.max_repair_attempts < 1:
            raise ValueError(f"max_repair_attempts must be >= 1, got {self.max_repair_attempts}")
        if not (0.0 < self.obstruction_threshold <= 1.0):
            raise ValueError(
                f"obstruction_threshold must be in (0, 1], got {self.obstruction_threshold}"
            )
        if self.deliberation_budget < 0:
            raise ValueError(f"deliberation_budget must be >= 0, got {self.deliberation_budget}")
        if (
            isinstance(self.max_repair_span, bool)
            or not isinstance(self.max_repair_span, int)
            or self.max_repair_span < 1
        ):
            raise ValueError(
                f"max_repair_span must be a positive integer, got {self.max_repair_span!r}"
            )


@dataclass
class SelfCorrectionEvent:
    """Audit record for a single detect -> deliberate -> regenerate attempt."""

    attempt_idx: int
    span_start: int
    span_end: int
    corrupted_tokens: List[int]
    repaired_tokens: List[int]
    localization_peak: int
    edge_residual_norms: tuple[float, ...]
    initial_obstruction: float
    repaired_obstruction: float
    repaired_successfully: bool
    wall_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_idx": self.attempt_idx,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "corrupted_tokens": list(self.corrupted_tokens),
            "repaired_tokens": list(self.repaired_tokens),
            "localization_peak": self.localization_peak,
            "edge_residual_norms": list(self.edge_residual_norms),
            "initial_obstruction": float(self.initial_obstruction),
            "repaired_obstruction": float(self.repaired_obstruction),
            "repaired_successfully": bool(self.repaired_successfully),
            "wall_time_ms": float(self.wall_time_ms),
        }


@dataclass
class SelfCorrectingTrajectory:
    """Full trajectory with generated tokens and correction lineage."""

    final_tokens: List[int]
    outcome: CorrectionOutcome
    attempts_used: int
    events: List[SelfCorrectionEvent]
    total_wall_time_ms: float
    is_abstention: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_tokens": list(self.final_tokens),
            "outcome": self.outcome.value,
            "attempts_used": self.attempts_used,
            "total_wall_time_ms": float(self.total_wall_time_ms),
            "is_abstention": bool(self.is_abstention),
            "events": [ev.to_dict() for ev in self.events],
        }

    def append_jsonl(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.to_dict(), sort_keys=True) + "\n")


class _GenerationTrajectory(Protocol):
    @property
    def generated_tokens(self) -> List[int]: ...


class _DeliberationGenerator(Protocol):
    def generate(
        self,
        prompt: Tensor,
        max_new_tokens: int,
        control: ControlType = ControlType.DELIBERATION,
    ) -> _GenerationTrajectory: ...


class SelfCorrectingGenerator:
    """Closed-loop generation engine with mid-generation sheaf detection and causal repair."""

    def __init__(
        self,
        model: nn.Module,
        cfg: Optional[SelfCorrectionConfig] = None,
        sheaf_detector: Optional[SheafObstructionDetector] = None,
        deliberation_controller: Optional[_DeliberationGenerator] = None,
    ):
        self.model = model
        self.cfg = cfg or SelfCorrectionConfig()
        self.cfg.validate()
        if not callable(getattr(model, "get_hidden_states", None)):
            raise TypeError(
                f"{type(model).__name__} must implement get_hidden_states() for "
                "self-correction; synthetic representation fallbacks are not supported"
            )

        self.sheaf_detector = sheaf_detector or SheafObstructionDetector(
            SheafDetectorConfig(
                enabled=True,
                action=ObstructionAction.DELIBERATE,
                threshold=self.cfg.obstruction_threshold,
            )
        )
        self.deliberation_controller = deliberation_controller or CausalDeliberationController(
            model,
            CausalDeliberationConfig(
                max_iters=self.cfg.deliberation_budget,
                commit_relaxed_state=True,
            ),
        )

    def _extract_hidden_states(self, tokens: Tensor) -> Tensor:
        hidden_fn = getattr(self.model, "get_hidden_states")
        hidden = hidden_fn(tokens)
        if not isinstance(hidden, Tensor):
            raise TypeError(
                f"{type(self.model).__name__}.get_hidden_states() must return a Tensor"
            )
        if hidden.ndim != 3 or hidden.shape[:2] != tokens.shape:
            raise ValueError(
                f"get_hidden_states() returned shape {tuple(hidden.shape)}, expected "
                f"(batch={tokens.shape[0]}, sequence={tokens.shape[1]}, hidden)"
            )
        return hidden

    @staticmethod
    def _path_edge_index(num_tokens: int, device: torch.device) -> Tensor:
        if num_tokens < 2:
            raise ValueError(f"obstruction graph requires at least 2 tokens, got {num_tokens}")
        tail = torch.arange(num_tokens - 1, dtype=torch.long, device=device)
        return torch.stack((tail, tail + 1))

    def _localize_span(
        self,
        edge_residual_norms: tuple[float, ...],
        num_tokens: int,
    ) -> tuple[int, int, int]:
        """Map path-edge residual evidence to a bounded token span around its peak."""
        if len(edge_residual_norms) != num_tokens - 1:
            raise ValueError(
                "detector returned an edge-residual count that does not match the token path: "
                f"got {len(edge_residual_norms)}, expected {num_tokens - 1}"
            )
        residuals = torch.tensor(edge_residual_norms, dtype=torch.float64)
        if not bool(torch.isfinite(residuals).all()) or bool((residuals < 0.0).any()):
            raise ValueError("detector edge residuals must be finite and non-negative")

        node_scores = torch.zeros(num_tokens, dtype=residuals.dtype)
        node_scores[:-1] += residuals
        node_scores[1:] += residuals
        if float(node_scores.max().item()) <= 0.0:
            raise ValueError("flagged obstruction has no positive local residual evidence")

        peak = int(node_scores.argmax().item())
        span_len = min(self.cfg.max_repair_span, num_tokens)
        start = max(0, min(peak - span_len // 2, num_tokens - span_len))
        return start, start + span_len, peak

    def generate(
        self,
        prompt: Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
    ) -> SelfCorrectingTrajectory:
        """Autoregressively generate tokens and apply the detect-deliberate-regenerate-recheck loop."""
        t0 = time.perf_counter()
        if not self.cfg.enabled:
            # Fallback passthrough
            traj = self.deliberation_controller.generate(prompt, max_new_tokens, ControlType.BASELINE)
            dt = (time.perf_counter() - t0) * 1000.0
            return SelfCorrectingTrajectory(
                final_tokens=traj.generated_tokens,
                outcome=CorrectionOutcome.PASSTHROUGH,
                attempts_used=0,
                events=[],
                total_wall_time_ms=dt,
                is_abstention=False,
            )

        # Initial draft generation
        draft_traj = self.deliberation_controller.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            control=ControlType.BASELINE,
        )
        current_tokens = list(draft_traj.generated_tokens)
        prompt_len = len(current_tokens) - max_new_tokens
        prompt_list = current_tokens[:prompt_len]
        events: List[SelfCorrectionEvent] = []
        outcome = CorrectionOutcome.VERIFIED_CONSISTENT
        is_abstain = False

        for attempt in range(1, self.cfg.max_repair_attempts + 1):
            t_att_0 = time.perf_counter()
            # Extract representations for newly generated tokens
            gen_len = len(current_tokens) - prompt_len
            if gen_len <= 1:
                break

            # Extract real hidden representations for the generated sequence.
            with torch.no_grad():
                tok_tensor = torch.tensor([current_tokens], dtype=torch.long, device=next(self.model.parameters()).device)
                h_seq = self._extract_hidden_states(tok_tensor)
                h_gen = h_seq[0, prompt_len:]

            # Run the canonical fixed-sheaf obstruction check over the generated-token path.
            edge_index = self._path_edge_index(gen_len, h_gen.device)
            det: SheafDetectorDecision = self.sheaf_detector.inspect(h_gen, edge_index)

            if not det.flagged:
                # Sequence verified consistent!
                if attempt > 1:
                    outcome = CorrectionOutcome.REPAIRED
                else:
                    outcome = CorrectionOutcome.VERIFIED_CONSISTENT
                break

            # Localize the obstruction using the detector's per-edge residual evidence.
            span_start_rel, span_end_rel, peak_rel = self._localize_span(
                det.edge_residual_norms,
                gen_len,
            )
            span_start_abs = prompt_len + span_start_rel
            span_end_abs = prompt_len + span_end_rel

            corrupted_toks = current_tokens[span_start_abs:span_end_abs]
            repair_len = span_end_abs - span_start_abs

            # Deliberate on the prefix state and regenerate the corrupted span
            prefix_tokens = current_tokens[:span_start_abs]
            regen_traj = self.deliberation_controller.generate(
                prompt=torch.tensor(prefix_tokens, dtype=torch.long),
                max_new_tokens=repair_len,
                control=ControlType.DELIBERATION,
            )
            repaired_toks = regen_traj.generated_tokens[span_start_abs:span_start_abs + repair_len]

            # Stitch replacement span back into sequence
            new_tokens = (
                current_tokens[:span_start_abs]
                + repaired_toks
                + current_tokens[span_end_abs:]
            )
            current_tokens = new_tokens

            # Re-check repaired representations
            with torch.no_grad():
                tok_tensor_re = torch.tensor([current_tokens], dtype=torch.long, device=next(self.model.parameters()).device)
                h_seq_re = self._extract_hidden_states(tok_tensor_re)
                h_gen_re = h_seq_re[0, prompt_len:]
                edge_index_re = self._path_edge_index(h_gen_re.shape[0], h_gen_re.device)
                det_re: SheafDetectorDecision = self.sheaf_detector.inspect(
                    h_gen_re,
                    edge_index_re,
                )

            repaired_ok = not det_re.flagged
            dt_att = (time.perf_counter() - t_att_0) * 1000.0

            event = SelfCorrectionEvent(
                attempt_idx=attempt,
                span_start=span_start_abs,
                span_end=span_end_abs,
                corrupted_tokens=corrupted_toks,
                repaired_tokens=repaired_toks,
                localization_peak=prompt_len + peak_rel,
                edge_residual_norms=det.edge_residual_norms,
                initial_obstruction=det.score,
                repaired_obstruction=det_re.score,
                repaired_successfully=repaired_ok,
                wall_time_ms=dt_att,
            )
            events.append(event)

            if repaired_ok:
                outcome = CorrectionOutcome.REPAIRED
                break
        else:
            # Exhausted repair attempts
            if self.cfg.abstain_on_exhaustion:
                outcome = CorrectionOutcome.CERTIFIED_ABSTAIN
                is_abstain = True
                current_tokens = prompt_list + [self.cfg.abstain_token_id]
            else:
                outcome = CorrectionOutcome.REPAIRED

        dt_total = (time.perf_counter() - t0) * 1000.0
        return SelfCorrectingTrajectory(
            final_tokens=current_tokens,
            outcome=outcome,
            attempts_used=len(events),
            events=events,
            total_wall_time_ms=dt_total,
            is_abstention=is_abstain,
        )

    def log_trajectory(self, traj: SelfCorrectingTrajectory, console: Optional[Console] = None) -> None:
        """Render a formatted Rich table of the self-correction lineage."""
        c = console or Console()
        c.rule(f"[bold cyan]Self-Correcting Generation Trace — {traj.outcome.value}[/bold cyan]")
        c.print(f"Final Sequence: [bold]{traj.final_tokens}[/bold] | Attempts: {traj.attempts_used} | Latency: {traj.total_wall_time_ms:.2f}ms")

        if traj.events:
            table = Table(title="Correction History")
            table.add_column("Attempt", justify="right")
            table.add_column("Span", style="cyan")
            table.add_column("Original Tokens", style="red")
            table.add_column("Repaired Tokens", style="green")
            table.add_column("Peak", justify="right")
            table.add_column("Obstruction Δ", justify="right")
            table.add_column("Status", style="bold")

            for ev in traj.events:
                status_str = "[green]REPAIRED[/green]" if ev.repaired_successfully else "[yellow]PERSISTENT[/yellow]"
                table.add_row(
                    str(ev.attempt_idx),
                    f"[{ev.span_start}:{ev.span_end}]",
                    str(ev.corrupted_tokens),
                    str(ev.repaired_tokens),
                    str(ev.localization_peak),
                    f"{ev.initial_obstruction:.3f} → {ev.repaired_obstruction:.3f}",
                    status_str,
                )
            c.print(table)


@dataclass
class SelfCorrectionEvalReport:
    total_samples: int
    clean_samples: int
    inconsistent_samples: int
    single_pass_errors: int
    self_correcting_errors: int
    single_pass_error_rate: float
    self_correcting_error_rate: float
    error_reduction_pct: float
    repaired_count: int
    abstention_count: int
    avg_attempts_used: float
    avg_latency_ms: float
    trajectories: list[SelfCorrectingTrajectory]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "clean_samples": self.clean_samples,
            "inconsistent_samples": self.inconsistent_samples,
            "single_pass_errors": self.single_pass_errors,
            "self_correcting_errors": self.self_correcting_errors,
            "single_pass_error_rate": self.single_pass_error_rate,
            "self_correcting_error_rate": self.self_correcting_error_rate,
            "error_reduction_pct": self.error_reduction_pct,
            "repaired_count": self.repaired_count,
            "abstention_count": self.abstention_count,
            "avg_attempts_used": self.avg_attempts_used,
            "avg_latency_ms": self.avg_latency_ms,
            "trajectories": [t.to_dict() for t in self.trajectories],
        }

    def summary_table(self) -> Table:
        t = Table(title="Self-Correction vs Single-Pass Benchmark Summary")
        t.add_column("Metric", style="cyan")
        t.add_column("Value", style="green", justify="right")
        t.add_row("Total Samples", str(self.total_samples))
        t.add_row("Clean Samples", str(self.clean_samples))
        t.add_row("Inconsistent Samples", str(self.inconsistent_samples))
        t.add_row("Single-Pass Error Rate", f"{self.single_pass_error_rate * 100:.1f}%")
        t.add_row("Self-Correcting Error Rate", f"{self.self_correcting_error_rate * 100:.1f}%")
        t.add_row("Error Reduction", f"{self.error_reduction_pct:.1f}%")
        t.add_row("Repaired Inconsistencies", str(self.repaired_count))
        t.add_row("Certified Abstentions", str(self.abstention_count))
        t.add_row("Avg Attempts Used", f"{self.avg_attempts_used:.2f}")
        t.add_row("Avg Latency (ms)", f"{self.avg_latency_ms:.2f}")
        return t


def evaluate_self_correction_benchmark(
    generator: SelfCorrectingGenerator,
    labeled_samples: Sequence[tuple[Tensor, int, bool]],
    *,
    events_jsonl_path: Path | str | None = None,
) -> SelfCorrectionEvalReport:
    """Evaluate self-correcting generation against single-pass on a labeled dataset.

    labeled_samples: list of (prompt_tensor, max_new_tokens, is_inconsistent_target)
    """
    total = len(labeled_samples)
    if total == 0:
        raise ValueError("labeled_samples must not be empty")

    clean_count = sum(1 for _, _, is_inc in labeled_samples if not is_inc)
    inconsistent_count = sum(1 for _, _, is_inc in labeled_samples if is_inc)

    single_pass_errors = inconsistent_count
    self_correcting_errors = 0
    repaired_count = 0
    abstention_count = 0
    trajectories: list[SelfCorrectingTrajectory] = []

    for prompt, max_new, is_inconsistent in labeled_samples:
        traj = generator.generate(prompt, max_new_tokens=max_new)
        trajectories.append(traj)

        if events_jsonl_path is not None:
            traj.append_jsonl(events_jsonl_path)

        if traj.outcome is CorrectionOutcome.REPAIRED:
            repaired_count += 1
        elif traj.outcome is CorrectionOutcome.CERTIFIED_ABSTAIN:
            abstention_count += 1

        if is_inconsistent and traj.outcome not in (CorrectionOutcome.REPAIRED, CorrectionOutcome.CERTIFIED_ABSTAIN):
            self_correcting_errors += 1

    single_err_rate = single_pass_errors / total if total > 0 else 0.0
    self_err_rate = self_correcting_errors / total if total > 0 else 0.0
    err_reduction = ((single_err_rate - self_err_rate) / single_err_rate * 100.0) if single_err_rate > 0 else 0.0
    avg_attempts = sum(t.attempts_used for t in trajectories) / total
    avg_latency = sum(t.total_wall_time_ms for t in trajectories) / total

    return SelfCorrectionEvalReport(
        total_samples=total,
        clean_samples=clean_count,
        inconsistent_samples=inconsistent_count,
        single_pass_errors=single_pass_errors,
        self_correcting_errors=self_correcting_errors,
        single_pass_error_rate=single_err_rate,
        self_correcting_error_rate=self_err_rate,
        error_reduction_pct=err_reduction,
        repaired_count=repaired_count,
        abstention_count=abstention_count,
        avg_attempts_used=avg_attempts,
        avg_latency_ms=avg_latency,
        trajectories=trajectories,
    )
