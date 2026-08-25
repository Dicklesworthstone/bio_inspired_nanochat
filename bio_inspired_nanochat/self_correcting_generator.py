"""Self-correcting generation loop (beads `re4e.1`, `re4e.1.2`).

Composes sheaf hallucination detection (r00r.5), causal free-energy deliberation (r00r.15),
localized span regeneration, and bounded abstention into an integrated closed-loop
self-healing inference engine.
"""

from __future__ import annotations

import json
import math
import numbers
import time
from collections.abc import Callable, Sequence
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

    NO_OBSTRUCTION_DETECTED = "NO_OBSTRUCTION_DETECTED"
    REPAIRED = "REPAIRED"
    ABSTAIN = "ABSTAIN"
    UNCHECKED = "UNCHECKED"
    UNRESOLVED = "UNRESOLVED"
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
        if (
            isinstance(self.max_repair_attempts, bool)
            or not isinstance(self.max_repair_attempts, int)
            or self.max_repair_attempts < 1
        ):
            raise ValueError(
                "max_repair_attempts must be a positive integer, "
                f"got {self.max_repair_attempts!r}"
            )
        if (
            isinstance(self.obstruction_threshold, bool)
            or not isinstance(self.obstruction_threshold, numbers.Real)
            or not math.isfinite(float(self.obstruction_threshold))
            or not 0.0 < self.obstruction_threshold <= 1.0
        ):
            raise ValueError(
                f"obstruction_threshold must be in (0, 1], got {self.obstruction_threshold}"
            )
        if (
            isinstance(self.deliberation_budget, bool)
            or not isinstance(self.deliberation_budget, int)
            or self.deliberation_budget < 0
        ):
            raise ValueError(
                "deliberation_budget must be a non-negative integer, "
                f"got {self.deliberation_budget!r}"
            )
        if (
            isinstance(self.max_repair_span, bool)
            or not isinstance(self.max_repair_span, int)
            or self.max_repair_span < 1
        ):
            raise ValueError(
                f"max_repair_span must be a positive integer, got {self.max_repair_span!r}"
            )
        if (
            isinstance(self.abstain_token_id, bool)
            or not isinstance(self.abstain_token_id, int)
            or self.abstain_token_id < 0
        ):
            raise ValueError(
                f"abstain_token_id must be a non-negative integer, got {self.abstain_token_id!r}"
            )
        if not isinstance(self.abstain_on_exhaustion, bool):
            raise ValueError("abstain_on_exhaustion must be a boolean")


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
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        rng: torch.Generator | None = None,
    ) -> _GenerationTrajectory: ...


class _ObstructionDetector(Protocol):
    def inspect(
        self,
        stalks: Tensor,
        edge_index: Tensor,
    ) -> SheafDetectorDecision: ...


class SelfCorrectingGenerator:
    """Closed-loop generation engine with mid-generation sheaf detection and causal repair."""

    def __init__(
        self,
        model: nn.Module,
        cfg: Optional[SelfCorrectionConfig] = None,
        sheaf_detector: Optional[_ObstructionDetector] = None,
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
        vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
        if not isinstance(vocab_size, int) or vocab_size < 1:
            raise TypeError(
                f"{type(model).__name__}.config.vocab_size must be a positive integer"
            )
        if self.cfg.abstain_token_id >= vocab_size:
            raise ValueError(
                f"abstain_token_id {self.cfg.abstain_token_id} is outside model vocabulary "
                f"[0, {vocab_size})"
            )
        self.vocab_size = vocab_size

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

    @staticmethod
    def _normalize_prompt(prompt: Tensor) -> Tensor:
        prompt_tensor = prompt if isinstance(prompt, Tensor) else torch.as_tensor(prompt)
        if prompt_tensor.ndim == 1:
            normalized = prompt_tensor
        elif prompt_tensor.ndim == 2 and prompt_tensor.shape[0] == 1:
            normalized = prompt_tensor.reshape(-1)
        elif prompt_tensor.ndim == 2:
            raise ValueError(
                "prompt must contain exactly one sequence; batched generation is not supported"
            )
        else:
            raise ValueError(
                f"prompt must have shape (T,) or (1, T), got {tuple(prompt_tensor.shape)}"
            )
        if normalized.numel() == 0:
            raise ValueError("prompt must contain at least one token")
        return normalized

    def _validated_prompt(self, prompt: Tensor) -> Tensor:
        normalized = self._normalize_prompt(prompt)
        if normalized.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError(f"prompt must use an integer dtype, got {normalized.dtype}")
        if bool((normalized < 0).any()) or bool((normalized >= self.vocab_size).any()):
            raise ValueError(
                f"prompt contains a token outside model vocabulary [0, {self.vocab_size})"
            )
        return normalized

    def _model_device(self, fallback: torch.device) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            try:
                return next(self.model.buffers()).device
            except StopIteration:
                return fallback

    def _validated_generated_tokens(
        self,
        trajectory: _GenerationTrajectory,
        *,
        prompt_tokens: list[int],
        expected_new_tokens: int,
    ) -> list[int]:
        generated = list(trajectory.generated_tokens)
        expected_length = len(prompt_tokens) + expected_new_tokens
        if generated[: len(prompt_tokens)] != prompt_tokens:
            raise ValueError("deliberation controller did not preserve the supplied prompt")
        if len(generated) != expected_length:
            raise ValueError(
                "deliberation controller returned an unexpected token count: "
                f"got {len(generated)}, expected {expected_length}"
            )
        if any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or not 0 <= token < self.vocab_size
            for token in generated
        ):
            raise ValueError("deliberation controller returned a token outside the model vocabulary")
        return generated

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
        top_k: int | None = None,
        rng: torch.Generator | None = None,
    ) -> SelfCorrectingTrajectory:
        """Autoregressively generate tokens and apply the detect-deliberate-regenerate-recheck loop."""
        t0 = time.perf_counter()
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise TypeError("max_new_tokens must be a non-negative integer")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, numbers.Real)
            or not math.isfinite(float(temperature))
            or temperature < 0.0
        ):
            raise ValueError("temperature must be finite and non-negative")
        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0
        ):
            raise ValueError("top_k must be a non-negative integer")
        prompt_tensor = self._validated_prompt(prompt)
        prompt_list = [int(token) for token in prompt_tensor.tolist()]
        if not self.cfg.enabled:
            # Fallback passthrough
            traj = self.deliberation_controller.generate(
                prompt_tensor,
                max_new_tokens,
                ControlType.BASELINE,
                temperature=temperature,
                top_k=top_k,
                rng=rng,
            )
            generated_tokens = self._validated_generated_tokens(
                traj,
                prompt_tokens=prompt_list,
                expected_new_tokens=max_new_tokens,
            )
            dt = (time.perf_counter() - t0) * 1000.0
            return SelfCorrectingTrajectory(
                final_tokens=generated_tokens,
                outcome=CorrectionOutcome.PASSTHROUGH,
                attempts_used=0,
                events=[],
                total_wall_time_ms=dt,
                is_abstention=False,
            )

        # Initial draft generation
        draft_traj = self.deliberation_controller.generate(
            prompt=prompt_tensor,
            max_new_tokens=max_new_tokens,
            control=ControlType.BASELINE,
            temperature=temperature,
            top_k=top_k,
            rng=rng,
        )
        current_tokens = self._validated_generated_tokens(
            draft_traj,
            prompt_tokens=prompt_list,
            expected_new_tokens=max_new_tokens,
        )
        prompt_len = len(prompt_list)
        events: List[SelfCorrectionEvent] = []
        outcome = CorrectionOutcome.UNCHECKED
        is_abstain = False
        model_device = self._model_device(prompt_tensor.device)

        for attempt in range(1, self.cfg.max_repair_attempts + 1):
            t_att_0 = time.perf_counter()
            # Extract representations for newly generated tokens
            gen_len = len(current_tokens) - prompt_len
            if gen_len <= 1:
                outcome = CorrectionOutcome.UNCHECKED
                break

            # Extract real hidden representations for the generated sequence.
            with torch.no_grad():
                tok_tensor = torch.tensor(
                    [current_tokens],
                    dtype=torch.long,
                    device=model_device,
                )
                h_seq = self._extract_hidden_states(tok_tensor)
                h_gen = h_seq[0, prompt_len:]

            # Run the canonical fixed-sheaf obstruction check over the generated-token path.
            edge_index = self._path_edge_index(gen_len, h_gen.device)
            det: SheafDetectorDecision = self.sheaf_detector.inspect(h_gen, edge_index)

            if not det.available:
                outcome = (
                    CorrectionOutcome.UNRESOLVED
                    if events
                    else CorrectionOutcome.UNCHECKED
                )
                break
            if not det.flagged:
                # The available detector found no above-threshold obstruction.
                if attempt > 1:
                    outcome = CorrectionOutcome.REPAIRED
                else:
                    outcome = CorrectionOutcome.NO_OBSTRUCTION_DETECTED
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
                prompt=torch.tensor(prefix_tokens, dtype=torch.long, device=model_device),
                max_new_tokens=repair_len,
                control=ControlType.DELIBERATION,
                temperature=temperature,
                top_k=top_k,
                rng=rng,
            )
            regenerated = self._validated_generated_tokens(
                regen_traj,
                prompt_tokens=prefix_tokens,
                expected_new_tokens=repair_len,
            )
            repaired_toks = regenerated[span_start_abs:]

            # Stitch replacement span back into sequence
            new_tokens = (
                current_tokens[:span_start_abs]
                + repaired_toks
                + current_tokens[span_end_abs:]
            )
            current_tokens = new_tokens

            # Re-check repaired representations
            with torch.no_grad():
                tok_tensor_re = torch.tensor(
                    [current_tokens],
                    dtype=torch.long,
                    device=model_device,
                )
                h_seq_re = self._extract_hidden_states(tok_tensor_re)
                h_gen_re = h_seq_re[0, prompt_len:]
                edge_index_re = self._path_edge_index(h_gen_re.shape[0], h_gen_re.device)
                det_re: SheafDetectorDecision = self.sheaf_detector.inspect(
                    h_gen_re,
                    edge_index_re,
                )

            repaired_ok = det_re.available and not det_re.flagged
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
                outcome = CorrectionOutcome.ABSTAIN
                is_abstain = True
                current_tokens = prompt_list + [self.cfg.abstain_token_id]
            else:
                outcome = CorrectionOutcome.UNRESOLVED

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
class SelfCorrectionEvalSample:
    """One paired case whose oracle scores the full prompt-plus-completion token sequence."""

    prompt: Tensor
    max_new_tokens: int
    is_error: Callable[[Sequence[int]], bool]
    expected_inconsistency: bool
    name: str = ""


@dataclass
class SelfCorrectionSampleResult:
    """Auditable outputs and scores for both arms of one paired sample."""

    name: str
    expected_inconsistency: bool
    baseline_tokens: list[int]
    self_correcting_tokens: list[int]
    baseline_error: bool
    self_correcting_output_error: bool
    self_correcting_failure: bool
    abstained: bool
    baseline_latency_ms: float
    self_correcting_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expected_inconsistency": self.expected_inconsistency,
            "baseline_tokens": list(self.baseline_tokens),
            "self_correcting_tokens": list(self.self_correcting_tokens),
            "baseline_error": self.baseline_error,
            "self_correcting_output_error": self.self_correcting_output_error,
            "self_correcting_failure": self.self_correcting_failure,
            "abstained": self.abstained,
            "baseline_latency_ms": self.baseline_latency_ms,
            "self_correcting_latency_ms": self.self_correcting_latency_ms,
        }


@dataclass
class SelfCorrectionEvalReport:
    total_samples: int
    clean_samples: int
    inconsistent_samples: int
    single_pass_errors: int
    self_correcting_errors: int
    single_pass_error_rate: float
    self_correcting_error_rate: float
    error_reduction_pct: float | None
    repaired_count: int
    abstention_count: int
    answered_count: int
    answered_error_count: int
    coverage: float
    answered_error_rate: float
    avg_baseline_latency_ms: float
    avg_attempts_used: float
    avg_latency_ms: float
    latency_overhead_ratio: float | None
    cumulative_single_pass_error_rate: tuple[float, ...]
    cumulative_self_correcting_error_rate: tuple[float, ...]
    verdict: str
    verdict_reason: str
    sample_results: list[SelfCorrectionSampleResult]
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
            "answered_count": self.answered_count,
            "answered_error_count": self.answered_error_count,
            "coverage": self.coverage,
            "answered_error_rate": self.answered_error_rate,
            "avg_baseline_latency_ms": self.avg_baseline_latency_ms,
            "avg_attempts_used": self.avg_attempts_used,
            "avg_latency_ms": self.avg_latency_ms,
            "latency_overhead_ratio": self.latency_overhead_ratio,
            "cumulative_single_pass_error_rate": list(
                self.cumulative_single_pass_error_rate
            ),
            "cumulative_self_correcting_error_rate": list(
                self.cumulative_self_correcting_error_rate
            ),
            "verdict": self.verdict,
            "verdict_reason": self.verdict_reason,
            "sample_results": [result.to_dict() for result in self.sample_results],
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
        reduction = (
            f"{self.error_reduction_pct:.1f}%"
            if self.error_reduction_pct is not None
            else "N/A (zero baseline errors)"
        )
        t.add_row("Error Reduction", reduction)
        t.add_row("Repaired Inconsistencies", str(self.repaired_count))
        t.add_row("Abstentions", str(self.abstention_count))
        t.add_row("Coverage", f"{self.coverage * 100:.1f}%")
        t.add_row("Answered Error Rate", f"{self.answered_error_rate * 100:.1f}%")
        t.add_row("Avg Baseline Latency (ms)", f"{self.avg_baseline_latency_ms:.2f}")
        t.add_row("Avg Attempts Used", f"{self.avg_attempts_used:.2f}")
        t.add_row("Avg Latency (ms)", f"{self.avg_latency_ms:.2f}")
        overhead = (
            f"{self.latency_overhead_ratio:.2f}x"
            if self.latency_overhead_ratio is not None
            else "N/A"
        )
        t.add_row("Latency Overhead", overhead)
        t.add_row("Verdict", self.verdict)
        return t


def evaluate_self_correction_benchmark(
    generator: SelfCorrectingGenerator,
    labeled_samples: Sequence[SelfCorrectionEvalSample],
    *,
    events_jsonl_path: Path | str | None = None,
    temperature: float = 0.8,
    top_k: int | None = None,
    seed: int = 0,
) -> SelfCorrectionEvalReport:
    """Run an oracle-scored paired baseline/self-correction evaluation.

    Both arms start from identical model state and RNG seeds. Abstention counts as
    a failure in the primary error rate, so selective refusal cannot manufacture
    an apparent quality improvement; coverage and answered-only risk are reported
    separately. The verdict is a descriptive comparison of observed paired counts,
    not a statistical-significance claim.
    """
    total = len(labeled_samples)
    if total == 0:
        raise ValueError("labeled_samples must not be empty")
    if not generator.cfg.enabled:
        raise ValueError("benchmark requires self-correction to be enabled")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, numbers.Real)
        or not math.isfinite(float(temperature))
        or temperature < 0.0
    ):
        raise ValueError("temperature must be finite and non-negative")
    if top_k is not None and (
        isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0
    ):
        raise ValueError("top_k must be a non-negative integer")
    if any(not isinstance(sample, SelfCorrectionEvalSample) for sample in labeled_samples):
        raise TypeError("labeled_samples must contain SelfCorrectionEvalSample values")
    for sample_index, sample in enumerate(labeled_samples):
        if not isinstance(sample.expected_inconsistency, bool):
            raise ValueError(
                f"sample {sample_index} expected_inconsistency must be a boolean"
            )
        if (
            isinstance(sample.max_new_tokens, bool)
            or not isinstance(sample.max_new_tokens, int)
            or sample.max_new_tokens < 0
        ):
            raise ValueError(
                f"sample {sample_index} max_new_tokens must be a non-negative integer"
            )
        if not callable(sample.is_error):
            raise TypeError(f"sample {sample_index} is_error must be callable")
        if not isinstance(sample.name, str):
            raise TypeError(f"sample {sample_index} name must be a string")

    clean_count = sum(not sample.expected_inconsistency for sample in labeled_samples)
    inconsistent_count = total - clean_count

    single_pass_errors = 0
    self_correcting_errors = 0
    repaired_count = 0
    abstention_count = 0
    answered_error_count = 0
    baseline_latencies: list[float] = []
    correction_latencies: list[float] = []
    trajectories: list[SelfCorrectingTrajectory] = []
    sample_results: list[SelfCorrectionSampleResult] = []
    cumulative_baseline: list[float] = []
    cumulative_corrected: list[float] = []

    model_state = {
        name: value.detach().clone()
        for name, value in generator.model.state_dict().items()
    }
    model_training = generator.model.training
    fork_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []

    def restore_model(*, for_evaluation: bool) -> None:
        generator.model.load_state_dict(model_state, strict=True)
        if for_evaluation:
            generator.model.eval()
        else:
            generator.model.train(model_training)
        if not for_evaluation:
            return
        reset_sequence_state = getattr(generator.model, "reset_sequence_state", None)
        if callable(reset_sequence_state):
            reset_sequence_state()

    def score_output(
        sample: SelfCorrectionEvalSample,
        tokens: list[int],
        *,
        sample_index: int,
        arm: str,
    ) -> bool:
        try:
            result = sample.is_error(tuple(tokens))
        except Exception as error:
            raise RuntimeError(
                f"error oracle failed for sample {sample_index} ({arm})"
            ) from error
        if not isinstance(result, bool):
            raise TypeError(
                f"error oracle for sample {sample_index} ({arm}) must return bool"
            )
        return result

    try:
        for sample_index, sample in enumerate(labeled_samples):
            sample_seed = seed + sample_index
            prompt = generator._validated_prompt(sample.prompt)
            prompt_tokens = [int(token) for token in prompt.tolist()]
            model_device = generator._model_device(prompt.device)

            restore_model(for_evaluation=True)
            baseline_rng = torch.Generator(device=model_device).manual_seed(sample_seed)
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(sample_seed)
                baseline_start = time.perf_counter()
                baseline_trajectory = generator.deliberation_controller.generate(
                    prompt=prompt,
                    max_new_tokens=sample.max_new_tokens,
                    control=ControlType.BASELINE,
                    temperature=temperature,
                    top_k=top_k,
                    rng=baseline_rng,
                )
                baseline_latency = (time.perf_counter() - baseline_start) * 1000.0
            baseline_tokens = generator._validated_generated_tokens(
                baseline_trajectory,
                prompt_tokens=prompt_tokens,
                expected_new_tokens=sample.max_new_tokens,
            )
            baseline_error = score_output(
                sample,
                baseline_tokens,
                sample_index=sample_index,
                arm="single_pass",
            )

            restore_model(for_evaluation=True)
            correction_rng = torch.Generator(device=model_device).manual_seed(sample_seed)
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(sample_seed)
                correction_start = time.perf_counter()
                trajectory = generator.generate(
                    prompt,
                    max_new_tokens=sample.max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    rng=correction_rng,
                )
                correction_latency = (time.perf_counter() - correction_start) * 1000.0
            trajectories.append(trajectory)

            output_error = score_output(
                sample,
                trajectory.final_tokens,
                sample_index=sample_index,
                arm="self_correcting",
            )
            correction_failure = trajectory.is_abstention or output_error

            single_pass_errors += int(baseline_error)
            self_correcting_errors += int(correction_failure)
            if not trajectory.is_abstention:
                answered_error_count += int(output_error)
            repaired_count += int(trajectory.outcome is CorrectionOutcome.REPAIRED)
            abstention_count += int(trajectory.is_abstention)
            baseline_latencies.append(baseline_latency)
            correction_latencies.append(correction_latency)

            sample_results.append(
                SelfCorrectionSampleResult(
                    name=sample.name or f"sample_{sample_index}",
                    expected_inconsistency=sample.expected_inconsistency,
                    baseline_tokens=baseline_tokens,
                    self_correcting_tokens=list(trajectory.final_tokens),
                    baseline_error=baseline_error,
                    self_correcting_output_error=output_error,
                    self_correcting_failure=correction_failure,
                    abstained=trajectory.is_abstention,
                    baseline_latency_ms=baseline_latency,
                    self_correcting_latency_ms=correction_latency,
                )
            )
            observed = sample_index + 1
            cumulative_baseline.append(single_pass_errors / observed)
            cumulative_corrected.append(self_correcting_errors / observed)

            if events_jsonl_path is not None:
                trajectory.append_jsonl(events_jsonl_path)
    finally:
        restore_model(for_evaluation=False)

    single_err_rate = single_pass_errors / total
    self_err_rate = self_correcting_errors / total
    err_reduction = (
        (single_err_rate - self_err_rate) / single_err_rate * 100.0
        if single_err_rate > 0
        else None
    )
    answered_count = total - abstention_count
    coverage = answered_count / total
    answered_error_rate = (
        answered_error_count / answered_count if answered_count > 0 else 0.0
    )
    avg_baseline_latency = sum(baseline_latencies) / total
    avg_attempts = sum(t.attempts_used for t in trajectories) / total
    avg_latency = sum(correction_latencies) / total
    overhead = avg_latency / avg_baseline_latency if avg_baseline_latency > 0.0 else None
    if self_correcting_errors < single_pass_errors:
        verdict = "improved"
    elif self_correcting_errors > single_pass_errors:
        verdict = "worse"
    else:
        verdict = "null"
    verdict_reason = (
        f"paired primary failures {single_pass_errors}/{total} single-pass versus "
        f"{self_correcting_errors}/{total} self-correcting; "
        f"coverage {answered_count}/{total}"
    )

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
        answered_count=answered_count,
        answered_error_count=answered_error_count,
        coverage=coverage,
        answered_error_rate=answered_error_rate,
        avg_baseline_latency_ms=avg_baseline_latency,
        avg_attempts_used=avg_attempts,
        avg_latency_ms=avg_latency,
        latency_overhead_ratio=overhead,
        cumulative_single_pass_error_rate=tuple(cumulative_baseline),
        cumulative_self_correcting_error_rate=tuple(cumulative_corrected),
        verdict=verdict,
        verdict_reason=verdict_reason,
        sample_results=sample_results,
        trajectories=trajectories,
    )
