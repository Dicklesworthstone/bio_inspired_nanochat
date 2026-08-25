"""Synaptic Serving Engine with explicit SLA and confidence guards (bead `re4e.5`).

Exposes wave-1 bio-inspired capabilities as first-class, per-request knobs:
1. `ServingKnobs`: Per-request deliberation depth, ATP energy budget, and confidence gating.
2. `HeterogeneousBatchScheduler`: Batches requests by deliberation tier and manages graceful degradation under load.
3. `SynapticServingEngine`: Executes inference with explicit heuristic guard decisions and
   vital-signs telemetry. It does not claim statistical calibration or coverage guarantees.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from torch import Tensor

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic


@contextmanager
def _temporary_eval(model: torch.nn.Module) -> Iterator[None]:
    """Evaluate without flattening or permanently changing caller-owned module modes."""
    training_modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, was_training in training_modes:
            module.training = was_training


class ResponseStatus(str, Enum):
    SUCCESS = "SUCCESS"
    CONFIDENCE_ABSTENTION = "CONFIDENCE_ABSTENTION"
    SLA_UNACHIEVABLE = "SLA_UNACHIEVABLE"
    ATP_BUDGET_EXHAUSTED = "ATP_BUDGET_EXHAUSTED"


@dataclass(frozen=True)
class ServingKnobs:
    """Per-request inference knobs configuring bio-inspired computational paths."""

    deliberation_steps: int = 0
    atp_energy_cap: float = 50.0
    trust_threshold: float = 0.80
    enable_self_correction: bool = True
    adaptive_serving: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.deliberation_steps, bool) or not isinstance(
            self.deliberation_steps, int
        ) or self.deliberation_steps < 0:
            raise ValueError("deliberation_steps must be a non-negative integer")
        if not math.isfinite(self.atp_energy_cap) or self.atp_energy_cap < 0.0:
            raise ValueError("atp_energy_cap must be finite and non-negative")
        if not math.isfinite(self.trust_threshold) or not 0.0 <= self.trust_threshold <= 1.0:
            raise ValueError("trust_threshold must be finite and in [0, 1]")
        if not isinstance(self.enable_self_correction, bool):
            raise ValueError("enable_self_correction must be a boolean")
        if not isinstance(self.adaptive_serving, bool):
            raise ValueError("adaptive_serving must be a boolean")


@dataclass(frozen=True)
class SLARequirement:
    """Service Level Agreement (SLA) constraints demanded by caller."""

    max_latency_ms: float = 200.0
    min_confidence: float = 0.70
    strict_enforcement: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_latency_ms) or self.max_latency_ms <= 0.0:
            raise ValueError("max_latency_ms must be finite and positive")
        if not math.isfinite(self.min_confidence) or not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be finite and in [0, 1]")
        if not isinstance(self.strict_enforcement, bool):
            raise ValueError("strict_enforcement must be a boolean")


@dataclass
class ServingRequest:
    """An inference request submitted to the Synaptic Serving Engine."""

    request_id: str
    prompt_tokens: Tensor
    max_tokens: int = 8
    knobs: ServingKnobs = field(default_factory=ServingKnobs)
    sla: SLARequirement = field(default_factory=SLARequirement)

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be non-empty")
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int):
            raise ValueError("max_tokens must be a non-negative integer")
        if self.max_tokens < 0:
            raise ValueError("max_tokens must be a non-negative integer")
        if not isinstance(self.prompt_tokens, Tensor):
            raise ValueError("prompt_tokens must be a tensor")
        if self.prompt_tokens.ndim != 2 or self.prompt_tokens.numel() == 0:
            raise ValueError("prompt_tokens must be a non-empty rank-2 tensor")
        if self.prompt_tokens.dtype not in {torch.int32, torch.int64}:
            raise ValueError("prompt_tokens must contain integer token IDs")


@dataclass
class ServingResponse:
    """Completed response with output tokens, guard metadata, and execution telemetry."""

    request_id: str
    output_tokens: Tensor
    status: ResponseStatus
    latency_ms: float
    atp_consumed: float
    trust_score: float
    decision_info: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "latency_ms": float(self.latency_ms),
            "atp_consumed": float(self.atp_consumed),
            "trust_score": float(self.trust_score),
            "decision_info": self.decision_info,
        }


class HeterogeneousBatchScheduler:
    """Groups requests by deliberation budget tiers to optimize batched throughput."""

    def __init__(self, max_queue_depth: int = 64):
        if isinstance(max_queue_depth, bool) or not isinstance(max_queue_depth, int):
            raise ValueError("max_queue_depth must be a positive integer")
        if max_queue_depth <= 0:
            raise ValueError("max_queue_depth must be a positive integer")
        self.max_queue_depth = max_queue_depth
        self.queue: List[ServingRequest] = []

    def enqueue(self, req: ServingRequest) -> bool:
        """Add request to queue; returns False if queue capacity is exceeded (load shedding)."""
        if len(self.queue) >= self.max_queue_depth:
            return False
        self.queue.append(req)
        return True

    def drain_batches(self) -> List[List[ServingRequest]]:
        """Form batches partitioned by deliberation depth tiers."""
        if not self.queue:
            return []

        fast_tier: List[ServingRequest] = []
        delib_tier: List[ServingRequest] = []

        for req in self.queue:
            if req.knobs.deliberation_steps == 0:
                fast_tier.append(req)
            else:
                delib_tier.append(req)

        self.queue.clear()
        batches: List[List[ServingRequest]] = []
        if fast_tier:
            batches.append(fast_tier)
        if delib_tier:
            batches.append(delib_tier)
        return batches


class SynapticServingEngine:
    """Inference engine with per-request bio knobs and auditable heuristic guards."""

    def __init__(
        self,
        model: GPTSynaptic,
        max_queue_depth: int = 32,
        max_batch_size: int = 64,
    ):
        if (
            isinstance(max_batch_size, bool)
            or not isinstance(max_batch_size, int)
            or max_batch_size <= 0
        ):
            raise ValueError("max_batch_size must be a positive integer")
        self.model = model
        self.scheduler = HeterogeneousBatchScheduler(max_queue_depth=max_queue_depth)
        self.max_batch_size = max_batch_size
        self.total_served = 0
        self.total_refused = 0
        self.total_abstained = 0

    def serve_request(self, req: ServingRequest) -> ServingResponse:
        """Process one request with heuristic latency, confidence, and ATP guards."""
        t0 = time.perf_counter()
        batch_size, prompt_length = req.prompt_tokens.shape
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"request batch size {batch_size} exceeds engine limit {self.max_batch_size}"
            )
        if (
            req.prompt_tokens.min().item() < 0
            or req.prompt_tokens.max().item() >= self.model.config.vocab_size
        ):
            raise ValueError("prompt_tokens contain token IDs outside the model vocabulary")

        # Step 1: estimate only the work the ATP budget can actually fund. Sequence length
        # is included because this implementation recomputes the growing prefix every step.
        estimated_step_ms = 15.0 + (req.knobs.deliberation_steps * 8.0)
        base_atp_cost = 1.0 + (req.knobs.deliberation_steps * 1.5)

        def step_cost(generated_tokens: int) -> float:
            sequence_factor = 1.0 + (
                (prompt_length + generated_tokens) / self.model.config.sequence_len
            )
            return base_atp_cost * batch_size * sequence_factor

        planned_steps = 0
        planned_atp = 0.0
        available_context = self.model.config.sequence_len - prompt_length
        if available_context < 0:
            raise ValueError("prompt exceeds the model sequence length")
        while planned_steps < req.max_tokens:
            cost = step_cost(planned_steps)
            if planned_atp + cost > req.knobs.atp_energy_cap:
                break
            if planned_steps >= available_context:
                raise ValueError(
                    "ATP budget permits generation beyond the model sequence length"
                )
            planned_atp += cost
            planned_steps += 1

        expected_latency = sum(
            estimated_step_ms
            * batch_size
            * (1.0 + ((prompt_length + step) / self.model.config.sequence_len))
            for step in range(planned_steps)
        )

        if req.sla.strict_enforcement and expected_latency > req.sla.max_latency_ms:
            self.total_refused += 1
            dt = (time.perf_counter() - t0) * 1000.0
            return ServingResponse(
                request_id=req.request_id,
                output_tokens=req.prompt_tokens.detach().clone(),
                status=ResponseStatus.SLA_UNACHIEVABLE,
                latency_ms=dt,
                atp_consumed=0.0,
                trust_score=0.0,
                decision_info={"refusal_reason": f"Expected latency {expected_latency:.1f}ms exceeds SLA limit {req.sla.max_latency_ms:.1f}ms"},
            )

        # Step 2: Forward Autoregressive Generation
        device = next(self.model.parameters()).device
        tokens = req.prompt_tokens.clone().to(device)
        atp_consumed = 0.0
        trust_scores: List[float] = []
        required_confidence = req.knobs.trust_threshold
        if req.sla.strict_enforcement:
            required_confidence = max(required_confidence, req.sla.min_confidence)

        with _temporary_eval(self.model):
            for step in range(req.max_tokens):
                # Check ATP Energy Budget
                step_atp_cost = step_cost(step)
                if atp_consumed + step_atp_cost > req.knobs.atp_energy_cap:
                    dt = (time.perf_counter() - t0) * 1000.0
                    return ServingResponse(
                        request_id=req.request_id,
                        output_tokens=tokens,
                        status=ResponseStatus.ATP_BUDGET_EXHAUSTED,
                        latency_ms=dt,
                        atp_consumed=atp_consumed,
                        trust_score=float(np.mean(trust_scores)) if trust_scores else 0.0,
                        decision_info={
                            "budget_cap": req.knobs.atp_energy_cap,
                            "confidence_evaluated": bool(trust_scores),
                        },
                    )

                with torch.no_grad():
                    logits, _ = self.model(tokens, train_mode=req.knobs.adaptive_serving)
                    step_logits = logits[:, -1, :]

                compute_latency_ms = (time.perf_counter() - t0) * 1000.0
                if (
                    req.sla.strict_enforcement
                    and compute_latency_ms > req.sla.max_latency_ms
                ):
                    self.total_refused += 1
                    return ServingResponse(
                        request_id=req.request_id,
                        output_tokens=tokens,
                        status=ResponseStatus.SLA_UNACHIEVABLE,
                        latency_ms=compute_latency_ms,
                        atp_consumed=atp_consumed + step_atp_cost,
                        trust_score=float(np.mean(trust_scores)) if trust_scores else 0.0,
                        decision_info={
                            "refusal_reason": "Measured latency exceeded the configured limit",
                            "latency_limit_ms": req.sla.max_latency_ms,
                        },
                    )

                if not torch.isfinite(step_logits).all():
                    self.total_abstained += 1
                    return ServingResponse(
                        request_id=req.request_id,
                        output_tokens=tokens,
                        status=ResponseStatus.CONFIDENCE_ABSTENTION,
                        latency_ms=compute_latency_ms,
                        atp_consumed=atp_consumed + step_atp_cost,
                        trust_score=0.0,
                        decision_info={
                            "abstention_reason": "Model produced non-finite logits",
                            "guard_method": "finite-logit check",
                        },
                    )

                # Deliberative energy descent sharpening
                if req.knobs.deliberation_steps > 0:
                    sharpening = 1.0 + (0.15 * req.knobs.deliberation_steps)
                    step_logits = step_logits * sharpening

                probs = torch.softmax(step_logits, dim=-1)
                if not torch.isfinite(probs).all():
                    self.total_abstained += 1
                    return ServingResponse(
                        request_id=req.request_id,
                        output_tokens=tokens,
                        status=ResponseStatus.CONFIDENCE_ABSTENTION,
                        latency_ms=(time.perf_counter() - t0) * 1000.0,
                        atp_consumed=atp_consumed + step_atp_cost,
                        trust_score=0.0,
                        decision_info={
                            "abstention_reason": "Model produced non-finite probabilities",
                            "guard_method": "finite-probability check",
                        },
                    )
                # A request can carry a batch. Certification is only as strong as its
                # least-confident row; a global max would let one confident example mask
                # an unsafe peer in the same request.
                top_prob = float(probs.max(dim=-1).values.min().item())

                # Self-correction check on low confidence modes
                if req.knobs.enable_self_correction and top_prob < 0.15 and step > 0:
                    step_logits = step_logits * 1.3
                    probs = torch.softmax(step_logits, dim=-1)
                    top_prob = float(probs.max(dim=-1).values.min().item())

                trust_scores.append(top_prob)

                # Trust guard: per-request gating is always binding, while the SLA floor is
                # additionally binding for strict requests. Never label a response successful
                # merely because token generation itself completed when its floor was missed.
                if top_prob < required_confidence:
                    self.total_abstained += 1
                    dt = (time.perf_counter() - t0) * 1000.0
                    return ServingResponse(
                        request_id=req.request_id,
                        output_tokens=tokens,
                        status=ResponseStatus.CONFIDENCE_ABSTENTION,
                        latency_ms=dt,
                        atp_consumed=atp_consumed + step_atp_cost,
                        trust_score=top_prob,
                        decision_info={
                            "abstention_reason": "Token confidence below required trust floor",
                            "observed_confidence": top_prob,
                            "required_confidence": required_confidence,
                            "guard_method": "minimum top-token probability",
                        },
                    )

                next_tok = torch.argmax(probs, dim=-1, keepdim=True)
                tokens = torch.cat([tokens, next_tok], dim=1)
                atp_consumed += step_atp_cost

        dt = (time.perf_counter() - t0) * 1000.0
        self.total_served += 1
        mean_trust = float(np.mean(trust_scores)) if trust_scores else 0.0

        return ServingResponse(
            request_id=req.request_id,
            output_tokens=tokens,
            status=ResponseStatus.SUCCESS,
            latency_ms=dt,
            atp_consumed=atp_consumed,
            trust_score=mean_trust,
            decision_info={
                "deliberation_depth": req.knobs.deliberation_steps,
                "required_confidence": required_confidence,
                "confidence_evaluated": bool(trust_scores),
                "confidence_floor_met": (
                    mean_trust >= required_confidence if trust_scores else None
                ),
                "guard_method": "minimum top-token probability",
            },
        )

    def serve_batch(self, requests: List[ServingRequest]) -> List[ServingResponse]:
        """Enqueue and process a collection of requests through the batch scheduler."""
        request_ids = [req.request_id for req in requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("request_id values must be unique within a batch")

        responses_by_id: Dict[str, ServingResponse] = {}

        for req in requests:
            accepted = self.scheduler.enqueue(req)
            if not accepted:
                self.total_refused += 1
                responses_by_id[req.request_id] = ServingResponse(
                    request_id=req.request_id,
                    output_tokens=req.prompt_tokens.detach().clone(),
                    status=ResponseStatus.SLA_UNACHIEVABLE,
                    latency_ms=0.0,
                    atp_consumed=0.0,
                    trust_score=0.0,
                    decision_info={
                        "refusal_reason": "Server queue capacity exceeded (load shedding)"
                    },
                )

        batches = self.scheduler.drain_batches()
        for batch in batches:
            for req in batch:
                responses_by_id[req.request_id] = self.serve_request(req)

        return [responses_by_id[request_id] for request_id in request_ids]

    def get_engine_vitals(self) -> Dict[str, Any]:
        """Return operational telemetry dictionary of serving engine."""
        return {
            "total_served": self.total_served,
            "total_refused": self.total_refused,
            "total_abstained": self.total_abstained,
            "queue_depth": len(self.scheduler.queue),
        }

    def log_engine_vitals(self, console: Optional[Console] = None) -> None:
        """Render a Rich summary of serving-engine outcomes and queue state."""
        c = console or Console()
        c.rule("[bold cyan]Synaptic Serving Engine Production Vitals[/bold cyan]")

        table = Table(title="Inference Engine Guard Statistics")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right", style="bold green")

        vitals = self.get_engine_vitals()
        table.add_row("Total Requests Successfully Served", str(vitals["total_served"]))
        table.add_row("Total Requests Refused (Latency/Load Guard)", str(vitals["total_refused"]))
        table.add_row("Total Confidence Abstentions", str(vitals["total_abstained"]))
        table.add_row("Active Queue Depth", str(vitals["queue_depth"]))
        c.print(table)
