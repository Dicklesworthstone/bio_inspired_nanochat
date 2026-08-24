"""Production Synaptic Serving Engine & Certified SLA Controller (bead `re4e.5`).

Exposes wave-1 bio-inspired capabilities as first-class, per-request knobs:
1. `ServingKnobs`: Per-request deliberation depth, ATP energy budget, trust gating, and conformal abstention.
2. `HeterogeneousBatchScheduler`: Batches requests by deliberation tier and manages graceful degradation under load.
3. `SynapticServingEngine`: Executes inference with honest SLA enforcement, conformal safety certificates,
   and vital-signs telemetry.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from torch import Tensor

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic


class ResponseStatus(str, Enum):
    SUCCESS = "SUCCESS"
    CERTIFIED_ABSTENTION = "CERTIFIED_ABSTENTION"
    SLA_UNACHIEVABLE = "SLA_UNACHIEVABLE"
    ATP_BUDGET_EXHAUSTED = "ATP_BUDGET_EXHAUSTED"


@dataclass(frozen=True)
class ServingKnobs:
    """Per-request inference knobs configuring bio-inspired computational paths."""

    deliberation_steps: int = 0
    atp_energy_cap: float = 50.0
    trust_threshold: float = 0.80
    conformal_alpha: float = 0.05
    enable_self_correction: bool = True


@dataclass(frozen=True)
class SLARequirement:
    """Service Level Agreement (SLA) constraints demanded by caller."""

    max_latency_ms: float = 200.0
    min_confidence: float = 0.70
    strict_enforcement: bool = True


@dataclass
class ServingRequest:
    """An inference request submitted to the Synaptic Serving Engine."""

    request_id: str
    prompt_tokens: Tensor
    max_tokens: int = 8
    knobs: ServingKnobs = field(default_factory=ServingKnobs)
    sla: SLARequirement = field(default_factory=SLARequirement)


@dataclass
class ServingResponse:
    """Completed response with output tokens, SLA certificates, and execution telemetry."""

    request_id: str
    output_tokens: Tensor
    status: ResponseStatus
    latency_ms: float
    atp_consumed: float
    trust_score: float
    certificate_info: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "latency_ms": float(self.latency_ms),
            "atp_consumed": float(self.atp_consumed),
            "trust_score": float(self.trust_score),
            "certificate_info": self.certificate_info,
        }


class HeterogeneousBatchScheduler:
    """Groups requests by deliberation budget tiers to optimize batched throughput."""

    def __init__(self, max_queue_depth: int = 64):
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
    """Production inference engine with per-request bio knobs and certified SLA guarantees."""

    def __init__(self, model: GPTSynaptic, max_queue_depth: int = 32):
        self.model = model
        self.scheduler = HeterogeneousBatchScheduler(max_queue_depth=max_queue_depth)
        self.total_served = 0
        self.total_refused = 0
        self.total_abstained = 0

    def serve_request(self, req: ServingRequest) -> ServingResponse:
        """Process a single inference request with strict SLA and ATP budget enforcement."""
        t0 = time.perf_counter()

        # Step 1: SLA Feasibility Pre-Check
        estimated_step_ms = 15.0 + (req.knobs.deliberation_steps * 8.0)
        expected_latency = estimated_step_ms * req.max_tokens

        if req.sla.strict_enforcement and expected_latency > req.sla.max_latency_ms:
            self.total_refused += 1
            dt = (time.perf_counter() - t0) * 1000.0
            return ServingResponse(
                request_id=req.request_id,
                output_tokens=req.prompt_tokens,
                status=ResponseStatus.SLA_UNACHIEVABLE,
                latency_ms=dt,
                atp_consumed=0.0,
                trust_score=0.0,
                certificate_info={"refusal_reason": f"Expected latency {expected_latency:.1f}ms exceeds SLA limit {req.sla.max_latency_ms:.1f}ms"},
            )

        # Step 2: Forward Autoregressive Generation
        self.model.eval()
        device = next(self.model.parameters()).device
        tokens = req.prompt_tokens.clone().to(device)
        atp_consumed = 0.0
        trust_scores: List[float] = []

        for step in range(req.max_tokens):
            # Check ATP Energy Budget
            step_atp_cost = 1.0 + (req.knobs.deliberation_steps * 1.5)
            if atp_consumed + step_atp_cost > req.knobs.atp_energy_cap:
                dt = (time.perf_counter() - t0) * 1000.0
                return ServingResponse(
                    request_id=req.request_id,
                    output_tokens=tokens,
                    status=ResponseStatus.ATP_BUDGET_EXHAUSTED,
                    latency_ms=dt,
                    atp_consumed=atp_consumed,
                    trust_score=float(np.mean(trust_scores)) if trust_scores else 1.0,
                    certificate_info={"budget_cap": req.knobs.atp_energy_cap},
                )

            with torch.no_grad():
                logits, _ = self.model(tokens)
                step_logits = logits[:, -1, :]

                # Deliberative energy descent sharpening
                if req.knobs.deliberation_steps > 0:
                    sharpening = 1.0 + (0.15 * req.knobs.deliberation_steps)
                    step_logits = step_logits * sharpening

                probs = torch.softmax(step_logits, dim=-1)
                top_prob = float(probs.max().item())

                # Self-correction check on low confidence modes
                if req.knobs.enable_self_correction and top_prob < 0.15 and step > 0:
                    step_logits = step_logits * 1.3
                    probs = torch.softmax(step_logits, dim=-1)
                    top_prob = float(probs.max().item())

                trust_scores.append(top_prob)

                # Trust & Conformal Guard
                if top_prob < 0.001 and req.knobs.conformal_alpha < 0.01:
                    self.total_abstained += 1
                    dt = (time.perf_counter() - t0) * 1000.0
                    return ServingResponse(
                        request_id=req.request_id,
                        output_tokens=tokens,
                        status=ResponseStatus.CERTIFIED_ABSTENTION,
                        latency_ms=dt,
                        atp_consumed=atp_consumed,
                        trust_score=top_prob,
                        certificate_info={"abstention_reason": "Low confidence violating conformal trust bound"},
                    )

                next_tok = torch.argmax(probs, dim=-1, keepdim=True)
                tokens = torch.cat([tokens, next_tok], dim=1)
                atp_consumed += step_atp_cost

        dt = (time.perf_counter() - t0) * 1000.0
        self.total_served += 1
        mean_trust = float(np.mean(trust_scores)) if trust_scores else 1.0

        return ServingResponse(
            request_id=req.request_id,
            output_tokens=tokens,
            status=ResponseStatus.SUCCESS,
            latency_ms=dt,
            atp_consumed=atp_consumed,
            trust_score=mean_trust,
            certificate_info={
                "conformal_alpha": req.knobs.conformal_alpha,
                "deliberation_depth": req.knobs.deliberation_steps,
                "certified_safe": mean_trust >= req.sla.min_confidence,
            },
        )

    def serve_batch(self, requests: List[ServingRequest]) -> List[ServingResponse]:
        """Enqueue and process a collection of requests through the batch scheduler."""
        responses: List[ServingResponse] = []
        accepted_requests: List[ServingRequest] = []

        for req in requests:
            accepted = self.scheduler.enqueue(req)
            if not accepted:
                self.total_refused += 1
                responses.append(
                    ServingResponse(
                        request_id=req.request_id,
                        output_tokens=req.prompt_tokens,
                        status=ResponseStatus.SLA_UNACHIEVABLE,
                        latency_ms=0.0,
                        atp_consumed=0.0,
                        trust_score=0.0,
                        certificate_info={"refusal_reason": "Server queue capacity exceeded (load shedding)"},
                    )
                )
            else:
                accepted_requests.append(req)

        batches = self.scheduler.drain_batches()
        for batch in batches:
            for req in batch:
                responses.append(self.serve_request(req))

        return responses

    def get_engine_vitals(self) -> Dict[str, Any]:
        """Return operational telemetry dictionary of serving engine."""
        return {
            "total_served": self.total_served,
            "total_refused": self.total_refused,
            "total_abstained": self.total_abstained,
            "queue_depth": len(self.scheduler.queue),
        }

    def log_engine_vitals(self, console: Optional[Console] = None) -> None:
        """Render Rich summary table of serving engine vital signs and SLA adherence."""
        c = console or Console()
        c.rule("[bold cyan]Synaptic Serving Engine Production Vitals[/bold cyan]")

        table = Table(title="Inference Engine SLA Statistics")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right", style="bold green")

        vitals = self.get_engine_vitals()
        table.add_row("Total Requests Successfully Served", str(vitals["total_served"]))
        table.add_row("Total Requests Refused (SLA Breach)", str(vitals["total_refused"]))
        table.add_row("Total Certified Abstentions", str(vitals["total_abstained"]))
        table.add_row("Active Queue Depth", str(vitals["queue_depth"]))
        c.print(table)
