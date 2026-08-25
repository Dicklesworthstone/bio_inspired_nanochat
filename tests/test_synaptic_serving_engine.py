"""Tests for the Synaptic Serving Engine and heuristic request guards (bead `re4e.5`)."""

import math
from typing import Any, cast

import pytest
import torch

from bio_inspired_nanochat import synaptic_serving_engine as serving_module
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.synaptic_serving_engine import (
    HeterogeneousBatchScheduler,
    ResponseStatus,
    ServingKnobs,
    ServingRequest,
    SLARequirement,
    SynapticServingEngine,
)


def _make_model() -> GPTSynaptic:
    cfg = GPTSynapticConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        synapses=True,
        use_moe=False,
    )
    return GPTSynaptic(cfg)


def test_serving_engine_successful_request():
    """Verify that a compliant request executes and returns auditable guard metadata."""
    model = _make_model()
    engine = SynapticServingEngine(model)

    req = ServingRequest(
        request_id="req_001",
        prompt_tokens=torch.randint(0, 32, (1, 3)),
        max_tokens=4,
        knobs=ServingKnobs(deliberation_steps=2, atp_energy_cap=30.0, trust_threshold=0.0),
        sla=SLARequirement(max_latency_ms=60_000.0, min_confidence=0.0),
    )

    resp = engine.serve_request(req)

    assert resp.status == ResponseStatus.SUCCESS
    assert resp.output_tokens.shape[1] == 7
    assert resp.atp_consumed > 0.0
    assert resp.decision_info["confidence_floor_met"] is True
    assert resp.decision_info["guard_method"] == "minimum top-token probability"


def test_serving_engine_sla_refusal():
    """Verify that unreasonable SLA constraints trigger honest rejection rather than violation."""
    model = _make_model()
    engine = SynapticServingEngine(model)

    req = ServingRequest(
        request_id="req_too_fast",
        prompt_tokens=torch.randint(0, 32, (1, 3)),
        max_tokens=10,
        knobs=ServingKnobs(deliberation_steps=5),  # High compute demand
        sla=SLARequirement(max_latency_ms=10.0, strict_enforcement=True),  # Impossible 10ms SLA
    )

    resp = engine.serve_request(req)

    assert resp.status == ResponseStatus.SLA_UNACHIEVABLE
    assert "refusal_reason" in resp.decision_info
    assert engine.total_refused == 1


def test_strict_sla_stops_on_measured_overrun(monkeypatch):
    model = _make_model()

    def controlled_forward(tokens, train_mode=True):
        logits = torch.zeros((*tokens.shape, model.config.vocab_size))
        return logits, None

    timestamps = iter((0.0, 0.2))
    monkeypatch.setattr(model, "forward", controlled_forward)
    monkeypatch.setattr(serving_module.time, "perf_counter", lambda: next(timestamps))

    response = SynapticServingEngine(model).serve_request(
        ServingRequest(
            "measured-overrun",
            torch.ones((1, 2), dtype=torch.long),
            max_tokens=1,
            knobs=ServingKnobs(trust_threshold=0.0),
            sla=SLARequirement(max_latency_ms=100.0, min_confidence=0.0),
        )
    )

    assert response.status == ResponseStatus.SLA_UNACHIEVABLE
    assert response.latency_ms == pytest.approx(200.0)
    assert "Measured latency" in response.decision_info["refusal_reason"]


def test_serving_engine_atp_energy_budget_exhaustion():
    """Verify that generation halts when per-request ATP energy cap is reached."""
    model = _make_model()
    engine = SynapticServingEngine(model)

    req = ServingRequest(
        request_id="req_capped",
        prompt_tokens=torch.randint(0, 32, (1, 2)),
        max_tokens=10,
        knobs=ServingKnobs(atp_energy_cap=3.5, trust_threshold=0.0),  # Only enough for ~3 tokens
        sla=SLARequirement(max_latency_ms=1000.0, min_confidence=0.0),
    )

    resp = engine.serve_request(req)

    assert resp.status == ResponseStatus.ATP_BUDGET_EXHAUSTED
    assert resp.atp_consumed <= 3.5
    assert resp.output_tokens.shape[1] < 12


def test_serving_engine_confidence_abstention():
    """Strict confidence requirements fail closed instead of returning unsafe success."""
    model = _make_model()
    engine = SynapticServingEngine(model)

    req = ServingRequest(
        request_id="req_abstain",
        prompt_tokens=torch.randint(0, 32, (1, 2)),
        max_tokens=4,
        knobs=ServingKnobs(trust_threshold=0.99),
        sla=SLARequirement(max_latency_ms=1000.0, min_confidence=0.99),
    )

    resp = engine.serve_request(req)
    assert resp.status == ResponseStatus.CONFIDENCE_ABSTENTION
    assert resp.atp_consumed > 0.0
    assert resp.decision_info["required_confidence"] == 0.99
    assert resp.decision_info["observed_confidence"] < 0.99
    assert resp.decision_info["guard_method"] == "minimum top-token probability"


@pytest.mark.parametrize("invalid_threshold", [-0.1, 1.1, math.nan, math.inf])
def test_serving_confidence_thresholds_reject_invalid_values(invalid_threshold):
    """Non-finite or out-of-range confidence floors cannot bypass fail-closed gating."""
    with pytest.raises(ValueError, match="trust_threshold"):
        ServingKnobs(trust_threshold=invalid_threshold)
    with pytest.raises(ValueError, match="min_confidence"):
        SLARequirement(min_confidence=invalid_threshold)


@pytest.mark.parametrize("invalid_steps", [-1, 1.5, True])
def test_serving_knobs_reject_invalid_deliberation_steps(invalid_steps):
    """Invalid compute depth cannot create negative ATP costs or bypass latency checks."""
    with pytest.raises(ValueError, match="deliberation_steps"):
        ServingKnobs(deliberation_steps=invalid_steps)


@pytest.mark.parametrize("invalid_cap", [-0.1, math.nan, math.inf])
def test_serving_knobs_reject_invalid_energy_cap(invalid_cap):
    with pytest.raises(ValueError, match="atp_energy_cap"):
        ServingKnobs(atp_energy_cap=invalid_cap)


@pytest.mark.parametrize("invalid_latency", [0.0, -1.0, math.nan, math.inf])
def test_serving_sla_rejects_invalid_latency(invalid_latency):
    with pytest.raises(ValueError, match="max_latency_ms"):
        SLARequirement(max_latency_ms=invalid_latency)


def test_serving_request_rejects_invalid_generation_shape_and_length():
    with pytest.raises(ValueError, match="max_tokens"):
        ServingRequest("negative", torch.ones((1, 2), dtype=torch.long), max_tokens=-1)
    with pytest.raises(ValueError, match="max_tokens"):
        ServingRequest(
            "fractional",
            torch.ones((1, 2), dtype=torch.long),
            max_tokens=cast(Any, 1.5),
        )
    with pytest.raises(ValueError, match="rank-2"):
        ServingRequest("empty", torch.empty((1, 0), dtype=torch.long))
    with pytest.raises(ValueError, match="rank-2"):
        ServingRequest("flat", torch.ones(2, dtype=torch.long))
    with pytest.raises(ValueError, match="integer token IDs"):
        ServingRequest("float", torch.ones((1, 2), dtype=torch.float32))
    with pytest.raises(ValueError, match="request_id"):
        ServingRequest("", torch.ones((1, 2), dtype=torch.long))
    with pytest.raises(ValueError, match="request_id"):
        ServingRequest("   ", torch.ones((1, 2), dtype=torch.long))
    with pytest.raises(ValueError, match="prompt_tokens must be a tensor"):
        ServingRequest("list", cast(Any, [[1, 2]]))


def test_serving_rejects_non_boolean_policy_switches():
    with pytest.raises(ValueError, match="enable_self_correction"):
        ServingKnobs(enable_self_correction=cast(Any, 1))
    with pytest.raises(ValueError, match="adaptive_serving"):
        ServingKnobs(adaptive_serving=cast(Any, 1))
    with pytest.raises(ValueError, match="strict_enforcement"):
        SLARequirement(strict_enforcement=cast(Any, 1))


def test_serving_request_rejects_context_overflow_and_invalid_token_ids():
    engine = SynapticServingEngine(_make_model())
    no_trust = ServingKnobs(trust_threshold=0.0)
    no_sla_floor = SLARequirement(min_confidence=0.0)

    with pytest.raises(ValueError, match="sequence length"):
        engine.serve_request(
            ServingRequest(
                "too-long",
                torch.ones((1, 7), dtype=torch.long),
                max_tokens=2,
                knobs=no_trust,
                sla=no_sla_floor,
            )
        )
    with pytest.raises(ValueError, match="outside the model vocabulary"):
        engine.serve_request(
            ServingRequest(
                "bad-token",
                torch.tensor([[32]], dtype=torch.long),
                max_tokens=0,
                knobs=no_trust,
                sla=no_sla_floor,
            )
        )

    with pytest.raises(ValueError, match="batch size"):
        SynapticServingEngine(_make_model(), max_batch_size=1).serve_request(
            ServingRequest(
                "oversized-batch",
                torch.ones((2, 1), dtype=torch.long),
                max_tokens=0,
                knobs=no_trust,
                sla=no_sla_floor,
            )
        )


def test_serving_is_non_adaptive_by_default():
    """Ordinary inference must not persist request-specific plasticity into later requests."""
    model = _make_model()
    engine = SynapticServingEngine(model)
    state_before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    request = ServingRequest(
        "isolated",
        torch.ones((1, 2), dtype=torch.long),
        max_tokens=1,
        knobs=ServingKnobs(trust_threshold=0.0),
        sla=SLARequirement(min_confidence=0.0, strict_enforcement=False),
    )

    response = engine.serve_request(request)

    assert response.status == ResponseStatus.SUCCESS
    assert all(
        torch.equal(value, state_before[name])
        for name, value in model.state_dict().items()
    )


def test_adaptive_serving_rejects_pending_training_plasticity_without_mutation():
    """Inference must not combine an unflushed training write with request adaptation."""
    model = _make_model()
    pending_module = next(
        module for module in model.modules() if hasattr(module, "_plasticity_pending")
    )
    pending_module._plasticity_pending = True
    state_before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    with pytest.raises(RuntimeError, match="clean post-backward plasticity boundary"):
        SynapticServingEngine(model).serve_request(
            ServingRequest(
                "unsafe-adaptation",
                torch.ones((1, 2), dtype=torch.long),
                max_tokens=1,
                knobs=ServingKnobs(adaptive_serving=True, trust_threshold=0.0),
                sla=SLARequirement(min_confidence=0.0, strict_enforcement=False),
            )
        )

    assert pending_module._plasticity_pending is True
    assert all(
        torch.equal(value, state_before[name])
        for name, value in model.state_dict().items()
    )


def test_serving_restores_exact_training_modes():
    model = _make_model()
    model.train()
    model.h[0].mlp.eval()
    modes_before = [module.training for module in model.modules()]
    engine = SynapticServingEngine(model)

    response = engine.serve_request(
        ServingRequest(
            "mode-safe",
            torch.ones((1, 2), dtype=torch.long),
            max_tokens=1,
            knobs=ServingKnobs(trust_threshold=0.0),
            sla=SLARequirement(min_confidence=0.0, strict_enforcement=False),
        )
    )

    assert response.status == ResponseStatus.SUCCESS
    assert [module.training for module in model.modules()] == modes_before


def test_serving_batch_confidence_uses_least_confident_row(monkeypatch):
    """One confident batch row must not mask a low-confidence sibling."""
    model = _make_model()

    def controlled_forward(tokens, train_mode=True):
        assert not train_mode
        logits = torch.zeros((*tokens.shape, model.config.vocab_size))
        logits[0, -1, 0] = 20.0
        return logits, None

    monkeypatch.setattr(model, "forward", controlled_forward)
    response = SynapticServingEngine(model).serve_request(
        ServingRequest(
            "mixed-confidence",
            torch.ones((2, 2), dtype=torch.long),
            max_tokens=1,
            knobs=ServingKnobs(trust_threshold=0.5),
            sla=SLARequirement(min_confidence=0.0, strict_enforcement=False),
        )
    )

    assert response.status == ResponseStatus.CONFIDENCE_ABSTENTION
    assert response.trust_score == pytest.approx(1.0 / model.config.vocab_size)


def test_serving_abstains_on_nonfinite_logits(monkeypatch):
    model = _make_model()

    def nonfinite_forward(tokens, train_mode=True):
        logits = torch.zeros((*tokens.shape, model.config.vocab_size))
        logits[:, -1, 0] = math.nan
        return logits, None

    monkeypatch.setattr(model, "forward", nonfinite_forward)
    response = SynapticServingEngine(model).serve_request(
        ServingRequest(
            "nonfinite",
            torch.ones((1, 2), dtype=torch.long),
            max_tokens=1,
            knobs=ServingKnobs(trust_threshold=0.0),
            sla=SLARequirement(min_confidence=0.0, strict_enforcement=False),
        )
    )

    assert response.status == ResponseStatus.CONFIDENCE_ABSTENTION
    assert response.trust_score == 0.0
    assert response.decision_info["guard_method"] == "finite-logit check"


def test_zero_token_request_does_not_claim_unmeasured_confidence():
    response = SynapticServingEngine(_make_model()).serve_request(
        ServingRequest(
            "no-generation",
            torch.ones((1, 2), dtype=torch.long),
            max_tokens=0,
        )
    )

    assert response.status == ResponseStatus.SUCCESS
    assert response.trust_score == 0.0
    assert response.decision_info["confidence_evaluated"] is False
    assert response.decision_info["confidence_floor_met"] is None


def test_heterogeneous_batch_scheduler_partitioning_and_shedding():
    """Verify scheduler separates fast and deliberative tiers and sheds load when full."""
    scheduler = HeterogeneousBatchScheduler(max_queue_depth=3)

    req_fast = ServingRequest(
        request_id="r1",
        prompt_tokens=torch.tensor([[1, 2]]),
        knobs=ServingKnobs(deliberation_steps=0),
    )
    req_delib = ServingRequest(
        request_id="r2",
        prompt_tokens=torch.tensor([[3, 4]]),
        knobs=ServingKnobs(deliberation_steps=3),
    )

    assert scheduler.enqueue(req_fast)
    assert scheduler.enqueue(req_delib)
    assert scheduler.enqueue(req_fast)
    assert not scheduler.enqueue(req_fast)  # Full queue -> shed load

    batches = scheduler.drain_batches()
    assert len(batches) == 2  # Fast tier batch and Deliberative tier batch
    assert len(batches[0]) == 2
    assert len(batches[1]) == 1

    with pytest.raises(ValueError, match="max_queue_depth"):
        HeterogeneousBatchScheduler(max_queue_depth=0)


def test_serving_engine_batch_processing_and_vitals():
    """Verify batch processing and operational telemetry vitals."""
    model = _make_model()
    engine = SynapticServingEngine(model, max_queue_depth=5)

    requests = [
        ServingRequest(
            request_id="b1",
            prompt_tokens=torch.randint(0, 32, (1, 2)),
            max_tokens=2,
            knobs=ServingKnobs(trust_threshold=0.0),
            sla=SLARequirement(min_confidence=0.0, strict_enforcement=False),
        ),
        ServingRequest(
            request_id="b2",
            prompt_tokens=torch.randint(0, 32, (1, 2)),
            max_tokens=2,
            knobs=ServingKnobs(deliberation_steps=2, trust_threshold=0.0),
            sla=SLARequirement(min_confidence=0.0, strict_enforcement=False),
        ),
    ]

    responses = engine.serve_batch(requests)
    assert len(responses) == 2
    assert responses[0].status == ResponseStatus.SUCCESS
    assert responses[1].status == ResponseStatus.SUCCESS

    vitals = engine.get_engine_vitals()
    assert vitals["total_served"] == 2
    assert vitals["queue_depth"] == 0

    engine.log_engine_vitals()


def test_serving_batch_preserves_input_order_across_tiers_and_shedding():
    engine = SynapticServingEngine(_make_model(), max_queue_depth=2)
    sla = SLARequirement(min_confidence=0.0, strict_enforcement=False)
    requests = [
        ServingRequest(
            "deliberative-first",
            torch.ones((1, 2), dtype=torch.long),
            max_tokens=1,
            knobs=ServingKnobs(deliberation_steps=1, trust_threshold=0.0),
            sla=sla,
        ),
        ServingRequest(
            "fast-second",
            torch.ones((1, 2), dtype=torch.long),
            max_tokens=1,
            knobs=ServingKnobs(trust_threshold=0.0),
            sla=sla,
        ),
        ServingRequest(
            "shed-third",
            torch.ones((1, 2), dtype=torch.long),
            max_tokens=1,
            knobs=ServingKnobs(trust_threshold=0.0),
            sla=sla,
        ),
    ]

    responses = engine.serve_batch(requests)
    assert [response.request_id for response in responses] == [
        "deliberative-first",
        "fast-second",
        "shed-third",
    ]
    assert responses[-1].status == ResponseStatus.SLA_UNACHIEVABLE

    with pytest.raises(ValueError, match="unique"):
        engine.serve_batch([requests[0], requests[0]])
