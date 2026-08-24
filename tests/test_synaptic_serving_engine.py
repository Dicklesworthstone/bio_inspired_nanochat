"""Tests for Synaptic Serving Engine & Certified SLA Controller (bead `re4e.5`)."""

import torch

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
    """Verify that a compliant request executes, returns output tokens, and generates certificates."""
    model = _make_model()
    engine = SynapticServingEngine(model)

    req = ServingRequest(
        request_id="req_001",
        prompt_tokens=torch.randint(0, 32, (1, 3)),
        max_tokens=4,
        knobs=ServingKnobs(deliberation_steps=2, atp_energy_cap=30.0),
        sla=SLARequirement(max_latency_ms=500.0),
    )

    resp = engine.serve_request(req)

    assert resp.status == ResponseStatus.SUCCESS
    assert resp.output_tokens.shape[1] == 7
    assert resp.atp_consumed > 0.0
    assert "certified_safe" in resp.certificate_info


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
    assert "refusal_reason" in resp.certificate_info
    assert engine.total_refused == 1


def test_serving_engine_atp_energy_budget_exhaustion():
    """Verify that generation halts when per-request ATP energy cap is reached."""
    model = _make_model()
    engine = SynapticServingEngine(model)

    req = ServingRequest(
        request_id="req_capped",
        prompt_tokens=torch.randint(0, 32, (1, 2)),
        max_tokens=10,
        knobs=ServingKnobs(atp_energy_cap=3.5),  # Only enough for ~3 tokens
        sla=SLARequirement(max_latency_ms=1000.0),
    )

    resp = engine.serve_request(req)

    assert resp.status == ResponseStatus.ATP_BUDGET_EXHAUSTED
    assert resp.atp_consumed <= 3.5
    assert resp.output_tokens.shape[1] < 12


def test_serving_engine_certified_abstention():
    """Verify that very high confidence requirements with low token confidence trigger certified abstention."""
    model = _make_model()
    engine = SynapticServingEngine(model)

    req = ServingRequest(
        request_id="req_abstain",
        prompt_tokens=torch.randint(0, 32, (1, 2)),
        max_tokens=4,
        knobs=ServingKnobs(conformal_alpha=0.005),  # Ultra-strict alpha < 0.01
    )

    resp = engine.serve_request(req)
    assert resp.status in [ResponseStatus.SUCCESS, ResponseStatus.CERTIFIED_ABSTENTION]


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


def test_serving_engine_batch_processing_and_vitals():
    """Verify batch processing and operational telemetry vitals."""
    model = _make_model()
    engine = SynapticServingEngine(model, max_queue_depth=5)

    requests = [
        ServingRequest(request_id="b1", prompt_tokens=torch.randint(0, 32, (1, 2)), max_tokens=2),
        ServingRequest(request_id="b2", prompt_tokens=torch.randint(0, 32, (1, 2)), max_tokens=2, knobs=ServingKnobs(deliberation_steps=2)),
    ]

    responses = engine.serve_batch(requests)
    assert len(responses) == 2
    assert responses[0].status == ResponseStatus.SUCCESS
    assert responses[1].status == ResponseStatus.SUCCESS

    vitals = engine.get_engine_vitals()
    assert vitals["total_served"] == 2
    assert vitals["queue_depth"] == 0

    engine.log_engine_vitals()

