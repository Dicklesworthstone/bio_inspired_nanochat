"""Tests for the Self-Correcting Generation Loop (beads `re4e.1`, `re4e.1.3`)."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.causal_deliberation import ControlType
from bio_inspired_nanochat.self_correcting_generator import (
    CorrectionOutcome,
    SelfCorrectingGenerator,
    SelfCorrectionConfig,
    SelfCorrectionEvent,
    SelfCorrectingTrajectory,
)


class _TokenStateModel(nn.Module):
    """Tiny real representation API with one planted token-level inconsistency."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(n_embd=2, vocab_size=128)
        self.lm_head = nn.Linear(2, 128, bias=False)

    def get_hidden_states(self, tokens: torch.Tensor) -> torch.Tensor:
        sign = torch.where(tokens == 99, -1.0, 1.0).to(dtype=torch.float32)
        return torch.stack((sign, torch.zeros_like(sign)), dim=-1)

    def hidden_to_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)


class _ScriptedController:
    """Plant a bad draft token, then return a clean deliberated replacement."""

    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        control: ControlType = ControlType.DELIBERATION,
    ) -> "_ScriptedTrajectory":
        prompt_tokens = prompt.reshape(-1).tolist()
        generated = (
            [10, 10, 99, 10, 10]
            if control is ControlType.BASELINE
            else [10] * max_new_tokens
        )
        return _ScriptedTrajectory(prompt_tokens + generated[:max_new_tokens])


@dataclass
class _ScriptedTrajectory:
    generated_tokens: list[int]


def test_rejects_models_without_real_hidden_states():
    """The detector must never run on fabricated random representations."""
    with pytest.raises(TypeError, match="get_hidden_states"):
        SelfCorrectingGenerator(nn.Linear(8, 8))


def test_config_rejects_nonpositive_repair_span():
    with pytest.raises(ValueError, match="max_repair_span"):
        SelfCorrectionConfig(max_repair_span=0).validate()


def test_self_correction_is_default_off_and_passes_through():
    """The production-safe default reduces exactly to baseline generation."""
    cfg = GPTSynapticConfig(vocab_size=32, n_layer=1, n_head=2, n_kv_head=2, n_embd=32, sequence_len=32)
    model = GPTSynaptic(cfg)
    model.eval()

    generator = SelfCorrectingGenerator(model)
    prompt = torch.tensor([1, 2, 3], dtype=torch.long)
    traj = generator.generate(prompt, max_new_tokens=4)

    assert traj.outcome == CorrectionOutcome.PASSTHROUGH
    assert traj.attempts_used == 0
    assert not traj.is_abstention


def test_verified_consistent_when_no_obstruction():
    """Verify that clean sequences return VERIFIED_CONSISTENT."""
    cfg = GPTSynapticConfig(vocab_size=32, n_layer=1, n_head=2, n_kv_head=2, n_embd=32, sequence_len=32)
    model = GPTSynaptic(cfg)
    model.eval()

    # Very high threshold -> never flags obstruction
    generator = SelfCorrectingGenerator(
        model,
        SelfCorrectionConfig(enabled=True, obstruction_threshold=1.0),
    )
    prompt = torch.tensor([1, 2, 3], dtype=torch.long)
    traj = generator.generate(prompt, max_new_tokens=4)

    assert traj.outcome == CorrectionOutcome.VERIFIED_CONSISTENT
    assert not traj.is_abstention


def test_certified_abstain_on_exhaustion():
    """Verify that persistent inconsistency terminates in CERTIFIED_ABSTAIN."""
    cfg = GPTSynapticConfig(vocab_size=32, n_layer=1, n_head=2, n_kv_head=2, n_embd=32, sequence_len=32)
    model = GPTSynaptic(cfg)
    model.eval()

    # Impossibly low threshold -> always flags obstruction
    generator = SelfCorrectingGenerator(
        model,
        SelfCorrectionConfig(
            enabled=True,
            obstruction_threshold=0.0001,
            max_repair_attempts=2,
            abstain_on_exhaustion=True,
            abstain_token_id=99,
        ),
    )
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    traj = generator.generate(prompt, max_new_tokens=4)

    assert traj.outcome == CorrectionOutcome.CERTIFIED_ABSTAIN
    assert traj.is_abstention
    assert traj.final_tokens == [1, 2, 3, 99]


@pytest.mark.parametrize(
    "prompt",
    [
        torch.tensor([1, 2], dtype=torch.long),
        torch.tensor([[1, 2]], dtype=torch.long),
    ],
)
def test_local_residuals_drive_middle_span_repair(prompt):
    """The repair target follows the planted interior obstruction, not position zero."""
    model = _TokenStateModel()
    generator = SelfCorrectingGenerator(
        model,
        SelfCorrectionConfig(
            enabled=True,
            max_repair_attempts=1,
            obstruction_threshold=0.05,
            max_repair_span=3,
        ),
        deliberation_controller=_ScriptedController(),
    )

    trajectory = generator.generate(
        prompt,
        max_new_tokens=5,
    )

    assert trajectory.outcome is CorrectionOutcome.REPAIRED
    assert trajectory.final_tokens == [1, 2, 10, 10, 10, 10, 10]
    assert trajectory.attempts_used == 1
    event = trajectory.events[0]
    assert (event.span_start, event.span_end) == (3, 6)
    assert event.localization_peak == 4
    assert event.corrupted_tokens == [10, 99, 10]
    assert event.repaired_tokens == [10, 10, 10]
    assert [
        index for index, residual in enumerate(event.edge_residual_norms) if residual > 0.0
    ] == [1, 2]
    assert event.repaired_successfully


def test_rich_table_lineage_logging():
    """Verify that logging history functions without exceptions."""
    cfg = GPTSynapticConfig(vocab_size=32, n_layer=1, n_head=2, n_kv_head=2, n_embd=32, sequence_len=32)
    model = GPTSynaptic(cfg)
    generator = SelfCorrectingGenerator(model)

    event = SelfCorrectionEvent(
        attempt_idx=1,
        span_start=3,
        span_end=5,
        corrupted_tokens=[10, 11],
        repaired_tokens=[12, 13],
        localization_peak=4,
        edge_residual_norms=(0.1, 0.7, 0.2),
        initial_obstruction=0.65,
        repaired_obstruction=0.20,
        repaired_successfully=True,
        wall_time_ms=12.5,
    )
    traj = SelfCorrectingTrajectory(
        final_tokens=[1, 2, 3, 12, 13, 14],
        outcome=CorrectionOutcome.REPAIRED,
        attempts_used=1,
        events=[event],
        total_wall_time_ms=25.0,
        is_abstention=False,
    )
    generator.log_trajectory(traj)


def test_engine_generate_with_self_correction_disabled_and_enabled():
    """Verify that Engine.generate integrates self-correction behind default-off toggle."""
    from bio_inspired_nanochat.engine import Engine

    class _MockTok:
        _special = {
            "<|python_start|>": -1, "<|python_end|>": -2,
            "<|output_start|>": -3, "<|output_end|>": -4, "<|assistant_end|>": -5,
        }
        def encode_special(self, s):
            return self._special[s]
        def get_bos_token_id(self):
            return -10
        def decode(self, toks):
            return " ".join(str(t) for t in toks)
        def encode(self, s):
            return [1, 2]

    cfg = GPTSynapticConfig(vocab_size=32, n_layer=1, n_head=2, n_kv_head=2, n_embd=32, sequence_len=32)
    model = GPTSynaptic(cfg)
    model.eval()

    engine = Engine(model, _MockTok())
    prompt = [1, 2, 3]

    # Baseline passthrough when self_correction is None or disabled
    out_default = list(engine.generate(prompt, num_samples=1, max_tokens=4, self_correction=None))
    assert len(out_default) == 4
    assert all(len(cols) == 1 for cols, _ in out_default)

    out_disabled = list(engine.generate(
        prompt,
        num_samples=1,
        max_tokens=4,
        self_correction=SelfCorrectionConfig(enabled=False),
    ))
    assert len(out_disabled) == 4

    # Enabled self-correction
    out_enabled = list(engine.generate(
        prompt,
        num_samples=1,
        max_tokens=4,
        self_correction=SelfCorrectionConfig(enabled=True, obstruction_threshold=1.0),
        yield_metrics=True,
    ))
    assert len(out_enabled) == 4
    for cols, masks, metrics in out_enabled:
        assert len(cols) == 1
        assert len(masks) == 1
        assert "self_correction" in metrics
        assert metrics["self_correction"]["outcome"] == CorrectionOutcome.VERIFIED_CONSISTENT.value

    # Non-streaming generate_batch integration
    batch_results, batch_masks = engine.generate_batch(
        prompt,
        num_samples=1,
        max_tokens=4,
        self_correction=SelfCorrectionConfig(enabled=True, obstruction_threshold=1.0),
    )
    assert len(batch_results) == 1
    assert len(batch_results[0]) == len(prompt) + 4
    assert len(batch_masks[0]) == len(prompt) + 4


def test_trajectory_append_jsonl_trace(tmp_path):
    """Verify that trajectory events dump valid structured JSONL records."""
    import json

    jsonl_file = tmp_path / "events.jsonl"
    event = SelfCorrectionEvent(
        attempt_idx=1,
        span_start=3,
        span_end=5,
        corrupted_tokens=[10, 99],
        repaired_tokens=[10, 10],
        localization_peak=4,
        edge_residual_norms=(0.1, 0.8),
        initial_obstruction=0.75,
        repaired_obstruction=0.15,
        repaired_successfully=True,
        wall_time_ms=10.5,
    )
    traj = SelfCorrectingTrajectory(
        final_tokens=[1, 2, 3, 10, 10],
        outcome=CorrectionOutcome.REPAIRED,
        attempts_used=1,
        events=[event],
        total_wall_time_ms=22.0,
        is_abstention=False,
    )
    traj.append_jsonl(jsonl_file)
    assert jsonl_file.exists()

    lines = jsonl_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["outcome"] == "REPAIRED"
    assert record["attempts_used"] == 1
    assert len(record["events"]) == 1
    assert record["events"][0]["localization_peak"] == 4


def test_evaluate_self_correction_benchmark_on_labeled_set(tmp_path):
    """Verify that the benchmark measures measurable error reduction vs single-pass."""
    from bio_inspired_nanochat.self_correcting_generator import (
        evaluate_self_correction_benchmark,
    )

    model = _TokenStateModel()
    ctrl = _ScriptedController()
    generator = SelfCorrectingGenerator(
        model,
        SelfCorrectionConfig(
            enabled=True,
            max_repair_attempts=2,
            obstruction_threshold=0.40,
            max_repair_span=3,
        ),
        deliberation_controller=ctrl,
    )

    # Labeled dataset: clean prompts vs inconsistency-planted prompts
    samples = [
        (torch.tensor([1, 2], dtype=torch.long), 5, True),   # triggers planted 99 -> repaired
        (torch.tensor([3, 4], dtype=torch.long), 5, True),   # triggers planted 99 -> repaired
        (torch.tensor([5, 6], dtype=torch.long), 5, True),   # triggers planted 99 -> repaired
    ]

    events_path = tmp_path / "events.jsonl"
    report = evaluate_self_correction_benchmark(
        generator,
        samples,
        events_jsonl_path=events_path,
    )

    assert report.total_samples == 3
    assert report.inconsistent_samples == 3
    assert report.single_pass_errors == 3
    assert report.self_correcting_errors == 0
    assert report.single_pass_error_rate == 1.0
    assert report.self_correcting_error_rate == 0.0
    assert report.error_reduction_pct == 100.0
    assert report.repaired_count == 3
    assert report.abstention_count == 0
    assert report.avg_attempts_used >= 1.0
    assert report.avg_latency_ms > 0.0
    assert events_path.exists()

    table = report.summary_table()
    assert table is not None


