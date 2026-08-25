"""Tests for the Self-Correcting Generation Loop (beads `re4e.1`, `re4e.1.3`)."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.nn as nn

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.causal_deliberation import ControlType
from bio_inspired_nanochat.sheaf_obstruction import (
    ObstructionAction,
    SheafDetectorDecision,
)
from bio_inspired_nanochat.self_correcting_generator import (
    CorrectionOutcome,
    SelfCorrectionEvalSample,
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

    def get_device(self) -> torch.device:
        return self.anchor.device


class _ScriptedController:
    """Plant a bad draft token, then return a clean deliberated replacement."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        control: ControlType = ControlType.DELIBERATION,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        rng: torch.Generator | None = None,
    ) -> "_ScriptedTrajectory":
        self.calls.append(
            {
                "control": control,
                "temperature": temperature,
                "top_k": top_k,
                "rng": rng,
            }
        )
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


class _FixedDetector:
    """Return a deliberate detector state without relying on representation values."""

    def __init__(self, *, available: bool, flagged: bool) -> None:
        self.available = available
        self.flagged = flagged

    def inspect(
        self,
        stalks: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> SheafDetectorDecision:
        flagged = self.available and self.flagged
        residuals = (1.0,) * int(edge_index.shape[1]) if self.available else ()
        score = 0.9 if flagged else 0.0
        return SheafDetectorDecision(
            enabled=True,
            available=self.available,
            flagged=flagged,
            score=score,
            score_after=score,
            threshold=0.4,
            calibrated_probability=None,
            requested_action=ObstructionAction.DELIBERATE,
            action_taken="deliberate" if flagged else "noop",
            output_stalks=stalks,
            edge_residual_norms=residuals,
            should_deliberate=flagged,
            fallback_reason=None if self.available else "test_unavailable",
        )


def test_rejects_models_without_real_hidden_states():
    """The detector must never run on fabricated random representations."""
    with pytest.raises(TypeError, match="get_hidden_states"):
        SelfCorrectingGenerator(nn.Linear(8, 8))


def test_config_rejects_nonpositive_repair_span():
    with pytest.raises(ValueError, match="max_repair_span"):
        SelfCorrectionConfig(max_repair_span=0).validate()


@pytest.mark.parametrize(
    "config",
    [
        SelfCorrectionConfig(max_repair_attempts=True),
        SelfCorrectionConfig(obstruction_threshold=float("nan")),
        SelfCorrectionConfig(deliberation_budget=cast(Any, 1.5)),
        SelfCorrectionConfig(abstain_token_id=-1),
        SelfCorrectionConfig(abstain_on_exhaustion=cast(Any, 1)),
    ],
)
def test_config_rejects_invalid_types_and_nonfinite_values(config):
    with pytest.raises(ValueError):
        config.validate()


def test_rejects_abstain_token_outside_model_vocabulary():
    model = _TokenStateModel()

    with pytest.raises(ValueError, match="outside model vocabulary"):
        SelfCorrectingGenerator(
            model,
            SelfCorrectionConfig(abstain_token_id=model.config.vocab_size),
        )


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


def test_reports_no_obstruction_when_score_is_below_threshold():
    """A below-threshold sequence is not overclaimed as globally verified."""
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

    assert traj.outcome == CorrectionOutcome.NO_OBSTRUCTION_DETECTED
    assert not traj.is_abstention


def test_abstain_on_exhaustion():
    """Persistent inconsistency terminates in an honest bounded abstention."""
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
            abstain_token_id=31,
        ),
    )
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    traj = generator.generate(prompt, max_new_tokens=4)

    assert traj.outcome == CorrectionOutcome.ABSTAIN
    assert traj.is_abstention
    assert traj.final_tokens == [1, 2, 3, 31]


def test_short_generation_is_reported_unchecked():
    generator = SelfCorrectingGenerator(
        _TokenStateModel(),
        SelfCorrectionConfig(enabled=True),
        deliberation_controller=_ScriptedController(),
    )

    trajectory = generator.generate(torch.tensor([1, 2]), max_new_tokens=1)

    assert trajectory.outcome is CorrectionOutcome.UNCHECKED
    assert not trajectory.is_abstention


def test_unavailable_detector_is_reported_unchecked():
    generator = SelfCorrectingGenerator(
        _TokenStateModel(),
        SelfCorrectionConfig(enabled=True),
        sheaf_detector=_FixedDetector(available=False, flagged=False),
        deliberation_controller=_ScriptedController(),
    )

    trajectory = generator.generate(torch.tensor([1, 2]), max_new_tokens=5)

    assert trajectory.outcome is CorrectionOutcome.UNCHECKED


def test_failed_repair_without_abstention_is_unresolved():
    generator = SelfCorrectingGenerator(
        _TokenStateModel(),
        SelfCorrectionConfig(
            enabled=True,
            max_repair_attempts=1,
            abstain_on_exhaustion=False,
        ),
        sheaf_detector=_FixedDetector(available=True, flagged=True),
        deliberation_controller=_ScriptedController(),
    )

    trajectory = generator.generate(torch.tensor([1, 2]), max_new_tokens=5)

    assert trajectory.outcome is CorrectionOutcome.UNRESOLVED
    assert trajectory.attempts_used == 1
    assert not trajectory.events[0].repaired_successfully


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
        assert (
            metrics["self_correction"]["outcome"]
            == CorrectionOutcome.NO_OBSTRUCTION_DETECTED.value
        )

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


def test_engine_self_correction_repairs_inside_inference_mode():
    """The low-threshold path must support the input gradients used by relaxation."""
    from bio_inspired_nanochat.engine import Engine

    class _MockTok:
        def encode_special(self, _value):
            return -5

        def get_bos_token_id(self):
            return -10

    model = GPTSynaptic(
        GPTSynapticConfig(
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_kv_head=2,
            n_embd=32,
            sequence_len=32,
        )
    ).eval()
    engine = Engine(model, _MockTok())

    output = list(
        engine.generate(
            [1, 2, 3],
            num_samples=1,
            max_tokens=4,
            temperature=0.0,
            self_correction=SelfCorrectionConfig(
                enabled=True,
                obstruction_threshold=0.0001,
                max_repair_attempts=1,
                abstain_token_id=31,
            ),
            yield_metrics=True,
        )
    )

    assert output
    assert output[0][2]["self_correction"]["outcome"] in {
        CorrectionOutcome.REPAIRED.value,
        CorrectionOutcome.ABSTAIN.value,
    }


def test_engine_forces_abstention_and_forwards_sampling_controls():
    from bio_inspired_nanochat.engine import Engine

    class _MockTok:
        def encode_special(self, _value):
            return -5

        def get_bos_token_id(self):
            return -10

    model = _TokenStateModel()
    controller = _ScriptedController()
    generator = SelfCorrectingGenerator(
        model,
        SelfCorrectionConfig(enabled=True, max_repair_attempts=1),
        sheaf_detector=_FixedDetector(available=True, flagged=True),
        deliberation_controller=controller,
    )
    engine = Engine(model, _MockTok())

    output = list(
        engine.generate(
            [1, 2],
            num_samples=1,
            max_tokens=5,
            temperature=0.25,
            top_k=3,
            seed=17,
            self_correction=generator,
            yield_metrics=True,
        )
    )

    assert len(output) == 1
    token_columns, masks, metrics = output[0]
    assert token_columns == [0]
    assert masks == [0]
    assert metrics["self_correction"]["outcome"] == CorrectionOutcome.ABSTAIN.value
    assert all(call["temperature"] == 0.25 for call in controller.calls)
    assert all(call["top_k"] == 3 for call in controller.calls)
    assert all(isinstance(call["rng"], torch.Generator) for call in controller.calls)


def test_engine_rejects_duplicate_self_correcting_samples():
    from bio_inspired_nanochat.engine import Engine

    class _MockTok:
        def encode_special(self, _value):
            return -5

        def get_bos_token_id(self):
            return -10

    engine = Engine(_TokenStateModel(), _MockTok())
    generator = SelfCorrectingGenerator(
        engine.model,
        SelfCorrectionConfig(enabled=True),
        deliberation_controller=_ScriptedController(),
    )

    with pytest.raises(ValueError, match="exactly one sample"):
        list(
            engine.generate(
                [1, 2],
                num_samples=2,
                max_tokens=2,
                self_correction=generator,
            )
        )


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

    # The oracle scores actual output tokens; labels are metadata only.
    samples = [
        SelfCorrectionEvalSample(
            prompt=torch.tensor([1, 2], dtype=torch.long),
            max_new_tokens=5,
            is_error=lambda tokens: 99 in tokens,
            expected_inconsistency=True,
            name="planted_0",
        ),
        SelfCorrectionEvalSample(
            prompt=torch.tensor([3, 4], dtype=torch.long),
            max_new_tokens=5,
            is_error=lambda tokens: 99 in tokens,
            expected_inconsistency=True,
            name="planted_1",
        ),
        SelfCorrectionEvalSample(
            prompt=torch.tensor([5, 6], dtype=torch.long),
            max_new_tokens=5,
            is_error=lambda tokens: 99 in tokens,
            expected_inconsistency=True,
            name="planted_2",
        ),
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
    assert report.coverage == 1.0
    assert report.answered_error_rate == 0.0
    assert report.avg_attempts_used >= 1.0
    assert report.avg_latency_ms > 0.0
    assert report.avg_baseline_latency_ms > 0.0
    assert report.latency_overhead_ratio is not None
    assert report.verdict == "improved"
    assert report.cumulative_single_pass_error_rate == (1.0, 1.0, 1.0)
    assert report.cumulative_self_correcting_error_rate == (0.0, 0.0, 0.0)
    assert all(result.baseline_error for result in report.sample_results)
    assert all(not result.self_correcting_failure for result in report.sample_results)
    assert events_path.exists()

    table = report.summary_table()
    assert table is not None


def test_benchmark_counts_abstention_as_primary_failure():
    from bio_inspired_nanochat.self_correcting_generator import (
        evaluate_self_correction_benchmark,
    )

    generator = SelfCorrectingGenerator(
        _TokenStateModel(),
        SelfCorrectionConfig(enabled=True, max_repair_attempts=1),
        sheaf_detector=_FixedDetector(available=True, flagged=True),
        deliberation_controller=_ScriptedController(),
    )
    sample = SelfCorrectionEvalSample(
        prompt=torch.tensor([1, 2]),
        max_new_tokens=5,
        is_error=lambda tokens: 99 in tokens,
        expected_inconsistency=True,
    )

    report = evaluate_self_correction_benchmark(generator, [sample])

    assert report.single_pass_errors == 1
    assert report.self_correcting_errors == 1
    assert report.abstention_count == 1
    assert report.coverage == 0.0
    assert report.error_reduction_pct == 0.0
    assert report.verdict == "null"
    assert not report.sample_results[0].self_correcting_output_error
    assert report.sample_results[0].self_correcting_failure


def test_benchmark_labels_do_not_predetermine_measured_errors():
    from bio_inspired_nanochat.self_correcting_generator import (
        evaluate_self_correction_benchmark,
    )

    generator = SelfCorrectingGenerator(
        _TokenStateModel(),
        SelfCorrectionConfig(enabled=True),
        deliberation_controller=_ScriptedController(),
    )
    sample = SelfCorrectionEvalSample(
        prompt=torch.tensor([1, 2]),
        max_new_tokens=5,
        is_error=lambda _tokens: False,
        expected_inconsistency=True,
    )

    report = evaluate_self_correction_benchmark(generator, [sample])

    assert report.inconsistent_samples == 1
    assert report.single_pass_errors == 0
    assert report.self_correcting_errors == 0
    assert report.error_reduction_pct is None
    assert report.verdict == "null"


def test_benchmark_reports_worse_without_fabricating_reduction_from_zero_baseline():
    from bio_inspired_nanochat.self_correcting_generator import (
        evaluate_self_correction_benchmark,
    )

    generator = SelfCorrectingGenerator(
        _TokenStateModel(),
        SelfCorrectionConfig(enabled=True, max_repair_attempts=1),
        sheaf_detector=_FixedDetector(available=True, flagged=True),
        deliberation_controller=_ScriptedController(),
    )
    sample = SelfCorrectionEvalSample(
        prompt=torch.tensor([1, 2]),
        max_new_tokens=5,
        is_error=lambda _tokens: False,
        expected_inconsistency=False,
    )

    report = evaluate_self_correction_benchmark(generator, [sample])

    assert report.single_pass_errors == 0
    assert report.self_correcting_errors == 1
    assert report.error_reduction_pct is None
    assert report.verdict == "worse"


def test_benchmark_pairs_identical_sampling_rng_for_both_arms():
    from bio_inspired_nanochat.self_correcting_generator import (
        evaluate_self_correction_benchmark,
    )

    class _StochasticController(_ScriptedController):
        def generate(
            self,
            prompt: torch.Tensor,
            max_new_tokens: int,
            control: ControlType = ControlType.DELIBERATION,
            *,
            temperature: float | None = None,
            top_k: int | None = None,
            rng: torch.Generator | None = None,
        ) -> _ScriptedTrajectory:
            if rng is None:
                raise AssertionError("benchmark must supply an explicit generator")
            sampled = torch.randint(0, 128, (max_new_tokens,), generator=rng).tolist()
            return _ScriptedTrajectory(prompt.reshape(-1).tolist() + sampled)

    generator = SelfCorrectingGenerator(
        _TokenStateModel(),
        SelfCorrectionConfig(enabled=True),
        sheaf_detector=_FixedDetector(available=True, flagged=False),
        deliberation_controller=_StochasticController(),
    )
    sample = SelfCorrectionEvalSample(
        prompt=torch.tensor([1, 2]),
        max_new_tokens=12,
        is_error=lambda _tokens: False,
        expected_inconsistency=False,
    )

    report = evaluate_self_correction_benchmark(generator, [sample], seed=1729)

    result = report.sample_results[0]
    assert result.baseline_tokens == result.self_correcting_tokens


def test_benchmark_rejects_non_boolean_oracle_result():
    from bio_inspired_nanochat.self_correcting_generator import (
        evaluate_self_correction_benchmark,
    )

    generator = SelfCorrectingGenerator(
        _TokenStateModel(),
        SelfCorrectionConfig(enabled=True),
        deliberation_controller=_ScriptedController(),
    )
    sample = SelfCorrectionEvalSample(
        prompt=torch.tensor([1, 2]),
        max_new_tokens=5,
        is_error=cast(Any, lambda _tokens: 0),
        expected_inconsistency=True,
    )

    with pytest.raises(TypeError, match="must return bool"):
        evaluate_self_correction_benchmark(generator, [sample])


@pytest.mark.parametrize(
    "prompt, message",
    [
        (torch.tensor([1.5]), "integer dtype"),
        (torch.tensor([128]), "outside model vocabulary"),
    ],
)
def test_benchmark_rejects_invalid_prompt_before_running_either_arm(prompt, message):
    from bio_inspired_nanochat.self_correcting_generator import (
        evaluate_self_correction_benchmark,
    )

    controller = _ScriptedController()
    generator = SelfCorrectingGenerator(
        _TokenStateModel(),
        SelfCorrectionConfig(enabled=True),
        deliberation_controller=controller,
    )
    sample = SelfCorrectionEvalSample(
        prompt=prompt,
        max_new_tokens=5,
        is_error=lambda _tokens: False,
        expected_inconsistency=False,
    )

    with pytest.raises(ValueError, match=message):
        evaluate_self_correction_benchmark(generator, [sample])

    assert controller.calls == []


def test_benchmark_restores_model_mode_state_and_global_rng():
    from bio_inspired_nanochat.self_correcting_generator import (
        evaluate_self_correction_benchmark,
    )

    model = _TokenStateModel().train()
    generator = SelfCorrectingGenerator(
        model,
        SelfCorrectionConfig(enabled=True),
        deliberation_controller=_ScriptedController(),
    )
    sample = SelfCorrectionEvalSample(
        prompt=torch.tensor([1, 2]),
        max_new_tokens=5,
        is_error=lambda tokens: 99 in tokens,
        expected_inconsistency=True,
    )
    model_state = {name: value.clone() for name, value in model.state_dict().items()}
    rng_state = torch.get_rng_state().clone()

    evaluate_self_correction_benchmark(generator, [sample], seed=123)

    assert model.training
    assert torch.equal(torch.get_rng_state(), rng_state)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, model_state[name])
