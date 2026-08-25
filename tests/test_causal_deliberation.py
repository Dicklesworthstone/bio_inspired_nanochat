"""Tests for full-state causal deliberation with compute-matched controls (bead `r00r.15`)."""

import pytest
import torch
import torch.nn as nn

from bio_inspired_nanochat.causal_deliberation import (
    CausalDeliberationConfig,
    CausalDeliberationController,
    ControlType,
    FullStateRelaxer,
)
from bio_inspired_nanochat.gpt import GPT, GPTConfig
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from scripts.e2e.causal_deliberation_eval import EvalConfig, run_full_causal_deliberation_eval


class TrackingGPTSynaptic(GPTSynaptic):
    """Real synaptic model that records representation reads without replacing them."""

    def __init__(self, config: GPTSynapticConfig):
        super().__init__(config)
        self.hidden_input_lengths: list[int] = []

    def get_hidden_states(self, idx, kv_cache=None, max_layers=None):
        self.hidden_input_lengths.append(idx.shape[1])
        return super().get_hidden_states(idx, kv_cache=kv_cache, max_layers=max_layers)


def test_model_hidden_state_api_is_the_real_language_head_input():
    """Both model trunks expose exactly the states consumed by their bounded heads."""
    torch.manual_seed(7)
    tokens = torch.tensor([[1, 3, 5, 7]], dtype=torch.long)
    models = (
        GPT(GPTConfig(vocab_size=32, n_layer=1, n_head=2, n_kv_head=2, n_embd=32, sequence_len=16)),
        GPTSynaptic(
            GPTSynapticConfig(
                vocab_size=32,
                n_layer=1,
                n_head=2,
                n_kv_head=2,
                n_embd=32,
                sequence_len=16,
            )
        ),
    )

    for model in models:
        model.eval()
        with torch.no_grad():
            hidden = model.get_hidden_states(tokens)
            repeated_hidden = model.get_hidden_states(tokens)
            direct_logits = model.hidden_to_logits(hidden)
            output = model(tokens, train_mode=False) if isinstance(model, GPTSynaptic) else model(tokens)
            forward_logits = output[0] if isinstance(output, tuple) else output

        assert hidden.shape == (1, 4, 32)
        assert torch.isfinite(hidden).all()
        assert torch.allclose(direct_logits, forward_logits)
        assert torch.allclose(hidden, repeated_hidden)


def test_baseline_generation_advances_real_transformer_context():
    """Every non-final token is fed back through the model instead of a noise proxy."""
    cfg = GPTSynapticConfig(
        vocab_size=32, n_layer=1, n_head=2, n_kv_head=2, n_embd=32, sequence_len=16
    )
    model = TrackingGPTSynaptic(cfg).eval()
    controller = CausalDeliberationController(
        model,
        CausalDeliberationConfig(max_iters=0, commit_relaxed_state=False),
    )

    trajectory = controller.generate(
        prompt=torch.tensor([1, 2, 3]),
        max_new_tokens=3,
        control=ControlType.BASELINE,
    )

    assert len(trajectory.generated_tokens) == 6
    assert model.hidden_input_lengths == [3, 4, 5]


def test_deliberation_rejects_models_without_representation_contract():
    """Unsupported models fail explicitly instead of silently fabricating state."""
    with pytest.raises(TypeError, match="get_hidden_states"):
        CausalDeliberationController(nn.Linear(8, 8), CausalDeliberationConfig())


def test_full_state_relaxer_energy_monotone_descent():
    """Verify that relaxation steps monotonically decrease or preserve the free energy."""
    torch.manual_seed(42)
    d_model = 32
    relaxer = FullStateRelaxer(d_model=d_model, step_size=0.05)

    h0 = torch.randn(1, d_model)
    fw0 = torch.zeros(d_model, d_model)

    e0 = float(relaxer.energy(h0).item())
    h_cur, fw_cur = h0, fw0
    e_prev = e0

    for _ in range(5):
        h_cur, e_new, fw_cur = relaxer.step(h_cur, fw_cur)
        e_val = float(e_new.item())
        assert e_val <= e_prev + 1e-6, f"Energy did not decrease: {e_val} vs {e_prev}"
        e_prev = e_val

    assert e_prev < e0, "Energy should strictly decrease after 5 relaxation steps"


def test_causal_commitment_alters_subsequent_logits():
    """Verify that committing the relaxed state causally changes subsequent logits."""
    torch.manual_seed(42)
    cfg = GPTSynapticConfig(
        vocab_size=32, n_layer=1, n_head=2, n_kv_head=2, n_embd=32, sequence_len=32
    )
    model = GPTSynaptic(cfg)
    model.eval()

    prompt = torch.tensor([1, 5, 9, 12], dtype=torch.long)

    # Compare identical deliberation loops that differ only in causal commitment.
    ctrl_committed = CausalDeliberationController(
        model,
        CausalDeliberationConfig(max_iters=5, commit_relaxed_state=True, temperature=0.5),
    )
    ctrl_discarded = CausalDeliberationController(
        model,
        CausalDeliberationConfig(max_iters=5, commit_relaxed_state=False, temperature=0.5),
    )
    ctrl_discarded.relaxer.load_state_dict(ctrl_committed.relaxer.state_dict())

    torch.manual_seed(123)
    traj_committed = ctrl_committed.generate(prompt=prompt, max_new_tokens=4, control=ControlType.DELIBERATION)
    torch.manual_seed(123)
    traj_discarded = ctrl_discarded.generate(prompt=prompt, max_new_tokens=4, control=ControlType.DELIBERATION)

    assert len(traj_committed.generated_tokens) == len(prompt) + 4
    assert len(traj_discarded.generated_tokens) == len(prompt) + 4
    assert traj_committed.total_iterations > 0
    assert traj_discarded.total_iterations > 0
    assert torch.allclose(
        traj_committed.step_results[0].logits,
        traj_discarded.step_results[0].logits,
    )
    assert not torch.allclose(
        traj_committed.step_results[1].logits,
        traj_discarded.step_results[1].logits,
    )


def test_compute_matched_controls_flops_and_support():
    """Verify compute accounting and support matching across control types."""
    torch.manual_seed(42)
    cfg = GPTSynapticConfig(
        vocab_size=32, n_layer=1, n_head=2, n_kv_head=2, n_embd=32, sequence_len=32
    )
    model = GPTSynaptic(cfg)
    model.eval()

    prompt = torch.tensor([2, 4, 6], dtype=torch.long)
    delib_cfg = CausalDeliberationConfig(max_iters=4, placebo_ops_per_iter=500)
    controller = CausalDeliberationController(model, delib_cfg)

    # Deliberation
    traj_delib = controller.generate(prompt=prompt, max_new_tokens=2, control=ControlType.DELIBERATION)
    # Placebo
    traj_placebo = controller.generate(prompt=prompt, max_new_tokens=2, control=ControlType.PLACEBO)
    # Baseline
    traj_base = controller.generate(prompt=prompt, max_new_tokens=2, control=ControlType.BASELINE)

    assert traj_delib.total_flops > traj_base.total_flops
    assert traj_placebo.total_flops > traj_base.total_flops


def test_eval_suite_end_to_end():
    """Verify that the preregistered evaluation suite runs end-to-end and outputs valid report statistics."""
    config = EvalConfig(
        seeds=(101, 102),
        budgets=(1, 4),
        vocab_size=32,
        eval_samples_per_task=2,
        n_embd=32,
        n_layer=1,
        n_head=2,
    )
    report = run_full_causal_deliberation_eval(config)

    assert report.verdict in ("improved", "null", "worse")
    assert len(report.task_results) > 0
    assert "copy" in [r.task_name for r in report.task_results]
    assert "associative_recall" in [r.task_name for r in report.task_results]
    assert "variable_binding" in [r.task_name for r in report.task_results]
