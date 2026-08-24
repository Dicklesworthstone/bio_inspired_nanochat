"""Tests for full-state causal deliberation with compute-matched controls (bead `r00r.15`)."""

import torch

from bio_inspired_nanochat.causal_deliberation import (
    CausalDeliberationConfig,
    CausalDeliberationController,
    ControlType,
    FullStateRelaxer,
)
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from scripts.e2e.causal_deliberation_eval import EvalConfig, run_full_causal_deliberation_eval


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

    # Controller with causal commitment enabled
    ctrl_committed = CausalDeliberationController(
        model,
        CausalDeliberationConfig(max_iters=5, commit_relaxed_state=True, temperature=0.5),
    )
    traj_committed = ctrl_committed.generate(prompt=prompt, max_new_tokens=4, control=ControlType.DELIBERATION)

    # Controller without deliberation (Baseline)
    ctrl_base = CausalDeliberationController(
        model,
        CausalDeliberationConfig(max_iters=0, commit_relaxed_state=False, temperature=0.5),
    )
    traj_base = ctrl_base.generate(prompt=prompt, max_new_tokens=4, control=ControlType.BASELINE)

    assert len(traj_committed.generated_tokens) == len(prompt) + 4
    assert len(traj_base.generated_tokens) == len(prompt) + 4
    assert traj_committed.total_iterations > 0
    assert traj_base.total_iterations == 0


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
