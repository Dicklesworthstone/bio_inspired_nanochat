"""Tests for the Kinetics Ablation Multi-Seed Evaluation (bead `yw9.6`).

Verifies:
1. Compute-matched evaluation across default, CMA-ES, and learned kinetics.
2. Paired statistical testing (mean deltas, bootstrap CIs, t-tests, Wilcoxon tests).
3. Absence of mode-dependent posthoc offsets or artificial penalties.
4. CLI entrypoint and JSON artifact serialization.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.e2e import kinetics_ablation_eval as kinetics_module
from scripts.e2e.kinetics_ablation_eval import (
    KineticsAblationConfig,
    _run_single_arm,
    main as kinetics_main,
    run_kinetics_ablation,
)


@pytest.mark.unit
def test_kinetics_ablation_eval_pipeline(tmp_path: Path):
    """The harness evaluates default, CMA-ES, and learned arms with paired statistics."""
    cfg = KineticsAblationConfig(
        seeds=(101, 103, 107),
        train_steps=3,
        eval_batches=2,
        batch_size=4,
        sequence_len=16,
        vocab_size=32,
        n_embd=16,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        bootstrap_samples=100,
    )
    report = run_kinetics_ablation(cfg, run_dir=tmp_path, verbose=False)

    assert set(report.arms.keys()) == {"default", "cmaes", "learned"}
    for mode in ("default", "cmaes", "learned"):
        arm = report.arms[mode]
        assert set(arm.losses.keys()) == {101, 103, 107}
        assert arm.loss_stats.mean > 0.0
        assert arm.loss_stats.ci_low <= arm.loss_stats.mean <= arm.loss_stats.ci_high

    assert "learned_vs_default" in report.comparisons
    assert "learned_vs_cmaes" in report.comparisons
    assert "cmaes_vs_default" in report.comparisons

    comp = report.comparisons["learned_vs_default"]
    assert comp.t_p_value is not None
    assert comp.delta_ci_low <= comp.mean_delta <= comp.delta_ci_high


@pytest.mark.unit
def test_arm_result_has_no_mode_dependent_posthoc_offset(monkeypatch):
    """Zero post-hoc offsets: identical model forward yields identical losses across arms."""
    class IdenticalModel(torch.nn.Module):
        def __init__(self, _config):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(1))

        def reset_sequence_state(self, **_kwargs):
            return 0

        def forward(self, tokens, **_kwargs):
            logits = self.bias.expand(*tokens.shape, 8)
            targets = _kwargs.get("targets")
            loss = None
            if targets is not None:
                loss = torch.tensor(1.2345, requires_grad=True)
            return logits, loss

    monkeypatch.setattr(kinetics_module, "GPTSynaptic", IdenticalModel)
    cfg = KineticsAblationConfig(
        seeds=(101, 103),
        train_steps=1,
        eval_batches=1,
        batch_size=2,
        sequence_len=4,
        vocab_size=8,
        n_embd=4,
        n_layer=1,
    )

    losses = {
        mode: _run_single_arm(mode, cfg, seed=101).val_loss
        for mode in ("default", "cmaes", "learned")
    }

    assert losses["default"] == pytest.approx(losses["cmaes"])
    assert losses["default"] == pytest.approx(losses["learned"])
    with pytest.raises(ValueError, match="Unknown kinetics mode"):
        _run_single_arm("invalid_mode", cfg, seed=101)


@pytest.mark.unit
def test_kinetics_config_validation():
    """Config validation rejects invalid seeds, odd sequence lengths, or non-divisible embeddings."""
    with pytest.raises(ValueError, match="at least two"):
        KineticsAblationConfig(seeds=(1,))
    with pytest.raises(ValueError, match="unique"):
        KineticsAblationConfig(seeds=(1, 1))
    with pytest.raises(ValueError, match="even"):
        KineticsAblationConfig(sequence_len=15)
    with pytest.raises(ValueError, match="divisible by 4"):
        KineticsAblationConfig(n_embd=6)


@pytest.mark.unit
def test_kinetics_ablation_cli_entrypoint(tmp_path: Path):
    """CLI entrypoint executes cleanly and writes JSON report."""
    json_path = tmp_path / "kinetics_report.json"
    ret = kinetics_main([
        "--run-dir", str(tmp_path),
        "--output-json", str(json_path),
        "--steps", "2",
        "--device", "cpu",
    ])
    assert ret == 0
    assert json_path.exists()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "verdict" in data
    assert "arms" in data
    assert "comparisons" in data
