"""Tests for the Factorial Mechanism Redundancy & Saturation Evaluation (bead `74f.5`).

Verifies:
1. Multi-seed factorial evaluation across individual and paired biological mechanisms.
2. Interaction / Synergy index calculation.
3. Zero posthoc penalty or fabricated offsets across arms.
4. CLI entrypoint and JSON artifact serialization.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.e2e import mechanism_redundancy_eval as redundancy_module
from scripts.e2e.mechanism_redundancy_eval import (
    RedundancyConfig,
    _run_single_seed_arm,
    main as redundancy_main,
    run_mechanism_redundancy_evaluation,
)


@pytest.mark.unit
def test_mechanism_redundancy_eval_pipeline(tmp_path: Path):
    """The factorial harness evaluates baseline, single, pairwise, and combined arms."""
    cfg = RedundancyConfig(
        seeds=(201, 203),
        train_steps=1,
        eval_batches=1,
        batch_size=2,
        sequence_len=16,
        vocab_size=32,
        n_embd=16,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        bootstrap_samples=100,
    )
    report = run_mechanism_redundancy_evaluation(cfg, run_dir=tmp_path, verbose=False)

    assert "vanilla_baseline" in report.arms
    assert "all_bio_active" in report.arms
    assert "presyn_only" in report.arms
    assert "hebbian_only" in report.arms
    assert "presyn_hebbian" in report.arms

    for name, arm in report.arms.items():
        assert set(arm.losses.keys()) == {201, 203}
        assert arm.loss_stats.mean > 0.0

    assert len(report.synergies) == 6
    for s in report.synergies:
        assert isinstance(s.synergy_index, float)
        assert s.interpretation in {
            "Independent (Additive)",
            "Diminishing Returns (Redundant / Sub-additive)",
            "Synergistic (Super-additive)",
        }


@pytest.mark.unit
def test_factorial_arms_have_no_posthoc_penalties(monkeypatch):
    """Identical model weights produce identical validation losses across all arms."""
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
                loss = torch.tensor(1.4567, requires_grad=True)
            return logits, loss

    monkeypatch.setattr(redundancy_module, "GPTSynaptic", IdenticalModel)
    cfg = RedundancyConfig(
        seeds=(201, 203),
        train_steps=1,
        eval_batches=1,
        batch_size=2,
        sequence_len=4,
        vocab_size=8,
        n_embd=4,
        n_layer=1,
    )

    flags_vanilla = {"presyn": False, "postsyn": False, "glial": False, "septin": False}
    flags_all = {"presyn": True, "postsyn": True, "glial": True, "septin": True}

    res_vanilla = _run_single_seed_arm("vanilla", flags_vanilla, cfg, seed=201)
    res_all = _run_single_seed_arm("all", flags_all, cfg, seed=201)

    assert res_vanilla.val_loss == pytest.approx(res_all.val_loss)


@pytest.mark.unit
def test_redundancy_config_validation():
    """Config validation rejects invalid seeds, odd sequence lengths, or non-divisible embeddings."""
    with pytest.raises(ValueError, match="at least two"):
        RedundancyConfig(seeds=(1,))
    with pytest.raises(ValueError, match="unique"):
        RedundancyConfig(seeds=(1, 1))
    with pytest.raises(ValueError, match="even"):
        RedundancyConfig(sequence_len=15)
    with pytest.raises(ValueError, match="divisible by 4"):
        RedundancyConfig(n_embd=6)


@pytest.mark.unit
def test_mechanism_redundancy_cli_entrypoint(tmp_path: Path):
    """CLI entrypoint runs cleanly and writes structured JSON."""
    json_path = tmp_path / "redundancy_report.json"
    ret = redundancy_main([
        "--run-dir", str(tmp_path),
        "--output-json", str(json_path),
        "--seeds", "201", "203",
        "--steps", "1",
        "--eval-batches", "1",
        "--device", "cpu",
    ])
    assert ret == 0
    assert json_path.exists()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "arms" in data
    assert "synergies" in data
    assert "correlation_matrix" in data
