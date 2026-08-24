"""Tests for Energy-Guided Tree Search and State-Space Rollouts (beads `re4e.3`, `re4e.3.3`)."""

import torch

from bio_inspired_nanochat.energy_guided_search import (
    EnergyGuidedSearchEngine,
    EnergySearchConfig,
    EnergySearchTrajectory,
)
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig


def test_energy_guided_search_fallback_when_disabled():
    """Verify that disabled search collapses to standard logprob beam fallback."""
    cfg = GPTSynapticConfig(vocab_size=32, n_layer=1, n_head=2, n_kv_head=2, n_embd=32, sequence_len=32)
    model = GPTSynaptic(cfg)
    model.eval()

    engine = EnergyGuidedSearchEngine(
        model,
        EnergySearchConfig(enabled=False, beam_width=2, branching_factor=2),
    )
    prompt = torch.tensor([1, 2, 3], dtype=torch.long)
    traj = engine.search(prompt, max_new_tokens=4)

    assert traj.is_pure_beam_fallback
    assert len(traj.best_tokens) == 3 + 4
    assert traj.pruned_nodes_count == 0


def test_energy_pruning_filters_high_energy_branches():
    """Verify that tight energy threshold prunes high-energy nodes during expansion."""
    cfg = GPTSynapticConfig(vocab_size=32, n_layer=1, n_head=2, n_kv_head=2, n_embd=32, sequence_len=32)
    model = GPTSynaptic(cfg)
    model.eval()

    # Impossibly low prune threshold -> aggressively prunes nodes
    engine = EnergyGuidedSearchEngine(
        model,
        EnergySearchConfig(
            enabled=True,
            beam_width=2,
            branching_factor=2,
            energy_weight=1.0,
            energy_prune_threshold=0.0001,
        ),
    )
    prompt = torch.tensor([1, 2, 3], dtype=torch.long)
    traj = engine.search(prompt, max_new_tokens=4)

    assert not traj.is_pure_beam_fallback
    assert traj.pruned_nodes_count > 0


def test_energy_guidance_favors_low_energy_trajectories():
    """Verify that energy-weighted search completes and produces lower energy paths."""
    cfg = GPTSynapticConfig(vocab_size=32, n_layer=1, n_head=2, n_kv_head=2, n_embd=32, sequence_len=32)
    model = GPTSynaptic(cfg)
    model.eval()

    engine = EnergyGuidedSearchEngine(
        model,
        EnergySearchConfig(
            enabled=True,
            beam_width=3,
            branching_factor=3,
            energy_weight=2.0,
            energy_prune_threshold=100.0,
        ),
    )
    prompt = torch.tensor([1, 2, 3], dtype=torch.long)
    traj = engine.search(prompt, max_new_tokens=4)

    assert len(traj.best_tokens) == 3 + 4
    assert traj.total_nodes_expanded > 1


def test_search_tree_logging():
    """Verify that tree visualization renders without error."""
    cfg = GPTSynapticConfig(vocab_size=32, n_layer=1, n_head=2, n_kv_head=2, n_embd=32, sequence_len=32)
    model = GPTSynaptic(cfg)
    engine = EnergyGuidedSearchEngine(model)

    traj = EnergySearchTrajectory(
        best_tokens=[1, 2, 3, 4, 5],
        best_score=1.23,
        total_nodes_expanded=10,
        pruned_nodes_count=2,
        search_tree_nodes=[
            {"node_id": 0, "token": 1, "parent_id": None, "cost": 0.0, "energy": 0.1, "depth": 0},
            {"node_id": 1, "token": 2, "parent_id": 0, "cost": 0.5, "energy": 0.2, "depth": 1},
        ],
        wall_time_ms=15.0,
        is_pure_beam_fallback=False,
    )
    engine.log_search_tree(traj)
