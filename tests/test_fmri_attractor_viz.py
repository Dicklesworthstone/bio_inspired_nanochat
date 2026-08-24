"""Tests for fMRI Free-Energy Landscape & Attractor Dynamics Visualizer (bead `r00r.10`)."""

import torch

from bio_inspired_nanochat.fmri_attractor_viz import (
    FreeEnergyLandscapeProjector,
)


def test_free_energy_surface_computation():
    """Verify free energy evaluation and surface grid generation."""
    proj = FreeEnergyLandscapeProjector(num_basins=3, grid_res=20)
    X, Y, Z = proj.compute_surface_grid(span=2.0)

    assert X.shape == (20, 20)
    assert Y.shape == (20, 20)
    assert Z.shape == (20, 20)

    # Free energy at basin center should be lower than distant origin base
    b0 = proj.basins[0]
    e_center = proj.compute_energy_at(b0.center_2d[0], b0.center_2d[1])
    e_distant = proj.compute_energy_at(b0.center_2d[0] + 3.0, b0.center_2d[1] + 3.0)

    assert float(e_center) < float(e_distant)


def test_hidden_states_projection_and_trajectory():
    """Verify projection of token hidden states into a cognitive trajectory."""
    proj = FreeEnergyLandscapeProjector()
    h = torch.randn(1, 6, 32)
    tokens = ["The", "quick", "brown", "fox", "jumps", "over"]

    traj = proj.project_hidden_states(h, tokens=tokens)

    assert len(traj) == 6
    assert traj[0].token_str == "The"
    assert len(traj[0].coord_2d) == 2
    assert isinstance(traj[0].free_energy, float)

    proj.log_trajectory(traj)


def test_plotly_landscape_dict_builder():
    """Verify Plotly landscape dictionary formatting."""
    proj = FreeEnergyLandscapeProjector(grid_res=10)
    h = torch.randn(4, 16)
    traj = proj.project_hidden_states(h)

    p_dict = proj.build_plotly_landscape(traj)

    assert "surface" in p_dict
    assert "trajectory" in p_dict
    assert "basins" in p_dict
    assert len(p_dict["trajectory"]["x"]) == 4
