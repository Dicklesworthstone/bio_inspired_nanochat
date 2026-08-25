"""Tests for the illustrative hidden-state landscape visualizer (bead `r00r.10`)."""

import math

import pytest
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


def test_projection_without_basins_reports_no_assignment():
    """An empty landscape must not fabricate a nearest basin or global minimum."""
    proj = FreeEnergyLandscapeProjector(num_basins=0)
    traj = proj.project_hidden_states(torch.randn(3, 8))

    assert proj.basins == []
    assert all(point.nearest_basin_id is None for point in traj)
    proj.log_trajectory(traj)


def test_projector_rejects_invalid_configuration_and_hidden_states():
    with pytest.raises(ValueError, match="num_basins"):
        FreeEnergyLandscapeProjector(num_basins=-1)
    with pytest.raises(ValueError, match="num_basins"):
        FreeEnergyLandscapeProjector(num_basins=5)
    with pytest.raises(ValueError, match="grid_res"):
        FreeEnergyLandscapeProjector(grid_res=1)

    projector = FreeEnergyLandscapeProjector()
    with pytest.raises(ValueError, match="singleton batch"):
        projector.project_hidden_states(torch.randn(2, 3, 8))
    with pytest.raises(ValueError, match="non-empty shape"):
        projector.project_hidden_states(torch.empty(0, 8))
    with pytest.raises(ValueError, match="finite values"):
        projector.project_hidden_states(torch.tensor([[1.0, math.nan]]))
    for invalid_span in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="span"):
            projector.compute_surface_grid(span=invalid_span)
    with pytest.raises(ValueError, match="energy coordinates"):
        projector.compute_energy_at(math.nan, 0.0)


def test_projected_prefix_does_not_move_when_future_states_are_appended():
    projector = FreeEnergyLandscapeProjector()
    prefix = torch.randn(3, 8)
    extended = torch.cat([prefix, torch.full((1, 8), 100.0)], dim=0)

    prefix_trajectory = projector.project_hidden_states(prefix)
    extended_trajectory = projector.project_hidden_states(extended)

    for before, after in zip(prefix_trajectory, extended_trajectory[: len(prefix_trajectory)]):
        assert before.coord_2d == pytest.approx(after.coord_2d)
        assert before.free_energy == pytest.approx(after.free_energy)
        assert before.nearest_basin_id == after.nearest_basin_id
