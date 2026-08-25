"""Illustrative hidden-state landscape visualizer (bead `r00r.10`).

Projects hidden states through a fixed, deterministic 2D map and overlays hand-authored
potential wells. This is a qualitative visualization, not fMRI, a learned semantic probe,
or evidence that the model follows a Lyapunov/free-energy objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from torch import Tensor



@dataclass
class AttractorBasin:
    """A hand-authored potential well on the illustrative 2D landscape."""

    basin_id: int
    name: str
    center_2d: Tuple[float, float]
    depth: float
    width: float


@dataclass
class CognitiveTrajectoryPoint:
    """A token position and its coordinates on the illustrative 2D landscape."""

    token_idx: int
    token_str: str
    coord_2d: Tuple[float, float]
    free_energy: float
    nearest_basin_id: Optional[int]


class FreeEnergyLandscapeProjector:
    """Projects hidden states onto a deterministic, hand-authored potential surface."""

    def __init__(self, num_basins: int = 4, grid_res: int = 40):
        if (
            isinstance(num_basins, bool)
            or not isinstance(num_basins, int)
            or not 0 <= num_basins <= 4
        ):
            raise ValueError("num_basins must be an integer in [0, 4]")
        if isinstance(grid_res, bool) or not isinstance(grid_res, int) or grid_res < 2:
            raise ValueError("grid_res must be an integer of at least 2")
        self.grid_res = grid_res
        self.basins: List[AttractorBasin] = [
            AttractorBasin(0, "Illustrative Basin A", (-1.5, -1.0), 3.0, 0.8),
            AttractorBasin(1, "Illustrative Basin B", (1.5, -1.2), 3.5, 0.7),
            AttractorBasin(2, "Illustrative Basin C", (0.0, 1.5), 2.5, 1.0),
            AttractorBasin(3, "Illustrative Basin D", (-1.2, 1.2), 2.8, 0.9),
        ][:num_basins]

    def compute_energy_at(self, x: float | np.ndarray, y: float | np.ndarray) -> np.ndarray:
        """Evaluate the illustrative potential V(x, y) = harmonic_base - sum(wells)."""
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if not np.isfinite(x_arr).all() or not np.isfinite(y_arr).all():
            raise ValueError("energy coordinates must contain only finite values")

        # Base parabolic harmonic confinement
        base_v = 0.5 * (x_arr**2 + y_arr**2)

        # Subtract Gaussian potential wells for each attractor basin
        well_sum = np.zeros_like(base_v)
        for b in self.basins:
            bx, by = b.center_2d
            dist_sq = (x_arr - bx) ** 2 + (y_arr - by) ** 2
            well = b.depth * np.exp(-dist_sq / (2.0 * (b.width**2)))
            well_sum += well

        return base_v - well_sum

    def compute_surface_grid(self, span: float = 3.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate (X, Y, Z) coordinate meshgrid for 3D surface rendering."""
        if not np.isfinite(span) or span <= 0.0:
            raise ValueError("span must be finite and positive")
        x = np.linspace(-span, span, self.grid_res)
        y = np.linspace(-span, span, self.grid_res)
        X, Y = np.meshgrid(x, y)
        Z = self.compute_energy_at(X, Y)
        return X, Y, Z

    def project_hidden_states(
        self,
        hidden_states: Tensor,
        tokens: Optional[List[str]] = None,
    ) -> List[CognitiveTrajectoryPoint]:
        """Project (T, D) token activations onto the 2D energy surface trajectory."""
        h = hidden_states.detach().cpu().float()
        if h.ndim == 3:
            if h.shape[0] != 1:
                raise ValueError("rank-3 hidden_states must have a singleton batch dimension")
            h = h[0]
        if h.ndim != 2 or h.shape[0] == 0 or h.shape[1] == 0:
            raise ValueError("hidden_states must have non-empty shape (T, D) or (1, T, D)")
        if not torch.isfinite(h).all():
            raise ValueError("hidden_states must contain only finite values")

        t_len, d_dim = h.shape

        # Use two fixed orthonormal projections when D >= 2; D == 1 occupies only x.
        proj_matrix = torch.zeros(d_dim, 2)
        if d_dim == 1:
            proj_matrix[0, 0] = 1.0
        else:
            half = max(1, d_dim // 2)
            proj_matrix[:half, 0] = 1.0 / np.sqrt(half)
            proj_matrix[half:, 1] = 1.0 / np.sqrt(d_dim - half)

        coords_2d = (h @ proj_matrix).numpy()

        # Bound each point independently. Whole-trajectory max normalization made an
        # already-recorded prefix move whenever a larger future activation was appended.
        coords_2d = np.tanh(coords_2d) * 2.0

        trajectory: List[CognitiveTrajectoryPoint] = []
        for i in range(t_len):
            cx, cy = float(coords_2d[i, 0]), float(coords_2d[i, 1])
            fe = float(self.compute_energy_at(cx, cy))

            # Nearest basin
            if self.basins:
                nearest_b = min(
                    self.basins,
                    key=lambda b: (cx - b.center_2d[0]) ** 2 + (cy - b.center_2d[1]) ** 2,
                )
                nearest_id = nearest_b.basin_id
            else:
                nearest_id = None

            t_str = tokens[i] if tokens and i < len(tokens) else f"tok_{i}"

            trajectory.append(
                CognitiveTrajectoryPoint(
                    token_idx=i,
                    token_str=t_str,
                    coord_2d=(cx, cy),
                    free_energy=fe,
                    nearest_basin_id=nearest_id,
                )
            )

        return trajectory

    def build_plotly_landscape(
        self,
        trajectory: List[CognitiveTrajectoryPoint],
    ) -> Dict[str, Any]:
        """Build a Plotly-compatible dictionary for the surface and projected trajectory."""
        X, Y, Z = self.compute_surface_grid()

        traj_x = [p.coord_2d[0] for p in trajectory]
        traj_y = [p.coord_2d[1] for p in trajectory]
        traj_z = [p.free_energy for p in trajectory]
        traj_labels = [f"Step {p.token_idx}: '{p.token_str}' (F={p.free_energy:.2f})" for p in trajectory]

        return {
            "surface": {"x": X.tolist(), "y": Y.tolist(), "z": Z.tolist()},
            "trajectory": {
                "x": traj_x,
                "y": traj_y,
                "z": traj_z,
                "labels": traj_labels,
            },
            "basins": [
                {"name": b.name, "center": b.center_2d, "depth": b.depth}
                for b in self.basins
            ],
        }

    def log_trajectory(
        self,
        trajectory: List[CognitiveTrajectoryPoint],
        console: Optional[Console] = None,
    ) -> None:
        """Render a Rich table of the projected hidden-state trajectory."""
        c = console or Console()
        c.rule("[bold cyan]Illustrative Hidden-State Landscape[/bold cyan]")

        table = Table(title="Token Positions on a Hand-Authored Potential Surface")
        table.add_column("Step", justify="right")
        table.add_column("Token", style="bold")
        table.add_column("2D Coordinates (x, y)", justify="center")
        table.add_column("Illustrative Potential", justify="right", style="bold green")
        table.add_column("Nearest Well", style="cyan")

        for p in trajectory:
            b_name = "Unassigned"
            if p.nearest_basin_id is not None:
                b_name = next(
                    (b.name for b in self.basins if b.basin_id == p.nearest_basin_id),
                    "Unknown",
                )
            table.add_row(
                str(p.token_idx),
                p.token_str,
                f"({p.coord_2d[0]:.2f}, {p.coord_2d[1]:.2f})",
                f"{p.free_energy:.3f}",
                b_name,
            )
        c.print(table)
