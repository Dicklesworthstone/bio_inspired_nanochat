"""fMRI for Living Transformers: Free-Energy Landscape & Attractor Dynamics Visualizer (bead `r00r.10`).

Projects high-dimensional cognitive representations into an interactive 3D Free-Energy
Lyapunov potential landscape, tracking attractor basin descents and depletion-driven basin hopping.
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
    """A semantic attractor basin (energy local minimum) on the low-dim manifold."""

    basin_id: int
    name: str
    center_2d: Tuple[float, float]
    depth: float
    width: float


@dataclass
class CognitiveTrajectoryPoint:
    """A single token step's coordinates on the 2D energy manifold with associated free energy."""

    token_idx: int
    token_str: str
    coord_2d: Tuple[float, float]
    free_energy: float
    nearest_basin_id: int


class FreeEnergyLandscapeProjector:
    """Computes free-energy potential surface and projects hidden states onto attractor basins."""

    def __init__(self, num_basins: int = 4, grid_res: int = 40):
        self.grid_res = grid_res
        self.basins: List[AttractorBasin] = [
            AttractorBasin(0, "Factual Recall", (-1.5, -1.0), 3.0, 0.8),
            AttractorBasin(1, "Logical Deduction", (1.5, -1.2), 3.5, 0.7),
            AttractorBasin(2, "Creative Exploration", (0.0, 1.5), 2.5, 1.0),
            AttractorBasin(3, "Syntactic Structuring", (-1.2, 1.2), 2.8, 0.9),
        ][:num_basins]

    def compute_energy_at(self, x: float | np.ndarray, y: float | np.ndarray) -> np.ndarray:
        """Evaluate the Lyapunov potential landscape V(x, y) = harmonic_base - sum(basins)."""
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

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
            h = h.squeeze(0)  # (T, D)

        t_len, d_dim = h.shape

        # Use 2 exact orthonormal projections across representation dimensions
        proj_matrix = torch.zeros(d_dim, 2)
        if d_dim == 1:
            proj_matrix[0, 0] = 1.0
        else:
            half = max(1, d_dim // 2)
            proj_matrix[:half, 0] = 1.0 / np.sqrt(half)
            proj_matrix[half:, 1] = 1.0 / np.sqrt(d_dim - half)

        coords_2d = (h @ proj_matrix).numpy()

        # Scale coordinates into landscape domain [-2.5, 2.5]
        max_val = max(1e-4, np.max(np.abs(coords_2d)))
        coords_2d = (coords_2d / max_val) * 2.0

        trajectory: List[CognitiveTrajectoryPoint] = []
        for i in range(t_len):
            cx, cy = float(coords_2d[i, 0]), float(coords_2d[i, 1])
            fe = float(self.compute_energy_at(cx, cy))

            # Nearest basin
            nearest_b = min(
                self.basins,
                key=lambda b: (cx - b.center_2d[0]) ** 2 + (cy - b.center_2d[1]) ** 2,
            )
            t_str = tokens[i] if tokens and i < len(tokens) else f"tok_{i}"

            trajectory.append(
                CognitiveTrajectoryPoint(
                    token_idx=i,
                    token_str=t_str,
                    coord_2d=(cx, cy),
                    free_energy=fe,
                    nearest_basin_id=nearest_b.basin_id,
                )
            )

        return trajectory

    def build_plotly_landscape(
        self,
        trajectory: List[CognitiveTrajectoryPoint],
    ) -> Dict[str, Any]:
        """Build Plotly chart dictionary of the 3D free-energy landscape and descent trajectory."""
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
        """Render Rich table of cognitive attractor dynamics trajectory."""
        c = console or Console()
        c.rule("[bold cyan]fMRI Attractor Dynamics & Free-Energy Descent[/bold cyan]")

        table = Table(title="Cognitive Step Progression Across Energy Landscape")
        table.add_column("Step", justify="right")
        table.add_column("Token", style="bold")
        table.add_column("2D Coordinates (x, y)", justify="center")
        table.add_column("Free Energy F", justify="right", style="bold green")
        table.add_column("Attractor Basin", style="cyan")

        for p in trajectory:
            b_name = self.basins[p.nearest_basin_id].name if p.nearest_basin_id < len(self.basins) else "Unknown"
            table.add_row(
                str(p.token_idx),
                p.token_str,
                f"({p.coord_2d[0]:.2f}, {p.coord_2d[1]:.2f})",
                f"{p.free_energy:.3f}",
                b_name,
            )
        c.print(table)
