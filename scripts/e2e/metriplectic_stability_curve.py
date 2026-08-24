"""Falsification curve for the live metriplectic recurrence (bead ``0642.1.3.1``).

This experiment advances a tiny closed calcium/buffer/heat system over a fixed physical horizon with
two implementations:

* ``baseline`` uses the exact clamped-Euler calcium/buffer update retained as the live fallback;
* ``metriplectic`` uses :func:`torch_guarded_step`, the guarded implicit-midpoint update wired into
  ``SynapticPresyn.release_canonical``.

The linear calcium/buffer subsystem has an analytic matrix-exponential solution. That gives the
curve a non-circular loss target and an analytic explicit-Euler stability prediction. The default
sweep straddles that prediction: the baseline starts increasing free energy at ``dt=0.5``, while the
metriplectic arm preserves energy, produces entropy, remains finite and physical, uses no fallback,
and has no worse endpoint loss at every matched step size.

Every state transition is written to ``events.jsonl``. A strict JSON report and a Rich summary table
make both the detailed proof evidence and the divergence boundary auditable.

Run with:

    uv run python -m scripts.e2e.metriplectic_stability_curve
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat.metriplectic_integrator import torch_guarded_step
from bio_inspired_nanochat.run_logging import RunLogger
from bio_inspired_nanochat.torch_imports import Tensor, torch


@dataclass(frozen=True)
class StabilitySweepConfig:
    """Deterministic controls for the matched fixed-horizon sweep."""

    step_sizes: tuple[float, ...] = (0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0)
    duration: float = 1.0
    calcium0: tuple[float, ...] = (0.8, 0.65, 0.5)
    buffer0: tuple[float, ...] = (0.2, 0.3, 0.4)
    heat0: tuple[float, ...] = (0.0, 0.05, 0.1)
    omega: float = -0.8
    gamma_calcium: float = 0.1
    gamma_buffer: float = 0.1
    temperature: float = 0.5
    certificate_tolerance: float = 1e-10

    def validate(self) -> None:
        if not self.step_sizes or any(not math.isfinite(dt) or dt <= 0.0 for dt in self.step_sizes):
            raise ValueError("step_sizes must contain finite positive values")
        if tuple(sorted(set(self.step_sizes))) != self.step_sizes:
            raise ValueError("step_sizes must be unique and strictly increasing")
        if not math.isfinite(self.duration) or self.duration <= 0.0:
            raise ValueError("duration must be finite and positive")
        for dt in self.step_sizes:
            count = self.duration / dt
            if not math.isclose(count, round(count), rel_tol=0.0, abs_tol=1e-10):
                raise ValueError(
                    f"duration must be an integer multiple of every step size; got {self.duration}/{dt}"
                )
        sizes = {len(self.calcium0), len(self.buffer0), len(self.heat0)}
        if sizes != {len(self.calcium0)} or not self.calcium0:
            raise ValueError("calcium0, buffer0, and heat0 must be non-empty and equally sized")
        values = (*self.calcium0, *self.buffer0, *self.heat0)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("initial state must be finite")
        if any(value < 0.0 for value in self.calcium0):
            raise ValueError("initial calcium must be non-negative")
        if any(value < 0.0 or value > 1.0 for value in self.buffer0):
            raise ValueError("initial buffer must lie in [0, 1]")
        if any(value < 0.0 for value in self.heat0):
            raise ValueError("initial heat must be non-negative")
        rates = (self.omega, self.gamma_calcium, self.gamma_buffer)
        if any(not math.isfinite(value) for value in rates):
            raise ValueError("omega and damping rates must be finite")
        if self.gamma_calcium < 0.0 or self.gamma_buffer < 0.0:
            raise ValueError("damping rates must be non-negative")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(self.certificate_tolerance) or self.certificate_tolerance <= 0.0:
            raise ValueError("certificate_tolerance must be finite and positive")


@dataclass(frozen=True)
class ArmResult:
    """One arm at one step size."""

    arm: str
    step_size: float
    steps: int
    endpoint_loss: float
    finite: bool
    physical_domain: bool
    stable: bool
    max_abs_energy_drift: float
    min_entropy_production: float
    max_free_energy_delta: float
    fallback_count: int
    clamp_count: int
    divergence_reasons: tuple[str, ...]


@dataclass(frozen=True)
class CurvePoint:
    """Matched baseline/metriplectic results at one step size."""

    step_size: float
    explicit_euler_spectral_radius: float
    predicted_euler_unstable: bool
    baseline: ArmResult
    metriplectic: ArmResult
    metriplectic_loss_no_worse: bool


@dataclass(frozen=True)
class StabilitySweepReport:
    """Complete stability curve and predeclared leapfrog verdict."""

    bead: str
    config: StabilitySweepConfig
    curve: tuple[CurvePoint, ...]
    predicted_baseline_boundary: float | None
    measured_baseline_boundary: float | None
    measured_metriplectic_boundary: float | None
    leapfrog_reproduced: bool
    report_path: str
    events_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def assert_leapfrog(self) -> None:
        if self.leapfrog_reproduced:
            return
        raise AssertionError(
            "metriplectic stability leapfrog was not reproduced: "
            f"predicted baseline boundary={self.predicted_baseline_boundary}, "
            f"measured baseline boundary={self.measured_baseline_boundary}, "
            f"metriplectic boundary={self.measured_metriplectic_boundary}"
        )


def _energy(calcium: Tensor, buffer: Tensor, heat: Tensor) -> Tensor:
    return 0.5 * (calcium.square() + buffer.square()) + heat


def _free_energy(calcium: Tensor, buffer: Tensor, heat: Tensor, temperature: float) -> Tensor:
    return _energy(calcium, buffer, heat) - temperature * heat


def _physical(calcium: Tensor, buffer: Tensor, heat: Tensor, tolerance: float) -> bool:
    return bool(
        torch.all(calcium >= -tolerance)
        and torch.all(buffer >= -tolerance)
        and torch.all(buffer <= 1.0 + tolerance)
        and torch.all(heat >= -tolerance)
    )


def _exact_endpoint(cfg: StabilitySweepConfig) -> tuple[Tensor, Tensor, Tensor]:
    """Analytic continuous-flow endpoint for the linear closed subsystem."""
    calcium0 = torch.tensor(cfg.calcium0, dtype=torch.float64)
    buffer0 = torch.tensor(cfg.buffer0, dtype=torch.float64)
    heat0 = torch.tensor(cfg.heat0, dtype=torch.float64)
    generator = torch.tensor(
        [
            [-cfg.gamma_calcium, cfg.omega],
            [-cfg.omega, -cfg.gamma_buffer],
        ],
        dtype=torch.float64,
    )
    mechanical = torch.matrix_exp(generator * cfg.duration) @ torch.stack((calcium0, buffer0))
    energy0 = _energy(calcium0, buffer0, heat0)
    heat = energy0 - 0.5 * (mechanical[0].square() + mechanical[1].square())
    return mechanical[0], mechanical[1], heat


def _euler_spectral_radius(cfg: StabilitySweepConfig, dt: float) -> float:
    generator = torch.tensor(
        [
            [-cfg.gamma_calcium, cfg.omega],
            [-cfg.omega, -cfg.gamma_buffer],
        ],
        dtype=torch.float64,
    )
    update = torch.eye(2, dtype=torch.float64) + dt * generator
    return float(torch.linalg.eigvals(update).abs().max().item())


def _endpoint_loss(calcium: Tensor, buffer: Tensor, reference: tuple[Tensor, Tensor, Tensor]) -> float:
    reference_mechanical = torch.stack(reference[:2])
    observed = torch.stack((calcium, buffer))
    return float(torch.mean((observed - reference_mechanical).square()).item())


def _simulate_arm(
    cfg: StabilitySweepConfig,
    *,
    arm: str,
    dt: float,
    reference: tuple[Tensor, Tensor, Tensor],
    logger: RunLogger,
) -> ArmResult:
    calcium = torch.tensor(cfg.calcium0, dtype=torch.float64)
    buffer = torch.tensor(cfg.buffer0, dtype=torch.float64)
    heat = torch.tensor(cfg.heat0, dtype=torch.float64)
    steps = round(cfg.duration / dt)
    max_abs_energy_drift = 0.0
    min_entropy_production = math.inf
    max_free_energy_delta = -math.inf
    fallback_count = 0
    clamp_count = 0
    finite = True
    physical_domain = True

    for local_step in range(steps):
        energy0 = _energy(calcium, buffer, heat)
        entropy0 = heat
        free_energy0 = _free_energy(calcium, buffer, heat, cfg.temperature)
        calcium_raw = calcium + dt * (
            cfg.omega * buffer - cfg.gamma_calcium * calcium
        )
        buffer_raw = buffer + dt * (
            -cfg.omega * calcium - cfg.gamma_buffer * buffer
        )
        calcium_euler = calcium_raw.clamp_min(0.0)
        buffer_euler = buffer_raw.clamp(0.0, 1.0)
        clamp_count += int(torch.count_nonzero(calcium_euler != calcium_raw).item())
        clamp_count += int(torch.count_nonzero(buffer_euler != buffer_raw).item())

        breach_codes: list[int] = []
        if arm == "baseline":
            calcium_next, buffer_next, heat_next = calcium_euler, buffer_euler, heat
            used_fallbacks = 0
        elif arm == "metriplectic":
            calcium_next, buffer_next, heat_next, record = torch_guarded_step(
                calcium,
                buffer,
                heat,
                dt=dt,
                omega=cfg.omega,
                gC=cfg.gamma_calcium,
                gB=cfg.gamma_buffer,
                temperature=cfg.temperature,
                fallback=(calcium_euler, buffer_euler, heat),
            )
            used_fallbacks = int(record.fallback_mask.sum().item())
            fallback_count += used_fallbacks
            breach_codes = [int(value) for value in record.breach_code.tolist()]
        else:
            raise ValueError(f"unknown arm {arm!r}")

        energy1 = _energy(calcium_next, buffer_next, heat_next)
        entropy1 = heat_next
        free_energy1 = _free_energy(calcium_next, buffer_next, heat_next, cfg.temperature)
        energy_drift = energy1 - energy0
        entropy_production = entropy1 - entropy0
        free_energy_delta = free_energy1 - free_energy0
        max_abs_energy_drift = max(
            max_abs_energy_drift, float(energy_drift.abs().max().item())
        )
        min_entropy_production = min(
            min_entropy_production, float(entropy_production.min().item())
        )
        max_free_energy_delta = max(
            max_free_energy_delta, float(free_energy_delta.max().item())
        )
        step_finite = bool(
            torch.all(torch.isfinite(calcium_next))
            and torch.all(torch.isfinite(buffer_next))
            and torch.all(torch.isfinite(heat_next))
        )
        step_physical = _physical(
            calcium_next, buffer_next, heat_next, cfg.certificate_tolerance
        )
        finite = finite and step_finite
        physical_domain = physical_domain and step_physical
        logger.event(
            "metriplectic_stability_step",
            arm=arm,
            step_size=dt,
            local_step=local_step,
            calcium=calcium_next.tolist(),
            buffer=buffer_next.tolist(),
            heat=heat_next.tolist(),
            energy_drift=energy_drift.tolist(),
            entropy_production=entropy_production.tolist(),
            free_energy_delta=free_energy_delta.tolist(),
            finite=step_finite,
            physical_domain=step_physical,
            fallbacks=used_fallbacks,
            breach_codes=breach_codes,
        )
        calcium, buffer, heat = calcium_next, buffer_next, heat_next

    reasons: list[str] = []
    if not finite:
        reasons.append("nonfinite")
    if not physical_domain:
        reasons.append("physical_domain")
    if max_free_energy_delta > cfg.certificate_tolerance:
        reasons.append("free_energy_increase")
    if arm == "metriplectic":
        if max_abs_energy_drift > cfg.certificate_tolerance:
            reasons.append("energy_drift")
        if min_entropy_production < -cfg.certificate_tolerance:
            reasons.append("negative_entropy_production")
        if fallback_count:
            reasons.append("fallback")
    return ArmResult(
        arm=arm,
        step_size=dt,
        steps=steps,
        endpoint_loss=_endpoint_loss(calcium, buffer, reference),
        finite=finite,
        physical_domain=physical_domain,
        stable=not reasons,
        max_abs_energy_drift=max_abs_energy_drift,
        min_entropy_production=min_entropy_production,
        max_free_energy_delta=max_free_energy_delta,
        fallback_count=fallback_count,
        clamp_count=clamp_count,
        divergence_reasons=tuple(reasons),
    )


def _first_boundary(curve: list[CurvePoint], arm: str) -> float | None:
    for point in curve:
        result = point.baseline if arm == "baseline" else point.metriplectic
        if not result.stable:
            return point.step_size
    return None


def run_stability_sweep(
    cfg: StabilitySweepConfig | None = None,
    *,
    run_dir: str | Path | None = None,
) -> StabilitySweepReport:
    """Run both arms, persist the curve, and return its falsification verdict."""
    cfg = cfg or StabilitySweepConfig()
    cfg.validate()
    output_dir = Path(run_dir) if run_dir is not None else Path("runs/e2e/metriplectic_stability")
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = _exact_endpoint(cfg)
    curve: list[CurvePoint] = []

    with RunLogger(
        output_dir,
        name="metriplectic_stability",
        console=False,
        provenance={"bead": "bio_inspired_nanochat-0642.1.3.1", "config": asdict(cfg)},
    ) as logger:
        for dt in cfg.step_sizes:
            spectral_radius = _euler_spectral_radius(cfg, dt)
            baseline = _simulate_arm(
                cfg, arm="baseline", dt=dt, reference=reference, logger=logger
            )
            metriplectic = _simulate_arm(
                cfg, arm="metriplectic", dt=dt, reference=reference, logger=logger
            )
            point = CurvePoint(
                step_size=dt,
                explicit_euler_spectral_radius=spectral_radius,
                predicted_euler_unstable=(
                    spectral_radius > 1.0 + cfg.certificate_tolerance
                ),
                baseline=baseline,
                metriplectic=metriplectic,
                metriplectic_loss_no_worse=(
                    metriplectic.endpoint_loss
                    <= baseline.endpoint_loss + cfg.certificate_tolerance
                ),
            )
            curve.append(point)
            logger.event("metriplectic_stability_curve_point", **asdict(point))

        predicted_boundary = next(
            (point.step_size for point in curve if point.predicted_euler_unstable), None
        )
        baseline_boundary = _first_boundary(curve, "baseline")
        metriplectic_boundary = _first_boundary(curve, "metriplectic")
        leapfrog = (
            predicted_boundary is not None
            and baseline_boundary == predicted_boundary
            and metriplectic_boundary is None
            and all(point.metriplectic_loss_no_worse for point in curve)
        )
        report_path = output_dir / "stability_curve.json"
        report = StabilitySweepReport(
            bead="bio_inspired_nanochat-0642.1.3.1",
            config=cfg,
            curve=tuple(curve),
            predicted_baseline_boundary=predicted_boundary,
            measured_baseline_boundary=baseline_boundary,
            measured_metriplectic_boundary=metriplectic_boundary,
            leapfrog_reproduced=leapfrog,
            report_path=str(report_path),
            events_path=str(output_dir / "events.jsonl"),
        )
        logger.event(
            "metriplectic_stability_summary",
            predicted_baseline_boundary=predicted_boundary,
            measured_baseline_boundary=baseline_boundary,
            measured_metriplectic_boundary=metriplectic_boundary,
            leapfrog_reproduced=leapfrog,
        )
        report_path.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return report


def render_report(report: StabilitySweepReport, *, console: Console | None = None) -> None:
    """Render the matched stability/loss curve through Rich."""
    console = console or Console()
    table = Table(title="Metriplectic stability falsification curve")
    table.add_column("dt", justify="right")
    table.add_column("rho(Euler)", justify="right")
    table.add_column("baseline", justify="center")
    table.add_column("baseline loss", justify="right")
    table.add_column("max baseline dF", justify="right")
    table.add_column("GENERIC", justify="center")
    table.add_column("GENERIC loss", justify="right")
    table.add_column("max |dE|", justify="right")
    table.add_column("fallbacks", justify="right")
    for point in report.curve:
        table.add_row(
            f"{point.step_size:g}",
            f"{point.explicit_euler_spectral_radius:.4f}",
            "stable" if point.baseline.stable else "DIVERGED",
            f"{point.baseline.endpoint_loss:.3e}",
            f"{point.baseline.max_free_energy_delta:.3e}",
            "stable" if point.metriplectic.stable else "DIVERGED",
            f"{point.metriplectic.endpoint_loss:.3e}",
            f"{point.metriplectic.max_abs_energy_drift:.3e}",
            str(point.metriplectic.fallback_count),
        )
    console.print(table)
    outcome = "REPRODUCED" if report.leapfrog_reproduced else "NOT REPRODUCED"
    console.print(
        f"Leapfrog {outcome}: predicted/measured baseline boundary "
        f"{report.predicted_baseline_boundary}/{report.measured_baseline_boundary}; "
        f"metriplectic boundary {report.measured_metriplectic_boundary}."
    )
    console.print(f"JSON report: {report.report_path}")
    console.print(f"Detailed events: {report.events_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Clamped-Euler versus guarded metriplectic stability curve"
    )
    parser.add_argument(
        "--run-dir",
        default="runs/e2e/metriplectic_stability",
        help="directory for stability_curve.json and events.jsonl",
    )
    parser.add_argument(
        "--step-sizes",
        nargs="+",
        type=float,
        default=None,
        help="strictly increasing step sizes that divide the fixed duration",
    )
    args = parser.parse_args(argv)
    cfg = StabilitySweepConfig(
        step_sizes=tuple(args.step_sizes)
        if args.step_sizes is not None
        else StabilitySweepConfig.step_sizes
    )
    report = run_stability_sweep(cfg, run_dir=args.run_dir)
    render_report(report)
    return 0 if report.leapfrog_reproduced else 1


if __name__ == "__main__":
    raise SystemExit(main())
