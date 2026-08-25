"""Falsification curve for the live metriplectic recurrence (bead ``0642.1.3``).

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

Every state transition is written to ``events.jsonl``. The multi-seed path routes paired endpoint
loss and stress-regime divergence through ``eval_stats``, emits a strict JSON report, and appends the
underlying observations to the committed results registry. Rich tables make the boundary and honest
verdict auditable.

Run with:

    uv run python -m scripts.e2e.metriplectic_stability_curve
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat.eval_stats import PairedResult, paired_comparison
from bio_inspired_nanochat.metriplectic_integrator import (
    GuardThresholds,
    M_op,
    field,
    guarded_step,
    torch_guarded_step,
)
from bio_inspired_nanochat.results_registry import DEFAULT_REGISTRY, append_record, make_record
from bio_inspired_nanochat.run_logging import RunLogger
from bio_inspired_nanochat.torch_imports import Tensor, torch


def _json_safe(v: Any) -> Any:
    """Map non-finite floats to null so allow_nan=False writers never crash on
    legitimate ±inf statistics (e.g. zero-variance paired effect sizes)."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    return v


def _dump_report_json(report: Any) -> str:
    return json.dumps(_json_safe(report.to_dict()), indent=2, sort_keys=True, allow_nan=False)


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
    max_degeneracy_residual: float | None
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
class FallbackInjectionResult:
    """Evidence from a deliberately degeneracy-breaking operator."""

    steps: int
    fallback_count: int
    max_residual: float
    every_breach_was_degeneracy: bool
    every_fallback_matched_baseline: bool
    trajectory_finite: bool
    physical_domain: bool
    verified: bool


@dataclass(frozen=True)
class ProofObligationResult:
    """Aggregate certificate and safety-net verdict over the real sweep."""

    max_abs_energy_drift: float
    min_entropy_production: float
    max_free_energy_delta: float
    max_degeneracy_residual: float
    structural_fallback_count: int
    fallback_injection: FallbackInjectionResult
    verified: bool


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
    proof_obligation: ProofObligationResult
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


@dataclass(frozen=True)
class SeededOutcome:
    """Matched headline outcomes for one independently sampled physical state batch."""

    seed: int
    baseline_endpoint_loss: float
    metriplectic_endpoint_loss: float
    baseline_divergence_rate: float
    metriplectic_divergence_rate: float
    baseline_boundary: float | None
    metriplectic_boundary: float | None


@dataclass(frozen=True)
class StatisticalSweepReport:
    """Multi-seed paired analysis and predeclared honest verdict."""

    bead: str
    run_id: str
    seeds: tuple[int, ...]
    stress_step_sizes: tuple[float, ...]
    outcomes: tuple[SeededOutcome, ...]
    endpoint_loss_comparison: PairedResult
    divergence_rate_comparison: PairedResult
    verdict: str
    verdict_reason: str
    report_path: str
    events_path: str
    registry_path: str | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # A constant non-zero paired shift has an infinite t statistic. JSON has no portable
        # infinity literal, so strict evidence records it as null while retaining p=0 and the
        # exact constant bootstrap interval.
        for key in ("endpoint_loss_comparison", "divergence_rate_comparison"):
            t_stat = payload[key]["t_stat"]
            if not math.isfinite(t_stat):
                payload[key]["t_stat"] = None
        return payload

    def assert_positive(self) -> None:
        if self.verdict != "positive":
            raise AssertionError(
                f"expected positive paired verdict, got {self.verdict}: "
                f"{self.verdict_reason}"
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
    max_degeneracy_residual: float | None = None
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
        res_l_grad_s: list[float] = []
        res_m_grad_e: list[float] = []
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
            res_l_grad_s = [float(value) for value in record.res_L_gradS.tolist()]
            res_m_grad_e = [float(value) for value in record.res_M_gradE.tolist()]
            step_residual = max((*res_l_grad_s, *res_m_grad_e), default=0.0)
            max_degeneracy_residual = max(max_degeneracy_residual or 0.0, step_residual)
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
            res_L_gradS=res_l_grad_s,
            res_M_gradE=res_m_grad_e,
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
        max_degeneracy_residual=max_degeneracy_residual,
        fallback_count=fallback_count,
        clamp_count=clamp_count,
        divergence_reasons=tuple(reasons),
    )


def _run_fallback_injection(
    cfg: StabilitySweepConfig,
    logger: RunLogger,
    *,
    steps: int = 16,
    dt: float = 0.05,
) -> FallbackInjectionResult:
    """Break ``M*grad(E)=0`` while leaving the Euler fallback vector field unchanged."""

    def bad_m(
        state: np.ndarray,
        g_c: float = cfg.gamma_calcium,
        g_b: float = cfg.gamma_buffer,
    ) -> np.ndarray:
        # The added C/B diagonal violates M*grad(E)=0, but annihilates grad(S)=(0,0,1).
        # Therefore the proposed structural step is rejected while the deterministic Euler
        # fallback remains exactly the engineering baseline rather than inheriting the defect.
        return M_op(state, g_c, g_b) + np.diag([0.3, 0.3, 0.0])

    state = np.array([cfg.calcium0[0], cfg.buffer0[0], cfg.heat0[0]], dtype=np.float64)
    fallback_count = 0
    max_residual = 0.0
    every_breach_was_degeneracy = True
    every_fallback_matched_baseline = True
    trajectory_finite = True
    physical_domain = True
    thresholds = GuardThresholds(
        eps_E=cfg.certificate_tolerance,
        eps_S=cfg.certificate_tolerance,
        eps_D=cfg.certificate_tolerance,
    )

    for local_step in range(steps):
        expected = state + dt * field(
            state,
            cfg.omega,
            cfg.gamma_calcium,
            cfg.gamma_buffer,
        )
        state, record = guarded_step(
            state,
            dt,
            local_step,
            thresholds,
            omega=cfg.omega,
            gC=cfg.gamma_calcium,
            gB=cfg.gamma_buffer,
            T=cfg.temperature,
            M_fn=bad_m,
        )
        fallback_count += int(record.used_fallback)
        max_residual = max(max_residual, record.res_L_gradS, record.res_M_gradE)
        every_breach_was_degeneracy &= record.breach == "degeneracy"
        matches_baseline = bool(np.array_equal(state, expected))
        every_fallback_matched_baseline &= matches_baseline
        step_finite = bool(np.all(np.isfinite(state)))
        step_physical = bool(
            state[0] >= -cfg.certificate_tolerance
            and -cfg.certificate_tolerance <= state[1] <= 1.0 + cfg.certificate_tolerance
            and state[2] >= -cfg.certificate_tolerance
        )
        trajectory_finite &= step_finite
        physical_domain &= step_physical
        logger.event(
            "metriplectic_fallback_injection_step",
            local_step=local_step,
            step_size=dt,
            state=state.tolist(),
            breach=record.breach,
            used_fallback=record.used_fallback,
            fallback_matches_baseline=matches_baseline,
            res_L_gradS=record.res_L_gradS,
            res_M_gradE=record.res_M_gradE,
            finite=step_finite,
            physical_domain=step_physical,
        )

    verified = (
        fallback_count == steps
        and max_residual > cfg.certificate_tolerance
        and every_breach_was_degeneracy
        and every_fallback_matched_baseline
        and trajectory_finite
        and physical_domain
    )
    result = FallbackInjectionResult(
        steps=steps,
        fallback_count=fallback_count,
        max_residual=max_residual,
        every_breach_was_degeneracy=every_breach_was_degeneracy,
        every_fallback_matched_baseline=every_fallback_matched_baseline,
        trajectory_finite=trajectory_finite,
        physical_domain=physical_domain,
        verified=verified,
    )
    logger.event("metriplectic_fallback_injection_summary", **asdict(result))
    return result


def _proof_obligation(
    cfg: StabilitySweepConfig,
    curve: list[CurvePoint],
    injection: FallbackInjectionResult,
) -> ProofObligationResult:
    structural = [point.metriplectic for point in curve]
    max_energy_drift = max(result.max_abs_energy_drift for result in structural)
    min_entropy_production = min(result.min_entropy_production for result in structural)
    max_free_energy_delta = max(result.max_free_energy_delta for result in structural)
    max_degeneracy_residual = max(
        result.max_degeneracy_residual or 0.0 for result in structural
    )
    fallback_count = sum(result.fallback_count for result in structural)
    verified = (
        max_energy_drift <= cfg.certificate_tolerance
        and min_entropy_production >= -cfg.certificate_tolerance
        and max_free_energy_delta <= cfg.certificate_tolerance
        and max_degeneracy_residual <= cfg.certificate_tolerance
        and fallback_count == 0
        and injection.verified
    )
    return ProofObligationResult(
        max_abs_energy_drift=max_energy_drift,
        min_entropy_production=min_entropy_production,
        max_free_energy_delta=max_free_energy_delta,
        max_degeneracy_residual=max_degeneracy_residual,
        structural_fallback_count=fallback_count,
        fallback_injection=injection,
        verified=verified,
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
        injection = _run_fallback_injection(cfg, logger)
        proof_obligation = _proof_obligation(cfg, curve, injection)
        report_path = output_dir / "stability_curve.json"
        report = StabilitySweepReport(
            bead="bio_inspired_nanochat-0642.1.3.1",
            config=cfg,
            curve=tuple(curve),
            predicted_baseline_boundary=predicted_boundary,
            measured_baseline_boundary=baseline_boundary,
            measured_metriplectic_boundary=metriplectic_boundary,
            leapfrog_reproduced=leapfrog,
            proof_obligation=proof_obligation,
            report_path=str(report_path),
            events_path=str(output_dir / "events.jsonl"),
        )
        logger.event(
            "metriplectic_stability_summary",
            predicted_baseline_boundary=predicted_boundary,
            measured_baseline_boundary=baseline_boundary,
            measured_metriplectic_boundary=metriplectic_boundary,
            leapfrog_reproduced=leapfrog,
            proof_obligation_verified=proof_obligation.verified,
        )
        report_path.write_text(
            _dump_report_json(report) + "\n",
            encoding="utf-8",
        )
    return report


def _seeded_config(cfg: StabilitySweepConfig, seed: int) -> StabilitySweepConfig:
    """Sample a conservative physical batch without changing the analytic stability boundary."""
    rng = np.random.default_rng(seed)
    batch_size = 8
    calcium0 = tuple(float(value) for value in rng.uniform(0.60, 0.82, batch_size))
    buffer0 = tuple(float(value) for value in rng.uniform(0.18, 0.32, batch_size))
    heat0 = tuple(float(value) for value in rng.uniform(0.0, 0.10, batch_size))
    seeded = replace(cfg, calcium0=calcium0, buffer0=buffer0, heat0=heat0)
    seeded.validate()
    return seeded


def _paired_verdict(loss: PairedResult, divergence: PairedResult) -> tuple[str, str]:
    """Apply the predeclared two-metric rule; inconclusive evidence is an honest null."""

    def supports_improvement(result: PairedResult) -> bool:
        return (
            result.mean_delta < 0.0
            and result.delta_ci_high < 0.0
            and result.t_p_value < 0.05
            and result.wilcoxon_p_value <= 0.05
        )

    def supports_harm(result: PairedResult) -> bool:
        return (
            result.mean_delta > 0.0
            and result.delta_ci_low > 0.0
            and result.t_p_value < 0.05
            and result.wilcoxon_p_value <= 0.05
        )

    if supports_improvement(loss) and supports_improvement(divergence):
        return (
            "positive",
            "both endpoint loss and stress-regime divergence rate improve; both paired "
            "bootstrap intervals exclude zero and both paired tests meet alpha=0.05",
        )
    if supports_harm(loss) and supports_harm(divergence):
        return (
            "worse",
            "both endpoint loss and stress-regime divergence rate worsen with supported "
            "paired effects",
        )
    return (
        "null",
        "the two predeclared metrics do not both support improvement or both support harm",
    )


def _append_statistical_records(
    report: StatisticalSweepReport,
    cfg: StabilitySweepConfig,
    registry_path: str,
) -> None:
    """Append the matched seed/arm observations that underlie the paired verdict."""
    loss = report.endpoint_loss_comparison
    divergence = report.divergence_rate_comparison
    comparison_note = (
        f"experiment=metriplectic_stability; paired_verdict={report.verdict}; "
        f"endpoint_delta={loss.mean_delta:.17g}; "
        f"endpoint_ci=[{loss.delta_ci_low:.17g},{loss.delta_ci_high:.17g}]; "
        f"divergence_delta={divergence.mean_delta:.17g}; "
        f"divergence_ci=[{divergence.delta_ci_low:.17g},{divergence.delta_ci_high:.17g}]; "
        f"artifact={report.report_path}"
    )
    for outcome in report.outcomes:
        seeded_cfg = _seeded_config(cfg, outcome.seed)
        for arm, endpoint_loss, divergence_rate in (
            (
                "baseline",
                outcome.baseline_endpoint_loss,
                outcome.baseline_divergence_rate,
            ),
            (
                "metriplectic",
                outcome.metriplectic_endpoint_loss,
                outcome.metriplectic_divergence_rate,
            ),
        ):
            record = make_record(
                "eval",
                {
                    "integrator_endpoint_loss": endpoint_loss,
                    "integrator_divergence_rate": divergence_rate,
                },
                run_id=f"{report.run_id}-{arm}-s{outcome.seed}",
                syn_cfg=seeded_cfg,
                seed=outcome.seed,
                notes=f"arm={arm}; {comparison_note}",
            )
            append_record(record, registry_path)


def run_statistical_stability_sweep(
    cfg: StabilitySweepConfig | None = None,
    *,
    seeds: tuple[int, ...] = (11, 23, 37, 53, 71, 89, 107, 131),
    bootstrap_samples: int = 10_000,
    run_dir: str | Path | None = None,
    registry_path: str | Path | None = None,
) -> StatisticalSweepReport:
    """Run matched seeded batches and persist paired loss/divergence statistics."""
    cfg = cfg or StabilitySweepConfig()
    cfg.validate()
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must contain at least two unique values")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")

    output_dir = (
        Path(run_dir)
        if run_dir is not None
        else Path("runs/e2e/metriplectic_stability/statistics")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "statistics.json"
    registry_path_str = str(registry_path) if registry_path is not None else None
    outcomes: list[SeededOutcome] = []
    stress_step_sizes: tuple[float, ...] | None = None

    with RunLogger(
        output_dir,
        name="metriplectic_stability_statistics",
        console=False,
        provenance={
            "bead": "bio_inspired_nanochat-0642.1.3.2",
            "config": asdict(cfg),
            "seeds": seeds,
        },
    ) as logger:
        for seed in seeds:
            seeded_report = run_stability_sweep(
                _seeded_config(cfg, seed),
                run_dir=output_dir / f"seed-{seed}",
            )
            seeded_stress = tuple(
                point.step_size
                for point in seeded_report.curve
                if point.predicted_euler_unstable
            )
            if not seeded_stress:
                raise ValueError("the sweep must include at least one predicted-unstable step size")
            if stress_step_sizes is None:
                stress_step_sizes = seeded_stress
            elif seeded_stress != stress_step_sizes:
                raise AssertionError("analytic stress regime changed across seeded states")
            stress_points = [
                point for point in seeded_report.curve if point.step_size in seeded_stress
            ]
            headline = seeded_report.curve[-1]
            outcome = SeededOutcome(
                seed=seed,
                baseline_endpoint_loss=headline.baseline.endpoint_loss,
                metriplectic_endpoint_loss=headline.metriplectic.endpoint_loss,
                baseline_divergence_rate=sum(
                    not point.baseline.stable for point in stress_points
                )
                / len(stress_points),
                metriplectic_divergence_rate=sum(
                    not point.metriplectic.stable for point in stress_points
                )
                / len(stress_points),
                baseline_boundary=seeded_report.measured_baseline_boundary,
                metriplectic_boundary=seeded_report.measured_metriplectic_boundary,
            )
            outcomes.append(outcome)
            logger.event("metriplectic_stability_seed_outcome", **asdict(outcome))

        baseline_loss = {outcome.seed: outcome.baseline_endpoint_loss for outcome in outcomes}
        metriplectic_loss = {
            outcome.seed: outcome.metriplectic_endpoint_loss for outcome in outcomes
        }
        baseline_divergence = {
            outcome.seed: outcome.baseline_divergence_rate for outcome in outcomes
        }
        metriplectic_divergence = {
            outcome.seed: outcome.metriplectic_divergence_rate for outcome in outcomes
        }
        loss_comparison = paired_comparison(
            metriplectic_loss,
            baseline_loss,
            lower_is_better=True,
            n_boot=bootstrap_samples,
            seed=0,
        )
        divergence_comparison = paired_comparison(
            metriplectic_divergence,
            baseline_divergence,
            lower_is_better=True,
            n_boot=bootstrap_samples,
            seed=1,
        )
        if loss_comparison is None or divergence_comparison is None:
            raise AssertionError("validated multi-seed inputs did not produce paired statistics")
        verdict, verdict_reason = _paired_verdict(loss_comparison, divergence_comparison)
        report = StatisticalSweepReport(
            bead="bio_inspired_nanochat-0642.1.3.2",
            run_id=logger.run_id,
            seeds=seeds,
            stress_step_sizes=stress_step_sizes or (),
            outcomes=tuple(outcomes),
            endpoint_loss_comparison=loss_comparison,
            divergence_rate_comparison=divergence_comparison,
            verdict=verdict,
            verdict_reason=verdict_reason,
            report_path=str(report_path),
            events_path=str(output_dir / "events.jsonl"),
            registry_path=registry_path_str,
        )
        report_path.write_text(
            _dump_report_json(report) + "\n",
            encoding="utf-8",
        )
        if registry_path_str is not None:
            _append_statistical_records(report, cfg, registry_path_str)
        logger.event("metriplectic_stability_statistical_summary", **report.to_dict())
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
    proof = report.proof_obligation
    console.print(
        f"Proof obligation {'VERIFIED' if proof.verified else 'FAILED'}: "
        f"max |dE|={proof.max_abs_energy_drift:.3e}, "
        f"min dS={proof.min_entropy_production:.3e}, "
        f"max degeneracy residual={proof.max_degeneracy_residual:.3e}; "
        f"injected fallbacks={proof.fallback_injection.fallback_count}/"
        f"{proof.fallback_injection.steps}."
    )
    console.print(f"JSON report: {report.report_path}")
    console.print(f"Detailed events: {report.events_path}")


def render_statistical_report(
    report: StatisticalSweepReport,
    *,
    console: Console | None = None,
) -> None:
    """Render the paired multi-seed evidence and honest verdict through Rich."""
    console = console or Console()
    table = Table(title="Metriplectic paired multi-seed verdict")
    table.add_column("metric")
    table.add_column("mean Δ (GENERIC - baseline)", justify="right")
    table.add_column("bootstrap 95% CI", justify="right")
    table.add_column("paired-t p", justify="right")
    table.add_column("Wilcoxon p", justify="right")
    table.add_column("favorable", justify="right")
    for name, comparison in (
        ("endpoint loss", report.endpoint_loss_comparison),
        ("divergence rate", report.divergence_rate_comparison),
    ):
        table.add_row(
            name,
            f"{comparison.mean_delta:.6g}",
            f"[{comparison.delta_ci_low:.6g}, {comparison.delta_ci_high:.6g}]",
            f"{comparison.t_p_value:.3g}",
            f"{comparison.wilcoxon_p_value:.3g}",
            f"{comparison.n_favorable}/{comparison.n_pairs}",
        )
    console.print(table)
    color = {"positive": "green", "null": "yellow", "worse": "red"}[report.verdict]
    console.print(
        f"[{color}]VERDICT: {report.verdict.upper()}[/{color}] — {report.verdict_reason}"
    )
    console.print(f"Statistical report: {report.report_path}")
    if report.registry_path is not None:
        console.print(f"Result observations appended to: {report.registry_path}")


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
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[11, 23, 37, 53, 71, 89, 107, 131],
        help="matched seeds for the paired statistical verdict",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
        help="paired-bootstrap resamples",
    )
    parser.add_argument(
        "--registry-path",
        default=DEFAULT_REGISTRY,
        help="append-only committed results registry",
    )
    args = parser.parse_args(argv)
    cfg = StabilitySweepConfig(
        step_sizes=tuple(args.step_sizes)
        if args.step_sizes is not None
        else StabilitySweepConfig.step_sizes
    )
    report = run_stability_sweep(cfg, run_dir=args.run_dir)
    render_report(report)
    statistical_report = run_statistical_stability_sweep(
        cfg,
        seeds=tuple(args.seeds),
        bootstrap_samples=args.bootstrap_samples,
        run_dir=Path(args.run_dir) / "statistics",
        registry_path=args.registry_path,
    )
    render_statistical_report(statistical_report)
    return (
        0
        if report.leapfrog_reproduced
        and report.proof_obligation.verified
        and statistical_report.verdict == "positive"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
