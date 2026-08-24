"""Stats-backed falsification of the tropical selection certificate (bead ``0642.6.3.1``).

The experiment is deliberately scoped to finite affine score families, which is the exact domain
licensed by :mod:`bio_inspired_nanochat.tropical_certificate`.  It predeclares three claims:

1. soft attention converges to the exact hard readout as temperature decreases;
2. the safety-adjusted dual-norm radius is positive and never exceeds a dense black-box estimate of
   the first adversarial decision flip; and
3. the active-vertex fingerprint agrees with a frozen-value winner-lesion target, while ordinary
   one-layer attention rollout remains a dense post-hoc attribution at a distinct fixed temperature.

The named baselines are therefore ordinary softmax/attention rollout and a black-box angular ray
search with bisection.  The ray search is empirical evidence, not a replacement for the proof.  A
counterexample to an exact invariant produces an ``invalidated`` verdict; merely underpowered
paired evidence produces an honest ``null`` verdict.  Every temperature and seed outcome is written
through :class:`~bio_inspired_nanochat.run_logging.RunLogger` before the strict-JSON summary and
optional committed-registry observations are emitted.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat.eval_stats import (
    Aggregate,
    PairedResult,
    aggregate,
    paired_comparison,
)
from bio_inspired_nanochat.results_registry import (
    DEFAULT_REGISTRY,
    append_record,
    make_record,
)
from bio_inspired_nanochat.run_logging import RunLogger
from bio_inspired_nanochat.synaptic import SynapticConfig
from bio_inspired_nanochat.tropical_certificate import (
    CertificateScope,
    GeometryScope,
    InputNorm,
    TropicalCertificateMonitor,
    TropicalRoutingConfig,
    TropicalRoutingController,
    certify_selection_geometry,
    deterministic_argmax,
    global_lipschitz_certificate,
    temperature_gate,
    tropical_readout_or_baseline,
)


@dataclass(frozen=True)
class TropicalFalsificationConfig:
    """Predeclared sampling, threat-model, and decision thresholds."""

    # Held out from the exploratory development seeds and first executed after the protocol commit.
    seeds: tuple[int, ...] = (149, 167, 181, 199, 223, 241, 263, 281)
    temperatures: tuple[float, ...] = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
    n_choices: int = 6
    input_dimension: int = 2
    angle_samples: int = 4096
    binary_steps: int = 48
    interior_trials: int = 512
    search_radius: float = 2.0
    safety_fraction: float = 0.05
    min_certified_radius: float = 0.05
    max_attack_resolution_error: float = 1e-3
    attribution_temperature: float = 0.5
    bootstrap_samples: int = 10_000
    tie_tol: float = 1e-10

    def validate(self) -> None:
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must contain at least two unique values")
        if len(self.temperatures) < 2:
            raise ValueError("temperatures must contain a baseline and a low-temperature endpoint")
        if any(not math.isfinite(tau) or tau <= 0.0 for tau in self.temperatures):
            raise ValueError("temperatures must be finite and positive")
        if any(
            current <= following
            for current, following in zip(self.temperatures, self.temperatures[1:])
        ):
            raise ValueError("temperatures must be strictly decreasing")
        if self.n_choices < 3:
            raise ValueError("n_choices must be at least three")
        if self.input_dimension != 2:
            raise ValueError("input_dimension must be two for the exhaustive angular baseline")
        if self.angle_samples < 32:
            raise ValueError("angle_samples must be at least 32")
        if self.binary_steps < 8:
            raise ValueError("binary_steps must be at least eight")
        if self.interior_trials < 1:
            raise ValueError("interior_trials must be positive")
        if not math.isfinite(self.search_radius) or self.search_radius <= 0.0:
            raise ValueError("search_radius must be finite and positive")
        if not math.isfinite(self.safety_fraction) or not 0.0 < self.safety_fraction < 1.0:
            raise ValueError("safety_fraction must be finite and strictly between zero and one")
        if not math.isfinite(self.min_certified_radius) or self.min_certified_radius <= 0.0:
            raise ValueError("min_certified_radius must be finite and positive")
        if (
            not math.isfinite(self.max_attack_resolution_error)
            or self.max_attack_resolution_error <= 0.0
        ):
            raise ValueError("max_attack_resolution_error must be finite and positive")
        if not math.isfinite(self.attribution_temperature) or self.attribution_temperature <= 0.0:
            raise ValueError("attribution_temperature must be finite and positive")
        if self.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be positive")
        if not math.isfinite(self.tie_tol) or self.tie_tol < 0.0:
            raise ValueError("tie_tol must be finite and non-negative")


@dataclass(frozen=True)
class TropicalSeedOutcome:
    """All independently replayable measurements for one seed."""

    seed: int
    winner_id: str
    exactness_rate: float
    temperature_errors: tuple[float, ...]
    temperature_error_bounds: tuple[float, ...]
    monotonic_convergence: bool
    convergence_bound_satisfied: bool
    baseline_readout_l1_error: float
    low_temperature_readout_l1_error: float
    certified_radius: float
    raw_decision_radius: float
    empirical_adversarial_radius: float
    attack_resolution_error: float
    certified_to_empirical_ratio: float
    interior_trials: int
    interior_flips: int
    tropical_attribution_l1_error: float
    attention_rollout_attribution_l1_error: float
    high_temperature_gate_passed: bool
    low_temperature_gate_passed: bool
    high_temperature_fallback_identity: bool
    low_temperature_hard_authorized: bool


@dataclass(frozen=True)
class TropicalFalsificationReport:
    """Strict statistical evidence and a predeclared honest verdict."""

    bead: str
    run_id: str
    config: TropicalFalsificationConfig
    outcomes: tuple[TropicalSeedOutcome, ...]
    exactness: Aggregate
    radius_ratio: Aggregate
    radius_safety_margin: Aggregate
    convergence_comparison: PairedResult
    attribution_comparison: PairedResult
    verdict: str
    verdict_reason: str
    report_path: str
    events_path: str
    registry_path: str | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("convergence_comparison", "attribution_comparison"):
            if not math.isfinite(payload[key]["t_stat"]):
                payload[key]["t_stat"] = None
        return payload

    def assert_positive(self) -> None:
        if self.verdict != "positive":
            raise AssertionError(
                f"expected positive tropical verdict, got {self.verdict}: {self.verdict_reason}"
            )


@dataclass(frozen=True)
class _AffineFamily:
    x: np.ndarray
    slopes: np.ndarray
    offsets: np.ndarray
    scores: np.ndarray
    choice_ids: tuple[str, ...]
    values: np.ndarray


def _softmax(scores: np.ndarray, tau: float) -> np.ndarray:
    shifted = (np.asarray(scores, dtype=np.float64) - float(np.max(scores))) / tau
    weights = np.exp(np.clip(shifted, -745.0, 0.0))
    return weights / float(np.sum(weights))


def _oracle_argmax(
    scores: np.ndarray,
    choice_ids: tuple[str, ...],
    *,
    eligible: np.ndarray | None = None,
) -> tuple[int, str]:
    """Independent NumPy argmax oracle with the public lexicographic tie rule."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size != len(choice_ids):
        raise ValueError("scores and choice_ids must be aligned one-dimensional sequences")
    if not np.all(np.isfinite(values)):
        raise ValueError("oracle scores must be finite")
    if eligible is None:
        indices = np.arange(values.size)
    else:
        mask = np.asarray(eligible, dtype=np.bool_)
        if mask.shape != values.shape:
            raise ValueError("eligible mask must match scores")
        indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError("at least one oracle choice must be eligible")
    best_score = float(np.max(values[indices]))
    candidates = [int(index) for index in indices if values[index] == best_score]
    winner_index = min(candidates, key=lambda index: (choice_ids[index], index))
    return winner_index, choice_ids[winner_index]


def _sample_affine_family(config: TropicalFalsificationConfig, seed: int) -> _AffineFamily:
    """Sample a well-spread 2-D family with a unique, non-vacuous decision cell."""
    rng = np.random.default_rng(seed)
    rotation = float(rng.uniform(0.0, 2.0 * math.pi))
    angles = rotation + 2.0 * math.pi * np.arange(config.n_choices) / config.n_choices
    slopes = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float64)
    x = rng.normal(0.0, 0.25, size=config.input_dimension).astype(np.float64)
    winner = int(rng.integers(0, config.n_choices))
    target_scores = 1.0 - rng.uniform(0.40, 0.90, size=config.n_choices)
    target_scores[winner] = 1.0
    offsets = target_scores - slopes @ x
    scores = slopes @ x + offsets
    choice_ids = tuple(f"choice-{index:02d}" for index in range(config.n_choices))
    values = rng.normal(size=(config.n_choices, config.input_dimension + 1)).astype(np.float64)
    return _AffineFamily(
        x=x,
        slopes=slopes,
        offsets=offsets.astype(np.float64),
        scores=scores.astype(np.float64),
        choice_ids=choice_ids,
        values=values,
    )


def _winner_at(family: _AffineFamily, x: np.ndarray) -> tuple[int, str]:
    return _oracle_argmax(
        family.slopes @ np.asarray(x, dtype=np.float64) + family.offsets,
        family.choice_ids,
    )


def _winner_lesion_target(family: _AffineFamily, winner_index: int) -> np.ndarray:
    """Normalize frozen-readout changes caused by removing each eligible choice."""
    baseline = family.values[winner_index]
    effects = np.zeros(len(family.choice_ids), dtype=np.float64)
    for removed_index in range(len(family.choice_ids)):
        eligible = np.ones(len(family.choice_ids), dtype=np.bool_)
        eligible[removed_index] = False
        lesion_index, _ = _oracle_argmax(
            family.scores,
            family.choice_ids,
            eligible=eligible,
        )
        effects[removed_index] = float(
            np.linalg.norm(baseline - family.values[lesion_index], ord=2)
        )
    total = float(np.sum(effects))
    if not math.isfinite(total) or total <= 0.0:
        raise AssertionError("winner lesion did not change the frozen hard readout")
    return effects / total


def _empirical_adversarial_radius(
    family: _AffineFamily,
    *,
    winner_id: str,
    angle_samples: int,
    binary_steps: int,
    search_radius: float,
) -> float:
    """Find the nearest observed black-box flip over a dense angular ray sweep."""
    nearest: float | None = None
    angles = 2.0 * math.pi * (np.arange(angle_samples) + 0.5) / angle_samples
    for angle in angles:
        direction = np.array((math.cos(float(angle)), math.sin(float(angle))))
        if _winner_at(family, family.x + search_radius * direction)[1] == winner_id:
            continue
        low = 0.0
        high = search_radius
        for _ in range(binary_steps):
            midpoint = 0.5 * (low + high)
            if _winner_at(family, family.x + midpoint * direction)[1] == winner_id:
                low = midpoint
            else:
                high = midpoint
        nearest = high if nearest is None else min(nearest, high)
    if nearest is None:
        raise AssertionError("black-box search radius did not observe an adversarial decision flip")
    return nearest


def _count_interior_flips(
    family: _AffineFamily,
    *,
    winner_id: str,
    certified_radius: float,
    trials: int,
    seed: int,
) -> int:
    rng = np.random.default_rng(seed ^ 0x5EED5EED)
    flips = 0
    for _ in range(trials):
        direction = rng.normal(size=family.x.size)
        direction /= float(np.linalg.norm(direction))
        radius = float(rng.uniform(0.0, 0.999 * certified_radius))
        flips += _winner_at(family, family.x + radius * direction)[1] != winner_id
    return int(flips)


def _exercise_runtime_gate_and_fallback(
    config: TropicalFalsificationConfig,
    family: _AffineFamily,
    *,
    selection: Any,
    winner_index: int,
    logger: RunLogger,
) -> tuple[bool, bool]:
    """Drive the real controller from high-temperature fallback to certified hard readout."""
    controller = TropicalRoutingController(
        SynapticConfig(tropical_skeleton=True, barrier_strength=0.0),
        TropicalRoutingConfig(
            tau_start=config.temperatures[0],
            tau_min=config.temperatures[-1],
            anneal_steps=1,
            entry_windows=1,
        ),
        logger=logger,
    )
    monitor = TropicalCertificateMonitor(logger=logger)
    lipschitz = global_lipschitz_certificate(
        family.slopes,
        choice_ids=family.choice_ids,
        input_norm=InputNorm.L2,
    )
    values = np.eye(config.n_choices, dtype=np.float64)

    high_point = controller.schedule_point()
    high_record = monitor.record(
        step=0,
        layer="tropical_falsification.attention",
        head=0,
        certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
        selection=selection,
        lipschitz=lipschitz,
        temperature=temperature_gate(
            family.scores,
            certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
            tau=high_point.tau,
            choice_ids=family.choice_ids,
        ),
        pre_dropout=True,
        values_frozen=True,
        schedule_digest=high_point.digest,
    )
    high_decision = controller.observe(
        high_record,
        observed_barrier_strength=high_point.barrier_strength,
    )
    baseline = _softmax(family.scores, high_point.tau)
    high_readout = tropical_readout_or_baseline(
        baseline,
        values,
        family.scores,
        high_decision,
        choice_ids=family.choice_ids,
    )
    fallback_identity = bool(
        high_readout.value is baseline
        and not high_readout.used_hard_path
        and high_decision.used_baseline
    )

    low_point = controller.schedule_point()
    low_record = monitor.record(
        step=1,
        layer="tropical_falsification.attention",
        head=0,
        certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
        selection=selection,
        lipschitz=lipschitz,
        temperature=temperature_gate(
            family.scores,
            certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
            tau=low_point.tau,
            choice_ids=family.choice_ids,
        ),
        pre_dropout=True,
        values_frozen=True,
        schedule_digest=low_point.digest,
    )
    low_decision = controller.observe(
        low_record,
        observed_barrier_strength=low_point.barrier_strength,
    )
    low_baseline = _softmax(family.scores, low_point.tau)
    low_readout = tropical_readout_or_baseline(
        low_baseline,
        values,
        family.scores,
        low_decision,
        choice_ids=family.choice_ids,
    )
    hard_authorized = bool(
        low_decision.use_hard_path
        and low_readout.used_hard_path
        and low_readout.choice_index == winner_index
        and np.array_equal(low_readout.value, values[winner_index])
    )
    return fallback_identity, hard_authorized


def _run_seed(
    config: TropicalFalsificationConfig,
    seed: int,
    logger: RunLogger,
) -> TropicalSeedOutcome:
    family = _sample_affine_family(config, seed)
    winner_index, winner_id = _oracle_argmax(family.scores, family.choice_ids)
    runtime_winner = deterministic_argmax(family.scores, choice_ids=family.choice_ids)
    if runtime_winner != (winner_index, winner_id):
        raise AssertionError("runtime argmax disagreed with the independent oracle")
    certificate = certify_selection_geometry(
        family.x,
        family.slopes,
        family.offsets,
        choice_ids=family.choice_ids,
        input_norm=InputNorm.L2,
        scope=GeometryScope.EXACT_AFFINE,
        tie_tol=config.tie_tol,
        safety_fraction=config.safety_fraction,
        min_certified_radius=config.min_certified_radius,
    )
    geometry = certificate.geometry
    if not geometry.certified or geometry.certified_radius is None:
        raise AssertionError(f"constructed affine family was not certified: {geometry.reason}")
    gap = certificate.fingerprint.selection_gap
    if gap is None or gap <= config.tie_tol:
        raise AssertionError("constructed affine family did not have a finite unique-winner gap")

    one_hot = np.zeros(config.n_choices, dtype=np.float64)
    one_hot[winner_index] = 1.0
    errors: list[float] = []
    bounds: list[float] = []
    argmax_matches: list[bool] = []
    for index, tau in enumerate(config.temperatures):
        probabilities = _softmax(family.scores, tau)
        soft_index, soft_id = _oracle_argmax(probabilities, family.choice_ids)
        if deterministic_argmax(probabilities, choice_ids=family.choice_ids) != (
            soft_index,
            soft_id,
        ):
            raise AssertionError("runtime soft argmax disagreed with the independent oracle")
        error = float(np.linalg.norm(probabilities - one_hot, ord=1))
        bound = 2.0 * (config.n_choices - 1) * math.exp(-gap / tau)
        errors.append(error)
        bounds.append(bound)
        argmax_matches.append(soft_index == winner_index and soft_id == winner_id)
        logger.event(
            "tropical_temperature_observation",
            step=index,
            seed=seed,
            tau=tau,
            winner_id=winner_id,
            soft_argmax_id=soft_id,
            winner_mass=float(probabilities[winner_index]),
            readout_l1_error=error,
            theoretical_l1_bound=bound,
            bound_satisfied=error <= bound + 1e-12,
        )

    high_gate = temperature_gate(
        family.scores,
        certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
        tau=config.temperatures[0],
        choice_ids=family.choice_ids,
    )
    low_gate = temperature_gate(
        family.scores,
        certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
        tau=config.temperatures[-1],
        choice_ids=family.choice_ids,
    )
    empirical_radius = _empirical_adversarial_radius(
        family,
        winner_id=winner_id,
        angle_samples=config.angle_samples,
        binary_steps=config.binary_steps,
        search_radius=config.search_radius,
    )
    certified_radius = geometry.certified_radius
    raw_radius = geometry.raw_radius
    if raw_radius is None:
        raise AssertionError("certified exact-affine geometry did not expose a raw radius")
    interior_flips = _count_interior_flips(
        family,
        winner_id=winner_id,
        certified_radius=certified_radius,
        trials=config.interior_trials,
        seed=seed,
    )

    lesion_target = _winner_lesion_target(family, winner_index)
    tropical_attribution = np.zeros(config.n_choices, dtype=np.float64)
    active_id = certificate.fingerprint.active_ids[0]
    tropical_attribution[family.choice_ids.index(active_id)] = 1.0
    attention_rollout = _softmax(family.scores, config.attribution_temperature)
    tropical_attribution_error = float(
        np.linalg.norm(tropical_attribution - lesion_target, ord=1)
    )
    rollout_attribution_error = float(
        np.linalg.norm(attention_rollout - lesion_target, ord=1)
    )
    fallback_identity, hard_authorized = _exercise_runtime_gate_and_fallback(
        config,
        family,
        selection=certificate,
        winner_index=winner_index,
        logger=logger,
    )

    outcome = TropicalSeedOutcome(
        seed=seed,
        winner_id=winner_id,
        exactness_rate=sum(argmax_matches) / len(argmax_matches),
        temperature_errors=tuple(errors),
        temperature_error_bounds=tuple(bounds),
        monotonic_convergence=all(
            following <= current + 1e-12
            for current, following in itertools.pairwise(errors)
        ),
        convergence_bound_satisfied=all(
            error <= bound + 1e-12 for error, bound in zip(errors, bounds)
        ),
        baseline_readout_l1_error=errors[0],
        low_temperature_readout_l1_error=errors[-1],
        certified_radius=certified_radius,
        raw_decision_radius=raw_radius,
        empirical_adversarial_radius=empirical_radius,
        attack_resolution_error=empirical_radius - raw_radius,
        certified_to_empirical_ratio=certified_radius / empirical_radius,
        interior_trials=config.interior_trials,
        interior_flips=interior_flips,
        tropical_attribution_l1_error=tropical_attribution_error,
        attention_rollout_attribution_l1_error=rollout_attribution_error,
        high_temperature_gate_passed=bool(high_gate.passed),
        low_temperature_gate_passed=bool(low_gate.passed),
        high_temperature_fallback_identity=fallback_identity,
        low_temperature_hard_authorized=hard_authorized,
    )
    logger.event(
        "tropical_seed_outcome",
        fingerprint_digest=certificate.fingerprint.digest,
        lesion_target=lesion_target,
        high_temperature_gate_reason=high_gate.reason,
        low_temperature_gate_reason=low_gate.reason,
        **asdict(outcome),
    )
    return outcome


def _supports_lower(result: PairedResult) -> bool:
    return bool(
        result.mean_delta < 0.0
        and result.delta_ci_high < 0.0
        and result.t_p_value < 0.05
        and result.wilcoxon_p_value <= 0.05
    )


def _verdict(
    config: TropicalFalsificationConfig,
    outcomes: tuple[TropicalSeedOutcome, ...],
    convergence: PairedResult,
    attribution: PairedResult,
) -> tuple[str, str]:
    violations: list[str] = []
    for outcome in outcomes:
        if outcome.exactness_rate != 1.0:
            violations.append(f"seed {outcome.seed}: soft argmax mismatch")
        if not outcome.monotonic_convergence or not outcome.convergence_bound_satisfied:
            violations.append(f"seed {outcome.seed}: readout convergence bound failed")
        if outcome.high_temperature_gate_passed or not outcome.low_temperature_gate_passed:
            violations.append(f"seed {outcome.seed}: temperature gate control failed")
        if (
            not outcome.high_temperature_fallback_identity
            or not outcome.low_temperature_hard_authorized
        ):
            violations.append(f"seed {outcome.seed}: runtime fallback/hard transition failed")
        if outcome.certified_radius <= config.min_certified_radius:
            violations.append(f"seed {outcome.seed}: certified radius was vacuous")
        if outcome.certified_radius > outcome.empirical_adversarial_radius + 1e-10:
            violations.append(f"seed {outcome.seed}: certificate exceeded empirical flip radius")
        if outcome.attack_resolution_error < -1e-10:
            violations.append(f"seed {outcome.seed}: attack underestimated the exact boundary")
        if outcome.attack_resolution_error > config.max_attack_resolution_error:
            violations.append(f"seed {outcome.seed}: attack did not resolve the exact boundary")
        if outcome.interior_flips:
            violations.append(f"seed {outcome.seed}: adversarial flip occurred inside certificate")
        if outcome.tropical_attribution_l1_error > 1e-12:
            violations.append(f"seed {outcome.seed}: active-vertex attribution was not exact")
    if violations:
        return "invalidated", "; ".join(violations[:4])
    if _supports_lower(convergence) and _supports_lower(attribution):
        return (
            "positive",
            (
                "all exact invariants held; low-temperature readout and the active-vertex "
                "fingerprint beat their matched soft baselines; every certified radius was "
                "non-vacuous, remained inside the independent attack radius, and passed the "
                "attack-resolution control"
            ),
        )
    return (
        "null",
        (
            "no exact claim was falsified, but the paired evidence did not clear every "
            "predeclared threshold"
        ),
    )


def _append_registry_records(
    report: TropicalFalsificationReport,
    registry_path: str,
) -> None:
    comparison_note = (
        f"experiment=tropical_falsification; verdict={report.verdict}; "
        f"convergence_delta={report.convergence_comparison.mean_delta:.17g}; "
        f"convergence_ci=[{report.convergence_comparison.delta_ci_low:.17g},"
        f"{report.convergence_comparison.delta_ci_high:.17g}]; "
        f"attribution_delta={report.attribution_comparison.mean_delta:.17g}; "
        f"attribution_ci=[{report.attribution_comparison.delta_ci_low:.17g},"
        f"{report.attribution_comparison.delta_ci_high:.17g}]; "
        f"radius_ratio_ci=[{report.radius_ratio.ci_low:.17g},{report.radius_ratio.ci_high:.17g}]; "
        f"artifact={report.report_path}"
    )
    for outcome in report.outcomes:
        record = make_record(
            "eval",
            {
                "tropical_exactness_rate": outcome.exactness_rate,
                "tropical_readout_l1_error": outcome.low_temperature_readout_l1_error,
                "soft_readout_l1_error": outcome.baseline_readout_l1_error,
                "tropical_certified_radius": outcome.certified_radius,
                "empirical_adversarial_radius": outcome.empirical_adversarial_radius,
                "tropical_radius_tightness": outcome.certified_to_empirical_ratio,
                "tropical_attribution_l1_error": outcome.tropical_attribution_l1_error,
                "attention_rollout_attribution_l1_error": (
                    outcome.attention_rollout_attribution_l1_error
                ),
            },
            run_id=f"{report.run_id}-tropical-s{outcome.seed}",
            config=report.config,
            seed=outcome.seed,
            verdict=report.verdict,
            eligible_for_best=report.verdict == "positive",
            notes=comparison_note,
        )
        append_record(record, registry_path)


def run_tropical_falsification(
    config: TropicalFalsificationConfig | None = None,
    *,
    run_dir: str | Path | None = None,
    registry_path: str | Path | None = None,
) -> TropicalFalsificationReport:
    """Run the matched multi-seed experiment and persist strict audit evidence."""
    config = config or TropicalFalsificationConfig()
    config.validate()
    run_id = uuid.uuid4().hex[:12]
    output_dir = (
        Path(run_dir)
        if run_dir is not None
        else Path("runs/e2e/tropical_falsification") / run_id
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to mix tropical run artifacts in {output_dir}")
    report_path = output_dir / "statistics.json"
    registry_path_str = str(registry_path) if registry_path is not None else None

    with RunLogger(
        output_dir,
        name="tropical_falsification",
        run_id=run_id,
        console=False,
        provenance={
            "bead": "bio_inspired_nanochat-0642.6.3.1",
            "config": asdict(config),
            "named_baselines": (
                "ordinary tau=1 soft readout",
                f"attention rollout at tau={config.attribution_temperature}",
                "dense angular adversarial search with an independent NumPy argmax oracle",
                "frozen-value winner-lesion selection target",
            ),
        },
    ) as logger:
        outcomes = tuple(_run_seed(config, seed, logger) for seed in config.seeds)
        baseline_error = {
            outcome.seed: outcome.baseline_readout_l1_error for outcome in outcomes
        }
        low_temperature_error = {
            outcome.seed: outcome.low_temperature_readout_l1_error for outcome in outcomes
        }
        rollout_error = {
            outcome.seed: outcome.attention_rollout_attribution_l1_error
            for outcome in outcomes
        }
        tropical_attribution_error = {
            outcome.seed: outcome.tropical_attribution_l1_error for outcome in outcomes
        }
        convergence = paired_comparison(
            low_temperature_error,
            baseline_error,
            lower_is_better=True,
            n_boot=config.bootstrap_samples,
            seed=0,
        )
        attribution = paired_comparison(
            tropical_attribution_error,
            rollout_error,
            lower_is_better=True,
            n_boot=config.bootstrap_samples,
            seed=1,
        )
        if convergence is None or attribution is None:
            raise AssertionError("validated multi-seed inputs did not produce paired statistics")
        exactness = aggregate([outcome.exactness_rate for outcome in outcomes])
        radius_ratio = aggregate(
            [outcome.certified_to_empirical_ratio for outcome in outcomes]
        )
        radius_margin = aggregate(
            [
                outcome.empirical_adversarial_radius - outcome.certified_radius
                for outcome in outcomes
            ]
        )
        verdict, reason = _verdict(
            config,
            outcomes,
            convergence,
            attribution,
        )
        report = TropicalFalsificationReport(
            bead="bio_inspired_nanochat-0642.6.3.1",
            run_id=logger.run_id,
            config=config,
            outcomes=outcomes,
            exactness=exactness,
            radius_ratio=radius_ratio,
            radius_safety_margin=radius_margin,
            convergence_comparison=convergence,
            attribution_comparison=attribution,
            verdict=verdict,
            verdict_reason=reason,
            report_path=str(report_path),
            events_path=str(output_dir / "events.jsonl"),
            registry_path=registry_path_str,
        )
        report_path.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if registry_path_str is not None:
            _append_registry_records(report, registry_path_str)
        logger.event("tropical_falsification_summary", **report.to_dict())
    return report


def render_report(
    report: TropicalFalsificationReport,
    *,
    console: Console | None = None,
) -> None:
    """Render the seed-level falsification evidence through Rich."""
    console = console or Console()
    table = Table(title="Tropical certificate falsification")
    table.add_column("seed", justify="right")
    table.add_column("soft L1 tau=1", justify="right")
    table.add_column("soft L1 tau=min", justify="right")
    table.add_column("cert radius", justify="right")
    table.add_column("empirical flip", justify="right")
    table.add_column("tightness", justify="right")
    table.add_column("rollout attr L1", justify="right")
    for outcome in report.outcomes:
        table.add_row(
            str(outcome.seed),
            f"{outcome.baseline_readout_l1_error:.4g}",
            f"{outcome.low_temperature_readout_l1_error:.4g}",
            f"{outcome.certified_radius:.4g}",
            f"{outcome.empirical_adversarial_radius:.4g}",
            f"{outcome.certified_to_empirical_ratio:.4f}",
            f"{outcome.attention_rollout_attribution_l1_error:.4g}",
        )
    console.print(table)
    color = {"positive": "green", "null": "yellow", "invalidated": "red"}[report.verdict]
    console.print(f"[{color}]VERDICT: {report.verdict.upper()}[/{color}] — {report.verdict_reason}")
    console.print(
        "Radius tightness mean/95% CI: "
        f"{report.radius_ratio.mean:.5f} "
        f"[{report.radius_ratio.ci_low:.5f}, {report.radius_ratio.ci_high:.5f}]"
    )
    console.print(f"Statistical report: {report.report_path}")
    console.print(f"Detailed events: {report.events_path}")
    if report.registry_path is not None:
        console.print(f"Result observations appended to: {report.registry_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Falsify the affine tropical certificate")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="fresh output directory (default: a unique run-ID subdirectory)",
    )
    parser.add_argument("--registry-path", default=DEFAULT_REGISTRY)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--angle-samples", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    args = parser.parse_args(argv)
    config = TropicalFalsificationConfig()
    if args.seeds is not None:
        config = replace(config, seeds=tuple(args.seeds))
    if args.angle_samples is not None:
        config = replace(config, angle_samples=args.angle_samples)
    if args.bootstrap_samples is not None:
        config = replace(config, bootstrap_samples=args.bootstrap_samples)
    report = run_tropical_falsification(
        config,
        run_dir=args.run_dir,
        registry_path=args.registry_path,
    )
    render_report(report)
    return 0 if report.verdict == "positive" else 1


if __name__ == "__main__":
    raise SystemExit(main())
