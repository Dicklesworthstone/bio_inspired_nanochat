"""Fail-closed live certificate bundle and model-card generator (bead ``r00r.7``).

The project has several deliberately narrow runtime certificates.  This module composes them
without widening their claims:

* metriplectic stability is accepted only from a non-empty, fallback-free torch runtime trace;
* cusp retention is recomputed from the exact live :class:`SynapticConfig`;
* predictive calibration retains complete per-head measurements and recomputes a digest-bound,
  fixed-policy multi-seed statistical gate;
* tropical robustness retains its exact protected scope, threat norm, and binding digests; and
* the A/E/F claims must pass the timescale-separation composition harness together.

A refused bundle is still rendered. That is intentional: operators receive a model card stating
the failed assumptions and deterministic fallbacks, while :meth:`GuaranteeBundle.require_deployable`
and the CLI's default exit policy refuse bounded deployment authorization. ``--allow-uncertified``
is an explicit report-only opt-out from the nonzero exit status, never from the card verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat.ablation_registry import (
    MAX_SAFE_INTEGER,
    SUPPORTED_SYNAPTIC_GRANULARITIES,
    synaptic_config_schema_errors,
    validate_config,
)
from bio_inspired_nanochat.checkpoint_manager import config_hash
from bio_inspired_nanochat.composition import (
    composition_eligibility,
    pairwise_compatible,
)
from bio_inspired_nanochat.cusp_certificate import certify_retention
from bio_inspired_nanochat.eval_stats import (
    bootstrap_ci,
    paired_t_test,
    wilcoxon_signed_rank,
)
from bio_inspired_nanochat.metriplectic_integrator import (
    GuardThresholds,
    LyapunovMonitor,
    TorchStepRecord,
)
from bio_inspired_nanochat.run_logging import RunLogger
from bio_inspired_nanochat.stochastic_thermo import (
    HeadPredictiveThermoEvidence,
    PredictiveEvidencePolicy,
    PredictiveEvidenceProvenance,
    PredictiveThermoEvidence,
    predictive_distribution_verdict,
)
from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticGranularity
from bio_inspired_nanochat.tropical_certificate import (
    CertificateScope,
    GeometryScope,
    InputNorm,
    TropicalCertificateMonitor,
    TropicalCertificateRecord,
)

SCHEMA_VERSION = 1
_REQUIRED_COMPOSITION_THRUSTS = ("A", "E", "F")
_REQUIRED_GATE_KEYS = (
    "provenance",
    "metriplectic_stability",
    "cusp_retention",
    "predictive_calibration",
    "tropical_robustness",
    "composition",
)
_SUPPORTED_PREDICTIVE_MODES = frozenset({"straight_through", "gumbel_sigmoid_ste"})
_PREDICTIVE_ALPHA = 0.05
_PREDICTIVE_BOOTSTRAP_SAMPLES = 10_000
_PREDICTIVE_BOOTSTRAP_SEED = 20260824
_MAX_PREDICTIVE_SEEDS = 256
_PREDICTIVE_DEPLOYMENT_SEEDS = (11, 23, 37, 41, 53, 67)
_TARGET_ECE_MAX = 0.10
_TARGET_OOD_AUROC_MIN = 0.70
_LIVE_FT_SCOPE = "one_step_local_detailed_balance"
_LIVE_CROOKS_TOLERANCE = 0.25
_LIVE_INTEGRAL_FT_TOLERANCE = 0.04
_LIVE_FT_TRAJECTORIES = 80_000
_LIVE_FT_MIN_COUNT = 100
_LIVE_TUR_POOL_SIZE = 6
_LIVE_TUR_SCOPE = (
    "classic_continuous_time_tur_on_exact_one_step_paired_binomial_moments"
)
_MAX_CUSP_EPS = SynapticConfig().cusp_eps_max
_MAX_COMPOSITION_EPS = 0.5
_MAX_EVIDENCE_BYTES = 16 * 1024 * 1024
_HEX_16_RE = re.compile(r"^[0-9a-f]{16}$")
_HEX_40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_TRUSTED_STABILITY_THRESHOLDS = GuardThresholds()


class CertificationRefused(RuntimeError):
    """Raised when a caller requests deployment certification from a failed bundle."""


class EventLogger(Protocol):
    """Subset of :class:`RunLogger` used for structured bundle evidence."""

    def event(
        self,
        event: str,
        *,
        level: str = "info",
        step: int | None = None,
        **fields: Any,
    ) -> dict[str, Any]: ...


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _exact_keys(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    """Reject both hidden defaults and unrecognized fields at an evidence boundary."""
    missing = sorted(expected - set(payload))
    unknown = sorted(set(payload) - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        raise ValueError(f"{name} schema mismatch ({'; '.join(details)})")


def _required_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a JSON boolean")
    return value


def _required_int(
    value: object,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = MAX_SAFE_INTEGER,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a JSON integer")
    result = value
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _required_finite_float(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a JSON number")
    if type(value) is int and abs(value) > MAX_SAFE_INTEGER:
        raise ValueError(
            f"{name} integer input must lie within "
            f"[-{MAX_SAFE_INTEGER}, {MAX_SAFE_INTEGER}]"
        )
    try:
        result = float(cast(float | int, value))
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _optional_finite_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    return _required_finite_float(value, name)


def _numeric_or_none(value: object, name: str) -> None:
    """Validate an in-process measurement while allowing non-finite refusal evidence."""
    if value is not None and type(value) not in (int, float):
        raise TypeError(f"{name} must be a number or None")
    if type(value) is int and abs(value) > MAX_SAFE_INTEGER:
        raise ValueError(
            f"{name} integer input must lie within "
            f"[-{MAX_SAFE_INTEGER}, {MAX_SAFE_INTEGER}]"
        )


def _required_string_array(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array")
    return tuple(_nonempty(item, f"{name} item") for item in value)


def _required_number_array(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array")
    return cast(list[object], value)


def _predictive_sites_from_json(
    value: object, name: str
) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array")
    sites: list[tuple[str, int]] = []
    for index, item in enumerate(value):
        if not isinstance(item, list) or len(item) != 2:
            raise TypeError(f"{name}[{index}] must be [layer, head]")
        sites.append(
            (
                _nonempty(item[0], f"{name}[{index}] layer"),
                _required_int(item[1], f"{name}[{index}] head"),
            )
        )
    return tuple(sites)


def _strict_json_loads(text: str) -> Mapping[str, Any]:
    """Load one strict JSON object, rejecting duplicates and non-standard constants."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value!r} is forbidden")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    payload = json.loads(
        text,
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )
    return _mapping(payload, "evidence")


def _canonical_digest(value: Any) -> str:
    blob = json.dumps(
        _json_safe(value),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _predictive_policy_from_dict(payload: Mapping[str, Any]) -> PredictiveEvidencePolicy:
    expected = {field.name for field in fields(PredictiveEvidencePolicy)}
    _exact_keys(payload, expected, "predictive policy")
    return PredictiveEvidencePolicy(
        min_samples=_required_int(payload.get("min_samples"), "policy.min_samples"),
        min_tested_fraction=_required_finite_float(
            payload.get("min_tested_fraction"),
            "policy.min_tested_fraction",
            minimum=0.0,
            maximum=1.0,
        ),
        min_symmetric_bins=_required_int(
            payload.get("min_symmetric_bins"), "policy.min_symmetric_bins"
        ),
        crooks_bins=_required_int(payload.get("crooks_bins"), "policy.crooks_bins"),
        crooks_min_count=_required_int(
            payload.get("crooks_min_count"), "policy.crooks_min_count"
        ),
        crooks_tolerance=_required_finite_float(
            payload.get("crooks_tolerance"),
            "policy.crooks_tolerance",
            minimum=0.0,
        ),
        min_tur_bound_ratio=_required_finite_float(
            payload.get("min_tur_bound_ratio"),
            "policy.min_tur_bound_ratio",
            minimum=0.0,
        ),
        max_events_per_head=_required_int(
            payload.get("max_events_per_head"), "policy.max_events_per_head"
        ),
    )


def _predictive_head_from_dict(
    payload: Mapping[str, Any],
) -> HeadPredictiveThermoEvidence:
    expected = {field.name for field in fields(HeadPredictiveThermoEvidence)}
    _exact_keys(payload, expected, "predictive head")
    return HeadPredictiveThermoEvidence(
        layer_address=_nonempty(payload.get("layer_address"), "head.layer_address"),
        head_index=_required_int(payload.get("head_index"), "head.head_index"),
        sampling_modes=_required_string_array(
            payload.get("sampling_modes"), "head.sampling_modes"
        ),
        sample_count=_required_int(payload.get("sample_count"), "head.sample_count"),
        observed_events=_required_int(
            payload.get("observed_events"), "head.observed_events"
        ),
        tested_events=_required_int(payload.get("tested_events"), "head.tested_events"),
        retained_events=_required_int(
            payload.get("retained_events"), "head.retained_events"
        ),
        degenerate_events=_required_int(
            payload.get("degenerate_events"), "head.degenerate_events"
        ),
        tested_fraction=_required_finite_float(
            payload.get("tested_fraction"),
            "head.tested_fraction",
            minimum=0.0,
            maximum=1.0,
        ),
        symmetric_bins=_required_int(
            payload.get("symmetric_bins"), "head.symmetric_bins"
        ),
        crooks_residual=_optional_finite_float(
            payload.get("crooks_residual"), "head.crooks_residual"
        ),
        tur_relative_variance=_optional_finite_float(
            payload.get("tur_relative_variance"), "head.tur_relative_variance"
        ),
        tur_entropy_bound=_optional_finite_float(
            payload.get("tur_entropy_bound"), "head.tur_entropy_bound"
        ),
        tur_bound_ratio=_optional_finite_float(
            payload.get("tur_bound_ratio"), "head.tur_bound_ratio"
        ),
        finite=_required_bool(payload.get("finite"), "head.finite"),
        passed=_required_bool(payload.get("passed"), "head.passed"),
        refusal_reasons=_required_string_array(
            payload.get("refusal_reasons"), "head.refusal_reasons"
        ),
    )


def _predictive_evidence_from_dict(
    payload: Mapping[str, Any],
) -> PredictiveThermoEvidence:
    expected = {field.name for field in fields(PredictiveThermoEvidence)}
    _exact_keys(payload, expected, "predictive evidence")
    provenance_payload = _mapping(payload.get("provenance"), "predictive provenance")
    _exact_keys(
        provenance_payload,
        {
            "run_id",
            "checkpoint_id",
            "synaptic_config_hash",
            "config_hash",
            "rng_seed",
        },
        "predictive provenance",
    )
    heads_payload = payload.get("heads")
    if not isinstance(heads_payload, list):
        raise TypeError("predictive evidence heads must be a JSON array")
    return PredictiveThermoEvidence(
        provenance=PredictiveEvidenceProvenance(
            run_id=_nonempty(provenance_payload.get("run_id"), "predictive run_id"),
            checkpoint_id=_nonempty(
                provenance_payload.get("checkpoint_id"), "predictive checkpoint_id"
            ),
            synaptic_config_hash=_nonempty(
                provenance_payload.get("synaptic_config_hash"),
                "predictive synaptic_config_hash",
            ),
            config_hash=_nonempty(
                provenance_payload.get("config_hash"), "predictive config_hash"
            ),
            rng_seed=_required_int(
                provenance_payload.get("rng_seed"), "predictive rng_seed"
            ),
        ),
        policy=_predictive_policy_from_dict(
            _mapping(payload.get("policy"), "predictive policy")
        ),
        heads=tuple(
            _predictive_head_from_dict(_mapping(item, "predictive head"))
            for item in heads_payload
        ),
        observed_events=_required_int(
            payload.get("observed_events"), "predictive observed_events"
        ),
        tested_events=_required_int(
            payload.get("tested_events"), "predictive tested_events"
        ),
        retained_events=_required_int(
            payload.get("retained_events"), "predictive retained_events"
        ),
        degenerate_events=_required_int(
            payload.get("degenerate_events"), "predictive degenerate_events"
        ),
        tested_fraction=_required_finite_float(
            payload.get("tested_fraction"),
            "predictive tested_fraction",
            minimum=0.0,
            maximum=1.0,
        ),
        fresh=_required_bool(payload.get("fresh"), "predictive fresh"),
        local_gates_passed=_required_bool(
            payload.get("local_gates_passed"), "predictive local_gates_passed"
        ),
        multi_seed_statistics_passed=_required_bool(
            payload.get("multi_seed_statistics_passed"),
            "predictive multi_seed_statistics_passed",
        ),
        predictive_distribution_claim=_required_bool(
            payload.get("predictive_distribution_claim"),
            "predictive predictive_distribution_claim",
        ),
        calibration_mode=_nonempty(
            payload.get("calibration_mode"), "predictive calibration_mode"
        ),
        refusal_reasons=_required_string_array(
            payload.get("refusal_reasons"), "predictive refusal_reasons"
        ),
    )


def _finite(value: float | None) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _failures(*checks: tuple[bool, str]) -> tuple[str, ...]:
    return tuple(message for passed, message in checks if not passed)


def _json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats so refusal artifacts remain strict JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _contains_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, Mapping):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_nonfinite(item) for item in value)
    return False


@dataclass(frozen=True)
class ModelIdentity:
    """Identity every evidence source must match before claims may compose."""

    run_id: str
    checkpoint_id: str
    config_hash: str
    predictive_config_hash: str
    git_sha: str

    def __post_init__(self) -> None:
        _nonempty(self.run_id, "run_id")
        if not _HEX_64_RE.fullmatch(self.checkpoint_id):
            raise ValueError("checkpoint_id must be the lowercase SHA-256 of the model artifact")
        if not _HEX_16_RE.fullmatch(self.config_hash):
            raise ValueError("config_hash must be the 16-character normalized config digest")
        if not _HEX_64_RE.fullmatch(self.predictive_config_hash):
            raise ValueError(
                "predictive_config_hash must be the full predictive protocol SHA-256"
            )
        if not _HEX_40_RE.fullmatch(self.git_sha):
            raise ValueError("git_sha must be a full 40-character lowercase revision digest")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ModelIdentity:
        _exact_keys(
            payload,
            {
                "run_id",
                "checkpoint_id",
                "config_hash",
                "predictive_config_hash",
                "git_sha",
            },
            "identity",
        )
        return cls(
            run_id=_nonempty(payload.get("run_id"), "identity.run_id"),
            checkpoint_id=_nonempty(
                payload.get("checkpoint_id"), "identity.checkpoint_id"
            ),
            config_hash=_nonempty(payload.get("config_hash"), "identity.config_hash"),
            predictive_config_hash=_nonempty(
                payload.get("predictive_config_hash"),
                "identity.predictive_config_hash",
            ),
            git_sha=_nonempty(payload.get("git_sha"), "identity.git_sha"),
        )


@dataclass(frozen=True)
class StabilityObservation:
    """Raw aggregate from a live :class:`LyapunovMonitor`."""

    identity: ModelIdentity
    source: str
    steps: int
    max_energy_drift: float | None
    min_entropy_production: float | None
    max_degeneracy_residual: float | None
    max_free_energy_delta: float | None
    n_fallbacks: int
    lyapunov_ok: bool
    thresholds: GuardThresholds
    _runtime_verified: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.source not in {"torch_runtime", "numpy_reference"}:
            raise ValueError("stability source must be torch_runtime or numpy_reference")
        _required_int(self.steps, "stability.steps")
        _required_int(self.n_fallbacks, "stability.n_fallbacks")
        _required_bool(self.lyapunov_ok, "stability.lyapunov_ok")
        for name in (
            "max_energy_drift",
            "min_entropy_production",
            "max_degeneracy_residual",
            "max_free_energy_delta",
        ):
            _numeric_or_none(getattr(self, name), f"stability.{name}")
        if not isinstance(self.thresholds, GuardThresholds):
            raise TypeError("stability.thresholds must be GuardThresholds")
        for name in ("eps_E", "eps_S", "eps_D"):
            value = getattr(self.thresholds, name)
            _required_finite_float(value, f"stability.thresholds.{name}", minimum=0.0)

    @classmethod
    def from_monitor(
        cls,
        identity: ModelIdentity,
        monitor: LyapunovMonitor,
        *,
        thresholds: GuardThresholds | None = None,
    ) -> StabilityObservation:
        summary = monitor.summary()
        free_energy = [record.F for record in monitor.records]
        free_energy_deltas = [
            free_energy[index + 1] - free_energy[index]
            for index in range(len(free_energy) - 1)
        ]
        return cls(
            identity=identity,
            source="numpy_reference",
            steps=int(summary.get("steps", 0)),
            max_energy_drift=(
                float(summary["max_energy_drift"])
                if "max_energy_drift" in summary
                else None
            ),
            min_entropy_production=(
                float(summary["min_entropy_production"])
                if "min_entropy_production" in summary
                else None
            ),
            max_degeneracy_residual=(
                float(summary["max_degeneracy_residual"])
                if "max_degeneracy_residual" in summary
                else None
            ),
            max_free_energy_delta=(
                float(max(free_energy_deltas)) if free_energy_deltas else None
            ),
            n_fallbacks=int(summary.get("n_fallbacks", 0)),
            lyapunov_ok=bool(summary.get("lyapunov_ok", False)),
            thresholds=thresholds or GuardThresholds(),
        )

    @classmethod
    def from_torch_records(
        cls,
        identity: ModelIdentity,
        records: Sequence[TorchStepRecord],
        *,
        thresholds: GuardThresholds | None = None,
    ) -> StabilityObservation:
        """Aggregate records captured from the actual torch-native guarded recurrence."""
        threshold = thresholds or GuardThresholds()
        steps = 0
        fallbacks = 0
        energy: list[float] = []
        entropy: list[float] = []
        degeneracy: list[float] = []
        free_energy_delta: list[float] = []
        for record in records:
            if record.energy_drift.numel() == 0:
                continue
            steps += record.energy_drift.numel()
            fallbacks += int(record.fallback_mask.detach().sum().item())
            energy.append(float(record.energy_drift.detach().abs().max().item()))
            entropy.append(float(record.entropy_production.detach().min().item()))
            degeneracy.append(
                max(
                    float(record.res_L_gradS.detach().abs().max().item()),
                    float(record.res_M_gradE.detach().abs().max().item()),
                )
            )
            free_energy_delta.append(
                float(record.free_energy_delta.detach().max().item())
            )
        finite_deltas = bool(free_energy_delta) and all(
            math.isfinite(value) for value in free_energy_delta
        )
        observation = cls(
            identity=identity,
            source="torch_runtime",
            steps=steps,
            max_energy_drift=max(energy) if energy else None,
            min_entropy_production=min(entropy) if entropy else None,
            max_degeneracy_residual=max(degeneracy) if degeneracy else None,
            max_free_energy_delta=(
                max(free_energy_delta) if free_energy_delta else None
            ),
            n_fallbacks=fallbacks,
            lyapunov_ok=finite_deltas
            and max(free_energy_delta) <= threshold.eps_E,
            thresholds=threshold,
        )
        object.__setattr__(observation, "_runtime_verified", True)
        return observation

    def to_manifest_dict(self) -> dict[str, Any]:
        """Serialize readings without pretending an offline manifest retains live provenance."""
        return {
            field.name: asdict(self)[field.name]
            for field in fields(self)
            if field.name != "_runtime_verified"
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StabilityObservation:
        _exact_keys(
            payload,
            {
                "identity",
                "source",
                "steps",
                "max_energy_drift",
                "min_entropy_production",
                "max_degeneracy_residual",
                "max_free_energy_delta",
                "n_fallbacks",
                "lyapunov_ok",
                "thresholds",
            },
            "stability",
        )
        threshold_payload = _mapping(payload.get("thresholds"), "stability.thresholds")
        _exact_keys(
            threshold_payload,
            {"eps_E", "eps_S", "eps_D"},
            "stability.thresholds",
        )
        return cls(
            identity=ModelIdentity.from_dict(
                _mapping(payload.get("identity"), "stability.identity")
            ),
            source=_nonempty(payload.get("source"), "stability.source"),
            steps=_required_int(payload.get("steps"), "stability.steps"),
            max_energy_drift=_optional_finite_float(
                payload.get("max_energy_drift"), "stability.max_energy_drift"
            ),
            min_entropy_production=_optional_finite_float(
                payload.get("min_entropy_production"),
                "stability.min_entropy_production",
            ),
            max_degeneracy_residual=_optional_finite_float(
                payload.get("max_degeneracy_residual"),
                "stability.max_degeneracy_residual",
            ),
            max_free_energy_delta=_optional_finite_float(
                payload.get("max_free_energy_delta"),
                "stability.max_free_energy_delta",
            ),
            n_fallbacks=_required_int(
                payload.get("n_fallbacks"), "stability.n_fallbacks"
            ),
            lyapunov_ok=_required_bool(
                payload.get("lyapunov_ok"), "stability.lyapunov_ok"
            ),
            thresholds=GuardThresholds(
                eps_E=_required_finite_float(
                    threshold_payload.get("eps_E"),
                    "stability.thresholds.eps_E",
                    minimum=0.0,
                ),
                eps_S=_required_finite_float(
                    threshold_payload.get("eps_S"),
                    "stability.thresholds.eps_S",
                    minimum=0.0,
                ),
                eps_D=_required_finite_float(
                    threshold_payload.get("eps_D"),
                    "stability.thresholds.eps_D",
                    minimum=0.0,
                ),
            ),
        )


@dataclass(frozen=True)
class PredictiveSeedObservation:
    """One complete live predictive artifact plus its canonical content digest."""

    evidence: PredictiveThermoEvidence
    artifact_sha256: str

    def __post_init__(self) -> None:
        parsed = _predictive_evidence_from_dict(
            _mapping(_json_safe(asdict(self.evidence)), "predictive evidence")
        )
        if parsed != self.evidence:
            raise ValueError("predictive evidence does not round-trip through the strict schema")
        if not _HEX_64_RE.fullmatch(self.artifact_sha256):
            raise ValueError("predictive artifact_sha256 must be a lowercase SHA-256")
        if not hmac.compare_digest(
            self.artifact_sha256, _canonical_digest(asdict(self.evidence))
        ):
            raise ValueError("predictive artifact_sha256 does not match the evidence content")

    @property
    def provenance(self) -> PredictiveEvidenceProvenance:
        return self.evidence.provenance

    @classmethod
    def from_evidence(
        cls, evidence: PredictiveThermoEvidence
    ) -> PredictiveSeedObservation:
        return cls(
            evidence=evidence,
            artifact_sha256=_canonical_digest(asdict(evidence)),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PredictiveSeedObservation:
        _exact_keys(payload, {"evidence", "artifact_sha256"}, "predictive seed")
        return cls(
            evidence=_predictive_evidence_from_dict(
                _mapping(payload.get("evidence"), "predictive seed evidence")
            ),
            artifact_sha256=_nonempty(
                payload.get("artifact_sha256"), "predictive artifact_sha256"
            ),
        )


@dataclass(frozen=True)
class TargetCalibrationObservation:
    """Fixed point-estimate calibration gate for the explicit deployed target report."""

    artifact_sha256: str
    target_artifact_sha256: str
    target_run_id: str
    target_checkpoint_id: str
    target_synaptic_config_hash: str
    target_experiment_config_hash: str
    target_rng_seed: int
    evaluation_distribution: str
    evaluation_predictions_per_split: int
    thermo_ece: float
    thermo_ood_auroc: float
    softmax_ece: float
    softmax_ood_auroc: float
    mc_dropout_ece: float
    mc_dropout_ood_auroc: float
    passed: bool

    def __post_init__(self) -> None:
        if not _HEX_64_RE.fullmatch(self.artifact_sha256):
            raise ValueError(
                "target calibration artifact_sha256 must be a lowercase SHA-256"
            )
        if not _HEX_64_RE.fullmatch(self.target_artifact_sha256):
            raise ValueError(
                "target calibration target_artifact_sha256 must be a lowercase SHA-256"
            )
        _nonempty(self.target_run_id, "target calibration target_run_id")
        if not _HEX_64_RE.fullmatch(self.target_checkpoint_id):
            raise ValueError(
                "target calibration target_checkpoint_id must be a lowercase SHA-256"
            )
        if not _HEX_16_RE.fullmatch(self.target_synaptic_config_hash):
            raise ValueError(
                "target calibration target_synaptic_config_hash must be normalized"
            )
        if not _HEX_64_RE.fullmatch(self.target_experiment_config_hash):
            raise ValueError(
                "target calibration target_experiment_config_hash must be a lowercase SHA-256"
            )
        _required_int(self.target_rng_seed, "target calibration target_rng_seed")
        _nonempty(
            self.evaluation_distribution,
            "target calibration evaluation_distribution",
        )
        _required_int(
            self.evaluation_predictions_per_split,
            "target calibration evaluation_predictions_per_split",
            minimum=1,
        )
        for name in (
            "thermo_ece",
            "thermo_ood_auroc",
            "softmax_ece",
            "softmax_ood_auroc",
            "mc_dropout_ece",
            "mc_dropout_ood_auroc",
        ):
            _required_finite_float(
                getattr(self, name),
                f"target calibration {name}",
                minimum=0.0,
                maximum=1.0,
            )
        _required_bool(self.passed, "target calibration passed")
        content = asdict(self)
        del content["artifact_sha256"]
        if not hmac.compare_digest(
            self.artifact_sha256,
            _canonical_digest(content),
        ):
            raise ValueError(
                "target calibration artifact_sha256 does not match its content"
            )

    @classmethod
    def from_measurements(
        cls,
        *,
        target_artifact_sha256: str,
        target_provenance: PredictiveEvidenceProvenance,
        evaluation_distribution: str,
        evaluation_predictions_per_split: int,
        thermo_ece: float,
        thermo_ood_auroc: float,
        softmax_ece: float,
        softmax_ood_auroc: float,
        mc_dropout_ece: float,
        mc_dropout_ood_auroc: float,
        passed: bool,
    ) -> TargetCalibrationObservation:
        values = {
            "target_artifact_sha256": target_artifact_sha256,
            "target_run_id": target_provenance.run_id,
            "target_checkpoint_id": target_provenance.checkpoint_id,
            "target_synaptic_config_hash": target_provenance.synaptic_config_hash,
            "target_experiment_config_hash": target_provenance.config_hash,
            "target_rng_seed": target_provenance.rng_seed,
            "evaluation_distribution": evaluation_distribution,
            "evaluation_predictions_per_split": evaluation_predictions_per_split,
            "thermo_ece": thermo_ece,
            "thermo_ood_auroc": thermo_ood_auroc,
            "softmax_ece": softmax_ece,
            "softmax_ood_auroc": softmax_ood_auroc,
            "mc_dropout_ece": mc_dropout_ece,
            "mc_dropout_ood_auroc": mc_dropout_ood_auroc,
            "passed": passed,
        }
        return cls(
            artifact_sha256=_canonical_digest(values),
            target_artifact_sha256=target_artifact_sha256,
            target_run_id=target_provenance.run_id,
            target_checkpoint_id=target_provenance.checkpoint_id,
            target_synaptic_config_hash=target_provenance.synaptic_config_hash,
            target_experiment_config_hash=target_provenance.config_hash,
            target_rng_seed=target_provenance.rng_seed,
            evaluation_distribution=evaluation_distribution,
            evaluation_predictions_per_split=evaluation_predictions_per_split,
            thermo_ece=thermo_ece,
            thermo_ood_auroc=thermo_ood_auroc,
            softmax_ece=softmax_ece,
            softmax_ood_auroc=softmax_ood_auroc,
            mc_dropout_ece=mc_dropout_ece,
            mc_dropout_ood_auroc=mc_dropout_ood_auroc,
            passed=passed,
        )

    @classmethod
    def from_experiment_report(
        cls,
        report: Any,
        target: PredictiveSeedObservation,
    ) -> TargetCalibrationObservation:
        thermo = report.methods["thermo_uq"]
        softmax = report.methods["softmax_entropy"]
        mc_dropout = report.methods["mc_dropout"]
        passed = bool(
            thermo.ece <= _TARGET_ECE_MAX
            and thermo.ood_auroc >= _TARGET_OOD_AUROC_MIN
            and thermo.ece < softmax.ece
            and thermo.ece < mc_dropout.ece
            and thermo.ood_auroc > softmax.ood_auroc
            and thermo.ood_auroc > mc_dropout.ood_auroc
        )
        return cls.from_measurements(
            target_artifact_sha256=target.artifact_sha256,
            target_provenance=target.provenance,
            evaluation_distribution=(
                "synthetic_modular_arithmetic_id_heldout_and_ood_half_vocab"
            ),
            evaluation_predictions_per_split=(
                report.config.batch_size
                * report.config.eval_pool_size
                * report.config.seq_len
            ),
            thermo_ece=float(thermo.ece),
            thermo_ood_auroc=float(thermo.ood_auroc),
            softmax_ece=float(softmax.ece),
            softmax_ood_auroc=float(softmax.ood_auroc),
            mc_dropout_ece=float(mc_dropout.ece),
            mc_dropout_ood_auroc=float(mc_dropout.ood_auroc),
            passed=passed,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TargetCalibrationObservation:
        expected = {data_field.name for data_field in fields(cls)}
        _exact_keys(payload, expected, "target calibration")
        return cls(
            artifact_sha256=_nonempty(
                payload.get("artifact_sha256"),
                "target calibration artifact_sha256",
            ),
            target_artifact_sha256=_nonempty(
                payload.get("target_artifact_sha256"),
                "target calibration target_artifact_sha256",
            ),
            target_run_id=_nonempty(
                payload.get("target_run_id"),
                "target calibration target_run_id",
            ),
            target_checkpoint_id=_nonempty(
                payload.get("target_checkpoint_id"),
                "target calibration target_checkpoint_id",
            ),
            target_synaptic_config_hash=_nonempty(
                payload.get("target_synaptic_config_hash"),
                "target calibration target_synaptic_config_hash",
            ),
            target_experiment_config_hash=_nonempty(
                payload.get("target_experiment_config_hash"),
                "target calibration target_experiment_config_hash",
            ),
            target_rng_seed=_required_int(
                payload.get("target_rng_seed"),
                "target calibration target_rng_seed",
            ),
            evaluation_distribution=_nonempty(
                payload.get("evaluation_distribution"),
                "target calibration evaluation_distribution",
            ),
            evaluation_predictions_per_split=_required_int(
                payload.get("evaluation_predictions_per_split"),
                "target calibration evaluation_predictions_per_split",
                minimum=1,
            ),
            thermo_ece=_required_finite_float(
                payload.get("thermo_ece"),
                "target calibration thermo_ece",
                minimum=0.0,
                maximum=1.0,
            ),
            thermo_ood_auroc=_required_finite_float(
                payload.get("thermo_ood_auroc"),
                "target calibration thermo_ood_auroc",
                minimum=0.0,
                maximum=1.0,
            ),
            softmax_ece=_required_finite_float(
                payload.get("softmax_ece"),
                "target calibration softmax_ece",
                minimum=0.0,
                maximum=1.0,
            ),
            softmax_ood_auroc=_required_finite_float(
                payload.get("softmax_ood_auroc"),
                "target calibration softmax_ood_auroc",
                minimum=0.0,
                maximum=1.0,
            ),
            mc_dropout_ece=_required_finite_float(
                payload.get("mc_dropout_ece"),
                "target calibration mc_dropout_ece",
                minimum=0.0,
                maximum=1.0,
            ),
            mc_dropout_ood_auroc=_required_finite_float(
                payload.get("mc_dropout_ood_auroc"),
                "target calibration mc_dropout_ood_auroc",
                minimum=0.0,
                maximum=1.0,
            ),
            passed=_required_bool(
                payload.get("passed"),
                "target calibration passed",
            ),
        )


@dataclass(frozen=True)
class PredictiveMetricComparisonObservation:
    """Raw paired deltas and reproducible statistics for one metric/baseline pair."""

    baseline: str
    metric: str
    seed_count: int
    paired_deltas: tuple[float, ...]
    bootstrap_samples: int
    bootstrap_seed: int
    paired_t_p_value: float
    wilcoxon_p_value: float
    effect_ci_low: float
    effect_ci_high: float
    favorable_direction: str
    passed: bool

    def __post_init__(self) -> None:
        if self.baseline not in {"softmax_entropy", "mc_dropout"}:
            raise ValueError("comparison baseline is outside the predictive policy")
        if self.metric not in {"ece", "ood_auroc"}:
            raise ValueError("comparison metric is outside the predictive policy")
        _required_int(
            self.seed_count,
            "comparison.seed_count",
            minimum=1,
            maximum=_MAX_PREDICTIVE_SEEDS,
        )
        _required_int(
            self.bootstrap_samples,
            "comparison.bootstrap_samples",
            minimum=1,
        )
        _required_int(self.bootstrap_seed, "comparison.bootstrap_seed")
        if self.bootstrap_samples != _PREDICTIVE_BOOTSTRAP_SAMPLES:
            raise ValueError(
                "comparison.bootstrap_samples must equal "
                f"{_PREDICTIVE_BOOTSTRAP_SAMPLES}"
            )
        if self.bootstrap_seed != _PREDICTIVE_BOOTSTRAP_SEED:
            raise ValueError(
                f"comparison.bootstrap_seed must equal {_PREDICTIVE_BOOTSTRAP_SEED}"
            )
        if not self.paired_deltas:
            raise ValueError("comparison.paired_deltas must be non-empty")
        if len(self.paired_deltas) != self.seed_count:
            raise ValueError(
                "comparison.paired_deltas must contain exactly seed_count values"
            )
        for index, value in enumerate(self.paired_deltas):
            _required_finite_float(
                value,
                f"comparison.paired_deltas[{index}]",
                minimum=-1.0,
                maximum=1.0,
            )
        for name in ("paired_t_p_value", "wilcoxon_p_value"):
            _required_finite_float(
                getattr(self, name), f"comparison.{name}", minimum=0.0, maximum=1.0
            )
        _required_finite_float(self.effect_ci_low, "comparison.effect_ci_low")
        _required_finite_float(self.effect_ci_high, "comparison.effect_ci_high")
        if self.effect_ci_low > self.effect_ci_high:
            raise ValueError("comparison effect CI bounds are reversed")
        expected_direction = "lower" if self.metric == "ece" else "higher"
        if self.favorable_direction != expected_direction:
            raise ValueError("comparison direction contradicts the canonical metric policy")
        _required_bool(self.passed, "comparison.passed")

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> PredictiveMetricComparisonObservation:
        expected = {field.name for field in fields(cls)}
        _exact_keys(payload, expected, "predictive metric comparison")
        return cls(
            baseline=_nonempty(payload.get("baseline"), "comparison.baseline"),
            metric=_nonempty(payload.get("metric"), "comparison.metric"),
            seed_count=_required_int(payload.get("seed_count"), "comparison.seed_count"),
            paired_deltas=tuple(
                _required_finite_float(
                    value,
                    f"comparison.paired_deltas[{index}]",
                    minimum=-1.0,
                    maximum=1.0,
                )
                for index, value in enumerate(
                    _required_number_array(
                        payload.get("paired_deltas"), "comparison.paired_deltas"
                    )
                )
            ),
            bootstrap_samples=_required_int(
                payload.get("bootstrap_samples"),
                "comparison.bootstrap_samples",
                minimum=1,
            ),
            bootstrap_seed=_required_int(
                payload.get("bootstrap_seed"), "comparison.bootstrap_seed"
            ),
            paired_t_p_value=_required_finite_float(
                payload.get("paired_t_p_value"),
                "comparison.paired_t_p_value",
                minimum=0.0,
                maximum=1.0,
            ),
            wilcoxon_p_value=_required_finite_float(
                payload.get("wilcoxon_p_value"),
                "comparison.wilcoxon_p_value",
                minimum=0.0,
                maximum=1.0,
            ),
            effect_ci_low=_required_finite_float(
                payload.get("effect_ci_low"), "comparison.effect_ci_low"
            ),
            effect_ci_high=_required_finite_float(
                payload.get("effect_ci_high"), "comparison.effect_ci_high"
            ),
            favorable_direction=_nonempty(
                payload.get("favorable_direction"), "comparison.favorable_direction"
            ),
            passed=_required_bool(payload.get("passed"), "comparison.passed"),
        )


@dataclass(frozen=True)
class LiveCrooksPointObservation:
    """One positive/negative current-count pair from the live Crooks curve."""

    current: int
    positive_count: int
    negative_count: int
    observed_log_ratio: float
    expected_log_ratio: float
    residual: float

    def __post_init__(self) -> None:
        _required_int(self.current, "live Crooks current", minimum=1)
        _required_int(self.positive_count, "live Crooks positive_count")
        _required_int(self.negative_count, "live Crooks negative_count")
        for name in ("observed_log_ratio", "expected_log_ratio", "residual"):
            _required_finite_float(getattr(self, name), f"live Crooks {name}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LiveCrooksPointObservation:
        expected = {data_field.name for data_field in fields(cls)}
        _exact_keys(payload, expected, "live Crooks point")
        return cls(
            current=_required_int(
                payload.get("current"), "live Crooks current", minimum=1
            ),
            positive_count=_required_int(
                payload.get("positive_count"), "live Crooks positive_count"
            ),
            negative_count=_required_int(
                payload.get("negative_count"), "live Crooks negative_count"
            ),
            observed_log_ratio=_required_finite_float(
                payload.get("observed_log_ratio"),
                "live Crooks observed_log_ratio",
            ),
            expected_log_ratio=_required_finite_float(
                payload.get("expected_log_ratio"),
                "live Crooks expected_log_ratio",
            ),
            residual=_required_finite_float(
                payload.get("residual"), "live Crooks residual"
            ),
        )


@dataclass(frozen=True)
class LiveFTSeedObservation:
    """Raw one-step Crooks/integral-FT result for one matched predictive seed."""

    experiment_seed: int
    paired_predictive_run_id: str
    experiment_config_hash: str
    release_protocol_config_hash: str
    paired_predictive_rng_seed: int
    forward_rng_seed: int
    reverse_rng_seed: int
    scope: str
    n_trajectories: int
    pool_size: int
    configured_forward_probability: float
    configured_reverse_probability: float
    forward_probability: float
    reverse_probability: float
    affinity: float
    current_counts: tuple[int, ...]
    integral_ft: float
    integral_ft_residual: float
    crooks_curve: tuple[LiveCrooksPointObservation, ...]
    max_crooks_residual: float | None
    crooks_min_count: int
    crooks_tolerance: float
    integral_ft_tolerance: float
    passed: bool

    def __post_init__(self) -> None:
        _required_int(self.experiment_seed, "live FT experiment_seed")
        _nonempty(
            self.paired_predictive_run_id,
            "live FT paired_predictive_run_id",
        )
        if not _HEX_64_RE.fullmatch(self.experiment_config_hash):
            raise ValueError(
                "live FT experiment_config_hash must be a lowercase SHA-256"
            )
        if not _HEX_16_RE.fullmatch(self.release_protocol_config_hash):
            raise ValueError(
                "live FT release_protocol_config_hash must be a normalized config digest"
            )
        for name in (
            "paired_predictive_rng_seed",
            "forward_rng_seed",
            "reverse_rng_seed",
        ):
            _required_int(getattr(self, name), f"live FT {name}")
        if self.forward_rng_seed != self.experiment_seed + 101:
            raise ValueError("live FT forward_rng_seed contradicts the source protocol")
        if self.reverse_rng_seed != self.experiment_seed + 102:
            raise ValueError("live FT reverse_rng_seed contradicts the source protocol")
        if self.paired_predictive_rng_seed != self.experiment_seed + 301:
            raise ValueError(
                "live FT paired_predictive_rng_seed contradicts the source protocol"
            )
        if self.scope != _LIVE_FT_SCOPE:
            raise ValueError("live FT scope is outside the canonical source policy")
        _required_int(self.n_trajectories, "live FT n_trajectories", minimum=1)
        if self.n_trajectories != _LIVE_FT_TRAJECTORIES:
            raise ValueError(
                f"live FT n_trajectories must equal {_LIVE_FT_TRAJECTORIES}"
            )
        _required_int(self.pool_size, "live FT pool_size", minimum=1)
        if self.pool_size != _LIVE_TUR_POOL_SIZE:
            raise ValueError(f"live FT pool_size must equal {_LIVE_TUR_POOL_SIZE}")
        for name in (
            "configured_forward_probability",
            "configured_reverse_probability",
            "forward_probability",
            "reverse_probability",
        ):
            _required_finite_float(
                getattr(self, name), f"live FT {name}", minimum=0.0, maximum=1.0
            )
        if not (
            0.0
            < self.configured_reverse_probability
            < self.configured_forward_probability
            < 1.0
        ):
            raise ValueError(
                "live FT configured probabilities must satisfy "
                "0 < reverse < forward < 1"
            )
        if not 0.0 < self.reverse_probability < self.forward_probability < 1.0:
            raise ValueError(
                "live FT probabilities must satisfy 0 < reverse < forward < 1"
            )
        for name in (
            "affinity",
            "integral_ft",
            "integral_ft_residual",
            "crooks_tolerance",
            "integral_ft_tolerance",
        ):
            _required_finite_float(
                getattr(self, name), f"live FT {name}", minimum=0.0
            )
        if self.max_crooks_residual is not None:
            _required_finite_float(
                self.max_crooks_residual,
                "live FT max_crooks_residual",
                minimum=0.0,
            )
        if len(self.current_counts) != 2 * self.pool_size + 1:
            raise ValueError(
                "live FT current_counts must cover every integer current from "
                "-pool_size through +pool_size"
            )
        for index, count in enumerate(self.current_counts):
            _required_int(count, f"live FT current_counts[{index}]")
        if sum(self.current_counts) != self.n_trajectories:
            raise ValueError("live FT current_counts must sum to n_trajectories")
        if not all(
            isinstance(item, LiveCrooksPointObservation)
            for item in self.crooks_curve
        ):
            raise TypeError("live FT crooks_curve must contain raw Crooks points")
        _required_int(self.crooks_min_count, "live FT crooks_min_count", minimum=1)
        if self.crooks_min_count != _LIVE_FT_MIN_COUNT:
            raise ValueError(
                f"live FT crooks_min_count must equal {_LIVE_FT_MIN_COUNT}"
            )
        if self.crooks_tolerance != _LIVE_CROOKS_TOLERANCE:
            raise ValueError(
                f"live FT crooks_tolerance must equal {_LIVE_CROOKS_TOLERANCE}"
            )
        if self.integral_ft_tolerance != _LIVE_INTEGRAL_FT_TOLERANCE:
            raise ValueError(
                "live FT integral_ft_tolerance must equal "
                f"{_LIVE_INTEGRAL_FT_TOLERANCE}"
            )
        _required_bool(self.passed, "live FT passed")

    @classmethod
    def from_experiment_report(cls, report: Any) -> LiveFTSeedObservation:
        """Adapt one canonical experiment report using only producer-owned provenance."""
        result = report.live_release_ft
        provenance = report.predictive_thermo_evidence.provenance
        if result.experiment_seed != report.config.seed:
            raise ValueError("live FT experiment seed contradicts its experiment report")
        return cls(
            experiment_seed=result.experiment_seed,
            paired_predictive_run_id=provenance.run_id,
            experiment_config_hash=provenance.config_hash,
            release_protocol_config_hash=result.release_protocol_config_hash,
            paired_predictive_rng_seed=provenance.rng_seed,
            forward_rng_seed=result.forward_rng_seed,
            reverse_rng_seed=result.reverse_rng_seed,
            scope=result.scope,
            n_trajectories=result.n_trajectories,
            pool_size=result.pool_size,
            configured_forward_probability=result.configured_forward_probability,
            configured_reverse_probability=result.configured_reverse_probability,
            forward_probability=result.forward_probability,
            reverse_probability=result.reverse_probability,
            affinity=result.affinity,
            current_counts=tuple(result.current_counts),
            integral_ft=result.integral_ft,
            integral_ft_residual=result.integral_ft_residual,
            crooks_curve=tuple(
                LiveCrooksPointObservation(
                    current=point.current,
                    positive_count=point.positive_count,
                    negative_count=point.negative_count,
                    observed_log_ratio=point.observed_log_ratio,
                    expected_log_ratio=point.expected_log_ratio,
                    residual=point.residual,
                )
                for point in result.curve
            ),
            max_crooks_residual=result.max_crooks_residual,
            crooks_min_count=result.crooks_min_count,
            crooks_tolerance=result.tolerance,
            integral_ft_tolerance=result.integral_tolerance,
            passed=result.passed,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LiveFTSeedObservation:
        expected = {data_field.name for data_field in fields(cls)}
        _exact_keys(payload, expected, "live FT seed")
        curve = payload.get("crooks_curve")
        if not isinstance(curve, list):
            raise TypeError("live FT crooks_curve must be a JSON array")
        current_counts = payload.get("current_counts")
        if not isinstance(current_counts, list):
            raise TypeError("live FT current_counts must be a JSON array")
        return cls(
            experiment_seed=_required_int(
                payload.get("experiment_seed"), "live FT experiment_seed"
            ),
            paired_predictive_run_id=_nonempty(
                payload.get("paired_predictive_run_id"),
                "live FT paired_predictive_run_id",
            ),
            experiment_config_hash=_nonempty(
                payload.get("experiment_config_hash"),
                "live FT experiment_config_hash",
            ),
            release_protocol_config_hash=_nonempty(
                payload.get("release_protocol_config_hash"),
                "live FT release_protocol_config_hash",
            ),
            paired_predictive_rng_seed=_required_int(
                payload.get("paired_predictive_rng_seed"),
                "live FT paired_predictive_rng_seed",
            ),
            forward_rng_seed=_required_int(
                payload.get("forward_rng_seed"), "live FT forward_rng_seed"
            ),
            reverse_rng_seed=_required_int(
                payload.get("reverse_rng_seed"), "live FT reverse_rng_seed"
            ),
            scope=_nonempty(payload.get("scope"), "live FT scope"),
            n_trajectories=_required_int(
                payload.get("n_trajectories"),
                "live FT n_trajectories",
                minimum=1,
            ),
            pool_size=_required_int(
                payload.get("pool_size"), "live FT pool_size", minimum=1
            ),
            configured_forward_probability=_required_finite_float(
                payload.get("configured_forward_probability"),
                "live FT configured_forward_probability",
                minimum=0.0,
                maximum=1.0,
            ),
            configured_reverse_probability=_required_finite_float(
                payload.get("configured_reverse_probability"),
                "live FT configured_reverse_probability",
                minimum=0.0,
                maximum=1.0,
            ),
            forward_probability=_required_finite_float(
                payload.get("forward_probability"),
                "live FT forward_probability",
                minimum=0.0,
                maximum=1.0,
            ),
            reverse_probability=_required_finite_float(
                payload.get("reverse_probability"),
                "live FT reverse_probability",
                minimum=0.0,
                maximum=1.0,
            ),
            affinity=_required_finite_float(
                payload.get("affinity"), "live FT affinity", minimum=0.0
            ),
            current_counts=tuple(
                _required_int(count, f"live FT current_counts[{index}]")
                for index, count in enumerate(current_counts)
            ),
            integral_ft=_required_finite_float(
                payload.get("integral_ft"), "live FT integral_ft", minimum=0.0
            ),
            integral_ft_residual=_required_finite_float(
                payload.get("integral_ft_residual"),
                "live FT integral_ft_residual",
                minimum=0.0,
            ),
            crooks_curve=tuple(
                LiveCrooksPointObservation.from_dict(
                    _mapping(item, "live Crooks point")
                )
                for item in curve
            ),
            max_crooks_residual=_optional_finite_float(
                payload.get("max_crooks_residual"),
                "live FT max_crooks_residual",
            ),
            crooks_min_count=_required_int(
                payload.get("crooks_min_count"),
                "live FT crooks_min_count",
                minimum=1,
            ),
            crooks_tolerance=_required_finite_float(
                payload.get("crooks_tolerance"),
                "live FT crooks_tolerance",
                minimum=0.0,
            ),
            integral_ft_tolerance=_required_finite_float(
                payload.get("integral_ft_tolerance"),
                "live FT integral_ft_tolerance",
                minimum=0.0,
            ),
            passed=_required_bool(payload.get("passed"), "live FT passed"),
        )


@dataclass(frozen=True)
class LiveTURObservation:
    """Raw finite-binomial diagnostic for the classic continuous-time TUR transfer."""

    scope: str
    pool_size: int
    forward_probability: float
    reverse_probability: float
    affinity: float
    relative_variance: float
    entropy_bound: float
    slack: float
    bound_ratio: float
    nonvacuous: bool
    satisfied: bool

    def __post_init__(self) -> None:
        if self.scope != _LIVE_TUR_SCOPE:
            raise ValueError("live TUR scope is outside the canonical source policy")
        _required_int(self.pool_size, "live TUR pool_size", minimum=1)
        if self.pool_size != _LIVE_TUR_POOL_SIZE:
            raise ValueError(f"live TUR pool_size must equal {_LIVE_TUR_POOL_SIZE}")
        _required_finite_float(
            self.forward_probability,
            "live TUR forward_probability",
            minimum=0.0,
            maximum=1.0,
        )
        _required_finite_float(
            self.reverse_probability,
            "live TUR reverse_probability",
            minimum=0.0,
            maximum=1.0,
        )
        if not 0.0 < self.reverse_probability < self.forward_probability < 1.0:
            raise ValueError(
                "live TUR probabilities must satisfy 0 < reverse < forward < 1"
            )
        _required_finite_float(self.affinity, "live TUR affinity", minimum=0.0)
        _required_finite_float(
            self.relative_variance,
            "live TUR relative_variance",
            minimum=0.0,
        )
        _required_finite_float(
            self.entropy_bound,
            "live TUR entropy_bound",
            minimum=0.0,
        )
        _required_finite_float(self.slack, "live TUR slack")
        _required_finite_float(self.bound_ratio, "live TUR bound_ratio", minimum=0.0)
        _required_bool(self.nonvacuous, "live TUR nonvacuous")
        _required_bool(self.satisfied, "live TUR satisfied")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LiveTURObservation:
        expected = {data_field.name for data_field in fields(cls)}
        _exact_keys(payload, expected, "live TUR")
        return cls(
            scope=_nonempty(payload.get("scope"), "live TUR scope"),
            pool_size=_required_int(
                payload.get("pool_size"), "live TUR pool_size", minimum=1
            ),
            forward_probability=_required_finite_float(
                payload.get("forward_probability"),
                "live TUR forward_probability",
                minimum=0.0,
                maximum=1.0,
            ),
            reverse_probability=_required_finite_float(
                payload.get("reverse_probability"),
                "live TUR reverse_probability",
                minimum=0.0,
                maximum=1.0,
            ),
            affinity=_required_finite_float(
                payload.get("affinity"), "live TUR affinity", minimum=0.0
            ),
            relative_variance=_required_finite_float(
                payload.get("relative_variance"),
                "live TUR relative_variance",
                minimum=0.0,
            ),
            entropy_bound=_required_finite_float(
                payload.get("entropy_bound"),
                "live TUR entropy_bound",
                minimum=0.0,
            ),
            slack=_required_finite_float(payload.get("slack"), "live TUR slack"),
            bound_ratio=_required_finite_float(
                payload.get("bound_ratio"),
                "live TUR bound_ratio",
                minimum=0.0,
            ),
            nonvacuous=_required_bool(
                payload.get("nonvacuous"), "live TUR nonvacuous"
            ),
            satisfied=_required_bool(payload.get("satisfied"), "live TUR satisfied"),
        )


@dataclass(frozen=True)
class MatchedSeedStatisticsObservation:
    """Canonical ECE/AUROC × two-baseline predictive report with live FT/TUR values."""

    artifact_sha256: str
    identity: ModelIdentity
    predictive_run_ids: tuple[str, ...]
    predictive_checkpoint_ids: tuple[str, ...]
    predictive_synaptic_config_hashes: tuple[str, ...]
    predictive_config_hashes: tuple[str, ...]
    predictive_rng_seeds: tuple[int, ...]
    expected_sites: tuple[tuple[str, int], ...]
    comparisons: tuple[PredictiveMetricComparisonObservation, ...]
    alpha: float
    fixed_policy_applied: bool
    live_ft_seeds: tuple[LiveFTSeedObservation, ...]
    live_tur: LiveTURObservation
    passed: bool

    def __post_init__(self) -> None:
        if not _HEX_64_RE.fullmatch(self.artifact_sha256):
            raise ValueError("statistics artifact_sha256 must be a lowercase SHA-256")
        if not isinstance(self.identity, ModelIdentity):
            raise TypeError("statistics.identity must be a ModelIdentity")
        if not self.predictive_run_ids or len(set(self.predictive_run_ids)) != len(
            self.predictive_run_ids
        ):
            raise ValueError("statistics predictive_run_ids must be non-empty and unique")
        if len(self.predictive_run_ids) > _MAX_PREDICTIVE_SEEDS:
            raise ValueError(
                "statistics predictive cohort exceeds the fixed maximum of "
                f"{_MAX_PREDICTIVE_SEEDS} seeds"
            )
        for run_id in self.predictive_run_ids:
            _nonempty(run_id, "statistics predictive run ID")
        if len(self.predictive_checkpoint_ids) != len(self.predictive_run_ids):
            raise ValueError(
                "statistics predictive checkpoint IDs must align with run IDs"
            )
        if len(self.predictive_config_hashes) != len(self.predictive_run_ids):
            raise ValueError(
                "statistics predictive config hashes must align with run IDs"
            )
        if len(self.predictive_synaptic_config_hashes) != len(
            self.predictive_run_ids
        ):
            raise ValueError(
                "statistics predictive synaptic config hashes must align with run IDs"
            )
        for checkpoint_id in self.predictive_checkpoint_ids:
            if not _HEX_64_RE.fullmatch(checkpoint_id):
                raise ValueError(
                    "statistics predictive checkpoint ID must be a lowercase SHA-256"
                )
        for predictive_config_hash in self.predictive_config_hashes:
            if not _HEX_64_RE.fullmatch(predictive_config_hash):
                raise ValueError(
                    "statistics predictive config hash must be a lowercase SHA-256"
                )
        for synaptic_config_hash in self.predictive_synaptic_config_hashes:
            if not _HEX_16_RE.fullmatch(synaptic_config_hash):
                raise ValueError(
                    "statistics predictive synaptic config hash must be normalized"
                )
        if not self.predictive_rng_seeds or len(set(self.predictive_rng_seeds)) != len(
            self.predictive_rng_seeds
        ):
            raise ValueError("statistics predictive_rng_seeds must be non-empty and unique")
        for rng_seed in self.predictive_rng_seeds:
            _required_int(rng_seed, "statistics predictive RNG seed")
        if not self.expected_sites or len(set(self.expected_sites)) != len(
            self.expected_sites
        ):
            raise ValueError("statistics expected_sites must be non-empty and unique")
        for layer, head in self.expected_sites:
            _nonempty(layer, "statistics expected site layer")
            _required_int(head, "statistics expected site head")
        _required_finite_float(self.alpha, "statistics.alpha")
        if self.alpha != _PREDICTIVE_ALPHA:
            raise ValueError(f"statistics.alpha must equal {_PREDICTIVE_ALPHA}")
        _required_bool(
            self.fixed_policy_applied, "statistics.fixed_policy_applied"
        )
        if not self.comparisons:
            raise ValueError("statistics.comparisons must be non-empty")
        if not all(
            isinstance(item, PredictiveMetricComparisonObservation)
            for item in self.comparisons
        ):
            raise TypeError("statistics.comparisons must contain comparison observations")
        if not self.live_ft_seeds:
            raise ValueError("statistics.live_ft_seeds must be non-empty")
        if not all(isinstance(item, LiveFTSeedObservation) for item in self.live_ft_seeds):
            raise TypeError("statistics.live_ft_seeds must contain live FT observations")
        if not isinstance(self.live_tur, LiveTURObservation):
            raise TypeError("statistics.live_tur must be a LiveTURObservation")
        _required_bool(self.passed, "statistics.passed")
        content = asdict(self)
        del content["artifact_sha256"]
        if not hmac.compare_digest(
            self.artifact_sha256, _canonical_digest(content)
        ):
            raise ValueError("statistics artifact_sha256 does not match its content")

    @classmethod
    def from_measurements(
        cls,
        *,
        identity: ModelIdentity,
        predictive_run_ids: Sequence[str],
        predictive_checkpoint_ids: Sequence[str],
        predictive_synaptic_config_hashes: Sequence[str],
        predictive_config_hashes: Sequence[str],
        predictive_rng_seeds: Sequence[int],
        expected_sites: Sequence[tuple[str, int]],
        comparisons: Sequence[PredictiveMetricComparisonObservation],
        live_ft_seeds: Sequence[LiveFTSeedObservation],
        live_tur: LiveTURObservation,
        alpha: float = _PREDICTIVE_ALPHA,
        fixed_policy_applied: bool,
        passed: bool,
    ) -> MatchedSeedStatisticsObservation:
        """Construct a digest-bound source report from its complete raw policy inputs."""
        comparison_tuple = tuple(comparisons)
        live_ft_tuple = tuple(live_ft_seeds)
        run_id_tuple = tuple(predictive_run_ids)
        checkpoint_id_tuple = tuple(predictive_checkpoint_ids)
        predictive_synaptic_config_hash_tuple = tuple(
            predictive_synaptic_config_hashes
        )
        predictive_config_hash_tuple = tuple(predictive_config_hashes)
        rng_seed_tuple = tuple(predictive_rng_seeds)
        expected_site_tuple = tuple(expected_sites)
        digest_values = {
            "identity": asdict(identity),
            "predictive_run_ids": run_id_tuple,
            "predictive_checkpoint_ids": checkpoint_id_tuple,
            "predictive_synaptic_config_hashes": (
                predictive_synaptic_config_hash_tuple
            ),
            "predictive_config_hashes": predictive_config_hash_tuple,
            "predictive_rng_seeds": rng_seed_tuple,
            "expected_sites": expected_site_tuple,
            "comparisons": [asdict(item) for item in comparison_tuple],
            "alpha": alpha,
            "fixed_policy_applied": fixed_policy_applied,
            "live_ft_seeds": [asdict(item) for item in live_ft_tuple],
            "live_tur": asdict(live_tur),
            "passed": passed,
        }
        return cls(
            artifact_sha256=_canonical_digest(digest_values),
            identity=identity,
            predictive_run_ids=run_id_tuple,
            predictive_checkpoint_ids=checkpoint_id_tuple,
            predictive_synaptic_config_hashes=(
                predictive_synaptic_config_hash_tuple
            ),
            predictive_config_hashes=predictive_config_hash_tuple,
            predictive_rng_seeds=rng_seed_tuple,
            expected_sites=expected_site_tuple,
            comparisons=comparison_tuple,
            alpha=alpha,
            fixed_policy_applied=fixed_policy_applied,
            live_ft_seeds=live_ft_tuple,
            live_tur=live_tur,
            passed=passed,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MatchedSeedStatisticsObservation:
        expected = {field.name for field in fields(cls)}
        _exact_keys(payload, expected, "predictive statistics")
        comparisons = payload.get("comparisons")
        if not isinstance(comparisons, list):
            raise TypeError("statistics.comparisons must be a JSON array")
        live_ft_seeds = payload.get("live_ft_seeds")
        if not isinstance(live_ft_seeds, list):
            raise TypeError("statistics.live_ft_seeds must be a JSON array")
        return cls(
            artifact_sha256=_nonempty(
                payload.get("artifact_sha256"), "statistics.artifact_sha256"
            ),
            identity=ModelIdentity.from_dict(
                _mapping(payload.get("identity"), "statistics.identity")
            ),
            predictive_run_ids=_required_string_array(
                payload.get("predictive_run_ids"), "statistics.predictive_run_ids"
            ),
            predictive_checkpoint_ids=_required_string_array(
                payload.get("predictive_checkpoint_ids"),
                "statistics.predictive_checkpoint_ids",
            ),
            predictive_synaptic_config_hashes=_required_string_array(
                payload.get("predictive_synaptic_config_hashes"),
                "statistics.predictive_synaptic_config_hashes",
            ),
            predictive_config_hashes=_required_string_array(
                payload.get("predictive_config_hashes"),
                "statistics.predictive_config_hashes",
            ),
            predictive_rng_seeds=tuple(
                _required_int(value, f"statistics.predictive_rng_seeds[{index}]")
                for index, value in enumerate(
                    _required_number_array(
                        payload.get("predictive_rng_seeds"),
                        "statistics.predictive_rng_seeds",
                    )
                )
            ),
            expected_sites=_predictive_sites_from_json(
                payload.get("expected_sites"), "statistics.expected_sites"
            ),
            comparisons=tuple(
                PredictiveMetricComparisonObservation.from_dict(
                    _mapping(item, "predictive metric comparison")
                )
                for item in comparisons
            ),
            alpha=_required_finite_float(
                payload.get("alpha"),
                "statistics.alpha",
            ),
            fixed_policy_applied=_required_bool(
                payload.get("fixed_policy_applied"),
                "statistics.fixed_policy_applied",
            ),
            live_ft_seeds=tuple(
                LiveFTSeedObservation.from_dict(_mapping(item, "live FT seed"))
                for item in live_ft_seeds
            ),
            live_tur=LiveTURObservation.from_dict(
                _mapping(payload.get("live_tur"), "live TUR")
            ),
            passed=_required_bool(payload.get("passed"), "statistics.passed"),
        )


def _with_predictive_statistics(
    evidence: PredictiveThermoEvidence,
    statistics_passed: bool,
) -> PredictiveThermoEvidence:
    """Finalize the group-bound claim fields of one producer-owned local artifact."""
    reasons: list[str] = []
    if not evidence.heads:
        reasons.append("empty_evidence")
    if not evidence.fresh:
        reasons.append("stale_evidence")
    if evidence.heads and not evidence.local_gates_passed:
        reasons.append("local_layer_head_gates_failed")
    if not statistics_passed:
        reasons.append("multi_seed_statistics_pending_or_failed")
    claim = not reasons
    return replace(
        evidence,
        multi_seed_statistics_passed=statistics_passed,
        predictive_distribution_claim=claim,
        calibration_mode=(
            "predictive_thermodynamic_calibration"
            if claim
            else "empirical_ece_fallback"
        ),
        refusal_reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class PredictiveCalibrationObservation:
    """Deployed-checkpoint local evidence plus a matched research cohort artifact."""

    identity: ModelIdentity
    expected_sites: tuple[tuple[str, int], ...]
    target: PredictiveSeedObservation
    target_calibration: TargetCalibrationObservation
    seeds: tuple[PredictiveSeedObservation, ...]
    statistics: MatchedSeedStatisticsObservation
    _runtime_verified: bool = field(default=False, init=False, repr=False, compare=False)
    _source_target: PredictiveSeedObservation | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _source_target_calibration: TargetCalibrationObservation | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _source_seeds: tuple[PredictiveSeedObservation, ...] = field(
        default=(), init=False, repr=False, compare=False
    )
    _source_statistics: MatchedSeedStatisticsObservation | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.expected_sites:
            raise ValueError("predictive expected_sites must be non-empty")
        if len(set(self.expected_sites)) != len(self.expected_sites):
            raise ValueError("predictive expected_sites must be unique")
        for layer, head in self.expected_sites:
            _nonempty(layer, "predictive expected site layer")
            _required_int(head, "predictive expected site head")
        if not isinstance(self.target_calibration, TargetCalibrationObservation):
            raise TypeError(
                "predictive target_calibration must be a TargetCalibrationObservation"
            )
        if len(self.seeds) > _MAX_PREDICTIVE_SEEDS:
            raise ValueError(
                "predictive cohort exceeds the fixed maximum of "
                f"{_MAX_PREDICTIVE_SEEDS} seeds"
            )

    @classmethod
    def from_evidences(
        cls,
        identity: ModelIdentity,
        evidences: Sequence[PredictiveThermoEvidence],
        *,
        target_evidence: PredictiveThermoEvidence,
        target_calibration: TargetCalibrationObservation,
        expected_sites: Sequence[tuple[str, int]],
        statistics: MatchedSeedStatisticsObservation,
    ) -> PredictiveCalibrationObservation:
        return cls(
            identity=identity,
            expected_sites=tuple(expected_sites),
            target=PredictiveSeedObservation.from_evidence(target_evidence),
            target_calibration=target_calibration,
            seeds=tuple(PredictiveSeedObservation.from_evidence(item) for item in evidences),
            statistics=statistics,
        )

    @classmethod
    def from_multi_seed_report(
        cls,
        identity: ModelIdentity,
        report: Any,
        *,
        target_report: Any,
    ) -> PredictiveCalibrationObservation:
        """Adapt explicit target and cohort producer reports into live-bound evidence."""
        reports = tuple(sorted(report.reports, key=lambda item: item.config.seed))
        if len(reports) < 2:
            raise ValueError("predictive cohort must contain at least two experiment reports")
        if len(reports) > _MAX_PREDICTIVE_SEEDS:
            raise ValueError(
                "predictive cohort exceeds the fixed maximum of "
                f"{_MAX_PREDICTIVE_SEEDS} seeds"
            )
        cohort_seeds = tuple(item.config.seed for item in reports)
        if len(set(cohort_seeds)) != len(cohort_seeds):
            raise ValueError("predictive cohort experiment seeds must be unique")
        if tuple(report.seeds) != cohort_seeds:
            raise ValueError(
                "predictive cohort top-level seeds must match canonical report order"
            )
        if report.alpha != _PREDICTIVE_ALPHA:
            raise ValueError(
                f"predictive cohort alpha must equal {_PREDICTIVE_ALPHA}"
            )
        if report.bootstrap_samples != _PREDICTIVE_BOOTSTRAP_SAMPLES:
            raise ValueError(
                "predictive cohort bootstrap_samples must equal "
                f"{_PREDICTIVE_BOOTSTRAP_SAMPLES}"
            )
        if report.bootstrap_seed != _PREDICTIVE_BOOTSTRAP_SEED:
            raise ValueError(
                "predictive cohort bootstrap_seed must equal "
                f"{_PREDICTIVE_BOOTSTRAP_SEED}"
            )
        source_evidences = tuple(
            item.predictive_thermo_evidence for item in reports
        )
        source_target_evidence = target_report.predictive_thermo_evidence
        for source_report, evidence in (
            (target_report, source_target_evidence),
            *zip(reports, source_evidences, strict=True),
        ):
            experiment_seed = _required_int(
                source_report.config.seed,
                "predictive producer experiment seed",
            )
            if evidence.provenance.rng_seed != experiment_seed + 301:
                raise ValueError(
                    "predictive producer RNG provenance contradicts its experiment seed"
                )
            expected_run_id = (
                "stochastic-thermo-predictive-"
                f"{evidence.provenance.config_hash[:12]}-s{experiment_seed}"
            )
            if evidence.provenance.run_id != expected_run_id:
                raise ValueError(
                    "predictive producer run ID contradicts its experiment/config provenance"
                )
        expected_sites = tuple(
            (head.layer_address, head.head_index)
            for head in source_evidences[0].heads
        )
        if not expected_sites or any(
            tuple((head.layer_address, head.head_index) for head in evidence.heads)
            != expected_sites
            for evidence in (*source_evidences[1:], source_target_evidence)
        ):
            raise ValueError(
                "predictive target/cohort reports must have identical non-empty layer/head sites"
            )

        comparisons: list[PredictiveMetricComparisonObservation] = []
        for baseline in ("softmax_entropy", "mc_dropout"):
            for metric in ("ece", "ood_auroc"):
                deltas = np.asarray(
                    [
                        float(getattr(item.methods["thermo_uq"], metric))
                        - float(getattr(item.methods[baseline], metric))
                        for item in reports
                    ],
                    dtype=np.float64,
                )
                _, paired_t_p_value = paired_t_test(deltas)
                wilcoxon_p_value = wilcoxon_signed_rank(deltas)
                effect_ci_low, effect_ci_high = bootstrap_ci(
                    deltas,
                    n_boot=_PREDICTIVE_BOOTSTRAP_SAMPLES,
                    seed=_PREDICTIVE_BOOTSTRAP_SEED,
                )
                favorable_direction = "lower" if metric == "ece" else "higher"
                ci_favorable = (
                    effect_ci_high < 0.0
                    if favorable_direction == "lower"
                    else effect_ci_low > 0.0
                )
                passed = bool(
                    len(deltas) >= 6
                    and int(np.count_nonzero(deltas)) >= 6
                    and paired_t_p_value <= _PREDICTIVE_ALPHA
                    and wilcoxon_p_value <= _PREDICTIVE_ALPHA
                    and ci_favorable
                )
                comparisons.append(
                    PredictiveMetricComparisonObservation(
                        baseline=baseline,
                        metric=metric,
                        seed_count=len(deltas),
                        paired_deltas=tuple(float(value) for value in deltas),
                        bootstrap_samples=_PREDICTIVE_BOOTSTRAP_SAMPLES,
                        bootstrap_seed=_PREDICTIVE_BOOTSTRAP_SEED,
                        paired_t_p_value=paired_t_p_value,
                        wilcoxon_p_value=wilcoxon_p_value,
                        effect_ci_low=effect_ci_low,
                        effect_ci_high=effect_ci_high,
                        favorable_direction=favorable_direction,
                        passed=passed,
                    )
                )

        live_ft_seeds = tuple(
            LiveFTSeedObservation.from_experiment_report(item) for item in reports
        )
        first_ft = reports[0].live_release_ft
        live_tur = LiveTURObservation(
            scope=report.live_tur.scope,
            pool_size=first_ft.pool_size,
            forward_probability=first_ft.forward_probability,
            reverse_probability=first_ft.reverse_probability,
            affinity=first_ft.affinity,
            relative_variance=report.live_tur.relative_variance,
            entropy_bound=report.live_tur.entropy_bound,
            slack=report.live_tur.slack,
            bound_ratio=report.live_tur.bound_ratio,
            nonvacuous=report.live_tur.nonvacuous,
            satisfied=report.live_tur.satisfied,
        )
        statistics_passed = bool(
            all(item.passed for item in comparisons)
            and all(item.passed for item in live_ft_seeds)
            and live_tur.nonvacuous
            and live_tur.satisfied
        )
        statistics = MatchedSeedStatisticsObservation.from_measurements(
            identity=identity,
            predictive_run_ids=tuple(
                evidence.provenance.run_id for evidence in source_evidences
            ),
            predictive_checkpoint_ids=tuple(
                evidence.provenance.checkpoint_id for evidence in source_evidences
            ),
            predictive_synaptic_config_hashes=tuple(
                evidence.provenance.synaptic_config_hash
                for evidence in source_evidences
            ),
            predictive_config_hashes=tuple(
                evidence.provenance.config_hash for evidence in source_evidences
            ),
            predictive_rng_seeds=tuple(
                evidence.provenance.rng_seed for evidence in source_evidences
            ),
            expected_sites=expected_sites,
            comparisons=comparisons,
            live_ft_seeds=live_ft_seeds,
            live_tur=live_tur,
            alpha=_PREDICTIVE_ALPHA,
            fixed_policy_applied=True,
            passed=statistics_passed,
        )
        evidences = tuple(
            _with_predictive_statistics(evidence, statistics_passed)
            for evidence in source_evidences
        )
        target_evidence = _with_predictive_statistics(
            source_target_evidence,
            statistics_passed,
        )
        target_seed = PredictiveSeedObservation.from_evidence(target_evidence)
        target_calibration = TargetCalibrationObservation.from_experiment_report(
            target_report,
            target_seed,
        )
        source_verdict = report.predictive_distribution
        expected_source_verdict = predictive_distribution_verdict(
            source_evidences,
            multi_seed_statistics_passed=statistics_passed,
        )
        if source_verdict != expected_source_verdict:
            raise ValueError(
                "predictive cohort verdict contradicts recomputed local/statistical gates"
            )
        observation = cls.from_evidences(
            identity,
            evidences,
            target_evidence=target_evidence,
            target_calibration=target_calibration,
            expected_sites=expected_sites,
            statistics=statistics,
        )
        object.__setattr__(observation, "_runtime_verified", True)
        object.__setattr__(observation, "_source_target", observation.target)
        object.__setattr__(
            observation,
            "_source_target_calibration",
            observation.target_calibration,
        )
        object.__setattr__(observation, "_source_seeds", observation.seeds)
        object.__setattr__(
            observation,
            "_source_statistics",
            observation.statistics,
        )
        return observation

    def to_manifest_dict(self) -> dict[str, Any]:
        """Serialize report values without the non-transferable producer capability."""
        return {
            "identity": asdict(self.identity),
            "expected_sites": [list(site) for site in self.expected_sites],
            "target": asdict(self.target),
            "target_calibration": asdict(self.target_calibration),
            "seeds": [asdict(seed) for seed in self.seeds],
            "statistics": asdict(self.statistics),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> PredictiveCalibrationObservation:
        _exact_keys(
            payload,
            {
                "identity",
                "expected_sites",
                "target",
                "target_calibration",
                "seeds",
                "statistics",
            },
            "predictive_calibration",
        )
        seeds = payload.get("seeds")
        if not isinstance(seeds, list):
            raise TypeError("predictive_calibration.seeds must be a JSON array")
        return cls(
            identity=ModelIdentity.from_dict(
                _mapping(payload.get("identity"), "predictive_calibration.identity")
            ),
            expected_sites=_predictive_sites_from_json(
                payload.get("expected_sites"),
                "predictive_calibration.expected_sites",
            ),
            target=PredictiveSeedObservation.from_dict(
                _mapping(
                    payload.get("target"), "predictive_calibration target"
                )
            ),
            target_calibration=TargetCalibrationObservation.from_dict(
                _mapping(
                    payload.get("target_calibration"),
                    "predictive_calibration target calibration",
                )
            ),
            seeds=tuple(
                PredictiveSeedObservation.from_dict(
                    _mapping(item, "predictive_calibration seed")
                )
                for item in seeds
            ),
            statistics=MatchedSeedStatisticsObservation.from_dict(
                _mapping(
                    payload.get("statistics"), "predictive_calibration.statistics"
                )
            ),
        )


@dataclass(frozen=True)
class RobustnessRecordObservation:
    """Scope-preserving aggregate of one tropical runtime certificate record."""

    step: int
    scope: str
    input_norm: str
    exact_affine: bool
    replayable: bool
    values_frozen: bool
    certified_radius: float | None
    certified_radius_unbounded: bool
    selection_certified: bool
    lipschitz_certified: bool
    readout_certified: bool | None
    output_stability_certified: bool
    artifacts_bound: bool
    certified: bool
    reason: str
    fingerprint_digest: str
    state_digest: str
    score_digest: str
    slope_digest: str
    lipschitz_slope_digest: str
    schedule_digest: str | None
    artifact_sha256: str

    def __post_init__(self) -> None:
        _required_int(self.step, "robustness record step")
        CertificateScope(self.scope)
        InputNorm(self.input_norm)
        for name in (
            "exact_affine",
            "replayable",
            "values_frozen",
            "certified_radius_unbounded",
            "selection_certified",
            "lipschitz_certified",
            "output_stability_certified",
            "artifacts_bound",
            "certified",
        ):
            _required_bool(getattr(self, name), f"robustness record {name}")
        if self.readout_certified is not None:
            _required_bool(self.readout_certified, "robustness record readout_certified")
        _numeric_or_none(
            self.certified_radius, "robustness record certified_radius"
        )
        _nonempty(self.reason, "robustness record reason")
        for name in (
            "fingerprint_digest",
            "state_digest",
            "score_digest",
            "slope_digest",
            "lipschitz_slope_digest",
            "artifact_sha256",
        ):
            if not _HEX_64_RE.fullmatch(getattr(self, name)):
                raise ValueError(f"robustness record {name} must be a lowercase SHA-256")
        if self.schedule_digest is not None and not _HEX_64_RE.fullmatch(
            self.schedule_digest
        ):
            raise ValueError("robustness record schedule_digest must be null or a SHA-256")
        if not hmac.compare_digest(
            self.artifact_sha256,
            _canonical_digest(
                {
                    field.name: getattr(self, field.name)
                    for field in fields(self)
                    if field.name != "artifact_sha256"
                }
            ),
        ):
            raise ValueError("robustness artifact_sha256 does not match its summary")

    @classmethod
    def from_record(
        cls, record: TropicalCertificateRecord
    ) -> RobustnessRecordObservation:
        values = {
            "step": record.step,
            "scope": record.certificate_scope.value,
            "input_norm": record.geometry.input_norm.value,
            "exact_affine": record.geometry.scope is GeometryScope.EXACT_AFFINE,
            "replayable": record.fingerprint.replayable,
            "values_frozen": record.values_frozen,
            "certified_radius": record.geometry.certified_radius,
            "certified_radius_unbounded": record.geometry.certified_radius_unbounded,
            "selection_certified": record.selection_certified,
            "lipschitz_certified": record.lipschitz_certified,
            "readout_certified": record.readout_certified,
            "output_stability_certified": record.output_stability_certified,
            "artifacts_bound": record.artifacts_bound,
            "certified": record.certified,
            "reason": record.reason,
            "fingerprint_digest": record.fingerprint.digest,
            "state_digest": record.fingerprint.state_digest,
            "score_digest": record.fingerprint.score_digest,
            "slope_digest": record.fingerprint.slope_digest,
            "lipschitz_slope_digest": record.lipschitz.slope_digest,
            "schedule_digest": record.schedule_digest,
        }
        return cls(
            step=record.step,
            scope=record.certificate_scope.value,
            input_norm=record.geometry.input_norm.value,
            exact_affine=record.geometry.scope is GeometryScope.EXACT_AFFINE,
            replayable=record.fingerprint.replayable,
            values_frozen=record.values_frozen,
            certified_radius=record.geometry.certified_radius,
            certified_radius_unbounded=record.geometry.certified_radius_unbounded,
            selection_certified=record.selection_certified,
            lipschitz_certified=record.lipschitz_certified,
            readout_certified=record.readout_certified,
            output_stability_certified=record.output_stability_certified,
            artifacts_bound=record.artifacts_bound,
            certified=record.certified,
            reason=record.reason,
            fingerprint_digest=record.fingerprint.digest,
            state_digest=record.fingerprint.state_digest,
            score_digest=record.fingerprint.score_digest,
            slope_digest=record.fingerprint.slope_digest,
            lipschitz_slope_digest=record.lipschitz.slope_digest,
            schedule_digest=record.schedule_digest,
            artifact_sha256=_canonical_digest(values),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RobustnessRecordObservation:
        expected = {field.name for field in fields(cls)}
        _exact_keys(payload, expected, "robustness record")
        readout = payload.get("readout_certified")
        if readout is not None and not isinstance(readout, bool):
            raise TypeError("readout_certified must be a bool or null")
        scope = CertificateScope(
            _nonempty(payload.get("scope"), "robustness record scope")
        ).value
        input_norm = InputNorm(
            _nonempty(payload.get("input_norm"), "robustness record input_norm")
        ).value
        schedule_digest = payload.get("schedule_digest")
        if schedule_digest is not None:
            schedule_digest = _nonempty(
                schedule_digest, "robustness record schedule_digest"
            )
        return cls(
            step=_required_int(payload.get("step"), "robustness record step"),
            scope=scope,
            input_norm=input_norm,
            exact_affine=_required_bool(
                payload.get("exact_affine"), "robustness record exact_affine"
            ),
            replayable=_required_bool(
                payload.get("replayable"), "robustness record replayable"
            ),
            values_frozen=_required_bool(
                payload.get("values_frozen"), "robustness record values_frozen"
            ),
            certified_radius=_optional_finite_float(
                payload.get("certified_radius"), "robustness record certified_radius"
            ),
            certified_radius_unbounded=_required_bool(
                payload.get("certified_radius_unbounded"),
                "robustness record certified_radius_unbounded",
            ),
            selection_certified=_required_bool(
                payload.get("selection_certified"),
                "robustness record selection_certified",
            ),
            lipschitz_certified=_required_bool(
                payload.get("lipschitz_certified"),
                "robustness record lipschitz_certified",
            ),
            readout_certified=readout,
            output_stability_certified=_required_bool(
                payload.get("output_stability_certified"),
                "robustness record output_stability_certified",
            ),
            artifacts_bound=_required_bool(
                payload.get("artifacts_bound"), "robustness record artifacts_bound"
            ),
            certified=_required_bool(
                payload.get("certified"), "robustness record certified"
            ),
            reason=_nonempty(payload.get("reason"), "robustness record reason"),
            fingerprint_digest=_nonempty(
                payload.get("fingerprint_digest"), "robustness record fingerprint_digest"
            ),
            state_digest=_nonempty(
                payload.get("state_digest"), "robustness record state_digest"
            ),
            score_digest=_nonempty(
                payload.get("score_digest"), "robustness record score_digest"
            ),
            slope_digest=_nonempty(
                payload.get("slope_digest"), "robustness record slope_digest"
            ),
            lipschitz_slope_digest=_nonempty(
                payload.get("lipschitz_slope_digest"),
                "robustness record lipschitz_slope_digest",
            ),
            schedule_digest=schedule_digest,
            artifact_sha256=_nonempty(
                payload.get("artifact_sha256"), "robustness record artifact_sha256"
            ),
        )


@dataclass(frozen=True)
class RobustnessObservation:
    """Checkpoint/config-bound tropical evidence for the protected decision scopes."""

    identity: ModelIdentity
    records: tuple[RobustnessRecordObservation, ...]
    _runtime_verified: bool = field(default=False, init=False, repr=False, compare=False)
    _source_records: tuple[TropicalCertificateRecord, ...] = field(
        default=(), init=False, repr=False, compare=False
    )

    @classmethod
    def from_monitor(
        cls, identity: ModelIdentity, monitor: TropicalCertificateMonitor
    ) -> RobustnessObservation:
        observation = cls(
            identity=identity,
            records=tuple(
                RobustnessRecordObservation.from_record(record)
                for record in monitor.records
            ),
        )
        object.__setattr__(observation, "_runtime_verified", True)
        object.__setattr__(observation, "_source_records", tuple(monitor.records))
        return observation

    def to_manifest_dict(self) -> dict[str, Any]:
        """Serialize a report while dropping non-transferable in-process attestation."""
        return {
            "identity": asdict(self.identity),
            "records": [asdict(record) for record in self.records],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RobustnessObservation:
        _exact_keys(payload, {"identity", "records"}, "robustness")
        records = payload.get("records")
        if not isinstance(records, list):
            raise TypeError("robustness.records must be a JSON array")
        return cls(
            identity=ModelIdentity.from_dict(
                _mapping(payload.get("identity"), "robustness.identity")
            ),
            records=tuple(
                RobustnessRecordObservation.from_dict(
                    _mapping(item, "robustness record")
                )
                for item in records
            ),
        )


@dataclass(frozen=True)
class GateResult:
    """One claim, its measured values, explicit assumptions, and fail-closed verdict."""

    key: str
    claim: str
    passed: bool
    values: dict[str, Any]
    assumptions: tuple[str, ...]
    failures: tuple[str, ...]
    fallback: str

    def __post_init__(self) -> None:
        _nonempty(self.key, "gate key")
        _nonempty(self.claim, "gate claim")
        _nonempty(self.fallback, "gate fallback")
        _required_bool(self.passed, f"{self.key}.passed")
        if self.passed != (not self.failures):
            raise ValueError(f"{self.key} passed verdict contradicts its failure list")
        if self.passed and _contains_nonfinite(self.values):
            raise ValueError(f"{self.key} cannot pass with non-finite values")

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class GuaranteeBundle:
    """Complete model-card payload and deployment decision."""

    identity: ModelIdentity
    generated_at: str
    gates: tuple[GateResult, ...]

    def __post_init__(self) -> None:
        keys = tuple(gate.key for gate in self.gates)
        if keys != _REQUIRED_GATE_KEYS:
            raise ValueError(
                "guarantee bundle must contain the six required unique gates in policy order"
            )
        if not isinstance(self.generated_at, str):
            raise TypeError("generated_at must be an ISO-8601 timestamp string")
        try:
            generated = datetime.fromisoformat(self.generated_at)
        except ValueError as exc:
            raise ValueError("generated_at must be an ISO-8601 timestamp") from exc
        if generated.tzinfo is None:
            raise ValueError("generated_at must include a timezone")

    @property
    def deployment_certified(self) -> bool:
        return all(gate.passed for gate in self.gates)

    @property
    def refusal_reasons(self) -> tuple[str, ...]:
        return tuple(
            f"{gate.key}: {failure}"
            for gate in self.gates
            for failure in gate.failures
        )

    def require_deployable(self) -> None:
        """Refuse deployment certification unless every required gate passes."""
        if not self.deployment_certified:
            detail = "; ".join(self.refusal_reasons) or "no certificate gates passed"
            raise CertificationRefused(f"deployment certification refused: {detail}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": self.generated_at,
            "identity": asdict(self.identity),
            "deployment_certified": self.deployment_certified,
            "refusal_reasons": list(self.refusal_reasons),
            "scope_notice": (
                "This is bounded authorization for the six named certificate policies, not a "
                "general model-safety guarantee. Each gate is valid only under its listed "
                "assumptions; tropical selection/readout certificates do not imply stable expert "
                "or model outputs unless output stability explicitly passes."
            ),
            "gates": [gate.to_dict() for gate in self.gates],
        }

    def to_markdown(self) -> str:
        status = "AUTHORIZED" if self.deployment_certified else "REFUSED"
        lines = [
            "# Live Certificate Model Card",
            "",
            f"Bounded certificate-policy deployment verdict: **{status}**",
            "",
            f"- Bundle/certificate run: {_markdown_text(self.identity.run_id)}",
            f"- Declared checkpoint: {_markdown_text(self.identity.checkpoint_id)}",
            f"- Synaptic config hash: {_markdown_text(self.identity.config_hash)}",
            (
                "- Predictive experiment config hash: "
                f"{_markdown_text(self.identity.predictive_config_hash)}"
            ),
            f"- Declared Git revision: {_markdown_text(self.identity.git_sha)}",
            f"- Generated: {_markdown_text(self.generated_at)}",
            "",
            ("This card is not a general model-safety guarantee. It composes narrow runtime "
            "certificates and does not promote a selection "
            "radius into an output guarantee, a local release theorem into predictive calibration, "
            "or a fallback-covered step into a structure-preserving proof."),
            "",
            "| Gate | Verdict | Claim | Fallback |",
            "| --- | --- | --- | --- |",
        ]
        for gate in self.gates:
            verdict = "PASS" if gate.passed else "REFUSED"
            lines.append(
                f"| `{gate.key}` | {verdict} | {_markdown_cell(gate.claim)} | "
                f"{_markdown_cell(gate.fallback)} |"
            )
        for gate in self.gates:
            lines.extend(["", f"## {gate.key}", "", gate.claim, "", "Measured values:", ""])
            measured_json = json.dumps(
                gate.to_dict()["values"],
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            lines.extend(f"    {line}" for line in measured_json.splitlines())
            lines.extend(["", "Assumptions:", ""])
            lines.extend(f"- {_markdown_text(item)}" for item in gate.assumptions)
            if gate.failures:
                lines.extend(["", "Failed gates:", ""])
                lines.extend(f"- {_markdown_text(item)}" for item in gate.failures)
        return "\n".join(lines) + "\n"

    def emit(self, logger: EventLogger) -> None:
        """Write one structured event per gate plus the aggregate verdict."""
        for gate in self.gates:
            logger.event(
                "certificate_gate",
                level="info" if gate.passed else "warning",
                gate=gate.to_dict(),
            )
        logger.event(
            "certificate_bundle",
            level="info" if self.deployment_certified else "warning",
            bundle=self.to_dict(),
        )

    def render(self, console: Console | None = None) -> None:
        """Render a compact Rich certificate table."""
        output = console or Console()
        table = Table(
            title="Live certificate model card",
            caption=(
                "Deployment certification is fail-closed; see model_card.json for values and "
                "assumptions"
            ),
        )
        table.add_column("Gate")
        table.add_column("Verdict")
        table.add_column("Failed assumption / scope")
        for gate in self.gates:
            table.add_row(
                gate.key,
                "[green]PASS[/green]" if gate.passed else "[red]REFUSED[/red]",
                "; ".join(gate.failures) if gate.failures else gate.claim,
            )
        output.print(table)

    def write_artifacts(self, output_dir: str | Path) -> tuple[Path, Path]:
        """Write strict JSON and Markdown model-card artifacts."""
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        json_path = destination / "model_card.json"
        markdown_path = destination / "MODEL_CARD.md"
        json_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        markdown_path.write_text(self.to_markdown(), encoding="utf-8")
        return json_path, markdown_path


def _markdown_text(value: str) -> str:
    """Render untrusted text as one escaped Markdown line."""
    collapsed = " ".join(value.replace("\r", "\n").splitlines())
    controls = frozenset("\\`*_[]<>#|~")
    return "".join(
        f"\\{character}" if character in controls else character
        for character in collapsed
    )


def _markdown_cell(value: str) -> str:
    return _markdown_text(value)


def _certificate_config_errors(cfg: SynapticConfig) -> list[str]:
    """Domains required by the cusp and composition certificate calculations."""
    errors: list[str] = []
    for name in ("tau_c", "tau_buf", "tau_rrp", "post_slow_lr"):
        if getattr(cfg, name) <= 0.0:
            errors.append(f"{name} must be > 0 for certificate calculations")
    if not 0.0 < cfg.post_trace_decay < 1.0:
        errors.append("post_trace_decay must lie strictly in (0, 1)")
    if cfg.structural_interval < 1:
        errors.append("structural_interval must be >= 1")
    if not 0.0 <= cfg.alpha_buf_on <= 1.0:
        errors.append("alpha_buf_on must lie in [0, 1]")
    if not 0.0 <= cfg.alpha_buf_off <= 1.0:
        errors.append("alpha_buf_off must lie in [0, 1]")
    if not 1.0 < cfg.latch_hill_n <= 16.0:
        errors.append("latch_hill_n must lie in (1, 16]")
    if not 0.0 < cfg.latch_hill_k <= 1.0:
        errors.append("latch_hill_k must lie in (0, 1]")
    if not 0.0 < cfg.latch_input_gain <= 500.0:
        errors.append("latch_input_gain must lie in (0, 500]")
    for name in ("camkii_thr", "latch_ltd_thr", "latch_pp1_basal"):
        if not 0.0 <= getattr(cfg, name) <= 1.0:
            errors.append(f"{name} must lie in [0, 1]")
    for name in ("latch_alpha_ca", "latch_beta_pp1", "latch_gamma_auto"):
        if not 0.0 <= getattr(cfg, name) <= 1_000_000.0:
            errors.append(f"{name} must lie in [0, 1000000]")
    if not 0.0 < cfg.cusp_eps_max <= _MAX_CUSP_EPS:
        errors.append(f"cusp_eps_max must lie in (0, {_MAX_CUSP_EPS}]")
    return errors


def _synaptic_config_from_dict(payload: Mapping[str, Any]) -> SynapticConfig:
    config_fields = {
        config_field.name: config_field for config_field in fields(SynapticConfig)
    }
    _exact_keys(payload, set(config_fields), "synaptic_config")
    values: dict[str, Any] = {}
    for name, config_field in config_fields.items():
        value = payload[name]
        default = config_field.default
        if type(default) is bool:
            values[name] = _required_bool(value, f"synaptic_config.{name}")
        elif type(default) is int:
            values[name] = _required_int(
                value,
                f"synaptic_config.{name}",
                minimum=0,
            )
        elif type(default) is float:
            values[name] = _required_finite_float(
                value,
                f"synaptic_config.{name}",
            )
        elif isinstance(default, SynapticGranularity):
            raw = _nonempty(value, f"synaptic_config.{name}")
            try:
                values[name] = SynapticGranularity(raw)
            except ValueError as exc:
                expected = ", ".join(sorted(SUPPORTED_SYNAPTIC_GRANULARITIES))
                raise ValueError(
                    f"synaptic_config.{name} must be one of {expected}"
                ) from exc
        elif type(default) is str:
            values[name] = _nonempty(value, f"synaptic_config.{name}")
        else:  # pragma: no cover - fail closed on future unsupported fields
            raise TypeError(f"unsupported SynapticConfig schema field {name}")
    cfg = SynapticConfig(**values)
    schema_errors = synaptic_config_schema_errors(cfg)
    if schema_errors:
        raise ValueError("invalid SynapticConfig schema: " + "; ".join(schema_errors))
    domain_errors = _certificate_config_errors(cfg)
    if domain_errors:
        raise ValueError("invalid certificate config: " + "; ".join(domain_errors))
    return cfg


def _identity_gate(
    identity: ModelIdentity,
    cfg: SynapticConfig,
    observations: Sequence[ModelIdentity],
) -> GateResult:
    normalized_hash = config_hash(asdict(cfg))
    schema_errors = synaptic_config_schema_errors(cfg)
    if schema_errors:
        config_errors, config_warnings = schema_errors, []
    else:
        config_errors, config_warnings = validate_config(cfg)
    mismatches = [
        f"evidence source {index} does not match the requested model identity"
        for index, observed in enumerate(observations, start=1)
        if observed != identity
    ]
    if normalized_hash != identity.config_hash:
        mismatches.append(
            "identity config_hash does not match the normalized live SynapticConfig"
        )
    mismatches.extend(f"invalid SynapticConfig: {error}" for error in config_errors)
    return GateResult(
        key="provenance",
        claim=(
            "All gate observations consistently declare the bundle identity and the live config "
            "recomputes its digest; predictive target/cohort provenance is retained separately."
        ),
        passed=not mismatches,
        values={
            "identity": asdict(identity),
            "normalized_config_hash": normalized_hash,
            "checkpoint_revision_binding": "operator_attested",
            "evidence_sources": len(observations),
            "config_warnings": config_warnings,
        },
        assumptions=(
            "the operator recomputed checkpoint_id from the deployed artifact and did not relabel runtime records",
            "config_hash covers the complete normalized SynapticConfig",
            "the bundle run ID names this certificate assembly, not every predictive source run",
            "the operator captured evidence without mutating the checkpoint between gates",
            "the local evidence directory and manifest are inside the operator's trusted boundary",
        ),
        failures=tuple(mismatches),
        fallback="refuse the aggregate certificate; retain source artifacts separately",
    )


def _stability_gate(cfg: SynapticConfig, observation: StabilityObservation) -> GateResult:
    threshold = observation.thresholds
    thresholds_trusted = bool(
        threshold.eps_E <= _TRUSTED_STABILITY_THRESHOLDS.eps_E
        and threshold.eps_S <= _TRUSTED_STABILITY_THRESHOLDS.eps_S
        and threshold.eps_D <= _TRUSTED_STABILITY_THRESHOLDS.eps_D
    )
    recomputed_lyapunov = bool(
        _finite(observation.max_free_energy_delta)
        and (observation.max_free_energy_delta or 0.0) <= threshold.eps_E
    )
    failures = _failures(
        (cfg.enable_presyn is True, "presynaptic runtime is disabled or malformed"),
        (
            cfg.metriplectic_integrator is True,
            "metriplectic_integrator is disabled or malformed",
        ),
        (
            observation.source == "torch_runtime" and observation._runtime_verified,
            "stability evidence is not a live in-process torch runtime attestation",
        ),
        (
            thresholds_trusted,
            "stability thresholds exceed the deployment certificate policy",
        ),
        (observation.steps > 0, "no live metriplectic steps were observed"),
        (
            observation.lyapunov_ok and recomputed_lyapunov,
            "free-energy verdict was false or inconsistent with its measured maximum delta",
        ),
        (
            _finite(observation.max_energy_drift)
            and abs(observation.max_energy_drift or 0.0) <= threshold.eps_E,
            "energy drift exceeded eps_E or was non-finite",
        ),
        (
            _finite(observation.min_entropy_production)
            and (observation.min_entropy_production or 0.0) >= -threshold.eps_S,
            "entropy production fell below -eps_S or was non-finite",
        ),
        (
            _finite(observation.max_degeneracy_residual)
            and (observation.max_degeneracy_residual or 0.0) <= threshold.eps_D,
            "a degeneracy residual exceeded eps_D or was non-finite",
        ),
        (
            observation.n_fallbacks == 0,
            "one or more steps used the baseline fallback and are outside the certificate",
        ),
    )
    return GateResult(
        key="metriplectic_stability",
        claim="Observed guarded steps conserve energy, produce entropy, and do not increase F.",
        passed=not failures,
        values={
            "source": observation.source,
            "runtime_attested": observation._runtime_verified,
            "state_updates": observation.steps,
            "max_energy_drift": observation.max_energy_drift,
            "min_entropy_production": observation.min_entropy_production,
            "max_degeneracy_residual": observation.max_degeneracy_residual,
            "max_free_energy_delta": observation.max_free_energy_delta,
            "n_fallbacks": observation.n_fallbacks,
            "lyapunov_ok": observation.lyapunov_ok,
            "thresholds": asdict(threshold),
            "maximum_policy_thresholds": asdict(_TRUSTED_STABILITY_THRESHOLDS),
        },
        assumptions=(
            "the monitor covers the deployment-relevant trajectory rather than a synthetic proxy",
            "the live operators and state dtype match those used to compute the guard ledger",
            "only fallback-free accepted steps carry the structure-preserving claim",
        ),
        failures=failures,
        fallback="use the clamped-Euler path with no metriplectic stability label",
    )


def _retention_gate(cfg: SynapticConfig) -> GateResult:
    certificate = certify_retention(cfg)
    failures = _failures(
        (cfg.enable_hebbian is True, "Hebbian runtime is disabled or malformed"),
        (cfg.bistable_latch is True, "bistable_latch is disabled or malformed"),
        (cfg.cusp_latch is True, "cusp_latch is disabled or malformed"),
        (certificate.certified, certificate.reason),
        (certificate.delta_star > 0.0, "retention half-width delta* is vacuous"),
        (
            not certificate.use_heuristic_fallback,
            "the live latch selected the uncertified heuristic fallback",
        ),
    )
    return GateResult(
        key="cusp_retention",
        claim="The active cusp latch has a non-vacuous retention half-width delta*.",
        passed=not failures,
        values=asdict(certificate),
        assumptions=(
            "the certificate is local to the stated resting operating point",
            "the CaMKII residual is represented by the certified cusp normal form",
            "the fast calcium/buffer subsystem remains inside the normal-hyperbolicity gate",
        ),
        failures=failures,
        fallback="use the heuristic latch with delta*=0 and no retention guarantee",
    )


def _head_predictive_failures(
    head: HeadPredictiveThermoEvidence,
    policy: PredictiveEvidencePolicy,
) -> tuple[str, ...]:
    expected_fraction = (
        head.tested_events / head.observed_events if head.observed_events else 0.0
    )
    recomputed_tur_ratio = (
        head.tur_relative_variance / head.tur_entropy_bound
        if head.tur_relative_variance is not None
        and math.isfinite(head.tur_relative_variance)
        and head.tur_entropy_bound is not None
        and math.isfinite(head.tur_entropy_bound)
        and head.tur_entropy_bound > 0.0
        else None
    )
    computed_reasons: list[str] = []
    if head.sample_count < policy.min_samples:
        computed_reasons.append("under_sampled")
    if head.tested_events == 0:
        computed_reasons.append("no_tested_events")
    if not head.sampling_modes or not set(head.sampling_modes) <= _SUPPORTED_PREDICTIVE_MODES:
        computed_reasons.append("unsupported_sampling_mode")
    if expected_fraction < policy.min_tested_fraction:
        computed_reasons.append("under_covered")
    if head.symmetric_bins < policy.min_symmetric_bins:
        computed_reasons.append("insufficient_symmetric_support")
    if head.crooks_residual is None or head.crooks_residual > policy.crooks_tolerance:
        computed_reasons.append("crooks_gate_failed")
    if (
        recomputed_tur_ratio is None
        or recomputed_tur_ratio < policy.min_tur_bound_ratio
    ):
        computed_reasons.append("tur_gate_failed")
    finite_values = (
        expected_fraction,
        head.crooks_residual,
        head.tur_relative_variance,
        head.tur_entropy_bound,
        recomputed_tur_ratio,
    )
    computed_finite = all(value is not None and math.isfinite(value) for value in finite_values)
    if not computed_finite:
        computed_reasons.append("non_finite_evidence")

    failures: list[str] = []
    if head.tested_events + head.degenerate_events != head.observed_events:
        failures.append("event counts do not conserve observed coverage")
    if head.retained_events > head.tested_events:
        failures.append("retained event count exceeds tested event count")
    if head.retained_events > policy.max_events_per_head:
        failures.append("retained event count exceeds the bounded reservoir policy")
    if head.retained_events != min(head.tested_events, policy.max_events_per_head):
        failures.append(
            "retained event count contradicts the collector reservoir invariant"
        )
    if len(set(head.sampling_modes)) != len(head.sampling_modes):
        failures.append("sampling_modes contains duplicates")
    if head.symmetric_bins > policy.crooks_bins // 2:
        failures.append("symmetric bin count exceeds the declared histogram support")
    minimum_crooks_support = (
        2 * head.symmetric_bins * policy.crooks_min_count
    )
    if head.retained_events < minimum_crooks_support:
        failures.append(
            "retained events cannot support the declared symmetric Crooks bins"
        )
    if head.tested_fraction != expected_fraction:
        failures.append("tested_fraction contradicts event counts")
    if head.crooks_residual is not None and head.crooks_residual < 0.0:
        failures.append("Crooks residual must be non-negative")
    if head.tur_relative_variance is not None and head.tur_relative_variance < 0.0:
        failures.append("TUR relative variance must be non-negative")
    if head.tur_entropy_bound is not None and head.tur_entropy_bound <= 0.0:
        failures.append("TUR entropy bound must be positive")
    if head.tur_bound_ratio is not None and head.tur_bound_ratio < 0.0:
        failures.append("TUR bound ratio must be non-negative")
    if (
        head.tur_relative_variance is not None
        and head.tur_entropy_bound is not None
        and head.tur_entropy_bound > 0.0
        and head.tur_bound_ratio != recomputed_tur_ratio
    ):
        failures.append("TUR bound ratio contradicts variance/entropy measurements")
    if head.finite != computed_finite:
        failures.append("finite verdict contradicts measured values")
    if head.refusal_reasons != tuple(computed_reasons):
        failures.append("refusal reasons contradict the declared policy and measurements")
    if head.passed != (not computed_reasons):
        failures.append("head pass verdict contradicts the recomputed local gates")
    return tuple(failures)


def _predictive_gate(
    cfg: SynapticConfig, observation: PredictiveCalibrationObservation
) -> GateResult:
    seeds = observation.seeds
    target_evidence = observation.target.evidence
    evidences = tuple(seed.evidence for seed in seeds)
    local_evidences = (target_evidence, *evidences)
    provenances = tuple(evidence.provenance for evidence in evidences)
    run_ids = {item.run_id for item in provenances}
    rng_seeds = {item.rng_seed for item in provenances}
    checkpoint_ids = {item.checkpoint_id for item in provenances}
    synaptic_config_hashes = {item.synaptic_config_hash for item in provenances}
    config_hashes = {item.config_hash for item in provenances}
    source_records_match = bool(
        observation._runtime_verified
        and observation._source_target == observation.target
        and observation._source_target_calibration == observation.target_calibration
        and observation._source_seeds == observation.seeds
        and observation._source_statistics == observation.statistics
    )
    target_calibration = observation.target_calibration
    target_calibration_bound = bool(
        target_calibration.target_artifact_sha256
        == observation.target.artifact_sha256
        and target_calibration.target_run_id
        == target_evidence.provenance.run_id
        and target_calibration.target_checkpoint_id
        == target_evidence.provenance.checkpoint_id
        and target_calibration.target_synaptic_config_hash
        == target_evidence.provenance.synaptic_config_hash
        and target_calibration.target_experiment_config_hash
        == target_evidence.provenance.config_hash
        and target_calibration.target_rng_seed
        == target_evidence.provenance.rng_seed
    )
    recomputed_target_calibration_pass = bool(
        target_calibration.thermo_ece <= _TARGET_ECE_MAX
        and target_calibration.thermo_ood_auroc >= _TARGET_OOD_AUROC_MIN
        and target_calibration.thermo_ece < target_calibration.softmax_ece
        and target_calibration.thermo_ece < target_calibration.mc_dropout_ece
        and target_calibration.thermo_ood_auroc
        > target_calibration.softmax_ood_auroc
        and target_calibration.thermo_ood_auroc
        > target_calibration.mc_dropout_ood_auroc
    )
    target_calibration_valid = bool(
        target_calibration.passed == recomputed_target_calibration_pass
        and recomputed_target_calibration_pass
    )
    cohort_experiment_seeds = tuple(item.rng_seed - 301 for item in provenances)
    cohort_policy_valid = cohort_experiment_seeds == _PREDICTIVE_DEPLOYMENT_SEEDS
    target_cohort_separate = bool(
        target_evidence.provenance.run_id not in run_ids
        and target_evidence.provenance.rng_seed not in rng_seeds
        and target_evidence.provenance.checkpoint_id not in checkpoint_ids
    )
    cohort_checkpoints_unique = len(checkpoint_ids) == len(seeds)
    modes = {
        mode
        for evidence in local_evidences
        for head in evidence.heads
        for mode in head.sampling_modes
    }
    expected_sites = set(observation.expected_sites)
    failures = list(
        _failures(
            (cfg.enable_presyn is True, "presynaptic runtime is disabled or malformed"),
            (
                source_records_match,
                "predictive evidence is not bound to explicit live target/cohort producer reports",
            ),
            (
                target_calibration_valid,
                "deployed target failed the fixed point-estimate ECE/OOD-AUROC policy",
            ),
            (
                target_calibration_bound,
                "deployed target calibration is not bound to its predictive artifact/provenance",
            ),
            (
                target_cohort_separate,
                "deployed target run/RNG/checkpoint overlaps the research cohort",
            ),
            (
                cohort_checkpoints_unique,
                "predictive cohort checkpoint digests are duplicated (pseudoreplication)",
            ),
            (
                cohort_policy_valid,
                "predictive cohort does not match the fixed deployment seed policy",
            ),
            (len(seeds) >= 2, "fewer than two predictive evidence seeds were supplied"),
            (len(run_ids) == len(seeds), "predictive run IDs are duplicated"),
            (len(rng_seeds) == len(seeds), "predictive RNG seeds are duplicated"),
            (
                target_evidence.provenance.checkpoint_id
                == observation.identity.checkpoint_id,
                "target predictive evidence is not bound to the deployed checkpoint digest",
            ),
            (
                target_evidence.provenance.synaptic_config_hash
                == observation.identity.config_hash,
                "target predictive evidence is not bound to the live SynapticConfig",
            ),
            (
                target_evidence.provenance.config_hash
                == observation.identity.predictive_config_hash,
                "target predictive evidence is not bound to the predictive protocol config",
            ),
            (
                synaptic_config_hashes == {observation.identity.config_hash},
                "predictive cohort used a different normalized SynapticConfig",
            ),
            (
                config_hashes == {observation.identity.predictive_config_hash},
                "predictive evidence is not bound to the declared predictive protocol config",
            ),
            (
                all(_HEX_64_RE.fullmatch(item) for item in checkpoint_ids),
                "predictive evidence contains a malformed checkpoint digest",
            ),
            (
                all(evidence.fresh for evidence in local_evidences),
                "target or cohort predictive evidence is stale",
            ),
            (
                bool(modes) and modes <= _SUPPORTED_PREDICTIVE_MODES,
                "predictive evidence used no exact release mode or an unsupported mode",
            ),
            (
                all(
                    cfg.stochastic_mode in head.sampling_modes
                    for evidence in local_evidences
                    for head in evidence.heads
                ),
                "the live stochastic_mode was absent from one or more layer/head artifacts",
            ),
            (
                all(
                    evidence.policy == PredictiveEvidencePolicy()
                    for evidence in local_evidences
                ),
                "one or more predictive artifacts used a non-deployment evidence policy",
            ),
        )
    )

    locally_valid: list[bool] = []
    for index, evidence in enumerate(local_evidences):
        evidence_label = "target" if index == 0 else f"cohort seed {index}"
        sites = {(head.layer_address, head.head_index) for head in evidence.heads}
        seed_failures: list[str] = []
        if sites != expected_sites or len(sites) != len(evidence.heads):
            seed_failures.append("layer/head coverage does not exactly match expected_sites")
        for head in evidence.heads:
            seed_failures.extend(
                f"{head.layer_address}/h{head.head_index}: {failure}"
                for failure in _head_predictive_failures(head, evidence.policy)
            )
        totals = (
            sum(head.observed_events for head in evidence.heads),
            sum(head.tested_events for head in evidence.heads),
            sum(head.retained_events for head in evidence.heads),
            sum(head.degenerate_events for head in evidence.heads),
        )
        if totals != (
            evidence.observed_events,
            evidence.tested_events,
            evidence.retained_events,
            evidence.degenerate_events,
        ):
            seed_failures.append("seed aggregate event counts contradict its head records")
        expected_fraction = (
            evidence.tested_events / evidence.observed_events
            if evidence.observed_events
            else 0.0
        )
        if evidence.tested_fraction != expected_fraction:
            seed_failures.append("seed tested_fraction contradicts aggregate event counts")
        computed_local = bool(evidence.heads) and all(head.passed for head in evidence.heads)
        if evidence.local_gates_passed != computed_local:
            seed_failures.append("seed local_gates_passed contradicts its head verdicts")
        expected_seed_claim = bool(
            evidence.fresh
            and computed_local
            and evidence.multi_seed_statistics_passed
        )
        if evidence.predictive_distribution_claim != expected_seed_claim:
            seed_failures.append("seed predictive claim contradicts its component gates")
        expected_mode = (
            "predictive_thermodynamic_calibration"
            if expected_seed_claim
            else "empirical_ece_fallback"
        )
        if evidence.calibration_mode != expected_mode:
            seed_failures.append("seed calibration_mode contradicts its component gates")
        expected_reasons: list[str] = []
        if not evidence.heads:
            expected_reasons.append("empty_evidence")
        if not evidence.fresh:
            expected_reasons.append("stale_evidence")
        if evidence.heads and not computed_local:
            expected_reasons.append("local_layer_head_gates_failed")
        if not evidence.multi_seed_statistics_passed:
            expected_reasons.append("multi_seed_statistics_pending_or_failed")
        if evidence.refusal_reasons != tuple(expected_reasons):
            seed_failures.append("seed refusal reasons contradict its component gates")
        finalized_claim_valid = bool(
            evidence.multi_seed_statistics_passed
            and evidence.predictive_distribution_claim
            and evidence.calibration_mode
            == "predictive_thermodynamic_calibration"
            and not evidence.refusal_reasons
        )
        if not finalized_claim_valid:
            seed_failures.append(
                "predictive artifact does not carry the finalized passing group claim"
            )
        locally_valid.append(
            not seed_failures and computed_local and finalized_claim_valid
        )
        failures.extend(
            f"{evidence_label}: {failure}" for failure in seed_failures
        )

    statistics = observation.statistics
    evidence_run_id_tuple = tuple(item.run_id for item in provenances)
    evidence_checkpoint_id_tuple = tuple(item.checkpoint_id for item in provenances)
    evidence_synaptic_config_hash_tuple = tuple(
        item.synaptic_config_hash for item in provenances
    )
    evidence_config_hash_tuple = tuple(item.config_hash for item in provenances)
    evidence_rng_seed_tuple = tuple(item.rng_seed for item in provenances)
    statistics_bound = bool(
        statistics.identity == observation.identity
        and statistics.predictive_run_ids == evidence_run_id_tuple
        and statistics.predictive_checkpoint_ids == evidence_checkpoint_id_tuple
        and statistics.predictive_synaptic_config_hashes
        == evidence_synaptic_config_hash_tuple
        and statistics.predictive_config_hashes == evidence_config_hash_tuple
        and statistics.predictive_rng_seeds == evidence_rng_seed_tuple
        and statistics.expected_sites == observation.expected_sites
    )
    failures.extend(
        _failures(
            (
                statistics_bound,
                "statistics artifact is not bound to this model identity, predictive cohort, and site set",
            ),
        )
    )
    required_comparisons = {
        ("softmax_entropy", "ece"),
        ("softmax_entropy", "ood_auroc"),
        ("mc_dropout", "ece"),
        ("mc_dropout", "ood_auroc"),
    }
    observed_comparisons = {
        (comparison.baseline, comparison.metric)
        for comparison in statistics.comparisons
    }
    comparison_results: list[dict[str, Any]] = []
    comparison_matrix_valid = bool(
        observed_comparisons == required_comparisons
        and len(statistics.comparisons) == len(required_comparisons)
    )
    comparisons_pass = comparison_matrix_valid
    for comparison in statistics.comparisons:
        comparison_shape_valid = bool(
            comparison_matrix_valid
            and comparison.seed_count == len(seeds)
            and len(comparison.paired_deltas) == len(seeds)
        )
        if not comparison_shape_valid:
            comparisons_pass = False
            comparison_results.append(
                {
                    "baseline": comparison.baseline,
                    "metric": comparison.metric,
                    "reported_values_match": False,
                    "passed": False,
                    "recomputation_skipped": (
                        "invalid comparison matrix or cohort-sized delta vector"
                    ),
                }
            )
            continue
        deltas = np.asarray(comparison.paired_deltas, dtype=np.float64)
        _, recomputed_t_p = paired_t_test(deltas)
        recomputed_wilcoxon_p = wilcoxon_signed_rank(deltas)
        recomputed_ci_low, recomputed_ci_high = bootstrap_ci(
            deltas,
            n_boot=comparison.bootstrap_samples,
            seed=comparison.bootstrap_seed,
        )
        reported_match = all(
            math.isclose(reported, recomputed, rel_tol=1e-12, abs_tol=1e-12)
            for reported, recomputed in (
                (comparison.paired_t_p_value, recomputed_t_p),
                (comparison.wilcoxon_p_value, recomputed_wilcoxon_p),
                (comparison.effect_ci_low, recomputed_ci_low),
                (comparison.effect_ci_high, recomputed_ci_high),
            )
        )
        ci_favorable = (
            recomputed_ci_high < 0.0
            if comparison.favorable_direction == "lower"
            else recomputed_ci_low > 0.0
        )
        recomputed_comparison_pass = bool(
            int(np.count_nonzero(deltas)) >= 6
            and reported_match
            and recomputed_t_p <= statistics.alpha
            and recomputed_wilcoxon_p <= statistics.alpha
            and ci_favorable
        )
        comparison_consistent = bool(
            comparison.passed == recomputed_comparison_pass
            and recomputed_comparison_pass
        )
        comparisons_pass = comparisons_pass and comparison_consistent
        comparison_results.append(
            {
                "baseline": comparison.baseline,
                "metric": comparison.metric,
                "paired_t_p_value": recomputed_t_p,
                "wilcoxon_p_value": recomputed_wilcoxon_p,
                "effect_ci_low": recomputed_ci_low,
                "effect_ci_high": recomputed_ci_high,
                "nonzero_pairs": int(np.count_nonzero(deltas)),
                "reported_values_match": reported_match,
                "passed": recomputed_comparison_pass,
            }
        )

    evidence_by_run_id = {item.run_id: item for item in provenances}
    observed_ft_run_ids = {
        item.paired_predictive_run_id for item in statistics.live_ft_seeds
    }
    observed_ft_source_provenance = {
        (
            item.experiment_seed,
            item.paired_predictive_rng_seed,
            item.forward_rng_seed,
            item.reverse_rng_seed,
        )
        for item in statistics.live_ft_seeds
    }
    live_tur = statistics.live_tur
    live_ft_results: list[dict[str, Any]] = []
    live_ft_pass = bool(
        len(statistics.live_ft_seeds) == len(seeds)
        and len(observed_ft_run_ids) == len(statistics.live_ft_seeds)
        and observed_ft_run_ids == set(evidence_by_run_id)
        and len(observed_ft_source_provenance) == len(statistics.live_ft_seeds)
    )
    for ft_seed in statistics.live_ft_seeds:
        bound_provenance = evidence_by_run_id.get(ft_seed.paired_predictive_run_id)
        ft_cohort_pair_bound = bool(
            bound_provenance is not None
            and ft_seed.experiment_config_hash == bound_provenance.config_hash
            and ft_seed.paired_predictive_rng_seed == bound_provenance.rng_seed
        )
        expected_release_protocol_hash = config_hash(
            asdict(
                SynapticConfig(
                    stochastic_train_frac=1.0,
                    stochastic_mode="straight_through",
                    stochastic_count_cap=8,
                    prime_rate=0.0,
                    endo_delay=0,
                    init_rrp=6.0,
                    rec_rate=ft_seed.configured_reverse_probability,
                )
            )
        )
        recomputed_ft_affinity = (
            math.log(ft_seed.forward_probability)
            + math.log1p(-ft_seed.reverse_probability)
            - math.log(ft_seed.reverse_probability)
            - math.log1p(-ft_seed.forward_probability)
        )
        ft_protocol_valid = bool(
            math.isfinite(recomputed_ft_affinity)
            and recomputed_ft_affinity > 0.0
            and ft_seed.release_protocol_config_hash
            == expected_release_protocol_hash
            and math.isclose(
                ft_seed.forward_probability,
                ft_seed.configured_forward_probability,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            and math.isclose(
                ft_seed.reverse_probability,
                ft_seed.configured_reverse_probability,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            and len(ft_seed.current_counts) == 2 * ft_seed.pool_size + 1
            and sum(ft_seed.current_counts) == ft_seed.n_trajectories
        )
        integral_terms: list[float] = []
        if ft_protocol_valid:
            try:
                for current, count in zip(
                    range(-ft_seed.pool_size, ft_seed.pool_size + 1),
                    ft_seed.current_counts,
                    strict=True,
                ):
                    if count:
                        integral_terms.append(
                            count * math.exp(-current * recomputed_ft_affinity)
                        )
                recomputed_integral_ft = (
                    math.fsum(integral_terms) / ft_seed.n_trajectories
                )
            except (OverflowError, ValueError):
                recomputed_integral_ft = None
        else:
            recomputed_integral_ft = None
        if recomputed_integral_ft is not None and not math.isfinite(
            recomputed_integral_ft
        ):
            recomputed_integral_ft = None
        recomputed_integral_residual = (
            abs(recomputed_integral_ft - 1.0)
            if recomputed_integral_ft is not None
            else None
        )

        recomputed_curve: list[dict[str, Any]] = []
        for current in range(1, ft_seed.pool_size + 1):
            positive_count = ft_seed.current_counts[ft_seed.pool_size + current]
            negative_count = ft_seed.current_counts[ft_seed.pool_size - current]
            if (
                positive_count < ft_seed.crooks_min_count
                or negative_count < ft_seed.crooks_min_count
            ):
                continue
            observed_log_ratio = math.log(positive_count / negative_count)
            expected_log_ratio = current * recomputed_ft_affinity
            recomputed_curve.append(
                {
                    "current": current,
                    "positive_count": positive_count,
                    "negative_count": negative_count,
                    "observed_log_ratio": observed_log_ratio,
                    "expected_log_ratio": expected_log_ratio,
                    "residual": observed_log_ratio - expected_log_ratio,
                }
            )
        curve_values_match = len(ft_seed.crooks_curve) == len(recomputed_curve)
        if curve_values_match:
            for reported, recomputed in zip(
                ft_seed.crooks_curve, recomputed_curve, strict=True
            ):
                curve_values_match = bool(
                    reported.current == recomputed["current"]
                    and reported.positive_count == recomputed["positive_count"]
                    and reported.negative_count == recomputed["negative_count"]
                    and math.isclose(
                        reported.observed_log_ratio,
                        cast(float, recomputed["observed_log_ratio"]),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                    and math.isclose(
                        reported.expected_log_ratio,
                        cast(float, recomputed["expected_log_ratio"]),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                    and math.isclose(
                        reported.residual,
                        cast(float, recomputed["residual"]),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                )
                if not curve_values_match:
                    break
        recomputed_max_crooks_residual = max(
            (
                abs(cast(float, point["residual"]))
                for point in recomputed_curve
            ),
            default=None,
        )
        max_crooks_summary_matches = bool(
            ft_seed.max_crooks_residual is None
            and recomputed_max_crooks_residual is None
        ) or bool(
            ft_seed.max_crooks_residual is not None
            and recomputed_max_crooks_residual is not None
            and math.isclose(
                ft_seed.max_crooks_residual,
                recomputed_max_crooks_residual,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )
        ft_summaries_match = bool(
            math.isclose(
                ft_seed.affinity,
                recomputed_ft_affinity,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            and recomputed_integral_ft is not None
            and recomputed_integral_residual is not None
            and max_crooks_summary_matches
            and math.isclose(
                ft_seed.integral_ft,
                recomputed_integral_ft,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            and math.isclose(
                ft_seed.integral_ft_residual,
                recomputed_integral_residual,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )
        ft_protocol_matches_tur = bool(
            ft_seed.pool_size == live_tur.pool_size
            and ft_seed.forward_probability == live_tur.forward_probability
            and ft_seed.reverse_probability == live_tur.reverse_probability
            and ft_seed.affinity == live_tur.affinity
        )
        recomputed_ft_pass = bool(
            ft_protocol_valid
            and ft_cohort_pair_bound
            and ft_protocol_matches_tur
            and bool(recomputed_curve)
            and curve_values_match
            and ft_summaries_match
            and recomputed_max_crooks_residual is not None
            and recomputed_max_crooks_residual <= _LIVE_CROOKS_TOLERANCE
            and recomputed_integral_residual is not None
            and recomputed_integral_residual <= _LIVE_INTEGRAL_FT_TOLERANCE
        )
        ft_consistent = bool(ft_seed.passed == recomputed_ft_pass and recomputed_ft_pass)
        live_ft_pass = live_ft_pass and ft_consistent
        live_ft_results.append(
            {
                **asdict(ft_seed),
                "protocol_valid": ft_protocol_valid,
                "cohort_pair_bound": ft_cohort_pair_bound,
                "expected_release_protocol_config_hash": (
                    expected_release_protocol_hash
                ),
                "protocol_matches_tur": ft_protocol_matches_tur,
                "curve_values_match": curve_values_match,
                "summaries_match": ft_summaries_match,
                "recomputed_affinity": recomputed_ft_affinity,
                "recomputed_integral_ft": recomputed_integral_ft,
                "recomputed_integral_ft_residual": recomputed_integral_residual,
                "recomputed_curve": recomputed_curve,
                "recomputed_max_crooks_residual": recomputed_max_crooks_residual,
                "recomputed_passed": recomputed_ft_pass,
            }
        )

    recomputed_affinity = (
        math.log(live_tur.forward_probability)
        + math.log1p(-live_tur.reverse_probability)
        - math.log(live_tur.reverse_probability)
        - math.log1p(-live_tur.forward_probability)
    )
    mean_current = live_tur.pool_size * (
        live_tur.forward_probability - live_tur.reverse_probability
    )
    current_variance = live_tur.pool_size * (
        live_tur.forward_probability * (1.0 - live_tur.forward_probability)
        + live_tur.reverse_probability * (1.0 - live_tur.reverse_probability)
    )
    mean_current_squared = mean_current * mean_current
    mean_entropy = mean_current * recomputed_affinity
    tur_primitives_valid = bool(
        math.isfinite(recomputed_affinity)
        and recomputed_affinity > 0.0
        and math.isfinite(mean_current)
        and mean_current > 0.0
        and math.isfinite(mean_current_squared)
        and mean_current_squared > 0.0
        and math.isfinite(current_variance)
        and current_variance > 0.0
        and math.isfinite(mean_entropy)
        and mean_entropy > 0.0
    )
    recomputed_relative_variance = (
        current_variance / mean_current_squared if tur_primitives_valid else None
    )
    recomputed_entropy_bound = (
        2.0 / mean_entropy if tur_primitives_valid else None
    )
    tur_slack = (
        recomputed_relative_variance - recomputed_entropy_bound
        if recomputed_relative_variance is not None
        and recomputed_entropy_bound is not None
        else None
    )
    tur_ratio = (
        recomputed_relative_variance / recomputed_entropy_bound
        if recomputed_relative_variance is not None
        and recomputed_entropy_bound is not None
        and recomputed_entropy_bound > 0.0
        else None
    )
    recomputed_tur_nonvacuous = bool(
        recomputed_entropy_bound is not None
        and math.isfinite(recomputed_entropy_bound)
        and recomputed_entropy_bound > 0.0
    )
    recomputed_tur_satisfied = bool(
        recomputed_tur_nonvacuous
        and tur_slack is not None
        and tur_slack >= 0.0
    )
    tur_consistent = bool(
        tur_primitives_valid
        and recomputed_relative_variance is not None
        and recomputed_entropy_bound is not None
        and tur_slack is not None
        and tur_ratio is not None
        and math.isclose(
            live_tur.affinity, recomputed_affinity, rel_tol=1e-12, abs_tol=1e-12
        )
        and math.isclose(
            live_tur.relative_variance,
            recomputed_relative_variance,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        and math.isclose(
            live_tur.entropy_bound,
            recomputed_entropy_bound,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        and math.isclose(
            live_tur.slack, tur_slack, rel_tol=1e-12, abs_tol=1e-12
        )
        and math.isclose(
            live_tur.bound_ratio, tur_ratio, rel_tol=1e-12, abs_tol=1e-12
        )
        and live_tur.nonvacuous == recomputed_tur_nonvacuous
        and live_tur.satisfied == recomputed_tur_satisfied
        and recomputed_tur_satisfied
    )
    recomputed_statistics_pass = bool(
        statistics_bound
        and statistics.fixed_policy_applied
        and statistics.alpha == _PREDICTIVE_ALPHA
        and comparisons_pass
        and live_ft_pass
        and tur_consistent
    )
    failures.extend(
        _failures(
            (
                comparison_matrix_valid,
                "statistics must contain exactly ECE and OOD-AUROC against both canonical baselines",
            ),
            (
                comparisons_pass,
                "one or more canonical comparison statistics failed recomputation",
            ),
            (
                live_ft_pass,
                "not every predictive seed passed the live fluctuation-theorem gate",
            ),
            (
                tur_consistent,
                "live TUR measurements are failed or internally inconsistent",
            ),
            (
                statistics.passed == recomputed_statistics_pass
                and recomputed_statistics_pass,
                "the fixed-policy matched-seed statistical gate failed or is inconsistent",
            ),
            (
                all(locally_valid),
                "one or more complete layer/head evidence artifacts failed recomputation",
            ),
        )
    )
    return GateResult(
        key="predictive_calibration",
        claim=(
            "The deployed target passes fixed local point-estimate ECE/OOD-AUROC and "
            "per-layer/head gates, while the fixed matched-seed cohort supplies separate "
            "population-level statistical support."
        ),
        passed=not failures,
        values={
            "seed_count": len(seeds),
            "runtime_attested": observation._runtime_verified,
            "source_records_match": source_records_match,
            "target_run_id": target_evidence.provenance.run_id,
            "target_checkpoint_id": target_evidence.provenance.checkpoint_id,
            "target_evidence": {
                "artifact_sha256": observation.target.artifact_sha256,
                "evidence": asdict(target_evidence),
            },
            "target_calibration": {
                **asdict(target_calibration),
                "ece_max": _TARGET_ECE_MAX,
                "ood_auroc_min": _TARGET_OOD_AUROC_MIN,
                "target_binding_matches": target_calibration_bound,
                "recomputed_passed": recomputed_target_calibration_pass,
            },
            "target_cohort_separate": target_cohort_separate,
            "cohort_checkpoints_unique": cohort_checkpoints_unique,
            "cohort_experiment_seeds": cohort_experiment_seeds,
            "required_cohort_experiment_seeds": _PREDICTIVE_DEPLOYMENT_SEEDS,
            "run_ids": sorted(run_ids),
            "rng_seeds": sorted(rng_seeds),
            "sampling_modes": sorted(modes),
            "expected_sites": [list(site) for site in observation.expected_sites],
            "cohort_artifact_sha256": [seed.artifact_sha256 for seed in seeds],
            "local_seed_pass_rate": (
                sum(locally_valid) / len(local_evidences)
                if local_evidences
                else 0.0
            ),
            "statistics": asdict(statistics),
            "recomputed_comparisons": comparison_results,
            "recomputed_live_ft": live_ft_results,
            "recomputed_live_tur": {
                "primitives_valid": tur_primitives_valid,
                "affinity": recomputed_affinity,
                "relative_variance": recomputed_relative_variance,
                "entropy_bound": recomputed_entropy_bound,
                "slack": tur_slack,
                "bound_ratio": tur_ratio,
                "nonvacuous": recomputed_tur_nonvacuous,
                "satisfied": recomputed_tur_satisfied,
            },
            "calibration_mode": (
                "predictive_thermodynamic_calibration"
                if not failures
                else "empirical_ece_fallback"
            ),
        },
        assumptions=(
            "release counts use an exact supported estimator rather than an approximate reparameterization",
            "every claimed layer/head has fresh symmetric Crooks support and a non-vacuous TUR check",
            "the fixed statistical policy was applied to matched held-out prediction data",
            "cohort membership is the fixed ordered deployment seed policy, not a selected subset",
            "target and cohort metrics are limited to the exact captured evaluation distribution and do not transfer to deployment traffic",
        ),
        failures=tuple(failures),
        fallback="report empirical ECE/uncertainty only; drop the thermodynamic predictive claim",
    )


def _robustness_gate(cfg: SynapticConfig, observation: RobustnessObservation) -> GateResult:
    records = observation.records
    source_records_match = bool(
        observation._runtime_verified
        and len(observation._source_records) == len(records)
        and tuple(
            RobustnessRecordObservation.from_record(record)
            for record in observation._source_records
        )
        == records
    )
    record_failures: list[str] = []
    for record in records:
        scope = CertificateScope(record.scope)
        radius_ok = bool(
            (
                record.certified_radius_unbounded
                and record.certified_radius is None
            )
            or (
                not record.certified_radius_unbounded
                and _finite(record.certified_radius)
                and (record.certified_radius or 0.0) > 0.0
            )
        )
        base_certified = bool(
            record.selection_certified
            and record.lipschitz_certified
            and record.artifacts_bound
        )
        if scope is CertificateScope.MOE_TOPK_MEMBERSHIP:
            scope_consistent = bool(
                record.readout_certified is None
                and record.certified == base_certified
                and not record.output_stability_certified
            )
        elif scope is CertificateScope.ATTENTION_HARD_READOUT:
            scope_consistent = bool(
                record.readout_certified is not None
                and record.certified == record.readout_certified
                and record.output_stability_certified
                == bool(record.certified and record.values_frozen)
            )
        else:
            scope_consistent = bool(
                record.readout_certified is not None
                and record.certified == record.readout_certified
                and not record.output_stability_certified
            )
        valid = bool(
            record.exact_affine
            and record.replayable
            and radius_ok
            and base_certified
            and record.certified
            and scope_consistent
            and hmac.compare_digest(
                record.slope_digest, record.lipschitz_slope_digest
            )
            and record.reason == "certified"
        )
        if not valid:
            record_failures.append(
                f"step {record.step} ({record.scope}) failed recomputed scope/binding/radius gates: "
                f"{record.reason}"
            )
    failures = list(
        _failures(
            (
                cfg.tropical_skeleton is True,
                "tropical_skeleton is disabled or malformed",
            ),
            (
                source_records_match,
                "tropical evidence is not bound to complete live in-process monitor records",
            ),
            (bool(records), "no tropical runtime certificate records were supplied"),
        )
    )
    failures.extend(record_failures)
    finite_radii = [
        record.certified_radius
        for record in records
        if record.certified_radius is not None and math.isfinite(record.certified_radius)
    ]
    return GateResult(
        key="tropical_robustness",
        claim=(
            "The listed exact affine decision scopes retain their selections/readouts inside the "
            "reported strict-ball radii."
        ),
        passed=not failures,
        values={
            "records": len(records),
            "runtime_attested": observation._runtime_verified,
            "source_records_match": source_records_match,
            "scopes": sorted({record.scope for record in records}),
            "input_norms": sorted({record.input_norm for record in records}),
            "min_finite_certified_radius": min(finite_radii) if finite_radii else None,
            "unbounded_radius_records": sum(
                record.certified_radius_unbounded for record in records
            ),
            "output_stability_all": bool(records)
            and all(record.output_stability_certified for record in records),
            "record_summaries": [asdict(record) for record in records],
            "live_record_details": [
                asdict(record) for record in observation._source_records
            ],
        },
        assumptions=(
            "the protected score family is exactly affine on the certified cell and replayable",
            "the threat model is the per-record input norm and protected decision scope",
            "stable selection does not imply stable selected-expert output",
            "attention output stability is claimed only where values were explicitly frozen",
        ),
        failures=tuple(failures),
        fallback="retain the soft/default router and report an uncertified pointwise fingerprint",
    )


def _composition_gate(cfg: SynapticConfig, *, eps_max: float) -> GateResult:
    eligibility = composition_eligibility(cfg, eps_max=eps_max)
    required = {key: eligibility[key] for key in _REQUIRED_COMPOSITION_THRUSTS}
    pairs = {
        f"{left}+{right}": pairwise_compatible(
            cfg, left, right, eps_max=eps_max
        )
        for index, left in enumerate(_REQUIRED_COMPOSITION_THRUSTS)
        for right in _REQUIRED_COMPOSITION_THRUSTS[index + 1 :]
    }
    failures = tuple(
        [
            f"thrust {key} is ineligible at {row.boundary} (eps={row.eps:.6g})"
            for key, row in required.items()
            if not row.eligible
        ]
        + [
            f"pair {key} is incompatible: {row.reason}"
            for key, row in pairs.items()
            if not row.compatible
        ]
    )
    return GateResult(
        key="composition",
        claim="The A/E/F certificates may compose under the configured timescale proxy.",
        passed=not failures,
        values={
            "eps_max": eps_max,
            "eligibility": {key: asdict(value) for key, value in required.items()},
            "pairwise": {key: asdict(value) for key, value in pairs.items()},
        },
        assumptions=(
            "configured timescales are an accepted proxy for the live deployment dynamics",
            "every boundary crossed by A/E/F remains below eps_max",
            "the tropical H certificate is separately gated by its exact-affine runtime evidence",
        ),
        failures=failures,
        fallback="disable the higher-risk incompatible thrust(s) named by the pairwise harness",
    )


def build_guarantee_bundle(
    *,
    identity: ModelIdentity,
    config: SynapticConfig,
    stability: StabilityObservation,
    predictive_calibration: PredictiveCalibrationObservation,
    robustness: RobustnessObservation,
    eps_max: float = 0.5,
    generated_at: str | None = None,
) -> GuaranteeBundle:
    """Compose all required gates without silently weakening any source certificate."""
    schema_errors = synaptic_config_schema_errors(config)
    if schema_errors:
        raise ValueError("invalid SynapticConfig schema: " + "; ".join(schema_errors))
    domain_errors = _certificate_config_errors(config)
    if domain_errors:
        raise ValueError("invalid certificate config: " + "; ".join(domain_errors))
    normalized_eps_max = _required_finite_float(
        eps_max, "eps_max", minimum=0.0, maximum=_MAX_COMPOSITION_EPS
    )
    if normalized_eps_max == 0.0:
        raise ValueError(
            f"eps_max must be finite and lie in (0, {_MAX_COMPOSITION_EPS}]"
        )
    provenance = _identity_gate(
        identity,
        config,
        (stability.identity, predictive_calibration.identity, robustness.identity),
    )
    gates = (
        provenance,
        _stability_gate(config, stability),
        _retention_gate(config),
        _predictive_gate(config, predictive_calibration),
        _robustness_gate(config, robustness),
        _composition_gate(config, eps_max=normalized_eps_max),
    )
    return GuaranteeBundle(
        identity=identity,
        generated_at=generated_at
        or datetime.now(UTC).isoformat(timespec="seconds"),
        gates=gates,
    )


def make_evidence_manifest(
    *,
    identity: ModelIdentity,
    config: SynapticConfig,
    stability: StabilityObservation,
    predictive_calibration: PredictiveCalibrationObservation,
    robustness: RobustnessObservation,
) -> dict[str, Any]:
    """Serialize an audit manifest without non-transferable runtime attestations."""
    return _json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "identity": asdict(identity),
            "synaptic_config": asdict(config),
            "stability": stability.to_manifest_dict(),
            "predictive_calibration": predictive_calibration.to_manifest_dict(),
            "robustness": robustness.to_manifest_dict(),
        }
    )


def bundle_from_manifest(
    payload: Mapping[str, Any], *, eps_max: float = 0.5
) -> GuaranteeBundle:
    """Validate a raw evidence manifest and produce its fail-closed model card."""
    _exact_keys(
        payload,
        {
            "schema_version",
            "identity",
            "synaptic_config",
            "stability",
            "predictive_calibration",
            "robustness",
        },
        "evidence manifest",
    )
    if type(payload.get("schema_version")) is not int or payload.get(
        "schema_version"
    ) != SCHEMA_VERSION:
        raise ValueError(f"schema_version must equal {SCHEMA_VERSION}")
    config = _synaptic_config_from_dict(
        _mapping(payload.get("synaptic_config"), "synaptic_config")
    )
    identity = ModelIdentity.from_dict(
        _mapping(payload.get("identity"), "identity")
    )
    return build_guarantee_bundle(
        identity=identity,
        config=config,
        stability=StabilityObservation.from_dict(
            _mapping(payload.get("stability"), "stability")
        ),
        predictive_calibration=PredictiveCalibrationObservation.from_dict(
            _mapping(
                payload.get("predictive_calibration"), "predictive_calibration"
            )
        ),
        robustness=RobustnessObservation.from_dict(
            _mapping(payload.get("robustness"), "robustness")
        ),
        eps_max=eps_max,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Generate a fail-closed JSON/Markdown audit card from an offline manifest."""
    parser = argparse.ArgumentParser(
        description="Generate a fail-closed certificate audit card"
    )
    parser.add_argument("evidence", help="raw evidence manifest JSON")
    parser.add_argument(
        "--output-dir", default="runs/certified_model_card", help="artifact directory"
    )
    parser.add_argument("--eps-max", type=float, default=0.5)
    parser.add_argument(
        "--allow-uncertified",
        action="store_true",
        help="return status 0 for a well-formed refusal card (never changes its verdict)",
    )
    args = parser.parse_args(argv)
    console = Console()
    try:
        source = Path(args.evidence)
        source_size = source.stat().st_size
        if source_size > _MAX_EVIDENCE_BYTES:
            raise ValueError(
                "evidence manifest exceeds the fixed "
                f"{_MAX_EVIDENCE_BYTES}-byte input limit"
            )
        payload = _strict_json_loads(source.read_text(encoding="utf-8"))
        bundle = bundle_from_manifest(payload, eps_max=args.eps_max)
        output_dir = Path(args.output_dir)
        with RunLogger(
            output_dir,
            name="certificate_model_card",
            provenance=asdict(bundle.identity),
        ) as logger:
            bundle.emit(logger)
            json_path, markdown_path = bundle.write_artifacts(output_dir)
        bundle.render(console)
        console.print(f"JSON: {json_path}")
        console.print(f"Markdown: {markdown_path}")
        if not args.allow_uncertified and not bundle.deployment_certified:
            console.print("[red]Deployment certification refused.[/red]")
            return 2
        return 0
    except (OSError, RecursionError, TypeError, ValueError, json.JSONDecodeError) as exc:
        console.print(f"[red]Certificate generation failed:[/red] {exc}")
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
