"""Read-only tropical selection certificates and interpretability fingerprints.

The affine score skeleton is ``z_j(x) = a_j @ x + b_j``.  This module records
the selected polyhedral cell, exact dual-norm distances to its relevant score
facets, a global max-plus Lipschitz bound, and the measured soft-to-hard
temperature regime.  It deliberately does not alter attention or MoE routing;
the default-off runtime toggle and fallback belong to bead ``0642.6.2.2``.

Claim scopes stay separate throughout:

* stable selection does not imply that a soft readout is hard;
* stable expert membership does not imply stable expert outputs; and
* a pointwise fingerprint of nonlinear scores is not an affine-region proof.

Unbounded quantities use ``None`` plus an explicit ``*_unbounded`` flag so all
records remain strict JSON (no ``NaN`` or ``Infinity``).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from fractions import Fraction
from typing import Any, Protocol

import numpy as np


class InputNorm(StrEnum):
    """Threat-model norm on the affine input perturbation."""

    L1 = "l1"
    L2 = "l2"
    LINF = "linf"


class GeometryScope(StrEnum):
    """Whether the observed score family licenses polyhedral geometry."""

    EXACT_AFFINE = "exact_affine"
    LOCAL_ONLY = "local_only"
    INVALID = "invalid"


class CertificateScope(StrEnum):
    """The protected decision; each scope has different temperature semantics."""

    ATTENTION_HARD_READOUT = "attention_hard_readout"
    MOE_TOPK_MEMBERSHIP = "moe_topk_membership"
    MOE_HARD_TOP1 = "moe_hard_top1"


class EventLogger(Protocol):
    """The subset of :class:`run_logging.RunLogger` used by the monitor."""

    def event(
        self,
        event: str,
        *,
        level: str = "info",
        step: int | None = None,
        **fields: Any,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class PairwiseFacet:
    """One selected-versus-unselected half-space and its boundary distance."""

    selected_id: str
    competitor_id: str
    normal: tuple[float, ...]
    rhs: float
    slack: float
    dual_slope_norm: float
    boundary_radius: float | None
    boundary_unbounded: bool
    equal_slope: bool
    duplicate_term: bool


@dataclass(frozen=True)
class ActiveFaceFingerprint:
    """Replayable active IDs and exposed lifted-face metadata for one decision."""

    valid: bool
    reason: str
    eligible_ids: tuple[str, ...]
    active_ids: tuple[str, ...]
    ambiguity_ids: tuple[str, ...]
    topk_order: tuple[str, ...]
    masked_scores: tuple[tuple[str, float], ...]
    selection_gap: float | None
    gap_unbounded: bool
    face_dimension: int | None
    active_vertex_certified: bool
    tie_tol: float
    tie_rule: str
    mask_digest: str
    score_digest: str
    slope_digest: str
    input_digest: str
    state_digest: str
    replayable: bool
    digest: str


@dataclass(frozen=True)
class SelectionGeometry:
    """Polyhedral selection cell and strict-ball robustness radius."""

    scope: GeometryScope
    input_norm: InputNorm
    input_dimension: int
    top_k: int
    facets: tuple[PairwiseFacet, ...]
    raw_radius: float | None
    radius_unbounded: bool
    certified_radius: float | None
    certified_radius_unbounded: bool
    safety_fraction: float
    tie_tol: float
    min_certified_radius: float
    support_required: bool
    support_radius: float | None
    support_radius_unbounded: bool
    support_top_k: int | None
    support_eligible_ids: tuple[str, ...]
    support_selected_ids: tuple[str, ...]
    support_fingerprint_digest: str | None
    exact_affine: bool
    certified: bool
    reason: str


@dataclass(frozen=True)
class SelectionCertificate:
    """The pointwise fingerprint and the geometry it is (or is not) allowed to claim."""

    fingerprint: ActiveFaceFingerprint
    geometry: SelectionGeometry


@dataclass(frozen=True)
class LipschitzCertificate:
    """Global max-plus Lipschitz constant, exact or conservatively over all slopes."""

    input_norm: InputNorm
    value: float | None
    exact: bool
    conservative: bool
    eligible_ids: tuple[str, ...]
    retained_ids: tuple[str, ...]
    slope_digest: str
    ledger_complete: bool
    valid: bool
    reason: str


@dataclass(frozen=True)
class TemperatureGate:
    """Measured soft-to-hard regime evidence for one masked score distribution."""

    certificate_scope: CertificateScope
    applicable: bool
    valid: bool
    singleton: bool
    m: int
    tau: float | None
    gap: float | None
    gap_unbounded: bool
    kappa: float | None
    kappa_unbounded: bool
    kappa_min: float | None
    winner_mass: float | None
    winner_mass_lower_bound: float | None
    measured_entropy: float | None
    normalized_entropy: float | None
    normalized_entropy_upper_bound: float | None
    min_winner_mass: float
    max_normalized_entropy: float
    tie_tol: float
    choice_ids: tuple[str, ...]
    score_digest: str
    passed: bool | None
    reason: str


@dataclass(frozen=True)
class TropicalCertificateRecord:
    """One strict-JSON decision record with deliberately separate verdicts."""

    step: int
    layer: str | None
    head: int | None
    certificate_scope: CertificateScope
    router_top_k: int | None
    pre_dropout: bool | None
    values_frozen: bool
    fingerprint: ActiveFaceFingerprint
    geometry: SelectionGeometry
    lipschitz: LipschitzCertificate
    temperature: TemperatureGate | None
    selection_certified: bool
    lipschitz_certified: bool
    readout_certified: bool | None
    output_stability_certified: bool
    output_stability_reason: str
    artifacts_bound: bool
    certified: bool
    reason: str


@dataclass(frozen=True)
class _PreparedScores:
    x: np.ndarray
    slopes: np.ndarray
    offsets: np.ndarray
    scores: np.ndarray
    eligible: np.ndarray
    choice_ids: tuple[str, ...]


def _enum_value[T: StrEnum](value: T | str, enum_type: type[T], name: str) -> T:
    try:
        return enum_type(value)
    except ValueError as exc:
        choices = ", ".join(member.value for member in enum_type)
        raise ValueError(f"{name} must be one of: {choices}") from exc


def _json_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prepare_scores(
    x: np.ndarray,
    slopes: np.ndarray,
    offsets: np.ndarray,
    *,
    eligible: np.ndarray | None,
    choice_ids: tuple[str, ...] | None,
) -> _PreparedScores:
    x_array = np.asarray(x, dtype=np.float64)
    slope_array = np.asarray(slopes, dtype=np.float64)
    offset_array = np.asarray(offsets, dtype=np.float64)
    if x_array.ndim != 1:
        raise ValueError(f"x must be one-dimensional, got shape {x_array.shape}")
    if x_array.size < 1:
        raise ValueError("x and slope input dimension must be non-empty")
    if slope_array.ndim != 2:
        raise ValueError(f"slopes must be two-dimensional, got shape {slope_array.shape}")
    if slope_array.shape[1] < 1:
        raise ValueError("slope input dimension must be non-empty")
    if slope_array.shape[1] != x_array.size:
        raise ValueError(
            "slopes input dimension must match x: "
            f"{slope_array.shape[1]} != {x_array.size}"
        )
    if offset_array.shape != (slope_array.shape[0],):
        raise ValueError(
            "offsets must have one value per slope, got "
            f"{offset_array.shape} for {slope_array.shape[0]} slopes"
        )

    n_choices = int(slope_array.shape[0])
    if eligible is None:
        eligible_array = np.ones(n_choices, dtype=bool)
    else:
        eligible_array = np.asarray(eligible)
        if eligible_array.dtype != np.bool_ or eligible_array.shape != (n_choices,):
            raise ValueError(
                "eligible must be a one-dimensional boolean mask with one entry per choice"
            )

    if choice_ids is None:
        ids = tuple(str(index) for index in range(n_choices))
    else:
        ids = tuple(str(choice_id) for choice_id in choice_ids)
        if len(ids) != n_choices:
            raise ValueError("choice_ids must have one entry per score")
        if len(set(ids)) != len(ids):
            raise ValueError("choice_ids must be unique")
    with np.errstate(over="ignore", invalid="ignore"):
        scores = slope_array @ x_array + offset_array
    return _PreparedScores(
        x=x_array,
        slopes=slope_array,
        offsets=offset_array,
        scores=scores,
        eligible=eligible_array,
        choice_ids=ids,
    )


def dual_norm(vector: np.ndarray, input_norm: InputNorm | str) -> float:
    """Return the dual norm corresponding to the declared input threat norm."""
    norm = _enum_value(input_norm, InputNorm, "input_norm")
    values = np.asarray(vector, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"vector must be one-dimensional, got shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("vector must contain only finite values")
    extended = values.astype(np.longdouble)
    if norm is InputNorm.L1:
        result = np.max(np.abs(extended), initial=np.longdouble(0.0))
    elif norm is InputNorm.L2:
        result = np.sqrt(np.sum(extended * extended, dtype=np.longdouble))
    else:
        result = np.sum(np.abs(extended), dtype=np.longdouble)
    if not np.isfinite(result) or result > np.finfo(np.float64).max:
        raise ValueError("dual norm exceeded the finite certificate range")
    return float(result)


def _score_digest(masked_scores: tuple[tuple[str, float], ...]) -> str:
    return _json_digest({"masked_scores": masked_scores})


def _slope_digest(
    choice_ids: tuple[str, ...],
    slopes: np.ndarray,
) -> str:
    return _json_digest(
        {"choice_ids": choice_ids, "slopes": np.asarray(slopes).tolist()}
    )


def _finite_float(value: np.longdouble) -> float | None:
    if not np.isfinite(value) or abs(value) > np.finfo(np.float64).max:
        return None
    return float(value)


def _finite_float_array(values: np.ndarray) -> np.ndarray | None:
    extended = np.asarray(values, dtype=np.longdouble)
    if (
        not np.isfinite(extended).all()
        or np.any(np.abs(extended) > np.finfo(np.float64).max)
    ):
        return None
    return extended.astype(np.float64)


def _exact_affine_dimension(points: np.ndarray) -> int | None:
    """Return exact affine dimension over IEEE-float rationals within a fixed work cap.

    Default SVD tolerances can collapse a mathematically nonzero but thin face.
    Every finite Python float is an exact rational, so fraction-free semantics
    avoid that false claim.  Large tied faces are withheld rather than making
    unbounded rational arithmetic part of a read-only runtime monitor.
    """
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        return None
    n_points, dimension = values.shape
    if n_points <= 1:
        return 0
    if (n_points - 1) * dimension > 4096:
        return None
    base = [Fraction.from_float(float(value)) for value in values[0]]
    rows = [
        [Fraction.from_float(float(value)) - base[column] for column, value in enumerate(row)]
        for row in values[1:]
    ]
    rank = 0
    for column in range(dimension):
        pivot = next(
            (row_index for row_index in range(rank, len(rows)) if rows[row_index][column]),
            None,
        )
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        pivot_value = rows[rank][column]
        for row_index in range(rank + 1, len(rows)):
            value = rows[row_index][column]
            if not value:
                continue
            factor = value / pivot_value
            for trailing in range(column, dimension):
                rows[row_index][trailing] -= factor * rows[rank][trailing]
        rank += 1
        if rank == len(rows):
            break
    return rank


def _invalid_fingerprint(
    prepared: _PreparedScores,
    *,
    reason: str,
    tie_tol: float,
    replayable: bool,
    state_digest: str | None,
) -> ActiveFaceFingerprint:
    eligible_ids = tuple(
        prepared.choice_ids[index]
        for index in np.flatnonzero(prepared.eligible)
    )
    mask_digest = _json_digest({"eligible_ids": eligible_ids})
    score_digest = _score_digest(())
    slope_digest = _json_digest(
        {"eligible_ids": eligible_ids, "invalid": reason, "shape": list(prepared.slopes.shape)}
    )
    input_digest = _json_digest(
        {"shape": list(prepared.x.shape), "finite": bool(np.isfinite(prepared.x).all())}
    )
    safe_state_digest = state_digest or _json_digest(
        {
            "shape": [list(prepared.slopes.shape), list(prepared.x.shape)],
            "eligible_ids": eligible_ids,
            "invalid": reason,
        }
    )
    payload = {
        "eligible_ids": eligible_ids,
        "reason": reason,
        "score_digest": score_digest,
        "slope_digest": slope_digest,
        "input_digest": input_digest,
        "tie_tol": tie_tol,
        "state_digest": safe_state_digest,
        "replayable": replayable,
    }
    return ActiveFaceFingerprint(
        valid=False,
        reason=reason,
        eligible_ids=eligible_ids,
        active_ids=(),
        ambiguity_ids=(),
        topk_order=(),
        masked_scores=(),
        selection_gap=None,
        gap_unbounded=False,
        face_dimension=None,
        active_vertex_certified=False,
        tie_tol=tie_tol,
        tie_rule="score_desc_choice_id_asc",
        mask_digest=mask_digest,
        score_digest=score_digest,
        slope_digest=slope_digest,
        input_digest=input_digest,
        state_digest=safe_state_digest,
        replayable=replayable,
        digest=_json_digest(payload),
    )


def _fingerprint_from_prepared(
    prepared: _PreparedScores,
    *,
    top_k: int,
    tie_tol: float,
    replayable: bool,
    state_digest: str | None,
) -> ActiveFaceFingerprint:
    eligible_indices = [int(index) for index in np.flatnonzero(prepared.eligible)]
    if not eligible_indices:
        return _invalid_fingerprint(
            prepared,
            reason="empty eligible set",
            tie_tol=tie_tol,
            replayable=replayable,
            state_digest=state_digest,
        )
    if top_k < 1 or top_k > len(eligible_indices):
        raise ValueError(
            f"top_k must be in [1, {len(eligible_indices)}], got {top_k}"
        )

    eligible_numeric = np.array(eligible_indices, dtype=np.int64)
    if (
        not np.isfinite(prepared.x).all()
        or not np.isfinite(prepared.slopes[eligible_numeric]).all()
        or not np.isfinite(prepared.offsets[eligible_numeric]).all()
        or not np.isfinite(prepared.scores[eligible_numeric]).all()
    ):
        return _invalid_fingerprint(
            prepared,
            reason="non-finite affine score evidence",
            tie_tol=tie_tol,
            replayable=replayable,
            state_digest=state_digest,
        )

    eligible_scores = prepared.scores[eligible_numeric].astype(np.longdouble)
    score_span = np.max(eligible_scores) - np.min(eligible_scores)
    if not np.isfinite(score_span) or score_span > np.finfo(np.float64).max:
        return _invalid_fingerprint(
            prepared,
            reason="score margin exceeded the finite certificate range",
            tie_tol=tie_tol,
            replayable=replayable,
            state_digest=state_digest,
        )

    ordered = sorted(
        eligible_indices,
        key=lambda index: (-float(prepared.scores[index]), prepared.choice_ids[index]),
    )
    selected = ordered[:top_k]
    maximum = float(prepared.scores[ordered[0]])
    active = [index for index in ordered if float(prepared.scores[index]) == maximum]
    ambiguous = [
        index
        for index in ordered
        if maximum - float(prepared.scores[index]) <= tie_tol
    ]
    if len(eligible_indices) == top_k:
        selection_gap = None
        gap_unbounded = True
    else:
        selection_gap = min(
            float(prepared.scores[selected_index] - prepared.scores[competitor_index])
            for selected_index in selected
            for competitor_index in ordered[top_k:]
        )
        gap_unbounded = False

    if len(ambiguous) != len(active):
        face_dimension = None
    elif len(active) == 1:
        face_dimension = 0
    else:
        lifted = np.concatenate(
            (prepared.slopes[active], prepared.offsets[active, None]),
            axis=1,
        )
        face_dimension = _exact_affine_dimension(lifted)

    eligible_ids = tuple(prepared.choice_ids[index] for index in eligible_indices)
    masked_scores = tuple(
        (prepared.choice_ids[index], float(prepared.scores[index]))
        for index in eligible_indices
    )
    mask_digest = _json_digest({"eligible_ids": eligible_ids})
    score_digest = _score_digest(masked_scores)
    slope_digest = _slope_digest(
        eligible_ids,
        prepared.slopes[eligible_numeric],
    )
    input_digest = _json_digest({"x": prepared.x.tolist()})
    if state_digest is None:
        state_digest = _json_digest(
            {
                "x": prepared.x.tolist(),
                "slopes": prepared.slopes[eligible_numeric].tolist(),
                "offsets": prepared.offsets[eligible_numeric].tolist(),
                "eligible_ids": eligible_ids,
            }
        )
    payload = {
        "eligible_ids": eligible_ids,
        "active_ids": [prepared.choice_ids[index] for index in active],
        "ambiguity_ids": [prepared.choice_ids[index] for index in ambiguous],
        "topk_order": [prepared.choice_ids[index] for index in selected],
        "masked_scores": masked_scores,
        "score_digest": score_digest,
        "slope_digest": slope_digest,
        "input_digest": input_digest,
        "tie_tol": tie_tol,
        "tie_rule": "score_desc_choice_id_asc",
        "mask_digest": mask_digest,
        "state_digest": state_digest,
        "replayable": replayable,
    }
    return ActiveFaceFingerprint(
        valid=True,
        reason="replayable" if replayable else "score-producing state was not replayable",
        eligible_ids=eligible_ids,
        active_ids=tuple(prepared.choice_ids[index] for index in active),
        ambiguity_ids=tuple(prepared.choice_ids[index] for index in ambiguous),
        topk_order=tuple(prepared.choice_ids[index] for index in selected),
        masked_scores=masked_scores,
        selection_gap=selection_gap,
        gap_unbounded=gap_unbounded,
        face_dimension=face_dimension,
        active_vertex_certified=bool(
            replayable and len(active) == 1 and len(ambiguous) == 1
        ),
        tie_tol=tie_tol,
        tie_rule="score_desc_choice_id_asc",
        mask_digest=mask_digest,
        score_digest=score_digest,
        slope_digest=slope_digest,
        input_digest=input_digest,
        state_digest=state_digest,
        replayable=replayable,
        digest=_json_digest(payload),
    )


def active_face_fingerprint(
    x: np.ndarray,
    slopes: np.ndarray,
    offsets: np.ndarray,
    *,
    eligible: np.ndarray | None = None,
    choice_ids: tuple[str, ...] | None = None,
    top_k: int = 1,
    tie_tol: float = 1e-9,
    replayable: bool = True,
    state_digest: str | None = None,
) -> ActiveFaceFingerprint:
    """Fingerprint the exact scores, deterministic selection, and lifted active face."""
    if not math.isfinite(tie_tol) or tie_tol < 0.0:
        raise ValueError("tie_tol must be finite and non-negative")
    prepared = _prepare_scores(
        x,
        slopes,
        offsets,
        eligible=eligible,
        choice_ids=choice_ids,
    )
    return _fingerprint_from_prepared(
        prepared,
        top_k=top_k,
        tie_tol=tie_tol,
        replayable=replayable,
        state_digest=state_digest,
    )


def _composed_radius(
    radius: float | None,
    radius_unbounded: bool,
    support: SelectionGeometry | None,
) -> tuple[float | None, bool]:
    if support is None:
        return radius, radius_unbounded
    if radius_unbounded and support.radius_unbounded:
        return None, True
    finite = []
    if not radius_unbounded and radius is not None:
        finite.append(radius)
    if not support.radius_unbounded and support.raw_radius is not None:
        finite.append(support.raw_radius)
    return (min(finite), False) if finite else (None, True)


def certify_selection_geometry(
    x: np.ndarray,
    slopes: np.ndarray,
    offsets: np.ndarray,
    *,
    eligible: np.ndarray | None = None,
    choice_ids: tuple[str, ...] | None = None,
    top_k: int = 1,
    input_norm: InputNorm | str = InputNorm.L2,
    scope: GeometryScope | str = GeometryScope.EXACT_AFFINE,
    tie_tol: float = 1e-9,
    safety_fraction: float = 0.05,
    min_certified_radius: float = 0.0,
    replayable: bool = True,
    state_digest: str | None = None,
    support_required: bool = False,
    support_certificate: SelectionCertificate | None = None,
) -> SelectionCertificate:
    """Certify the strict-ball radius preserving a top-1 or unordered top-k set.

    ``support_certificate`` composes the outer support cell required by
    standard synaptic attention or MoE hard-top-1 routing.  Its raw radius is
    intersected with the inner selection radius before the safety fraction is
    applied.  Both artifacts must share a replay-state digest and compatible
    eligible/selected IDs.
    """
    norm = _enum_value(input_norm, InputNorm, "input_norm")
    geometry_scope = _enum_value(scope, GeometryScope, "scope")
    if not math.isfinite(tie_tol) or tie_tol < 0.0:
        raise ValueError("tie_tol must be finite and non-negative")
    if not math.isfinite(safety_fraction) or not 0.0 < safety_fraction < 1.0:
        raise ValueError("safety_fraction must be finite and strictly between 0 and 1")
    if not math.isfinite(min_certified_radius) or min_certified_radius < 0.0:
        raise ValueError("min_certified_radius must be finite and non-negative")

    prepared = _prepare_scores(
        x,
        slopes,
        offsets,
        eligible=eligible,
        choice_ids=choice_ids,
    )
    fingerprint = _fingerprint_from_prepared(
        prepared,
        top_k=top_k,
        tie_tol=tie_tol,
        replayable=replayable,
        state_digest=state_digest,
    )
    facets: list[PairwiseFacet] = []
    finite_radii: list[float] = []
    invalid_boundary = False
    derived_nonfinite = False

    if fingerprint.valid:
        index_by_id = {choice_id: index for index, choice_id in enumerate(prepared.choice_ids)}
        selected = [index_by_id[choice_id] for choice_id in fingerprint.topk_order]
        unselected = [
            int(index)
            for index in np.flatnonzero(prepared.eligible)
            if int(index) not in selected
        ]
        for selected_index in selected:
            for competitor_index in unselected:
                normal_array = _finite_float_array(
                    prepared.slopes[selected_index].astype(np.longdouble)
                    - prepared.slopes[competitor_index].astype(np.longdouble)
                )
                rhs = _finite_float(
                    np.longdouble(prepared.offsets[competitor_index])
                    - np.longdouble(prepared.offsets[selected_index])
                )
                slack = _finite_float(
                    np.longdouble(prepared.scores[selected_index])
                    - np.longdouble(prepared.scores[competitor_index])
                )
                if normal_array is None or rhs is None or slack is None:
                    derived_nonfinite = True
                    invalid_boundary = True
                    break
                equal_slope = bool(
                    np.array_equal(
                        prepared.slopes[selected_index],
                        prepared.slopes[competitor_index],
                    )
                )
                duplicate = bool(
                    equal_slope
                    and prepared.offsets[selected_index] == prepared.offsets[competitor_index]
                )
                if equal_slope:
                    slope_norm = 0.0
                    boundary_radius = None
                    boundary_unbounded = not duplicate and slack > tie_tol
                    invalid_boundary |= not boundary_unbounded
                else:
                    try:
                        slope_norm = dual_norm(normal_array, norm)
                    except ValueError:
                        derived_nonfinite = True
                        invalid_boundary = True
                        break
                    boundary_radius = slack / slope_norm
                    if not math.isfinite(boundary_radius):
                        derived_nonfinite = True
                        invalid_boundary = True
                        break
                    boundary_unbounded = False
                    finite_radii.append(boundary_radius)
                    invalid_boundary |= slack <= tie_tol or boundary_radius <= 0.0
                facets.append(
                    PairwiseFacet(
                        selected_id=prepared.choice_ids[selected_index],
                        competitor_id=prepared.choice_ids[competitor_index],
                        normal=tuple(float(value) for value in normal_array),
                        rhs=rhs,
                        slack=slack,
                        dual_slope_norm=slope_norm,
                        boundary_radius=boundary_radius,
                        boundary_unbounded=boundary_unbounded,
                        equal_slope=equal_slope,
                        duplicate_term=duplicate,
                    )
                )
            if derived_nonfinite:
                break

    if derived_nonfinite:
        facets.clear()
        finite_radii.clear()

    raw_radius = min(finite_radii) if finite_radii else None
    radius_unbounded = bool(
        fingerprint.valid
        and not invalid_boundary
        and not derived_nonfinite
        and not finite_radii
    )
    support_geometry = (
        None if support_certificate is None else support_certificate.geometry
    )
    if fingerprint.valid and not derived_nonfinite and (
        support_geometry is None or support_geometry.certified
    ):
        raw_radius, radius_unbounded = _composed_radius(
            raw_radius,
            radius_unbounded,
            support_geometry,
        )
    if support_certificate is None:
        support_radius = None
        support_radius_unbounded = False
        support_top_k = None
        support_eligible_ids: tuple[str, ...] = ()
        support_selected_ids: tuple[str, ...] = ()
        support_fingerprint_digest = None
    else:
        support_geometry = support_certificate.geometry
        support_radius = support_geometry.raw_radius
        support_radius_unbounded = support_geometry.radius_unbounded
        support_top_k = support_geometry.top_k
        support_eligible_ids = support_certificate.fingerprint.eligible_ids
        support_selected_ids = support_certificate.fingerprint.topk_order
        support_fingerprint_digest = support_certificate.fingerprint.digest

    certified_radius = (
        None if radius_unbounded or raw_radius is None
        else (1.0 - safety_fraction) * raw_radius
    )
    support_id_binding = bool(
        support_certificate is None
        or fingerprint.eligible_ids == support_certificate.fingerprint.eligible_ids
        or fingerprint.eligible_ids == support_certificate.fingerprint.topk_order
    )
    support_ok = support_certificate is None or bool(
        support_geometry is not None
        and support_geometry.certified
        and support_geometry.input_norm is norm
        and support_geometry.input_dimension == prepared.x.size
        and support_certificate.fingerprint.replayable
        and hmac.compare_digest(
            fingerprint.input_digest,
            support_certificate.fingerprint.input_digest,
        )
        and hmac.compare_digest(
            fingerprint.state_digest,
            support_certificate.fingerprint.state_digest,
        )
        and support_id_binding
    )
    if support_required:
        support_ok = support_geometry is not None and support_ok
    radius_nonvacuous = radius_unbounded or bool(
        certified_radius is not None
        and certified_radius > min_certified_radius
    )
    certified = bool(
        fingerprint.valid
        and fingerprint.replayable
        and geometry_scope is GeometryScope.EXACT_AFFINE
        and not invalid_boundary
        and not derived_nonfinite
        and support_ok
        and radius_nonvacuous
    )
    if not fingerprint.valid:
        reason = fingerprint.reason
        geometry_scope = GeometryScope.INVALID
    elif geometry_scope is GeometryScope.INVALID:
        reason = "score geometry was declared invalid"
    elif geometry_scope is GeometryScope.LOCAL_ONLY:
        reason = "pointwise fingerprint only; scores were not certified affine"
    elif not fingerprint.replayable:
        reason = "score-producing state was not replayable"
    elif derived_nonfinite:
        reason = "derived facet evidence exceeded the finite certificate range"
    elif support_required and support_certificate is None:
        reason = "required base-support geometry was missing"
    elif support_certificate is not None and not support_ok:
        reason = "support geometry was not certified or bound to the same state, norm, and IDs"
    elif invalid_boundary:
        reason = "selection boundary was tied, near-tied, or duplicated"
    elif not radius_nonvacuous:
        reason = "safety-adjusted radius did not exceed the non-vacuity floor"
    else:
        reason = "certified"

    return SelectionCertificate(
        fingerprint=fingerprint,
        geometry=SelectionGeometry(
            scope=geometry_scope,
            input_norm=norm,
            input_dimension=int(prepared.x.size),
            top_k=top_k,
            facets=tuple(facets),
            raw_radius=raw_radius,
            radius_unbounded=radius_unbounded,
            certified_radius=certified_radius,
            certified_radius_unbounded=radius_unbounded,
            safety_fraction=safety_fraction,
            tie_tol=tie_tol,
            min_certified_radius=min_certified_radius,
            support_required=support_required,
            support_radius=support_radius,
            support_radius_unbounded=support_radius_unbounded,
            support_top_k=support_top_k,
            support_eligible_ids=support_eligible_ids,
            support_selected_ids=support_selected_ids,
            support_fingerprint_digest=support_fingerprint_digest,
            exact_affine=geometry_scope is GeometryScope.EXACT_AFFINE,
            certified=certified,
            reason=reason,
        ),
    )


def global_lipschitz_certificate(
    slopes: np.ndarray,
    *,
    eligible: np.ndarray | None = None,
    choice_ids: tuple[str, ...] | None = None,
    input_norm: InputNorm | str = InputNorm.L2,
    nonempty_region_ids: tuple[str, ...] | None = None,
    ledger_complete: bool = False,
) -> LipschitzCertificate:
    """Bound ``max_j(a_j @ x + b_j)`` under the declared input norm.

    Without an externally verified complete region-feasibility ledger the
    maximum is conservatively taken over every eligible slope.  Merely passing
    a subset is not enough to label the result exact.
    """
    norm = _enum_value(input_norm, InputNorm, "input_norm")
    slope_array = np.asarray(slopes, dtype=np.float64)
    if slope_array.ndim != 2:
        raise ValueError(f"slopes must be two-dimensional, got shape {slope_array.shape}")
    if slope_array.shape[1] < 1:
        raise ValueError("slope input dimension must be non-empty")
    n_choices = int(slope_array.shape[0])
    if eligible is None:
        eligible_array = np.ones(n_choices, dtype=bool)
    else:
        eligible_array = np.asarray(eligible)
        if eligible_array.dtype != np.bool_ or eligible_array.shape != (n_choices,):
            raise ValueError(
                "eligible must be a one-dimensional boolean mask with one entry per choice"
            )
    ids = (
        tuple(str(index) for index in range(n_choices))
        if choice_ids is None
        else tuple(str(choice_id) for choice_id in choice_ids)
    )
    if len(ids) != n_choices or len(set(ids)) != len(ids):
        raise ValueError("choice_ids must be unique and have one entry per slope")

    eligible_ids = tuple(ids[index] for index in np.flatnonzero(eligible_array))
    eligible_indices = np.array(
        [index for index in np.flatnonzero(eligible_array)],
        dtype=np.int64,
    )
    if eligible_indices.size and np.isfinite(slope_array[eligible_indices]).all():
        slope_digest = _slope_digest(eligible_ids, slope_array[eligible_indices])
    else:
        slope_digest = _json_digest(
            {"eligible_ids": eligible_ids, "invalid": "non-finite slope evidence"}
        )
    if not eligible_ids:
        return LipschitzCertificate(
            input_norm=norm,
            value=None,
            exact=False,
            conservative=False,
            eligible_ids=(),
            retained_ids=(),
            slope_digest=slope_digest,
            ledger_complete=False,
            valid=False,
            reason="empty eligible set",
        )
    id_to_index = {choice_id: index for index, choice_id in enumerate(ids)}
    if ledger_complete:
        if not nonempty_region_ids:
            raise ValueError(
                "ledger_complete=True requires nonempty_region_ids from a complete feasibility ledger"
            )
        retained = tuple(str(choice_id) for choice_id in nonempty_region_ids)
        if len(set(retained)) != len(retained) or any(
            choice_id not in eligible_ids for choice_id in retained
        ):
            raise ValueError("nonempty_region_ids must be unique eligible choice IDs")
        exact = True
        conservative = False
        reason = "exact over the attested complete nonempty-region ledger"
    else:
        retained = eligible_ids
        exact = False
        conservative = True
        reason = "conservative maximum over all eligible slopes"

    retained_indices = np.array([id_to_index[choice_id] for choice_id in retained])
    if not np.isfinite(slope_array[retained_indices]).all():
        return LipschitzCertificate(
            input_norm=norm,
            value=None,
            exact=False,
            conservative=conservative,
            eligible_ids=eligible_ids,
            retained_ids=retained,
            slope_digest=slope_digest,
            ledger_complete=ledger_complete,
            valid=False,
            reason="non-finite retained slope",
        )
    try:
        value = max(dual_norm(slope_array[index], norm) for index in retained_indices)
    except ValueError:
        return LipschitzCertificate(
            input_norm=norm,
            value=None,
            exact=False,
            conservative=conservative,
            eligible_ids=eligible_ids,
            retained_ids=retained,
            slope_digest=slope_digest,
            ledger_complete=ledger_complete,
            valid=False,
            reason="retained slope norm exceeded the finite certificate range",
        )
    return LipschitzCertificate(
        input_norm=norm,
        value=value,
        exact=exact,
        conservative=conservative,
        eligible_ids=eligible_ids,
        retained_ids=retained,
        slope_digest=slope_digest,
        ledger_complete=ledger_complete,
        valid=True,
        reason=reason,
    )


def _entropy_upper_bound(losing_mass: float, m: int) -> float:
    if losing_mass <= 0.0:
        return 0.0
    if losing_mass >= 1.0:
        return math.log(m)
    binary = -losing_mass * math.log(losing_mass) - (1.0 - losing_mass) * math.log(
        1.0 - losing_mass
    )
    return binary + losing_mass * math.log(m - 1)


def temperature_gate(
    scores: np.ndarray,
    *,
    certificate_scope: CertificateScope | str,
    tau: float,
    min_winner_mass: float = 0.95,
    max_normalized_entropy: float = 0.20,
    tie_tol: float = 1e-9,
    choice_ids: tuple[str, ...] | None = None,
) -> TemperatureGate:
    """Measure the soft-to-hard regime and apply scope-correct validity gates."""
    scope = _enum_value(certificate_scope, CertificateScope, "certificate_scope")
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.ndim != 1:
        raise ValueError(f"scores must be one-dimensional, got shape {score_array.shape}")
    if not math.isfinite(min_winner_mass) or not 0.0 < min_winner_mass < 1.0:
        raise ValueError("min_winner_mass must be finite and strictly between 0 and 1")
    if not math.isfinite(max_normalized_entropy) or not 0.0 <= max_normalized_entropy <= 1.0:
        raise ValueError("max_normalized_entropy must be finite and in [0, 1]")
    if not math.isfinite(tie_tol) or tie_tol < 0.0:
        raise ValueError("tie_tol must be finite and non-negative")
    applicable = scope is not CertificateScope.MOE_TOPK_MEMBERSHIP
    m = int(score_array.size)
    ids = tuple(str(index) for index in range(m)) if choice_ids is None else tuple(
        str(choice_id) for choice_id in choice_ids
    )
    if len(ids) != m or len(set(ids)) != len(ids):
        raise ValueError("choice_ids must be unique and have one entry per score")
    invalid_reason = None
    if m == 0:
        invalid_reason = "empty masked score distribution"
    elif not np.isfinite(score_array).all():
        invalid_reason = "non-finite masked score distribution"
    elif not math.isfinite(tau) or tau <= 0.0:
        invalid_reason = "temperature must be finite and positive"
    elif m >= 2 and min_winner_mass <= 1.0 / m:
        raise ValueError(f"min_winner_mass must be greater than the uniform mass 1/{m}")
    if invalid_reason is None:
        score_digest = _score_digest(
            tuple((choice_id, float(score)) for choice_id, score in zip(ids, score_array))
        )
    else:
        score_digest = _json_digest(
            {"choice_ids": ids, "invalid": invalid_reason, "shape": list(score_array.shape)}
        )
    if invalid_reason is not None:
        return TemperatureGate(
            certificate_scope=scope,
            applicable=applicable,
            valid=False,
            singleton=False,
            m=m,
            tau=float(tau) if math.isfinite(tau) else None,
            gap=None,
            gap_unbounded=False,
            kappa=None,
            kappa_unbounded=False,
            kappa_min=None,
            winner_mass=None,
            winner_mass_lower_bound=None,
            measured_entropy=None,
            normalized_entropy=None,
            normalized_entropy_upper_bound=None,
            min_winner_mass=min_winner_mass,
            max_normalized_entropy=max_normalized_entropy,
            tie_tol=tie_tol,
            choice_ids=ids,
            score_digest=score_digest,
            passed=False if applicable else None,
            reason=invalid_reason,
        )
    if m == 1:
        return TemperatureGate(
            certificate_scope=scope,
            applicable=applicable,
            valid=True,
            singleton=True,
            m=1,
            tau=float(tau),
            gap=None,
            gap_unbounded=True,
            kappa=None,
            kappa_unbounded=False,
            kappa_min=None,
            winner_mass=1.0,
            winner_mass_lower_bound=1.0,
            measured_entropy=0.0,
            normalized_entropy=0.0,
            normalized_entropy_upper_bound=0.0,
            min_winner_mass=min_winner_mass,
            max_normalized_entropy=max_normalized_entropy,
            tie_tol=tie_tol,
            choice_ids=ids,
            score_digest=score_digest,
            passed=True if applicable else None,
            reason="singleton bypass" if applicable else "singleton membership diagnostic",
        )

    ordered_scores = np.sort(score_array.astype(np.longdouble), kind="stable")[::-1]
    gap_extended = ordered_scores[0] - ordered_scores[1]
    gap = _finite_float(gap_extended)
    kappa_extended = gap_extended / np.longdouble(tau)
    kappa = _finite_float(kappa_extended)
    derived_finite = gap is not None and kappa is not None
    kappa_min = math.log((m - 1) * min_winner_mass / (1.0 - min_winner_mass))
    shifted_extended = (
        score_array.astype(np.longdouble) - np.max(score_array.astype(np.longdouble))
    ) / np.longdouble(tau)
    shifted = np.clip(shifted_extended, -745.0, 0.0).astype(np.float64)
    weights = np.exp(shifted)
    probabilities = weights / float(np.sum(weights))
    winner_mass = float(np.max(probabilities))
    positive = probabilities[probabilities > 0.0]
    entropy = float(-np.sum(positive * np.log(positive)))
    normalized_entropy = entropy / math.log(m)
    u = 0.0 if kappa is None else (m - 1) * math.exp(-kappa)
    mass_lower_bound = 1.0 / (1.0 + u)
    losing_mass_bound = u / (1.0 + u)
    entropy_bound = _entropy_upper_bound(losing_mass_bound, m) / math.log(m)
    analytic_pass = bool(
        mass_lower_bound >= min_winner_mass
        and entropy_bound <= max_normalized_entropy
    )
    measured_pass = bool(
        winner_mass >= min_winner_mass
        and normalized_entropy <= max_normalized_entropy
    )
    gap_pass = bool(derived_finite and gap is not None and gap > tie_tol)
    passed = bool(gap_pass and analytic_pass and measured_pass) if applicable else None
    if not derived_finite:
        reason = "derived gap or kappa exceeded the finite certificate range"
    elif not applicable:
        reason = "temperature is diagnostic only for MoE top-k membership"
    elif not gap_pass:
        reason = "winner gap did not strictly exceed tie tolerance"
    elif not analytic_pass:
        reason = "analytic mass/entropy bound did not pass"
    elif not measured_pass:
        reason = "measured mass/entropy did not pass"
    else:
        reason = "passed"
    return TemperatureGate(
        certificate_scope=scope,
        applicable=applicable,
        valid=derived_finite,
        singleton=False,
        m=m,
        tau=float(tau),
        gap=gap,
        gap_unbounded=False,
        kappa=kappa,
        kappa_unbounded=False,
        kappa_min=kappa_min,
        winner_mass=winner_mass,
        winner_mass_lower_bound=mass_lower_bound,
        measured_entropy=entropy,
        normalized_entropy=normalized_entropy,
        normalized_entropy_upper_bound=entropy_bound,
        min_winner_mass=min_winner_mass,
        max_normalized_entropy=max_normalized_entropy,
        tie_tol=tie_tol,
        choice_ids=ids,
        score_digest=score_digest,
        passed=passed,
        reason=reason,
    )


class TropicalCertificateMonitor:
    """Collect, log, serialize, and render tropical certificates without changing routing."""

    def __init__(self, logger: EventLogger | None = None) -> None:
        self.logger = logger
        self.records: list[TropicalCertificateRecord] = []

    def record(
        self,
        *,
        step: int,
        certificate_scope: CertificateScope | str,
        selection: SelectionCertificate,
        lipschitz: LipschitzCertificate,
        temperature: TemperatureGate | None = None,
        layer: str | None = None,
        head: int | None = None,
        router_top_k: int | None = None,
        pre_dropout: bool | None = None,
        values_frozen: bool = False,
    ) -> TropicalCertificateRecord:
        """Compose scope-correct verdicts and append one structured record."""
        scope = _enum_value(certificate_scope, CertificateScope, "certificate_scope")
        if temperature is not None and temperature.certificate_scope is not scope:
            raise ValueError("temperature gate scope does not match certificate scope")
        if router_top_k is not None and router_top_k < 1:
            raise ValueError("router_top_k must be positive when supplied")

        lipschitz_binding = bool(
            selection.geometry.input_norm is lipschitz.input_norm
            and selection.fingerprint.eligible_ids == lipschitz.eligible_ids
            and hmac.compare_digest(
                selection.fingerprint.slope_digest,
                lipschitz.slope_digest,
            )
        )
        temperature_binding = bool(
            temperature is None
            or (
                temperature.choice_ids == selection.fingerprint.eligible_ids
                and hmac.compare_digest(
                    temperature.score_digest,
                    selection.fingerprint.score_digest,
                )
            )
        )
        if (
            scope is CertificateScope.MOE_TOPK_MEMBERSHIP
            and temperature is not None
            and not temperature_binding
        ):
            raise ValueError(
                "MoE membership temperature diagnostic was not bound to the same IDs and scores"
            )
        if scope is CertificateScope.ATTENTION_HARD_READOUT:
            scope_binding = selection.geometry.top_k == 1
        elif scope is CertificateScope.MOE_TOPK_MEMBERSHIP:
            scope_binding = bool(
                router_top_k is not None
                and selection.geometry.top_k == router_top_k
            )
        else:
            scope_binding = bool(
                router_top_k is not None
                and selection.geometry.top_k == 1
                and selection.geometry.support_required
                and selection.geometry.support_top_k == router_top_k
                and selection.geometry.support_selected_ids
                == selection.fingerprint.eligible_ids
            )
        artifacts_bound = lipschitz_binding and temperature_binding and scope_binding
        selection_certified = selection.geometry.certified and scope_binding
        lipschitz_certified = lipschitz.valid and lipschitz_binding
        base_certified = selection_certified and lipschitz_certified

        if scope is CertificateScope.MOE_TOPK_MEMBERSHIP:
            readout_certified = None
            certified = base_certified
        else:
            readout_certified = bool(
                base_certified
                and artifacts_bound
                and pre_dropout
                and temperature is not None
                and temperature.applicable
                and temperature.valid
                and temperature.passed
            )
            certified = readout_certified

        if scope is CertificateScope.ATTENTION_HARD_READOUT:
            output_stability = bool(certified and values_frozen)
            output_reason = (
                "selected value was frozen under the declared query-conditional threat model"
                if output_stability
                else "attention output stability requires a certified readout and frozen selected value"
            )
        elif scope is CertificateScope.MOE_HARD_TOP1:
            output_stability = False
            output_reason = "stable expert selection does not bound the selected expert output"
        else:
            output_stability = False
            output_reason = "MoE membership alone does not bound expert outputs"

        if not scope_binding:
            reason = "certificate scope did not match the protected top-k geometry"
        elif not lipschitz_binding:
            reason = "Lipschitz evidence was not bound to the same norm, IDs, and slope family"
        elif not temperature_binding:
            reason = "temperature evidence was not bound to the same IDs and measured scores"
        elif not selection_certified:
            reason = selection.geometry.reason
        elif not lipschitz_certified:
            reason = lipschitz.reason
        elif readout_certified is not None and not readout_certified:
            if not pre_dropout:
                reason = "pre-dropout readout was not explicitly attested"
            elif temperature is None:
                reason = "measured temperature evidence was missing"
            else:
                reason = temperature.reason
        else:
            reason = "certified"
        record = TropicalCertificateRecord(
            step=int(step),
            layer=layer,
            head=None if head is None else int(head),
            certificate_scope=scope,
            router_top_k=router_top_k,
            pre_dropout=pre_dropout,
            values_frozen=values_frozen,
            fingerprint=selection.fingerprint,
            geometry=selection.geometry,
            lipschitz=lipschitz,
            temperature=temperature,
            selection_certified=selection_certified,
            lipschitz_certified=lipschitz_certified,
            readout_certified=readout_certified,
            output_stability_certified=output_stability,
            output_stability_reason=output_reason,
            artifacts_bound=artifacts_bound,
            certified=certified,
            reason=reason,
        )
        self.records.append(record)
        if self.logger is not None:
            self.logger.event(
                "tropical_certificate",
                level="info" if certified else "warning",
                step=record.step,
                certificate=asdict(record),
            )
        return record

    def all_certified(self) -> bool:
        """Return false for empty evidence rather than accepting vacuous truth."""
        return bool(self.records) and all(record.certified for record in self.records)

    def summary(self) -> dict[str, Any]:
        """Return a compact JSON-safe aggregate for run summaries."""
        return {
            "steps": len(self.records),
            "all_certified": self.all_certified(),
            "certified_steps": sum(record.certified for record in self.records),
            "selection_certified_steps": sum(
                record.selection_certified for record in self.records
            ),
            "readout_certified_steps": sum(
                bool(record.readout_certified) for record in self.records
            ),
            "output_stability_certified_steps": sum(
                record.output_stability_certified for record in self.records
            ),
            "unique_fingerprints": len(
                {record.fingerprint.digest for record in self.records}
            ),
        }

    def to_jsonl(self) -> list[str]:
        """Return strict, deterministically keyed JSONL records."""
        return [
            json.dumps(
                asdict(record),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
            for record in self.records
        ]

    def render(self, console: Any | None = None) -> None:
        """Render certificate verdicts with Rich."""
        from rich.console import Console
        from rich.table import Table

        output = console or Console()
        table = Table(
            title="Tropical selection certificates",
            caption="z_j = a_j^T x + b_j(h); radius is a strict-ball selection guarantee",
        )
        table.add_column("Step", justify="right")
        table.add_column("Layer/head")
        table.add_column("Claim / active IDs")
        table.add_column("Geometry / nearest flip")
        table.add_column("Lipschitz")
        table.add_column("Measured regime")
        table.add_column("Assumptions")
        table.add_column("Verdict / failed gate")
        for record in self.records:
            radius = (
                "unbounded"
                if record.geometry.certified_radius_unbounded
                else (
                    f"{record.geometry.certified_radius:.6g}"
                    if record.geometry.certified_radius is not None
                    else "n/a"
                )
            )
            location = record.layer or "-"
            if record.head is not None:
                location = f"{location}/{record.head}"
            gap = (
                "unbounded"
                if record.fingerprint.gap_unbounded
                else (
                    f"{record.fingerprint.selection_gap:.6g}"
                    if record.fingerprint.selection_gap is not None
                    else "n/a"
                )
            )
            reachable = [
                facet
                for facet in record.geometry.facets
                if facet.boundary_radius is not None
            ]
            if reachable:
                nearest = min(reachable, key=lambda facet: facet.boundary_radius or 0.0)
                flip = f"{nearest.selected_id}={nearest.competitor_id}"
            elif record.geometry.radius_unbounded:
                flip = "no reachable facet"
            else:
                flip = "unavailable"
            lipschitz_value = (
                f"{record.lipschitz.value:.6g}"
                if record.lipschitz.value is not None
                else "n/a"
            )
            lipschitz_kind = "exact" if record.lipschitz.exact else "conservative"
            if record.temperature is None:
                regime = "not measured"
            elif not record.temperature.valid:
                tau = (
                    f"{record.temperature.tau:g}"
                    if record.temperature.tau is not None
                    else "n/a"
                )
                regime = f"invalid; tau={tau}; {record.temperature.reason}"
            elif record.temperature.singleton:
                tau = (
                    f"{record.temperature.tau:g}"
                    if record.temperature.tau is not None
                    else "n/a"
                )
                regime = f"tau={tau}; singleton"
            else:
                tau = (
                    f"{record.temperature.tau:g}"
                    if record.temperature.tau is not None
                    else "n/a"
                )
                kappa = (
                    f"{record.temperature.kappa:.4g}"
                    if record.temperature.kappa is not None
                    else "n/a"
                )
                mass = (
                    f"{record.temperature.winner_mass:.4g}"
                    if record.temperature.winner_mass is not None
                    else "n/a"
                )
                entropy = (
                    f"{record.temperature.normalized_entropy:.4g}"
                    if record.temperature.normalized_entropy is not None
                    else "n/a"
                )
                regime = (
                    f"tau={tau}; kappa={kappa}; "
                    f"p*={mass}; Hn={entropy}"
                )
            assumptions = (
                f"scope={record.geometry.scope.value}; replay={record.fingerprint.replayable}; "
                f"pre_dropout={record.pre_dropout}; artifacts_bound={record.artifacts_bound}"
            )
            table.add_row(
                str(record.step),
                location,
                f"{record.certificate_scope.value}\n{', '.join(record.fingerprint.topk_order) or '-'}",
                (
                    f"gap={gap}; radius={radius}; safety={record.geometry.safety_fraction:g}\n"
                    f"nearest: {flip}"
                ),
                f"L={lipschitz_value} ({lipschitz_kind})",
                regime,
                assumptions,
                (
                    "[green]certified[/green]"
                    if record.certified
                    else f"[red]refused[/red]\n{record.reason}"
                ),
            )
        output.print(table)
