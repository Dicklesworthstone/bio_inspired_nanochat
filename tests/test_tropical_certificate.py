"""Focused tests for the read-only tropical certificate runtime."""

from __future__ import annotations

import json
import hmac
import math
from dataclasses import asdict
from typing import Any

import numpy as np
import pytest
from rich.console import Console

from bio_inspired_nanochat.tropical_certificate import (
    CertificateScope,
    GeometryScope,
    InputNorm,
    TropicalCertificateMonitor,
    certify_selection_geometry,
    global_lipschitz_certificate,
    temperature_gate,
)


def _basic_selection(
    *,
    input_norm: InputNorm = InputNorm.L2,
    scope: GeometryScope = GeometryScope.EXACT_AFFINE,
    **kwargs: Any,
):
    kwargs.setdefault("safety_fraction", 0.1)
    return certify_selection_geometry(
        np.array([0.0, 0.0]),
        np.array([[0.0, 0.0], [1.0, -2.0]]),
        np.array([2.0, 0.0]),
        input_norm=input_norm,
        scope=scope,
        choice_ids=("winner", "runner_up"),
        **kwargs,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("input_norm", "expected_radius", "boundary_delta"),
    [
        (InputNorm.L1, 1.0, np.array([0.0, -1.0])),
        (InputNorm.L2, 2.0 / math.sqrt(5.0), np.array([0.4, -0.8])),
        (InputNorm.LINF, 2.0 / 3.0, np.array([2.0 / 3.0, -2.0 / 3.0])),
    ],
)
def test_dual_norm_radius_reaches_constructed_facet(
    input_norm: InputNorm,
    expected_radius: float,
    boundary_delta: np.ndarray,
) -> None:
    selection = _basic_selection(input_norm=input_norm)

    assert selection.geometry.certified
    assert selection.geometry.raw_radius == pytest.approx(expected_radius)
    assert selection.geometry.certified_radius == pytest.approx(0.9 * expected_radius)
    facet = selection.geometry.facets[0]
    assert facet.slack + np.dot(np.array(facet.normal), boundary_delta) == pytest.approx(0.0)

    inside = 0.99 * boundary_delta
    boundary_scores = np.array([2.0, 0.0]) + np.array(
        [[0.0, 0.0], [1.0, -2.0]]
    ) @ inside
    assert boundary_scores[0] > boundary_scores[1]


@pytest.mark.unit
def test_singleton_and_all_selected_sets_have_explicit_unbounded_radius() -> None:
    singleton = certify_selection_geometry(
        np.array([1.0]),
        np.array([[2.0]]),
        np.array([0.0]),
    )
    all_selected = certify_selection_geometry(
        np.array([0.0]),
        np.array([[0.0], [1.0]]),
        np.array([2.0, 0.0]),
        top_k=2,
    )

    for selection in (singleton, all_selected):
        assert selection.geometry.certified
        assert selection.geometry.radius_unbounded
        assert selection.geometry.raw_radius is None
        assert selection.geometry.certified_radius is None
        assert selection.fingerprint.gap_unbounded


@pytest.mark.unit
def test_exact_tie_reports_face_and_refuses_unique_selection() -> None:
    selection = certify_selection_geometry(
        np.array([0.0]),
        np.array([[0.0], [1.0]]),
        np.array([0.0, 0.0]),
        choice_ids=("b", "a"),
    )

    assert selection.fingerprint.active_ids == ("a", "b")
    assert selection.fingerprint.topk_order == ("a",)
    assert selection.fingerprint.face_dimension == 1
    assert not selection.fingerprint.active_vertex_certified
    assert not selection.geometry.certified
    assert selection.geometry.raw_radius == 0.0


@pytest.mark.unit
def test_thin_exact_tropical_face_uses_exact_float_rational_rank() -> None:
    selection = certify_selection_geometry(
        np.array([0.0, 0.0]),
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1e-20]]),
        np.array([0.0, 0.0, 0.0]),
    )

    assert selection.fingerprint.active_ids == ("0", "1", "2")
    assert selection.fingerprint.face_dimension == 2
    assert not selection.geometry.certified


@pytest.mark.unit
def test_near_tie_withholds_face_dimension_and_fails_strict_gap() -> None:
    selection = certify_selection_geometry(
        np.array([0.0]),
        np.array([[0.0], [1.0]]),
        np.array([1.0, 1.0 - 1e-10]),
        tie_tol=1e-9,
    )

    assert selection.fingerprint.active_ids == ("0",)
    assert selection.fingerprint.ambiguity_ids == ("0", "1")
    assert selection.fingerprint.face_dimension is None
    assert not selection.geometry.certified


@pytest.mark.unit
def test_equal_slope_lower_term_is_unreachable_but_duplicate_is_invalid() -> None:
    unreachable = certify_selection_geometry(
        np.array([3.0]),
        np.array([[2.0], [2.0]]),
        np.array([1.0, 0.0]),
    )
    duplicate = certify_selection_geometry(
        np.array([3.0]),
        np.array([[2.0], [2.0]]),
        np.array([1.0, 1.0]),
    )

    assert unreachable.geometry.certified
    assert unreachable.geometry.radius_unbounded
    assert unreachable.geometry.facets[0].boundary_unbounded
    assert not unreachable.geometry.facets[0].duplicate_term
    assert not duplicate.geometry.certified
    assert duplicate.geometry.facets[0].duplicate_term


@pytest.mark.unit
def test_topk_membership_radius_uses_selected_unselected_facets() -> None:
    selection = certify_selection_geometry(
        np.array([0.0]),
        np.array([[0.0], [1.0], [-1.0]]),
        np.array([3.0, 2.0, 0.0]),
        top_k=2,
        safety_fraction=0.1,
        choice_ids=("first", "second", "third"),
    )

    assert selection.fingerprint.topk_order == ("first", "second")
    assert selection.geometry.raw_radius == pytest.approx(1.0)
    assert selection.geometry.certified_radius == pytest.approx(0.9)
    assert selection.geometry.certified


@pytest.mark.unit
def test_empty_mask_and_nonfinite_evidence_fail_closed_and_stay_json_safe() -> None:
    empty = certify_selection_geometry(
        np.array([0.0]),
        np.array([[0.0], [1.0]]),
        np.array([1.0, 0.0]),
        eligible=np.array([False, False]),
    )
    nonfinite = certify_selection_geometry(
        np.array([0.0]),
        np.array([[0.0], [1.0]]),
        np.array([math.nan, 0.0]),
    )

    assert not empty.geometry.certified
    assert empty.geometry.scope is GeometryScope.INVALID
    assert not nonfinite.geometry.certified
    assert nonfinite.geometry.scope is GeometryScope.INVALID
    json.dumps(asdict(empty), allow_nan=False)
    json.dumps(asdict(nonfinite), allow_nan=False)


@pytest.mark.unit
def test_malformed_shapes_and_configuration_raise_value_error() -> None:
    with pytest.raises(ValueError, match="input dimension"):
        certify_selection_geometry(
            np.array([0.0, 0.0]),
            np.array([[1.0]]),
            np.array([0.0]),
        )
    with pytest.raises(ValueError, match="strictly between"):
        _basic_selection(safety_fraction=1.0)
    with pytest.raises(ValueError, match="boolean mask"):
        certify_selection_geometry(
            np.array([0.0]),
            np.array([[0.0]]),
            np.array([0.0]),
            eligible=np.array([1]),
        )


@pytest.mark.unit
def test_support_cell_radius_is_composed_and_missing_support_fails_closed() -> None:
    support = certify_selection_geometry(
        np.array([0.0]),
        np.array([[0.0], [2.0]]),
        np.array([1.0, 0.0]),
        safety_fraction=0.1,
        state_digest="shared-support-state",
    )
    augmented = certify_selection_geometry(
        np.array([0.0]),
        np.array([[0.0], [1.0]]),
        np.array([2.0, 0.0]),
        safety_fraction=0.1,
        state_digest="shared-support-state",
        support_required=True,
        support_certificate=support,
    )
    missing = certify_selection_geometry(
        np.array([0.0]),
        np.array([[0.0], [1.0]]),
        np.array([2.0, 0.0]),
        support_required=True,
    )

    assert support.geometry.raw_radius == pytest.approx(0.5)
    assert augmented.geometry.raw_radius == pytest.approx(0.5)
    assert augmented.geometry.certified_radius == pytest.approx(0.45)
    assert augmented.geometry.certified
    assert not missing.geometry.certified
    assert "missing" in missing.geometry.reason


@pytest.mark.unit
def test_uncertified_supplied_support_also_fails_closed() -> None:
    local_support = certify_selection_geometry(
        np.array([0.0]),
        np.array([[0.0], [2.0]]),
        np.array([1.0, 0.0]),
        scope=GeometryScope.LOCAL_ONLY,
        state_digest="shared-support-state",
    )
    selection = certify_selection_geometry(
        np.array([0.0]),
        np.array([[0.0], [1.0]]),
        np.array([2.0, 0.0]),
        state_digest="shared-support-state",
        support_required=True,
        support_certificate=local_support,
    )

    assert not selection.geometry.certified
    assert "support" in selection.geometry.reason


@pytest.mark.unit
def test_local_only_keeps_deterministic_fingerprint_but_refuses_geometry() -> None:
    first = _basic_selection(scope=GeometryScope.LOCAL_ONLY)
    second = _basic_selection(scope=GeometryScope.LOCAL_ONLY)

    assert first.fingerprint.valid
    assert first.fingerprint.replayable
    assert hmac.compare_digest(first.fingerprint.digest, second.fingerprint.digest)
    assert not first.geometry.certified
    assert first.geometry.scope is GeometryScope.LOCAL_ONLY


@pytest.mark.unit
def test_nonreplayable_state_refuses_geometry() -> None:
    selection = _basic_selection(replayable=False)

    assert selection.fingerprint.valid
    assert not selection.fingerprint.replayable
    assert not selection.geometry.certified
    assert "not replayable" in selection.geometry.reason


@pytest.mark.unit
def test_global_lipschitz_is_conservative_without_complete_ledger() -> None:
    slopes = np.array([[1.0, 0.0], [100.0, 0.0]])
    conservative = global_lipschitz_certificate(
        slopes,
        choice_ids=("active", "dominated"),
        input_norm=InputNorm.L2,
    )
    exact = global_lipschitz_certificate(
        slopes,
        choice_ids=("active", "dominated"),
        input_norm=InputNorm.L2,
        nonempty_region_ids=("active",),
        ledger_complete=True,
    )

    assert conservative.valid and conservative.conservative and not conservative.exact
    assert conservative.value == pytest.approx(100.0)
    assert exact.valid and exact.exact and not exact.conservative
    assert exact.value == pytest.approx(1.0)
    with pytest.raises(ValueError, match="requires nonempty_region_ids"):
        global_lipschitz_certificate(slopes, ledger_complete=True)


@pytest.mark.unit
def test_temperature_gate_passes_low_temperature_and_fails_high_temperature() -> None:
    low = temperature_gate(
        np.array([5.0, 0.0]),
        certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
        tau=0.1,
    )
    high = temperature_gate(
        np.array([5.0, 0.0]),
        certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
        tau=10.0,
    )

    assert low.valid and low.passed
    assert low.winner_mass is not None and low.winner_mass > 0.99
    assert high.valid and not high.passed


@pytest.mark.unit
def test_temperature_thresholds_are_inclusive_but_gap_is_strict() -> None:
    baseline = temperature_gate(
        np.array([3.0, 0.0]),
        certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
        tau=0.5,
        min_winner_mass=0.8,
        max_normalized_entropy=0.5,
    )
    assert baseline.winner_mass_lower_bound is not None
    assert baseline.normalized_entropy_upper_bound is not None
    on_thresholds = temperature_gate(
        np.array([3.0, 0.0]),
        certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
        tau=0.5,
        min_winner_mass=baseline.winner_mass_lower_bound,
        max_normalized_entropy=baseline.normalized_entropy_upper_bound,
    )
    strict_gap = temperature_gate(
        np.array([1.0, 0.0]),
        certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
        tau=0.1,
        tie_tol=1.0,
    )

    assert on_thresholds.passed
    assert not strict_gap.passed
    assert "strictly" in strict_gap.reason


@pytest.mark.unit
def test_temperature_rejects_values_just_beyond_inclusive_thresholds() -> None:
    baseline = temperature_gate(
        np.array([3.0, 0.0]),
        certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
        tau=0.5,
        min_winner_mass=0.8,
        max_normalized_entropy=0.5,
    )
    assert baseline.winner_mass_lower_bound is not None
    assert baseline.normalized_entropy_upper_bound is not None

    mass_too_strict = temperature_gate(
        np.array([3.0, 0.0]),
        certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
        tau=0.5,
        min_winner_mass=np.nextafter(baseline.winner_mass_lower_bound, 1.0),
        max_normalized_entropy=0.5,
    )
    entropy_too_strict = temperature_gate(
        np.array([3.0, 0.0]),
        certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
        tau=0.5,
        min_winner_mass=0.8,
        max_normalized_entropy=np.nextafter(
            baseline.normalized_entropy_upper_bound,
            0.0,
        ),
    )

    assert not mass_too_strict.passed
    assert not entropy_too_strict.passed


@pytest.mark.unit
def test_derived_numeric_overflow_fails_closed_and_serializes_strictly() -> None:
    selection = certify_selection_geometry(
        np.array([0.0]),
        np.array([[0.0], [0.0]]),
        np.array([1e308, -1e308]),
    )
    lipschitz = global_lipschitz_certificate(
        np.array([[1.3e308, 1.3e308]]),
        input_norm=InputNorm.L2,
    )
    gate = temperature_gate(
        np.array([1.0, 0.0]),
        certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
        tau=np.nextafter(0.0, 1.0),
    )

    assert not selection.geometry.certified
    assert not selection.fingerprint.valid
    assert not lipschitz.valid and lipschitz.value is None
    assert not gate.valid and not gate.passed
    assert gate.kappa is None
    json.dumps(asdict(selection), allow_nan=False)
    json.dumps(asdict(lipschitz), allow_nan=False)
    json.dumps(asdict(gate), allow_nan=False)


@pytest.mark.unit
def test_singleton_temperature_bypass_is_labeled() -> None:
    gate = temperature_gate(
        np.array([4.0]),
        certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
        tau=1.0,
    )

    assert gate.valid and gate.singleton and gate.passed
    assert gate.gap is None and gate.gap_unbounded
    assert gate.winner_mass == 1.0
    assert gate.normalized_entropy == 0.0
    assert "singleton" in gate.reason


class _FakeLogger:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def event(
        self,
        event: str,
        *,
        level: str = "info",
        step: int | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        record = {"event": event, "level": level, "step": step, **fields}
        self.events.append(record)
        return record


@pytest.mark.unit
def test_monitor_distinguishes_attention_readout_and_output_stability() -> None:
    selection = _basic_selection()
    lipschitz = global_lipschitz_certificate(
        np.array([[0.0, 0.0], [1.0, -2.0]]),
        choice_ids=("winner", "runner_up"),
    )
    low = temperature_gate(
        np.array([2.0, 0.0]),
        certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
        tau=0.1,
        choice_ids=("winner", "runner_up"),
    )
    logger = _FakeLogger()
    monitor = TropicalCertificateMonitor(logger)

    record = monitor.record(
        step=7,
        layer="block.0.attn",
        head=2,
        certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
        selection=selection,
        lipschitz=lipschitz,
        temperature=low,
        pre_dropout=True,
        values_frozen=False,
    )

    assert record.selection_certified
    assert record.readout_certified
    assert record.certified
    assert not record.output_stability_certified
    assert monitor.all_certified()
    assert logger.events[0]["event"] == "tropical_certificate"
    assert logger.events[0]["level"] == "info"


@pytest.mark.unit
def test_monitor_refuses_unbound_lipschitz_and_temperature_artifacts() -> None:
    selection = _basic_selection()
    unrelated_lipschitz = global_lipschitz_certificate(
        np.array([[0.0, 0.0], [1.0, -2.0]]),
        input_norm=InputNorm.L1,
        choice_ids=("other-a", "other-b"),
    )
    stale_temperature = temperature_gate(
        np.array([5.0, 0.0]),
        certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
        tau=0.1,
        choice_ids=("winner", "runner_up"),
    )

    record = TropicalCertificateMonitor().record(
        step=0,
        certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
        selection=selection,
        lipschitz=unrelated_lipschitz,
        temperature=stale_temperature,
        pre_dropout=True,
    )

    assert not record.artifacts_bound
    assert not record.certified
    assert "Lipschitz evidence" in record.reason


@pytest.mark.unit
def test_attention_readout_rejects_membership_only_topk_geometry() -> None:
    selection = certify_selection_geometry(
        np.array([0.0]),
        np.array([[0.0], [1.0]]),
        np.array([2.0, 0.0]),
        top_k=2,
    )
    lipschitz = global_lipschitz_certificate(np.array([[0.0], [1.0]]))
    gate = temperature_gate(
        np.array([2.0, 0.0]),
        certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
        tau=0.1,
    )

    record = TropicalCertificateMonitor().record(
        step=0,
        certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
        selection=selection,
        lipschitz=lipschitz,
        temperature=gate,
        pre_dropout=True,
    )

    assert selection.geometry.certified and selection.geometry.radius_unbounded
    assert not record.selection_certified
    assert not record.certified
    assert "top-k geometry" in record.reason


@pytest.mark.unit
def test_moe_hard_top1_requires_and_composes_outer_membership_geometry() -> None:
    state_digest = "shared-moe-router-state"
    outer = certify_selection_geometry(
        np.array([0.0]),
        np.array([[0.0], [1.0], [-1.0]]),
        np.array([3.0, 2.0, 0.0]),
        choice_ids=("e0", "e1", "e2"),
        top_k=2,
        state_digest=state_digest,
    )
    inner = certify_selection_geometry(
        np.array([0.0]),
        np.array([[0.0], [1.0]]),
        np.array([3.0, 2.0]),
        choice_ids=("e0", "e1"),
        top_k=1,
        state_digest=state_digest,
        support_required=True,
        support_certificate=outer,
    )
    lipschitz = global_lipschitz_certificate(
        np.array([[0.0], [1.0]]),
        choice_ids=("e0", "e1"),
    )
    gate = temperature_gate(
        np.array([3.0, 2.0]),
        certificate_scope=CertificateScope.MOE_HARD_TOP1,
        tau=0.01,
        choice_ids=("e0", "e1"),
    )

    record = TropicalCertificateMonitor().record(
        step=0,
        certificate_scope=CertificateScope.MOE_HARD_TOP1,
        router_top_k=2,
        selection=inner,
        lipschitz=lipschitz,
        temperature=gate,
        pre_dropout=True,
    )

    assert outer.geometry.certified and inner.geometry.certified
    assert outer.geometry.raw_radius == pytest.approx(1.0)
    assert inner.geometry.raw_radius == pytest.approx(1.0)
    assert record.selection_certified and record.readout_certified
    assert record.certified
    assert not record.output_stability_certified


@pytest.mark.unit
def test_support_composition_rejects_different_input_threat_source() -> None:
    support = certify_selection_geometry(
        np.array([0.0]),
        np.array([[0.0], [1.0]]),
        np.array([1.0, 0.0]),
        state_digest="claimed-shared-state",
    )
    selection = certify_selection_geometry(
        np.array([0.0, 0.0]),
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([2.0, 0.0]),
        state_digest="claimed-shared-state",
        support_required=True,
        support_certificate=support,
    )

    assert not selection.geometry.certified
    assert "support geometry" in selection.geometry.reason


@pytest.mark.unit
def test_attention_requires_explicit_pre_dropout_attestation() -> None:
    selection = _basic_selection()
    lipschitz = global_lipschitz_certificate(
        np.array([[0.0, 0.0], [1.0, -2.0]]),
        choice_ids=("winner", "runner_up"),
    )
    gate = temperature_gate(
        np.array([2.0, 0.0]),
        certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
        tau=0.1,
        choice_ids=("winner", "runner_up"),
    )

    record = TropicalCertificateMonitor().record(
        step=0,
        certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
        selection=selection,
        lipschitz=lipschitz,
        temperature=gate,
    )

    assert not record.certified
    assert record.pre_dropout is None
    assert "explicitly attested" in record.reason


@pytest.mark.unit
def test_high_temperature_refuses_attention_but_not_moe_membership() -> None:
    selection = _basic_selection()
    lipschitz = global_lipschitz_certificate(
        np.array([[0.0, 0.0], [1.0, -2.0]]),
        choice_ids=("winner", "runner_up"),
    )
    attention_gate = temperature_gate(
        np.array([2.0, 0.0]),
        certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
        tau=100.0,
        choice_ids=("winner", "runner_up"),
    )
    membership_gate = temperature_gate(
        np.array([2.0, 0.0]),
        certificate_scope=CertificateScope.MOE_TOPK_MEMBERSHIP,
        tau=100.0,
        choice_ids=("winner", "runner_up"),
    )
    monitor = TropicalCertificateMonitor()

    attention = monitor.record(
        step=1,
        certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
        selection=selection,
        lipschitz=lipschitz,
        temperature=attention_gate,
        pre_dropout=True,
    )
    membership = monitor.record(
        step=2,
        certificate_scope=CertificateScope.MOE_TOPK_MEMBERSHIP,
        selection=selection,
        lipschitz=lipschitz,
        temperature=membership_gate,
        router_top_k=1,
    )

    assert not attention.certified
    assert membership.temperature is not None
    assert membership.temperature.passed is None
    assert membership.readout_certified is None
    assert membership.certified


@pytest.mark.unit
def test_monitor_fails_nonvacuously_and_requires_measured_temperature() -> None:
    empty = TropicalCertificateMonitor()
    assert not empty.all_certified()
    selection = _basic_selection()
    lipschitz = global_lipschitz_certificate(
        np.array([[0.0, 0.0], [1.0, -2.0]]),
        choice_ids=("winner", "runner_up"),
    )
    record = empty.record(
        step=0,
        certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
        selection=selection,
        lipschitz=lipschitz,
        pre_dropout=True,
    )

    assert not record.certified
    assert not record.readout_certified
    assert "missing" in record.reason


@pytest.mark.unit
def test_monitor_emits_strict_json_and_rich_table() -> None:
    selection = certify_selection_geometry(
        np.array([0.0]),
        np.array([[1.0]]),
        np.array([0.0]),
    )
    lipschitz = global_lipschitz_certificate(np.array([[1.0]]))
    gate = temperature_gate(
        np.array([0.0]),
        certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
        tau=1.0,
    )
    monitor = TropicalCertificateMonitor()
    monitor.record(
        step=3,
        certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
        selection=selection,
        lipschitz=lipschitz,
        temperature=gate,
        pre_dropout=True,
        values_frozen=True,
    )

    line = monitor.to_jsonl()[0]
    parsed = json.loads(line)
    assert parsed["geometry"]["raw_radius"] is None
    assert parsed["geometry"]["radius_unbounded"]
    assert "NaN" not in line and "Infinity" not in line
    console = Console(record=True, width=120)
    monitor.render(console)
    rendered = console.export_text()
    assert "Tropical selection certificates" in rendered
    assert "certified" in rendered


@pytest.mark.unit
def test_monitor_rejects_mismatched_temperature_scope() -> None:
    selection = _basic_selection()
    lipschitz = global_lipschitz_certificate(
        np.array([[0.0, 0.0], [1.0, -2.0]]),
        choice_ids=("winner", "runner_up"),
    )
    wrong_gate = temperature_gate(
        np.array([2.0, 0.0]),
        certificate_scope=CertificateScope.MOE_TOPK_MEMBERSHIP,
        tau=1.0,
    )

    with pytest.raises(ValueError, match="scope"):
        TropicalCertificateMonitor().record(
            step=0,
            certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
            selection=selection,
            lipschitz=lipschitz,
            temperature=wrong_gate,
        )
