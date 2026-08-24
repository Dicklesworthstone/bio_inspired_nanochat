"""Focused tests for the read-only tropical certificate runtime."""

from __future__ import annotations

import json
import hmac
import math
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest
from rich.console import Console

from bio_inspired_nanochat.synaptic import SynapticConfig
from bio_inspired_nanochat.tropical_certificate import (
    CertificateScope,
    GeometryScope,
    InputNorm,
    TropicalCertificateMonitor,
    TropicalRoutingConfig,
    TropicalRoutingController,
    TropicalRoutingMode,
    TropicalRoutingTransition,
    certify_selection_geometry,
    deterministic_argmax,
    global_lipschitz_certificate,
    temperature_gate,
    tropical_readout_or_baseline,
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


def _tropical_syn_cfg(
    *,
    enabled: bool = True,
    barrier_strength: float = 0.1,
) -> SynapticConfig:
    return SynapticConfig(
        tropical_skeleton=enabled,
        barrier_strength=barrier_strength,
    )


def _attention_runtime_record(
    tau: float,
    *,
    step: int,
    schedule_digest: str,
    score_gap: float = 2.0,
    layer: str = "test.attn",
    head: int = 0,
):
    offsets = np.array([score_gap, 0.0])
    slopes = np.array([[0.0, 0.0], [1.0, -2.0]])
    ids = ("winner", "runner_up")
    selection = certify_selection_geometry(
        np.array([0.0, 0.0]),
        slopes,
        offsets,
        choice_ids=ids,
        state_digest="runtime-attention-state",
    )
    lipschitz = global_lipschitz_certificate(slopes, choice_ids=ids)
    gate = temperature_gate(
        offsets,
        certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
        tau=tau,
        choice_ids=ids,
    )
    return TropicalCertificateMonitor().record(
        step=step,
        layer=layer,
        head=head,
        certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
        selection=selection,
        lipschitz=lipschitz,
        temperature=gate,
        pre_dropout=True,
        values_frozen=True,
        schedule_digest=schedule_digest,
    )


def _membership_runtime_record(
    tau: float,
    *,
    step: int,
    schedule_digest: str,
):
    slopes = np.array([[0.0, 0.0], [1.0, -2.0]])
    ids = ("winner", "runner_up")
    selection = certify_selection_geometry(
        np.array([0.0, 0.0]),
        slopes,
        np.array([2.0, 0.0]),
        choice_ids=ids,
        state_digest="runtime-moe-state",
    )
    lipschitz = global_lipschitz_certificate(slopes, choice_ids=ids)
    gate = temperature_gate(
        np.array([2.0, 0.0]),
        certificate_scope=CertificateScope.MOE_TOPK_MEMBERSHIP,
        tau=tau,
        choice_ids=ids,
    )
    return TropicalCertificateMonitor().record(
        step=step,
        layer="test.moe",
        certificate_scope=CertificateScope.MOE_TOPK_MEMBERSHIP,
        router_top_k=1,
        selection=selection,
        lipschitz=lipschitz,
        temperature=gate,
        schedule_digest=schedule_digest,
    )


@pytest.mark.unit
def test_tropical_routing_config_rejects_malformed_schedules() -> None:
    with pytest.raises(ValueError, match="tau_start"):
        TropicalRoutingConfig(tau_start=0.0)
    with pytest.raises(ValueError, match="tau_min"):
        TropicalRoutingConfig(tau_start=0.1, tau_min=0.2)
    with pytest.raises(ValueError, match="anneal_steps"):
        TropicalRoutingConfig(anneal_steps=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="barrier_end"):
        TropicalRoutingConfig(barrier_end=math.nan)
    with pytest.raises(ValueError, match="entry_windows"):
        TropicalRoutingConfig(entry_windows=0)


@pytest.mark.unit
def test_geometric_temperature_and_linear_barrier_schedule_reach_endpoints() -> None:
    controller = TropicalRoutingController(
        _tropical_syn_cfg(),
        TropicalRoutingConfig(
            tau_start=1.0,
            tau_min=0.125,
            anneal_steps=3,
            barrier_end=0.4,
            entry_windows=1,
        ),
    )
    points = []
    for _ in range(4):
        points.append(controller.schedule_point())
        controller.observe(None)

    assert [point.progress for point in points] == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])
    assert [point.tau for point in points] == pytest.approx([1.0, 0.5, 0.25, 0.125])
    assert [point.barrier_strength for point in points] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert all(left.tau > right.tau for left, right in zip(points, points[1:]))


@pytest.mark.unit
def test_disabled_controller_is_state_identity_and_returns_exact_baseline_object() -> None:
    controller = TropicalRoutingController(
        _tropical_syn_cfg(enabled=False),
        TropicalRoutingConfig(tau_start=0.1, tau_min=0.1, entry_windows=1),
    )
    before = controller.state()
    point = controller.schedule_point()
    decision = controller.observe(
        _attention_runtime_record(
            point.tau,
            step=0,
            schedule_digest=point.digest,
        ),
        observed_barrier_strength=point.barrier_strength,
    )
    baseline = np.array([4.0, -1.0])
    readout = tropical_readout_or_baseline(
        baseline,
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([2.0, 0.0]),
        decision,
    )

    assert controller.state() == before
    assert decision.mode is TropicalRoutingMode.DISABLED
    assert decision.transition is TropicalRoutingTransition.DISABLED
    assert decision.used_baseline and not decision.use_hard_path
    assert readout.value is baseline


@pytest.mark.unit
def test_consecutive_certificates_enter_hard_mode_and_attribute_exactly() -> None:
    controller = TropicalRoutingController(
        _tropical_syn_cfg(),
        TropicalRoutingConfig(
            tau_start=0.1,
            tau_min=0.1,
            entry_windows=2,
        ),
    )
    first_point = controller.schedule_point()
    first = controller.observe(
        _attention_runtime_record(
            first_point.tau,
            step=0,
            schedule_digest=first_point.digest,
        ),
        observed_barrier_strength=first_point.barrier_strength,
    )
    second_point = controller.schedule_point()
    second = controller.observe(
        _attention_runtime_record(
            second_point.tau,
            step=1,
            schedule_digest=second_point.digest,
        ),
        observed_barrier_strength=second_point.barrier_strength,
    )
    values = np.array([[7.0, 11.0], [-2.0, 5.0]])
    readout = tropical_readout_or_baseline(
        np.array([0.0, 0.0]),
        values,
        np.array([2.0, 0.0]),
        second,
        choice_ids=("winner", "runner_up"),
    )

    assert first.mode is TropicalRoutingMode.SOFT_APPROXIMATION
    assert first.gate_passed and not first.use_hard_path
    assert second.transition is TropicalRoutingTransition.ENTER_HARD
    assert second.use_hard_path and second.authorized_choice_id == "winner"
    assert readout.used_hard_path and readout.choice_id == "winner"
    assert readout.choice_index == 0
    assert np.shares_memory(readout.value, values)
    np.testing.assert_array_equal(readout.value, values[0])


@pytest.mark.unit
def test_any_hard_mode_gate_failure_falls_back_on_the_same_decision() -> None:
    controller = TropicalRoutingController(
        _tropical_syn_cfg(),
        TropicalRoutingConfig(
            tau_start=0.1,
            tau_min=0.1,
            entry_windows=1,
        ),
    )
    point = controller.schedule_point()
    entered = controller.observe(
        _attention_runtime_record(
            point.tau,
            step=0,
            schedule_digest=point.digest,
        ),
        observed_barrier_strength=point.barrier_strength,
    )
    failed = controller.observe(None)
    baseline = np.array([3.0])
    readout = tropical_readout_or_baseline(
        baseline,
        np.array([[8.0], [1.0]]),
        np.array([2.0, 0.0]),
        failed,
    )

    assert entered.use_hard_path
    assert failed.transition is TropicalRoutingTransition.EXIT_TO_SOFT
    assert failed.used_baseline and not failed.hard_active
    assert "immediate certificate fallback" in failed.reason
    assert readout.value is baseline


@pytest.mark.unit
def test_high_temperature_and_stale_schedule_evidence_stay_soft() -> None:
    high_temperature = TropicalRoutingController(
        _tropical_syn_cfg(),
        TropicalRoutingConfig(
            tau_start=1.0,
            tau_min=1.0,
            entry_windows=1,
        ),
    )
    point = high_temperature.schedule_point()
    high = high_temperature.observe(
        _attention_runtime_record(
            point.tau,
            step=0,
            schedule_digest=point.digest,
        ),
        observed_barrier_strength=point.barrier_strength,
    )
    stale_controller = TropicalRoutingController(
        _tropical_syn_cfg(),
        TropicalRoutingConfig(
            tau_start=0.2,
            tau_min=0.2,
            entry_windows=1,
        ),
    )
    stale_point = stale_controller.schedule_point()
    stale = stale_controller.observe(
        _attention_runtime_record(
            0.1,
            step=0,
            schedule_digest=stale_point.digest,
        ),
        observed_barrier_strength=stale_point.barrier_strength,
    )

    assert high.mode is TropicalRoutingMode.SOFT_APPROXIMATION
    assert not high.gate_passed and "soft approximation" in high.reason
    assert not stale.gate_passed
    assert "temperature did not match" in stale.reason


@pytest.mark.unit
def test_barrier_mismatch_and_membership_scope_cannot_authorize_hard_readout() -> None:
    config = TropicalRoutingConfig(
        tau_start=0.1,
        tau_min=0.1,
        entry_windows=1,
    )
    barrier_controller = TropicalRoutingController(
        _tropical_syn_cfg(),
        config,
    )
    point = barrier_controller.schedule_point()
    mismatch = barrier_controller.observe(
        _attention_runtime_record(
            point.tau,
            step=0,
            schedule_digest=point.digest,
        ),
        observed_barrier_strength=point.barrier_strength + 0.01,
    )
    membership_controller = TropicalRoutingController(
        _tropical_syn_cfg(),
        config,
    )
    membership_point = membership_controller.schedule_point()
    membership = membership_controller.observe(
        _membership_runtime_record(
            0.1,
            step=0,
            schedule_digest=membership_point.digest,
        )
    )

    assert not mismatch.gate_passed
    assert "barrier did not match" in mismatch.reason
    assert not membership.gate_passed
    assert "membership cannot authorize" in membership.reason


@pytest.mark.unit
def test_local_only_and_nonfinite_records_fail_closed_without_advancing_hard_state() -> None:
    config = TropicalRoutingConfig(
        tau_start=0.1,
        tau_min=0.1,
        entry_windows=1,
    )
    local_controller = TropicalRoutingController(_tropical_syn_cfg(), config)
    slopes = np.array([[0.0, 0.0], [1.0, -2.0]])
    ids = ("winner", "runner_up")
    local_selection = certify_selection_geometry(
        np.array([0.0, 0.0]),
        slopes,
        np.array([2.0, 0.0]),
        scope=GeometryScope.LOCAL_ONLY,
        choice_ids=ids,
    )
    local_record = TropicalCertificateMonitor().record(
        step=0,
        layer="test.attn",
        head=0,
        certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
        selection=local_selection,
        lipschitz=global_lipschitz_certificate(slopes, choice_ids=ids),
        temperature=temperature_gate(
            np.array([2.0, 0.0]),
            certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
            tau=0.1,
            choice_ids=ids,
        ),
        pre_dropout=True,
        schedule_digest=local_controller.schedule_point().digest,
    )
    local_point = local_controller.schedule_point()
    local = local_controller.observe(
        local_record,
        observed_barrier_strength=local_point.barrier_strength,
    )

    nonfinite_controller = TropicalRoutingController(_tropical_syn_cfg(), config)
    nonfinite_point = nonfinite_controller.schedule_point()
    valid = _attention_runtime_record(
        0.1,
        step=0,
        schedule_digest=nonfinite_point.digest,
    )
    assert valid.temperature is not None
    forged_nonfinite = replace(
        valid,
        temperature=replace(valid.temperature, tau=math.nan),
    )
    nonfinite = nonfinite_controller.observe(
        forged_nonfinite,
        observed_barrier_strength=nonfinite_point.barrier_strength,
    )

    assert not local.gate_passed and "exact_affine" in local.reason
    assert not nonfinite.gate_passed
    assert "strict finite JSON" in nonfinite.reason
    assert not nonfinite_controller.state().hard_active


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "bad_value", "reason_fragment"),
    [
        ("layer", 7, "layer/site ID"),
        ("head", -1, "non-negative head ID"),
        ("schedule_digest", 7, "schedule digest"),
    ],
)
def test_wrong_typed_certificate_metadata_falls_back_without_raising(
    field: str,
    bad_value: Any,
    reason_fragment: str,
) -> None:
    config = TropicalRoutingConfig(
        tau_start=0.1,
        tau_min=0.1,
        entry_windows=1,
    )
    controller = TropicalRoutingController(_tropical_syn_cfg(), config)
    point = controller.schedule_point()
    valid = _attention_runtime_record(
        point.tau,
        step=0,
        schedule_digest=point.digest,
    )
    forged = replace(valid, **{field: bad_value})

    decision = controller.observe(
        forged,
        observed_barrier_strength=point.barrier_strength,
    )

    assert not decision.gate_passed and decision.used_baseline
    assert reason_fragment in decision.reason


@pytest.mark.unit
def test_hard_readout_rechecks_bound_scores_and_falls_back_on_drift() -> None:
    controller = TropicalRoutingController(
        _tropical_syn_cfg(),
        TropicalRoutingConfig(
            tau_start=0.1,
            tau_min=0.1,
            entry_windows=1,
        ),
    )
    point = controller.schedule_point()
    decision = controller.observe(
        _attention_runtime_record(
            point.tau,
            step=0,
            schedule_digest=point.digest,
        ),
        observed_barrier_strength=point.barrier_strength,
    )
    baseline = np.array([13.0, -8.0])
    readout = tropical_readout_or_baseline(
        baseline,
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([1.9, 0.0]),
        decision,
        choice_ids=("winner", "runner_up"),
    )

    assert decision.use_hard_path
    assert not readout.used_hard_path
    assert readout.value is baseline
    assert "scores did not match" in readout.reason


@pytest.mark.unit
def test_stale_certificate_replay_cannot_accumulate_entry_windows() -> None:
    controller = TropicalRoutingController(
        _tropical_syn_cfg(),
        TropicalRoutingConfig(
            tau_start=0.1,
            tau_min=0.1,
            entry_windows=2,
        ),
    )
    first_point = controller.schedule_point()
    first_record = _attention_runtime_record(
        first_point.tau,
        step=0,
        schedule_digest=first_point.digest,
    )
    first = controller.observe(
        first_record,
        observed_barrier_strength=first_point.barrier_strength,
    )
    second_point = controller.schedule_point()
    replay = controller.observe(
        replace(first_record, step=1),
        observed_barrier_strength=second_point.barrier_strength,
    )

    assert first.entry_streak == 1
    assert not replay.gate_passed and not replay.use_hard_path
    assert "schedule digest did not match" in replay.reason
    assert controller.state().entry_streak == 0


@pytest.mark.unit
def test_entry_hysteresis_is_bound_to_one_routing_site() -> None:
    controller = TropicalRoutingController(
        _tropical_syn_cfg(),
        TropicalRoutingConfig(
            tau_start=0.1,
            tau_min=0.1,
            entry_windows=2,
        ),
    )
    first_point = controller.schedule_point()
    first = controller.observe(
        _attention_runtime_record(
            first_point.tau,
            step=0,
            schedule_digest=first_point.digest,
            layer="block.0.attn",
        ),
        observed_barrier_strength=first_point.barrier_strength,
    )
    second_point = controller.schedule_point()
    changed_site = controller.observe(
        _attention_runtime_record(
            second_point.tau,
            step=1,
            schedule_digest=second_point.digest,
            layer="block.1.attn",
        ),
        observed_barrier_strength=second_point.barrier_strength,
    )
    third_point = controller.schedule_point()
    entered = controller.observe(
        _attention_runtime_record(
            third_point.tau,
            step=2,
            schedule_digest=third_point.digest,
            layer="block.1.attn",
        ),
        observed_barrier_strength=third_point.barrier_strength,
    )
    fourth_point = controller.schedule_point()
    exited = controller.observe(
        _attention_runtime_record(
            fourth_point.tau,
            step=3,
            schedule_digest=fourth_point.digest,
            layer="block.0.attn",
        ),
        observed_barrier_strength=fourth_point.barrier_strength,
    )

    assert first.entry_streak == 1
    assert changed_site.entry_streak == 1 and not changed_site.use_hard_path
    assert entered.transition is TropicalRoutingTransition.ENTER_HARD
    assert exited.transition is TropicalRoutingTransition.EXIT_TO_SOFT
    assert "routing site changed" in exited.reason


@pytest.mark.unit
def test_anonymous_routing_sites_cannot_collide_into_hard_entry() -> None:
    controller = TropicalRoutingController(
        _tropical_syn_cfg(),
        TropicalRoutingConfig(
            tau_start=0.1,
            tau_min=0.1,
            entry_windows=2,
        ),
    )
    decisions = []
    for step in range(2):
        point = controller.schedule_point()
        decisions.append(
            controller.observe(
                _attention_runtime_record(
                    point.tau,
                    step=step,
                    schedule_digest=point.digest,
                    layer="",
                ),
                observed_barrier_strength=point.barrier_strength,
            )
        )

    assert all(not decision.gate_passed for decision in decisions)
    assert all("layer/site ID" in decision.reason for decision in decisions)
    assert controller.state().entry_streak == 0
    assert not controller.state().hard_active


@pytest.mark.unit
def test_state_loader_rejects_unreachable_schedule_and_route_states() -> None:
    config = TropicalRoutingConfig(
        tau_start=0.1,
        tau_min=0.1,
        anneal_steps=5,
        entry_windows=2,
    )
    controller = TropicalRoutingController(_tropical_syn_cfg(), config)
    point = controller.schedule_point()
    controller.observe(
        _attention_runtime_record(
            point.tau,
            step=0,
            schedule_digest=point.digest,
        ),
        observed_barrier_strength=point.barrier_strength,
    )
    valid = controller.state_dict()

    unreachable_step = {**valid, "schedule_step": 0}
    with pytest.raises(ValueError, match="not reachable"):
        controller.load_state_dict(unreachable_step)
    missing_route = {**valid, "route_digest": None}
    with pytest.raises(ValueError, match="route_digest"):
        controller.load_state_dict(missing_route)
    excessive_streak = {**valid, "entry_streak": 2}
    with pytest.raises(ValueError):
        controller.load_state_dict(excessive_streak)

    disabled = TropicalRoutingController(_tropical_syn_cfg(enabled=False), config)
    forged_disabled = {
        **disabled.state_dict(),
        "schedule_step": 1,
        "decision_count": 1,
    }
    with pytest.raises(ValueError, match="disabled"):
        disabled.load_state_dict(forged_disabled)


@pytest.mark.unit
def test_controller_state_roundtrip_preserves_entry_streak_and_schedule() -> None:
    config = TropicalRoutingConfig(
        tau_start=0.1,
        tau_min=0.1,
        anneal_steps=5,
        entry_windows=2,
    )
    original = TropicalRoutingController(_tropical_syn_cfg(), config)
    point = original.schedule_point()
    original.observe(
        _attention_runtime_record(
            point.tau,
            step=0,
            schedule_digest=point.digest,
        ),
        observed_barrier_strength=point.barrier_strength,
    )
    serialized = json.loads(json.dumps(original.state_dict(), allow_nan=False))
    resumed = TropicalRoutingController(_tropical_syn_cfg(), config)
    resumed.load_state_dict(serialized)

    assert resumed.state() == original.state()
    assert resumed.schedule_point() == original.schedule_point()
    resumed_point = resumed.schedule_point()
    decision = resumed.observe(
        _attention_runtime_record(
            resumed_point.tau,
            step=1,
            schedule_digest=resumed_point.digest,
        ),
        observed_barrier_strength=resumed_point.barrier_strength,
    )
    assert decision.transition is TropicalRoutingTransition.ENTER_HARD

    mismatched = TropicalRoutingController(
        _tropical_syn_cfg(),
        TropicalRoutingConfig(
            tau_start=0.1,
            tau_min=0.05,
            anneal_steps=5,
            entry_windows=2,
        ),
    )
    with pytest.raises(ValueError, match="different schedule config"):
        mismatched.load_state_dict(serialized)


@pytest.mark.unit
def test_transition_logging_json_and_rich_output_are_strict_and_detailed() -> None:
    logger = _FakeLogger()
    controller = TropicalRoutingController(
        _tropical_syn_cfg(),
        TropicalRoutingConfig(
            tau_start=0.1,
            tau_min=0.1,
            entry_windows=1,
        ),
        logger=logger,
    )
    point = controller.schedule_point()
    controller.observe(
        _attention_runtime_record(
            point.tau,
            step=0,
            schedule_digest=point.digest,
        ),
        observed_barrier_strength=point.barrier_strength,
    )

    line = controller.to_jsonl()[0]
    parsed = json.loads(line)
    assert parsed["mode"] == "hard"
    assert parsed["schedule"]["tau"] == 0.1
    assert parsed["certificate_digest"]
    assert "NaN" not in line and "Infinity" not in line
    assert logger.events[0]["event"] == "tropical_routing_transition"
    console = Console(record=True, width=120)
    controller.render(console)
    assert "Tropical routing transitions" in console.export_text()


@pytest.mark.unit
def test_deterministic_argmax_uses_choice_id_tie_rule_but_ties_do_not_certify() -> None:
    index, choice_id = deterministic_argmax(
        np.array([4.0, 4.0, 1.0]),
        choice_ids=("z", "a", "m"),
    )
    tied = certify_selection_geometry(
        np.array([0.0]),
        np.array([[0.0], [0.0], [0.0]]),
        np.array([4.0, 4.0, 1.0]),
        choice_ids=("z", "a", "m"),
    )

    assert (index, choice_id) == (1, "a")
    assert tied.fingerprint.active_ids == ("a", "z")
    assert not tied.geometry.certified


@pytest.mark.unit
def test_seeded_soft_readout_converges_to_exact_hard_attribution() -> None:
    rng = np.random.default_rng(642622)
    values = rng.normal(size=(3, 7))
    scores = np.array([2.0, 0.5, -1.0])
    winner = values[0]
    errors = []
    for tau in (1.0, 0.5, 0.2, 0.1, 0.02):
        shifted = (scores - scores.max()) / tau
        weights = np.exp(shifted) / np.exp(shifted).sum()
        errors.append(float(np.linalg.norm(weights @ values - winner)))

    assert all(left > right for left, right in zip(errors, errors[1:]))
    assert errors[-1] < 1e-12


@pytest.mark.unit
def test_seeded_certified_radius_lower_bounds_measured_adversarial_flip() -> None:
    rng = np.random.default_rng(642623)
    selection = _basic_selection()
    certified = selection.geometry.certified_radius
    raw = selection.geometry.raw_radius
    assert certified is not None and raw is not None
    x = np.zeros(2)
    slopes = np.array([[0.0, 0.0], [1.0, -2.0]])
    offsets = np.array([2.0, 0.0])
    baseline_winner = int(np.argmax(slopes @ x + offsets))

    for _ in range(2_000):
        direction = rng.normal(size=2)
        direction /= np.linalg.norm(direction)
        radius = rng.uniform(0.0, certified * 0.999)
        assert int(np.argmax(slopes @ (x + radius * direction) + offsets)) == baseline_winner

    nearest_facet_direction = np.array([1.0, -2.0]) / math.sqrt(5.0)
    measured_flip = raw * (1.0 + 1e-8)
    flipped = int(
        np.argmax(slopes @ (x + measured_flip * nearest_facet_direction) + offsets)
    )
    assert flipped != baseline_winner
    assert certified < measured_flip
