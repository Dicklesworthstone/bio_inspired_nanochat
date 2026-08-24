"""Fail-closed guarantee-bundle and model-card acceptance tests (bead ``r00r.7``)."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, replace

import numpy as np
import pytest

from bio_inspired_nanochat.certificate_bundle import (
    CertificationRefused,
    LiveCrooksPointObservation,
    LiveFTSeedObservation,
    LiveTURObservation,
    MatchedSeedStatisticsObservation,
    ModelIdentity,
    PredictiveCalibrationObservation,
    PredictiveMetricComparisonObservation,
    PredictiveSeedObservation,
    RobustnessObservation,
    StabilityObservation,
    TargetCalibrationObservation,
    build_guarantee_bundle,
    bundle_from_manifest,
    main,
    make_evidence_manifest,
)
from bio_inspired_nanochat.checkpoint_manager import config_hash
from bio_inspired_nanochat.eval_stats import bootstrap_ci, paired_t_test, wilcoxon_signed_rank
from bio_inspired_nanochat.metriplectic_integrator import (
    GuardThresholds,
    run_monitored,
    torch_guarded_step,
)
from bio_inspired_nanochat.stochastic_thermo import (
    HeadPredictiveThermoEvidence,
    PredictiveEvidencePolicy,
    PredictiveEvidenceProvenance,
    PredictiveThermoEvidence,
)
from bio_inspired_nanochat.synaptic import SynapticConfig
from bio_inspired_nanochat.torch_imports import torch
from bio_inspired_nanochat.tropical_certificate import (
    CertificateScope,
    TropicalCertificateMonitor,
    certify_selection_geometry,
    global_lipschitz_certificate,
    temperature_gate,
)
from scripts.e2e.stochastic_thermo_uq import (
    ExperimentConfig,
    run_experiment,
    run_multi_seed,
)

pytestmark = pytest.mark.unit


def _certified_config() -> SynapticConfig:
    return SynapticConfig(
        enable_presyn=True,
        enable_hebbian=True,
        bistable_latch=True,
        cusp_latch=True,
        metriplectic_integrator=True,
        tropical_skeleton=True,
        stochastic_mode="straight_through",
        # A genuinely separated calcium -> release -> fast -> slow -> structure hierarchy.
        tau_rrp=20.0,
        post_trace_decay=0.99,
        post_slow_lr=0.0025,
        structural_interval=4000,
    )


def _identity(cfg: SynapticConfig, *, run_id: str = "cert-run") -> ModelIdentity:
    return ModelIdentity(
        run_id=run_id,
        checkpoint_id="c" * 64,
        config_hash=config_hash(asdict(cfg)),
        predictive_config_hash="d" * 64,
        git_sha="a" * 40,
    )


def _stability(identity: ModelIdentity) -> StabilityObservation:
    calcium = torch.tensor([0.8, 0.6], dtype=torch.float64)
    buffer = torch.tensor([0.1, 0.2], dtype=torch.float64)
    heat = torch.ones(2, dtype=torch.float64)
    records = []
    for _ in range(4):
        calcium, buffer, heat, record = torch_guarded_step(
            calcium, buffer, heat, dt=0.02
        )
        records.append(record)
    return StabilityObservation.from_torch_records(identity, records)


def _predictive(identity: ModelIdentity) -> PredictiveCalibrationObservation:
    policy = PredictiveEvidencePolicy()
    expected_sites = (("transformer.h.0.attn.pre", 0),)
    evidences = tuple(
        PredictiveThermoEvidence(
            provenance=PredictiveEvidenceProvenance(
                run_id=f"predictive-seed-{seed}",
                checkpoint_id=f"{seed:064x}",
                synaptic_config_hash=identity.config_hash,
                config_hash=identity.predictive_config_hash,
                rng_seed=seed + 301,
            ),
            policy=policy,
            heads=(
                HeadPredictiveThermoEvidence(
                    layer_address="transformer.h.0.attn.pre",
                    head_index=0,
                    sampling_modes=("straight_through",),
                    sample_count=8,
                    observed_events=100,
                    tested_events=90,
                    retained_events=90,
                    degenerate_events=10,
                    tested_fraction=0.9,
                    symmetric_bins=2,
                    crooks_residual=0.1,
                    tur_relative_variance=1.0,
                    tur_entropy_bound=1.0,
                    tur_bound_ratio=1.0,
                    finite=True,
                    passed=True,
                    refusal_reasons=(),
                ),
            ),
            observed_events=100,
            tested_events=90,
            retained_events=90,
            degenerate_events=10,
            tested_fraction=0.9,
            fresh=True,
            local_gates_passed=True,
            multi_seed_statistics_passed=True,
            predictive_distribution_claim=True,
            calibration_mode="predictive_thermodynamic_calibration",
            refusal_reasons=(),
        )
        for seed in (11, 23, 37, 41, 53, 67)
    )
    def comparison(
        baseline: str, metric: str, paired_deltas: np.ndarray
    ) -> PredictiveMetricComparisonObservation:
        _, paired_t_p_value = paired_t_test(paired_deltas)
        wilcoxon_p_value = wilcoxon_signed_rank(paired_deltas)
        effect_ci_low, effect_ci_high = bootstrap_ci(
            paired_deltas, n_boot=10_000, seed=20260824
        )
        return PredictiveMetricComparisonObservation(
            baseline=baseline,
            metric=metric,
            seed_count=len(evidences),
            paired_deltas=tuple(float(value) for value in paired_deltas),
            bootstrap_samples=10_000,
            bootstrap_seed=20260824,
            paired_t_p_value=paired_t_p_value,
            wilcoxon_p_value=wilcoxon_p_value,
            effect_ci_low=effect_ci_low,
            effect_ci_high=effect_ci_high,
            favorable_direction="lower" if metric == "ece" else "higher",
            passed=True,
        )

    favorable_lower = np.array([-0.12, -0.08, -0.11, -0.09, -0.13, -0.10])
    favorable_higher = -favorable_lower
    ft_pool_size = 6
    ft_forward_probability = 0.2
    ft_reverse_probability = 0.1
    ft_affinity = (
        math.log(ft_forward_probability)
        + math.log1p(-ft_reverse_probability)
        - math.log(ft_reverse_probability)
        - math.log1p(-ft_forward_probability)
    )
    ft_current_counts = (
        0,
        1,
        27,
        345,
        2547,
        10821,
        24322,
        24347,
        12893,
        3930,
        697,
        67,
        3,
    )
    ft_release_config_hash = config_hash(
        asdict(
            SynapticConfig(
                stochastic_train_frac=1.0,
                stochastic_mode="straight_through",
                stochastic_count_cap=8,
                prime_rate=0.0,
                endo_delay=0,
                init_rrp=6.0,
                rec_rate=ft_reverse_probability,
            )
        )
    )
    ft_integral = math.fsum(
        count * math.exp(-current * ft_affinity)
        for current, count in zip(
            range(-ft_pool_size, ft_pool_size + 1),
            ft_current_counts,
            strict=True,
        )
    ) / sum(ft_current_counts)
    ft_curve = tuple(
        LiveCrooksPointObservation(
            current=current,
            positive_count=ft_current_counts[ft_pool_size + current],
            negative_count=ft_current_counts[ft_pool_size - current],
            observed_log_ratio=math.log(
                ft_current_counts[ft_pool_size + current]
                / ft_current_counts[ft_pool_size - current]
            ),
            expected_log_ratio=current * ft_affinity,
            residual=(
                math.log(
                    ft_current_counts[ft_pool_size + current]
                    / ft_current_counts[ft_pool_size - current]
                )
                - current * ft_affinity
            ),
        )
        for current in (1, 2, 3)
    )
    statistics = MatchedSeedStatisticsObservation.from_measurements(
        identity=identity,
        predictive_run_ids=tuple(
            evidence.provenance.run_id for evidence in evidences
        ),
        predictive_checkpoint_ids=tuple(
            evidence.provenance.checkpoint_id for evidence in evidences
        ),
        predictive_synaptic_config_hashes=tuple(
            evidence.provenance.synaptic_config_hash for evidence in evidences
        ),
        predictive_config_hashes=tuple(
            evidence.provenance.config_hash for evidence in evidences
        ),
        predictive_rng_seeds=tuple(
            evidence.provenance.rng_seed for evidence in evidences
        ),
        expected_sites=expected_sites,
        comparisons=tuple(
            comparison(baseline, metric, deltas)
            for baseline in ("softmax_entropy", "mc_dropout")
            for metric, deltas in (
                ("ece", favorable_lower),
                ("ood_auroc", favorable_higher),
            )
        ),
        alpha=0.05,
        fixed_policy_applied=True,
        live_ft_seeds=tuple(
            LiveFTSeedObservation(
                experiment_seed=seed,
                paired_predictive_run_id=evidence.provenance.run_id,
                experiment_config_hash=evidence.provenance.config_hash,
                release_protocol_config_hash=ft_release_config_hash,
                paired_predictive_rng_seed=evidence.provenance.rng_seed,
                forward_rng_seed=seed + 101,
                reverse_rng_seed=seed + 102,
                scope="one_step_local_detailed_balance",
                n_trajectories=80_000,
                pool_size=ft_pool_size,
                configured_forward_probability=ft_forward_probability,
                configured_reverse_probability=ft_reverse_probability,
                forward_probability=ft_forward_probability,
                reverse_probability=ft_reverse_probability,
                affinity=ft_affinity,
                current_counts=ft_current_counts,
                integral_ft=ft_integral,
                integral_ft_residual=abs(ft_integral - 1.0),
                crooks_curve=ft_curve,
                max_crooks_residual=max(abs(point.residual) for point in ft_curve),
                crooks_min_count=100,
                crooks_tolerance=0.25,
                integral_ft_tolerance=0.04,
                passed=True,
            )
            for seed, evidence in zip(
                (11, 23, 37, 41, 53, 67), evidences, strict=True
            )
        ),
        live_tur=LiveTURObservation(
            scope=(
                "classic_continuous_time_tur_on_exact_one_step_"
                "paired_binomial_moments"
            ),
            pool_size=6,
            forward_probability=0.2,
            reverse_probability=0.1,
            affinity=ft_affinity,
            relative_variance=4.166666666666667,
            entropy_bound=4.110505770627386,
            slack=0.056160896039281205,
            bound_ratio=1.013662770270411,
            nonvacuous=True,
            satisfied=True,
        ),
        passed=True,
    )
    target_evidence = replace(
        evidences[0],
        provenance=PredictiveEvidenceProvenance(
            run_id="deployed-target-seed-10000",
            checkpoint_id=identity.checkpoint_id,
            synaptic_config_hash=identity.config_hash,
            config_hash=identity.predictive_config_hash,
            rng_seed=10_301,
        ),
    )
    target_seed = PredictiveSeedObservation.from_evidence(target_evidence)
    target_calibration = TargetCalibrationObservation.from_measurements(
        target_artifact_sha256=target_seed.artifact_sha256,
        target_provenance=target_evidence.provenance,
        evaluation_distribution=(
            "synthetic_modular_arithmetic_id_heldout_and_ood_half_vocab"
        ),
        evaluation_predictions_per_split=128,
        thermo_ece=0.05,
        thermo_ood_auroc=0.85,
        softmax_ece=0.12,
        softmax_ood_auroc=0.65,
        mc_dropout_ece=0.11,
        mc_dropout_ood_auroc=0.68,
        passed=True,
    )
    observation = PredictiveCalibrationObservation.from_evidences(
        identity,
        evidences,
        target_evidence=target_evidence,
        target_calibration=target_calibration,
        expected_sites=expected_sites,
        statistics=statistics,
    )
    # The controlled fixture stands in for the canonical producer adapter. Production callers
    # cannot authorize through from_evidences(); only from_multi_seed_report() sets this capability.
    return _mark_predictive_source_verified(observation)


def _robustness(identity: ModelIdentity) -> RobustnessObservation:
    x = np.array([0.0, 0.0])
    slopes = np.array([[0.0, 0.0], [1.0, -2.0]])
    offsets = np.array([2.0, 0.0])
    choice_ids = ("winner", "runner_up")
    selection = certify_selection_geometry(
        x,
        slopes,
        offsets,
        choice_ids=choice_ids,
        safety_fraction=0.05,
    )
    lipschitz = global_lipschitz_certificate(slopes, choice_ids=choice_ids)
    temperature = temperature_gate(
        offsets,
        certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
        tau=0.05,
        choice_ids=choice_ids,
    )
    monitor = TropicalCertificateMonitor()
    monitor.record(
        step=7,
        layer="transformer.h.0.attn",
        head=0,
        certificate_scope=CertificateScope.ATTENTION_HARD_READOUT,
        selection=selection,
        lipschitz=lipschitz,
        temperature=temperature,
        pre_dropout=True,
        values_frozen=True,
    )
    return RobustnessObservation.from_monitor(identity, monitor)


def _passing_inputs():
    cfg = _certified_config()
    identity = _identity(cfg)
    return cfg, identity, _stability(identity), _predictive(identity), _robustness(identity)


def _replace_statistics(
    statistics: MatchedSeedStatisticsObservation,
    *,
    comparisons=None,
    alpha=None,
    fixed_policy_applied=None,
    live_ft_seeds=None,
    live_tur=None,
    passed=None,
) -> MatchedSeedStatisticsObservation:
    return MatchedSeedStatisticsObservation.from_measurements(
        identity=statistics.identity,
        predictive_run_ids=statistics.predictive_run_ids,
        predictive_checkpoint_ids=statistics.predictive_checkpoint_ids,
        predictive_synaptic_config_hashes=(
            statistics.predictive_synaptic_config_hashes
        ),
        predictive_config_hashes=statistics.predictive_config_hashes,
        predictive_rng_seeds=statistics.predictive_rng_seeds,
        expected_sites=statistics.expected_sites,
        comparisons=statistics.comparisons if comparisons is None else comparisons,
        alpha=statistics.alpha if alpha is None else alpha,
        fixed_policy_applied=(
            statistics.fixed_policy_applied
            if fixed_policy_applied is None
            else fixed_policy_applied
        ),
        live_ft_seeds=(
            statistics.live_ft_seeds if live_ft_seeds is None else live_ft_seeds
        ),
        live_tur=statistics.live_tur if live_tur is None else live_tur,
        passed=statistics.passed if passed is None else passed,
    )


def _rebind_statistics_to_evidences(
    statistics: MatchedSeedStatisticsObservation,
    evidences: tuple[PredictiveThermoEvidence, ...],
) -> MatchedSeedStatisticsObservation:
    return MatchedSeedStatisticsObservation.from_measurements(
        identity=statistics.identity,
        predictive_run_ids=tuple(item.provenance.run_id for item in evidences),
        predictive_checkpoint_ids=tuple(
            item.provenance.checkpoint_id for item in evidences
        ),
        predictive_synaptic_config_hashes=tuple(
            item.provenance.synaptic_config_hash for item in evidences
        ),
        predictive_config_hashes=tuple(
            item.provenance.config_hash for item in evidences
        ),
        predictive_rng_seeds=tuple(item.provenance.rng_seed for item in evidences),
        expected_sites=statistics.expected_sites,
        comparisons=statistics.comparisons,
        alpha=statistics.alpha,
        fixed_policy_applied=statistics.fixed_policy_applied,
        live_ft_seeds=statistics.live_ft_seeds,
        live_tur=statistics.live_tur,
        passed=statistics.passed,
    )


def _mark_predictive_source_verified(
    observation: PredictiveCalibrationObservation,
) -> PredictiveCalibrationObservation:
    object.__setattr__(observation, "_runtime_verified", True)
    object.__setattr__(observation, "_source_target", observation.target)
    object.__setattr__(
        observation,
        "_source_target_calibration",
        observation.target_calibration,
    )
    object.__setattr__(observation, "_source_seeds", observation.seeds)
    object.__setattr__(observation, "_source_statistics", observation.statistics)
    return observation


def _build(
    cfg: SynapticConfig,
    identity: ModelIdentity,
    stability: StabilityObservation,
    predictive: PredictiveCalibrationObservation,
    robustness: RobustnessObservation,
):
    return build_guarantee_bundle(
        identity=identity,
        config=cfg,
        stability=stability,
        predictive_calibration=predictive,
        robustness=robustness,
        generated_at="2026-08-24T12:00:00+00:00",
    )


def test_controlled_complete_evidence_composes_into_bounded_authorization() -> None:
    bundle = _build(*_passing_inputs())

    assert bundle.deployment_certified
    assert [gate.key for gate in bundle.gates] == [
        "provenance",
        "metriplectic_stability",
        "cusp_retention",
        "predictive_calibration",
        "tropical_robustness",
        "composition",
    ]
    assert all(gate.passed and not gate.failures for gate in bundle.gates)
    bundle.require_deployable()
    payload = bundle.to_dict()
    json.dumps(payload, allow_nan=False)
    assert payload["deployment_certified"]
    assert '"forward_drive"' not in json.dumps(payload)
    assert '"reverse_drive"' not in json.dumps(payload)
    markdown = bundle.to_markdown()
    assert "deployment verdict: **AUTHORIZED**" in markdown
    assert "not a general model-safety guarantee" in markdown
    assert "stable selection does not imply stable selected-expert output" in markdown
    robustness_gate = next(
        gate for gate in bundle.gates if gate.key == "tropical_robustness"
    )
    assert robustness_gate.values["source_records_match"]
    live_record = robustness_gate.values["live_record_details"][0]
    assert live_record["layer"] == "transformer.h.0.attn"
    assert live_record["head"] == 0
    assert live_record["fingerprint"]["eligible_ids"] == ("winner", "runner_up")


def test_model_card_markdown_escapes_untrusted_identity_and_failure_text() -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    injected_reason = "failed\n# Fake Gate\n**AUTHORIZED**"
    record_values = asdict(robustness.records[0])
    record_values["reason"] = injected_reason
    record_values.pop("artifact_sha256")
    record_digest = hashlib.sha256(
        json.dumps(
            record_values,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    injected_record = replace(
        robustness.records[0],
        reason=injected_reason,
        artifact_sha256=record_digest,
    )
    injected_identity = replace(
        identity,
        run_id="run`\n# Injected Heading\n**AUTHORIZED**",
    )
    bundle = _build(
        cfg,
        injected_identity,
        stability,
        predictive,
        replace(robustness, records=(injected_record,)),
    )

    markdown = bundle.to_markdown()
    assert markdown.count("# Live Certificate Model Card") == 1
    assert "\n# Injected Heading" not in markdown
    assert "\n# Fake Gate" not in markdown
    assert r"\# Injected Heading" in markdown
    assert r"\*\*AUTHORIZED\*\*" in markdown
    assert json.dumps(injected_reason) in markdown


def test_empty_or_fallback_stability_trace_is_refused() -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    failed = replace(stability, steps=0, n_fallbacks=2, lyapunov_ok=False)
    bundle = _build(cfg, identity, failed, predictive, robustness)

    gate = next(item for item in bundle.gates if item.key == "metriplectic_stability")
    assert not gate.passed
    assert any("no live" in failure for failure in gate.failures)
    assert any("fallback" in failure for failure in gate.failures)
    with pytest.raises(CertificationRefused, match="deployment certification refused"):
        bundle.require_deployable()


def test_torch_native_guard_records_feed_the_live_stability_gate() -> None:
    cfg, identity, _, predictive, robustness = _passing_inputs()
    calcium = torch.tensor([0.8, 0.6], dtype=torch.float64)
    buffer = torch.tensor([0.1, 0.2], dtype=torch.float64)
    heat = torch.ones(2, dtype=torch.float64)
    records = []
    for _ in range(4):
        calcium, buffer, heat, record = torch_guarded_step(
            calcium, buffer, heat, dt=0.02
        )
        records.append(record)
    stability = StabilityObservation.from_torch_records(identity, records)
    bundle = _build(cfg, identity, stability, predictive, robustness)

    assert stability.steps == 8 and stability.n_fallbacks == 0
    assert stability.lyapunov_ok
    assert next(
        gate for gate in bundle.gates if gate.key == "metriplectic_stability"
    ).passed


def test_numpy_reference_stability_is_reported_but_never_deployment_authorized() -> None:
    cfg, identity, _, predictive, robustness = _passing_inputs()
    _, monitor = run_monitored(np.array([0.8, 0.1, 0.0]), 0.02, 8)
    reference = StabilityObservation.from_monitor(identity, monitor)
    bundle = _build(cfg, identity, reference, predictive, robustness)

    gate = next(item for item in bundle.gates if item.key == "metriplectic_stability")
    assert not gate.passed
    assert any("torch runtime attestation" in failure for failure in gate.failures)


def test_predictive_gate_rejects_duplicates_staleness_and_wrong_checkpoint() -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    first = predictive.seeds[0]
    stale_evidence = replace(
        first.evidence,
        provenance=replace(
            first.provenance,
            checkpoint_id="e" * 64,
        ),
        fresh=False,
        multi_seed_statistics_passed=False,
        predictive_distribution_claim=False,
        calibration_mode="empirical_ece_fallback",
        refusal_reasons=(
            "stale_evidence",
            "multi_seed_statistics_pending_or_failed",
        ),
    )
    bad = PredictiveCalibrationObservation(
        identity=identity,
        expected_sites=predictive.expected_sites,
        target=PredictiveSeedObservation.from_evidence(stale_evidence),
        target_calibration=predictive.target_calibration,
        seeds=(first, PredictiveSeedObservation.from_evidence(stale_evidence)),
        statistics=_replace_statistics(predictive.statistics, passed=False),
    )
    bundle = _build(cfg, identity, stability, bad, robustness)

    gate = next(item for item in bundle.gates if item.key == "predictive_calibration")
    assert not gate.passed
    assert any("duplicated" in failure for failure in gate.failures)
    assert any("stale" in failure for failure in gate.failures)
    assert any("checkpoint" in failure for failure in gate.failures)


def test_target_predictive_source_must_match_live_synaptic_config() -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    mismatched_target = replace(
        predictive.target.evidence,
        provenance=replace(
            predictive.target.provenance,
            synaptic_config_hash="f" * 16,
        ),
    )
    bundle = _build(
        cfg,
        identity,
        stability,
        replace(
            predictive,
            target=PredictiveSeedObservation.from_evidence(mismatched_target),
        ),
        robustness,
    )

    gate = next(item for item in bundle.gates if item.key == "predictive_calibration")
    assert not gate.passed
    assert any("live SynapticConfig" in failure for failure in gate.failures)
    assert gate.values["calibration_mode"] == "empirical_ece_fallback"


def test_direct_predictive_observations_are_report_only() -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    unverified = PredictiveCalibrationObservation.from_evidences(
        identity,
        tuple(seed.evidence for seed in predictive.seeds),
        target_evidence=predictive.target.evidence,
        target_calibration=predictive.target_calibration,
        expected_sites=predictive.expected_sites,
        statistics=predictive.statistics,
    )
    bundle = _build(cfg, identity, stability, unverified, robustness)

    gate = next(item for item in bundle.gates if item.key == "predictive_calibration")
    assert not gate.passed
    assert not gate.values["runtime_attested"]
    assert any("producer reports" in failure for failure in gate.failures)


def test_predictive_cohort_membership_is_fixed_before_inference() -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    shifted_evidence = replace(
        predictive.seeds[0].evidence,
        provenance=replace(
            predictive.seeds[0].provenance,
            rng_seed=predictive.seeds[0].provenance.rng_seed + 1,
        ),
    )
    shifted = replace(
        predictive,
        seeds=(
            PredictiveSeedObservation.from_evidence(shifted_evidence),
            *predictive.seeds[1:],
        ),
    )
    bundle = _build(cfg, identity, stability, shifted, robustness)

    gate = next(item for item in bundle.gates if item.key == "predictive_calibration")
    assert not gate.passed
    assert any("fixed deployment seed policy" in failure for failure in gate.failures)


def test_deployed_target_must_be_distinct_from_research_cohort() -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    overlap_target = predictive.seeds[0]
    source_metrics = predictive.target_calibration
    overlap_calibration = TargetCalibrationObservation.from_measurements(
        target_artifact_sha256=overlap_target.artifact_sha256,
        target_provenance=overlap_target.provenance,
        evaluation_distribution=source_metrics.evaluation_distribution,
        evaluation_predictions_per_split=(
            source_metrics.evaluation_predictions_per_split
        ),
        thermo_ece=source_metrics.thermo_ece,
        thermo_ood_auroc=source_metrics.thermo_ood_auroc,
        softmax_ece=source_metrics.softmax_ece,
        softmax_ood_auroc=source_metrics.softmax_ood_auroc,
        mc_dropout_ece=source_metrics.mc_dropout_ece,
        mc_dropout_ood_auroc=source_metrics.mc_dropout_ood_auroc,
        passed=True,
    )
    overlap = _mark_predictive_source_verified(
        PredictiveCalibrationObservation(
            identity=identity,
            expected_sites=predictive.expected_sites,
            target=overlap_target,
            target_calibration=overlap_calibration,
            seeds=predictive.seeds,
            statistics=predictive.statistics,
        )
    )
    bundle = _build(cfg, identity, stability, overlap, robustness)

    gate = next(item for item in bundle.gates if item.key == "predictive_calibration")
    assert not gate.passed
    assert not gate.values["target_cohort_separate"]
    assert any("overlaps the research cohort" in failure for failure in gate.failures)


def test_predictive_cohort_rejects_duplicate_checkpoint_units() -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    evidences = [seed.evidence for seed in predictive.seeds]
    evidences[1] = replace(
        evidences[1],
        provenance=replace(
            evidences[1].provenance,
            checkpoint_id=evidences[0].provenance.checkpoint_id,
        ),
    )
    evidence_tuple = tuple(evidences)
    duplicated = _mark_predictive_source_verified(
        PredictiveCalibrationObservation.from_evidences(
            identity,
            evidence_tuple,
            target_evidence=predictive.target.evidence,
            target_calibration=predictive.target_calibration,
            expected_sites=predictive.expected_sites,
            statistics=_rebind_statistics_to_evidences(
                predictive.statistics,
                evidence_tuple,
            ),
        )
    )
    bundle = _build(cfg, identity, stability, duplicated, robustness)

    gate = next(item for item in bundle.gates if item.key == "predictive_calibration")
    assert not gate.passed
    assert not gate.values["cohort_checkpoints_unique"]
    assert any("pseudoreplication" in failure for failure in gate.failures)


def test_target_calibration_cannot_be_transplanted_in_manifest() -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    transplanted = replace(predictive, target=predictive.seeds[0])
    manifest = make_evidence_manifest(
        identity=identity,
        config=cfg,
        stability=stability,
        predictive_calibration=transplanted,
        robustness=robustness,
    )

    gate = next(
        item
        for item in bundle_from_manifest(manifest).gates
        if item.key == "predictive_calibration"
    )
    assert not gate.passed
    assert not gate.values["target_calibration"]["target_binding_matches"]
    assert any("not bound to its predictive artifact" in failure for failure in gate.failures)


def test_target_predictive_refusal_flag_cannot_be_promoted() -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    refused_evidence = replace(
        predictive.target.evidence,
        multi_seed_statistics_passed=False,
        predictive_distribution_claim=False,
        calibration_mode="empirical_ece_fallback",
        refusal_reasons=("multi_seed_statistics_pending_or_failed",),
    )
    bundle = _build(
        cfg,
        identity,
        stability,
        replace(
            predictive,
            target=PredictiveSeedObservation.from_evidence(refused_evidence),
        ),
        robustness,
    )

    gate = next(item for item in bundle.gates if item.key == "predictive_calibration")
    assert not gate.passed
    assert any("finalized passing group claim" in failure for failure in gate.failures)


def test_cohort_predictive_refusal_flag_cannot_be_promoted() -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    refused_evidence = replace(
        predictive.seeds[0].evidence,
        multi_seed_statistics_passed=False,
        predictive_distribution_claim=False,
        calibration_mode="empirical_ece_fallback",
        refusal_reasons=("multi_seed_statistics_pending_or_failed",),
    )
    refused_seeds = (
        PredictiveSeedObservation.from_evidence(refused_evidence),
        *predictive.seeds[1:],
    )
    bundle = _build(
        cfg,
        identity,
        stability,
        replace(predictive, seeds=refused_seeds),
        robustness,
    )

    gate = next(item for item in bundle.gates if item.key == "predictive_calibration")
    assert not gate.passed
    assert any("finalized passing group claim" in failure for failure in gate.failures)


def test_replacing_deployed_target_drops_live_source_binding() -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    target_evidence = replace(
        predictive.target.evidence,
        provenance=replace(
            predictive.target.provenance,
            run_id="deployed-target-only",
            rng_seed=9_901,
        ),
        heads=(replace(predictive.target.evidence.heads[0], sample_count=9),),
    )
    target = PredictiveSeedObservation.from_evidence(target_evidence)
    bundle = _build(
        cfg,
        identity,
        stability,
        replace(predictive, target=target),
        robustness,
    )

    assert not bundle.deployment_certified
    gate = next(item for item in bundle.gates if item.key == "predictive_calibration")
    assert not gate.values["source_records_match"]
    target_values = gate.values["target_evidence"]
    assert target_values["artifact_sha256"] == target.artifact_sha256
    assert target_values["evidence"]["provenance"]["run_id"] == "deployed-target-only"
    assert target_values["evidence"]["heads"][0]["sample_count"] == 9
    assert "deployed-target-only" not in gate.values["run_ids"]
    markdown = bundle.to_markdown()
    assert "deployed-target-only" in markdown
    assert target.artifact_sha256 in markdown


def test_tropical_scope_is_preserved_without_promoting_output_stability() -> None:
    cfg, identity, stability, predictive, _ = _passing_inputs()
    x = np.array([0.0, 0.0])
    slopes = np.array([[0.0, 0.0], [1.0, -2.0]])
    offsets = np.array([2.0, 0.0])
    choice_ids = ("winner", "runner_up")
    selection = certify_selection_geometry(
        x, slopes, offsets, choice_ids=choice_ids, safety_fraction=0.05
    )
    monitor = TropicalCertificateMonitor()
    monitor.record(
        step=3,
        layer="transformer.h.0.mlp",
        certificate_scope=CertificateScope.MOE_TOPK_MEMBERSHIP,
        selection=selection,
        lipschitz=global_lipschitz_certificate(slopes, choice_ids=choice_ids),
        router_top_k=1,
    )
    robustness = RobustnessObservation.from_monitor(identity, monitor)
    bundle = _build(cfg, identity, stability, predictive, robustness)

    gate = next(item for item in bundle.gates if item.key == "tropical_robustness")
    assert gate.passed
    assert gate.values["scopes"] == ["moe_topk_membership"]
    assert not gate.values["output_stability_all"]
    assert "does not imply stable selected-expert" in gate.assumptions[2]


def test_default_timescales_and_disabled_toggles_fail_closed() -> None:
    _, _, old_stability, old_predictive, old_robustness = _passing_inputs()
    cfg = SynapticConfig(stochastic_mode="straight_through")
    identity = _identity(cfg, run_id="default-config")
    stability = replace(old_stability, identity=identity)
    predictive = replace(
        old_predictive,
        identity=identity,
        seeds=tuple(
            PredictiveSeedObservation.from_evidence(
                replace(
                    seed.evidence,
                    provenance=replace(
                        seed.provenance,
                        checkpoint_id=identity.checkpoint_id,
                        synaptic_config_hash=identity.config_hash,
                        config_hash=identity.predictive_config_hash,
                    ),
                )
            )
            for seed in old_predictive.seeds
        ),
    )
    robustness = replace(old_robustness, identity=identity)
    bundle = _build(cfg, identity, stability, predictive, robustness)

    failed = {gate.key for gate in bundle.gates if not gate.passed}
    assert {
        "metriplectic_stability",
        "cusp_retention",
        "tropical_robustness",
        "composition",
    } <= failed
    assert not bundle.deployment_certified


def test_provenance_mismatch_and_invalid_config_hash_refuse_aggregation() -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    wrong = replace(identity, run_id="different-run")
    bundle = _build(cfg, wrong, stability, predictive, robustness)

    gate = bundle.gates[0]
    assert gate.key == "provenance" and not gate.passed
    assert len(gate.failures) == 3

    wrong_hash = replace(identity, config_hash="f" * 16)
    bundle = _build(
        cfg,
        wrong_hash,
        replace(stability, identity=wrong_hash),
        replace(predictive, identity=wrong_hash),
        replace(robustness, identity=wrong_hash),
    )
    assert any("normalized" in failure for failure in bundle.gates[0].failures)


def test_nonfinite_refusal_still_serializes_as_strict_json() -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    failed = replace(stability, max_energy_drift=float("nan"))
    bundle = _build(cfg, identity, failed, predictive, robustness)

    assert not next(
        gate for gate in bundle.gates if gate.key == "metriplectic_stability"
    ).passed
    payload = bundle.to_dict()
    json.dumps(payload, allow_nan=False)
    stability_payload = next(
        gate for gate in payload["gates"] if gate["key"] == "metriplectic_stability"
    )
    assert stability_payload["values"]["max_energy_drift"] is None


def test_manifest_roundtrip_emits_artifacts_but_loses_live_attestation(tmp_path) -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    manifest = make_evidence_manifest(
        identity=identity,
        config=cfg,
        stability=stability,
        predictive_calibration=predictive,
        robustness=robustness,
    )
    source = tmp_path / "evidence.json"
    source.write_text(json.dumps(manifest, allow_nan=False), encoding="utf-8")
    output_dir = tmp_path / "card"

    round_trip = bundle_from_manifest(manifest)
    assert not round_trip.deployment_certified
    assert main([str(source), "--output-dir", str(output_dir)]) == 2
    payload = json.loads((output_dir / "model_card.json").read_text(encoding="utf-8"))
    assert not payload["deployment_certified"]
    assert any("live in-process" in reason for reason in payload["refusal_reasons"])
    assert (output_dir / "MODEL_CARD.md").exists()
    events = [
        json.loads(line)
        for line in (output_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert sum(event["event"] == "certificate_gate" for event in events) == 6
    assert any(event["event"] == "certificate_bundle" for event in events)


def test_cli_writes_refusal_card_and_returns_two_by_default(tmp_path) -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    failed = replace(stability, steps=0)
    manifest = make_evidence_manifest(
        identity=identity,
        config=cfg,
        stability=failed,
        predictive_calibration=predictive,
        robustness=robustness,
    )
    source = tmp_path / "failed.json"
    source.write_text(json.dumps(manifest, allow_nan=False), encoding="utf-8")
    output_dir = tmp_path / "refused"

    assert main([str(source), "--output-dir", str(output_dir)]) == 2
    payload = json.loads((output_dir / "model_card.json").read_text(encoding="utf-8"))
    assert not payload["deployment_certified"]
    assert any("no live" in reason for reason in payload["refusal_reasons"])


def test_manifest_rejects_unknown_config_fields() -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    manifest = make_evidence_manifest(
        identity=identity,
        config=cfg,
        stability=stability,
        predictive_calibration=predictive,
        robustness=robustness,
    )
    manifest["synaptic_config"]["invented_certificate_bypass"] = True

    with pytest.raises(ValueError, match="unknown fields"):
        bundle_from_manifest(manifest)


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        ("tau_c", 0.0, "tau_c"),
        ("tau_c", 10**400, "safe integer"),
        ("tau_buf", 0.0, "tau_buf"),
        ("tau_rrp", 0.0, "tau_rrp"),
        ("post_slow_lr", 0.0, "post_slow_lr"),
        ("post_trace_decay", 1.0, "post_trace_decay"),
        ("structural_interval", 0, "structural_interval"),
        ("structural_interval", 10**400, "non-negative int"),
        ("alpha_buf_on", 1.1, "alpha_buf_on"),
        ("latch_hill_n", 1.0, "latch_hill_n"),
        ("latch_hill_k", 0.0, "latch_hill_k"),
        ("latch_input_gain", 0.0, "latch_input_gain"),
        ("camkii_thr", 1.1, "camkii_thr"),
        ("latch_ltd_thr", -0.1, "latch_ltd_thr"),
        ("latch_pp1_basal", 1.1, "latch_pp1_basal"),
        ("latch_alpha_ca", -0.1, "latch_alpha_ca"),
        ("latch_beta_pp1", -0.1, "latch_beta_pp1"),
        ("latch_gamma_auto", -0.1, "latch_gamma_auto"),
        ("cusp_eps_max", 0.0, "cusp_eps_max"),
        ("cusp_eps_max", 0.99, "cusp_eps_max"),
    ],
)
def test_certificate_config_domains_refuse_before_certificate_math(
    field_name, value, match
) -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    invalid = replace(cfg, **{field_name: value})

    with pytest.raises(ValueError, match=match):
        _build(invalid, identity, stability, predictive, robustness)


def test_cli_reports_invalid_certificate_domain_without_uncaught_math_error(
    tmp_path,
) -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    manifest = make_evidence_manifest(
        identity=identity,
        config=cfg,
        stability=stability,
        predictive_calibration=predictive,
        robustness=robustness,
    )
    manifest["synaptic_config"]["tau_buf"] = 0.0
    source = tmp_path / "invalid-domain.json"
    source.write_text(json.dumps(manifest, allow_nan=False), encoding="utf-8")

    assert main([str(source), "--output-dir", str(tmp_path / "card")]) == 2

    manifest["synaptic_config"]["tau_c"] = 10**400
    oversized_source = tmp_path / "oversized-float-input.json"
    oversized_source.write_text(
        json.dumps(manifest, allow_nan=False), encoding="utf-8"
    )
    assert (
        main(
            [
                str(oversized_source),
                "--output-dir",
                str(tmp_path / "oversized-card"),
            ]
        )
        == 2
    )


def test_composition_policy_cannot_be_widened_by_the_caller() -> None:
    cfg = replace(_certified_config(), tau_rrp=9.0)
    identity = _identity(cfg, run_id="composition-policy")
    stability = _stability(identity)
    predictive = _predictive(identity)
    robustness = _robustness(identity)

    default_bundle = _build(
        cfg, identity, stability, predictive, robustness
    )
    composition = next(
        gate for gate in default_bundle.gates if gate.key == "composition"
    )
    assert not composition.passed
    assert composition.values["eligibility"]["A"]["eps"] > 0.5

    with pytest.raises(ValueError, match="must be <= 0.5"):
        build_guarantee_bundle(
            identity=identity,
            config=cfg,
            stability=stability,
            predictive_calibration=predictive,
            robustness=robustness,
            eps_max=0.9,
        )
    with pytest.raises(ValueError, match="integer input"):
        build_guarantee_bundle(
            identity=identity,
            config=cfg,
            stability=stability,
            predictive_calibration=predictive,
            robustness=robustness,
            eps_max=10**400,
        )


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("synaptic_config", "enable_presyn"), "false", "JSON boolean"),
        (
            (
                "predictive_calibration",
                "seeds",
                0,
                "evidence",
                "provenance",
                "rng_seed",
            ),
            -1,
            ">= 0",
        ),
        (("robustness", "records", 0, "scope"), "whole_model_logits", "valid"),
        (("robustness", "records", 0, "input_norm"), "magic-norm", "valid"),
        (("stability", "thresholds", "eps_E"), float("inf"), "finite"),
        (
            (
                "predictive_calibration",
                "statistics",
                "comparisons",
                0,
                "paired_deltas",
                0,
            ),
            -2.0,
            ">= -1.0",
        ),
    ],
)
def test_untrusted_manifest_rejects_coercions_and_out_of_domain_values(
    path, value, match
) -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    manifest = make_evidence_manifest(
        identity=identity,
        config=cfg,
        stability=stability,
        predictive_calibration=predictive,
        robustness=robustness,
    )
    target = manifest
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises((TypeError, ValueError), match=match):
        bundle_from_manifest(manifest)


@pytest.mark.parametrize(
    ("head_changes", "match"),
    [
        ({"crooks_residual": -0.1}, "Crooks residual must be non-negative"),
        (
            {"tur_relative_variance": 1.0, "tur_entropy_bound": 0.5, "tur_bound_ratio": 1.0},
            "TUR bound ratio contradicts",
        ),
    ],
)
def test_predictive_head_physics_is_recomputed_from_raw_values(
    head_changes, match
) -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    first = predictive.seeds[0].evidence
    bad_head = replace(first.heads[0], **head_changes)
    bad_seed = PredictiveSeedObservation.from_evidence(
        replace(first, heads=(bad_head,))
    )
    bad_predictive = replace(
        predictive,
        seeds=(bad_seed, *predictive.seeds[1:]),
    )

    gate = next(
        item
        for item in _build(
            cfg, identity, stability, bad_predictive, robustness
        ).gates
        if item.key == "predictive_calibration"
    )
    assert not gate.passed
    assert any(match in failure for failure in gate.failures)


def test_predictive_head_thresholds_use_exact_recomputed_values() -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    first = predictive.seeds[0].evidence
    bad_head = replace(
        first.heads[0],
        tur_relative_variance=0.9499999999995,
        tur_entropy_bound=1.0,
        tur_bound_ratio=0.95,
    )
    bad_seed = PredictiveSeedObservation.from_evidence(
        replace(first, heads=(bad_head,))
    )
    bad_predictive = replace(
        predictive,
        seeds=(bad_seed, *predictive.seeds[1:]),
    )

    gate = next(
        item
        for item in _build(
            cfg, identity, stability, bad_predictive, robustness
        ).gates
        if item.key == "predictive_calibration"
    )
    assert not gate.passed
    assert any("TUR bound ratio contradicts" in failure for failure in gate.failures)
    assert any("refusal reasons contradict" in failure for failure in gate.failures)


def test_predictive_coverage_threshold_uses_count_derived_fraction() -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    first = predictive.seeds[0].evidence
    observed = 4_000_000_000_000
    tested = 2_999_999_999_999
    retained = first.policy.max_events_per_head
    bad_head = replace(
        first.heads[0],
        observed_events=observed,
        tested_events=tested,
        retained_events=retained,
        degenerate_events=observed - tested,
        tested_fraction=0.75,
    )
    bad_evidence = replace(
        first,
        heads=(bad_head,),
        observed_events=observed,
        tested_events=tested,
        retained_events=retained,
        degenerate_events=observed - tested,
        tested_fraction=0.75,
    )
    bad_predictive = replace(
        predictive,
        seeds=(
            PredictiveSeedObservation.from_evidence(bad_evidence),
            *predictive.seeds[1:],
        ),
    )

    gate = next(
        item
        for item in _build(
            cfg, identity, stability, bad_predictive, robustness
        ).gates
        if item.key == "predictive_calibration"
    )
    assert not gate.passed
    assert any("tested_fraction contradicts" in failure for failure in gate.failures)
    assert any("refusal reasons contradict" in failure for failure in gate.failures)


def test_predictive_head_requires_the_collector_reservoir_invariant() -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    first = predictive.seeds[0].evidence
    bad_head = replace(first.heads[0], retained_events=0)
    bad_seed = PredictiveSeedObservation.from_evidence(
        replace(first, heads=(bad_head,), retained_events=0)
    )
    bad_predictive = replace(
        predictive,
        seeds=(bad_seed, *predictive.seeds[1:]),
    )

    gate = next(
        item
        for item in _build(
            cfg, identity, stability, bad_predictive, robustness
        ).gates
        if item.key == "predictive_calibration"
    )
    assert not gate.passed
    assert any("collector reservoir invariant" in failure for failure in gate.failures)


def test_predictive_head_requires_enough_retained_crooks_support() -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    first = predictive.seeds[0].evidence
    bad_head = replace(
        first.heads[0],
        observed_events=2,
        tested_events=2,
        retained_events=2,
        degenerate_events=0,
        tested_fraction=1.0,
    )
    bad_seed = PredictiveSeedObservation.from_evidence(
        replace(
            first,
            heads=(bad_head,),
            observed_events=2,
            tested_events=2,
            retained_events=2,
            degenerate_events=0,
            tested_fraction=1.0,
        )
    )
    bad_predictive = replace(
        predictive,
        seeds=(bad_seed, *predictive.seeds[1:]),
    )

    gate = next(
        item
        for item in _build(
            cfg, identity, stability, bad_predictive, robustness
        ).gates
        if item.key == "predictive_calibration"
    )
    assert not gate.passed
    assert any("symmetric Crooks bins" in failure for failure in gate.failures)


def test_predictive_statistics_require_exact_canonical_comparison_matrix() -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    statistics = _replace_statistics(
        predictive.statistics,
        comparisons=predictive.statistics.comparisons[:-1],
        passed=False,
    )
    bundle = _build(
        cfg,
        identity,
        stability,
        replace(predictive, statistics=statistics),
        robustness,
    )

    gate = next(item for item in bundle.gates if item.key == "predictive_calibration")
    assert not gate.passed
    assert any("exactly ECE and OOD-AUROC" in failure for failure in gate.failures)


def test_one_seed_predictive_artifact_renders_power_refusal() -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    bundle = _build(
        cfg,
        identity,
        stability,
        replace(predictive, seeds=predictive.seeds[:1]),
        robustness,
    )

    gate = next(item for item in bundle.gates if item.key == "predictive_calibration")
    assert not gate.passed
    assert any(
        "fewer than two predictive evidence seeds" in failure
        for failure in gate.failures
    )
    assert "fewer than two predictive evidence seeds" in bundle.to_markdown()


def test_invalid_comparison_matrix_skips_bootstrap_recomputation(monkeypatch) -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    statistics = _replace_statistics(
        predictive.statistics,
        comparisons=(
            *predictive.statistics.comparisons,
            predictive.statistics.comparisons[0],
        ),
        passed=False,
    )

    def unexpected_bootstrap(*args, **kwargs):
        raise AssertionError("bootstrap must not run for an invalid comparison matrix")

    monkeypatch.setattr(
        "bio_inspired_nanochat.certificate_bundle.bootstrap_ci",
        unexpected_bootstrap,
    )
    bundle = _build(
        cfg,
        identity,
        stability,
        replace(predictive, statistics=statistics),
        robustness,
    )

    gate = next(item for item in bundle.gates if item.key == "predictive_calibration")
    assert not gate.passed
    assert all(
        "recomputation_skipped" in item
        for item in gate.values["recomputed_comparisons"]
    )


def test_comparison_vectors_have_a_fixed_resource_bound() -> None:
    _, _, _, predictive, _ = _passing_inputs()
    comparison = predictive.statistics.comparisons[0]

    with pytest.raises(ValueError, match="must be <= 256"):
        replace(
            comparison,
            seed_count=257,
            paired_deltas=(-0.1,) * 257,
        )


def test_predictive_statistics_recompute_p_values_and_confidence_intervals() -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    comparisons = list(predictive.statistics.comparisons)
    comparisons[0] = replace(comparisons[0], paired_t_p_value=0.04)
    statistics = _replace_statistics(
        predictive.statistics,
        comparisons=tuple(comparisons),
        passed=False,
    )
    bundle = _build(
        cfg,
        identity,
        stability,
        replace(predictive, statistics=statistics),
        robustness,
    )

    gate = next(item for item in bundle.gates if item.key == "predictive_calibration")
    assert not gate.passed
    assert not gate.values["recomputed_comparisons"][0]["reported_values_match"]


def test_predictive_statistics_recompute_each_live_ft_seed_and_tur() -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    ft_seeds = list(predictive.statistics.live_ft_seeds)
    bad_curve = list(ft_seeds[0].crooks_curve)
    bad_curve[0] = replace(bad_curve[0], residual=0.0)
    ft_seeds[0] = replace(
        ft_seeds[0], crooks_curve=tuple(bad_curve), passed=False
    )
    invalid_tur = replace(
        predictive.statistics.live_tur,
        slack=0.5,
        bound_ratio=1.0,
        satisfied=True,
    )
    statistics = _replace_statistics(
        predictive.statistics,
        live_ft_seeds=tuple(ft_seeds),
        live_tur=invalid_tur,
        passed=False,
    )
    bundle = _build(
        cfg,
        identity,
        stability,
        replace(predictive, statistics=statistics),
        robustness,
    )

    gate = next(item for item in bundle.gates if item.key == "predictive_calibration")
    assert not gate.passed
    assert any("not every predictive seed" in failure for failure in gate.failures)
    assert any("TUR measurements" in failure for failure in gate.failures)


def test_live_ft_protocol_hash_cannot_be_detached_from_measured_probabilities() -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    forged_protocol_hash = config_hash(
        asdict(
            SynapticConfig(
                stochastic_train_frac=1.0,
                stochastic_mode="straight_through",
                stochastic_count_cap=8,
                prime_rate=0.0,
                endo_delay=0,
                init_rrp=6.0,
                rec_rate=0.9,
            )
        )
    )
    ft_seeds = list(predictive.statistics.live_ft_seeds)
    ft_seeds[0] = replace(
        ft_seeds[0],
        configured_forward_probability=0.95,
        configured_reverse_probability=0.9,
        release_protocol_config_hash=forged_protocol_hash,
        passed=False,
    )
    statistics = _replace_statistics(
        predictive.statistics,
        live_ft_seeds=tuple(ft_seeds),
        passed=False,
    )
    gate = next(
        item
        for item in _build(
            cfg,
            identity,
            stability,
            replace(predictive, statistics=statistics),
            robustness,
        ).gates
        if item.key == "predictive_calibration"
    )

    assert not gate.passed
    assert not gate.values["recomputed_live_ft"][0]["protocol_valid"]


def test_canonical_multi_seed_report_adapts_distinct_ft_sources_and_refusals() -> None:
    base_config = ExperimentConfig(
        vocab_size=16,
        seq_len=4,
        batch_size=1,
        pool_size=1,
        eval_pool_size=1,
        train_steps=0,
        n_head=1,
        n_embd=8,
        dropout=0.1,
        mc_samples=2,
        ece_bins=4,
        ft_forward_probability=0.9,
        ft_reverse_probability=0.01,
    )
    source = run_multi_seed(base_config, (11, 23))
    target_source = run_experiment(replace(base_config, seed=31))
    source_evidences = tuple(
        report.predictive_thermo_evidence for report in source.reports
    )
    target_evidence = target_source.predictive_thermo_evidence
    actual_cfg = SynapticConfig(stochastic_mode="straight_through")
    identity = ModelIdentity(
        run_id="canonical-multi-seed-adapter",
        checkpoint_id=target_evidence.provenance.checkpoint_id,
        config_hash=config_hash(asdict(actual_cfg)),
        predictive_config_hash=source_evidences[0].provenance.config_hash,
        git_sha="a" * 40,
    )
    predictive = PredictiveCalibrationObservation.from_multi_seed_report(
        identity,
        source,
        target_report=target_source,
    )
    target_calibration = predictive.target_calibration
    assert (
        target_calibration.evaluation_distribution
        == "synthetic_modular_arithmetic_id_heldout_and_ood_half_vocab"
    )
    assert target_calibration.evaluation_predictions_per_split == 4
    transplanted_calibration = TargetCalibrationObservation.from_measurements(
        target_artifact_sha256=predictive.target.artifact_sha256,
        target_provenance=predictive.target.provenance,
        evaluation_distribution="different_distribution",
        evaluation_predictions_per_split=5,
        thermo_ece=target_calibration.thermo_ece,
        thermo_ood_auroc=target_calibration.thermo_ood_auroc,
        softmax_ece=target_calibration.softmax_ece,
        softmax_ood_auroc=target_calibration.softmax_ood_auroc,
        mc_dropout_ece=target_calibration.mc_dropout_ece,
        mc_dropout_ood_auroc=target_calibration.mc_dropout_ood_auroc,
        passed=target_calibration.passed,
    )
    transplanted_gate = next(
        item
        for item in build_guarantee_bundle(
            identity=identity,
            config=actual_cfg,
            stability=_stability(identity),
            predictive_calibration=replace(
                predictive,
                target_calibration=transplanted_calibration,
            ),
            robustness=_robustness(identity),
            generated_at="2026-08-24T12:00:00+00:00",
        ).gates
        if item.key == "predictive_calibration"
    )
    assert not transplanted_gate.passed
    assert not transplanted_gate.values["source_records_match"]
    assert transplanted_gate.values["target_calibration"]["target_binding_matches"]
    worst_target_methods = dict(target_source.methods)
    worst_target_methods["thermo_uq"] = replace(
        worst_target_methods["thermo_uq"],
        ece=1.0,
        ood_auroc=0.0,
    )
    worst_target = replace(target_source, methods=worst_target_methods)
    worst_target_predictive = PredictiveCalibrationObservation.from_multi_seed_report(
        identity,
        source,
        target_report=worst_target,
    )
    worst_target_gate = next(
        item
        for item in build_guarantee_bundle(
            identity=identity,
            config=actual_cfg,
            stability=_stability(identity),
            predictive_calibration=worst_target_predictive,
            robustness=_robustness(identity),
            generated_at="2026-08-24T12:00:00+00:00",
        ).gates
        if item.key == "predictive_calibration"
    )
    assert not worst_target_gate.passed
    assert not worst_target_gate.values["target_calibration"]["recomputed_passed"]
    assert any(
        "deployed target failed" in failure
        for failure in worst_target_gate.failures
    )
    reordered = PredictiveCalibrationObservation.from_multi_seed_report(
        identity,
        replace(source, reports=list(reversed(source.reports))),
        target_report=target_source,
    )
    assert (
        reordered.statistics.artifact_sha256
        == predictive.statistics.artifact_sha256
    )
    with pytest.raises(ValueError, match="bootstrap_seed must equal"):
        PredictiveCalibrationObservation.from_multi_seed_report(
            identity,
            replace(source, bootstrap_seed=0),
            target_report=target_source,
        )
    contradictory_summary = replace(
        source,
        predictive_distribution=replace(
            source.predictive_distribution,
            local_seed_pass_rate=0.123,
            refusal_reasons=("forged_summary",),
        ),
    )
    with pytest.raises(ValueError, match="cohort verdict contradicts"):
        PredictiveCalibrationObservation.from_multi_seed_report(
            identity,
            contradictory_summary,
            target_report=target_source,
        )
    bad_target_rng = replace(
        target_source,
        predictive_thermo_evidence=replace(
            target_evidence,
            provenance=replace(
                target_evidence.provenance,
                rng_seed=target_evidence.provenance.rng_seed + 1,
            ),
        ),
    )
    with pytest.raises(ValueError, match="RNG provenance"):
        PredictiveCalibrationObservation.from_multi_seed_report(
            identity,
            source,
            target_report=bad_target_rng,
        )
    bad_target_run_id = replace(
        target_source,
        predictive_thermo_evidence=replace(
            target_evidence,
            provenance=replace(
                target_evidence.provenance,
                run_id="forged-target-run",
            ),
        ),
    )
    with pytest.raises(ValueError, match="run ID contradicts"):
        PredictiveCalibrationObservation.from_multi_seed_report(
            identity,
            source,
            target_report=bad_target_run_id,
        )

    ft_seeds = predictive.statistics.live_ft_seeds
    assert tuple(item.experiment_seed for item in ft_seeds) == (11, 23)
    assert tuple(item.forward_rng_seed for item in ft_seeds) == (112, 124)
    assert tuple(item.reverse_rng_seed for item in ft_seeds) == (113, 125)
    assert tuple(item.paired_predictive_rng_seed for item in ft_seeds) == (312, 324)
    assert len({item.paired_predictive_run_id for item in ft_seeds}) == 2
    assert target_evidence.provenance.run_id not in {
        seed.paired_predictive_run_id for seed in ft_seeds
    }
    assert all(sum(item.current_counts) == item.n_trajectories for item in ft_seeds)
    assert all(not item.crooks_curve and item.max_crooks_residual is None for item in ft_seeds)

    gate = next(
        item
        for item in build_guarantee_bundle(
            identity=identity,
            config=actual_cfg,
            stability=_stability(identity),
            predictive_calibration=predictive,
            robustness=_robustness(identity),
            generated_at="2026-08-24T12:00:00+00:00",
        ).gates
        if item.key == "predictive_calibration"
    )
    assert not gate.passed
    target_values = gate.values["target_evidence"]
    assert target_values["artifact_sha256"] == predictive.target.artifact_sha256
    assert (
        target_values["evidence"]["provenance"]["run_id"]
        == target_evidence.provenance.run_id
    )
    assert target_evidence.provenance.run_id in build_guarantee_bundle(
        identity=identity,
        config=actual_cfg,
        stability=_stability(identity),
        predictive_calibration=predictive,
        robustness=_robustness(identity),
        generated_at="2026-08-24T12:00:00+00:00",
    ).to_markdown()
    assert all(
        result["curve_values_match"]
        and result["summaries_match"]
        and result["cohort_pair_bound"]
        and not result["recomputed_passed"]
        for result in gate.values["recomputed_live_ft"]
    )


def test_live_tur_preserves_the_source_exact_nonnegative_slack_predicate() -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    forward_probability = 0.3576637104179802
    reverse_probability = 0.1
    pool_size = 6
    affinity = (
        math.log(forward_probability)
        + math.log1p(-reverse_probability)
        - math.log(reverse_probability)
        - math.log1p(-forward_probability)
    )
    mean_current = pool_size * (forward_probability - reverse_probability)
    variance = pool_size * (
        forward_probability * (1.0 - forward_probability)
        + reverse_probability * (1.0 - reverse_probability)
    )
    relative_variance = variance / (mean_current * mean_current)
    entropy_bound = 2.0 / (mean_current * affinity)
    slack = relative_variance - entropy_bound
    assert -1e-12 < slack < 0.0
    boundary_tur = replace(
        predictive.statistics.live_tur,
        forward_probability=forward_probability,
        reverse_probability=reverse_probability,
        affinity=affinity,
        relative_variance=relative_variance,
        entropy_bound=entropy_bound,
        slack=slack,
        bound_ratio=relative_variance / entropy_bound,
        nonvacuous=True,
        satisfied=True,
    )
    statistics = _replace_statistics(
        predictive.statistics,
        live_tur=boundary_tur,
        passed=False,
    )
    bundle = _build(
        cfg,
        identity,
        stability,
        replace(predictive, statistics=statistics),
        robustness,
    )

    gate = next(item for item in bundle.gates if item.key == "predictive_calibration")
    assert not gate.passed
    assert not gate.values["recomputed_live_tur"]["satisfied"]


def test_live_tur_underflow_and_oversized_integer_fail_closed(tmp_path) -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    subnormal_tur = replace(
        predictive.statistics.live_tur,
        reverse_probability=5e-324,
        forward_probability=1e-323,
        affinity=float(np.log(2.0)),
        relative_variance=1.0,
        entropy_bound=1.0,
        slack=0.0,
        bound_ratio=1.0,
        nonvacuous=True,
        satisfied=True,
    )
    statistics = _replace_statistics(
        predictive.statistics,
        live_tur=subnormal_tur,
        passed=False,
    )
    refused_predictive = replace(predictive, statistics=statistics)
    bundle = _build(
        cfg, identity, stability, refused_predictive, robustness
    )
    gate = next(item for item in bundle.gates if item.key == "predictive_calibration")
    assert not gate.passed
    assert not gate.values["recomputed_live_tur"]["primitives_valid"]

    manifest = make_evidence_manifest(
        identity=identity,
        config=cfg,
        stability=stability,
        predictive_calibration=refused_predictive,
        robustness=robustness,
    )
    source = tmp_path / "subnormal-tur.json"
    source.write_text(json.dumps(manifest, allow_nan=False), encoding="utf-8")
    assert main([str(source), "--output-dir", str(tmp_path / "subnormal")]) == 2

    with pytest.raises(ValueError, match="must be <="):
        replace(subnormal_tur, pool_size=10**400)
    manifest["predictive_calibration"]["statistics"]["live_tur"]["pool_size"] = (
        10**400
    )
    oversized = tmp_path / "oversized-pool.json"
    oversized.write_text(json.dumps(manifest, allow_nan=False), encoding="utf-8")
    assert main([str(oversized), "--output-dir", str(tmp_path / "oversized")]) == 2


def test_predictive_statistics_cannot_be_transplanted_across_checkpoints() -> None:
    cfg = _certified_config()
    identity_a = _identity(cfg, run_id="stats-source-a")
    identity_b = replace(
        identity_a,
        run_id="stats-target-b",
        checkpoint_id="e" * 64,
        git_sha="b" * 40,
    )
    statistics_a = _predictive(identity_a).statistics
    predictive_b = _predictive(identity_b)
    transplanted = replace(predictive_b, statistics=statistics_a)

    bundle = _build(
        cfg,
        identity_b,
        _stability(identity_b),
        transplanted,
        _robustness(identity_b),
    )
    gate = next(item for item in bundle.gates if item.key == "predictive_calibration")
    assert not gate.passed
    assert any("not bound to this model identity" in failure for failure in gate.failures)


def test_predictive_statistics_policy_and_content_digest_are_fixed() -> None:
    _, _, _, predictive, _ = _passing_inputs()
    with pytest.raises(ValueError, match="alpha must equal"):
        _replace_statistics(predictive.statistics, alpha=0.01)
    with pytest.raises(ValueError, match="direction contradicts"):
        replace(
            predictive.statistics.comparisons[0], favorable_direction="higher"
        )
    with pytest.raises(ValueError, match="must be >= -1.0"):
        replace(
            predictive.statistics.comparisons[0],
            paired_deltas=(-1e100,) * 6,
        )
    with pytest.raises(ValueError, match="bootstrap_samples must equal"):
        replace(
            predictive.statistics.comparisons[0],
            bootstrap_samples=1,
            bootstrap_seed=0,
        )
    with pytest.raises(ValueError, match="does not match its content"):
        replace(predictive.statistics, artifact_sha256="0" * 64)
    with pytest.raises(ValueError, match="crooks_tolerance must equal"):
        replace(
            predictive.statistics.live_ft_seeds[0],
            crooks_tolerance=1e101,
            integral_ft_tolerance=1e101,
            passed=True,
        )


def test_offline_source_labels_cannot_restore_live_runtime_attestation() -> None:
    cfg, identity, _, predictive, robustness = _passing_inputs()
    _, monitor = run_monitored(np.array([0.8, 0.1, 0.0]), 0.02, 8)
    reference = StabilityObservation.from_monitor(identity, monitor)
    manifest = make_evidence_manifest(
        identity=identity,
        config=cfg,
        stability=reference,
        predictive_calibration=predictive,
        robustness=robustness,
    )
    manifest["stability"]["source"] = "torch_runtime"

    bundle = bundle_from_manifest(manifest)
    stability_gate = next(
        item for item in bundle.gates if item.key == "metriplectic_stability"
    )
    robustness_gate = next(
        item for item in bundle.gates if item.key == "tropical_robustness"
    )
    assert not stability_gate.passed and not stability_gate.values["runtime_attested"]
    assert not robustness_gate.passed
    assert not robustness_gate.values["source_records_match"]


def test_tropical_summary_cannot_self_attest_an_inconsistent_readout() -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    manifest = make_evidence_manifest(
        identity=identity,
        config=cfg,
        stability=stability,
        predictive_calibration=predictive,
        robustness=robustness,
    )
    record = manifest["robustness"]["records"][0]
    record["readout_certified"] = False
    content = {key: value for key, value in record.items() if key != "artifact_sha256"}
    record["artifact_sha256"] = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    bundle = bundle_from_manifest(manifest)
    gate = next(item for item in bundle.gates if item.key == "tropical_robustness")
    assert not gate.passed
    assert any("scope/binding/radius" in failure for failure in gate.failures)


def test_stability_threshold_policy_is_fixed_and_finite() -> None:
    cfg, identity, stability, predictive, robustness = _passing_inputs()
    with pytest.raises(ValueError, match="finite"):
        replace(stability, thresholds=GuardThresholds(float("inf"), 1e-10, 1e-8))

    relaxed = replace(stability, thresholds=GuardThresholds(1e-4, 1e-10, 1e-8))
    bundle = _build(cfg, identity, relaxed, predictive, robustness)
    gate = next(item for item in bundle.gates if item.key == "metriplectic_stability")
    assert not gate.passed
    assert any("exceed" in failure for failure in gate.failures)


def test_cli_rejects_duplicate_keys_and_nonstandard_json_constants(tmp_path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"schema_version": NaN}', encoding="utf-8")

    assert main([str(duplicate), "--output-dir", str(tmp_path / "duplicate")]) == 2
    assert main([str(nonfinite), "--output-dir", str(tmp_path / "nonfinite")]) == 2


def test_cli_rejects_excessive_json_nesting_without_artifacts(tmp_path) -> None:
    nested = tmp_path / "nested.json"
    nested.write_text("[" * 10_000 + "0" + "]" * 10_000, encoding="utf-8")
    output_dir = tmp_path / "nested-card"

    assert main([str(nested), "--output-dir", str(output_dir)]) == 2
    assert not output_dir.exists()
