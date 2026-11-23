"""Tests for the statistical testing layer (bead 74f.3).

The distribution functions are validated against KNOWN reference values (standard
t-tables / textbook Wilcoxon results) so the pure-numpy implementations are provably
correct, then the aggregation/comparison logic is checked on synthetic matrices.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from typing import Any, cast

import numpy as np
import pytest

from bio_inspired_nanochat.eval_stats import (
    D1_MECHANISMS,
    D1_PARAMETER_MATCH_RULE,
    D1_PROTOCOL_VERSION,
    _format_markdown_report,
    _strict_json_value,
    aggregate,
    bootstrap_ci,
    compare_matrix,
    d1_confirm,
    d1_confirmation_plan,
    d1_csv_report,
    d1_screen,
    holm_adjust,
    load_matrix_csv,
    main,
    paired_comparison,
    paired_t_test,
    structural_descriptive,
    t_cdf,
    t_ppf,
    t_sf_two_sided,
    wilcoxon_signed_rank,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Student-t against known reference values
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "df,expected",
    [(1, 12.7062), (2, 4.3027), (4, 2.7764), (9, 2.2622), (30, 2.0423), (100, 1.9840)],
)
def test_t_ppf_matches_tables(df, expected):
    assert t_ppf(0.975, df) == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize(
    "t,df,expected_two_sided_p",
    [(2.0, 4, 0.11611), (3.0, 2, 0.09547), (1.0, 10, 0.34086), (0.0, 5, 1.0)],
)
def test_t_sf_two_sided_matches_known(t, df, expected_two_sided_p):
    assert t_sf_two_sided(t, df) == pytest.approx(expected_two_sided_p, abs=1e-4)


def test_t_cdf_symmetry_and_center():
    assert t_cdf(0.0, 7) == pytest.approx(0.5, abs=1e-9)
    assert t_cdf(2.0, 4) == pytest.approx(0.94195, abs=1e-4)
    assert t_cdf(-2.0, 4) == pytest.approx(1.0 - 0.94195, abs=1e-4)


# --------------------------------------------------------------------------- #
# Paired t-test
# --------------------------------------------------------------------------- #
def test_paired_t_identical_is_t0_p1():
    t, p = paired_t_test(np.zeros(5))
    assert t == 0.0 and p == 1.0


def test_paired_t_known_value():
    # deltas = [-1,-2,-3,-4,-5]: mean -3, sd sqrt(2.5), t = -4.2426, df=4, p≈0.0132.
    t, p = paired_t_test(np.array([-1.0, -2.0, -3.0, -4.0, -5.0]))
    assert t == pytest.approx(-4.2426, abs=1e-3)
    assert p == pytest.approx(0.0132, abs=1e-3)


def test_paired_t_constant_shift_is_significant():
    t, p = paired_t_test(np.full(4, 0.5))  # zero variance, non-zero mean
    assert math.isinf(t) and p == 0.0


# --------------------------------------------------------------------------- #
# Wilcoxon signed-rank
# --------------------------------------------------------------------------- #
def test_wilcoxon_all_positive_small_n_exact():
    # n=5 all-positive differences: exact two-sided p = 2 * (1/32) = 0.0625.
    assert wilcoxon_signed_rank(np.array([1.0, 2.0, 3.0, 4.0, 5.0])) == pytest.approx(
        0.0625, abs=1e-9
    )


def test_wilcoxon_drops_zeros_and_symmetric_is_one():
    assert wilcoxon_signed_rank(np.zeros(6)) == 1.0
    # Perfectly symmetric ranks -> p == 1.0.
    assert wilcoxon_signed_rank(np.array([1.0, -1.0, 2.0, -2.0])) == pytest.approx(1.0)


def test_wilcoxon_large_n_normal_approx_runs():
    rng = np.random.default_rng(0)
    d = rng.normal(0.0, 1.0, size=40)
    p = wilcoxon_signed_rank(d)
    assert 0.0 <= p <= 1.0


# --------------------------------------------------------------------------- #
# Bootstrap + aggregate
# --------------------------------------------------------------------------- #
def test_bootstrap_ci_deterministic_and_brackets_mean():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lo1, hi1 = bootstrap_ci(a, seed=42, n_boot=2000)
    lo2, hi2 = bootstrap_ci(a, seed=42, n_boot=2000)
    assert (lo1, hi1) == (lo2, hi2)  # deterministic
    assert lo1 <= a.mean() <= hi1


def test_aggregate_single_value_has_zero_width_ci():
    agg = aggregate([3.5])
    assert agg.n == 1 and agg.mean == 3.5 and agg.ci_low == 3.5 and agg.ci_high == 3.5


def test_aggregate_known_ci():
    # [2,4,6]: mean 4, std 2, sem 2/sqrt(3), t_.975,2 = 4.3027 -> half-width 4.969.
    agg = aggregate([2.0, 4.0, 6.0])
    assert agg.mean == pytest.approx(4.0)
    assert agg.std == pytest.approx(2.0)
    half = 4.3027 * (2.0 / math.sqrt(3))
    assert agg.ci_low == pytest.approx(4.0 - half, abs=1e-2)
    assert agg.ci_high == pytest.approx(4.0 + half, abs=1e-2)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_aggregate_rejects_nonfinite_observations(bad):
    with pytest.raises(ValueError, match="non-finite observation"):
        aggregate([1.0, bad])


def test_aggregate_rejects_non_vector_evidence_and_invalid_confidence():
    with pytest.raises(ValueError, match="one-dimensional"):
        aggregate(np.ones((2, 2)))
    with pytest.raises(ValueError, match="confidence"):
        aggregate([1.0, 2.0], confidence=1.0)


# --------------------------------------------------------------------------- #
# paired_comparison + compare_matrix
# --------------------------------------------------------------------------- #
def test_paired_comparison_needs_two_shared_seeds():
    assert paired_comparison({1: 1.0}, {1: 2.0}, lower_is_better=True) is None
    res = paired_comparison({1: 1.0, 2: 1.0}, {1: 2.0, 2: 2.0}, lower_is_better=True)
    assert res is not None and res.n_pairs == 2

    with pytest.raises(TypeError, match="lower_is_better"):
        paired_comparison(
            {1: 1.0},
            {1: 2.0},
            lower_is_better=cast(Any, 1),
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_paired_comparison_rejects_nonfinite_matched_observations(bad):
    with pytest.raises(ValueError, match="non-finite observation"):
        paired_comparison(
            {1: 1.0, 2: bad, 3: 1.2},
            {1: 1.1, 2: 1.1, 3: 1.1},
            lower_is_better=True,
        )


def test_compare_matrix_detects_consistent_improvement():
    # Six non-zero pairs are the mathematical minimum for an exact two-sided Wilcoxon p < .05.
    baseline = {seed: 1.0 + seed * 0.01 for seed in range(1, 7)}
    deltas = (-0.08, -0.09, -0.07, -0.10, -0.06, -0.11)
    data = {
        "vanilla": baseline,
        "bio_all": {
            seed: baseline[seed] + delta
            for seed, delta in zip(baseline, deltas, strict=True)
        },
    }
    rep = compare_matrix(data, baseline="vanilla", metric="val_bpb")
    assert rep["lower_is_better"] is True
    bio = rep["presets"]["bio_all"]
    assert bio["better"] is True
    assert bio["significant"] is True
    assert bio["supported_gain"] is True
    assert bio["verdict"] == "supported_gain"
    assert bio["ci_favorable"] is True
    assert bio["tests_pass"] is True
    assert bio["paired_vs_baseline"]["n_favorable"] == 6
    assert bio["paired_vs_baseline"]["mean_delta"] < 0  # lower bpb
    # baseline row carries no paired comparison.
    assert "paired_vs_baseline" not in rep["presets"]["vanilla"]


def test_compare_matrix_no_difference_is_not_significant():
    data = {
        "vanilla": {1: 1.0, 2: 1.1, 3: 0.9},
        "bio_all": {1: 1.0, 2: 1.1, 3: 0.9},
    }
    rep = compare_matrix(data, baseline="vanilla", metric="val_bpb")
    assert rep["presets"]["bio_all"]["significant"] is False
    assert rep["presets"]["bio_all"]["better"] is False
    assert rep["presets"]["bio_all"]["verdict"] == "null"


def test_compare_matrix_does_not_cherry_pick_the_paired_t_test():
    # A constant three-seed improvement has paired-t p=0, but exact two-sided Wilcoxon p=.25.
    # The old ``min(p_t, p_w)`` rule incorrectly called this significant.
    data = {
        "vanilla": {1: 1.0, 2: 1.1, 3: 0.9},
        "bio_all": {1: 0.8, 2: 0.9, 3: 0.7},
    }
    entry = compare_matrix(data, baseline="vanilla")["presets"]["bio_all"]
    assert entry["paired_vs_baseline"]["t_p_value"] < 0.05
    assert entry["paired_vs_baseline"]["wilcoxon_p_value"] == pytest.approx(0.25)
    assert entry["ci_favorable"] is True
    assert entry["tests_pass"] is False
    assert entry["significant"] is False
    assert entry["verdict"] == "null"


def test_compare_matrix_reports_supported_regression():
    baseline = {seed: 1.0 + seed * 0.01 for seed in range(1, 7)}
    deltas = (0.08, 0.09, 0.07, 0.10, 0.06, 0.11)
    data = {
        "vanilla": baseline,
        "bio_all": {
            seed: baseline[seed] + delta
            for seed, delta in zip(baseline, deltas, strict=True)
        },
    }
    entry = compare_matrix(data, baseline="vanilla")["presets"]["bio_all"]
    assert entry["better"] is False
    assert entry["ci_adverse"] is True
    assert entry["supported_regression"] is True
    assert entry["verdict"] == "supported_regression"


@pytest.mark.parametrize("pairs", [1, 2])
def test_compare_matrix_fails_closed_on_too_few_pairs(pairs):
    data = {
        "vanilla": {seed: 1.0 for seed in range(pairs)},
        "bio_all": {seed: 0.8 for seed in range(pairs)},
    }
    entry = compare_matrix(data, baseline="vanilla")["presets"]["bio_all"]
    assert entry["evidence_sufficient"] is False
    assert entry["significant"] is False
    assert entry["verdict"] == "insufficient_evidence"


def test_holm_adjust_known_family_and_rejects_invalid_values():
    assert holm_adjust({"a": 0.01, "b": 0.03, "c": 0.04}) == pytest.approx(
        {"a": 0.03, "b": 0.06, "c": 0.06}
    )
    with pytest.raises(ValueError, match="p-values"):
        holm_adjust({"impossible": 1.1})


def test_compare_matrix_holm_corrects_across_all_treatments():
    # At n=6 an all-favorable exact Wilcoxon has raw p=.03125. With two treatments in the
    # family, Holm adjusts both to .0625, so neither can be promoted on its raw p-value alone.
    baseline = {seed: 1.0 + seed * 0.01 for seed in range(6)}
    data = {
        "vanilla": baseline,
        "bio_a": {seed: value - 0.10 - seed * 0.001 for seed, value in baseline.items()},
        "bio_b": {seed: value - 0.08 - seed * 0.001 for seed, value in baseline.items()},
    }
    report = compare_matrix(data, baseline="vanilla")
    for preset in ("bio_a", "bio_b"):
        entry = report["presets"][preset]
        assert entry["paired_vs_baseline"]["wilcoxon_p_value"] == pytest.approx(0.03125)
        assert entry["wilcoxon_p_adjusted"] == pytest.approx(0.0625)
        assert entry["ci_favorable"] is True
        assert entry["tests_pass"] is False
        assert entry["verdict"] == "null"


def test_compare_matrix_higher_better_metric_direction():
    # niah_acc is higher-better: a consistent accuracy gain must read as better.
    data = {
        "vanilla": {1: 0.50, 2: 0.55, 3: 0.45},
        "bio_all": {1: 0.70, 2: 0.72, 3: 0.68},
    }
    rep = compare_matrix(data, baseline="vanilla", metric="niah_acc")
    assert rep["lower_is_better"] is False
    assert rep["presets"]["bio_all"]["better"] is True
    assert rep["presets"]["bio_all"]["paired_vs_baseline"]["mean_delta"] > 0
    assert rep["presets"]["bio_all"]["verdict"] == "null"


def test_compare_matrix_neutral_metric_requires_explicit_direction():
    data = {
        "vanilla": {seed: 1.0 for seed in range(6)},
        "bio_all": {seed: 2.0 + seed * 0.01 for seed in range(6)},
    }

    with pytest.raises(ValueError, match="neutral direction"):
        compare_matrix(
            data,
            baseline="vanilla",
            metric="live_tur_relative_variance",
        )

    report = compare_matrix(
        data,
        baseline="vanilla",
        metric="live_tur_relative_variance",
        lower_is_better=False,
    )
    assert report["presets"]["bio_all"]["supported_gain"] is True


def test_compare_matrix_rejects_nonboolean_direction_override():
    with pytest.raises(TypeError, match="lower_is_better"):
        compare_matrix(
            {"vanilla": {1: 1.0}},
            baseline="vanilla",
            metric="val_bpb",
            lower_is_better=cast(Any, 1),
        )


def test_compare_matrix_unknown_baseline_raises():
    with pytest.raises(ValueError, match="baseline"):
        compare_matrix({"a": {1: 1.0}}, baseline="nope", metric="val_bpb")


@pytest.mark.parametrize(
    "kwargs,match",
    [({"alpha": 0.0}, "alpha"), ({"alpha": 1.0}, "alpha"), ({"min_pairs": 1}, "min_pairs")],
)
def test_compare_matrix_rejects_invalid_decision_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        compare_matrix({"vanilla": {1: 1.0}}, baseline="vanilla", **kwargs)


def test_report_serialization_is_strict_json_and_markdown():
    data = {
        "vanilla": {seed: 1.0 for seed in range(6)},
        "bio_all": {seed: 0.8 for seed in range(6)},
    }
    report = compare_matrix(data, baseline="vanilla")
    encoded = json.dumps(_strict_json_value(report), allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded["presets"]["bio_all"]["paired_vs_baseline"]["t_stat"] is None
    markdown = _format_markdown_report(report)
    assert "`supported_gain`" in markdown
    assert "Holm correction" in markdown


# --------------------------------------------------------------------------- #
# CSV ingestion
# --------------------------------------------------------------------------- #
def test_load_matrix_csv(tmp_path):
    p = tmp_path / "summary.csv"
    p.write_text(
        "status,preset,seed,val_bpb\n"
        "ok,vanilla,1,1.00\n"
        "ok,vanilla,2,1.10\n"
        "ok,bio_all,1,0.80\n"
        "error,bio_all,2,nan\n"          # error row skipped
        "ok,bio_all,2,0.85\n"            # finite row kept
        "ok,bio_all,3,notanumber\n",     # unparseable skipped
        encoding="utf-8",
    )
    data = load_matrix_csv(p, "val_bpb")
    assert data["vanilla"] == {1: 1.0, 2: 1.1}
    assert data["bio_all"] == {1: 0.8, 2: 0.85}


def test_load_matrix_csv_missing_metric_raises(tmp_path):
    p = tmp_path / "summary.csv"
    p.write_text("preset,seed,val_bpb\nbio_all,1,0.8\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a column"):
        load_matrix_csv(p, "eval_accuracy")


def test_load_matrix_csv_rejects_conflicting_duplicate_cells(tmp_path):
    p = tmp_path / "summary.csv"
    p.write_text(
        "status,preset,seed,val_bpb\n"
        "ok,bio_all,7,0.8\n"
        "ok,bio_all,7,1.2\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"bio_all.*seed=7.*summary\.csv: lines 2 and 3.*0\.8.*1\.2",
    ):
        load_matrix_csv(p, "val_bpb")


def test_load_matrix_csv_accepts_exact_idempotent_duplicate(tmp_path):
    p = tmp_path / "summary.csv"
    p.write_text(
        "status,preset,seed,val_bpb\n"
        "ok,bio_all,7,0.8\n"
        "ok,bio_all,7,0.800\n",
        encoding="utf-8",
    )

    assert load_matrix_csv(p, "val_bpb") == {"bio_all": {7: 0.8}}


def test_cli_writes_strict_json_and_markdown_reports(tmp_path, monkeypatch):
    csv_path = tmp_path / "summary.csv"
    rows = ["status,preset,seed,recipe_source,data,val_bpb"]
    for seed in range(6):
        rows.extend(
            [
                f"ok,vanilla,{seed},inline_smoke_noncanonical,synthetic,1.0",
                f"ok,bio_all,{seed},inline_smoke_noncanonical,synthetic,0.8",
            ]
        )
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    json_path = tmp_path / "reports" / "stats.json"
    markdown_path = tmp_path / "reports" / "stats.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_stats",
            str(csv_path),
            "--json-out",
            str(json_path),
            "--markdown-out",
            str(markdown_path),
        ],
    )

    assert main() == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["presets"]["bio_all"]["verdict"] == "supported_gain"
    assert payload["presets"]["bio_all"]["paired_vs_baseline"]["t_stat"] is None
    assert payload["canonical_recipe_evidence"] is False
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "`supported_gain`" in markdown
    assert "NONCANONICAL OR UNKNOWN EVIDENCE" in markdown
    assert "sample SD" in markdown


def test_cli_fails_closed_when_requested_baseline_is_missing(tmp_path, monkeypatch):
    csv_path = tmp_path / "summary.csv"
    csv_path.write_text("preset,seed,val_bpb\nbio_all,1,0.8\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["eval_stats", str(csv_path), "--baseline", "vanilla"])
    assert main() == 2


# Handwritten roster and numeric fixtures are independent of the contrast registry.
# These observations check the decision algorithm, and are never experimental rows.
SCREEN_ARMS = (
    "vanilla", "synaptic_off", "bio_all", "bio_no_presyn", "bio_no_stochastic_release",
    "bio_no_doc2", "bio_no_septin_barrier", "add_hebbian", "add_bdnf", "add_bistable_latch",
    "add_stdp", "add_cusp_latch", "add_learnable_kinetics", "add_differentiable_recurrence",
    "add_metriplectic_integrator", "add_neuromod", "control_presyn", "control_presyn_hebbian",
)


def _fixture_screen():
    return {arm: {1337: 4.0, 1338: 8.0} for arm in SCREEN_ARMS}


def test_d1_screen_refuses_scaffold_only_and_one_favorable_seed():
    data = _fixture_screen()
    data["vanilla"] = {1337: 6.0, 1338: 10.0}
    data["add_bdnf"] = {1337: 3.0, 1338: 8.0}
    screen = d1_screen(data)
    assert screen["status"] == "no_go"
    assert screen["survivors"] == []
    assert screen["contrasts"]["scaffold"]["paired_deltas"] == [-2.0, -2.0]
    assert screen["contrasts"]["add_bdnf"]["paired_deltas"] == [-1.0, 0.0]


def test_d1_nominations_use_prerequisites_and_reverse_leave_one_out():
    data = _fixture_screen()
    data["add_hebbian"] = {1337: 5.0, 1338: 10.0}
    for child in ("add_bistable_latch", "add_cusp_latch", "add_stdp"):
        data[child] = data["add_hebbian"].copy()
    data["bio_no_doc2"] = {1337: 4.5, 1338: 8.25}
    screen = d1_screen(data)
    assert screen["status"] == "go"
    assert screen["survivors"] == ["bio_no_doc2", "add_bdnf"]
    assert screen["contrasts"]["add_bdnf"]["paired_deltas"] == [-1.0, -2.0]
    assert screen["contrasts"]["bio_no_doc2"]["paired_deltas"] == [-0.5, -0.25]
    assert screen["contrasts"]["bio_no_presyn"]["role"] == "presynaptic_stack"
    plan = d1_confirmation_plan(screen["survivors"])
    assert plan["family_size"] == 4
    assert plan["seeds"] == list(range(2337, 2345))
    assert plan["minimum_first_holm_p"] == 4 / 128
    assert set(plan["columns"]) == {"vanilla", "bio_all", "synaptic_off", "add_bdnf", "add_hebbian", "bio_no_doc2"}


def test_d1_registered_treatment_really_differs_from_prerequisite_only_control():
    from bio_inspired_nanochat import ablation_matrix as am
    from bio_inspired_nanochat.ablation_registry import MECHANISMS

    arms = {c.config_id: c for c in am.d1_screening_columns()}
    assert set(arms) == set(SCREEN_ARMS)
    by_field = {m.field: m for m in MECHANISMS}

    def active(cfg, field):
        mechanism = by_field[field]
        return (
            not mechanism.moe_only and getattr(cfg, field) != mechanism.off_value
            and all(active(cfg, prereq) for prereq in mechanism.requires)
        )

    for c in D1_MECHANISMS:
        treatment, control = arms[c.treatment].build_syn_cfg(), arms[c.control].build_syn_cfg()
        assert treatment is not None and control is not None
        changed = {field for field in by_field if active(treatment, field) != active(control, field)}
        if c.name == "bio_no_presyn":
            assert "enable_presyn" in changed and len(changed) > 1
        else:
            assert len(changed) == 1, (c, changed)
        assert not treatment.native_presyn and not control.native_presyn


@pytest.mark.parametrize("arm,values", [("add_bdnf", {1337: 3.0}), ("new_arm", {1337: 1.0, 1338: 1.0}),
                                         ("add_bdnf", {1337: float("nan"), 1338: 0.0})])
def test_d1_partial_extra_or_invalid_screen_cannot_select(arm, values):
    data = _fixture_screen()
    data[arm] = values
    assert d1_screen(data)["status"] == "inconclusive"


@pytest.mark.parametrize("delta,verdict", [(-1, "supported_gain"), (0, "null"), (1, "supported_regression")])
def test_d1_confirmation_independent_fixture_and_exact_family(delta, verdict):
    plan = d1_confirmation_plan(["add_bdnf"])
    assert plan["family_size"] == 3
    assert plan["seeds"] == list(range(2337, 2344))
    assert plan["minimum_raw_wilcoxon_p"] == 1 / 64
    assert plan["minimum_first_holm_p"] == 3 / 64
    assert set(plan["columns"]) == {"bio_all", "vanilla", "synaptic_off", "add_hebbian", "add_bdnf"}
    data = {arm: {s: float(i + 4) for i, s in enumerate(plan["seeds"])} for arm in plan["columns"]}
    data["add_bdnf"] = {s: v + delta for s, v in data["add_hebbian"].items()}
    report = d1_confirm(data, ["add_bdnf"])
    assert report["status"] == "evaluated"
    assert set(report["results"]) == {"total_bio", "scaffold", "add_bdnf"}
    result = report["results"]["add_bdnf"]
    assert result["paired_deltas"] == [delta] * 7
    assert result["n_nonzero"] == (7 if delta else 0)
    assert result["wilcoxon_p_value"] == (1 / 64 if delta else 1)
    assert result["wilcoxon_p_adjusted"] == (3 / 64 if delta else 1)
    assert result["delta_ci_low"] == result["delta_ci_high"] == delta
    assert result["verdict"] == verdict
    assert report["results"]["scaffold"]["verdict"] == "null"


def test_d1_ties_can_make_family_gate_unattainable_even_at_planned_sample_size():
    plan = d1_confirmation_plan(["add_bdnf"])
    data = {arm: {s: 2.0 for s in plan["seeds"]} for arm in plan["columns"]}
    data["add_bdnf"] = {s: 1.0 if s != 2337 else 2.0 for s in plan["seeds"]}
    result = d1_confirm(data, ["add_bdnf"])["results"]["add_bdnf"]
    assert result["n_pairs"] == 7 and result["n_nonzero"] == 6
    assert result["minimum_raw_wilcoxon_p"] == 1 / 32
    assert result["minimum_holm_p_given_other_results"] == 3 / 32
    assert result["verdict"] == "inconclusive"
    data["add_bdnf"][1337] = data["add_bdnf"].pop(2337)
    assert d1_confirm(data, ["add_bdnf"])["status"] == "inconclusive"
    with pytest.raises(ValueError, match="distinct"):
        d1_confirmation_plan(["add_bdnf", "add_bdnf"])
    with pytest.raises(ValueError, match="unregistered"):
        d1_confirmation_plan(["add_native_presyn"])


def test_exploratory_structural_effects_have_reference_and_sign():
    scores = {"moe_fixed": 4, "moe_splitmerge": 5, "moe_no_genome": 3,
              "moe_no_metabolism": 6, "moe_add_glial_homeostasis": 4}
    report = structural_descriptive({arm: {42: float(value)} for arm, value in scores.items()})
    assert report["status"] == "exploratory"
    assert {k: v["paired_deltas"] for k, v in report["contrasts"].items()} == {
        "lifecycle": [1], "genome": [1], "metabolism": [-2], "glial_homeostasis": [0],
    }


def test_old_cpu_summary_cannot_enter_registered_scientific_protocol(tmp_path):
    path = tmp_path / "cpu-summary.csv"
    path.write_text("status,preset,seed,recipe_source,data,val_bpb\nok,bio_all,1337,base_train_checkpoint,fineweb,1.0\n")
    with pytest.raises(ValueError, match="missing.*matrix_manifest_digest"):
        d1_csv_report(path, stage="screening")


def _d1_csv_fixture_row():
    # Invented metadata for validating the reader, with no training or performance claim.
    return {
        "preset": "vanilla", "seed": "1337", "val_bpb": "2", "status": "ok",
        "matrix_stage": "screening", "protocol_id": D1_PROTOCOL_VERSION,
        "recipe_source": "base_train_checkpoint", "data": "fineweb", "train_device_type": "cuda",
        "train_world_size": "2", "device_type": "cuda", "world_size": "1",
        "sequence_len": "1024", "n_layer": "10", "n_head": "5", "n_embd": "640",
        "vocab_size": "65536", "use_moe": "False", "eval_bpb": "True",
        "parameter_match_rule": D1_PARAMETER_MATCH_RULE, "matrix_manifest_digest": "a" * 64,
        "matrix_source_sha256": "1" * 64, "model_file_sha256": "f" * 64, "checkpoint_git_sha": "2" * 40,
        "checkpoint_config_hash": "3" * 16, "trained_cell_id": f"matrix_screening_{'a' * 20}_vanilla_s1337",
        "train_data_digest": "d" * 64, "eval_data_digest": "e" * 64, "tokenizer_digest": "b" * 64,
        "optimizer_recipe_digest": "c" * 64, "optimizer_names": "Muon,AdamW",
        "trainable_params": "100000", "active_params": "100000",
        "total_batch_size_tokens": "32768", "device_batch_size": "8",
        "eval_device_batch_size": "8", "eval_tokens_processed": "1048576", "eval_steps": "128",
        "steps": "305", "checkpoint_step": "305", "train_tokens_processed": "9994240",
        "train_tokens_requested": "9994240",
    }


@pytest.mark.parametrize("changes,error", [
    ({"train_device_type": "cpu"}, "train_device_type"),
    ({"train_data_digest": "e" * 64}, "identical train and validation"),
    ({"eval_tokens_processed": "4096"}, "1,000,000"),
    ({"eval_steps": "1"}, "evaluation token count"),
    ({"steps": "200"}, "steps must equal"),
    ({"trained_cell_id": "matrix_vanilla_s1337"}, "trained_cell_id"),
    ({"checkpoint_config_hash": "unknown"}, "checkpoint_config_hash"),
    ({"model_file_sha256": "bad"}, "model_file_sha256"),
    ({"matrix_manifest_digest": "b" * 64}, "trained_cell_id"),
    ({"status": "error"}, "status"),
    ({}, "missing columns"),
])
def test_invalid_or_incomplete_score_receipt_cannot_open_confirmation(tmp_path, changes, error):
    row = {**_d1_csv_fixture_row(), **changes}
    path = tmp_path / "partial.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError, match=error):
        d1_csv_report(path, stage="screening")


def _write_d1_fixture(path, data, *, screen_csv=None, manifest_digest=None):
    from scripts.eval_matrix import SUMMARY_FIELDS, _normalize_row_for_csv

    stage = "confirmation" if screen_csv else "screening"
    run_id = manifest_digest or ("c" if screen_csv else "a") * 64
    rows = []
    for preset, cells in data.items():
        for seed, value in cells.items():
            row = {
                **_d1_csv_fixture_row(), "matrix_stage": stage, "matrix_manifest_digest": run_id,
                "preset": preset, "seed": seed, "val_bpb": value,
                "optimizer_names": ["Muon", "AdamW"],
                "checkpoint_config_hash": hashlib.sha256(preset.encode()).hexdigest()[:16],
                "trained_cell_id": f"matrix_{stage}_{run_id[:20]}_{preset}_s{seed}",
            }
            if screen_csv:
                row.update(screen_summary_sha256=hashlib.sha256(screen_csv.read_bytes()).hexdigest(),
                           steps=3051, checkpoint_step=3051, train_tokens_processed=99975168,
                           train_tokens_requested=99975168)
            rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(_normalize_row_for_csv(row) for row in rows)
    path.with_suffix(".jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_complete_csv_and_mirror_select_fixture_then_detect_bpb_or_provenance_edit(tmp_path):
    data = _fixture_screen()
    data["add_bdnf"] = {1337: 3.5, 1338: 7.0}
    screen_csv = tmp_path / "screen.csv"
    _write_d1_fixture(screen_csv, data)
    selection = d1_csv_report(screen_csv, stage="screening")
    assert selection["status"] == "go" and selection["survivors"] == ["add_bdnf"]
    names = ("bio_all", "vanilla", "synaptic_off", "add_hebbian", "add_bdnf")
    heldout = {name: {s: 2.0 for s in range(2337, 2344)} for name in names}
    heldout["add_bdnf"] = {s: 1.75 for s in range(2337, 2344)}
    confirmation = tmp_path / "confirm.csv"
    _write_d1_fixture(confirmation, heldout, screen_csv=screen_csv)
    report = d1_csv_report(confirmation, stage="confirmation", screen_csv=screen_csv)
    assert report["results"]["add_bdnf"]["wilcoxon_p_adjusted"] == 3 / 64

    # A disagreement in either direction invalidates the registered record. The
    # altered numeric value is finite and would still pass a range/shape check.
    with screen_csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["val_bpb"] = "4.1"
    with screen_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="CSV disagrees on val_bpb"):
        d1_csv_report(screen_csv, stage="screening")

    _write_d1_fixture(screen_csv, data)
    mirror = screen_csv.with_suffix(".jsonl")
    records = [json.loads(line) for line in mirror.read_text().splitlines()]
    records[0]["matrix_source_sha256"] = "e" * 64
    mirror.write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(ValueError, match="CSV disagrees on matrix_source_sha256"):
        d1_csv_report(screen_csv, stage="screening")

    # Even if both scorer files change in agreement, confirmation still binds
    # the exact selection digest from the admitted campaign.
    data["bio_all"][1337] += 0.125
    _write_d1_fixture(screen_csv, data)
    with pytest.raises(ValueError, match="differs from the selection admitted"):
        d1_csv_report(confirmation, stage="confirmation", screen_csv=screen_csv)
