"""Tests for the statistical testing layer (bead 74f.3).

The distribution functions are validated against KNOWN reference values (standard
t-tables / textbook Wilcoxon results) so the pure-numpy implementations are provably
correct, then the aggregation/comparison logic is checked on synthetic matrices.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bio_inspired_nanochat.eval_stats import (
    aggregate,
    bootstrap_ci,
    compare_matrix,
    load_matrix_csv,
    paired_comparison,
    paired_t_test,
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


# --------------------------------------------------------------------------- #
# paired_comparison + compare_matrix
# --------------------------------------------------------------------------- #
def test_paired_comparison_needs_two_shared_seeds():
    assert paired_comparison({1: 1.0}, {1: 2.0}, lower_is_better=True) is None
    res = paired_comparison({1: 1.0, 2: 1.0}, {1: 2.0, 2: 2.0}, lower_is_better=True)
    assert res is not None and res.n_pairs == 2


def test_compare_matrix_detects_consistent_improvement():
    # bio_all has lower val_bpb than vanilla on every matched seed -> better + favorable.
    data = {
        "vanilla": {1: 1.00, 2: 1.10, 3: 0.95, 4: 1.05},
        "bio_all": {1: 0.80, 2: 0.88, 3: 0.78, 4: 0.84},
    }
    rep = compare_matrix(data, baseline="vanilla", metric="val_bpb")
    assert rep["lower_is_better"] is True
    bio = rep["presets"]["bio_all"]
    assert bio["better"] is True
    assert bio["significant"] is True
    assert bio["paired_vs_baseline"]["n_favorable"] == 4
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


def test_compare_matrix_unknown_baseline_raises():
    with pytest.raises(ValueError, match="baseline"):
        compare_matrix({"a": {1: 1.0}}, baseline="nope", metric="val_bpb")


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
