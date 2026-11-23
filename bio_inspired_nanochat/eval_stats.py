"""Statistical testing layer for bio-vs-vanilla comparisons (bead 74f.3).

Before this module there was ZERO statistical testing in the codebase — headline
"bio beats vanilla" claims shipped without significance or uncertainty. This adds:

* multi-seed aggregation with a Student-t 95% confidence interval,
* paired significance vs a baseline on MATCHED seeds (paired t-test + Wilcoxon
  signed-rank), plus a paired bootstrap CI on the delta and Cohen's d_z effect size,
* a direction-aware matrix comparison (`compare_matrix`) and a CLI that reads an
  ``eval_matrix`` ``summary.csv`` and prints per-preset mean ± CI and significance.

Pure-numpy (no SciPy dependency): the Student-t CDF/quantile use the regularized
incomplete beta function; Wilcoxon is exact for small n and normal-approximated
(with tie + continuity correction) for large n. Validated against known reference
values in ``tests/test_eval_stats.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from rich.console import Console

from bio_inspired_nanochat.metrics_schema import Direction, get_metric

# --------------------------------------------------------------------------- #
# Student-t distribution via the regularized incomplete beta function
# --------------------------------------------------------------------------- #
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Numerical Recipes)."""
    maxit, eps, fpmin = 300, 3.0e-16, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) in [0, 1]."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf_two_sided(t: float, df: float) -> float:
    """Two-sided tail P(|T| > |t|) for Student-t with ``df`` degrees of freedom."""
    if df <= 0:
        return float("nan")
    if t == 0.0:
        return 1.0
    return _betai(df / 2.0, 0.5, df / (df + t * t))


def t_cdf(t: float, df: float) -> float:
    """CDF P(T <= t) for Student-t."""
    tail = t_sf_two_sided(t, df) / 2.0  # P(T > |t|)
    return 1.0 - tail if t >= 0 else tail


def t_ppf(p: float, df: float) -> float:
    """Inverse CDF (quantile) for Student-t via bisection on :func:`t_cdf`."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0,1), got {p}")
    if p == 0.5:
        return 0.0
    lo, hi = -1.0e4, 1.0e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# Aggregation, paired tests, bootstrap
# --------------------------------------------------------------------------- #
@dataclass
class Aggregate:
    n: int
    mean: float
    std: float       # sample std (ddof=1)
    sem: float       # standard error of the mean
    ci_low: float    # Student-t 95% CI
    ci_high: float


def _finite_observation_array(
    values: list[float] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Return a one-dimensional float array, rejecting invalid evidence."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence")
    if not np.isfinite(array).all():
        bad_positions = np.flatnonzero(~np.isfinite(array)).tolist()
        raise ValueError(
            f"{name} contains non-finite observation(s) at positions {bad_positions}"
        )
    return array


def aggregate(values: list[float] | np.ndarray, confidence: float = 0.95) -> Aggregate:
    """Mean and Student-t confidence interval over seeds."""
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 < float(confidence) < 1.0
    ):
        raise ValueError(f"confidence must be finite and in (0, 1), got {confidence!r}")
    a = _finite_observation_array(values, name="aggregate values")
    n = int(a.size)
    if n == 0:
        raise ValueError("aggregate() needs at least one value")
    mean = float(a.mean())
    if n == 1:
        return Aggregate(1, mean, 0.0, 0.0, mean, mean)
    std = float(a.std(ddof=1))
    sem = std / math.sqrt(n)
    crit = t_ppf(0.5 + confidence / 2.0, n - 1)
    return Aggregate(n, mean, std, sem, mean - crit * sem, mean + crit * sem)


@dataclass
class PairedResult:
    n_pairs: int
    mean_delta: float       # mean(treatment - baseline) over matched seeds
    delta_ci_low: float     # paired bootstrap CI of the mean delta
    delta_ci_high: float
    t_stat: float
    t_p_value: float        # paired t-test, two-sided
    wilcoxon_p_value: float  # Wilcoxon signed-rank, two-sided
    cohen_dz: float         # paired effect size mean(d)/std(d)
    n_favorable: int        # pairs where treatment beat baseline (direction-aware)


def paired_t_test(deltas: np.ndarray) -> tuple[float, float]:
    """Two-sided paired t-test on per-pair differences. Returns (t_stat, p_value)."""
    deltas = _finite_observation_array(deltas, name="paired deltas")
    n = deltas.size
    if n < 2:
        return float("nan"), float("nan")
    sd = deltas.std(ddof=1)
    if sd == 0.0:
        # No variance: either an exact tie (t=0, p=1) or a constant non-zero shift.
        mean = float(deltas.mean())
        return (0.0, 1.0) if mean == 0.0 else (math.copysign(math.inf, mean), 0.0)
    t = float(deltas.mean() / (sd / math.sqrt(n)))
    return t, float(t_sf_two_sided(t, n - 1))


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties shared — like scipy.stats.rankdata('average')."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=np.float64)
    sa = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0  # mean of 1-based ranks i+1..j+1
        i = j + 1
    return ranks


def wilcoxon_signed_rank(deltas: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value on paired differences.

    Zero differences are dropped. Exact (enumeration) for n <= 18; otherwise a
    normal approximation with tie and continuity correction.
    """
    deltas = _finite_observation_array(deltas, name="Wilcoxon deltas")
    d = deltas[deltas != 0.0]
    n = d.size
    if n == 0:
        return 1.0
    ranks = _rankdata(np.abs(d))
    w_plus = float(ranks[d > 0].sum())
    total = n * (n + 1) / 2.0

    if n <= 18:
        # Exact null: each rank is +/- with equal probability. Enumerate the
        # distribution of W+ (the sum of ranks assigned the positive sign).
        sums: dict[float, int] = {0.0: 1}
        for r in ranks:
            nxt: dict[float, int] = {}
            for s, cnt in sums.items():
                nxt[s] = nxt.get(s, 0) + cnt          # this rank negative
                nxt[s + r] = nxt.get(s + r, 0) + cnt  # this rank positive
            sums = nxt
        denom = float(2**n)
        le = sum(c for s, c in sums.items() if s <= w_plus) / denom
        ge = sum(c for s, c in sums.items() if s >= w_plus) / denom
        return float(min(1.0, 2.0 * min(le, ge)))

    mean = total / 2.0
    # Variance with tie correction.
    _, counts = np.unique(np.abs(d), return_counts=True)
    tie_term = float((counts**3 - counts).sum())
    var = (n * (n + 1) * (2 * n + 1) - tie_term / 2.0) / 24.0
    if var <= 0:
        return 1.0
    z = (w_plus - mean - math.copysign(0.5, w_plus - mean)) / math.sqrt(var)
    return float(math.erfc(abs(z) / math.sqrt(2.0)))  # two-sided normal tail


def bootstrap_ci(
    values: np.ndarray,
    *,
    n_boot: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean (deterministic given ``seed``)."""
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot <= 0:
        raise ValueError(f"n_boot must be a positive integer, got {n_boot!r}")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 < float(confidence) < 1.0
    ):
        raise ValueError(f"confidence must be finite and in (0, 1), got {confidence!r}")
    a = _finite_observation_array(values, name="bootstrap values")
    if a.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    means = a[idx].mean(axis=1)
    lo = (1.0 - confidence) / 2.0
    return float(np.quantile(means, lo)), float(np.quantile(means, 1.0 - lo))


def paired_comparison(
    treatment: dict[int, float],
    baseline: dict[int, float],
    *,
    lower_is_better: bool,
    n_boot: int = 10000,
    seed: int = 0,
) -> Optional[PairedResult]:
    """Compare a treatment to a baseline on the seeds they share.

    Returns ``None`` if fewer than 2 seeds are shared (no paired test possible).
    """
    if not isinstance(lower_is_better, bool):
        raise TypeError("lower_is_better must be a bool")
    seeds = sorted(set(treatment) & set(baseline))
    if len(seeds) < 2:
        return None
    t_vals = np.array([treatment[s] for s in seeds], dtype=np.float64)
    b_vals = np.array([baseline[s] for s in seeds], dtype=np.float64)
    deltas = t_vals - b_vals
    t_stat, t_p = paired_t_test(deltas)
    w_p = wilcoxon_signed_rank(deltas)
    ci_low, ci_high = bootstrap_ci(deltas, n_boot=n_boot, seed=seed)
    sd = deltas.std(ddof=1)
    mean_d = float(deltas.mean())
    # Zero-variance deltas: match paired_t_test's convention (infinite effect in
    # the direction of the constant shift) instead of the misleading 0.0 that
    # read as "no effect" next to p=0.
    dz = (
        float(mean_d / sd)
        if sd > 0
        else (float("inf") if mean_d > 0 else float("-inf") if mean_d < 0 else 0.0)
    )
    favorable = int((deltas < 0).sum() if lower_is_better else (deltas > 0).sum())
    return PairedResult(
        n_pairs=len(seeds),
        mean_delta=float(deltas.mean()),
        delta_ci_low=ci_low,
        delta_ci_high=ci_high,
        t_stat=t_stat,
        t_p_value=t_p,
        wilcoxon_p_value=w_p,
        cohen_dz=dz,
        n_favorable=favorable,
    )


def _direction_lower_better(metric: str, lower_is_better: Optional[bool]) -> bool:
    if lower_is_better is not None:
        if not isinstance(lower_is_better, bool):
            raise TypeError("lower_is_better must be a bool or None")
        return lower_is_better
    spec = get_metric(metric)
    if spec is None:
        # Fail CLOSED: guessing a direction silently INVERTS the favorable/adverse
        # logic for any higher-is-better metric that misses the substring heuristic
        # (e.g. "working_memory" accuracy), promoting regressions to gains. The
        # caller must register the metric or pass ``lower_is_better`` explicitly.
        raise ValueError(
            f"Unknown metric {metric!r}: no schema entry and no explicit "
            f"lower_is_better override. Register it in metrics_schema or pass "
            f"lower_is_better explicitly."
        )
    if spec.direction == Direction.NEUTRAL:
        raise ValueError(
            f"Metric {metric!r} has neutral direction: comparison requires an explicit "
            "lower_is_better override grounded in the experiment's hypothesis."
        )
    return spec.direction == Direction.LOWER_BETTER


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-adjust a named family of p-values while preserving its keys.

    Non-finite values are retained as ``nan`` and excluded from the correction family. The
    step-down running maximum is required for monotonic adjusted p-values.
    """
    invalid = {
        name: value
        for name, value in p_values.items()
        if math.isfinite(value) and not 0.0 <= value <= 1.0
    }
    if invalid:
        raise ValueError(f"p-values must be in [0, 1], got {invalid}")
    adjusted = {name: float("nan") for name in p_values}
    finite = sorted(
        ((name, value) for name, value in p_values.items() if math.isfinite(value)),
        key=lambda item: item[1],
    )
    family_size = len(finite)
    running_max = 0.0
    for rank, (name, value) in enumerate(finite):
        running_max = max(running_max, (family_size - rank) * value)
        adjusted[name] = min(1.0, running_max)
    return adjusted


def compare_matrix(
    data: dict[str, dict[int, float]],
    *,
    baseline: str,
    metric: str = "val_bpb",
    lower_is_better: Optional[bool] = None,
    alpha: float = 0.05,
    min_pairs: int = 3,
    seed: int = 0,
) -> dict:
    """Aggregate every preset and test each against ``baseline`` on matched seeds.

    ``data`` maps preset -> {seed: metric_value}. Returns a structured report with a
    per-preset aggregate and (for non-baseline presets) a paired comparison whose verdict is
    direction-aware, Holm-corrected across presets, and gated by both paired tests plus the
    paired-bootstrap interval. Fewer than ``min_pairs`` matched seeds always yields
    ``insufficient_evidence``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if min_pairs < 2:
        raise ValueError(f"min_pairs must be at least 2, got {min_pairs}")
    if baseline not in data:
        raise ValueError(f"baseline preset {baseline!r} not in data ({sorted(data)})")
    lower = _direction_lower_better(metric, lower_is_better)
    report: dict = {
        "metric": metric,
        "lower_is_better": lower,
        "baseline": baseline,
        "alpha": alpha,
        "min_pairs": min_pairs,
        "decision_rule": {
            "multiple_comparison_correction": "holm",
            "supported_gain": (
                "at least min_pairs matched seeds; mean delta and paired-bootstrap 95% CI "
                "favorable; Holm-adjusted paired-t and Wilcoxon p-values both <= alpha"
            ),
            "supported_regression": (
                "at least min_pairs matched seeds; mean delta and paired-bootstrap 95% CI "
                "adverse; Holm-adjusted paired-t and Wilcoxon p-values both <= alpha"
            ),
            "null": "sufficient matched seeds, but neither directional support rule passed",
            "insufficient_evidence": "fewer than min_pairs matched seeds",
        },
        "presets": {},
    }
    paired_by_preset: dict[str, PairedResult] = {}
    for preset, by_seed in data.items():
        agg = aggregate(list(by_seed.values()))
        entry: dict = {"aggregate": asdict(agg)}
        if preset not in (baseline,):  # every preset except the baseline gets a paired test
            paired = paired_comparison(
                by_seed, data[baseline], lower_is_better=lower, seed=seed
            )
            if paired is not None:
                entry["paired_vs_baseline"] = asdict(paired)
                paired_by_preset[preset] = paired
            else:
                matched = len(set(by_seed) & set(data[baseline]))
                entry.update(
                    {
                        "matched_pairs": matched,
                        "better": None,
                        "evidence_sufficient": False,
                        "ci_favorable": False,
                        "ci_adverse": False,
                        "paired_t_p_adjusted": None,
                        "wilcoxon_p_adjusted": None,
                        "tests_pass": False,
                        "significant": False,
                        "supported_gain": False,
                        "supported_regression": False,
                        "verdict": "insufficient_evidence",
                    }
                )
        report["presets"][preset] = entry

    adjusted_t = holm_adjust(
        {preset: paired.t_p_value for preset, paired in paired_by_preset.items()}
    )
    adjusted_wilcoxon = holm_adjust(
        {preset: paired.wilcoxon_p_value for preset, paired in paired_by_preset.items()}
    )
    for preset, paired in paired_by_preset.items():
        entry = report["presets"][preset]
        improvement = -paired.mean_delta if lower else paired.mean_delta
        ci_favorable = (
            paired.delta_ci_high < 0.0 if lower else paired.delta_ci_low > 0.0
        )
        ci_adverse = (
            paired.delta_ci_low > 0.0 if lower else paired.delta_ci_high < 0.0
        )
        enough_pairs = paired.n_pairs >= min_pairs
        tests_pass = (
            adjusted_t[preset] <= alpha and adjusted_wilcoxon[preset] <= alpha
        )
        supported_gain = enough_pairs and improvement > 0.0 and ci_favorable and tests_pass
        supported_regression = (
            enough_pairs and improvement < 0.0 and ci_adverse and tests_pass
        )
        if not enough_pairs:
            verdict = "insufficient_evidence"
        elif supported_gain:
            verdict = "supported_gain"
        elif supported_regression:
            verdict = "supported_regression"
        else:
            verdict = "null"
        entry.update(
            {
                "better": improvement > 0.0,
                "evidence_sufficient": enough_pairs,
                "ci_favorable": ci_favorable,
                "ci_adverse": ci_adverse,
                "paired_t_p_adjusted": adjusted_t[preset],
                "wilcoxon_p_adjusted": adjusted_wilcoxon[preset],
                "tests_pass": tests_pass,
                "significant": supported_gain or supported_regression,
                "supported_gain": supported_gain,
                "supported_regression": supported_regression,
                "verdict": verdict,
            }
        )
    return report


# --------------------------------------------------------------------------- #
# Prospective D1 comparisons and decisions
# --------------------------------------------------------------------------- #
D1_PROTOCOL_VERSION = "d1-paired-bpb-2026-11-22"
D1_SCREEN_SEEDS = (1337, 1338)
D1_ALPHA = 0.05
D1_PARAMETER_MATCH_RULE = "report_exact_trainable_and_active_counts"


@dataclass(frozen=True)
class D1Contrast:
    """Treatment minus control, so negative BPB always favors the intervention."""

    name: str
    treatment: str
    control: str
    role: str


D1_ANCHORS = (
    D1Contrast("total_bio", "bio_all", "vanilla", "primary"),
    D1Contrast("scaffold", "synaptic_off", "vanilla", "architecture"),
)
# Fixed before D1. The name of a leave-one-out hypothesis is its ablated column
# for compatibility with matrix_launch; *full* is the treated model.
D1_MECHANISMS = (
    D1Contrast("bio_no_presyn", "bio_all", "bio_no_presyn", "presynaptic_stack"),
    D1Contrast("bio_no_stochastic_release", "bio_all", "bio_no_stochastic_release", "mechanism"),
    D1Contrast("bio_no_doc2", "bio_all", "bio_no_doc2", "mechanism"),
    D1Contrast("bio_no_septin_barrier", "bio_all", "bio_no_septin_barrier", "mechanism"),
    D1Contrast("add_hebbian", "add_hebbian", "synaptic_off", "mechanism"),
    D1Contrast("add_bdnf", "add_bdnf", "add_hebbian", "mechanism"),
    D1Contrast("add_bistable_latch", "add_bistable_latch", "add_hebbian", "mechanism"),
    D1Contrast("add_stdp", "add_stdp", "add_hebbian", "mechanism"),
    D1Contrast("add_cusp_latch", "add_cusp_latch", "add_bistable_latch", "mechanism"),
    D1Contrast("add_learnable_kinetics", "add_learnable_kinetics", "control_presyn", "mechanism"),
    D1Contrast("add_differentiable_recurrence", "add_differentiable_recurrence", "add_learnable_kinetics", "mechanism"),
    D1Contrast("add_metriplectic_integrator", "add_metriplectic_integrator", "control_presyn", "mechanism"),
    D1Contrast("add_neuromod", "add_neuromod", "control_presyn_hebbian", "mechanism"),
)
D1_STRUCTURAL = (
    D1Contrast("lifecycle", "moe_splitmerge", "moe_fixed", "exploratory"),
    D1Contrast("metabolism", "moe_fixed", "moe_no_metabolism", "exploratory"),
    D1Contrast("genome", "moe_fixed", "moe_no_genome", "exploratory"),
    D1Contrast("glial_homeostasis", "moe_add_glial_homeostasis", "moe_fixed", "exploratory"),
)


def _contrast_columns(contrasts: tuple[D1Contrast, ...]) -> list[str]:
    return list(dict.fromkeys(arm for contrast in contrasts for arm in (contrast.control, contrast.treatment)))


def _d1_cell_errors(
    data: dict[str, dict[int, float]], columns: list[str], seeds: tuple[int, ...]
) -> list[str]:
    errors = []
    if set(data) != set(columns):
        errors.append(f"missing columns {sorted(set(columns) - set(data))}, extra columns {sorted(set(data) - set(columns))}")
    for column in columns:
        cells = data.get(column, {})
        if set(cells) != set(seeds):
            errors.append(f"{column} expected seeds {list(seeds)}, received {sorted(cells)}")
        if any(not math.isfinite(value) or value < 0 for value in cells.values()):
            errors.append(f"{column} has invalid validation BPB")
    return errors


def d1_screen(data: dict[str, dict[int, float]]) -> dict:
    """Select candidates descriptively: strictly lower BPB on both registered seeds."""
    contrasts = D1_ANCHORS + D1_MECHANISMS
    errors = _d1_cell_errors(data, _contrast_columns(contrasts), D1_SCREEN_SEEDS)
    report: dict = {
        "protocol_version": D1_PROTOCOL_VERSION, "stage": "screening", "status": "inconclusive",
        "metric": "val_bpb", "lower_is_better": True,
        "seeds": list(D1_SCREEN_SEEDS), "survivors": [], "contrasts": {}, "errors": errors,
    }
    if errors:
        return report
    for contrast in contrasts:
        deltas = [data[contrast.treatment][s] - data[contrast.control][s] for s in D1_SCREEN_SEEDS]
        advances = contrast in D1_MECHANISMS and all(delta < 0 for delta in deltas)
        if advances:
            report["survivors"].append(contrast.name)
        report["contrasts"][contrast.name] = {**asdict(contrast), "paired_deltas": deltas, "advances": advances}
    report["status"] = "go" if report["survivors"] else "no_go"
    return report


def structural_descriptive(data: dict[str, dict[int, float]]) -> dict:
    """Exploratory MoE contrasts. No dense nominations or confirmatory tests."""
    seeds = tuple(sorted(data.get("moe_fixed", {})))
    errors = _d1_cell_errors(data, _contrast_columns(D1_STRUCTURAL), seeds)
    if not seeds:
        errors.append("no fixed-MoE seed observations")
    report: dict = {"stage": "structural", "seeds": list(seeds), "status": "inconclusive",
                    "errors": errors, "contrasts": {}}
    if errors:
        return report
    for c in D1_STRUCTURAL:
        deltas = [data[c.treatment][s] - data[c.control][s] for s in seeds]
        report["contrasts"][c.name] = {**asdict(c), "paired_deltas": deltas}
    report["status"] = "exploratory"
    return report


def _d1_confirmation_contrasts(survivors: list[str] | tuple[str, ...]) -> tuple[D1Contrast, ...]:
    selected = set(survivors)
    if not selected or len(selected) != len(survivors):
        raise ValueError("D1 confirmation requires one or more distinct selected mechanisms")
    unknown = selected - {c.name for c in D1_MECHANISMS}
    if unknown:
        raise ValueError(f"unregistered D1 mechanism(s): {sorted(unknown)}")
    return D1_ANCHORS + tuple(c for c in D1_MECHANISMS if c.name in selected)


def d1_confirmation_plan(survivors: list[str] | tuple[str, ...]) -> dict:
    """Keep every selected contrast's control and size one prespecified Holm family."""
    contrasts = _d1_confirmation_contrasts(survivors)
    family_size = len(contrasts)
    min_n = math.ceil(math.log2(2 * family_size / D1_ALPHA))
    return {
        "protocol_version": D1_PROTOCOL_VERSION, "stage": "confirmation",
        "columns": _contrast_columns(contrasts), "seeds": list(range(2337, 2337 + min_n)),
        "family_size": family_size, "alpha": D1_ALPHA,
        "minimum_raw_wilcoxon_p": 2 ** (1 - min_n),
        "minimum_first_holm_p": min(1.0, family_size * 2 ** (1 - min_n)),
        "contrasts": [asdict(c) for c in contrasts],
    }


def d1_confirm(data: dict[str, dict[int, float]], survivors: list[str] | tuple[str, ...]) -> dict:
    """Evaluate only the registered family on complete, independent confirmation pairs."""
    plan = d1_confirmation_plan(survivors)
    seeds = tuple(plan["seeds"])
    errors = _d1_cell_errors(data, plan["columns"], seeds)
    report: dict = {**plan, "status": "inconclusive", "errors": errors, "results": {},
                    "metric": "val_bpb", "lower_is_better": True}
    if errors:
        return report
    contrasts = _d1_confirmation_contrasts(survivors)
    results = {}
    for c in contrasts:
        result = paired_comparison(data[c.treatment], data[c.control], lower_is_better=True)
        if result is None:
            raise RuntimeError("complete registered grid yielded fewer than two paired observations")
        results[c.name] = result
    adjusted_t = holm_adjust({name: r.t_p_value for name, r in results.items()})
    raw_w = {name: r.wilcoxon_p_value for name, r in results.items()}
    adjusted_w = holm_adjust(raw_w)
    for c in contrasts:
        r = results[c.name]
        deltas = [data[c.treatment][s] - data[c.control][s] for s in seeds]
        nonzero = sum(d != 0.0 for d in deltas)
        floor = min(1.0, 2.0 ** (1 - nonzero))
        family_floor = holm_adjust({**raw_w, c.name: floor})[c.name]
        passes = max(adjusted_t[c.name], adjusted_w[c.name]) <= D1_ALPHA
        if passes and r.mean_delta < 0 and r.delta_ci_high < 0:
            verdict = "supported_gain"
        elif passes and r.mean_delta > 0 and r.delta_ci_low > 0:
            verdict = "supported_regression"
        elif nonzero > 0 and family_floor > D1_ALPHA:
            verdict = "inconclusive"
        else:
            verdict = "null"
        report["results"][c.name] = {
            **asdict(r), "paired_deltas": deltas, "n_nonzero": nonzero,
            "minimum_raw_wilcoxon_p": floor, "minimum_holm_p_given_other_results": family_floor,
            "paired_t_p_adjusted": adjusted_t[c.name],
            "wilcoxon_p_adjusted": adjusted_w[c.name], "verdict": verdict,
        }
    report["status"] = "evaluated"
    return report


@dataclass
class _D1Evidence:
    data: dict[str, dict[int, float]]
    common: dict[str, str]
    arms: dict[str, dict[str, str]]
    manifest: str
    screen_summary_sha256: str


def _verify_score_mirror(csv_path: Path, rows: dict[tuple[str, int], dict[str, str]]) -> None:
    """Detect a partial publish or edited CSV against the scorer's existing JSONL records."""
    mirror = csv_path.with_suffix(".jsonl")
    if not mirror.is_file():
        raise ValueError(f"registered D1 analysis requires the scorer's paired summary {mirror}")
    seen = set()
    with mirror.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{mirror} line {number}: expected one score record")
            key = (record.get("preset"), record.get("seed"))
            if key not in rows or key in seen:
                raise ValueError(f"{mirror} line {number}: extra or duplicate score record {key}")
            for field, value in rows[key].items():
                actual = record.get(field)
                if isinstance(actual, (list, dict)):
                    actual = json.dumps(actual, sort_keys=True, separators=(",", ":"))
                if str(actual) != value:
                    raise ValueError(f"{mirror} line {number}: CSV disagrees on {field} for {key}")
            seen.add(key)
    if seen != set(rows):
        raise ValueError(f"{mirror}: missing recorded scorer cells")


def _read_d1_evidence(path: Path, stage: str, survivors: list[str] | tuple[str, ...]) -> _D1Evidence:
    """Validate scored rows as one complete campaign before computing any contrasts."""
    from bio_inspired_nanochat.matrix_manifest import registered_tokens

    if stage == "screening":
        columns = _contrast_columns(D1_ANCHORS + D1_MECHANISMS)
        seeds = D1_SCREEN_SEEDS
    elif stage == "confirmation":
        plan = d1_confirmation_plan(survivors)
        columns, seeds = plan["columns"], tuple(plan["seeds"])
    else:
        raise ValueError(f"unregistered D1 stage {stage!r}")

    fixed = {
        "status": "ok", "matrix_stage": stage, "protocol_id": D1_PROTOCOL_VERSION,
        "recipe_source": "base_train_checkpoint", "data": "fineweb",
        "train_device_type": "cuda", "train_world_size": "2", "device_type": "cuda",
        "sequence_len": "1024", "n_layer": "10", "n_head": "5", "n_embd": "640",
        "parameter_match_rule": D1_PARAMETER_MATCH_RULE,
    }
    digest_fields = ("matrix_manifest_digest", "matrix_source_sha256", "model_file_sha256", "train_data_digest",
                     "eval_data_digest", "tokenizer_digest", "optimizer_recipe_digest")
    common_fields = ("matrix_source_sha256", "checkpoint_git_sha", "train_data_digest",
                     "eval_data_digest", "tokenizer_digest", "train_device_type", "train_world_size",
                     "device_type", "world_size", "eval_device_batch_size", "eval_tokens_processed", "total_batch_size_tokens",
                     "device_batch_size", "sequence_len", "n_layer", "n_head", "n_embd", "vocab_size")
    arm_fields = ("checkpoint_config_hash", "optimizer_recipe_digest", "optimizer_names",
                  "trainable_params", "active_params")
    required = (set(fixed) | set(digest_fields) | set(common_fields) | set(arm_fields)
                | {"trained_cell_id", "preset", "seed", "val_bpb", "eval_bpb", "use_moe",
                   "train_tokens_processed", "train_tokens_requested", "steps", "checkpoint_step", "eval_steps"})
    if stage == "confirmation":
        required.add("screen_summary_sha256")
    evidence = _D1Evidence({}, {}, {}, "", "")
    rows: dict[tuple[str, int], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        missing = required.difference(header)
        if missing or len(header) != len(set(header)):
            raise ValueError(f"D1 CSV has missing or duplicate headers: missing {sorted(missing)}")
        for row in reader:
            try:
                if None in row or any(not row[k] for k in required):
                    raise ValueError("missing or unlabelled cell field")
                for key, expected in fixed.items():
                    if row[key] != expected:
                        raise ValueError(f"{key} must equal {expected!r}")
                lengths = {k: 64 for k in digest_fields}
                lengths.update(checkpoint_config_hash=16, checkpoint_git_sha=40)
                if stage == "confirmation":
                    lengths["screen_summary_sha256"] = 64
                for key, length in lengths.items():
                    if not re.fullmatch(rf"[0-9a-f]{{{length}}}", row[key]):
                        raise ValueError(f"invalid {key}")
                for key in ("world_size", "n_head", "vocab_size", "device_batch_size",
                            "trainable_params", "active_params"):
                    if int(row[key]) <= 0:
                        raise ValueError(f"{key} must be positive")
                if row["eval_bpb"].lower() not in ("1", "true") or row["use_moe"].lower() not in ("0", "false"):
                    raise ValueError("D1 requires held-out BPB on dense models")
                if row["train_data_digest"] == row["eval_data_digest"]:
                    raise ValueError("identical train and validation shard digests")
                if int(row["eval_tokens_processed"]) < 1_000_000:
                    raise ValueError("D1 requires at least 1,000,000 held-out evaluation tokens")
                eval_micro = int(row["world_size"]) * int(row["sequence_len"]) * int(row["eval_device_batch_size"])
                if eval_micro <= 0 or int(row["eval_steps"]) * eval_micro != int(row["eval_tokens_processed"]):
                    raise ValueError("held-out evaluation token count disagrees with batch geometry")
                batch = int(row["total_batch_size_tokens"])
                if batch % (int(row["device_batch_size"]) * 1024 * 2):
                    raise ValueError("global training batch disagrees with per-rank batch geometry")
                tokens = registered_tokens(stage, batch)
                for key, expected_count in (("train_tokens_processed", tokens), ("train_tokens_requested", tokens),
                                            ("steps", tokens // batch), ("checkpoint_step", tokens // batch)):
                    if int(row[key]) != expected_count:
                        raise ValueError(f"{key} must equal {expected_count}")
                preset, seed = row["preset"], int(row["seed"])
                manifest = row["matrix_manifest_digest"]
                if row["trained_cell_id"] != f"matrix_{stage}_{manifest[:20]}_{preset}_s{seed}":
                    raise ValueError("trained_cell_id disagrees with manifest, preset or seed")
                common = {k: row[k] for k in common_fields}
                arm = {k: row[k] for k in arm_fields}
                if evidence.common and common != evidence.common:
                    raise ValueError("inconsistent source, shard, tokenizer, batch, evaluation or model dimensions")
                if evidence.manifest and manifest != evidence.manifest:
                    raise ValueError("mixed stage manifests")
                source_screen = row.get("screen_summary_sha256", "") or ""
                if evidence.screen_summary_sha256 and source_screen != evidence.screen_summary_sha256:
                    raise ValueError("mixed screening selections")
                if preset in evidence.arms and arm != evidence.arms[preset]:
                    raise ValueError(f"configuration, optimizer or capacity differs between seeds of {preset}")
                cells = evidence.data.setdefault(preset, {})
                if seed in cells:
                    raise ValueError(f"duplicate cell {preset} seed {seed}")
                cells[seed] = float(row["val_bpb"])
                rows[preset, seed] = {field: row[field] for field in required}
                evidence.common, evidence.manifest = common, manifest
                evidence.arms[preset], evidence.screen_summary_sha256 = arm, source_screen
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path} line {reader.line_num}: {exc}") from exc
    errors = _d1_cell_errors(evidence.data, columns, seeds)
    if errors:
        raise ValueError("; ".join(errors))
    if len({arm["checkpoint_config_hash"] for arm in evidence.arms.values()}) != len(evidence.arms):
        raise ValueError("distinct D1 columns report identical checkpoint configurations")
    _verify_score_mirror(path, rows)
    return evidence


def load_d1_csv(
    path: Path, *, stage: str, survivors: list[str] | tuple[str, ...] = ()
) -> dict[str, dict[int, float]]:
    """Load a complete registered grid for the launcher admission or statistical analysis."""
    return _read_d1_evidence(path, stage, survivors).data


def _check_d1_continuity(screen: _D1Evidence, confirmation: _D1Evidence, screen_digest: str) -> None:
    if confirmation.screen_summary_sha256 != screen_digest:
        raise ValueError("supplied screening CSV differs from the selection admitted to confirmation")
    if confirmation.common != screen.common or any(
        arm != screen.arms[preset] for preset, arm in confirmation.arms.items()
    ):
        raise ValueError("source, data, evaluation or per-arm recipe changed after screening")


def d1_csv_report(path: Path, *, stage: str, screen_csv: Path | None = None) -> dict:
    """Make a D1 decision from the scorer's completed checkpoint rows."""
    from bio_inspired_nanochat.matrix_manifest import file_digest, registered_tokens

    if stage == "screening" and screen_csv is None:
        evidence = _read_d1_evidence(path, stage, ())
        report = d1_screen(evidence.data)
        contrasts = D1_ANCHORS + D1_MECHANISMS
    elif stage == "confirmation" and screen_csv is not None:
        screen = _read_d1_evidence(screen_csv, "screening", ())
        selection = d1_screen(screen.data)
        if selection["status"] != "go":
            raise ValueError("screen nominated no survivor for confirmation")
        survivors = selection["survivors"]
        evidence = _read_d1_evidence(path, stage, survivors)
        _check_d1_continuity(screen, evidence, file_digest(screen_csv))
        report = d1_confirm(evidence.data, survivors)
        report["screen_summary_sha256"] = evidence.screen_summary_sha256
        contrasts = _d1_confirmation_contrasts(survivors)
    else:
        raise ValueError("confirmation requires --d1-screen-csv; screening takes only its own CSV")
    report.update(
        manifest_digest=evidence.manifest, common=evidence.common, arms=evidence.arms,
        val_bpb_by_seed=evidence.data,
        train_tokens_per_cell=registered_tokens(stage, int(evidence.common["total_batch_size_tokens"])),
    )
    report["matching"] = {}
    for c in contrasts:
        treatment, control = evidence.arms[c.treatment], evidence.arms[c.control]
        report["matching"][c.name] = {
            "equal_parameter_counts": all(treatment[k] == control[k] for k in ("active_params", "trainable_params")),
            "same_optimizer_recipe": all(treatment[k] == control[k] for k in ("optimizer_names", "optimizer_recipe_digest")),
        }
    return report


def load_matrix_csv(path: Path, metric: str) -> dict[str, dict[int, float]]:
    """Read preset/seed/<metric> rows from an eval_matrix summary.csv.

    Successful rows only (``status == ok`` when present); non-finite metrics skipped.
    Exact repeated cells are idempotent; conflicting finite values fail closed.
    """
    data: dict[str, dict[int, float]] = {}
    source_lines: dict[tuple[str, int], int] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or metric not in reader.fieldnames:
            raise ValueError(
                f"metric {metric!r} not a column in {path} "
                f"(columns: {reader.fieldnames})"
            )
        for row in reader:
            line_number = reader.line_num
            if row.get("status", "ok") not in ("ok", "", None):
                continue
            raw = row.get(metric, "")
            preset, seed_s = row.get("preset"), row.get("seed")
            if not raw or preset is None or not seed_s:
                continue
            try:
                value, seed = float(raw), int(seed_s)
            except ValueError:
                continue
            if not math.isfinite(value):
                continue
            by_seed = data.setdefault(preset, {})
            if seed in by_seed:
                previous = by_seed[seed]
                if value != previous:
                    previous_line = source_lines[(preset, seed)]
                    raise ValueError(
                        f"conflicting duplicate matrix cell {preset!r}, seed={seed} "
                        f"in {path}: lines {previous_line} and {line_number} contain "
                        f"{previous!r} and {value!r}"
                    )
                continue
            by_seed[seed] = value
            source_lines[(preset, seed)] = line_number
    return data


def _load_matrix_provenance(path: Path, metric: str) -> dict[str, Any]:
    """Summarize recipe/dataset provenance for the same usable rows as ``load_matrix_csv``."""
    recipe_sources: set[str] = set()
    data_sources: set[str] = set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or metric not in reader.fieldnames:
            raise ValueError(f"metric {metric!r} not a column in {path}")
        for row in reader:
            if row.get("status", "ok") not in ("ok", "", None):
                continue
            try:
                value = float(row.get(metric, ""))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            recipe_sources.add(row.get("recipe_source") or "unknown")
            data_sources.add(row.get("data") or "unknown")
    canonical = recipe_sources == {"base_train_checkpoint"} and data_sources == {"fineweb"}
    if canonical:
        warning = (
            "CHECKPOINT SUMMARY: rows identify the base_train checkpoint scorer and FineWeb loader. "
            "Use the D1 analysis mode to verify the registered grid, budgets and matched controls."
        )
    else:
        warning = (
            "NONCANONICAL OR UNKNOWN EVIDENCE: inferential calculations are shown for pipeline "
            "inspection only; do not promote them to a recipe-faithful FineWeb bio-vs-vanilla claim"
        )
    return {
        "recipe_sources": sorted(recipe_sources),
        "data_sources": sorted(data_sources),
        "canonical_recipe_evidence": canonical,
        "registered_d1_evidence": False,
        "scope_warning": warning,
    }


def _format_report(report: dict) -> str:
    lines = [
        f"metric={report['metric']} "
        f"({'lower' if report['lower_is_better'] else 'higher'} is better)  "
        f"baseline={report['baseline']}  alpha={report['alpha']}  "
        f"min_pairs={report['min_pairs']}  correction=Holm",
        "",
        f"{'preset':<28}{'n':>3}  {'mean':>10}  {'sample SD':>10}  {'95% CI':>22}  "
        f"{'Δ vs base':>11}  {'t p(adj)':>10}  {'W p(adj)':>10}  verdict",
    ]
    if report.get("scope_warning"):
        lines[1:1] = [str(report["scope_warning"]), ""]
    for preset, e in report["presets"].items():
        a = e["aggregate"]
        ci = f"[{a['ci_low']:.4g}, {a['ci_high']:.4g}]"
        row = (
            f"{preset:<28}{a['n']:>3}  {a['mean']:>10.5g}  {a['std']:>10.5g}  "
            f"{ci:>22}  "
        )
        if "paired_vs_baseline" in e:
            p = e["paired_vs_baseline"]
            row += (
                f"{p['mean_delta']:>+11.4g}  {e['paired_t_p_adjusted']:>10.3g}  "
                f"{e['wilcoxon_p_adjusted']:>10.3g}  {e['verdict']}"
            )
        elif preset == report["baseline"]:
            row += f"{'—':>11}  {'—':>10}  {'—':>10}  baseline"
        else:
            row += f"{'—':>11}  {'—':>10}  {'—':>10}  {e['verdict']}"
        lines.append(row)
    return "\n".join(lines)


def _strict_json_value(value: Any) -> Any:
    """Return a recursively strict-JSON-safe value (non-finite floats become null)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    return value


def _format_markdown_report(report: dict) -> str:
    direction = "lower" if report["lower_is_better"] else "higher"
    lines = [
        f"# Statistical comparison: `{report['metric']}`",
        "",
        f"- Baseline: `{report['baseline']}`",
        f"- Direction: {direction} is better",
        f"- Familywise alpha: `{report['alpha']}` with Holm correction",
        f"- Minimum matched seeds for an inferential verdict: `{report['min_pairs']}`",
        "- Support requires a favorable paired-bootstrap 95% CI and both adjusted paired tests.",
        "",
        "| Preset | n | Mean ± sample SD (Student-t 95% CI) | Delta vs baseline | "
        "Adjusted paired-t p | Adjusted Wilcoxon p | Verdict |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    if report.get("scope_warning"):
        lines[1:1] = ["", f"> **WARNING:** {report['scope_warning']}"]
    for preset, entry in report["presets"].items():
        aggregate_result = entry["aggregate"]
        aggregate_text = (
            f"{aggregate_result['mean']:.6g} ± {aggregate_result['std']:.6g} "
            f"[{aggregate_result['ci_low']:.6g}, {aggregate_result['ci_high']:.6g}]"
        )
        if preset == report["baseline"]:
            lines.append(
                f"| `{preset}` | {aggregate_result['n']} | {aggregate_text} | — | — | — | baseline |"
            )
            continue
        if "paired_vs_baseline" not in entry:
            lines.append(
                f"| `{preset}` | {aggregate_result['n']} | {aggregate_text} | — | — | — | "
                f"`{entry['verdict']}` |"
            )
            continue
        paired = entry["paired_vs_baseline"]
        lines.append(
            f"| `{preset}` | {aggregate_result['n']} | {aggregate_text} | "
            f"{paired['mean_delta']:+.6g} "
            f"[{paired['delta_ci_low']:+.6g}, {paired['delta_ci_high']:+.6g}] | "
            f"{entry['paired_t_p_adjusted']:.4g} | "
            f"{entry['wilcoxon_p_adjusted']:.4g} | `{entry['verdict']}` |"
        )
    lines.extend(
        [
            "",
            "`null` means the preregistered support rule did not pass; it is not evidence of equivalence. "
            "`insufficient_evidence` means too few matched seeds were available for the declared minimum.",
            "",
        ]
    )
    return "\n".join(lines)


def _format_d1_report(report: dict) -> str:
    lines = [
        f"# {report['protocol_version']} · {report['stage']} · {report['status']}",
        "", f"Manifest `{report['manifest_digest']}`. Paired training seeds {report['seeds']}.",
        "", "Lower validation BPB favors the treatment. Parameter and optimizer details appear in the JSON.",
        "", "| Contrast | Treatment − control | Paired BPB differences | Parameter counts equal | Optimizer matches | Outcome |",
        "|---|---|---|---|---|---|",
    ]
    screen = report["stage"] == "screening"
    contrasts = report["contrasts"].values() if screen else report["contrasts"]
    for c in contrasts:
        name = c["name"]
        r = c if screen else report["results"][name]
        if screen:
            outcome = "nominated" if c["advances"] else "descriptive"
        else:
            outcome = (
                f"{r['verdict']} (n nonzero {r['n_nonzero']}, "
                f"adjusted t p={r['paired_t_p_adjusted']:.4g}, "
                f"adjusted W p={r['wilcoxon_p_adjusted']:.4g}, "
                f"95% interval [{r['delta_ci_low']:+.6g}, {r['delta_ci_high']:+.6g}])"
            )
        matching = report["matching"][name]
        deltas = ", ".join(f"{v:+.6g}" for v in r["paired_deltas"])
        lines.append(
            f"| {name} | `{c['treatment']}` − `{c['control']}` | {deltas} | "
            f"{matching['equal_parameter_counts']} | {matching['same_optimizer_recipe']} | {outcome} |"
        )
    if screen:
        lines.extend(["", "Screening nominates candidates only. A no-go yields no confirmatory test."])
    else:
        lines.extend(["", f"Holm family size {report['family_size']}. "
                      "A null leaves the effect unresolved. Unequal capacity or optimizer changes "
                      "limit attribution to the tested configuration."])
    return "\n".join(lines) + "\n"


def main() -> int:
    console = Console()
    ap = argparse.ArgumentParser(description="Bio-vs-vanilla statistical comparison.")
    ap.add_argument("csv", type=Path, help="eval_matrix summary.csv")
    ap.add_argument("--metric", default="val_bpb")
    ap.add_argument("--baseline", default="vanilla")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-pairs", type=int, default=3)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--markdown-out", type=Path)
    ap.add_argument("--d1-stage", choices=("screening", "confirmation"))
    ap.add_argument("--d1-screen-csv", type=Path, help="screening summary admitted to confirmation")
    ap.add_argument(
        "--higher-better", action="store_true", help="force higher-is-better direction"
    )
    args = ap.parse_args()

    try:
        if args.d1_stage:
            if (args.metric, args.baseline, args.alpha, args.min_pairs, args.higher_better) != (
                "val_bpb", "vanilla", D1_ALPHA, 3, False
            ):
                raise ValueError("metric, direction, contrasts, alpha and sample size are frozen for D1")
            report = d1_csv_report(args.csv, stage=args.d1_stage, screen_csv=args.d1_screen_csv)
            formatted = _format_d1_report(report)
        else:
            if args.d1_screen_csv is not None:
                raise ValueError("--d1-screen-csv requires --d1-stage confirmation")
            data = load_matrix_csv(args.csv, args.metric)
            report = compare_matrix(
                data, baseline=args.baseline, metric=args.metric,
                lower_is_better=(False if args.higher_better else None),
                alpha=args.alpha, min_pairs=args.min_pairs,
            )
            report.update(_load_matrix_provenance(args.csv, args.metric))
            formatted = _format_report(report)
    except (OSError, ValueError) as exc:
        console.print(f"Inconclusive: {exc}", markup=False, style="red")
        return 2
    console.print(formatted, markup=False)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                _strict_json_value(report), sort_keys=True, indent=2, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
        console.print(f"[green]Wrote strict JSON report:[/green] {args.json_out}")
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown = formatted if args.d1_stage else _format_markdown_report(report)
        args.markdown_out.write_text(markdown, encoding="utf-8")
        console.print(f"[green]Wrote Markdown report:[/green] {args.markdown_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
