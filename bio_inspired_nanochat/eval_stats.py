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
        return (0.0, 1.0) if deltas.mean() == 0.0 else (float("inf"), 0.0)
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
# CLI: read an eval_matrix summary.csv and print the comparison
# --------------------------------------------------------------------------- #
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
    warning = None
    if not canonical:
        warning = (
            "NONCANONICAL OR UNKNOWN EVIDENCE: inferential calculations are shown for pipeline "
            "inspection only; do not promote them to a recipe-faithful FineWeb bio-vs-vanilla claim"
        )
    return {
        "recipe_sources": sorted(recipe_sources),
        "data_sources": sorted(data_sources),
        "canonical_recipe_evidence": canonical,
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
    ap.add_argument(
        "--higher-better", action="store_true", help="force higher-is-better direction"
    )
    args = ap.parse_args()

    data = load_matrix_csv(args.csv, args.metric)
    if args.baseline not in data:
        console.print(
            f"[red]Baseline {args.baseline!r} has no usable rows in {args.csv}.[/red]"
        )
        return 2
    try:
        report = compare_matrix(
            data,
            baseline=args.baseline,
            metric=args.metric,
            lower_is_better=(False if args.higher_better else None),
            alpha=args.alpha,
            min_pairs=args.min_pairs,
        )
        report.update(_load_matrix_provenance(args.csv, args.metric))
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    console.print(_format_report(report), markup=False)
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
        args.markdown_out.write_text(_format_markdown_report(report), encoding="utf-8")
        console.print(f"[green]Wrote Markdown report:[/green] {args.markdown_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
