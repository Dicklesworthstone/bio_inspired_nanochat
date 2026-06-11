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
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

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


def aggregate(values: list[float] | np.ndarray, confidence: float = 0.95) -> Aggregate:
    """Mean and Student-t confidence interval over seeds."""
    a = np.asarray(values, dtype=np.float64)
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
    a = np.asarray(values, dtype=np.float64)
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
    dz = float(deltas.mean() / sd) if sd > 0 else 0.0
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
        return lower_is_better
    spec = get_metric(metric)
    if spec is None:
        # Unknown metric: default to lower-is-better (losses/bpb dominate) and let the
        # caller override. Common bpb/loss names also matched defensively.
        return not any(k in metric for k in ("acc", "accuracy", "tok_per_sec", "mfu"))
    return spec.direction == Direction.LOWER_BETTER


def compare_matrix(
    data: dict[str, dict[int, float]],
    *,
    baseline: str,
    metric: str = "val_bpb",
    lower_is_better: Optional[bool] = None,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Aggregate every preset and test each against ``baseline`` on matched seeds.

    ``data`` maps preset -> {seed: metric_value}. Returns a structured report with a
    per-preset aggregate and (for non-baseline presets) a paired comparison whose
    ``better``/``significant`` flags are direction-aware.
    """
    if baseline not in data:
        raise ValueError(f"baseline preset {baseline!r} not in data ({sorted(data)})")
    lower = _direction_lower_better(metric, lower_is_better)
    report: dict = {
        "metric": metric,
        "lower_is_better": lower,
        "baseline": baseline,
        "alpha": alpha,
        "presets": {},
    }
    for preset, by_seed in data.items():
        agg = aggregate(list(by_seed.values()))
        entry: dict = {"aggregate": asdict(agg)}
        if preset not in (baseline,):  # every preset except the baseline gets a paired test
            paired = paired_comparison(
                by_seed, data[baseline], lower_is_better=lower, seed=seed
            )
            if paired is not None:
                improvement = -paired.mean_delta if lower else paired.mean_delta
                entry["paired_vs_baseline"] = asdict(paired)
                entry["better"] = improvement > 0
                entry["significant"] = (
                    min(paired.t_p_value, paired.wilcoxon_p_value) < alpha
                )
        report["presets"][preset] = entry
    return report


# --------------------------------------------------------------------------- #
# CLI: read an eval_matrix summary.csv and print the comparison
# --------------------------------------------------------------------------- #
def load_matrix_csv(path: Path, metric: str) -> dict[str, dict[int, float]]:
    """Read preset/seed/<metric> rows from an eval_matrix summary.csv.

    Successful rows only (``status == ok`` when present); non-finite metrics skipped.
    Repeated (preset, seed) keep the last finite value.
    """
    data: dict[str, dict[int, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or metric not in reader.fieldnames:
            raise ValueError(
                f"metric {metric!r} not a column in {path} "
                f"(columns: {reader.fieldnames})"
            )
        for row in reader:
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
            data.setdefault(preset, {})[seed] = value
    return data


def _format_report(report: dict) -> str:
    lines = [
        f"metric={report['metric']} "
        f"({'lower' if report['lower_is_better'] else 'higher'} is better)  "
        f"baseline={report['baseline']}  alpha={report['alpha']}",
        "",
        f"{'preset':<28}{'n':>3}  {'mean':>10}  {'95% CI':>22}  "
        f"{'Δ vs base':>11}  {'t p':>8}  {'W p':>8}  verdict",
    ]
    for preset, e in report["presets"].items():
        a = e["aggregate"]
        ci = f"[{a['ci_low']:.4g}, {a['ci_high']:.4g}]"
        row = f"{preset:<28}{a['n']:>3}  {a['mean']:>10.5g}  {ci:>22}  "
        if "paired_vs_baseline" in e:
            p = e["paired_vs_baseline"]
            verdict = (
                ("BETTER" if e["better"] else "WORSE")
                + (" *" if e["significant"] else "")
            )
            row += (
                f"{p['mean_delta']:>+11.4g}  {p['t_p_value']:>8.3g}  "
                f"{p['wilcoxon_p_value']:>8.3g}  {verdict}"
            )
        else:
            row += f"{'—':>11}  {'—':>8}  {'—':>8}  baseline"
        lines.append(row)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Bio-vs-vanilla statistical comparison.")
    ap.add_argument("csv", type=Path, help="eval_matrix summary.csv")
    ap.add_argument("--metric", default="val_bpb")
    ap.add_argument("--baseline", default="vanilla")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument(
        "--higher-better", action="store_true", help="force higher-is-better direction"
    )
    args = ap.parse_args()

    data = load_matrix_csv(args.csv, args.metric)
    if args.baseline not in data:
        # Fall back to bio_all so the tool still works on baseline-less matrices.
        alt = "bio_all" if "bio_all" in data else next(iter(data), None)
        if alt is None:
            print("No usable rows.")
            return 1
        args.baseline = alt
    report = compare_matrix(
        data,
        baseline=args.baseline,
        metric=args.metric,
        lower_is_better=(False if args.higher_better else None),
        alpha=args.alpha,
    )
    print(_format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
