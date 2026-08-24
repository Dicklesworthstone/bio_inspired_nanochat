"""Stochastic thermodynamics of vesicle release — reference implementation (Thrust E, bead `0642.3.1`).

Turns each stochastic vesicle-release event into a nonequilibrium Markov jump process and makes its
**entropy production** `Σ` a first-class, computable quantity, then derives the two falsifiable
guarantees of `docs/theory/stochastic_thermodynamics.md`:

  - the **Thermodynamic Uncertainty Relation** (TUR) `Var(J)/⟨J⟩² ≥ 2/⟨Σ⟩` — a *precision costs
    entropy* lower bound on the relative uncertainty of any release current `J` (bead `0642.3.1.2`);
  - the **Crooks / Jarzynski** fluctuation identities `P_F(Σ)/P_R(−Σ) = e^Σ` and `⟨e^{−Σ}⟩ = 1`,
    which turn into an analytic **calibration guarantee** the empirical work histogram must satisfy
    (bead `0642.3.1.3`).

The physical engine is the existing stochastic release `K ~ Binomial(N=RRP, p=release_prob)`
(`synaptic._sample_binomial_counts`) with recovery rate `rec_rate`. In the Poisson (rare-release)
limit the release current `J = N₊ − N₋` is a **Skellam** process — the difference of a release
Poisson `N₊ ~ Poisson(a·t)` and a recovery Poisson `N₋ ~ Poisson(b·t)`, with `a ∝ p` the release
propensity and `b ∝ rec_rate` the recovery propensity. The metabolic drive makes `a > b`, breaking
detailed balance, so `Σ = J·ln(a/b) > 0`. For this model the fluctuation identities hold **exactly**
(proved in the note §1–§3), which is why the corroboration tests are exact rather than statistical
flukes. This module is the theory + reference math; the runtime TUR certificate + Crooks monitor
(`0642.3.2.1`) and the falsification vs the softmax/MC baseline (`0642.3.3.1`) build on it.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


# =========================================================================== #
# §1. The Markov jump model + entropy production Σ (bead 0642.3.1.1)
# =========================================================================== #
@dataclass(frozen=True)
class ReleaseRates:
    """The two competing jump propensities of the driven vesicle cycle (per unit time).

    `a` = release propensity (forward jumps `Docked → Released`, ∝ release prob `p` × pool size);
    `b` = recovery propensity (reverse jumps `Released → Docked`, ∝ `rec_rate` × released count).
    The drive (calcium/ATP) sustains `a > b`, which is exactly what breaks detailed balance and makes
    the release a genuine nonequilibrium process with positive entropy production.
    """

    a: float
    b: float

    def __post_init__(self) -> None:
        if self.a <= 0.0 or self.b <= 0.0:
            raise ValueError(f"release/recovery propensities must be positive, got a={self.a}, b={self.b}")


def affinity(rates: ReleaseRates) -> float:
    """The thermodynamic affinity `A = ln(a/b)` — entropy produced per *net* forward (release) jump.

    `A > 0` ⟺ release-biased (dissipative, the driven regime); `A = 0` ⟺ detailed balance
    (equilibrium, `a = b`); `A < 0` ⟺ recovery-biased. This is the per-jump building block of `Σ`.
    """
    return math.log(rates.a / rates.b)


def mean_current(rates: ReleaseRates, steps: float) -> float:
    """⟨J⟩ = (a − b)·t — the mean net release current over `steps` units of time."""
    return (rates.a - rates.b) * steps


def var_current(rates: ReleaseRates, steps: float) -> float:
    """Var(J) = (a + b)·t — the variance of the net release current (sum of the two Poisson variances)."""
    return (rates.a + rates.b) * steps


def mean_entropy_production(rates: ReleaseRates, steps: float) -> float:
    """⟨Σ⟩ = ⟨J⟩·A = (a − b)·t·ln(a/b) ≥ 0 — the mean entropy produced over `steps` (note §1).

    Non-negative for all `a, b > 0` (equal sign of `(a−b)` and `ln(a/b)`), i.e. the second law: the
    driven release always dissipates, with equality only at detailed balance `a = b`.
    """
    return mean_current(rates, steps) * affinity(rates)


def entropy_production_samples(currents: np.ndarray, rates: ReleaseRates) -> np.ndarray:
    """`Σ[ω] = J[ω]·ln(a/b)` — the medium entropy production of each sampled release trajectory.

    A release jump `D→R` (forward prob `a`) against its time reverse `R→D` (prob `b`) contributes
    `+ln(a/b)`; a recovery jump contributes `−ln(a/b)`. So a trajectory with net displacement `J`
    produces `J·ln(a/b)` — the log-ratio of the forward to the time-reversed path probability.
    """
    return np.asarray(currents, dtype=np.float64) * affinity(rates)


def simulate_currents(rates: ReleaseRates, steps: float, n_traj: int, *, seed: int) -> np.ndarray:
    """Sample `n_traj` release currents `J = N₊ − N₋` of the driven Markov jump process (Skellam).

    `N₊ ~ Poisson(a·t)` forward (release) jumps, `N₋ ~ Poisson(b·t)` reverse (recovery) jumps, summed
    over `t = steps` — the Poisson (rare-release-per-step) limit of the binomial release/recovery.
    """
    rng = np.random.default_rng(seed)
    n_plus = rng.poisson(rates.a * steps, size=n_traj)
    n_minus = rng.poisson(rates.b * steps, size=n_traj)
    return (n_plus - n_minus).astype(np.float64)


def integral_fluctuation_theorem(sigmas: np.ndarray) -> float:
    """The Jarzynski-type integral fluctuation theorem statistic `⟨e^{−Σ}⟩` (must be `≈ 1`).

    Exact for the Skellam model: `⟨e^{−J·ln(a/b)}⟩ = ⟨(b/a)^J⟩ = e^{a t(b/a − 1) + b t(a/b − 1)} = 1`
    (note §1). The second law `⟨Σ⟩ ≥ 0` is then Jensen's inequality on this identity. Note: the
    Monte-Carlo estimator converges slowly far from equilibrium (rare negative-`Σ` trajectories
    dominate the average), so verify it by simulation only in the near-equilibrium regime; the
    closed form below is the exact statement.
    """
    s = np.asarray(sigmas, dtype=np.float64)
    return float(np.mean(np.exp(-s)))


def integral_ft_closed_form(rates: ReleaseRates, steps: float) -> float:
    """The *exact* `⟨e^{−Σ}⟩` from the Skellam moment generating function — `≡ 1` for all `a, b, t`.

    `⟨e^{−Σ}⟩ = ⟨(b/a)^J⟩ = exp(a t (b/a − 1)) · exp(b t (a/b − 1)) = exp(t(b−a) + t(a−b)) = 1`.
    This is the analytic counterpart of `integral_fluctuation_theorem` (which only converges by MC
    near equilibrium); together they show the identity holds and that the simulator reproduces it.
    The two exponents are summed *before* exponentiating: each alone can overflow for an extreme drive
    `a/b`, but their sum is identically 0 — so the combined form returns 1.0 for any `a, b, t`.
    """
    a, b, t = rates.a, rates.b, steps
    return math.exp(a * t * (b / a - 1.0) + b * t * (a / b - 1.0))


def detailed_fluctuation_ratio(currents: np.ndarray, rates: ReleaseRates, k: int) -> tuple[float, float]:
    """Empirical vs predicted `P(J=+k)/P(J=−k)` — the detailed fluctuation theorem `P(Σ)/P(−Σ)=e^Σ`.

    Returns `(empirical_ratio, e^{k·A})`. For the Skellam current the ratio is exactly `(a/b)^k`
    (the Bessel-`I` factors cancel), i.e. `e^{Σ}` with `Σ = k·ln(a/b)`.
    """
    c = np.asarray(currents, dtype=np.float64)
    n_pos = int(np.sum(c == k))
    n_neg = int(np.sum(c == -k))
    empirical = float("inf") if n_neg == 0 else n_pos / n_neg
    predicted = math.exp(k * affinity(rates))
    return empirical, predicted


# =========================================================================== #
# §2. The Thermodynamic Uncertainty Relation (bead 0642.3.1.2)
# =========================================================================== #
@dataclass(frozen=True)
class TURCertificate:
    """The per-current TUR verdict: measured relative variance vs the entropy bound `2/⟨Σ⟩`."""

    relative_variance: float   # ε² = Var(J)/⟨J⟩² — the squared relative uncertainty (the "precision")
    entropy_bound: float       # 2/⟨Σ⟩ — the TUR lower bound on ε²
    mean_entropy: float        # ⟨Σ⟩
    satisfied: bool            # ε² ≥ 2/⟨Σ⟩ (the TUR holds; a theorem, so this must be True)
    slack: float               # ε² − 2/⟨Σ⟩ ≥ 0 (how far above the bound — the achievable precision margin)


def tur_bound(mean_sigma: float) -> float:
    """The TUR lower bound on the relative variance: `2/⟨Σ⟩`. *Precision costs entropy.*"""
    if mean_sigma <= 0.0:
        raise ValueError(f"⟨Σ⟩ must be positive (a driven current), got {mean_sigma}")
    return 2.0 / mean_sigma


def tur_certificate(rates: ReleaseRates, steps: float) -> TURCertificate:
    """Analytic TUR certificate for the release current over `steps` (note §2).

    `ε² = Var(J)/⟨J⟩² = (a+b)/((a−b)²·t)` is lower-bounded by `2/⟨Σ⟩ = 2/((a−b)·t·ln(a/b))`. The TUR
    is `ε² ≥ 2/⟨Σ⟩` itself (always non-negative since `⟨Σ⟩ = (a−b)·ln(a/b) ≥ 0`); cleared of the
    denominators it reads `(a+b)·|ln(a/b)| ≥ 2·|a−b|`, a theorem for all `a,b>0` (the sign of
    `(a−b)` and `ln(a/b)` always agree). So `satisfied` is always True and `slack` is the precision the
    head can actually buy beyond the thermodynamic floor.
    """
    mean_j = mean_current(rates, steps)
    if mean_j == 0.0:
        raise ValueError("TUR needs a non-zero mean current (a ≠ b); got a == b (equilibrium)")
    rel_var = var_current(rates, steps) / (mean_j * mean_j)
    mean_sig = mean_entropy_production(rates, steps)
    bound = tur_bound(mean_sig)
    return TURCertificate(
        relative_variance=rel_var, entropy_bound=bound, mean_entropy=mean_sig,
        satisfied=bool(rel_var >= bound - 1e-12), slack=rel_var - bound,
    )


def empirical_tur(currents: np.ndarray, mean_sigma: float) -> TURCertificate:
    """The TUR certificate from *sampled* currents (the runtime-measurable form, `0642.3.2.1`).

    Uses the empirical mean/variance of `J` against the analytic ⟨Σ⟩; this is what a per-head monitor
    evaluates online to certify that a head's release precision is thermodynamically honest. Requires
    at least 2 samples: with `n < 2` the empirical variance is undefined / unreliable and the TUR is
    not testable (a 1-sample `Var=0` would otherwise read as a spurious "violation" of a theorem).
    """
    c = np.asarray(currents, dtype=np.float64)
    if c.size < 2:
        raise ValueError(f"empirical TUR needs >= 2 current samples, got {c.size}")
    mean_j = float(np.mean(c))
    if mean_j == 0.0:
        raise ValueError("empirical TUR needs a non-zero mean current")
    rel_var = float(np.var(c)) / (mean_j * mean_j)
    bound = tur_bound(mean_sigma)
    return TURCertificate(
        relative_variance=rel_var, entropy_bound=bound, mean_entropy=mean_sigma,
        satisfied=bool(rel_var >= bound - 1e-9), slack=rel_var - bound,
    )


# =========================================================================== #
# §3. Crooks / Jarzynski → the calibration guarantee (bead 0642.3.1.3)
# =========================================================================== #
def jarzynski_free_energy(work: np.ndarray, kT: float = 1.0) -> float:
    """Jarzynski free-energy estimate `ΔF = −kT·ln⟨e^{−w/kT}⟩` from work samples (numerically stable).

    For the steady-state release the dissipated work is `w = kT·Σ` and `ΔF = 0`, so this returns ≈0 —
    the equilibrium free-energy difference recovered from purely nonequilibrium release fluctuations.
    """
    if kT <= 0.0:
        raise ValueError(f"kT must be positive, got {kT}")
    w = np.asarray(work, dtype=np.float64) / kT
    w_min = float(np.min(w))
    # −kT·ln(mean(exp(−w))) with the min shifted out for stability.
    return -kT * (float(np.log(np.mean(np.exp(-(w - w_min))))) - w_min)


@dataclass(frozen=True)
class CrooksCalibration:
    """The Crooks calibration check: does the empirical `Σ` histogram obey `ln(P(+Σ)/P(−Σ)) = Σ`?"""

    bins: np.ndarray            # the positive Σ-bin centers tested (paired with their −Σ mirror)
    log_ratio: np.ndarray       # ln(P(+Σ)/P(−Σ)) measured
    predicted: np.ndarray       # Σ (the detailed-FT line; = w/kT with ΔF = 0)
    max_abs_residual: float     # sup |measured − predicted| over populated symmetric bins
    calibrated: bool            # residual within tolerance ⟹ the calibration guarantee holds


def crooks_calibration(sigma: np.ndarray, *, n_bins: int = 21, tol: float = 0.25,
                       min_count: int = 30) -> CrooksCalibration:
    """Test the detailed fluctuation theorem `ln(P(+Σ)/P(−Σ)) = Σ` on the entropy-production histogram.

    This is the analytic **calibration guarantee** the runtime monitor (`0642.3.2.1`) evaluates: the
    empirical forward/reverse symmetry of the `Σ` histogram (with `Σ = w/kT`, `ΔF = 0` for a
    steady-state current) must reproduce the line `Σ` on every populated symmetric-bin pair. Bins are
    symmetric about 0 so bin `i` (center `+c`) pairs with bin `n−1−i` (center `−c`); equal widths make
    the count ratio the density ratio. A guarantee the post-hoc-ECE baselines structurally cannot give.
    """
    s = np.asarray(sigma, dtype=np.float64)
    m = float(np.percentile(np.abs(s), 99.0))
    if m <= 0.0:
        return CrooksCalibration(np.array([]), np.array([]), np.array([]), float("inf"), False)
    edges = np.linspace(-m, m, n_bins + 1)          # symmetric about 0
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(s, bins=edges)
    measured, predicted, kept = [], [], []
    for i in range(n_bins):
        c = centers[i]
        j = n_bins - 1 - i                          # the mirror bin (center −c)
        # require both bins populated (and the mirror non-empty regardless of min_count) to avoid a
        # log(count/0) = inf when the caller passes min_count == 0.
        if c > 0 and counts[i] >= max(1, min_count) and counts[j] >= max(1, min_count):
            measured.append(math.log(counts[i] / counts[j]))
            predicted.append(c)
            kept.append(c)
    if not measured:
        return CrooksCalibration(np.array([]), np.array([]), np.array([]), float("inf"), False)
    measured = np.array(measured)
    predicted = np.array(predicted)
    resid = float(np.max(np.abs(measured - predicted)))
    return CrooksCalibration(
        bins=np.array(kept), log_ratio=measured, predicted=predicted,
        max_abs_residual=resid, calibrated=bool(resid <= tol),
    )


# =========================================================================== #
# §4. Energy-optimal (Landauer) release temperature (bead 0642.3.1.4)
# =========================================================================== #
def _rate_distortion_residual(snr: float) -> float:
    """`2·SNR/(1+SNR) − ln(1+SNR)`: zero at the bits-per-joule-optimal SNR (positive below, negative above)."""
    return 2.0 * snr / (1.0 + snr) - math.log1p(snr)


def optimal_exploration_snr(tol: float = 1e-13) -> float:
    """The signal-to-noise ratio that maximizes the release's bits-per-joule: `SNR* ≈ 3.9215`.

    Maximizing `bits/energy ∝ log₂(1+SNR)/√SNR` (the Gaussian-channel information over the TUR energy
    floor `2·SNR·kT` at the matched temperature `kT = σ/√SNR`) gives the transcendental stationarity
    `2·SNR/(1+SNR) = ln(1+SNR)`, whose unique positive root is the universal rate-distortion operating
    point of the release. Solved by bisection (no SciPy dependency).
    """
    lo, hi = 1.0, 100.0  # residual > 0 at lo, < 0 at hi (bracket the root)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _rate_distortion_residual(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def bits_per_joule(snr: float) -> float:
    """The (unnormalized) bits-per-joule of a release at signal-to-noise ratio `snr`.

    `½·log₂(1+SNR)` bits delivered over an energy `∝ √SNR` (the TUR floor `2·SNR·kT` evaluated at the
    matched temperature `kT = σ/√SNR`). A single interior maximum at `optimal_exploration_snr()`.
    """
    if snr <= 0.0:
        return 0.0
    return 0.5 * math.log2(1.0 + snr) / math.sqrt(snr)


def landauer_optimal_temperature(drive_uncertainty: float, *, snr_star: float | None = None) -> float:
    """The energy-optimal release temperature `kT* = σ/√SNR* ≈ 0.505·σ` (`σ` = drive uncertainty).

    The release resolves the signal exactly to the level of its uncertainty: any finer wastes
    metabolic energy on spurious precision (the Landauer cost), any coarser throws away signal. The
    matched point is the bits-per-joule maximum — a *thermodynamically optimal* attention temperature.
    """
    if drive_uncertainty <= 0.0:
        raise ValueError(f"drive uncertainty must be positive, got {drive_uncertainty}")
    s = optimal_exploration_snr() if snr_star is None else snr_star
    return drive_uncertainty / math.sqrt(s)


def ach_coupled_temperature(drive_uncertainty_base: float, ach_level: float, *, ach_gain: float = 1.0) -> float:
    """ACh-coupled energy-optimal release temperature (couples `0642.3.1.4` to `hy8.5`).

    Acetylcholine signals uncertainty/attention; here it scales the effective drive uncertainty
    `σ(ACh) = σ_base·(1 + ach_gain·ACh)`, so the optimal temperature `kT*(ACh) = σ(ACh)/√SNR*` rises
    with ACh: more uncertainty ⟹ hotter release ⟹ more exploration — Landauer-optimal *and*
    state-dependent. Default-neutral at `ach_level = 0` (`kT* = σ_base/√SNR*`).
    """
    sigma = drive_uncertainty_base * (1.0 + ach_gain * max(0.0, ach_level))
    return landauer_optimal_temperature(sigma)


# =========================================================================== #
# Bridge to the live release subsystem
# =========================================================================== #
def rates_from_release(p_release: float, rec_rate: float, pool: float) -> ReleaseRates:
    """Map the live release parameters to the Markov-jump propensities.

    `a = p_release · pool` (expected forward releases per step from a pool of `pool` docked vesicles),
    `b = rec_rate · pool` (expected recoveries). The drive `a > b ⟺ p_release > rec_rate` is exactly
    the high-calcium / metabolically-driven regime where the release dissipates and `Σ > 0`.
    """
    return ReleaseRates(a=max(p_release * pool, 1e-12), b=max(rec_rate * pool, 1e-12))


# =========================================================================== #
# Predictive-ensemble evidence bridge (bead 0642.3.4)
# =========================================================================== #
@dataclass(frozen=True)
class PredictiveEvidenceProvenance:
    """Immutable identity binding for one predictive-distribution evidence stream."""

    run_id: str
    checkpoint_id: str
    config_hash: str
    rng_seed: int

    def __post_init__(self) -> None:
        if not self.run_id or not self.checkpoint_id or not self.config_hash:
            raise ValueError("run, checkpoint, and config identities must be non-empty")
        if self.rng_seed < 0:
            raise ValueError("rng_seed must be nonnegative")


@dataclass(frozen=True)
class PredictiveEvidencePolicy:
    """Predeclared local calibration gates for layer/head release evidence."""

    min_samples: int = 8
    min_tested_fraction: float = 0.75
    min_symmetric_bins: int = 2
    crooks_bins: int = 21
    crooks_min_count: int = 5
    crooks_tolerance: float = 0.35
    min_tur_bound_ratio: float = 0.95
    max_events_per_head: int = 100_000

    def __post_init__(self) -> None:
        if self.min_samples < 2:
            raise ValueError("min_samples must be at least two")
        if not 0.0 < self.min_tested_fraction <= 1.0:
            raise ValueError("min_tested_fraction must lie in (0, 1]")
        if self.min_symmetric_bins < 1 or self.crooks_bins < 3:
            raise ValueError("Crooks evidence needs positive support and at least three bins")
        if self.min_symmetric_bins > self.crooks_bins // 2:
            raise ValueError("min_symmetric_bins exceeds the positive half of crooks_bins")
        if self.crooks_min_count < 1 or self.crooks_tolerance < 0.0:
            raise ValueError("Crooks count must be positive and tolerance nonnegative")
        if self.min_tur_bound_ratio <= 0.0:
            raise ValueError("min_tur_bound_ratio must be positive")
        if self.max_events_per_head < 2:
            raise ValueError("max_events_per_head must be at least two")


@dataclass(frozen=True)
class HeadPredictiveThermoEvidence:
    """One named presynaptic layer/head's tested predictive-ensemble evidence."""

    layer_address: str
    head_index: int
    sampling_modes: tuple[str, ...]
    sample_count: int
    observed_events: int
    tested_events: int
    retained_events: int
    degenerate_events: int
    tested_fraction: float
    symmetric_bins: int
    crooks_residual: float | None
    tur_relative_variance: float | None
    tur_entropy_bound: float | None
    tur_bound_ratio: float | None
    finite: bool
    passed: bool
    refusal_reasons: tuple[str, ...]


@dataclass(frozen=True)
class PredictiveThermoEvidence:
    """Seed-local, provenance-bound evidence and its fail-closed claim state."""

    provenance: PredictiveEvidenceProvenance
    policy: PredictiveEvidencePolicy
    heads: tuple[HeadPredictiveThermoEvidence, ...]
    observed_events: int
    tested_events: int
    retained_events: int
    degenerate_events: int
    tested_fraction: float
    fresh: bool
    local_gates_passed: bool
    multi_seed_statistics_passed: bool
    predictive_distribution_claim: bool
    calibration_mode: str
    refusal_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_jsonl(self) -> list[str]:
        """Emit one summary and one record per layer/head without NaN/Inf sentinels."""
        summary = self.to_dict()
        summary.pop("heads")
        lines = [json.dumps({"record_type": "predictive_thermo_summary", **summary})]
        lines.extend(
            json.dumps(
                {
                    "record_type": "predictive_thermo_head",
                    "run_id": self.provenance.run_id,
                    **asdict(head),
                }
            )
            for head in self.heads
        )
        return lines

    def render(self, console=None) -> None:
        """Render compact Rich evidence; the structured JSONL remains the audit source."""
        from rich.console import Console
        from rich.table import Table

        console = console or Console()
        table = Table(title="Predictive thermodynamic evidence by layer/head")
        table.add_column("Layer/head")
        table.add_column("Samples", justify="right")
        table.add_column("Tested/retained/observed", justify="right")
        table.add_column("Symmetric bins", justify="right")
        table.add_column("Crooks residual", justify="right")
        table.add_column("TUR ratio", justify="right")
        table.add_column("Gate")
        for head in self.heads:
            table.add_row(
                f"{head.layer_address}/h{head.head_index}",
                str(head.sample_count),
                f"{head.tested_events}/{head.retained_events}/{head.observed_events}",
                str(head.symmetric_bins),
                "untested" if head.crooks_residual is None else f"{head.crooks_residual:.4f}",
                "untested" if head.tur_bound_ratio is None else f"{head.tur_bound_ratio:.4f}",
                "PASS" if head.passed else ", ".join(head.refusal_reasons),
            )
        console.print(table)
        console.print(
            f"Predictive-distribution claim={self.predictive_distribution_claim}; "
            f"mode={self.calibration_mode}; fresh={self.fresh}; "
            f"reasons={list(self.refusal_reasons)}"
        )


@dataclass(frozen=True)
class MultiSeedPredictiveThermoVerdict:
    """Group claim: local release evidence AND matched-seed prediction statistics."""

    seed_count: int
    local_seed_pass_rate: float
    multi_seed_statistics_passed: bool
    predictive_distribution_claim: bool
    calibration_mode: str
    refusal_reasons: tuple[str, ...]


def predictive_distribution_verdict(
    evidences: list[PredictiveThermoEvidence] | tuple[PredictiveThermoEvidence, ...],
    *,
    multi_seed_statistics_passed: bool,
) -> MultiSeedPredictiveThermoVerdict:
    """Join seed-local evidence to the predeclared matched-seed statistical obligation."""
    reasons: list[str] = []
    if len(evidences) < 2:
        reasons.append("insufficient_seed_coverage")
    if not evidences:
        local_pass_rate = 0.0
        reasons.append("empty_evidence")
    else:
        local_pass_rate = sum(
            evidence.fresh and evidence.local_gates_passed for evidence in evidences
        ) / len(evidences)
        if any(not evidence.fresh for evidence in evidences):
            reasons.append("stale_evidence")
        run_ids = {evidence.provenance.run_id for evidence in evidences}
        rng_seeds = {evidence.provenance.rng_seed for evidence in evidences}
        if len(run_ids) != len(evidences) or len(rng_seeds) != len(evidences):
            reasons.append("duplicate_seed_evidence")
        if local_pass_rate < 1.0:
            reasons.append("local_layer_head_gates_failed")
    if not multi_seed_statistics_passed:
        reasons.append("multi_seed_statistics_failed")
    claim = not reasons
    return MultiSeedPredictiveThermoVerdict(
        seed_count=len(evidences),
        local_seed_pass_rate=local_pass_rate,
        multi_seed_statistics_passed=multi_seed_statistics_passed,
        predictive_distribution_claim=claim,
        calibration_mode=(
            "predictive_thermodynamic_calibration"
            if claim
            else "empirical_ece_fallback"
        ),
        refusal_reasons=tuple(reasons),
    )


class PredictiveThermoCollector:
    """Collect exact-binomial MC release events without perturbing the model RNG.

    The forward counts are the counts that actually shaped each predictive draw. A private NumPy
    generator supplies the matched reverse-protocol counts after capture; it is seeded by the
    provenance record and is isolated from PyTorch's inference generator. Each event uses the exact
    paired-binomial affinity ``log(p(1-q)/(q(1-p)))``. Approximate/non-integral release modes are
    recorded as degenerate rather than silently promoted to thermodynamic evidence.
    """

    def __init__(
        self,
        provenance: PredictiveEvidenceProvenance,
        policy: PredictiveEvidencePolicy | None = None,
    ) -> None:
        self.provenance = provenance
        self.policy = policy or PredictiveEvidencePolicy()
        self._rng = np.random.default_rng(provenance.rng_seed)
        self._current_sample: int | None = None
        self._observed: dict[tuple[str, int], int] = {}
        self._tested: dict[tuple[str, int], int] = {}
        self._degenerate: dict[tuple[str, int], int] = {}
        self._sample_ids: dict[tuple[str, int], set[int]] = {}
        self._sampling_modes: dict[tuple[str, int], set[str]] = {}
        self._currents: dict[tuple[str, int], list[np.ndarray]] = {}
        self._sigmas: dict[tuple[str, int], list[np.ndarray]] = {}
        self._priorities: dict[tuple[str, int], list[np.ndarray]] = {}

    def begin_sample(self, sample_index: int) -> None:
        if sample_index < 0:
            raise ValueError("sample_index must be nonnegative")
        self._current_sample = sample_index

    def record(
        self,
        *,
        layer_address: str,
        probabilities,
        pool_sizes,
        forward_counts,
        reverse_probability: float,
        sampling_mode: str,
        valid=None,
        sampled=None,
    ) -> None:
        """Record one live release call; inputs may be Torch tensors or NumPy arrays."""
        if self._current_sample is None:
            raise RuntimeError("begin_sample() must be called before recording release evidence")
        if not layer_address:
            raise ValueError("layer_address must be non-empty")
        if not 0.0 < reverse_probability < 1.0:
            raise ValueError("reverse_probability must lie strictly between zero and one")
        p = np.asarray(_to_numpy(probabilities), dtype=np.float64)
        n_raw = np.asarray(_to_numpy(pool_sizes), dtype=np.float64)
        forward = np.asarray(_to_numpy(forward_counts), dtype=np.float64)
        if p.shape != n_raw.shape or p.shape != forward.shape or p.ndim != 4:
            raise ValueError("release evidence tensors must share shape (B,H,T,K)")
        mask = (
            np.ones(p.shape, dtype=bool)
            if valid is None
            else np.asarray(_to_numpy(valid), dtype=bool)
        )
        if mask.shape != p.shape:
            raise ValueError("valid mask must match release evidence tensors")
        sampled_mask = (
            np.ones(p.shape, dtype=bool)
            if sampled is None
            else np.asarray(_to_numpy(sampled), dtype=bool)
        )
        if sampled_mask.shape != p.shape:
            raise ValueError("sampled mask must match release evidence tensors")

        exact_mode = sampling_mode in {"straight_through", "gumbel_sigmoid_ste"}
        for head in range(p.shape[1]):
            key = (layer_address, head)
            observed_mask = mask[:, head].reshape(-1)
            observed = int(observed_mask.sum())
            self._observed[key] = self._observed.get(key, 0) + observed
            self._sample_ids.setdefault(key, set()).add(self._current_sample)
            self._sampling_modes.setdefault(key, set()).add(sampling_mode)
            if observed == 0:
                continue
            p_head = p[:, head].reshape(-1)[observed_mask]
            n_head = n_raw[:, head].reshape(-1)[observed_mask]
            f_head = forward[:, head].reshape(-1)[observed_mask]
            sampled_head = sampled_mask[:, head].reshape(-1)[observed_mask]
            n_round = np.rint(n_head)
            f_round = np.rint(f_head)
            eligible = (
                sampled_head
                & exact_mode
                & np.isfinite(p_head)
                & np.isfinite(n_head)
                & np.isfinite(f_head)
                & (p_head > 0.0)
                & (p_head < 1.0)
                & (n_round >= 1.0)
                & np.isclose(n_head, n_round, atol=1e-5)
                & np.isclose(f_head, f_round, atol=1e-5)
                & (f_round >= 0.0)
                & (f_round <= n_round)
            )
            tested = int(np.sum(eligible))
            self._tested[key] = self._tested.get(key, 0) + tested
            self._degenerate[key] = self._degenerate.get(key, 0) + observed - tested
            if tested == 0:
                continue
            p_test = p_head[eligible]
            n_test = n_round[eligible].astype(np.int64)
            forward_test = f_round[eligible].astype(np.int64)
            reverse = self._rng.binomial(n_test, reverse_probability)
            current = (forward_test - reverse).astype(np.float64)
            local_affinity = np.log(
                p_test * (1.0 - reverse_probability)
                / (reverse_probability * (1.0 - p_test))
            )
            sigma = current * local_affinity
            priority = self._rng.random(current.size)
            old_currents = self._currents.get(key, [])
            old_sigmas = self._sigmas.get(key, [])
            old_priorities = self._priorities.get(key, [])
            combined_currents = np.concatenate([*old_currents, current])
            combined_sigmas = np.concatenate([*old_sigmas, sigma])
            combined_priorities = np.concatenate([*old_priorities, priority])
            if combined_currents.size > self.policy.max_events_per_head:
                keep = np.argpartition(
                    combined_priorities, -self.policy.max_events_per_head
                )[-self.policy.max_events_per_head :]
                combined_currents = combined_currents[keep]
                combined_sigmas = combined_sigmas[keep]
                combined_priorities = combined_priorities[keep]
            self._currents[key] = [combined_currents]
            self._sigmas[key] = [combined_sigmas]
            self._priorities[key] = [combined_priorities]

    def finalize(
        self,
        *,
        current_checkpoint_id: str,
        current_config_hash: str,
        current_rng_seed: int,
        multi_seed_statistics_passed: bool = False,
    ) -> PredictiveThermoEvidence:
        """Freeze evidence, refusing empty, stale, sparse, non-finite, or failed streams."""
        fresh = bool(
            current_checkpoint_id == self.provenance.checkpoint_id
            and current_config_hash == self.provenance.config_hash
            and current_rng_seed == self.provenance.rng_seed
        )
        heads = tuple(
            self._finalize_head(key)
            for key in sorted(self._observed, key=lambda item: (item[0], item[1]))
        )
        observed = sum(head.observed_events for head in heads)
        tested = sum(head.tested_events for head in heads)
        retained = sum(head.retained_events for head in heads)
        degenerate = sum(head.degenerate_events for head in heads)
        tested_fraction = tested / observed if observed else 0.0
        local_gates_passed = bool(heads and all(head.passed for head in heads))
        reasons: list[str] = []
        if not heads:
            reasons.append("empty_evidence")
        if not fresh:
            reasons.append("stale_evidence")
        if heads and not local_gates_passed:
            reasons.append("local_layer_head_gates_failed")
        if not multi_seed_statistics_passed:
            reasons.append("multi_seed_statistics_pending_or_failed")
        claim = not reasons
        return PredictiveThermoEvidence(
            provenance=self.provenance,
            policy=self.policy,
            heads=heads,
            observed_events=observed,
            tested_events=tested,
            retained_events=retained,
            degenerate_events=degenerate,
            tested_fraction=tested_fraction,
            fresh=fresh,
            local_gates_passed=local_gates_passed,
            multi_seed_statistics_passed=multi_seed_statistics_passed,
            predictive_distribution_claim=claim,
            calibration_mode=(
                "predictive_thermodynamic_calibration"
                if claim
                else "empirical_ece_fallback"
            ),
            refusal_reasons=tuple(reasons),
        )

    def _finalize_head(self, key: tuple[str, int]) -> HeadPredictiveThermoEvidence:
        observed = self._observed[key]
        degenerate = self._degenerate.get(key, observed)
        currents = np.concatenate(self._currents.get(key, [])) if key in self._currents else np.array([])
        sigmas = np.concatenate(self._sigmas.get(key, [])) if key in self._sigmas else np.array([])
        tested = self._tested.get(key, 0)
        retained = int(currents.size)
        tested_fraction = tested / observed if observed else 0.0
        crooks = (
            crooks_calibration(
                sigmas,
                n_bins=self.policy.crooks_bins,
                tol=self.policy.crooks_tolerance,
                min_count=self.policy.crooks_min_count,
            )
            if retained
            else None
        )
        crooks_residual = (
            crooks.max_abs_residual
            if crooks is not None and math.isfinite(crooks.max_abs_residual)
            else None
        )
        tur_relative_variance: float | None = None
        tur_entropy_bound: float | None = None
        tur_bound_ratio: float | None = None
        if retained >= 2:
            mean_current_value = float(np.mean(currents))
            mean_sigma = float(np.mean(sigmas))
            if mean_current_value != 0.0 and mean_sigma > 0.0:
                tur_relative_variance = float(np.var(currents)) / (
                    mean_current_value * mean_current_value
                )
                tur_entropy_bound = 2.0 / mean_sigma
                tur_bound_ratio = tur_relative_variance / tur_entropy_bound
        finite_values = [
            tested_fraction,
            crooks_residual,
            tur_relative_variance,
            tur_entropy_bound,
            tur_bound_ratio,
        ]
        finite = all(value is not None and math.isfinite(value) for value in finite_values)
        reasons: list[str] = []
        sample_count = len(self._sample_ids.get(key, set()))
        if sample_count < self.policy.min_samples:
            reasons.append("under_sampled")
        if tested == 0:
            reasons.append("no_tested_events")
        if not self._sampling_modes.get(key, set()) <= {
            "straight_through",
            "gumbel_sigmoid_ste",
        }:
            reasons.append("unsupported_sampling_mode")
        if tested_fraction < self.policy.min_tested_fraction:
            reasons.append("under_covered")
        symmetric_bins = 0 if crooks is None else int(crooks.bins.size)
        if symmetric_bins < self.policy.min_symmetric_bins:
            reasons.append("insufficient_symmetric_support")
        if crooks_residual is None or crooks_residual > self.policy.crooks_tolerance:
            reasons.append("crooks_gate_failed")
        if tur_bound_ratio is None or tur_bound_ratio < self.policy.min_tur_bound_ratio:
            reasons.append("tur_gate_failed")
        if not finite:
            reasons.append("non_finite_evidence")
        return HeadPredictiveThermoEvidence(
            layer_address=key[0],
            head_index=key[1],
            sampling_modes=tuple(sorted(self._sampling_modes.get(key, set()))),
            sample_count=sample_count,
            observed_events=observed,
            tested_events=tested,
            retained_events=retained,
            degenerate_events=degenerate,
            tested_fraction=tested_fraction,
            symmetric_bins=symmetric_bins,
            crooks_residual=crooks_residual,
            tur_relative_variance=tur_relative_variance,
            tur_entropy_bound=tur_entropy_bound,
            tur_bound_ratio=tur_bound_ratio,
            finite=finite,
            passed=not reasons,
            refusal_reasons=tuple(reasons),
        )


def _to_numpy(value):
    """Detach Torch-like tensors while keeping this reference module Torch-optional."""
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy = getattr(value, "numpy", None)
    if not callable(numpy):
        return value
    try:
        return numpy()
    except TypeError:
        # NumPy has no native bfloat16 dtype. Evidence is statistical float64 downstream, so an
        # explicit float32 bridge preserves all representable bfloat16 values without ambiguity.
        to_float = getattr(value, "float", None)
        if not callable(to_float):
            raise
        return to_float().numpy()


# =========================================================================== #
# Runtime TUR certificate + Crooks calibration monitor (bead 0642.3.2.1)
# =========================================================================== #
#
# Turns the theory into observable per-head/per-step evidence (the CuspMonitor / LyapunovMonitor
# discipline). Each `record` ingests a batch of release-current samples `J` (from the MC ensemble,
# `mc_ensemble`, or any live stochastic-release readout) with their jump propensities, certifies the
# TUR (the one-sided, always-computable precision/cost bound), and accumulates the entropy production
# for the Crooks detailed-FT residual (the calibration guarantee — computable only where the `Σ`
# histogram has support on both signs, i.e. near equilibrium; flagged otherwise). Emits rich + JSONL.


@dataclass
class ThermoStepRecord:
    """Auditable per-step stochastic-thermodynamics record (the runtime UQ-certificate evidence)."""

    step: int
    n_samples: int
    affinity: float            # A = ln(a/b)
    mean_current: float        # ⟨J⟩
    relative_variance: float   # ε² = Var(J)/⟨J⟩² (the precision)
    mean_entropy: float        # ⟨Σ⟩
    entropy_bound: float       # 2/⟨Σ⟩ (the TUR floor on ε²)
    tur_satisfied: bool        # ε² ≥ 2/⟨Σ⟩
    tur_slack: float           # ε² − 2/⟨Σ⟩ ≥ 0


class StochasticThermoMonitor:
    """Runtime TUR certificate + Crooks calibration monitor for the stochastic release (`0642.3.2.1`).

    Accumulates release-current batches; per batch it emits a `ThermoStepRecord` with the (non-vacuous)
    TUR certificate, and accumulates the entropy production for the Crooks detailed-FT residual. Audit
    surface mirrors `CuspMonitor`: predicates, `summary()`, `to_jsonl()`, `render()`.
    """

    def __init__(self) -> None:
        self.records: list[ThermoStepRecord] = []
        self._sigma: list[np.ndarray] = []

    def record(self, currents: np.ndarray, rates: ReleaseRates, *, step: int | None = None) -> ThermoStepRecord:
        """Ingest one batch of release currents `J` at jump propensities `rates`; certify the TUR."""
        c = np.asarray(currents, dtype=np.float64)
        if c.size == 0:
            raise ValueError("record needs a non-empty batch of currents")
        a = affinity(rates)
        sig = entropy_production_samples(c, rates)
        self._sigma.append(sig)
        mean_j = float(c.mean())
        mean_sig = float(sig.mean())
        # The empirical TUR is only meaningful with >= 2 samples AND non-zero spread: a degenerate
        # batch (n<2 or Var=0) gives an unreliable variance that would read as a spurious "violation"
        # of a theorem, so treat it as non-informative (satisfied, bound=∞) rather than a violation.
        if c.size >= 2 and float(np.var(c)) > 0.0 and mean_j != 0.0 and mean_sig > 0.0:
            cert = empirical_tur(c, mean_sig)
            rel_var, bound, satisfied, slack = (
                cert.relative_variance, cert.entropy_bound, cert.satisfied, cert.slack,
            )
        else:  # undriven / degenerate batch: the TUR is not testable here
            rel_var = float(np.var(c)) / (mean_j * mean_j) if mean_j != 0.0 else float("inf")
            bound, satisfied, slack = float("inf"), True, float("inf")
        rec = ThermoStepRecord(
            step=step if step is not None else len(self.records),
            n_samples=int(c.size), affinity=a, mean_current=mean_j,
            relative_variance=rel_var, mean_entropy=mean_sig, entropy_bound=bound,
            tur_satisfied=bool(satisfied), tur_slack=slack,
        )
        self.records.append(rec)
        return rec

    # -- audit predicates ----------------------------------------------------- #
    def all_currents_satisfy_tur(self) -> bool:
        """Every recorded batch honored the TUR (a theorem, so this must hold — a self-consistency check)."""
        return all(r.tur_satisfied for r in self.records)

    def crooks_calibration(self, **kw) -> CrooksCalibration:
        """The Crooks detailed-FT calibration over ALL accumulated entropy production."""
        if not self._sigma:
            return CrooksCalibration(np.array([]), np.array([]), np.array([]), float("inf"), False)
        return crooks_calibration(np.concatenate(self._sigma), **kw)

    def ft_residual(self, **kw) -> float:
        """The tracked fluctuation-theorem residual `sup|ln(P(+Σ)/P(−Σ)) − Σ|` (nan if no symmetric support)."""
        return self.crooks_calibration(**kw).max_abs_residual

    def assert_tur(self) -> None:
        if not self.all_currents_satisfy_tur():
            bad = next(r for r in self.records if not r.tur_satisfied)
            raise AssertionError(
                f"TUR violated at step {bad.step}: ε²={bad.relative_variance:.4g} < 2/⟨Σ⟩={bad.entropy_bound:.4g}"
            )

    def summary(self) -> dict:
        if not self.records:
            return {"steps": 0}
        finite = [r for r in self.records if math.isfinite(r.entropy_bound)]
        cal = self.crooks_calibration()
        return {
            "steps": len(self.records),
            "tur_all_satisfied": self.all_currents_satisfy_tur(),
            "mean_relative_variance": float(np.mean([r.relative_variance for r in finite])) if finite else float("nan"),
            "mean_entropy_bound": float(np.mean([r.entropy_bound for r in finite])) if finite else float("nan"),
            "min_tur_slack": min((r.tur_slack for r in finite), default=float("nan")),
            "mean_affinity": float(np.mean([r.affinity for r in self.records])),
            "ft_residual": cal.max_abs_residual,
            "ft_calibrated": cal.calibrated,
            "ft_bins": int(cal.bins.size),
        }

    def to_jsonl(self) -> list[str]:
        """Per-step records as JSONL lines (the eqyk.2 detailed-logging artifact)."""
        return [json.dumps(asdict(r), ensure_ascii=False) for r in self.records]

    def render(self, console=None) -> None:
        """Rich summary of the monitor (falls back to plain print without rich)."""
        s = self.summary()
        try:
            from rich.console import Console
            from rich.table import Table
            console = console or Console()
            t = Table(title="Stochastic-thermo monitor (TUR certificate + Crooks FT)")
            t.add_column("metric")
            t.add_column("value", justify="right")
            for k, v in s.items():
                t.add_row(k, f"{v:.5g}" if isinstance(v, float) else str(v))
            console.print(t)
        except Exception:  # pragma: no cover - rich is a project dep; stay usable without it
            print("stochastic-thermo monitor summary:", s)


# =========================================================================== #
# Energy-optimal temperature schedule + toggle + deterministic fallback (bead 0642.3.2.2)
# =========================================================================== #
@dataclass(frozen=True)
class ThermoUQConfig:
    """Toggle + knobs for thermodynamically-calibrated UQ (`thermo_uq`, `hm4.7`). Default-off."""

    enabled: bool = False              # the master toggle; off ⟹ a neutral temperature and no claim
    ft_tol: float = 0.25               # Crooks FT residual tolerance for the analytic-calibration claim
    drive_uncertainty_base: float = 1.0  # σ_base for the Landauer schedule
    ach_gain: float = 1.0              # how strongly ACh scales the effective uncertainty


@dataclass(frozen=True)
class CalibrationVerdict:
    """The deterministic calibration verdict: analytic FT guarantee, or the empirical-ECE fallback."""

    calibrated: bool          # the FT test passed within tolerance (the analytic guarantee holds)
    ft_residual: float        # the measured Crooks residual (nan if no symmetric support)
    bins: int                 # number of populated symmetric FT bins
    mode: str                 # "analytic_fluctuation_theorem" | "empirical_ece_fallback"
    reason: str


class ThermoUQController:
    """Energy-optimal release-temperature schedule + the fail-closed calibration gate (`0642.3.2.2`).

    Produces the Landauer-optimal release temperature as a function of the ACh-signaled uncertainty
    (the schedule), and gates the *analytic calibration claim* on the empirical fluctuation-theorem
    test: if the release `Σ` histogram obeys the Crooks identity within `ft_tol`, the predictive
    distribution carries the analytic FT-calibration guarantee; otherwise (the non-Markov /
    rate-misspecified regime, ledger E1/E3/R) the controller **deterministically falls back** to
    reporting empirical ECE only and flags the drop. Default-off (`enabled=False`) ⟹ a neutral
    temperature (1.0) and no calibration claim — the baseline path.
    """

    def __init__(self, cfg: ThermoUQConfig | None = None) -> None:
        self.cfg = cfg or ThermoUQConfig()

    def optimal_temperature(self, ach_level: float = 0.0) -> float:
        """The energy-optimal (Landauer) release temperature for the current ACh/uncertainty signal.

        Neutral (`1.0`) when disabled; otherwise `kT*(ACh) = σ(ACh)/√SNR*` (the `0642.3.1.4` law).
        """
        if not self.cfg.enabled:
            return 1.0
        return ach_coupled_temperature(
            self.cfg.drive_uncertainty_base, ach_level, ach_gain=self.cfg.ach_gain
        )

    def temperature_schedule(self, ach_levels) -> list[float]:
        """The optimal release temperature over a sequence of per-step ACh/uncertainty signals."""
        return [self.optimal_temperature(float(a)) for a in ach_levels]

    def calibration_verdict(self, sigma: np.ndarray, **crooks_kw) -> CalibrationVerdict:
        """Gate the analytic-calibration claim on the empirical FT test (the deterministic fallback).

        `sigma` is the accumulated entropy production of the predictive ensemble (e.g. from a
        `StochasticThermoMonitor` or the MC release readout). Passes the Crooks detailed-FT test ⟹
        analytic guarantee; fails (or no symmetric support) ⟹ the empirical-ECE fallback, flagged.
        """
        cal = crooks_calibration(np.asarray(sigma, dtype=np.float64), tol=self.cfg.ft_tol, **crooks_kw)
        if cal.calibrated:
            return CalibrationVerdict(
                calibrated=True, ft_residual=cal.max_abs_residual, bins=int(cal.bins.size),
                mode="analytic_fluctuation_theorem",
                reason=(f"FT test passed (residual={cal.max_abs_residual:.3g} ≤ {self.cfg.ft_tol:g} on "
                        f"{cal.bins.size} bins): predictive distribution is FT-calibrated"),
            )
        resid = cal.max_abs_residual
        why = ("no symmetric Σ support (insufficient near-equilibrium data)" if not math.isfinite(resid)
               else f"FT residual {resid:.3g} > {self.cfg.ft_tol:g} (non-Markov / rate-misspecified)")
        return CalibrationVerdict(
            calibrated=False, ft_residual=resid, bins=int(cal.bins.size),
            mode="empirical_ece_fallback",
            reason=f"FT test failed — {why}; dropping the analytic claim, report empirical ECE only",
        )

    def assess(self, currents: np.ndarray, rates: ReleaseRates, *, ach_level: float = 0.0,
               **crooks_kw) -> dict:
        """One-shot: the optimal temperature for `ach_level` + the calibration verdict for `currents`."""
        sigma = entropy_production_samples(np.asarray(currents, dtype=np.float64), rates)
        verdict = self.calibration_verdict(sigma, **crooks_kw)
        return {
            "enabled": self.cfg.enabled,
            "optimal_temperature": self.optimal_temperature(ach_level),
            "calibration_mode": verdict.mode,
            "ft_calibrated": verdict.calibrated,
            "ft_residual": verdict.ft_residual,
            "reason": verdict.reason,
        }
