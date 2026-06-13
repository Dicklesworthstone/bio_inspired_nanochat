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
    """
    a, b, t = rates.a, rates.b, steps
    return math.exp(a * t * (b / a - 1.0)) * math.exp(b * t * (a / b - 1.0))


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

    `ε² = Var(J)/⟨J⟩² = (a+b)/((a−b)²·t)` is lower-bounded by `2/⟨Σ⟩ = 2/((a−b)·t·ln(a/b))`; the TUR
    `(a+b)·ln(a/b) ≥ 2(a−b)` is a theorem for all `a,b>0`, so `satisfied` is always True and `slack`
    is the precision the head can actually buy beyond the thermodynamic floor.
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
    evaluates online to certify that a head's release precision is thermodynamically honest.
    """
    c = np.asarray(currents, dtype=np.float64)
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
        if c > 0 and counts[i] >= min_count and counts[j] >= min_count:
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
        a = affinity(rates)
        sig = entropy_production_samples(c, rates)
        self._sigma.append(sig)
        mean_j = float(c.mean())
        mean_sig = float(sig.mean())
        if mean_j != 0.0 and mean_sig > 0.0:
            cert = empirical_tur(c, mean_sig)
            rel_var, bound, satisfied, slack = (
                cert.relative_variance, cert.entropy_bound, cert.satisfied, cert.slack,
            )
        else:  # undriven / degenerate batch: the TUR is not informative here
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
