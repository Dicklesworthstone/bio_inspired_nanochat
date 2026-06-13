"""Free-probability + persistent-homology + optimal-transport structural plasticity (Thrust C, `0642.5.1`).

Replaces the heuristic `health = fatigue × energy` thresholds of the MoE expert lifecycle (`uta`) with
three *principled* geometric signals, each playing a distinct role:

  - **FREE PROBABILITY (birth conditioning):** splitting/cloning an expert `W` with antisymmetric noise
    `±δN` perturbs its singular spectrum; the singular values move by at most `‖δN‖` (Weyl), so the
    child condition number `κ` is **controllable by construction** — a spectral-conditioning
    certificate bounds `κ` and gives the largest noise scale that keeps a target `κ` (§1, `0642.5.1.1`).
  - **PERSISTENT HOMOLOGY (growth trigger):** the routing point cloud has a shape; its `H0` persistence
    (the gaps in its minimum spanning tree) reveals regions of input space with no expert coverage. A
    high-persistence gap — *stable* under perturbation by the bottleneck-stability theorem — is a
    principled signal to GROW capacity there (§2, `0642.5.1.2`).
  - **OPTIMAL TRANSPORT (merge):** merging two experts is the **Wasserstein barycenter** of their weight
    distributions (the geodesic midpoint), which preserves the marginal shape — unlike naive averaging,
    which collapses spread (§3, `0642.5.1.3`).

All three are pure-numpy (free convolution via the Weyl/RMT bound, `H0` via the MST, the 1D `W2`
barycenter via quantile interpolation — no SciPy/ripser). This is the theory + reference math; the
runtime certificates/monitors (`0642.5.2.1`) and the falsification vs the `uta` heuristic lifecycle
(`0642.5.3`) build on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# =========================================================================== #
# §1. Free-probability spectral conditioning of an expert split (bead 0642.5.1.1)
# =========================================================================== #
def condition_number(w: np.ndarray) -> float:
    """The spectral condition number `κ(W) = σ_max / σ_min` (∞ if singular)."""
    s = np.linalg.svd(np.asarray(w, dtype=np.float64), compute_uv=False)
    s_min = float(s[-1])
    return float(s[0]) / s_min if s_min > 0 else math.inf


@dataclass(frozen=True)
class SpectralCertificate:
    """The birth-conditioning certificate for a noisy expert split (Weyl singular-value bound)."""

    sigma_max: float        # σ_max(W)
    sigma_min: float        # σ_min(W)
    noise_norm: float       # ‖δN‖_2 (the perturbation spectral norm)
    kappa_parent: float     # κ(W)
    kappa_bound: float      # certified upper bound on κ(W ± δN) (∞ if the noise can zero a sing. value)
    well_conditioned: bool  # noise_norm < σ_min ⟹ the child stays full-rank with a finite κ bound


def spectral_conditioning_certificate(w: np.ndarray, noise_norm: float) -> SpectralCertificate:
    """Certify the child condition number after a split with perturbation of spectral norm `noise_norm`.

    By Weyl's inequality every singular value moves by at most `‖δN‖`, so
    `κ(W ± δN) ≤ (σ_max + ‖δN‖)/(σ_min − ‖δN‖)` whenever `‖δN‖ < σ_min` (else the child may be singular).
    This is the free-probability *birth conditioning* in its rigorous, always-valid form — free
    convolution sharpens the bulk prediction, but the Weyl envelope is what the certificate guarantees.
    """
    if noise_norm < 0:
        raise ValueError(f"noise_norm must be ≥ 0, got {noise_norm}")
    s = np.linalg.svd(np.asarray(w, dtype=np.float64), compute_uv=False)
    s_max, s_min = float(s[0]), float(s[-1])
    well = noise_norm < s_min
    kappa_bound = (s_max + noise_norm) / (s_min - noise_norm) if well else math.inf
    return SpectralCertificate(
        sigma_max=s_max, sigma_min=s_min, noise_norm=noise_norm,
        kappa_parent=(s_max / s_min if s_min > 0 else math.inf),
        kappa_bound=kappa_bound, well_conditioned=well,
    )


def max_noise_for_kappa(w: np.ndarray, kappa_target: float) -> float:
    """The largest split-noise spectral norm `‖δN‖` that still certifies `κ(child) ≤ kappa_target`.

    Solving `(σ_max + x)/(σ_min − x) ≤ κ_t` for `x` gives `x ≤ (κ_t·σ_min − σ_max)/(κ_t + 1)`
    (0 if even a zero-noise split already exceeds `κ_t`).
    """
    if kappa_target <= 1.0:
        raise ValueError(f"kappa_target must be > 1, got {kappa_target}")
    s = np.linalg.svd(np.asarray(w, dtype=np.float64), compute_uv=False)
    s_max, s_min = float(s[0]), float(s[-1])
    x = (kappa_target * s_min - s_max) / (kappa_target + 1.0)
    return max(0.0, x)


def function_preserving_split(w: np.ndarray, noise_norm: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A spectrally-controlled, antisymmetric split: `(W + δN, W − δN)` with `‖δN‖_2 = noise_norm`.

    The pair averages back to `W` (output-preserving in the dense regime, as in `uta.3`), while the
    antisymmetric noise lets the twins diverge under SGD — now with a *certified* child `κ` (the noise
    norm is set, not ad hoc). `δN` is a random matrix rescaled to the exact target spectral norm.
    """
    w = np.asarray(w, dtype=np.float64)
    n = rng.standard_normal(w.shape)
    sn = float(np.linalg.svd(n, compute_uv=False)[0])
    if sn > 0:
        n *= noise_norm / sn
    return w + n, w - n


# =========================================================================== #
# §2. Persistent-homology coverage signal (bead 0642.5.1.2)
# =========================================================================== #
def mst_edge_lengths(points: np.ndarray) -> np.ndarray:
    """Sorted edge lengths of the Euclidean minimum spanning tree (Prim, `O(n²)`).

    These are exactly the `H0` *death* times of the Vietoris–Rips/Čech filtration: each of the `n`
    points is an `H0` feature born at 0, and dies (merges) when an MST edge connects its component.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    if n < 2:
        return np.array([], dtype=np.float64)
    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    best = np.linalg.norm(pts - pts[0], axis=1)
    best[0] = math.inf
    edges = []
    for _ in range(n - 1):
        j = int(np.argmin(best))
        edges.append(float(best[j]))
        in_tree[j] = True
        best[j] = math.inf
        d = np.linalg.norm(pts - pts[j], axis=1)
        upd = (~in_tree) & (d < best)
        best[upd] = d[upd]
    return np.sort(np.array(edges, dtype=np.float64))


@dataclass(frozen=True)
class CoverageSignal:
    """The topology-triggered growth signal: the most persistent `H0` gap in the routing manifold."""

    max_gap: float          # the largest H0 persistence (longest MST edge) — the coverage hole
    typical_gap: float      # the median MST edge length (the bulk spacing scale)
    persistence_ratio: float  # max_gap / typical_gap — significance (>> 1 ⟹ a genuine, stable feature)
    n_points: int
    significant: bool       # persistence_ratio ≥ threshold (bottleneck-stable ⟹ grow here)


def coverage_signal(points: np.ndarray, *, ratio_threshold: float = 3.0) -> CoverageSignal:
    """The persistent-homology growth signal: the largest `H0` gap, scored for significance.

    A large `max_gap` relative to the bulk `typical_gap` is a region of input space the experts do not
    cover (a topological hole). By the **bottleneck-stability theorem** the persistence diagram moves
    by at most the data perturbation, so a gap with `persistence_ratio ≫ 1` is a *genuine* feature
    (robust to noise) — a principled, noise-stable trigger to grow capacity there.
    """
    e = mst_edge_lengths(points)
    if e.size == 0:
        return CoverageSignal(0.0, 0.0, 0.0, int(np.asarray(points).shape[0]), False)
    max_gap = float(e[-1])
    typical = float(np.median(e)) or 1e-12
    ratio = max_gap / typical
    return CoverageSignal(
        max_gap=max_gap, typical_gap=typical, persistence_ratio=ratio,
        n_points=int(np.asarray(points).shape[0]), significant=bool(ratio >= ratio_threshold),
    )


# =========================================================================== #
# §3. Optimal-transport (Wasserstein barycenter) merge (bead 0642.5.1.3)
# =========================================================================== #
def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    """The 1D 2-Wasserstein distance `W2` between two empirical distributions (sorted-quantile form).

    `W2(a,b)² = ∫₀¹ (F_a^{-1}(q) − F_b^{-1}(q))² dq` — for 1D samples this is the RMS difference of the
    sorted values (resampled to a common grid), the optimal-transport cost of monotone rearrangement.
    """
    qa = np.quantile(np.asarray(a, dtype=np.float64), np.linspace(0, 1, 512))
    qb = np.quantile(np.asarray(b, dtype=np.float64), np.linspace(0, 1, 512))
    return float(np.sqrt(np.mean((qa - qb) ** 2)))


def wasserstein_barycenter_1d(a: np.ndarray, b: np.ndarray, *, t: float = 0.5, n_grid: int = 1024) -> np.ndarray:
    """The 1D `W2` barycenter (McCann geodesic) of `a, b` at weight `t`: `(1−t)·F_a^{-1} + t·F_b^{-1}`.

    The barycenter's quantile function is the linear interpolation of the two quantile functions — the
    optimal-transport (function-preserving) merge, distinct from naive value averaging. Returned as the
    barycenter's quantiles on a uniform grid (its inverse-CDF samples).
    """
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"t must be in [0,1], got {t}")
    q = np.linspace(0.0, 1.0, n_grid)
    qa = np.quantile(np.asarray(a, dtype=np.float64), q)
    qb = np.quantile(np.asarray(b, dtype=np.float64), q)
    return (1.0 - t) * qa + t * qb


@dataclass(frozen=True)
class MergeCertificate:
    """The OT-merge certificate: the barycenter is the minimum-transport-cost (function-preserving) merge."""

    transport_cost: float       # the barycenter's weighted W2 cost Σ W2(bary, expert)²  (always ≤ naive_cost)
    naive_cost: float           # the same cost for the naive value-average merge
    barycenter_std: float       # spread of the OT-merged distribution: (1−t)σ_a + tσ_b
    naive_std: float            # spread of the naive average (≈ ½√(σ_a²+σ_b²); collapses when experts differ)
    ot_preserves_spread: bool   # barycenter_std ≥ naive_std — holds in the population limit (see note)


def ot_merge_certificate(a: np.ndarray, b: np.ndarray) -> MergeCertificate:
    """Certify the OT (Wasserstein-barycenter) merge of two experts against the naive value average.

    The W2 barycenter minimizes `½·W2(·,a)² + ½·W2(·,b)²` (the OT-optimal merge — `transport_cost ≤
    naive_cost` holds always), and — being the geodesic midpoint — its spread is `(1−t)σ_a + tσ_b`,
    whereas the naive elementwise average `(a+b)/2` of two same-size samples has spread
    `≈ ½√(σ_a²+σ_b²) ≤ ½(σ_a+σ_b)`, so it *shrinks* the variance. Hence `ot_preserves_spread` holds in
    the population limit; on small finite samples it can occasionally flip (sampling noise in the
    elementwise pairing), so read it as the typical, not a guaranteed, behavior. The naive baseline is
    elementwise (order-dependent); when the two experts have different sizes it is undefined here and
    falls back to the barycenter itself (so `naive_cost == transport_cost`, a no-contrast degenerate case).
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    bary = wasserstein_barycenter_1d(a, b, t=0.5)
    naive_vals = 0.5 * (a + b) if a.size == b.size else bary  # naive elementwise value average
    transport = 0.5 * (wasserstein_1d(bary, a) ** 2 + wasserstein_1d(bary, b) ** 2)
    naive_cost = 0.5 * (wasserstein_1d(naive_vals, a) ** 2 + wasserstein_1d(naive_vals, b) ** 2)
    return MergeCertificate(
        transport_cost=transport, naive_cost=naive_cost,
        barycenter_std=float(np.std(bary)), naive_std=float(np.std(naive_vals)),
        ot_preserves_spread=bool(np.std(bary) >= np.std(naive_vals) - 1e-12),
    )
