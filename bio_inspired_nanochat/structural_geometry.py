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

import json
import math
from dataclasses import asdict, dataclass

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
    n = int(np.asarray(points).shape[0])
    if e.size == 0:
        return CoverageSignal(0.0, 0.0, 0.0, n, False)
    max_gap = float(e[-1])
    typical = float(np.median(e))
    # A genuinely zero median (e.g. a tight cluster of (near-)duplicate points plus one far outlier)
    # makes the ratio undefined; report it as ∞ honestly — there IS an isolated hole — rather than a
    # meaningless ~1e14 from dividing by an epsilon floor.
    ratio = (max_gap / typical) if typical > 0.0 else (float("inf") if max_gap > 0.0 else 0.0)
    return CoverageSignal(
        max_gap=max_gap, typical_gap=typical, persistence_ratio=ratio,
        n_points=n, significant=bool(ratio >= ratio_threshold),
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
    comparator_available: bool  # equal-sized samples make the elementwise naive comparator well-defined
    transport_optimal: bool     # the quantile plan costs no more than the naive elementwise merge
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
    falls back to the barycenter itself for cost accounting. In that no-contrast case
    ``comparator_available`` and ``transport_optimal`` are both false, so the runtime monitor fails
    closed rather than treating equality-by-construction as evidence.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == b.size and a.size > 0:
        # Equal-cardinality empirical measures have an exact monotone coupling:
        # pair every order statistic. This is also the operation installed by
        # the runtime merge, so its threshold and audit record are literal.
        sorted_a = np.sort(a, kind="stable")
        sorted_b = np.sort(b, kind="stable")
        bary = 0.5 * (sorted_a + sorted_b)
        naive_vals = 0.5 * (a + b)
        sorted_naive = np.sort(naive_vals, kind="stable")
        transport = 0.5 * (
            float(np.mean((bary - sorted_a) ** 2))
            + float(np.mean((bary - sorted_b) ** 2))
        )
        naive_cost = 0.5 * (
            float(np.mean((sorted_naive - sorted_a) ** 2))
            + float(np.mean((sorted_naive - sorted_b) ** 2))
        )
        return MergeCertificate(
            transport_cost=transport,
            naive_cost=naive_cost,
            barycenter_std=float(np.std(bary)),
            naive_std=float(np.std(naive_vals)),
            comparator_available=True,
            transport_optimal=bool(transport <= naive_cost + 1e-12),
            ot_preserves_spread=bool(
                np.std(bary) >= np.std(naive_vals) - 1e-12
            ),
        )
    bary = wasserstein_barycenter_1d(a, b, t=0.5)
    comparator_available = False
    naive_vals = bary
    transport = 0.5 * (wasserstein_1d(bary, a) ** 2 + wasserstein_1d(bary, b) ** 2)
    naive_cost = 0.5 * (wasserstein_1d(naive_vals, a) ** 2 + wasserstein_1d(naive_vals, b) ** 2)
    return MergeCertificate(
        transport_cost=transport, naive_cost=naive_cost,
        barycenter_std=float(np.std(bary)), naive_std=float(np.std(naive_vals)),
        comparator_available=comparator_available,
        transport_optimal=bool(comparator_available and transport <= naive_cost + 1e-12),
        ot_preserves_spread=bool(np.std(bary) >= np.std(naive_vals) - 1e-12),
    )


# =========================================================================== #
# Runtime certificates + bounded routing-manifold monitor (bead 0642.5.2.1)
# =========================================================================== #


@dataclass(frozen=True)
class StructuralGeometryMonitorConfig:
    """Cost and significance bounds for :class:`StructuralGeometryMonitor`.

    H0 persistence uses an ``O(n² d)`` minimum-spanning-tree calculation. ``max_points`` and
    ``max_dim`` therefore cap the only data-dependent quadratic work. Rows are sampled at evenly
    spaced indices and dimensions are selected by variance, both deterministically.
    """

    persistence_ratio_threshold: float = 3.0
    max_points: int = 256
    max_dim: int = 8
    max_persistence_features: int = 8

    def __post_init__(self) -> None:
        if not math.isfinite(self.persistence_ratio_threshold) or self.persistence_ratio_threshold <= 0.0:
            raise ValueError("persistence_ratio_threshold must be finite and positive")
        if self.max_points < 2:
            raise ValueError("max_points must be >= 2")
        if self.max_dim < 1:
            raise ValueError("max_dim must be >= 1")
        if self.max_persistence_features < 1:
            raise ValueError("max_persistence_features must be >= 1")


@dataclass(frozen=True)
class StructuralGeometryRecord:
    """One JSON-safe structural decision record.

    ``None`` denotes an unbounded quantity (for example a singular parent's ``kappa``), avoiding
    non-standard JSON ``Infinity`` values while the boolean certificate fields retain the verdict.
    """

    step: int
    routing_points_input: int
    routing_dim_input: int
    routing_points_used: int
    routing_dim_used: int
    routing_was_capped: bool
    homology_dimension: int
    kappa_parent: float | None
    kappa_bound: float | None
    split_noise_norm: float
    split_well_conditioned: bool
    max_persistence: float
    typical_persistence: float
    persistence_ratio: float | None
    persistence_significant: bool
    top_persistence_features: tuple[float, ...]
    merge_transport_cost: float
    merge_naive_cost: float
    merge_cost_saving: float
    merge_comparator_available: bool
    merge_transport_optimal: bool
    merge_preserves_spread: bool


class StructuralGeometryMonitor:
    """Bounded runtime monitor for split conditioning, routing coverage, and OT merges.

    The three signals are recorded together because they gate one structural-lifecycle decision:
    the proposed child must have a finite Weyl condition-number bound, the routing manifold may
    request growth only for a thresholded persistent H0 gap, and a proposed merge must use a
    no-more-expensive quantile transport plan. Records are directly consumable by
    ``run_logging.RunLogger`` or emitted as standalone JSONL.
    """

    def __init__(self, cfg: StructuralGeometryMonitorConfig | None = None) -> None:
        self.cfg = cfg or StructuralGeometryMonitorConfig()
        self.records: list[StructuralGeometryRecord] = []

    def _bounded_routing_points(self, points: np.ndarray) -> tuple[np.ndarray, int, int]:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2:
            raise ValueError(f"routing_points must be a 2D array, got shape {pts.shape}")
        n_input, d_input = (int(pts.shape[0]), int(pts.shape[1]))
        if n_input < 2 or d_input < 1:
            raise ValueError(
                f"routing_points need >=2 rows and >=1 dimension, got shape {pts.shape}"
            )
        if not np.isfinite(pts).all():
            raise ValueError("routing_points must contain only finite values")

        if d_input > self.cfg.max_dim:
            # Stable sort makes tied-variance dimensions deterministic. Sorting the selected indices
            # restores source-column order after choosing the most informative dimensions.
            variances = np.var(pts, axis=0)
            dims = np.sort(np.argsort(-variances, kind="stable")[: self.cfg.max_dim])
            pts = pts[:, dims]
        if n_input > self.cfg.max_points:
            rows = np.linspace(0, n_input - 1, self.cfg.max_points, dtype=np.int64)
            pts = pts[rows]
        return pts, n_input, d_input

    @staticmethod
    def _finite_or_none(value: float) -> float | None:
        return float(value) if math.isfinite(value) else None

    def record(
        self,
        *,
        step: int,
        parent_weight: np.ndarray | None,
        split_noise_norm: float,
        routing_points: np.ndarray,
        merge_a: np.ndarray,
        merge_b: np.ndarray,
        split_certificate: SpectralCertificate | None = None,
        merge_certificate: MergeCertificate | None = None,
    ) -> StructuralGeometryRecord:
        """Compute and store all three certificates for one lifecycle decision."""
        bounded, n_input, d_input = self._bounded_routing_points(routing_points)
        if not math.isfinite(split_noise_norm):
            raise ValueError("split_noise_norm must be finite")
        if split_certificate is None:
            if parent_weight is None:
                raise ValueError(
                    "parent_weight is required when split_certificate is not supplied"
                )
            parent = np.asarray(parent_weight, dtype=np.float64)
            if parent.ndim != 2 or min(parent.shape, default=0) < 1:
                raise ValueError(
                    "parent_weight must be a non-empty 2D array, "
                    f"got shape {parent.shape}"
                )
            if not np.isfinite(parent).all():
                raise ValueError("parent_weight must contain only finite values")
            split = spectral_conditioning_certificate(parent, split_noise_norm)
        else:
            split = split_certificate
            if not all(
                math.isfinite(value)
                for value in (split.sigma_max, split.sigma_min, split.noise_norm)
            ):
                raise ValueError("split_certificate spectrum and noise must be finite")
            if split.sigma_max < 0.0 or split.sigma_min < 0.0 or split.noise_norm < 0.0:
                raise ValueError("split_certificate spectrum and noise must be non-negative")
            if not math.isclose(
                split.noise_norm,
                split_noise_norm,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "split_certificate noise does not match split_noise_norm"
                )
        merge_a_array = np.asarray(merge_a, dtype=np.float64)
        merge_b_array = np.asarray(merge_b, dtype=np.float64)
        if merge_a_array.size == 0 or merge_b_array.size == 0:
            raise ValueError("merge samples must be non-empty")
        if not np.isfinite(merge_a_array).all() or not np.isfinite(merge_b_array).all():
            raise ValueError("merge samples must contain only finite values")

        edges = mst_edge_lengths(bounded)
        max_gap = float(edges[-1]) if edges.size else 0.0
        typical = float(np.median(edges)) if edges.size else 0.0
        ratio = max_gap / typical if typical > 0.0 else (math.inf if max_gap > 0.0 else 0.0)
        significant = bool(ratio >= self.cfg.persistence_ratio_threshold)
        top = tuple(float(x) for x in edges[::-1][: self.cfg.max_persistence_features])

        merge = (
            merge_certificate
            if merge_certificate is not None
            else ot_merge_certificate(merge_a_array, merge_b_array)
        )
        rec = StructuralGeometryRecord(
            step=int(step),
            routing_points_input=n_input,
            routing_dim_input=d_input,
            routing_points_used=int(bounded.shape[0]),
            routing_dim_used=int(bounded.shape[1]),
            routing_was_capped=(bounded.shape != (n_input, d_input)),
            homology_dimension=0,
            kappa_parent=self._finite_or_none(split.kappa_parent),
            kappa_bound=self._finite_or_none(split.kappa_bound),
            split_noise_norm=float(split.noise_norm),
            split_well_conditioned=split.well_conditioned,
            max_persistence=max_gap,
            typical_persistence=typical,
            persistence_ratio=self._finite_or_none(ratio),
            persistence_significant=significant,
            top_persistence_features=top,
            merge_transport_cost=merge.transport_cost,
            merge_naive_cost=merge.naive_cost,
            merge_cost_saving=merge.naive_cost - merge.transport_cost,
            merge_comparator_available=merge.comparator_available,
            merge_transport_optimal=merge.transport_optimal,
            merge_preserves_spread=merge.ot_preserves_spread,
        )
        self.records.append(rec)
        return rec

    def all_births_well_conditioned(self) -> bool:
        return all(r.split_well_conditioned for r in self.records)

    def all_merge_plans_optimal(self) -> bool:
        return all(r.merge_transport_optimal for r in self.records)

    def assert_certificates(self) -> None:
        """Fail closed when a proposed split or merge lacks its required certificate."""
        if not self.records:
            raise AssertionError("no structural geometry records were observed")
        if not self.all_births_well_conditioned():
            bad = next(r for r in self.records if not r.split_well_conditioned)
            raise AssertionError(
                f"split conditioning failed at step {bad.step}: noise={bad.split_noise_norm:g}, "
                f"kappa_bound={bad.kappa_bound}"
            )
        if not self.all_merge_plans_optimal():
            bad = next(r for r in self.records if not r.merge_transport_optimal)
            raise AssertionError(
                f"OT merge certificate failed at step {bad.step}: "
                f"transport_cost={bad.merge_transport_cost:g} > naive_cost={bad.merge_naive_cost:g}"
            )

    def summary(self) -> dict:
        bounded_work = {
            "homology_dimension": 0,
            "routing_point_cap": self.cfg.max_points,
            "routing_dimension_cap": self.cfg.max_dim,
            "persistence_feature_cap": self.cfg.max_persistence_features,
        }
        if not self.records:
            return {"steps": 0, **bounded_work}
        finite_kappa = [r.kappa_bound for r in self.records if r.kappa_bound is not None]
        return {
            "steps": len(self.records),
            **bounded_work,
            "births_well_conditioned": self.all_births_well_conditioned(),
            "merge_plans_optimal": self.all_merge_plans_optimal(),
            "max_kappa_bound": max(finite_kappa) if finite_kappa else None,
            "max_persistence": max(r.max_persistence for r in self.records),
            "significant_persistence_steps": sum(r.persistence_significant for r in self.records),
            "mean_merge_cost": float(np.mean([r.merge_transport_cost for r in self.records])),
            "capped_steps": sum(r.routing_was_capped for r in self.records),
        }

    def to_jsonl(self) -> list[str]:
        """Return standard-compliant JSONL records for the structured run log."""
        return [json.dumps(asdict(r), ensure_ascii=False, allow_nan=False) for r in self.records]

    def render(self, console=None) -> None:
        """Render the current certificate summary with Rich."""
        from rich.console import Console
        from rich.table import Table

        console = console or Console()
        table = Table(title="Structural geometry certificates (spectral / H0 / OT)")
        table.add_column("metric")
        table.add_column("value", justify="right")
        for key, value in self.summary().items():
            table.add_row(key, f"{value:.5g}" if isinstance(value, float) else str(value))
        console.print(table)
