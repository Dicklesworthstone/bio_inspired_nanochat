"""Free-energy deliberation + energy-based decoding — live decode-path wiring (bead `r00r.1.2`).

Implements the live per-token wiring of the design note `docs/theory/free_energy_deliberation.md`
(`r00r.1.1`) on top of the reference `deliberate()` / `boltzmann_weights()` API in
`metriplectic_integrator.py`. Per token, the model's synaptic state is mapped to the metriplectic core
`z = (C, B, h)`, relaxed by extra free-energy-minimization steps ("ponder") until it self-consistently
halts (`|ΔF| < eps`) or a compute budget (`max_iters`, the latency bound) is hit. For each token, the
engine also advances a bounded model-top-k set on isolated KV/presynaptic cache branches and relaxes
each continuation. By default its `F_final` is added to model energy
(`-logit + lambda*F_final`). A separately calibrated, leakage-safe candidate readout may instead map
the model-energy prior plus branch-local synaptic statistics to a task energy; it is fitted outside
the decode path and remains an explicit opt-in. Effort/confidence still modulate temperature.

Convergence is guaranteed by Thrust A (the structure-preserving step makes `F` monotonically
non-increasing and bounded below — `docs/theory/metriplectic.md` §5), so the ponder always halts in a
bounded number of steps; a guard trip inside a step deterministically falls back to clamped Euler. The
whole mechanism is **default-off**: with no `DeliberationController` the engine decodes exactly as
before (single-step). Nothing here mutates the model — deliberation only runs the existing descent
longer and reads the result.

**Scope, stated honestly.** The candidate path (`r00r.1.6/1.7`) is one-step energy-guided decoding,
not a learned hidden-state recurrent ponder: it branches only the model's top-k candidates and leaves
the committed cache untouched until a token is selected normally. The raw path scores the
low-dimensional aggregate `z = (mean C, mean B, 0)`. The calibrated path is a linear pairwise-rank
readout over the existing model-energy prior, relaxed-energy trajectory, and branch-local state
moments; it does not alter or train the language model. The work per sequence/token is bounded by
`(candidate_top_k + 1) * max_iters`; `max_iters=0` bypasses both branching and pondering exactly.
Deeper tree search remains downstream `re4e.3` work.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from bio_inspired_nanochat.metriplectic_integrator import (
    TEMP,
    DeliberationResult,
    deliberate,
    free_energy,
)
from bio_inspired_nanochat.torch_imports import torch


@dataclass(frozen=True)
class DifficultyRouterConfig:
    """Calibration knobs for the bounded per-token difficulty signal.

    Entropy is already normalized by ``log(vocab_size)``.  Positive free energy is mapped to
    ``[0, 1)`` by a saturating exponential so an unusually large state cannot produce an unbounded
    compute request.  When free energy is unavailable, the router uses entropy alone rather than
    treating the missing measurement as false confidence.
    """

    entropy_weight: float = 0.75
    free_energy_scale: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.entropy_weight <= 1.0:
            raise ValueError(f"entropy_weight must be in [0, 1], got {self.entropy_weight}")
        if not np.isfinite(self.free_energy_scale) or self.free_energy_scale <= 0.0:
            raise ValueError(f"free_energy_scale must be finite and positive, got {self.free_energy_scale}")


@dataclass(frozen=True)
class TokenDifficulty:
    """One token's auditable uncertainty/free-energy difficulty measurement."""

    entropy_nats: float
    normalized_entropy: float
    free_energy: float | None
    normalized_free_energy: float | None
    score: float


@dataclass(frozen=True)
class ATPDebitRecord:
    """One exact integer debit from a sequence-local ATP account."""

    token_index: int
    action: str
    difficulty_score: float
    requested_units: int
    granted_units: int
    unit_cost_atp: int
    spent_atp: int
    remaining_atp: int


class ATPBudget:
    """Hard per-sequence compute budget in exact, integer ATP accounting units.

    A caller requests some number of homogeneous compute units (layers, experts, deliberation steps,
    or Monte-Carlo samples) and supplies that action's integer unit cost.  The account grants as many
    complete units as it can afford and never goes negative.  Integers are deliberate here: the
    invariant ``spent_atp + remaining_atp == total_atp`` is exact, with no floating-point tolerance.

    Instantiate one account per generated sequence.  The downstream adaptive-compute bead assigns
    concrete costs to its compute levers; this class owns only allocation and accounting.
    """

    def __init__(self, total_atp: int) -> None:
        self.total_atp = self._nonnegative_int("total_atp", total_atp)
        self.remaining_atp = self.total_atp
        self._spent_atp = 0
        self.records: list[ATPDebitRecord] = []

    @staticmethod
    def _nonnegative_int(name: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
        return value

    @property
    def spent_atp(self) -> int:
        return self._spent_atp

    @property
    def exhausted(self) -> bool:
        return self.remaining_atp == 0

    def debit(
        self,
        *,
        token_index: int,
        action: str,
        difficulty_score: float,
        requested_units: int,
        unit_cost_atp: int,
    ) -> ATPDebitRecord:
        """Grant affordable whole units and debit their exact ATP cost in one account operation."""
        token_index = self._nonnegative_int("token_index", token_index)
        requested_units = self._nonnegative_int("requested_units", requested_units)
        unit_cost_atp = self._nonnegative_int("unit_cost_atp", unit_cost_atp)
        if unit_cost_atp == 0:
            raise ValueError("unit_cost_atp must be positive")
        action = action.strip()
        if not action:
            raise ValueError("action must be non-empty")
        if not np.isfinite(difficulty_score) or not 0.0 <= difficulty_score <= 1.0:
            raise ValueError(f"difficulty_score must be finite and in [0, 1], got {difficulty_score}")

        affordable_units = self.remaining_atp // unit_cost_atp
        granted_units = min(requested_units, affordable_units)
        spent_atp = granted_units * unit_cost_atp
        self.remaining_atp -= spent_atp
        self._spent_atp += spent_atp
        record = ATPDebitRecord(
            token_index=token_index,
            action=action,
            difficulty_score=float(difficulty_score),
            requested_units=requested_units,
            granted_units=granted_units,
            unit_cost_atp=unit_cost_atp,
            spent_atp=spent_atp,
            remaining_atp=self.remaining_atp,
        )
        self.records.append(record)
        if self.spent_atp + self.remaining_atp != self.total_atp:
            raise RuntimeError("ATP accounting invariant violated")
        return record

    def to_jsonl(self) -> list[str]:
        """Return strict JSONL records for per-token/action energy telemetry."""
        return [json.dumps(asdict(record), ensure_ascii=False, allow_nan=False) for record in self.records]

    def summary(self) -> dict:
        return {
            "total_atp": self.total_atp,
            "spent_atp": self.spent_atp,
            "remaining_atp": self.remaining_atp,
            "exhausted": self.exhausted,
            "debits": len(self.records),
        }


class DifficultyRouter:
    """Measure token difficulty and convert it to a bounded compute-unit request."""

    def __init__(self, cfg: DifficultyRouterConfig | None = None) -> None:
        self.cfg = cfg or DifficultyRouterConfig()

    def measure(self, logits, *, free_energy_value: float | None = None) -> TokenDifficulty:
        """Combine predictive entropy and optional free energy into a score in ``[0, 1]``.

        ``logits`` must describe exactly one token distribution.  Rejecting batches here avoids
        silently averaging together easy and hard rows; callers route each generated sequence with
        its own ATP account.
        """
        values = torch.as_tensor(logits, dtype=torch.float64)
        if values.ndim != 1 or values.numel() < 2:
            raise ValueError(
                "logits must describe exactly one token distribution with at least two entries, "
                f"got shape {tuple(values.shape)}"
            )
        if not bool(torch.isfinite(values).all()):
            raise ValueError("logits must be finite")

        probabilities = torch.softmax(values, dim=-1)
        entropy = float(-(probabilities * probabilities.clamp_min(torch.finfo(values.dtype).tiny).log()).sum())
        normalized_entropy = min(1.0, max(0.0, entropy / float(np.log(values.numel()))))

        normalized_free_energy: float | None = None
        measured_free_energy: float | None = None
        if free_energy_value is not None:
            measured_free_energy = float(free_energy_value)
            if not np.isfinite(measured_free_energy):
                raise ValueError(f"free_energy_value must be finite, got {free_energy_value}")
            positive_free_energy = max(0.0, measured_free_energy)
            normalized_free_energy = float(-np.expm1(-positive_free_energy / self.cfg.free_energy_scale))

        if normalized_free_energy is None:
            score = normalized_entropy
        else:
            entropy_weight = self.cfg.entropy_weight
            score = entropy_weight * normalized_entropy + (1.0 - entropy_weight) * normalized_free_energy
        score = min(1.0, max(0.0, float(score)))
        return TokenDifficulty(
            entropy_nats=entropy,
            normalized_entropy=normalized_entropy,
            free_energy=measured_free_energy,
            normalized_free_energy=normalized_free_energy,
            score=score,
        )

    @staticmethod
    def requested_units(difficulty: TokenDifficulty, *, min_units: int, max_units: int) -> int:
        """Interpolate difficulty into an inclusive integer compute range, deterministically."""
        min_units = ATPBudget._nonnegative_int("min_units", min_units)
        max_units = ATPBudget._nonnegative_int("max_units", max_units)
        if min_units > max_units:
            raise ValueError(f"min_units must not exceed max_units, got {min_units} > {max_units}")
        span = max_units - min_units
        return min_units + int(np.floor(difficulty.score * span + 0.5))

    def route(
        self,
        budget: ATPBudget,
        *,
        token_index: int,
        action: str,
        difficulty: TokenDifficulty,
        min_units: int,
        max_units: int,
        unit_cost_atp: int,
    ) -> ATPDebitRecord:
        """Request difficulty-proportional compute, capped by the sequence's remaining ATP."""
        requested_units = self.requested_units(difficulty, min_units=min_units, max_units=max_units)
        return budget.debit(
            token_index=token_index,
            action=action,
            difficulty_score=difficulty.score,
            requested_units=requested_units,
            unit_cost_atp=unit_cost_atp,
        )


@dataclass(frozen=True)
class DeliberationConfig:
    """The compute-vs-quality knobs for per-token deliberation (default-off; see design §5)."""

    enabled: bool = False
    eps: float = 1e-4          # halting threshold on |ΔF| (smaller ⟹ deliberate longer)
    max_iters: int = 64        # per-token compute budget (the worst-case latency bound)
    dt: float = 0.5            # deliberation step size; tuned so typical tokens halt in ~25–55 steps
                               # (effort scales with calcium/difficulty without saturating the budget)
    T: float = TEMP            # free-energy temperature in F = E − T·S
    # Adaptive decode temperature: easy (low-effort) tokens sharpen toward `temp_floor` (commit),
    # hard (budget-hitting) tokens widen toward `temp_ceil` (explore). Bounds the multiplier on the
    # caller's base temperature, so base_temp=0 (greedy) stays greedy and `enabled=False` is identity.
    temp_floor: float = 0.7
    temp_ceil: float = 1.3
    # r00r.1.6: evaluate a bounded model-top-k set by the relaxed free energy of each
    # continuation.  The model logit is the base energy (-logit); candidate free energy is an
    # additive physical energy term with this coefficient (design note section 4).
    candidate_top_k: int = 8
    candidate_energy_weight: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.candidate_top_k, bool) or not isinstance(self.candidate_top_k, int):
            raise TypeError("candidate_top_k must be a positive integer")
        if self.candidate_top_k < 1:
            raise ValueError("candidate_top_k must be a positive integer")
        if not np.isfinite(self.candidate_energy_weight) or self.candidate_energy_weight < 0.0:
            raise ValueError("candidate_energy_weight must be finite and non-negative")


@dataclass(frozen=True)
class CandidateEnergyBatch:
    """Relaxed free-energy results for a `(batch, candidate)` continuation grid."""

    F_initial: np.ndarray
    F_final: np.ndarray
    effort: np.ndarray
    halted_converged: np.ndarray
    # Last axis is a stable, named feature vector. The first five columns describe the relaxation;
    # the remaining columns are per-row moments of the live branch-local synaptic state. Keeping the
    # raw arrays alongside the features preserves the original physical-energy path exactly.
    features: np.ndarray | None = None
    feature_names: tuple[str, ...] = ()

    @property
    def shape(self) -> tuple[int, ...]:
        return self.F_final.shape

    @property
    def candidate_count(self) -> int:
        return int(self.F_final.shape[-1])

    @property
    def max_effort_per_row(self) -> int:
        return int(np.max(np.sum(self.effort, axis=-1)))


@dataclass(frozen=True)
class CandidateEnergyReadout:
    """Leakage-safe linear rank readout fitted on an explicit calibration split.

    The readout predicts a *total* candidate energy from the model-energy prior (`-logit`) and
    branch-local synaptic features. Fitting uses within-candidate-set pairwise differences, so the
    learned direction directly asks for the correct continuation to have lower energy than every
    incorrect continuation. A ridge prior keeps the first coefficient pointed along model energy;
    the synaptic terms must earn any residual correction from calibration data.

    This object is deliberately immutable and has no optimizer/model reference. Consequently it
    cannot update during evaluation: callers must fit it on a disjoint calibration split and pass
    the frozen result to :class:`DeliberationController`.
    """

    feature_names: tuple[str, ...]
    center: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[float, ...]
    output_scale: float
    output_offset: float
    blend_weight: float
    l2: float
    calibration_groups: int
    calibration_pairs: int

    @classmethod
    def fit(
        cls,
        *,
        model_logits: np.ndarray,
        synaptic_features: np.ndarray,
        correct_mask: np.ndarray,
        feature_names: Sequence[str],
        l2: float = 1.0,
    ) -> CandidateEnergyReadout:
        """Fit on grouped candidates; groups without exactly one gold candidate are excluded."""
        logits = np.asarray(model_logits, dtype=np.float64)
        features = np.asarray(synaptic_features, dtype=np.float64)
        correct = np.asarray(correct_mask, dtype=np.bool_)
        names = tuple(feature_names)
        if logits.ndim != 2:
            raise ValueError("model_logits must have shape (groups, candidates)")
        if correct.shape != logits.shape:
            raise ValueError("correct_mask must match model_logits")
        if features.shape[:2] != logits.shape or features.ndim != 3:
            raise ValueError("synaptic_features must have shape (groups, candidates, features)")
        if features.shape[-1] != len(names):
            raise ValueError("feature_names must name every synaptic feature")
        if len(set(names)) != len(names):
            raise ValueError("feature_names must be unique")
        if not np.isfinite(logits).all() or not np.isfinite(features).all():
            raise ValueError("calibration inputs must be finite")
        if not np.isfinite(l2) or l2 <= 0.0:
            raise ValueError("l2 must be finite and positive")

        valid_groups = np.flatnonzero(correct.sum(axis=1) == 1)
        if valid_groups.size < 2:
            raise ValueError("at least two calibration groups must contain exactly one gold candidate")
        design = np.concatenate((-logits[..., None], features), axis=-1)
        flat = design[valid_groups].reshape(-1, design.shape[-1])
        center = flat.mean(axis=0)
        scale = flat.std(axis=0)
        scale = np.where(scale > 1e-8, scale, 1.0)
        standardized = (design - center) / scale

        differences: list[np.ndarray] = []
        for group_idx in valid_groups:
            group = standardized[group_idx]
            gold = group[correct[group_idx]][0]
            differences.extend(group[~correct[group_idx]] - gold)
        pairwise = np.asarray(differences, dtype=np.float64)
        if pairwise.ndim != 2 or pairwise.shape[0] < 2:
            raise ValueError("calibration must contain at least two correct/incorrect candidate pairs")

        # Target a unit energy margin for each incorrect-minus-correct pair. The prior says model
        # energy is useful while all residual synaptic weights start at zero.
        prior = np.zeros(pairwise.shape[1], dtype=np.float64)
        prior[0] = 1.0
        gram = pairwise.T @ pairwise + l2 * np.eye(pairwise.shape[1], dtype=np.float64)
        rhs = pairwise.T @ np.ones(pairwise.shape[0], dtype=np.float64) + l2 * prior
        weights = np.linalg.solve(gram, rhs)
        if not np.isfinite(weights).all():
            raise ValueError("calibration produced non-finite readout weights")
        # Pairwise ranking determines ordering but not absolute scale. Sampling consumes logits, so
        # an arbitrary energy scale would silently change effective temperature. Match the frozen
        # readout's mean/std to model energy on calibration data; this affine map preserves ranking.
        predicted_energy = standardized[valid_groups] @ weights
        calibration_model_energy = design[valid_groups, :, 0]
        predicted_std = float(predicted_energy.std())
        model_energy_std = float(calibration_model_energy.std())
        output_scale = model_energy_std / predicted_std if predicted_std > 1e-8 else 1.0
        output_offset = float(
            calibration_model_energy.mean() - output_scale * predicted_energy.mean()
        )
        full_calibrated_energy = output_scale * predicted_energy + output_offset
        # Ranking fit alone does not identify probability margins. Select a conservative blend by
        # calibration cross-entropy; ties resolve toward model-only energy, preventing a noisy
        # synaptic residual from flattening a well-calibrated language-model distribution.
        calibration_correct = correct[valid_groups]
        blend_weight = 0.0
        best_nll = float("inf")
        for blend in np.linspace(0.0, 1.0, 17):
            blended = calibration_model_energy + blend * (
                full_calibrated_energy - calibration_model_energy
            )
            row_min = blended.min(axis=1, keepdims=True)
            log_normalizer = (
                -row_min[:, 0]
                + np.log(np.exp(-(blended - row_min)).sum(axis=1))
            )
            gold_energy = blended[calibration_correct]
            nll = float(np.mean(gold_energy + log_normalizer))
            if nll < best_nll - 1e-12:
                best_nll = nll
                blend_weight = float(blend)
        return cls(
            feature_names=("model_energy", *names),
            center=tuple(float(value) for value in center),
            scale=tuple(float(value) for value in scale),
            weights=tuple(float(value) for value in weights),
            output_scale=output_scale,
            output_offset=output_offset,
            blend_weight=blend_weight,
            l2=float(l2),
            calibration_groups=int(valid_groups.size),
            calibration_pairs=int(pairwise.shape[0]),
        )

    def energy(
        self,
        model_logits: np.ndarray,
        scores: CandidateEnergyBatch,
    ) -> np.ndarray:
        """Return calibrated total energies with the same candidate-grid shape as the logits."""
        if scores.features is None:
            raise ValueError("candidate scores do not contain calibrated-readout features")
        expected_names = self.feature_names[1:]
        if scores.feature_names != expected_names:
            raise ValueError(
                "candidate feature schema does not match calibrated readout: "
                f"expected {expected_names}, got {scores.feature_names}"
            )
        logits = np.asarray(model_logits, dtype=np.float64)
        if logits.shape != scores.shape:
            raise ValueError("model_logits must match the candidate energy grid")
        design = np.concatenate((-logits[..., None], scores.features), axis=-1)
        center = np.asarray(self.center, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        weights = np.asarray(self.weights, dtype=np.float64)
        model_energy = design[..., 0]
        full_calibrated = (
            self.output_scale * (((design - center) / scale) @ weights) + self.output_offset
        )
        energies = model_energy + self.blend_weight * (full_calibrated - model_energy)
        if not np.isfinite(energies).all():
            raise ValueError("calibrated candidate energies must be finite")
        return energies

    def to_dict(self) -> dict:
        """Strict-JSON-ready calibration artifact."""
        return asdict(self)


@dataclass
class DeliberationRecord:
    """Auditable per-token deliberation trace (F-trajectory + effort; the `eqyk.2` schema)."""

    token_index: int
    effort: int                # iterations actually used (the token-difficulty estimate)
    halted_converged: bool     # True ⟹ self-consistent; False ⟹ budget hit (still "thinking")
    F_initial: float
    F_final: float             # the confidence signal (lower ⟹ more self-consistent)
    F_drop: float              # F_initial − F_final, the free energy released by pondering
    base_temperature: float
    effective_temperature: float
    calcium: float             # the aggregated synaptic state that seeded z
    buffer: float
    candidate_count: int = 0
    candidate_effort: int = 0
    candidate_F_min: float | None = None
    candidate_F_max: float | None = None

    @property
    def total_effort(self) -> int:
        """Worst-row bounded ponder effort, including candidate continuations."""
        return self.effort + self.candidate_effort


class DeliberationController:
    """Per-token free-energy deliberation for the engine decode path (bead `r00r.1.2`).

    Stateless across tokens except for the F-trajectory log; safe to reuse across a generation. The
    engine calls `effective_temperature(presyn_state, base_temp, token_index)` once per decoded token;
    when no synaptic state is present (vanilla model) it returns `base_temp` unchanged — the
    deterministic fallback to single-step decode.
    """

    _STATE_FEATURE_KEYS = ("C", "BUF", "RRP", "RES", "PR", "CL", "E", "AMP")

    def __init__(
        self,
        cfg: DeliberationConfig | None = None,
        *,
        candidate_readout: CandidateEnergyReadout | None = None,
    ) -> None:
        self.cfg = cfg or DeliberationConfig()
        self.candidate_readout = candidate_readout
        self.records: list[DeliberationRecord] = []

    # -- synaptic-state readout ---------------------------------------------- #
    @staticmethod
    def synaptic_z(presyn_state) -> np.ndarray | None:
        """Map the live presyn state to the metriplectic core `z = (C, B, h)`.

        Aggregates the mean calcium `C` and buffer `B` over layers/heads/edges; `h = 0` so the
        "kinetic" calcium energy `½(C²+B²)` is what relaxes into effort/entropy during the ponder
        (an active, high-calcium token is far from equilibrium ⟹ harder ⟹ more deliberation steps).
        Returns ``None`` when there is no synaptic state (a vanilla model ⟹ fall back to single-step).
        """
        if presyn_state is None:
            return None
        layers = presyn_state if isinstance(presyn_state, list) else [presyn_state]
        cs, bs = [], []
        for st in layers:
            if not isinstance(st, dict):
                continue
            c, b = st.get("C"), st.get("BUF")
            # Only accept finite means: an empty or all-NaN calcium tensor must NOT become a NaN `z`
            # (which would silently drive the latch to the explore-ceiling and log NaN). Skip it, and
            # if no usable calcium remains, fall back to None ⟹ single-step decode (vanilla behavior).
            if c is not None:
                cm = float(torch.as_tensor(c, dtype=torch.float64).mean())
                if np.isfinite(cm):
                    cs.append(cm)
            if b is not None:
                bm = float(torch.as_tensor(b, dtype=torch.float64).mean())
                if np.isfinite(bm):
                    bs.append(bm)
        if not cs:
            return None
        c_mean = float(np.mean(cs))
        b_mean = float(np.mean(bs)) if bs else 0.0
        return np.array([c_mean, b_mean, 0.0], dtype=np.float64)

    @staticmethod
    def synaptic_z_rows(presyn_state) -> np.ndarray | None:
        """Map each cache row to its own `(C, B, h)` core without mixing candidates.

        Candidate branches are packed as rows in one batched forward.  Preserving that leading
        dimension is essential: averaging it away would assign every candidate the same energy and
        make the apparent logit feedback a no-op.
        """
        if presyn_state is None:
            return None
        layers = presyn_state if isinstance(presyn_state, list) else [presyn_state]
        c_sum = b_sum = None
        c_count = b_count = None
        batch_size = None
        for state in layers:
            if not isinstance(state, dict) or state.get("C") is None:
                continue
            calcium = torch.as_tensor(state["C"], dtype=torch.float64)
            if calcium.ndim == 0:
                calcium = calcium.reshape(1, 1)
            else:
                calcium = calcium.reshape(calcium.shape[0], -1)
            if batch_size is None:
                batch_size = int(calcium.shape[0])
                c_sum = np.zeros(batch_size, dtype=np.float64)
                b_sum = np.zeros(batch_size, dtype=np.float64)
                c_count = np.zeros(batch_size, dtype=np.int64)
                b_count = np.zeros(batch_size, dtype=np.int64)
            elif calcium.shape[0] != batch_size:
                return None
            calcium_mean = calcium.mean(dim=1).detach().cpu().numpy()
            calcium_valid = np.isfinite(calcium_mean)
            c_sum[calcium_valid] += calcium_mean[calcium_valid]
            c_count[calcium_valid] += 1

            buffer = state.get("BUF")
            if buffer is None:
                continue
            buffer_tensor = torch.as_tensor(buffer, dtype=torch.float64)
            if buffer_tensor.ndim == 0:
                buffer_tensor = buffer_tensor.reshape(1, 1)
            else:
                buffer_tensor = buffer_tensor.reshape(buffer_tensor.shape[0], -1)
            if buffer_tensor.shape[0] != batch_size:
                return None
            buffer_mean = buffer_tensor.mean(dim=1).detach().cpu().numpy()
            buffer_valid = np.isfinite(buffer_mean)
            b_sum[buffer_valid] += buffer_mean[buffer_valid]
            b_count[buffer_valid] += 1

        if (
            batch_size is None
            or c_sum is None
            or b_sum is None
            or c_count is None
            or b_count is None
            or np.any(c_count == 0)
        ):
            return None
        calcium_rows = c_sum / c_count
        buffer_rows = np.divide(
            b_sum,
            b_count,
            out=np.zeros_like(b_sum),
            where=b_count > 0,
        )
        return np.column_stack((calcium_rows, buffer_rows, np.zeros(batch_size)))

    @classmethod
    def candidate_state_features(
        cls,
        presyn_state,
        *,
        expected_rows: int,
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        """Extract stable per-row state moments without mixing candidate branches.

        Each state key contributes its within-row mean and standard deviation, averaged across
        layers. Missing optional state is represented by zero, which keeps the feature schema stable
        across configurations. The physical raw-energy path never consumes these columns.
        """
        layers = presyn_state if isinstance(presyn_state, list) else [presyn_state]
        columns: list[np.ndarray] = []
        names: list[str] = []
        for key in cls._STATE_FEATURE_KEYS:
            layer_means: list[np.ndarray] = []
            layer_stds: list[np.ndarray] = []
            for state in layers:
                if not isinstance(state, dict):
                    continue
                value = state.get(key)
                if value is None or not torch.is_tensor(value):
                    continue
                tensor = value.detach().to(dtype=torch.float64)
                if tensor.ndim == 0 or tensor.shape[0] != expected_rows:
                    continue
                rows = tensor.reshape(expected_rows, -1)
                layer_means.append(
                    np.nan_to_num(rows.mean(dim=1).cpu().numpy(), copy=False)
                )
                layer_stds.append(
                    np.nan_to_num(rows.std(dim=1, unbiased=False).cpu().numpy(), copy=False)
                )
            if layer_means:
                columns.extend((np.mean(layer_means, axis=0), np.mean(layer_stds, axis=0)))
            else:
                columns.extend((np.zeros(expected_rows), np.zeros(expected_rows)))
            names.extend((f"{key}_mean", f"{key}_std"))
        return np.column_stack(columns), tuple(names)

    # -- the ponder ----------------------------------------------------------- #
    def ponder(self, z: np.ndarray) -> DeliberationResult:
        """Run the bounded free-energy-minimization loop on `z` (the design §1 deliberation loop)."""
        return deliberate(z, self.cfg.dt, eps=self.cfg.eps, max_iters=self.cfg.max_iters, T=self.cfg.T)

    def adaptive_temperature(self, base_temp: float, result: DeliberationResult) -> float:
        """Decode temperature from the deliberation outcome — commit when confident, explore when not.

        Multiplier interpolates `temp_floor → temp_ceil` with the effort fraction `iters/max_iters`:
        an easy token (halts in ~1 step) sharpens toward `temp_floor`; a budget-hitting hard token
        widens toward `temp_ceil`. The multiplier scales the caller's base temperature, so greedy
        (`base_temp == 0`) stays greedy and the bounds keep the effective temperature well-conditioned.
        """
        if base_temp <= 0.0:
            return base_temp  # greedy decode is unaffected (argmax regardless of temperature)
        frac = min(1.0, max(0.0, result.iters / max(1, self.cfg.max_iters)))
        mult = self.cfg.temp_floor + (self.cfg.temp_ceil - self.cfg.temp_floor) * frac
        return base_temp * mult

    def relax_candidate_states(
        self,
        presyn_state,
        *,
        candidate_shape: tuple[int, int],
    ) -> CandidateEnergyBatch | None:
        """Relax every packed candidate row, preserving its `(batch, top-k)` identity."""
        if not self.cfg.enabled or self.cfg.max_iters == 0:
            return None
        rows = self.synaptic_z_rows(presyn_state)
        expected_rows = int(np.prod(candidate_shape))
        if rows is None:
            return None
        if rows.shape != (expected_rows, 3):
            raise ValueError(
                "candidate presynaptic state does not match candidate grid: "
                f"got {rows.shape[0]} rows for shape {candidate_shape}"
            )
        results = [self.ponder(row) for row in rows]
        F_initial = np.asarray(
            [free_energy(row, self.cfg.T) for row in rows], dtype=np.float64
        ).reshape(candidate_shape)
        F_final = np.asarray(
            [result.F_final for result in results], dtype=np.float64
        ).reshape(candidate_shape)
        effort = np.asarray(
            [result.iters for result in results], dtype=np.int64
        ).reshape(candidate_shape)
        halted_converged = np.asarray(
            [result.halted_converged for result in results], dtype=np.bool_
        ).reshape(candidate_shape)
        state_features, state_feature_names = self.candidate_state_features(
            presyn_state,
            expected_rows=expected_rows,
        )
        relaxation_features = np.column_stack((
            F_initial.reshape(-1),
            F_final.reshape(-1),
            (F_initial - F_final).reshape(-1),
            effort.reshape(-1) / max(1, self.cfg.max_iters),
            halted_converged.reshape(-1).astype(np.float64),
            state_features,
        )).reshape(*candidate_shape, -1)
        return CandidateEnergyBatch(
            F_initial=F_initial,
            F_final=F_final,
            effort=effort,
            halted_converged=halted_converged,
            features=relaxation_features,
            feature_names=(
                "F_initial",
                "F_final",
                "F_drop",
                "effort_fraction",
                "halted_converged",
                *state_feature_names,
            ),
        )

    def candidate_energy_logits(
        self,
        logits,
        candidate_ids,
        scores: CandidateEnergyBatch,
    ):
        """Add relaxed candidate free energy to the model energy and return shape-safe logits.

        `-model_logit` is the base candidate energy.  The physical continuation energy is additive,
        so the adjusted score is `model_logit - lambda * F_final`; candidates outside the bounded
        model-top-k set receive `-inf` and cannot be sampled.
        """
        values = torch.as_tensor(logits)
        ids = torch.as_tensor(candidate_ids, device=values.device)
        if values.ndim != 2 or ids.ndim != 2 or ids.shape[0] != values.shape[0]:
            raise ValueError("logits and candidate_ids must have shapes (batch, vocab) and (batch, k)")
        if tuple(ids.shape) != scores.shape:
            raise ValueError(
                f"candidate energy shape {scores.shape} does not match ids shape {tuple(ids.shape)}"
            )
        if self.candidate_readout is None:
            candidate_energies = scores.F_final
        else:
            selected_model_logits = values.gather(1, ids).detach().to(dtype=torch.float64).cpu().numpy()
            calibrated_total = self.candidate_readout.energy(selected_model_logits, scores)
            # Interpolate from the original model energy to the calibrated total energy. This keeps
            # weight=0 an exact identity on selected candidates and weight=1 equal to `-E_cal`.
            model_energy = -selected_model_logits
            candidate_energies = calibrated_total - model_energy
        energies = torch.as_tensor(candidate_energies, dtype=values.dtype, device=values.device)
        if not bool(torch.isfinite(energies).all()):
            raise ValueError("candidate free energies must be finite")
        selected_logits = values.gather(1, ids)
        selected_logits = selected_logits - self.cfg.candidate_energy_weight * energies
        adjusted = torch.full_like(values, -torch.inf)
        return adjusted.scatter(1, ids, selected_logits)

    def effective_temperature(
        self,
        presyn_state,
        base_temp: float,
        *,
        token_index: int | None = None,
        candidate_scores: CandidateEnergyBatch | None = None,
    ) -> float:
        """The per-token engine hook: ponder the synaptic state and return the decode temperature.

        Falls back to `base_temp` (single-step decode) when deliberation is disabled or there is no
        synaptic state. Logs an auditable F-trajectory record when it ponders.
        """
        if not self.cfg.enabled or self.cfg.max_iters == 0:
            return base_temp
        z = self.synaptic_z(presyn_state)
        if z is None:
            return base_temp
        res = self.ponder(z)
        temp_eff = self.adaptive_temperature(base_temp, res)
        self.records.append(DeliberationRecord(
            token_index=self._next_index(token_index),
            effort=res.iters,
            halted_converged=res.halted_converged,
            F_initial=float(free_energy(z, self.cfg.T)),
            F_final=res.F_final,
            F_drop=res.F_drop,
            base_temperature=base_temp,
            effective_temperature=temp_eff,
            calcium=float(z[0]),
            buffer=float(z[1]),
            candidate_count=(0 if candidate_scores is None else candidate_scores.candidate_count),
            candidate_effort=(
                0 if candidate_scores is None else candidate_scores.max_effort_per_row
            ),
            candidate_F_min=(
                None if candidate_scores is None else float(np.min(candidate_scores.F_final))
            ),
            candidate_F_max=(
                None if candidate_scores is None else float(np.max(candidate_scores.F_final))
            ),
        ))
        return temp_eff

    # -- energy-based decoding (Boltzmann) ------------------------------------ #
    @staticmethod
    def boltzmann_token_weights(logits, kT: float = 1.0):
        """Energy-based decode weights `p ∝ exp(−F/kT)` over the last (vocab) axis, with the model
        logits as negative energy (`F = −logit`).

        This is exactly the temperature-`kT` softmax of the logits, normalized **per distribution**
        (the last axis), so it is correct for a single `(vocab,)` logit vector *and* a batch
        `(..., vocab)` of them (each row sums to 1). For a single vector it equals
        `metriplectic_integrator.boltzmann_weights(−logits, kT)`. It composes with
        `effective_temperature`, which supplies a deliberation-derived `kT`. For candidate-level energy
        decoding (score each relaxed continuation by its own `F`), pass those free energies to
        `metriplectic_integrator.boltzmann_weights` directly (the `re4e.3` energy-guided search path).
        """
        if kT <= 0.0:
            raise ValueError(f"kT must be positive, got {kT}")
        t = torch.as_tensor(logits, dtype=torch.float64)
        return torch.softmax(t / kT, dim=-1)

    # -- traces --------------------------------------------------------------- #
    def _next_index(self, token_index: int | None) -> int:
        if token_index is not None:
            return token_index
        idx = len(self.records)
        return idx

    def f_trajectory(self) -> list[dict]:
        """The per-token F-trajectory + effort log (JSONL-ready dicts)."""
        return [asdict(r) for r in self.records]

    def to_jsonl(self) -> list[str]:
        """The per-token F-trajectory as JSONL lines — the detailed-logging artifact (`eqyk.2`)."""
        return [json.dumps(asdict(r), ensure_ascii=False) for r in self.records]

    def write_trajectory(self, path) -> None:
        """Write the per-token F-trajectory artifact to ``path`` as JSONL (one record per token)."""
        from pathlib import Path
        Path(path).write_text("\n".join(self.to_jsonl()) + ("\n" if self.records else ""), encoding="utf-8")

    def summary(self) -> dict:
        if not self.records:
            return {"tokens": 0, "enabled": self.cfg.enabled}
        efforts = [r.total_effort for r in self.records]
        return {
            "tokens": len(self.records),
            "enabled": self.cfg.enabled,
            "mean_effort": float(np.mean(efforts)),
            "max_effort": int(np.max(efforts)),
            "frac_converged": sum(r.halted_converged for r in self.records) / len(self.records),
            "mean_F_drop": float(np.mean([r.F_drop for r in self.records])),
            "max_budget": (self.cfg.candidate_top_k + 1) * self.cfg.max_iters,
        }


def make_controller(cfg: DeliberationConfig | None) -> DeliberationController | None:
    """Build a controller iff deliberation is enabled; else ``None`` (the engine decodes as baseline)."""
    if cfg is None or not cfg.enabled:
        return None
    return DeliberationController(cfg)
