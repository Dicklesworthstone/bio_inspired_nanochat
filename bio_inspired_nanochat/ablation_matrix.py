"""Pre-registered bio-vs-vanilla ablation matrix (bead hwxb.5.1).

This is the **machine-readable specification** of the headline experiment: *does the biology help,
and which mechanism?* It is the single source of truth that the ablation runner (hwxb.5.2) and the
dry-run validator (hwxb.7.4) consume; the human-readable rationale lives in
``docs/ablation_matrix.md`` and the metric definitions in ``docs/eval_benchmark_matrix.md``.

Design (honoring the bead's pinned review comments):

1. **Three anchors, to disentangle ARCHITECTURE from MECHANISMS.** ``GPTSynaptic`` with every bio
   flag off is still a *different architecture* than vanilla ``GPT`` (presyn attention augmentation,
   router probe, Xi, MoE). So a bare ``vanilla`` vs ``bio_all`` contrast confounds the two. We add a
   ``synaptic_off`` anchor (the synaptic architecture, all mechanisms off — byte-identical-default
   per the unit tests) so the experiment decomposes cleanly:
     - ``synaptic_off − vanilla``      = effect of the synaptic ARCHITECTURE alone,
     - ``mechanism − synaptic_off``    = the CLEAN isolated effect of a mechanism (same arch, one flag),
     - ``bio_all − vanilla``           = the TOTAL bio effect.
   ``vanilla`` is **param-matched** to ``GPTSynaptic`` by adjusting depth/width (the synaptic stack
   adds parameters); the runner records both param counts.

2. **Both ablation directions.** LEAVE-ONE-OUT (``bio_all`` minus each default-on mechanism) measures
   a mechanism's marginal contribution *given the others*; ADD-ONE-IN (``synaptic_off`` plus each
   opt-in mechanism, with its prerequisites) measures a mechanism's standalone effect on the clean
   architecture anchor. Both are derived directly from the ablation registry, so this matrix cannot
   drift from the code's notion of what is ablatable.

3. **Staged compute, with a go/no-go gate.** A cheap SCREENING pass (small budget, few seeds) drops
   clearly-non-helping mechanisms; an expensive CONFIRMATION pass (full budget, >=3 seeds) runs only
   on the survivors plus the anchors. Commit the GPU-hour estimate and pass the gate before the
   confirmation pass — never burn days of 4090 time blindly.

The mechanism set is the authoritative ``ablation_registry.MECHANISMS`` list. ``flex_attention``,
``native_genetics``, ``tropical_skeleton``, and ``recurrence_checkpoint`` are EXCLUDED from the
science matrix: they are performance/runtime toggles, not standalone biological mechanisms (flex
is prefill-only and incompatible with KV-cache decode eval; native needs CUDA; tropical routing
needs a certified exact-affine score adapter; checkpointing changes activation storage rather than
the forward dynamics). Global neuromodulation (hy8.1) and NeuroScore are NOT yet registered ablation
mechanisms, so they are not in this matrix; registering them is a documented prerequisite for
including them (see ``docs/ablation_matrix.md`` §"Known gaps").
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from bio_inspired_nanochat.ablation_registry import (
    MECHANISMS,
    _BY_FIELD,
    validate_config,
)
from bio_inspired_nanochat.synaptic import SynapticConfig

# Mechanisms that are infrastructure/perf toggles, not biology — excluded from the science matrix.
INFRA_MECHANISMS: frozenset[str] = frozenset(
    {
        "flex_attention",
        "native_genetics",
        "recurrence_checkpoint",
        "tropical_skeleton",
    }
)
# The generic matrix only changes SynapticConfig; it does not construct an MoE
# lifecycle controller. Keep topology out until its equal-FLOP structural runner
# can activate MoE + scheduling, otherwise ``add_topological_nas`` is a silent no-op.
MATRIX_EXCLUDED_MECHANISMS: frozenset[str] = (
    INFRA_MECHANISMS | {"topological_nas"}
)

# Reproducible seeds. >=3 for research-grade statistical significance (matches eval_benchmark_matrix).
SCREENING_SEEDS: tuple[int, ...] = (1337, 1338)
CONFIRMATION_SEEDS: tuple[int, ...] = (1337, 1338, 1339)

# Equal-compute token budgets per run (the final value is pinned by the Phase-0 decision rule and the
# feasible set from hwxb.4.5; these are the planning defaults from docs/eval_benchmark_matrix.md).
SCREENING_TOKENS: int = 10_000_000      # "smoke": fast triage, ~<=30 min/seed on one high-end GPU
CONFIRMATION_TOKENS: int = 100_000_000  # "short": usable signal, ~<=4 h/seed on dual 4090

# Quality + health metrics the runner must emit per cell (defined in docs/eval_benchmark_matrix.md).
METRICS: tuple[str, ...] = (
    "val_bpb",            # PRIMARY: bits/byte on the held-out FineWeb split (loss_eval.evaluate_bpb)
    "niah_acc",           # long-context needle-in-a-haystack accuracy (synthetic_tasks)
    "working_memory",     # associative-recall delta (working_memory_suite; honest, may be null)
    "moe_gini",           # expert-routing Gini (neuroscore) — specialization health
    "dead_expert_frac",   # fraction of experts below a routing-frequency floor
    "tok_per_sec",        # throughput, for the equal-compute accounting
    "peak_mem_gb",        # peak VRAM, for feasibility
)

# Upper bound on the confirmation-pass cost before the experiment must be re-scoped (go/no-go gate).
DEFAULT_GPU_HOUR_CAP: float = 72.0  # 3 GPU-days; tune per the available 4090 allocation.


class Base(str, Enum):
    """The base model/architecture a config is built on."""

    VANILLA = "vanilla"            # standard GPT (synapses off at the model level)
    SYNAPTIC_OFF = "synaptic_off"  # GPTSynaptic architecture, every bio mechanism off
    BIO_ALL = "bio_all"            # GPTSynaptic with the default synaptic stack on


# The synaptic-OFF anchor's overrides: turn off every DEFAULT-ON registered mechanism. The opt-in
# mechanisms are already off by default, so this neutralizes the whole stack on the synaptic arch.
SYNAPTIC_OFF_OVERRIDES: dict[str, Any] = {
    m.field: m.off_value for m in MECHANISMS if m.default_on
}


@dataclass(frozen=True)
class AblationConfig:
    """One cell-column of the matrix: an identified model config + its role in the decomposition."""

    config_id: str
    base: Base
    overrides: dict[str, Any]  # SynapticConfig field -> value, applied on top of `base`
    role: str                  # "anchor" | "leave_one_out" | "add_one_in"
    rationale: str

    def build_syn_cfg(self) -> Optional[SynapticConfig]:
        """Materialize the SynapticConfig for this column (``None`` for the vanilla GPT baseline).

        The result is validated; an invalid column is a spec bug and raises.
        """
        if self.base is Base.VANILLA:
            return None
        cfg = SynapticConfig()
        if self.base is Base.SYNAPTIC_OFF:
            for field, value in SYNAPTIC_OFF_OVERRIDES.items():
                setattr(cfg, field, value)
        for field, value in self.overrides.items():
            if field not in {f.name for f in dataclasses.fields(SynapticConfig)}:
                raise ValueError(f"{self.config_id}: unknown SynapticConfig field {field!r}")
            setattr(cfg, field, value)
        errors, _ = validate_config(cfg)
        if errors:
            raise ValueError(f"{self.config_id}: invalid config -> {errors}")
        return cfg


def _on_value(field: str) -> Any:
    """The value that turns a registered mechanism ON (i.e. ``!= off_value``).

    For default-on mechanisms and scalar opt-ins this is the config default (the on-value); for a
    default-off boolean opt-in (``default == off_value == False``) it is ``True``.
    """
    m = _BY_FIELD[field]
    if m.default != m.off_value:
        return m.default
    if isinstance(m.off_value, bool):
        return not m.off_value
    raise ValueError(f"cannot infer an on-value for {field!r} (default == off_value == {m.off_value!r})")


def _prereq_closure(field: str) -> set[str]:
    """All prerequisite fields that must be ON for ``field`` to do anything (transitive)."""
    out: set[str] = set()
    stack = list(_BY_FIELD[field].requires)
    while stack:
        f = stack.pop()
        if f in out:
            continue
        out.add(f)
        if f in _BY_FIELD:
            stack.extend(_BY_FIELD[f].requires)
    return out


def anchors() -> list[AblationConfig]:
    """The three decomposition anchors."""
    return [
        AblationConfig(
            "vanilla", Base.VANILLA, {}, "anchor",
            "Param-matched standard GPT — the silicon baseline the biology must beat.",
        ),
        AblationConfig(
            "synaptic_off", Base.SYNAPTIC_OFF, {}, "anchor",
            "GPTSynaptic architecture with every bio mechanism OFF; isolates the architecture's "
            "effect from the mechanisms'. Byte-identical-default per the unit tests.",
        ),
        AblationConfig(
            "bio_all", Base.BIO_ALL, {}, "anchor",
            "The full default synaptic stack; (bio_all - vanilla) is the total bio effect.",
        ),
    ]


def leave_one_out() -> list[AblationConfig]:
    """``bio_all`` minus each DEFAULT-ON biological mechanism (marginal contribution)."""
    out: list[AblationConfig] = []
    for m in MECHANISMS:
        if not m.default_on or m.mechanism in MATRIX_EXCLUDED_MECHANISMS:
            continue
        out.append(AblationConfig(
            f"bio_no_{m.mechanism}", Base.BIO_ALL, {m.field: m.off_value}, "leave_one_out",
            f"bio_all with {m.mechanism} ablated ({m.field}={m.off_value!r}); its marginal "
            f"contribution given the rest of the stack.",
        ))
    return out


def add_one_in() -> list[AblationConfig]:
    """``synaptic_off`` plus each OPT-IN biological mechanism (+ its prerequisites): standalone effect."""
    out: list[AblationConfig] = []
    for m in MECHANISMS:
        if m.default_on or m.mechanism in MATRIX_EXCLUDED_MECHANISMS:
            continue
        overrides: dict[str, Any] = {}
        # Turn the mechanism's prerequisites back ON (they were neutralized by synaptic_off).
        for prereq in _prereq_closure(m.field):
            overrides[prereq] = _on_value(prereq)
        # Turn the mechanism itself ON.
        overrides[m.field] = _on_value(m.field)
        prereq_note = (
            f" with its prerequisite(s) {sorted(_prereq_closure(m.field))} also on"
            if _prereq_closure(m.field) else ""
        )
        out.append(AblationConfig(
            f"add_{m.mechanism}", Base.SYNAPTIC_OFF, overrides, "add_one_in",
            f"synaptic_off plus {m.mechanism}{prereq_note}; the mechanism's standalone effect on "
            f"the clean architecture anchor.",
        ))
    return out


def screening_columns() -> list[AblationConfig]:
    """Cheap first pass: all anchors + both ablation directions (drop non-helpers here)."""
    return anchors() + leave_one_out() + add_one_in()


def confirmation_columns(survivors: list[str]) -> list[AblationConfig]:
    """Expensive second pass: the anchors plus only the ``survivors`` that screening kept."""
    keep = set(survivors)
    cols = [c for c in screening_columns() if c.role == "anchor" or c.config_id in keep]
    return cols


def estimate_gpu_hours(columns: list[AblationConfig], seeds: tuple[int, ...], train_tokens: int,
                       tok_per_sec: float) -> float:
    """GPU-hours for ``len(columns) x len(seeds)`` runs of ``train_tokens`` at ``tok_per_sec``.

    ``tok_per_sec`` is the measured aggregate training throughput (hwxb.2.2) — or a planning estimate
    from docs/eval_benchmark_matrix.md §5 when no measurement exists yet.
    """
    if tok_per_sec <= 0:
        raise ValueError(f"tok_per_sec must be positive, got {tok_per_sec}")
    runs = len(columns) * len(seeds)
    return runs * (train_tokens / tok_per_sec) / 3600.0


@dataclass(frozen=True)
class GoNoGo:
    proceed: bool
    estimated_gpu_hours: float
    cap_gpu_hours: float
    n_survivors: int
    reason: str


def go_no_go(survivors: list[str], tok_per_sec: float, *,
             cap_gpu_hours: float = DEFAULT_GPU_HOUR_CAP,
             train_tokens: int = CONFIRMATION_TOKENS,
             seeds: tuple[int, ...] = CONFIRMATION_SEEDS) -> GoNoGo:
    """Gate the expensive confirmation pass: proceed only if >=1 mechanism survived screening AND the
    estimated GPU-hours fit the cap."""
    cols = confirmation_columns(survivors)
    hours = estimate_gpu_hours(cols, seeds, train_tokens, tok_per_sec)
    n_surv = len([s for s in survivors if s])
    if n_surv == 0:
        return GoNoGo(False, hours, cap_gpu_hours, 0,
                      "No mechanism survived screening — nothing to confirm; report the null result.")
    if hours > cap_gpu_hours:
        return GoNoGo(False, hours, cap_gpu_hours, n_surv,
                      f"Estimated {hours:.1f} GPU-h exceeds the {cap_gpu_hours:.1f} cap — cut seeds, "
                      f"budget, or survivors before committing.")
    return GoNoGo(True, hours, cap_gpu_hours, n_surv,
                  f"{n_surv} survivor(s); {hours:.1f} GPU-h within the {cap_gpu_hours:.1f} cap.")


# --------------------------------------------------------------------------- #
# Pre-registered decision rule (consistent with the Phase-0 success criterion).
# --------------------------------------------------------------------------- #
# Evaluated by the stats layer (bio_inspired_nanochat/eval_stats.py: aggregate + paired_t_test over
# per-seed deltas) once the runs complete (hwxb.5.3). Pinned BEFORE running so it cannot be
# rationalized after the fact:
#
#   PRIMARY metric: val_bpb (lower is better). All deltas are direction-aware (improvement = bpb down).
#
#   A mechanism "HELPS" iff, across the confirmation seeds:
#     (1) its clean isolated effect is an improvement: mean(metric_synaptic_off - metric_add_mechanism) > 0
#         with a paired 95% CI excluding 0 (paired t AND Wilcoxon agree on direction), AND
#     (2) the effect is attributable to the MECHANISM, not the architecture: the same contrast taken
#         against synaptic_off (not vanilla) is what (1) already uses, so a positive (synaptic_off -
#         vanilla) architecture effect cannot masquerade as a mechanism win.
#
#   "BIO HELPS" overall iff (bio_all - vanilla) on val_bpb is an improvement with a 95% CI excluding 0;
#   the per-mechanism rule above then attributes that win. A null or negative result is reported
#   honestly (the experiment is designed to be able to say "biology did not help at this scale").
DECISION_PRIMARY_METRIC: str = "val_bpb"
DECISION_CONFIDENCE: float = 0.95
