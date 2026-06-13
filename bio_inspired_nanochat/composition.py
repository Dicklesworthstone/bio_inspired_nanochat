"""Composition keystone — timescale-separation guard table + pairwise-interference harness (`0642.10`).

The alien-artifact discipline requires a *timescale-separation statement* before composing multiple
math families. Each thrust applies its esoteric math to one stratum of the synaptic hierarchy

    calcium ≪ release ≪ fast_weights ≪ slow_weights ≪ structure

and its certificate is only valid while its stratum is **separated** from the adjacent faster one (so
the faster level can be treated as slaved / quasi-static). This module turns the per-coupling
separation gauge (`separation_gauge`, bead `0642.10.1`) into:

  - a **composition-eligibility guard table** (`0642.10.2`): for each thrust, is its required coupling
    separated (`ε_k < eps_max`)? Below threshold ⟹ the certificate composes freely; above ⟹ the
    thrust's deterministic fallback trips.
  - a **pairwise-interference harness** (`0642.10.3`): two thrusts may co-activate only if *every*
    boundary between their strata is separated (no cross-timescale coupling); otherwise the
    **higher-risk** thrust is auto-disabled (the lower-risk keystone keeps running).

This is the runtime guard that makes Thrusts A/F/E/B/C safe to combine — composition is *gated on
measured separation*, never assumed. Honest by construction: where the configured timescales do not
actually separate (e.g. `release→fast_weights` at the defaults), the guard says so and trips the
fallback rather than silently composing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from bio_inspired_nanochat.separation_gauge import HIERARCHY, separation_table
from bio_inspired_nanochat.synaptic import SynapticConfig


@dataclass(frozen=True)
class Thrust:
    """A theory thrust: the stratum it governs, the incoming coupling it relies on, and its risk."""

    key: str            # "A", "F", "E", "B", "C"
    name: str
    stratum: str        # the hierarchy level the thrust operates on
    requires: str       # the SLOW side of the incoming boundary it needs separated (its stratum)
    risk: float         # disrupt-risk; on a conflict the HIGHER-risk thrust is disabled first


# Each thrust is gated on the boundary *into* its stratum (the faster level must be slaved). Risk
# ranks by how disruptive a mis-fire is (structural plasticity rewrites the architecture ⟹ highest;
# the metriplectic keystone is stability-by-construction ⟹ lowest).
THRUSTS: dict[str, Thrust] = {
    "A": Thrust("A", "Metriplectic/GENERIC dynamics", "calcium", "release", risk=0.10),
    "E": Thrust("E", "Stochastic-thermodynamic UQ", "release", "release", risk=0.40),
    "F": Thrust("F", "Singular-perturbation/cusp retention", "fast_weights", "fast_weights", risk=0.30),
    "B": Thrust("B", "Ultrametric/RSB memory", "slow_weights", "slow_weights", risk=0.50),
    "C": Thrust("C", "Free-prob/TDA structural plasticity", "structure", "structure", risk=0.70),
}


def _stratum_index(name: str) -> int:
    return HIERARCHY.index(name)


def _separation_rows(cfg: SynapticConfig, eps_max: float):
    """Per-boundary rows keyed by the slow stratum (the boundary *into* that stratum)."""
    return {row.slow: row for row in separation_table(cfg, eps_max=eps_max)}


# =========================================================================== #
# 0642.10.2 — composition-eligibility guard table
# =========================================================================== #
@dataclass(frozen=True)
class ThrustEligibility:
    """Whether a thrust's certificate composes, given the measured separation at its boundary."""

    key: str
    name: str
    boundary: str        # "<fast> → <slow>" the thrust is gated on
    eps: float           # ε_k = τ_fast / τ_slow at that boundary
    eps_max: float
    eligible: bool       # ε_k < eps_max ⟹ compose; else fallback
    verdict: str         # "compose" | "fallback"


def composition_eligibility(cfg: SynapticConfig, *, eps_max: float = 0.5) -> dict[str, ThrustEligibility]:
    """The guard table: for every thrust, is its required coupling separated? (`0642.10.2`)."""
    rows = _separation_rows(cfg, eps_max)
    out: dict[str, ThrustEligibility] = {}
    for key, t in THRUSTS.items():
        row = rows.get(t.requires)
        if row is None:  # the calcium fast-end (no incoming boundary) is always eligible
            out[key] = ThrustEligibility(key, t.name, "fast-end", 0.0, eps_max, True, "compose")
            continue
        out[key] = ThrustEligibility(
            key, t.name, f"{row.fast} → {row.slow}", row.eps, eps_max,
            eligible=row.separated, verdict="compose" if row.separated else "fallback",
        )
    return out


# =========================================================================== #
# 0642.10.3 — pairwise interference harness
# =========================================================================== #
@dataclass(frozen=True)
class CompatibilityVerdict:
    """Whether two thrusts may co-activate, and which (higher-risk) one is disabled if not."""

    thrust_a: str
    thrust_b: str
    compatible: bool
    composite_eps: float        # the worst ε over the boundaries between the two strata
    unseparated_boundaries: tuple  # the boundaries that block composition (empty ⟹ compatible)
    disabled: str | None        # the auto-disabled (higher-risk) thrust if incompatible
    reason: str


def pairwise_compatible(cfg: SynapticConfig, a: str, b: str, *, eps_max: float = 0.5) -> CompatibilityVerdict:
    """Can thrusts `a` and `b` co-activate? Every boundary *between* their strata must be separated.

    If any intermediate coupling has `ε_k ≥ eps_max`, the two strata co-move (cross-timescale
    coupling), so the certificates are not jointly valid and the **higher-risk** thrust is disabled —
    the lower-risk one (typically the metriplectic keystone) keeps running with its guarantees intact.
    """
    ta, tb = THRUSTS[a], THRUSTS[b]
    i, j = sorted((_stratum_index(ta.stratum), _stratum_index(tb.stratum)))
    rows = separation_table(cfg, eps_max=eps_max)
    # boundaries strictly between strata i and j (rows are indexed by the fast stratum, 0..len-1).
    span = [r for r in rows if i <= _stratum_index(r.fast) < j]
    unsep = tuple(f"{r.fast}→{r.slow}" for r in span if not r.separated)
    composite = max((r.eps for r in span), default=0.0)
    compatible = not unsep
    disabled = None
    if not compatible:
        disabled = a if ta.risk >= tb.risk else b
    reason = (f"separated across {ta.stratum}↔{tb.stratum} (worst ε={composite:.3g} < {eps_max:g})"
              if compatible else
              f"unseparated boundary {','.join(unsep)} (worst ε={composite:.3g} ≥ {eps_max:g}) — "
              f"disable higher-risk thrust {disabled}")
    return CompatibilityVerdict(
        thrust_a=a, thrust_b=b, compatible=compatible, composite_eps=composite,
        unseparated_boundaries=unsep, disabled=disabled, reason=reason,
    )


def compatibility_matrix(cfg: SynapticConfig, *, eps_max: float = 0.5) -> dict[str, CompatibilityVerdict]:
    """All pairwise compatibility verdicts (keyed `"A+F"`), for the composition harness + logging."""
    keys = list(THRUSTS)
    out: dict[str, CompatibilityVerdict] = {}
    for x in range(len(keys)):
        for y in range(x + 1, len(keys)):
            a, b = keys[x], keys[y]
            out[f"{a}+{b}"] = pairwise_compatible(cfg, a, b, eps_max=eps_max)
    return out


# =========================================================================== #
# Reporting
# =========================================================================== #
def composition_report(cfg: SynapticConfig, *, eps_max: float = 0.5) -> dict:
    """A JSON-able composition report: per-thrust eligibility + the pairwise compatibility matrix."""
    elig = {k: asdict(v) for k, v in composition_eligibility(cfg, eps_max=eps_max).items()}
    pairs = {k: asdict(v) for k, v in compatibility_matrix(cfg, eps_max=eps_max).items()}
    return {
        "eps_max": eps_max,
        "eligibility": elig,
        "pairwise": pairs,
        "all_eligible": all(e["eligible"] for e in elig.values()),
        "all_pairs_compatible": all(p["compatible"] for p in pairs.values()),
    }


def composition_report_jsonl(cfg: SynapticConfig, *, eps_max: float = 0.5) -> list[str]:
    """The composition report as JSONL lines (one eligibility row + one pairwise row each)."""
    rep = composition_report(cfg, eps_max=eps_max)
    lines = [json.dumps({"kind": "eligibility", **e}, ensure_ascii=False) for e in rep["eligibility"].values()]
    lines += [json.dumps({"kind": "pairwise", **p}, ensure_ascii=False) for p in rep["pairwise"].values()]
    return lines


def render_composition_report(cfg: SynapticConfig, *, eps_max: float = 0.5, console=None) -> None:
    """Rich rendering of the composition guard table (falls back to plain print without rich)."""
    elig = composition_eligibility(cfg, eps_max=eps_max)
    pairs = compatibility_matrix(cfg, eps_max=eps_max)
    try:
        from rich.console import Console
        from rich.table import Table
        console = console or Console()
        t = Table(title="Composition eligibility (thrust ↔ separation)")
        t.add_column("thrust")
        t.add_column("boundary")
        t.add_column("ε", justify="right")
        t.add_column("verdict")
        for e in elig.values():
            t.add_row(f"{e.key} {e.name}", e.boundary, f"{e.eps:.3g}", e.verdict)
        console.print(t)
        pt = Table(title="Pairwise compatibility")
        pt.add_column("pair")
        pt.add_column("compatible")
        pt.add_column("disabled")
        for k, v in pairs.items():
            pt.add_row(k, str(v.compatible), v.disabled or "—")
        console.print(pt)
    except Exception:  # pragma: no cover - rich is a project dep; stay usable without it
        print("composition eligibility:", {k: v.verdict for k, v in elig.items()})
        print("pairwise compatible:", {k: v.compatible for k, v in pairs.items()})
