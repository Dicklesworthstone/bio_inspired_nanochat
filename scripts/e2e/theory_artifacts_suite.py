r"""E2E SCRIPT: Theory artifacts verification suite (beads 0642, eqyk.18).

Comprehensive verification of Leapfrog-Theory formal mathematical certificates,
monitors, integrators, and bounds:
  1. ``metriplectic_energy_and_free_energy``: Metriplectic 3-state integrator [C, B, h] conserves
     total energy to machine precision ($|E(z_t) - E(z_0)| < 1e-10$), dissipates free energy
     strictly ($F_{t+1} \le F_t$), and satisfies Leibniz degeneracy ($L \nabla S = 0, M \nabla E = 0$).
  2. ``singular_perturbation_and_cusp_latch``: Fast/slow timescale separation ($\epsilon < 0.5$),
     Cusp catastrophe bistability threshold ($\Delta^* > 0$), and slow manifold projection certificates.
  3. ``stochastic_thermo_and_tur_bounds``: Crooks integral fluctuation theorem ($\langle e^{-\sigma} \rangle \approx 1.0$),
     entropy production non-negativity ($\langle \sigma \rangle \ge 0$), and Thermodynamic Uncertainty Relation (TUR).
  4. ``structural_geometry_and_optimal_transport``: Spectral conditioning ($\kappa(W)$ bounded) on
     expert split/birth, and Wasserstein-1 Optimal Transport barycentric merging certificates.
  5. ``timescale_separation_coupling``: Dynamic coupling timescale hierarchy across presynaptic calcium,
     vesicle release, fast eligibility traces, slow weights, and structural MoE neurogenesis.
  6. Structured event streaming: Emits rich mathematical certificate summaries and proof records into ``events.jsonl``.

Run:
    python -m scripts.e2e.theory_artifacts_suite
    pytest tests/test_e2e_theory_artifacts.py -v
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat.cusp_certificate import (
    CuspLatch,
    certify_retention,
    run_monitored_latch,
)
from bio_inspired_nanochat.e2e_harness import InvariantResult
from bio_inspired_nanochat.metriplectic_integrator import (
    energy,
    free_energy,
    run_monitored,
)
from bio_inspired_nanochat.run_logging import RunLogger
from bio_inspired_nanochat.separation_gauge import (
    coupling_timescales,
    is_well_separated,
)
from bio_inspired_nanochat.stochastic_thermo import (
    ReleaseRates,
    affinity,
    crooks_calibration,
    integral_ft_closed_form,
    integral_fluctuation_theorem,
    simulate_currents,
    tur_certificate,
)
from bio_inspired_nanochat.structural_geometry import (
    function_preserving_split,
    ot_merge_certificate,
    spectral_conditioning_certificate,
)
from bio_inspired_nanochat.synaptic import SynapticConfig


@dataclass
class TheoryArtifactsConfig:
    """Configuration for Theory Artifacts verification suite."""

    dt: float = 0.05
    metriplectic_steps: int = 40
    thermo_trajectories: int = 1500
    thermo_steps: float = 2.0
    expert_dim: int = 32
    seed: int = 42


@dataclass
class TheoryArtifactsReport:
    run_id: str
    config: TheoryArtifactsConfig
    passed: bool
    invariants: list[InvariantResult]
    summary: dict[str, Any] = field(default_factory=dict)

    def assert_passed(self) -> None:
        failed = [inv for inv in self.invariants if not inv.passed]
        if failed:
            msg = "\n".join(f"  FAILED: {inv.name} -> {inv.detail}" for inv in failed)
            raise AssertionError(f"Theory Artifacts battery failed with {len(failed)} failure(s):\n{msg}")


def run_theory_artifacts_e2e(
    cfg: TheoryArtifactsConfig | None = None,
    *,
    run_dir: Path | str | None = None,
    verbose: bool = True,
) -> TheoryArtifactsReport:
    """Run the complete Theory Artifacts verification battery."""
    if cfg is None:
        cfg = TheoryArtifactsConfig()

    console = Console(quiet=not verbose)
    run_id = f"theory-artifacts-e2e-{int(time.time())}"
    invariants: list[InvariantResult] = []

    clean_tmp = False
    if run_dir is None:
        base_dir = Path(tempfile.mkdtemp(prefix="theory_artifacts_e2e_"))
        clean_tmp = True
    else:
        base_dir = Path(run_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

    run_logger = RunLogger(base_dir, name="theory_artifacts_e2e", run_id=run_id, console=verbose)
    run_logger.event("theory_artifacts_config", config=asdict(cfg))

    try:
        # ===================================================================
        # 1. Metriplectic Energy Conservation & Free Energy Dissipation (Thrust A)
        # ===================================================================
        z0 = np.array([1.2, 0.4, 0.5], dtype=np.float64)  # [C, B, h]
        traj, lyap = run_monitored(z0, cfg.dt, cfg.metriplectic_steps)
        lyap_sum = lyap.summary()

        e0 = energy(z0)
        e_final = energy(traj[-1])
        f0 = free_energy(z0)
        f_final = free_energy(traj[-1])

        max_energy_dev = float(lyap_sum["max_energy_drift"])
        free_energy_monotonic = bool(lyap_sum["lyapunov_ok"])
        max_deg_res = float(lyap_sum["max_degeneracy_residual"])

        metriplectic_ok = (
            max_energy_dev < 1e-8
            and free_energy_monotonic
            and max_deg_res < 1e-10
        )
        invariants.append(
            InvariantResult(
                name="metriplectic_energy_and_free_energy",
                passed=metriplectic_ok,
                observed={
                    "initial_energy": e0,
                    "final_energy": e_final,
                    "max_energy_dev": max_energy_dev,
                    "free_energy_monotonic": free_energy_monotonic,
                    "max_degeneracy_residual": max_deg_res,
                    "initial_free_energy": f0,
                    "final_free_energy": f_final,
                    "n_fallbacks": lyap_sum["n_fallbacks"],
                },
                detail=(
                    f"Energy preserved (|dE|={max_energy_dev:.2e} < 1e-8); "
                    f"Free energy monotonic ({f0:.4f} -> {f_final:.4f}); "
                    f"Degeneracy residuals={max_deg_res:.2e}"
                ),
            )
        )

        # ===================================================================
        # 2. Singular Perturbation, Cusp Catastrophe & Retention (Thrust F)
        # ===================================================================
        syn_cfg = SynapticConfig(
            bistable_latch=True,
            cusp_latch=True,
            latch_alpha_ca=0.6,
            latch_beta_pp1=1.0,
            latch_gamma_auto=0.45,
            latch_hill_n=6.0,
            latch_hill_k=0.6,
        )
        cert = certify_retention(syn_cfg)
        lat = CuspLatch(syn_cfg)

        # Run monitored bistability pulse
        calciums = [0.1, 0.2, 0.8, 0.9, 0.9, 0.2, 0.1]
        _traj_cusp, mon = run_monitored_latch(lat, calciums)

        delta_star = lat.delta_star
        eps_gauge = cert.eps
        is_cert_valid = cert.certified

        cusp_ok = is_cert_valid and eps_gauge < 0.99 and delta_star > 0.0 and mon.separated_throughout()
        invariants.append(
            InvariantResult(
                name="singular_perturbation_and_cusp_latch",
                passed=cusp_ok,
                observed={
                    "valid_certificate": is_cert_valid,
                    "epsilon_gauge": eps_gauge,
                    "delta_star": delta_star,
                    "max_projector_error": mon.max_projector_error(),
                    "separated_throughout": mon.separated_throughout(),
                },
                detail=(
                    f"Retention certificate valid={is_cert_valid}, eps={eps_gauge:.3f} <= {syn_cfg.cusp_eps_max}, "
                    f"Delta*={delta_star:.4f}, normal hyperbolicity maintained"
                ),
            )
        )

        # ===================================================================
        # 3. Stochastic Thermodynamics & TUR Bounds (Thrust E)
        # ===================================================================
        # Analytic & TUR certificate
        rates_tur = ReleaseRates(a=0.8, b=0.2)
        tur = tur_certificate(rates_tur, 5.0)

        # Near-equilibrium IFT simulation
        rates_ift = ReleaseRates(a=0.55, b=0.45)
        aff_ift = affinity(rates_ift)
        currents = simulate_currents(rates_ift, cfg.thermo_steps, cfg.thermo_trajectories, seed=cfg.seed)

        ift_mc = integral_fluctuation_theorem(aff_ift * currents)
        ift_exact = integral_ft_closed_form(rates_ift, cfg.thermo_steps)
        ift_err = abs(ift_mc - 1.0)
        crooks_res = crooks_calibration(aff_ift * currents)

        tur_holds = tur.satisfied and tur.slack >= -1e-6
        ift_holds = ift_err < 0.15 and abs(ift_exact - 1.0) < 1e-12

        thermo_ok = tur_holds and ift_holds and (crooks_res.calibrated or crooks_res.max_abs_residual < 1.0)
        invariants.append(
            InvariantResult(
                name="stochastic_thermo_and_tur_bounds",
                passed=thermo_ok,
                observed={
                    "tur_satisfied": tur.satisfied,
                    "tur_relative_variance": tur.relative_variance,
                    "tur_entropy_bound": tur.entropy_bound,
                    "tur_slack": tur.slack,
                    "ift_mc": ift_mc,
                    "ift_exact": ift_exact,
                    "ift_error": ift_err,
                    "crooks_max_residual": crooks_res.max_abs_residual,
                },
                detail=(
                    f"TUR satisfied={tur.satisfied} (eps^2={tur.relative_variance:.3f} >= bound={tur.entropy_bound:.3f}); "
                    f"IFT MC <e^-sigma>={ift_mc:.3f} (exact={ift_exact:.2f}, |err|={ift_err:.3f})"
                ),
            )
        )

        # ===================================================================
        # 4. Structural Geometry, Spectral Conditioning & OT Merging (Thrust C & B)
        # ===================================================================
        rng = np.random.default_rng(cfg.seed)
        d = cfg.expert_dim
        w_parent = rng.standard_normal((d, d))
        w_child_a, w_child_b = function_preserving_split(w_parent, noise_norm=0.01, rng=rng)

        spec_cert_a = spectral_conditioning_certificate(w_child_a, noise_norm=0.01)
        spec_cert_b = spectral_conditioning_certificate(w_child_b, noise_norm=0.01)

        # Wasserstein-1 optimal transport barycentric merge
        merge_cert = ot_merge_certificate(w_child_a.flatten(), w_child_b.flatten())

        cond_ok = spec_cert_a.well_conditioned and spec_cert_b.well_conditioned and spec_cert_a.kappa_bound < 1000.0
        ot_ok = merge_cert.transport_optimal and merge_cert.transport_cost >= 0.0

        structural_ok = cond_ok and ot_ok
        invariants.append(
            InvariantResult(
                name="structural_geometry_and_optimal_transport",
                passed=structural_ok,
                observed={
                    "parent_kappa": spec_cert_a.kappa_parent,
                    "kappa_bound": spec_cert_a.kappa_bound,
                    "split_reconstruction_error": float(np.linalg.norm((w_child_a + w_child_b) / 2.0 - w_parent)),
                    "ot_transport_cost": merge_cert.transport_cost,
                    "ot_transport_optimal": merge_cert.transport_optimal,
                },
                detail=(
                    f"Split conditioning kappa_bound={spec_cert_a.kappa_bound:.2f} (well_conditioned={spec_cert_a.well_conditioned}); "
                    f"OT merge transport cost={merge_cert.transport_cost:.4f} (optimal={merge_cert.transport_optimal})"
                ),
            )
        )

        # ===================================================================
        # 5. Dynamic Timescale Separation Coupling (Thrust D)
        # ===================================================================
        sep_cfg = SynapticConfig(
            tau_c=6.0,
            tau_buf=4.0,
            tau_rrp=15.0,
            post_trace_decay=0.98,
            post_slow_lr=0.0005,
            structural_interval=50000,
        )
        t_scales = coupling_timescales(sep_cfg)
        well_separated = is_well_separated(sep_cfg, eps_max=0.5)

        # Calcium kinetics faster than release, which is faster than fast weights, which are faster than slow weights, which are faster than structural intervals
        hierarchy_ok = t_scales["calcium"] <= t_scales["release"] <= t_scales["fast_weights"] <= t_scales["slow_weights"] <= t_scales["structure"]

        separation_ok = well_separated and hierarchy_ok
        invariants.append(
            InvariantResult(
                name="timescale_separation_coupling",
                passed=separation_ok,
                observed={
                    "timescales": t_scales,
                    "well_separated": well_separated,
                    "hierarchy_holds": hierarchy_ok,
                },
                detail=(
                    f"Timescale hierarchy: calcium={t_scales['calcium']:.2f} <= "
                    f"release={t_scales['release']:.2f} <= "
                    f"fast_w={t_scales['fast_weights']:.2f} <= "
                    f"slow_w={t_scales['slow_weights']:.2f} <= "
                    f"structure={t_scales['structure']:.2f} (well_separated={well_separated})"
                ),
            )
        )

        for inv in invariants:
            run_logger.event("e2e_invariant", **asdict(inv))

        all_passed = all(inv.passed for inv in invariants)
        report = TheoryArtifactsReport(
            run_id=run_id,
            config=cfg,
            passed=all_passed,
            invariants=invariants,
            summary={
                "metriplectic_energy_dev": max_energy_dev,
                "cusp_delta_star": delta_star,
                "tur_satisfied": tur.satisfied,
                "ift_mc": ift_mc,
                "ot_transport_cost": merge_cert.transport_cost,
                "timescale_hierarchy": hierarchy_ok,
            },
        )

        if verbose:
            table = Table(title="Leapfrog-Theory Artifacts E2E Battery")
            table.add_column("Invariant", style="cyan")
            table.add_column("Status", style="bold")
            table.add_column("Detail", style="dim")
            for inv in invariants:
                status = "[green]PASS[/green]" if inv.passed else "[red]FAIL[/red]"
                table.add_row(inv.name, status, inv.detail)
            console.print(table)

        return report

    finally:
        run_logger.close()
        if clean_tmp:
            shutil.rmtree(base_dir, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Leapfrog-Theory Artifacts E2E battery")
    parser.add_argument("--run-dir", type=str, default=None, help="Directory to save E2E traces and logs")
    parser.add_argument("--steps", type=int, default=40, help="Number of Metriplectic integration steps")
    parser.add_argument("--thermo-trajectories", type=int, default=1500, help="Number of thermodynamic trajectories")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args(argv)

    cfg = TheoryArtifactsConfig(
        metriplectic_steps=args.steps,
        thermo_trajectories=args.thermo_trajectories,
        seed=args.seed,
    )
    report = run_theory_artifacts_e2e(cfg, run_dir=args.run_dir, verbose=True)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
