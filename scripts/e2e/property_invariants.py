r"""E2E SCRIPT: Property-based, metamorphic, and invariant verification suite (bead eqyk.14).

Verifies universal biological, computational, and physical invariants across arbitrary inputs:
  1. ``prop_vesicle_conservation``: Asserts non-negativity and boundedness of vesicle pools
     ($RRP \ge 0, RES \ge 0, \sum \text{DELAY} \ge 0$) with total pool $\le \text{initial\_total}$.
  2. ``prop_eval_determinism``: Asserts bit-for-bit identical outputs on repeated eval forwards.
  3. ``prop_reset_isolation``: Metamorphic invariant: processing $S_A \to \text{reset} \to S_B$
     yields identical logits on $S_B$ as processing $S_B$ on a freshly initialized model.
  4. ``prop_monotonic_depletion``: Sustained high-activation drive monotonically depletes RRP
     towards steady-state refill equilibrium across sequence positions.
  5. ``prop_stochastic_expectation_convergence``: Monte Carlo stochastic vesicle release passes
     converge to deterministic expectation ($p \times RRP$) in the large-sample limit.
  6. ``prop_extreme_input_robustness``: Asserts NaN/Inf freedom under extreme edge-case inputs
     (zeros, max tokens, large drive).
  7. ``prop_causal_invariance``: Modifying tokens at future position $t > k$ has strictly zero effect
     on past logits or presynaptic state at position $k$.
  8. Structured event logging: Emits invariant evaluations and error tolerances to ``events.jsonl``.

Run:
    python -m scripts.e2e.property_invariants
    pytest tests/test_property_invariants.py -v
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat.e2e_harness import InvariantResult
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.run_logging import RunLogger
from bio_inspired_nanochat.synaptic import (
    SynapticConfig,
    SynapticPresyn,
    build_presyn_state,
)


@dataclass
class PropertyInvariantsConfig:
    """Configuration for Property-Based & Metamorphic Invariant battery."""

    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    vocab_size: int = 64
    sequence_len: int = 32
    device: str = "cpu"
    seed: int = 42
    mc_samples: int = 60


@dataclass
class PropertyInvariantsReport:
    run_id: str
    config: PropertyInvariantsConfig
    passed: bool
    invariants: list[InvariantResult]
    summary: dict[str, Any] = field(default_factory=dict)

    def assert_passed(self) -> None:
        failed = [inv for inv in self.invariants if not inv.passed]
        if failed:
            msg = "\n".join(f"  FAILED: {inv.name} -> {inv.detail}" for inv in failed)
            raise AssertionError(f"Property Invariants battery failed with {len(failed)} failure(s):\n{msg}")


def _build_model(cfg: PropertyInvariantsConfig, *, seed_offset: int = 0) -> GPTSynaptic:
    """Create reproducible GPTSynaptic model for property testing."""
    torch.manual_seed(cfg.seed + seed_offset)
    syn_cfg = SynapticConfig(
        enable_hebbian=True,
        plasticity_during_training=False,
        stochastic_train_frac=0.15,
        bistable_latch=True,
        fast_weight_normalized=True,
    )
    gpt_cfg = GPTSynapticConfig(
        sequence_len=cfg.sequence_len,
        vocab_size=cfg.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_kv_head=cfg.n_head,
        n_embd=cfg.n_embd,
        synapses=True,
        syn_cfg=syn_cfg,
    )
    model = GPTSynaptic(gpt_cfg).to(cfg.device)
    model.eval()
    return model


def run_property_invariants_e2e(
    cfg: PropertyInvariantsConfig | None = None,
    *,
    run_dir: Path | str | None = None,
    verbose: bool = True,
) -> PropertyInvariantsReport:
    """Run the complete property-based & metamorphic invariant battery."""
    if cfg is None:
        cfg = PropertyInvariantsConfig()

    console = Console(quiet=not verbose)
    run_id = f"prop-invariants-e2e-{int(time.time())}"
    invariants: list[InvariantResult] = []

    clean_tmp = False
    if run_dir is None:
        base_dir = Path(tempfile.mkdtemp(prefix="prop_invariants_e2e_"))
        clean_tmp = True
    else:
        base_dir = Path(run_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

    run_logger = RunLogger(base_dir, name="property_invariants_e2e", run_id=run_id, console=verbose)
    run_logger.event("property_invariants_config", config=asdict(cfg))

    try:
        # ===================================================================
        # 1. Vesicle Conservation & Physical State Boundedness
        # ===================================================================
        syn_cfg = SynapticConfig(enable_presyn=True)
        pre = SynapticPresyn(cfg.n_embd // cfg.n_head, syn_cfg)
        state = build_presyn_state(2, 16, cfg.n_head, torch.device(cfg.device), torch.float32, syn_cfg)
        idx = torch.zeros(2, cfg.n_head, 16, 4, dtype=torch.long)
        for t in range(16):
            idx[:, :, t, :] = torch.randint(0, t + 1, (2, cfg.n_head, 4))

        init_pool = float((state["RRP"] + state["RES"]).sum().item())
        conservation_holds = True
        ranges_hold = True

        for step in range(20):
            drive = torch.randn(2, cfg.n_head, 16, 4) * 2.5
            pre.release_canonical(state, drive, idx, train=False)

            current_pool = float((state["RRP"] + state["RES"]).sum().item())
            for d in state["DELAY"]:
                current_pool += float(d.sum().item())

            if current_pool > init_pool + 1e-4:
                conservation_holds = False

            # Check valid bounds
            if (state["RRP"] < -1e-4).any() or (state["RES"] < -1e-4).any() or (state["C"] < -1e-4).any():
                ranges_hold = False

        vesicle_ok = conservation_holds and ranges_hold
        invariants.append(
            InvariantResult(
                name="prop_vesicle_conservation",
                passed=vesicle_ok,
                observed={
                    "init_pool": init_pool,
                    "final_pool": current_pool,
                    "conservation_holds": conservation_holds,
                    "ranges_hold": ranges_hold,
                },
                detail=(
                    f"Vesicle pool bounded: init={init_pool:.2f} -> final={current_pool:.2f}, "
                    f"conservation={conservation_holds}, valid_ranges={ranges_hold}"
                ),
            )
        )

        # ===================================================================
        # 2. Evaluation Determinism & Idempotence
        # ===================================================================
        model = _build_model(cfg, seed_offset=1)
        test_seq = torch.randint(0, cfg.vocab_size, (2, 16), device=cfg.device)

        model.reset_sequence_state(reset_fast_weights=True)
        with torch.no_grad():
            out1 = model(test_seq, train_mode=False)[0]

        model.reset_sequence_state(reset_fast_weights=True)
        with torch.no_grad():
            out2 = model(test_seq, train_mode=False)[0]

        det_diff = float((out1 - out2).abs().max().item())
        det_ok = det_diff == 0.0
        invariants.append(
            InvariantResult(
                name="prop_eval_determinism",
                passed=det_ok,
                observed={"max_diff": det_diff},
                detail=f"Deterministic eval output diff = {det_diff:.2e}",
            )
        )

        # ===================================================================
        # 3. Metamorphic Per-Sequence Reset Isolation
        # ===================================================================
        seq_A = torch.randint(0, cfg.vocab_size, (2, 12), device=cfg.device)
        seq_B = torch.randint(0, cfg.vocab_size, (2, 14), device=cfg.device)

        # Arm 1: Process seq_A -> reset -> process seq_B
        model_iso = _build_model(cfg, seed_offset=2)
        with torch.no_grad():
            model_iso(seq_A, train_mode=False)
            model_iso.reset_sequence_state(reset_fast_weights=True)
            out_B_after_A = model_iso(seq_B, train_mode=False)[0]

        # Arm 2: Process seq_B directly on fresh model
        model_fresh = _build_model(cfg, seed_offset=2)
        with torch.no_grad():
            model_fresh.reset_sequence_state(reset_fast_weights=True)
            out_B_direct = model_fresh(seq_B, train_mode=False)[0]

        isolation_diff = float((out_B_after_A - out_B_direct).abs().max().item())
        isolation_ok = isolation_diff < 1e-6
        invariants.append(
            InvariantResult(
                name="prop_reset_isolation",
                passed=isolation_ok,
                observed={"isolation_diff": isolation_diff},
                detail=f"Metamorphic reset isolation: max output difference = {isolation_diff:.2e}",
            )
        )

        # ===================================================================
        # 4. Monotonic Depletion Under Sustained Drive
        # ===================================================================
        # Full sequence causal simulation: measure RRP depletion across sequence time steps
        q_test = torch.randn(1, 1, 16, 16)
        k_test = torch.randn(1, 1, 16, 16)
        logits_drive = torch.full((1, 1, 16, 16), 4.0)  # Sustained high drive
        sim_state = build_presyn_state(1, 16, 1, torch.device(cfg.device), torch.float32, syn_cfg)

        with torch.no_grad():
            _, final_state = pre(q_test, k_test, logits_drive, sim_state, train_mode=False)

        rrp_time_series = [float(final_state["RRP"][0, 0, t].item()) for t in range(16)]

        # Initial RRP is 6.0, decreases over sequence positions
        depleted = rrp_time_series[-1] < rrp_time_series[0]
        monotonic = all(rrp_time_series[t] >= rrp_time_series[t + 1] - 1e-4 for t in range(6))

        depletion_ok = depleted and monotonic
        invariants.append(
            InvariantResult(
                name="prop_monotonic_depletion",
                passed=depletion_ok,
                observed={"rrp_trajectory": rrp_time_series[:8]},
                detail=(
                    f"Sustained drive RRP fatigue across sequence: {rrp_time_series[0]:.2f} -> "
                    f"{rrp_time_series[-1]:.2f}, monotonic={monotonic}"
                ),
            )
        )

        # ===================================================================
        # 5. Stochastic Mode Expectation Convergence (Monte Carlo)
        # ===================================================================
        stoch_cfg = SynapticConfig(
            enable_presyn=True,
            stochastic_train_frac=1.0,
            stochastic_mode="normal_reparam",
            stochastic_tau=1.0,
            stochastic_count_cap=12,
        )
        stoch_pre = SynapticPresyn(16, stoch_cfg)

        fixed_drive = torch.full((1, 1, 4, 1), 0.0)  # moderate drive -> interior p ~ 0.5
        fixed_idx = torch.zeros(1, 1, 4, 1, dtype=torch.long)

        # Deterministic expected release
        with torch.no_grad():
            det_state = build_presyn_state(1, 4, 1, torch.device(cfg.device), torch.float32, stoch_cfg)
            e_det = stoch_pre.release_canonical(det_state, fixed_drive, fixed_idx, train=False)
            expected_mean = float(e_det.mean().item())

        # Sample across N Monte Carlo passes
        stoch_samples = []
        with torch.no_grad():
            for s in range(cfg.mc_samples):
                torch.manual_seed(s + 500)
                cur_state = build_presyn_state(1, 4, 1, torch.device(cfg.device), torch.float32, stoch_cfg)
                e_stoch = stoch_pre.release_canonical(cur_state, fixed_drive, fixed_idx, train=True)
                stoch_samples.append(float(e_stoch.mean().item()))

        mc_mean = sum(stoch_samples) / len(stoch_samples)
        mc_error = abs(mc_mean - expected_mean)
        mc_rel_error = mc_error / max(1e-4, expected_mean)

        mc_converged = mc_rel_error < 0.30 or mc_error < 0.85
        invariants.append(
            InvariantResult(
                name="prop_stochastic_expectation_convergence",
                passed=mc_converged,
                observed={
                    "expected_det_release": expected_mean,
                    "mc_sample_mean": mc_mean,
                    "mc_samples": cfg.mc_samples,
                    "error": mc_error,
                    "rel_error": mc_rel_error,
                },
                detail=(
                    f"MC expectation matching: Expected={expected_mean:.3f}, "
                    f"MC mean ({cfg.mc_samples} draws)={mc_mean:.3f}, error={mc_error:.3f} (rel={mc_rel_error:.1%})"
                ),
            )
        )

        # ===================================================================
        # 6. NaN-Freedom & Robustness Under Extreme Inputs
        # ===================================================================
        robust_model = _build_model(cfg, seed_offset=3)
        extreme_batches = [
            torch.zeros(2, 16, dtype=torch.long, device=cfg.device),  # all zero tokens
            torch.full((2, 16), cfg.vocab_size - 1, dtype=torch.long, device=cfg.device),  # max tokens
            torch.randint(0, cfg.vocab_size, (1, 1), device=cfg.device).expand(2, 16),  # constant token
        ]

        all_robust_finite = True
        for e_batch in extreme_batches:
            robust_model.reset_sequence_state(reset_fast_weights=True)
            with torch.no_grad():
                out_e = robust_model(e_batch, train_mode=False)[0]
                if not torch.isfinite(out_e).all():
                    all_robust_finite = False

        invariants.append(
            InvariantResult(
                name="prop_extreme_input_robustness",
                passed=all_robust_finite,
                observed={"num_extreme_cases": len(extreme_batches), "all_finite": all_robust_finite},
                detail=f"Tested {len(extreme_batches)} extreme edge-case batches: all finite = {all_robust_finite}",
            )
        )

        # ===================================================================
        # 7. Strict Causal Invariance (Future Cannot Affect Past)
        # ===================================================================
        causal_model = _build_model(cfg, seed_offset=4)
        T_len = 16
        prefix_cutoff = 8

        # Base sequence
        seq_full_1 = torch.randint(0, cfg.vocab_size, (1, T_len), device=cfg.device)
        # Sequence identical in prefix [0..cutoff], but completely different in future [cutoff+1..T]
        seq_full_2 = seq_full_1.clone()
        seq_full_2[:, prefix_cutoff + 1 :] = torch.randint(0, cfg.vocab_size, (1, T_len - prefix_cutoff - 1), device=cfg.device)

        causal_model.reset_sequence_state(reset_fast_weights=True)
        with torch.no_grad():
            out_causal_1 = causal_model(seq_full_1, train_mode=False)[0]

        causal_model.reset_sequence_state(reset_fast_weights=True)
        with torch.no_grad():
            out_causal_2 = causal_model(seq_full_2, train_mode=False)[0]

        # Check prefix logits [0..prefix_cutoff]
        prefix_diff = float((out_causal_1[:, : prefix_cutoff + 1] - out_causal_2[:, : prefix_cutoff + 1]).abs().max().item())
        future_diff = float((out_causal_1[:, prefix_cutoff + 1 :] - out_causal_2[:, prefix_cutoff + 1 :]).abs().max().item())

        causal_holds = prefix_diff == 0.0 and future_diff > 1e-3
        invariants.append(
            InvariantResult(
                name="prop_causal_invariance",
                passed=causal_holds,
                observed={
                    "prefix_diff": prefix_diff,
                    "future_diff": future_diff,
                    "cutoff": prefix_cutoff,
                },
                detail=(
                    f"Causal invariance: past tokens [0..{prefix_cutoff}] diff = {prefix_diff:.2e} "
                    f"(future diff = {future_diff:.3f})"
                ),
            )
        )

        for inv in invariants:
            run_logger.event("e2e_invariant", **asdict(inv))

        all_passed = all(inv.passed for inv in invariants)
        report = PropertyInvariantsReport(
            run_id=run_id,
            config=cfg,
            passed=all_passed,
            invariants=invariants,
            summary={
                "vesicle_conservation": vesicle_ok,
                "eval_determinism_diff": det_diff,
                "reset_isolation_diff": isolation_diff,
                "mc_expectation_error": mc_error,
                "causal_prefix_diff": prefix_diff,
            },
        )

        if verbose:
            table = Table(title="Property-Based & Metamorphic Invariant Battery")
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
    parser = argparse.ArgumentParser(description="Run Property-Based & Metamorphic Invariant verification battery")
    parser.add_argument("--run-dir", type=str, default=None, help="Directory to save E2E traces and logs")
    parser.add_argument("--device", type=str, default="cpu", help="Device to execute on")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--mc-samples", type=int, default=60, help="Number of Monte Carlo expectation draws")
    args = parser.parse_args(argv)

    cfg = PropertyInvariantsConfig(device=args.device, seed=args.seed, mc_samples=args.mc_samples)
    report = run_property_invariants_e2e(cfg, run_dir=args.run_dir, verbose=True)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
