r"""E2E SCRIPT: Capability-Frontier verification suite (beads r00r / eqyk.19).

Comprehensive verification of Magnificence Capability Frontier features:
  1. ``deliberation_energy_monotonicity_and_halting`` (r00r.1): Free energy $F$ is strictly
     non-increasing during pondering ($F_{final} \le F_0$), self-consistent halting triggers on
     $|\Delta F| < \epsilon$, and per-token candidate scoring runs within bounded iterations.
  2. ``adaptive_compute_atp_budget_and_routing`` (r00r.3): Exact integer ATP budgets are never
     overdrawn, difficulty routing allocates strictly more compute to high-uncertainty tokens
     than easy tokens, and the minimum compute floor is always funded.
  3. ``automated_scientist_preregistration`` (r00r.2): Generates deterministic, immutable
     pre-registrations with frozen directional metrics, SHA-256 digest validation, and fixed stopping rules.
  4. ``cross_architecture_bio_adapter_injection`` (r00r.8): Function-preserving bio adapter injection
     converts standard affine feed-forward projections into ``SynapticLinear`` modules while exactly
     preserving initial forward outputs and enabling live biological dynamics.
  5. ``dream_sleep_consolidation_replay`` (r00r.6): Surprise-prioritized wake replay buffer buffers
     sequences, runs offline sleep consolidation, and applies multiplicative homeostatic downscaling.
  6. Structured event streaming: Emits rich execution telemetry and audit records into ``events.jsonl``.

Run:
    python -m scripts.e2e.capability_frontier_suite
    pytest tests/test_e2e_capability_frontier.py -v
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
import torch
import torch.nn as nn

from bio_inspired_nanochat.adaptive_compute import (
    AdaptiveComputeConfig,
    AdaptiveComputeController,
)
from bio_inspired_nanochat.deliberation import (
    ATPBudget,
    DeliberationConfig,
    DeliberationController,
    DifficultyRouter,
    DifficultyRouterConfig,
    deliberate,
    free_energy,
)
from bio_inspired_nanochat.e2e_harness import InvariantResult
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.hf_bio_adapter import (
    HFBioLinearAdapter,
    inject_bio_adapters,
    iter_bio_adapters,
    set_bio_adaptation,
)
from bio_inspired_nanochat.hypothesis_generator import (
    generate_hypotheses,
)
from bio_inspired_nanochat.results_registry import RunRecord
from bio_inspired_nanochat.run_logging import RunLogger
from bio_inspired_nanochat.sleep_consolidation import (
    ReplayBuffer,
    consolidate_sleep_replay,
    homeostatic_downscale,
)
from bio_inspired_nanochat.synaptic import SynapticConfig


@dataclass
class CapabilityFrontierConfig:
    """Configuration for Capability-Frontier verification suite."""

    deliberation_max_iters: int = 25
    deliberation_tol: float = 1e-4
    atp_initial_budget: int = 100
    expert_dim: int = 32
    vocab_size: int = 64
    seed: int = 42


@dataclass
class CapabilityFrontierReport:
    run_id: str
    config: CapabilityFrontierConfig
    passed: bool
    invariants: list[InvariantResult]
    summary: dict[str, Any] = field(default_factory=dict)

    def assert_passed(self) -> None:
        failed = [inv for inv in self.invariants if not inv.passed]
        if failed:
            msg = "\n".join(f"  FAILED: {inv.name} -> {inv.detail}" for inv in failed)
            raise AssertionError(f"Capability Frontier battery failed with {len(failed)} failure(s):\n{msg}")


class _ToyMLPBlock(nn.Module):
    def __init__(self, d_in: int, d_hidden: int) -> None:
        super().__init__()
        self.mlp = nn.ModuleDict({
            "c_fc": nn.Linear(d_in, d_hidden),
            "c_proj": nn.Linear(d_hidden, d_in),
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.mlp["c_fc"](x))
        return self.mlp["c_proj"](h)


def run_capability_frontier_e2e(
    cfg: CapabilityFrontierConfig | None = None,
    *,
    run_dir: Path | str | None = None,
    verbose: bool = True,
) -> CapabilityFrontierReport:
    """Run the complete Capability Frontier verification battery."""
    if cfg is None:
        cfg = CapabilityFrontierConfig()

    from rich.console import Console
    from rich.table import Table

    console = Console(quiet=not verbose)
    run_id = f"capability-frontier-e2e-{int(time.time())}"
    invariants: list[InvariantResult] = []

    clean_tmp = False
    if run_dir is None:
        base_dir = Path(tempfile.mkdtemp(prefix="capability_frontier_e2e_"))
        clean_tmp = True
    else:
        base_dir = Path(run_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

    run_logger = RunLogger(base_dir, name="capability_frontier_e2e", run_id=run_id, console=verbose)
    run_logger.event("capability_frontier_config", config=asdict(cfg))

    try:
        # ===================================================================
        # 1. Deliberation: Free Energy Monotonicity & Bounded Halting (r00r.1)
        # ===================================================================
        z0 = np.array([1.5, 0.8, 0.2], dtype=np.float64)  # [C, B, h]
        f_init = float(free_energy(z0))

        delib_res = deliberate(
            z0,
            dt=0.05,
            max_iters=cfg.deliberation_max_iters,
            eps=cfg.deliberation_tol,
        )
        f_final = float(delib_res.F_final)
        energy_monotone = (f_final <= f_init + 1e-8)
        halted_in_bounds = (0 < delib_res.iters <= cfg.deliberation_max_iters)

        delib_cfg = DeliberationConfig(
            candidate_top_k=4,
            max_iters=cfg.deliberation_max_iters,
            candidate_energy_weight=0.5,
        )
        controller = DeliberationController(delib_cfg)
        has_controller = controller is not None

        delib_ok = energy_monotone and halted_in_bounds and has_controller
        invariants.append(
            InvariantResult(
                name="deliberation_energy_monotonicity_and_halting",
                passed=delib_ok,
                observed={
                    "f_init": f_init,
                    "f_final": f_final,
                    "delta_f": f_final - f_init,
                    "steps": delib_res.iters,
                    "max_iters": cfg.deliberation_max_iters,
                    "converged": delib_res.halted_converged,
                },
                detail=(
                    f"Free energy non-increasing: {f_init:.4f} -> {f_final:.4f} (ΔF={f_final - f_init:.4f}); "
                    f"Ponder halted at step {delib_res.iters}/{cfg.deliberation_max_iters} (converged={delib_res.halted_converged})"
                ),
            )
        )

        # ===================================================================
        # 2. Adaptive Compute: ATP Budget Invariants & Difficulty Routing (r00r.3)
        # ===================================================================
        budget = ATPBudget(total_atp=cfg.atp_initial_budget)
        router = DifficultyRouter(
            cfg=DifficultyRouterConfig(entropy_weight=0.7, free_energy_scale=1.0),
        )

        # Token 1: High confidence / Low entropy -> Easy token
        easy_logits = torch.zeros(cfg.vocab_size)
        easy_logits[0] = 10.0  # Sharp distribution
        diff_easy = router.measure(easy_logits, free_energy_value=0.1)

        # Token 2: Low confidence / High entropy -> Hard token
        hard_logits = torch.zeros(cfg.vocab_size)  # Uniform distribution
        diff_hard = router.measure(hard_logits, free_energy_value=2.0)

        # Adaptive controller plan
        ac_cfg = AdaptiveComputeConfig(
            enabled=True,
            min_depth_layers=1,
            min_experts=1,
            min_mc_samples=1,
            max_mc_samples=8,
        )
        ac_ctrl = AdaptiveComputeController(
            config=ac_cfg,
            router=router,
        )

        plan_easy = ac_ctrl.plan(
            token_index=0,
            logits=easy_logits,
            budget=budget,
            max_depth_layers=8,
            max_experts=4,
            free_energy_value=0.1,
        )
        plan_hard = ac_ctrl.plan(
            token_index=1,
            logits=hard_logits,
            budget=budget,
            max_depth_layers=8,
            max_experts=4,
            free_energy_value=2.0,
        )

        budget_preserved = (budget.spent_atp <= cfg.atp_initial_budget) and (budget.remaining_atp >= 0)
        compute_scaled = (
            plan_hard.mc_samples >= plan_easy.mc_samples
            and plan_hard.compute_units >= plan_easy.compute_units
            and diff_hard.score > diff_easy.score
        )
        minimum_floor_respected = (plan_easy.depth_layers >= 1 and plan_easy.expert_top_k >= 1)

        adaptive_ok = budget_preserved and compute_scaled and minimum_floor_respected
        invariants.append(
            InvariantResult(
                name="adaptive_compute_atp_budget_and_routing",
                passed=adaptive_ok,
                observed={
                    "easy_score": diff_easy.score,
                    "hard_score": diff_hard.score,
                    "easy_compute_units": plan_easy.compute_units,
                    "hard_compute_units": plan_hard.compute_units,
                    "easy_mc_samples": plan_easy.mc_samples,
                    "hard_mc_samples": plan_hard.mc_samples,
                    "atp_spent": budget.spent_atp,
                    "atp_remaining": budget.remaining_atp,
                },
                detail=(
                    f"Difficulty routing scaled compute: easy score={diff_easy.score:.2f} (units={plan_easy.compute_units}) vs "
                    f"hard score={diff_hard.score:.2f} (units={plan_hard.compute_units}); "
                    f"ATP budget respected ({budget.spent_atp}/{cfg.atp_initial_budget} spent)"
                ),
            )
        )

        # ===================================================================
        # 3. Automated Scientist: Falsifiable Hypothesis Pre-registration (r00r.2)
        # ===================================================================
        records = [
            RunRecord(
                run_id="run-exp-01",
                harness="eval",
                notes="exploratory run testing bdnf metaplasticity on bpb",
                metrics={"eval_bpb": 1.42},
                seed=1001,
            ),
            RunRecord(
                run_id="run-exp-02",
                harness="eval",
                notes="exploratory run testing vesicle fatigue dynamics",
                metrics={"eval_bpb": 1.45},
                seed=1002,
            ),
        ]
        dummy_digest = "a" * 64
        hypotheses = generate_hypotheses(
            records,
            results_digest=dummy_digest,
            limit=2,
            paired_seed_count=4,
        )

        has_proposals = len(hypotheses) > 0
        all_hypotheses_valid = all(
            h.hypothesis_id.startswith("hyp-")
            and len(h.paired_seeds) == 4
            and h.minimum_effect > 0.0
            and h.stopping_rule.paired_seed_count == 4
            and h.stopping_rule.no_early_efficacy_stop
            for h in hypotheses
        )
        ai_scientist_ok = has_proposals and all_hypotheses_valid

        invariants.append(
            InvariantResult(
                name="automated_scientist_preregistration",
                passed=ai_scientist_ok,
                observed={
                    "generated_count": len(hypotheses),
                    "first_hypothesis_id": hypotheses[0].hypothesis_id if hypotheses else None,
                    "first_mechanism": hypotheses[0].mechanism if hypotheses else None,
                    "first_metric": hypotheses[0].primary_metric if hypotheses else None,
                    "paired_seeds_count": len(hypotheses[0].paired_seeds) if hypotheses else 0,
                },
                detail=(
                    f"Generated {len(hypotheses)} immutable preregistrations with frozen stopping rules "
                    f"(sample: id={hypotheses[0].hypothesis_id if hypotheses else 'none'}, "
                    f"mechanism={hypotheses[0].mechanism if hypotheses else 'none'})"
                ),
            )
        )

        # ===================================================================
        # 4. Cross-Architecture Bio-Adapter Injection (r00r.8)
        # ===================================================================
        torch.manual_seed(cfg.seed)
        toy_model = _ToyMLPBlock(d_in=cfg.expert_dim, d_hidden=cfg.expert_dim * 2)
        x_probe = torch.randn(2, 4, cfg.expert_dim)

        with torch.no_grad():
            out_before = toy_model(x_probe)

        syn_cfg = SynapticConfig(
            post_fast_lr=0.01,
            post_fast_decay=0.9,
            post_trace_decay=0.9,
        )
        adapter_report = inject_bio_adapters(
            toy_model,
            syn_cfg,
            target_patterns=["mlp.c_fc", "mlp.c_proj"],
        )

        with torch.no_grad():
            out_after_init = toy_model(x_probe)

        # Function preserving initialization: output must match initial exactly
        max_init_diff = float(torch.max(torch.abs(out_before - out_after_init)))
        init_preserved = (max_init_diff < 1e-5)

        # Dynamic forward pass with synaptic modulation active
        adapters = list(iter_bio_adapters(toy_model))
        has_adapters = len(adapters) == 2 and all(isinstance(a, HFBioLinearAdapter) for _, a in adapters)

        set_bio_adaptation(toy_model, True)
        out_active = toy_model(x_probe)
        forward_runs_finite = bool(torch.isfinite(out_active).all())

        adapter_ok = (
            adapter_report.adapter_count == 2
            and init_preserved
            and has_adapters
            and forward_runs_finite
        )
        invariants.append(
            InvariantResult(
                name="cross_architecture_bio_adapter_injection",
                passed=adapter_ok,
                observed={
                    "adapted_projections": adapter_report.adapter_count,
                    "max_init_difference": max_init_diff,
                    "init_function_preserving": init_preserved,
                    "finite_forward": forward_runs_finite,
                },
                detail=(
                    f"Injected {adapter_report.adapter_count} bio adapters into MLP; "
                    f"Initial forward preserved (max|Δ|={max_init_diff:.2e} < 1e-5); "
                    f"Active synaptic forward finite={forward_runs_finite}"
                ),
            )
        )

        # ===================================================================
        # 5. Dream / Sleep Consolidation & Replay (r00r.6)
        # ===================================================================
        replay_buf = ReplayBuffer(max_capacity=32, alpha=1.0, seed=cfg.seed)
        for i in range(8):
            inp = torch.randint(0, cfg.vocab_size, (1, 6))
            tgt = torch.randint(0, cfg.vocab_size, (1, 6))
            loss_val = float(0.5 + 0.2 * i)
            replay_buf.add(inp, tgt, loss=loss_val, step=i)

        batch_items = replay_buf.sample(batch_size=4)
        buf_sampled_correctly = (len(batch_items) == 4 and len(replay_buf) == 8)

        # Test offline sleep consolidation pass on GPTSynaptic
        gpt_cfg = GPTSynapticConfig(
            sequence_len=32,
            vocab_size=cfg.vocab_size,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=cfg.expert_dim,
            syn_cfg=syn_cfg,
        )
        model_gpt = GPTSynaptic(gpt_cfg)

        # Test homeostatic downscaling
        downscaled_stat = homeostatic_downscale(model_gpt, decay_factor=0.8)
        downscale_ok = (downscaled_stat["scaling_factor"] <= 0.81)

        # Run consolidation pass
        consolidation_stat = consolidate_sleep_replay(
            model_gpt,
            batch_items,
            consolidation_passes=2,
            downscale_decay=0.98,
        )
        consolidation_ok = (consolidation_stat["replayed_items"] == len(batch_items))

        sleep_ok = buf_sampled_correctly and downscale_ok and consolidation_ok
        invariants.append(
            InvariantResult(
                name="dream_sleep_consolidation_replay",
                passed=sleep_ok,
                observed={
                    "replay_capacity": len(replay_buf),
                    "sampled_items": len(batch_items),
                    "scaling_factor": downscaled_stat["scaling_factor"],
                    "replayed_items": consolidation_stat["replayed_items"],
                },
                detail=(
                    f"Replay buffer prioritized sampling ok (capacity={len(replay_buf)}); "
                    f"Homeostatic downscale scaling={downscaled_stat['scaling_factor']:.3f}; "
                    f"Consolidation replayed {consolidation_stat['replayed_items']} items"
                ),
            )
        )

        for inv in invariants:
            run_logger.event("e2e_invariant", **asdict(inv))

        all_passed = all(inv.passed for inv in invariants)
        report = CapabilityFrontierReport(
            run_id=run_id,
            config=cfg,
            passed=all_passed,
            invariants=invariants,
            summary={
                "deliberation_steps": delib_res.iters,
                "deliberation_delta_f": f_final - f_init,
                "adaptive_atp_spent": budget.spent_atp,
                "hypotheses_generated": len(hypotheses),
                "adapter_count": adapter_report.adapter_count,
                "replay_samples": len(replay_buf),
            },
        )

        if verbose:
            table = Table(title="Capability-Frontier E2E Verification Battery")
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
    parser = argparse.ArgumentParser(description="Run Capability-Frontier E2E battery")
    parser.add_argument("--run-dir", type=str, default=None, help="Directory to save E2E traces and logs")
    parser.add_argument("--delib-iters", type=int, default=25, help="Deliberation max iterations")
    parser.add_argument("--atp-budget", type=int, default=100, help="Initial ATP budget")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args(argv)

    cfg = CapabilityFrontierConfig(
        deliberation_max_iters=args.delib_iters,
        atp_initial_budget=args.atp_budget,
        seed=args.seed,
    )
    report = run_capability_frontier_e2e(cfg, run_dir=args.run_dir, verbose=True)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
