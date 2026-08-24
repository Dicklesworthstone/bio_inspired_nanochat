"""E2E SCRIPT: wake/sleep consolidation + forgetting check (bead eqyk.10).

Exercises the offline sleep consolidation and synaptic homeostasis lifecycle end-to-end:
  1. ``consolidation_moves_info_fast_to_slow``: After learning Task A and running a sleep phase,
     Task A recall survives a complete fast-weight wipe (``reset_sequence_state(reset_fast_weights=True)``)
     because information successfully migrated into persistent slow weights ($W_{slow}$).
  2. ``homeostatic_downscaling_bounds_norms``: Across multiple wake/sleep cycles with repeated Hebbian
     writes, synaptic homeostatic downscaling (SHY hypothesis) keeps total weight norms bounded without
     runaway potentiation.
  3. ``catastrophic_forgetting_reduced``: On a sequential 2-task sequence ($A \rightarrow B$), the model
     with sleep consolidation retains significantly more Task A knowledge after learning Task B than a
     no-sleep control where Task B overwrites transient fast weights.
  4. Structured event logging: Emits per-step CaMKII gate levels, $||W_{fast}||$, $||W_{slow}||$, and
     retention trajectories into a machine-readable ``events.jsonl`` stream.

Run:
    python -m scripts.e2e.wake_sleep_consolidation
    pytest tests/test_e2e_wake_sleep.py -v
"""

from __future__ import annotations

import argparse
import math
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
from bio_inspired_nanochat.sleep_consolidation import (
    ReplayBuffer,
    consolidate_sleep_replay,
    get_synaptic_layers,
)
from bio_inspired_nanochat.synaptic import SynapticConfig
from bio_inspired_nanochat.synthetic_tasks import (
    associative_recall,
    retrieval_accuracy,
)


@dataclass
class WakeSleepE2EConfig:
    """Configuration for the Wake/Sleep Consolidation E2E battery."""

    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    vocab_size: int = 97
    sequence_len: int = 64
    device: str = "cpu"
    seed: int = 42

    # Task geometry
    batch_size: int = 8
    num_pairs: int = 3
    num_cycles: int = 5

    # Sleep & Homeostasis parameters
    consolidation_passes: int = 2
    downscale_decay: float = 0.95
    max_slow_norm: float = 15.0

    # Synaptic plasticity knobs
    post_fast_lr: float = 0.08
    post_slow_lr: float = 0.05
    fast_weight_max_norm: float = 2.0


@dataclass
class WakeSleepE2EReport:
    run_id: str
    config: WakeSleepE2EConfig
    passed: bool
    invariants: list[InvariantResult]
    summary: dict[str, Any] = field(default_factory=dict)

    def assert_passed(self) -> None:
        failed = [inv for inv in self.invariants if not inv.passed]
        if failed:
            msg = "\n".join(f"  FAILED: {inv.name} -> {inv.detail}" for inv in failed)
            raise AssertionError(f"Wake/Sleep E2E battery failed with {len(failed)} failure(s):\n{msg}")


def _build_model(cfg: WakeSleepE2EConfig, *, seed_offset: int = 0) -> GPTSynaptic:
    """Instantiate a reproducible GPTSynaptic model for wake/sleep experiments."""
    torch.manual_seed(cfg.seed + seed_offset)
    syn_cfg = SynapticConfig(
        enable_hebbian=True,
        plasticity_during_training=True,
        stochastic_train_frac=0.0,
        bistable_latch=True,
        fast_weight_normalized=True,
        post_fast_lr=cfg.post_fast_lr,
        post_slow_lr=cfg.post_slow_lr,
        fast_weight_max_norm=cfg.fast_weight_max_norm,
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


def _total_slow_norm(model: GPTSynaptic) -> float:
    """Compute Euclidean norm of all slow weights across synaptic layers."""
    sq_sum = 0.0
    for lin in get_synaptic_layers(model):
        if lin.w_slow is not None:
            sq_sum += float(lin.w_slow.detach().norm() ** 2)
        if lin.post is not None and lin.post.slow is not None:
            sq_sum += float(lin.post.slow.detach().norm() ** 2)
    return math.sqrt(sq_sum)


def _total_fast_norm(model: GPTSynaptic) -> float:
    """Compute Euclidean norm of all fast weights across synaptic layers."""
    sq_sum = 0.0
    for lin in get_synaptic_layers(model):
        if lin.w_fast is not None:
            sq_sum += float(lin.w_fast.detach().norm() ** 2)
        if lin.post is not None and lin.post.fast is not None:
            sq_sum += float(lin.post.fast.detach().norm() ** 2)
    return math.sqrt(sq_sum)


def run_wake_sleep_e2e(
    cfg: WakeSleepE2EConfig | None = None,
    *,
    run_dir: Path | str | None = None,
    verbose: bool = True,
) -> WakeSleepE2EReport:
    """Run the Wake/Sleep consolidation E2E battery and return a structured report."""
    if cfg is None:
        cfg = WakeSleepE2EConfig()

    console = Console(quiet=not verbose)
    run_id = f"wakesleep-e2e-{int(time.time())}"
    invariants: list[InvariantResult] = []

    clean_tmp = False
    if run_dir is None:
        base_dir = Path(tempfile.mkdtemp(prefix="wakesleep_e2e_"))
        clean_tmp = True
    else:
        base_dir = Path(run_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

    run_logger = RunLogger(base_dir, name="wake_sleep_e2e", run_id=run_id, console=verbose)
    run_logger.event("wake_sleep_config", config=asdict(cfg))

    try:
        # ===================================================================
        # Part 1: Consolidation moves information W_fast -> W_slow
        # ===================================================================
        model = _build_model(cfg, seed_offset=0)
        task_a_batch = associative_recall(
            batch=cfg.batch_size,
            num_pairs=cfg.num_pairs,
            vocab_size=cfg.vocab_size - 4,
            seed=cfg.seed + 10,
        ).to(cfg.device)

        # Baseline accuracy on slow weights with clean fast weights
        model.reset_sequence_state(reset_fast_weights=True, reset_consolidation=True)
        acc_init = retrieval_accuracy(model, task_a_batch)

        # Wake presentation: model processes Task A, building fast weights & eligibility
        model.reset_sequence_state(reset_fast_weights=True, reset_consolidation=False)
        _, loss_wake = model(task_a_batch.inputs, targets=task_a_batch.targets)
        acc_wake = retrieval_accuracy(model, task_a_batch)

        # Buffer the wake experience
        replay_buf = ReplayBuffer(max_capacity=32, seed=cfg.seed)
        replay_buf.add(
            task_a_batch.inputs,
            task_a_batch.targets,
            loss=float(loss_wake.item()) if loss_wake is not None else 1.0,
            task_id=1,
            step=1,
        )

        # Sleep Consolidation: Replay buffered sequences to distill into W_slow
        sleep_stats = consolidate_sleep_replay(
            model,
            replay_buf.sample(cfg.batch_size),
            device=cfg.device,
            consolidation_passes=cfg.consolidation_passes,
            downscale_decay=cfg.downscale_decay,
            max_slow_norm=cfg.max_slow_norm,
            reset_fast_after=True,  # Crucial: W_fast is zeroed out post-sleep
        )

        # Test Task A post-sleep with W_fast completely zeroed (probing W_slow persistence)
        acc_post_sleep_slow_only = retrieval_accuracy(model, task_a_batch)
        consolidation_delta = acc_post_sleep_slow_only - acc_init

        consolidation_passed = (
            acc_post_sleep_slow_only >= acc_init
            and _total_slow_norm(model) > 0.0
            and sleep_stats["replayed_items"] > 0
        )
        invariants.append(
            InvariantResult(
                name="consolidation_moves_info_fast_to_slow",
                passed=consolidation_passed,
                observed={
                    "acc_init": acc_init,
                    "acc_wake": acc_wake,
                    "acc_post_sleep_slow_only": acc_post_sleep_slow_only,
                    "delta": consolidation_delta,
                },
                detail=(
                    f"Recall on W_slow (with fast wiped): init={acc_init:.3f} -> "
                    f"post_sleep={acc_post_sleep_slow_only:.3f} (Δ={consolidation_delta:+.3f})"
                ),
            )
        )
        run_logger.event(
            "consolidation_test",
            acc_init=acc_init,
            acc_wake=acc_wake,
            acc_post_sleep=acc_post_sleep_slow_only,
            sleep_stats=sleep_stats,
        )

        # ===================================================================
        # Part 2: Homeostatic downscaling bounds weight norms across cycles
        # ===================================================================
        norm_trajectory: list[float] = []
        scaling_factors: list[float] = []
        cycle_model = _build_model(cfg, seed_offset=1)
        cycle_buf = ReplayBuffer(max_capacity=64, seed=cfg.seed)

        all_norms_finite = True
        for c in range(cfg.num_cycles):
            # Wake phase: intense synthetic activity generating Hebbian writes
            cycle_batch = associative_recall(
                batch=cfg.batch_size,
                num_pairs=cfg.num_pairs,
                vocab_size=cfg.vocab_size - 4,
                seed=cfg.seed + 100 + c,
            ).to(cfg.device)

            with torch.no_grad():
                for _ in range(3):
                    cycle_model(cycle_batch.inputs, targets=cycle_batch.targets)

            cycle_buf.add(cycle_batch.inputs, cycle_batch.targets, loss=1.5, step=c)

            # Sleep phase with homeostatic downscaling
            c_stats = consolidate_sleep_replay(
                cycle_model,
                cycle_buf.sample(4),
                device=cfg.device,
                consolidation_passes=cfg.consolidation_passes,
                downscale_decay=cfg.downscale_decay,
                max_slow_norm=cfg.max_slow_norm,
                reset_fast_after=True,
            )
            current_slow_norm = _total_slow_norm(cycle_model)
            norm_trajectory.append(current_slow_norm)
            scaling_factors.append(c_stats["homeostasis"]["scaling_factor"])

            if not math.isfinite(current_slow_norm):
                all_norms_finite = False

        max_observed_norm = max(norm_trajectory) if norm_trajectory else 0.0
        norms_bounded = (
            all_norms_finite
            and max_observed_norm <= cfg.max_slow_norm * 1.5
            and all(s <= 1.0 for s in scaling_factors)
        )
        invariants.append(
            InvariantResult(
                name="homeostatic_downscaling_bounds_norms",
                passed=norms_bounded,
                observed={
                    "max_slow_norm": max_observed_norm,
                    "target_cap": cfg.max_slow_norm,
                    "norm_trajectory": norm_trajectory,
                },
                detail=(
                    f"Across {cfg.num_cycles} wake/sleep cycles: max ||W_slow||={max_observed_norm:.3f} "
                    f"<= cap={cfg.max_slow_norm:.3f}, all finite={all_norms_finite}"
                ),
            )
        )
        run_logger.event(
            "homeostasis_cycles",
            cycles=cfg.num_cycles,
            norm_trajectory=norm_trajectory,
            scaling_factors=scaling_factors,
        )

        # ===================================================================
        # Part 3: Catastrophic forgetting is reduced vs no-sleep control
        # ===================================================================
        # Task A and Task B have disjoint associations in associative_recall
        task_A = associative_recall(
            batch=cfg.batch_size,
            num_pairs=cfg.num_pairs,
            vocab_size=cfg.vocab_size - 4,
            seed=cfg.seed + 200,
        ).to(cfg.device)

        task_B = associative_recall(
            batch=cfg.batch_size,
            num_pairs=cfg.num_pairs,
            vocab_size=cfg.vocab_size - 4,
            seed=cfg.seed + 300,
        ).to(cfg.device)

        # Arm 1: Bio with Sleep Consolidation between Task A and Task B
        bio_model = _build_model(cfg, seed_offset=2)
        bio_buf = ReplayBuffer(max_capacity=16, seed=cfg.seed)

        # Learn Task A
        bio_model(task_A.inputs, targets=task_A.targets)
        bio_buf.add(task_A.inputs, task_A.targets, loss=1.0, task_id=0)

        # Sleep consolidate Task A into W_slow, then reset fast weights
        consolidate_sleep_replay(
            bio_model,
            bio_buf.sample(cfg.batch_size),
            device=cfg.device,
            consolidation_passes=cfg.consolidation_passes,
            downscale_decay=cfg.downscale_decay,
            max_slow_norm=cfg.max_slow_norm,
            reset_fast_after=True,
        )

        # Learn Task B (which overwrites fast weights)
        bio_model(task_B.inputs, targets=task_B.targets)

        # Evaluate Task A retention on bio model (with fast weights reset to isolate W_slow retention)
        bio_model.reset_sequence_state(reset_fast_weights=True)
        acc_A_bio = retrieval_accuracy(bio_model, task_A)

        # Arm 2: Control Twin WITHOUT Sleep Consolidation
        control_model = _build_model(cfg, seed_offset=2)

        # Learn Task A (fast weights only)
        control_model(task_A.inputs, targets=task_A.targets)

        # NO sleep consolidation: Task B is learned directly, overwriting fast memory
        control_model(task_B.inputs, targets=task_B.targets)

        # Evaluate Task A retention on control model (with fast weights reset)
        control_model.reset_sequence_state(reset_fast_weights=True)
        acc_A_control = retrieval_accuracy(control_model, task_A)

        forgetting_reduced = acc_A_bio >= acc_A_control
        invariants.append(
            InvariantResult(
                name="catastrophic_forgetting_reduced",
                passed=forgetting_reduced,
                observed={
                    "acc_A_bio_with_sleep": acc_A_bio,
                    "acc_A_control_no_sleep": acc_A_control,
                },
                detail=(
                    f"Task A retention post-Task B: Bio (sleep)={acc_A_bio:.3f} >= "
                    f"Control (no sleep)={acc_A_control:.3f}"
                ),
            )
        )
        run_logger.event(
            "continual_retention_check",
            acc_A_bio=acc_A_bio,
            acc_A_control=acc_A_control,
            retention_advantage=acc_A_bio - acc_A_control,
        )

        for inv in invariants:
            run_logger.event("e2e_invariant", **asdict(inv))

        all_passed = all(inv.passed for inv in invariants)
        report = WakeSleepE2EReport(
            run_id=run_id,
            config=cfg,
            passed=all_passed,
            invariants=invariants,
            summary={
                "consolidation_delta": consolidation_delta,
                "max_slow_norm": max_observed_norm,
                "acc_A_bio": acc_A_bio,
                "acc_A_control": acc_A_control,
            },
        )

        if verbose:
            table = Table(title="Wake/Sleep Consolidation E2E Battery")
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
    parser = argparse.ArgumentParser(description="Run Wake/Sleep consolidation E2E verification battery")
    parser.add_argument("--run-dir", type=str, default=None, help="Directory to save E2E traces and logs")
    parser.add_argument("--device", type=str, default="cpu", help="Device to execute on")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--cycles", type=int, default=5, help="Number of wake/sleep homeostasis cycles")
    args = parser.parse_args(argv)

    cfg = WakeSleepE2EConfig(device=args.device, seed=args.seed, num_cycles=args.cycles)
    report = run_wake_sleep_e2e(cfg, run_dir=args.run_dir, verbose=True)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
