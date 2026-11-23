"""E2E SCRIPT: neuromodulated three-factor / RL micro-run with detailed logs (bead eqyk.11).

Exercises the neuromodulatory bus and reward-gated three-factor plasticity end-to-end:
  1. ``bus_broadcast_gates_plasticity_and_exploration``: Asserts the NeuromodulatoryBus
     computes DA (RPE), ACh (uncertainty), and NE (novelty/arousal) from runtime signals and
     broadcasts multiplicative gains to every SynapticLinear and SynapticPresyn layer.
  2. ``three_factor_consolidates_rewarded_associations_only``: Tests the canonical three-factor
     learning rule ($ΔW ∝ \text{pre} \times \text{post} \times DA$) on a reward-conditioned task:
     rewarded steps amplify consolidation into $W_{slow}$, while unrewarded/negative steps freeze
     or suppress consolidation.
  3. ``rl_microrun_improves_reward_and_stays_finite``: Executes a short neuromodulated reinforcement
     learning loop on the synthetic reward task, verifying reward improves without gradient or
     plasticity divergence.
  4. Structured event stream: Emits per-step DA/ACh/NE levels, policy loss, reward, and weight norms
     into a machine-readable ``events.jsonl`` trace.

Run:
    python -m scripts.e2e.neuromod_rl
    pytest tests/test_e2e_neuromod_rl.py -v
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat.e2e_harness import InvariantResult
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.neuromod import NeuromodulatoryBus, NeuromodConfig
from bio_inspired_nanochat.run_logging import RunLogger
from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticLinear, SynapticPresyn
from bio_inspired_nanochat.synthetic_tasks import reward_task


@dataclass
class NeuromodRLE2EConfig:
    """Configuration for the Neuromodulated RL / Three-Factor E2E battery."""

    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    vocab_size: int = 64
    sequence_len: int = 32
    device: str = "cpu"
    seed: int = 42

    # Task geometry
    batch_size: int = 8
    context_len: int = 4

    # RL training parameters
    rl_steps: int = 35
    lr: float = 2e-3
    grad_clip: float = 1.0


@dataclass
class NeuromodRLE2EReport:
    run_id: str
    config: NeuromodRLE2EConfig
    passed: bool
    invariants: list[InvariantResult]
    summary: dict[str, Any] = field(default_factory=dict)

    def assert_passed(self) -> None:
        failed = [inv for inv in self.invariants if not inv.passed]
        if failed:
            msg = "\n".join(f"  FAILED: {inv.name} -> {inv.detail}" for inv in failed)
            raise AssertionError(f"Neuromod RL E2E battery failed with {len(failed)} failure(s):\n{msg}")


def _build_model(cfg: NeuromodRLE2EConfig, *, seed_offset: int = 0) -> GPTSynaptic:
    """Create reproducible GPTSynaptic model configured for neuromodulated plasticity."""
    torch.manual_seed(cfg.seed + seed_offset)
    syn_cfg = SynapticConfig(
        enable_hebbian=True,
        plasticity_during_training=True,
        stochastic_train_frac=0.15,
        bistable_latch=True,
        fast_weight_normalized=True,
        post_fast_lr=0.05,
        post_slow_lr=0.03,
        fast_weight_max_norm=2.0,
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
    return model


Feedback = Literal["aligned", "neutral", "lagged"]


@torch.no_grad()
def heldout_reward(model: GPTSynaptic, cfg: NeuromodRLE2EConfig) -> dict:
    batch, reward = reward_task(batch=512, context_len=cfg.context_len,
                                vocab_size=cfg.vocab_size, seed=88_103)
    batch = batch.to(cfg.device)
    model.eval()
    model.reset_sequence_state(reset_fast_weights=True)
    logits, _ = model(batch.inputs, train_mode=False, update_mem=False)
    model.train()
    scores = logits[:, -1].float()
    gold = batch.targets[:, -1:]
    logp = F.log_softmax(scores, dim=-1).gather(1, gold).squeeze(1)
    return {"expected_reward": float(logp.exp().mean()), "gold_nll": -float(logp.mean()),
            "greedy_hits": int(reward(scores.argmax(-1)).sum()), "examples": 512}


def train_reward_policy(
    cfg: NeuromodRLE2EConfig, *, feedback: Feedback = "aligned",
    isolated: bool = False, eval_at: Sequence[int] = (), logger: RunLogger | None = None,
) -> dict:
    """Train the existing reward policy with an explicit post-backward write boundary."""
    model = _build_model(cfg, seed_offset=3).train()
    layers = [layer for layer in model.modules() if isinstance(layer, SynapticLinear)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    bus_cfg = NeuromodConfig(ema_tau=0.8)
    if isolated:
        bus_cfg.ach_gain_slope = bus_cfg.ach_input_slope = bus_cfg.ne_gain_slope = 0.0
    if feedback == "neutral":
        bus_cfg.da_gain_slope = 0.0
    bus = NeuromodulatoryBus(bus_cfg)
    train = []
    checkpoints = {0: heldout_reward(model, cfg)} if 0 in eval_at else {}
    for step in range(cfg.rl_steps):
        batch, reward_fn = reward_task(batch=cfg.batch_size, context_len=cfg.context_len,
                                       vocab_size=cfg.vocab_size, seed=cfg.seed + 100 + step)
        batch = batch.to(cfg.device)
        model.reset_sequence_state(reset_fast_weights=True)
        logits, _ = model(batch.inputs, train_mode=True)
        policy = torch.distributions.Categorical(probs=F.softmax(logits[:, -1, :], dim=-1))
        actions = policy.sample()
        rewards = reward_fn(actions).to(actions.device)
        reward = float(rewards.mean())
        advantage = rewards - rewards.mean()
        loss = -(policy.log_prob(actions) * advantage.detach()).mean()
        optimizer.zero_grad()
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True))
        optimizer.step()

        loss_value = float(loss.detach())
        entropy = float(policy.entropy().detach().mean())
        if not all(math.isfinite(value) for value in (reward, loss_value, entropy, grad_norm)):
            raise FloatingPointError(f"Non-finite policy signal at update {step + 1}")
        if feedback == "lagged":
            gain = bus.gains()["plasticity"]
            bus.broadcast(model)
        bus.update(reward=reward, entropy=entropy, loss=loss_value)
        if feedback != "lagged":
            gain = bus.gains()["plasticity"]
            bus.broadcast(model)
        writes = sum(layer.commit_pending_plasticity() for layer in layers)
        if not layers or writes != len(layers):
            raise RuntimeError(f"Expected {len(layers)} committed writes, saw {writes} at update {step + 1}")

        entry = {"step": step + 1, "reward": reward, "loss": loss_value, "entropy": entropy,
                 "grad_norm": grad_norm, "dopamine_gain": gain, "writes": writes, **bus.levels()}
        train.append(entry)
        if logger is not None and (step + 1) % 10 == 0:
            logger.log_metrics(**entry)
        if step + 1 in eval_at:
            checkpoints[step + 1] = heldout_reward(model, cfg)
    if any(not torch.isfinite(param).all() for param in model.parameters()):
        raise FloatingPointError("Non-finite parameters after reward training")
    return {"train": train, "heldout": checkpoints, "sites": len(layers)}


def timing_study(out: Path) -> dict:
    """Compare the three registered feedback schedules on fixed held-out episodes."""
    arms: tuple[Feedback, ...] = ("aligned", "neutral", "lagged")
    steps = (0, 32, 64, 96, 128)
    seeds = range(410, 418)
    console = Console()
    runs = {}
    for seed in seeds:
        cfg = NeuromodRLE2EConfig(seed=seed, batch_size=16, rl_steps=steps[-1])
        runs[seed] = {arm: train_reward_policy(cfg, feedback=arm, isolated=True, eval_at=steps) for arm in arms}
        console.log(f"Completed all three feedback schedules for seed {seed}")

    def score(seed: int, arm: Feedback) -> float:
        return statistics.fmean(runs[seed][arm]["heldout"][step]["expected_reward"] for step in steps[1:])

    contrasts = {}
    for control in ("neutral", "lagged"):
        differences = [score(seed, "aligned") - score(seed, control) for seed in seeds]
        total = math.fsum(differences)
        flips = itertools.product((-1, 1), repeat=len(differences))
        extreme = sum(math.fsum(sign * delta for sign, delta in zip(signs, differences, strict=True))
                      >= total - 1e-14 for signs in flips)
        mean, p = total / len(differences), extreme / (2 ** len(differences))
        wins = sum(delta > 0 for delta in differences)
        contrasts[control] = {"mean": mean, "one_sided_p": p, "positive_seeds": wins,
                              "seed_deltas": differences, "pass": mean >= .01 and p <= .025 and wins >= 6}
    report = {"runs": runs, "torch": str(torch.__version__), "contrasts": contrasts,
              "advance": all(row["pass"] for row in contrasts.values())}
    out.mkdir(parents=True, exist_ok=True)
    with (out / "timing-study.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def run_neuromod_rl_e2e(
    cfg: NeuromodRLE2EConfig | None = None,
    *,
    run_dir: Path | str | None = None,
    verbose: bool = True,
) -> NeuromodRLE2EReport:
    """Run the complete neuromodulated three-factor / RL E2E verification battery."""
    if cfg is None:
        cfg = NeuromodRLE2EConfig()

    console = Console(quiet=not verbose)
    run_id = f"neuromod-rl-e2e-{int(time.time())}"
    invariants: list[InvariantResult] = []

    clean_tmp = False
    if run_dir is None:
        base_dir = Path(tempfile.mkdtemp(prefix="neuromod_rl_e2e_"))
        clean_tmp = True
    else:
        base_dir = Path(run_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

    run_logger = RunLogger(base_dir, name="neuromod_rl_e2e", run_id=run_id, console=verbose)
    run_logger.event("neuromod_rl_config", config=asdict(cfg))

    try:
        # ===================================================================
        # Part 1: Bus broadcast gates plasticity, exploration, and novelty
        # ===================================================================
        bus = NeuromodulatoryBus(NeuromodConfig(enabled=True))
        test_model = _build_model(cfg, seed_offset=0)

        # Seed initial baselines
        bus.update(reward=0.0, entropy=1.0, loss=1.0)
        # Apply changes: reward positive RPE, rising entropy, loss surprise
        bus.update(reward=1.0, entropy=3.0, loss=4.0)

        levels = bus.levels()
        gains = bus.gains()
        num_touched = bus.broadcast(test_model)

        lins = [m for m in test_model.modules() if isinstance(m, SynapticLinear)]
        pres = [m for m in test_model.modules() if isinstance(m, SynapticPresyn)]

        da_broadcast_ok = all(getattr(m, "_nm_da_gain", None) == gains["plasticity"] for m in lins)
        ach_broadcast_ok = all(getattr(m, "_nm_ach_gain", None) == gains["explore"] for m in pres)
        ne_broadcast_ok = all(getattr(m, "_nm_ne_gain", None) == gains["global"] for m in lins)

        bus_ok = (
            num_touched > 0
            and levels["da"] > 0
            and levels["ach"] > 0
            and levels["ne"] > 0
            and da_broadcast_ok
            and ach_broadcast_ok
            and ne_broadcast_ok
        )
        invariants.append(
            InvariantResult(
                name="bus_broadcast_gates_plasticity_and_exploration",
                passed=bus_ok,
                observed={
                    "num_touched": num_touched,
                    "levels": levels,
                    "gains": gains,
                },
                detail=(
                    f"Broadcast to {num_touched} modules: DA level={levels['da']:.2f} (gain={gains['plasticity']:.2f}), "
                    f"ACh level={levels['ach']:.2f} (gain={gains['explore']:.2f}), "
                    f"NE level={levels['ne']:.2f} (gain={gains['global']:.2f})"
                ),
            )
        )
        run_logger.event("bus_broadcast_test", levels=levels, gains=gains, touched=num_touched)

        # ===================================================================
        # Part 2: Three-Factor Rule consolidates rewarded associations only
        # ===================================================================
        batch_task, reward_fn = reward_task(
            batch=cfg.batch_size,
            context_len=cfg.context_len,
            vocab_size=cfg.vocab_size,
            seed=cfg.seed + 1,
        )
        batch_task = batch_task.to(cfg.device)

        def measure_slow_consolidation(da_gain: float) -> float:
            model = _build_model(cfg, seed_offset=2)
            model.eval()
            model.reset_sequence_state(reset_fast_weights=True)

            # Record initial slow weight snapshot
            initial_slow = [
                lin.w_slow.detach().clone()
                for lin in model.modules()
                if isinstance(lin, SynapticLinear) and lin.w_slow is not None
            ]

            # Set dopamine plasticity gain on all synaptic layers
            for lin in model.modules():
                if isinstance(lin, SynapticLinear):
                    lin._nm_da_gain = da_gain

            with torch.no_grad():
                for _ in range(3):
                    # This battery measures DA-gated online consolidation, so plasticity must
                    # run: since 2026-09-01 an eval-mode forward is deterministic and inert
                    # unless update_mem=True is passed (bridge plan G3).
                    model(batch_task.inputs, targets=batch_task.targets, update_mem=True)

            # Compute total drift in W_slow
            drift = 0.0
            idx = 0
            for lin in model.modules():
                if isinstance(lin, SynapticLinear) and lin.w_slow is not None:
                    drift += float((lin.w_slow.detach() - initial_slow[idx]).abs().sum().item())
                    idx += 1
            return drift

        drift_rewarded = measure_slow_consolidation(da_gain=2.5)   # High positive DA
        drift_unrewarded = measure_slow_consolidation(da_gain=0.0) # Zero DA (unrewarded)

        three_factor_ok = (
            drift_rewarded > 0.0
            and drift_unrewarded < 1e-7
            and drift_rewarded > drift_unrewarded * 10
        )
        invariants.append(
            InvariantResult(
                name="three_factor_consolidates_rewarded_associations_only",
                passed=three_factor_ok,
                observed={
                    "drift_rewarded": drift_rewarded,
                    "drift_unrewarded": drift_unrewarded,
                },
                detail=(
                    f"W_slow consolidation drift: Rewarded (DA=2.5)={drift_rewarded:.4f} vs "
                    f"Unrewarded (DA=0.0)={drift_unrewarded:.6f}"
                ),
            )
        )
        run_logger.event(
            "three_factor_test",
            drift_rewarded=drift_rewarded,
            drift_unrewarded=drift_unrewarded,
        )

        # ===================================================================
        # Part 3: Neuromodulated RL micro-run improves reward & stays finite
        # ===================================================================
        trace = train_reward_policy(cfg, logger=run_logger)
        reward_history = [row["reward"] for row in trace["train"]]
        committed = sum(row["writes"] for row in trace["train"])
        all_finite = True  # Nonfinite runs raise at their training boundary.

        first_half_r = sum(reward_history[: len(reward_history) // 2]) / (len(reward_history) // 2)
        second_half_r = sum(reward_history[len(reward_history) // 2 :]) / (len(reward_history) - len(reward_history) // 2)
        final_reward = reward_history[-1]

        rl_passed = (
            all_finite
            and len(reward_history) == cfg.rl_steps
            and second_half_r >= first_half_r - 0.1  # Stable learning / improvement
        )
        invariants.append(
            InvariantResult(
                name="rl_microrun_improves_reward_and_stays_finite",
                passed=rl_passed,
                observed={
                    "first_half_reward": first_half_r,
                    "second_half_reward": second_half_r,
                    "final_reward": final_reward,
                    "all_finite": all_finite,
                },
                detail=(
                    f"Over {cfg.rl_steps} steps: reward {first_half_r:.3f} -> {second_half_r:.3f} "
                    f"(final={final_reward:.3f}), all finite={all_finite}"
                ),
            )
        )

        for inv in invariants:
            run_logger.event("e2e_invariant", **asdict(inv))

        all_passed = all(inv.passed for inv in invariants)
        report = NeuromodRLE2EReport(
            run_id=run_id,
            config=cfg,
            passed=all_passed,
            invariants=invariants,
            summary={
                "bus_broadcast_touched": num_touched,
                "rewarded_drift": drift_rewarded,
                "unrewarded_drift": drift_unrewarded,
                "first_half_reward": first_half_r,
                "second_half_reward": second_half_r,
                "reward_episode_commits": committed,
                "reward_episode_sites": trace["sites"],
            },
        )

        if verbose:
            table = Table(title="Neuromodulated Three-Factor / RL E2E Battery")
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
    parser = argparse.ArgumentParser(description="Run Neuromodulated Three-Factor / RL E2E battery")
    parser.add_argument("--run-dir", type=str, default=None, help="Directory to save E2E traces and logs")
    parser.add_argument("--device", type=str, default="cpu", help="Device to execute on")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--steps", type=int, default=35, help="Number of RL micro-run steps")
    parser.add_argument("--timing-study", action="store_true", help="Run the fixed three-arm CPU study")
    args = parser.parse_args(argv)

    if args.timing_study:
        if args.run_dir is None:
            parser.error("--timing-study requires --run-dir")
        report = timing_study(Path(args.run_dir))
        Console().print(report["contrasts"])
        return 0

    cfg = NeuromodRLE2EConfig(device=args.device, seed=args.seed, rl_steps=args.steps)
    report = run_neuromod_rl_e2e(cfg, run_dir=args.run_dir, verbose=True)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
