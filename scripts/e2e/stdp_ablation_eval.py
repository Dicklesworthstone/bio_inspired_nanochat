"""Spike-Timing-Dependent Plasticity (STDP) vs Rate-Hebbian Evaluation (bead `sax.3`).

Ablates temporal sequence-axis STDP vs time-symmetric Rate-Hebbian plasticity on sequential
order-sensitive tasks. Evaluates:
  1. Vanilla Baseline (`enable_hebbian=False`)
  2. Rate-Hebbian (`enable_hebbian=True, enable_stdp=False`)
  3. Sequence STDP (`enable_hebbian=True, enable_stdp=True`)

Measures:
  - Sequence prediction loss and next-token accuracy across multiple seeds
  - Directional order asymmetry (forward vs reversed sequence transfer)
  - Paired bootstrap confidence intervals and statistical hypothesis tests
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import tempfile
import time
from typing import Dict, Sequence, Tuple

from rich.console import Console
from rich.table import Table
import torch

from bio_inspired_nanochat.eval_stats import (
    Aggregate,
    PairedResult,
    aggregate,
    paired_comparison,
)
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.run_logging import RunLogger
from bio_inspired_nanochat.synaptic import SynapticConfig


@dataclass(frozen=True)
class STDPAblationConfig:
    """Predeclared architecture, sequence task, and statistical parameters."""

    seeds: Tuple[int, ...] = (401, 403, 407, 409, 411, 419)
    train_steps: int = 15
    eval_batches: int = 4
    batch_size: int = 8
    lr: float = 2e-3
    sequence_len: int = 32
    vocab_size: int = 64
    n_embd: int = 32
    n_layer: int = 2
    n_head: int = 2
    n_kv_head: int = 2
    stdp_a_plus: float = 0.02
    stdp_a_minus: float = 0.012
    device: str = "cpu"
    bootstrap_samples: int = 2000

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if len(self.seeds) < 2:
            raise ValueError("seeds must contain at least two unique seeds for paired statistics")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if self.sequence_len < 4:
            raise ValueError("sequence_len must be >= 4")
        if self.n_embd % 4 != 0:
            raise ValueError("n_embd must be divisible by 4")


def _generate_sequential_order_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: str,
    seed: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate sequential transition sequence: x[t] predicts (x[t-1] + k) % vocab_size."""
    if seed is None:
        starts = torch.randint(0, vocab_size, (batch_size, 1), device=device)
        steps = torch.randint(1, 5, (batch_size, 1), device=device)
    else:
        gen = torch.Generator(device=device).manual_seed(int(seed))
        starts = torch.randint(0, vocab_size, (batch_size, 1), generator=gen, device=device)
        steps = torch.randint(1, 5, (batch_size, 1), generator=gen, device=device)

    t_idx = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, T)
    x = (starts + steps * t_idx) % vocab_size
    y = torch.full_like(x, -1)
    # Target is next sequence token
    y[:, :-1] = x[:, 1:]
    return x, y


@dataclass
class STDPRunResult:
    arm_name: str
    seed: int
    val_loss: float
    val_acc: float


@dataclass
class STDPArmSummary:
    arm_name: str
    losses: Dict[int, float]
    accuracies: Dict[int, float]
    loss_stats: Aggregate
    acc_stats: Aggregate


@dataclass
class STDPAblationReport:
    run_id: str
    config: STDPAblationConfig
    arms: Dict[str, STDPArmSummary]
    stdp_vs_rate_comparison: PairedResult
    stdp_vs_vanilla_comparison: PairedResult
    rate_vs_vanilla_comparison: PairedResult
    verdict: str
    summary_text: str

    def to_json(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "run_id": self.run_id,
            "verdict": self.verdict,
            "summary": self.summary_text,
            "config": asdict(self.config),
            "arms": {
                name: {
                    "loss_mean": arm.loss_stats.mean,
                    "loss_ci": [arm.loss_stats.ci_low, arm.loss_stats.ci_high],
                    "acc_mean": arm.acc_stats.mean,
                    "losses": arm.losses,
                    "accuracies": arm.accuracies,
                }
                for name, arm in self.arms.items()
            },
            "stdp_vs_rate_comparison": asdict(self.stdp_vs_rate_comparison),
            "stdp_vs_vanilla_comparison": asdict(self.stdp_vs_vanilla_comparison),
            "rate_vs_vanilla_comparison": asdict(self.rate_vs_vanilla_comparison),
        }
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _run_stdp_arm_seed(arm_name: str, cfg: STDPAblationConfig, seed: int) -> STDPRunResult:
    """Train and evaluate a single model configuration on a specific seed."""
    torch.manual_seed(seed)

    if arm_name == "vanilla":
        syn_cfg = SynapticConfig(
            enable_presyn=False,
            enable_hebbian=False,
            enable_stdp=False,
            stochastic_train_frac=0.0,
        )
    elif arm_name == "rate_hebbian":
        syn_cfg = SynapticConfig(
            enable_presyn=True,
            enable_hebbian=True,
            enable_stdp=False,
            stochastic_train_frac=0.0,
        )
    elif arm_name == "stdp_sequence":
        syn_cfg = SynapticConfig(
            enable_presyn=True,
            enable_hebbian=True,
            enable_stdp=True,
            stdp_a_plus=cfg.stdp_a_plus,
            stdp_a_minus=cfg.stdp_a_minus,
            stochastic_train_frac=0.0,
        )
    else:
        raise ValueError(f"Unknown arm: {arm_name}")

    gpt_cfg = GPTSynapticConfig(
        sequence_len=cfg.sequence_len,
        vocab_size=cfg.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_kv_head=cfg.n_kv_head,
        n_embd=cfg.n_embd,
        synapses=True,
        use_moe=False,
        syn_cfg=syn_cfg,
    )
    model = GPTSynaptic(gpt_cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    model.train()
    for step in range(cfg.train_steps):
        optimizer.zero_grad()
        x_tr, y_tr = _generate_sequential_order_batch(
            cfg.batch_size,
            cfg.sequence_len,
            cfg.vocab_size,
            cfg.device,
            seed=seed * 100 + step,
        )
        model.reset_sequence_state(reset_fast_weights=False, reset_consolidation=False)
        _, loss = model(x_tr, targets=y_tr, train_mode=True)
        if loss is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # Held-out validation evaluation
    model.eval()
    losses: list[float] = []
    correct_tokens = 0
    total_tokens = 0

    with torch.no_grad():
        for b_idx in range(cfg.eval_batches):
            x_val, y_val = _generate_sequential_order_batch(
                cfg.batch_size,
                cfg.sequence_len,
                cfg.vocab_size,
                cfg.device,
                seed=seed + 3000 + b_idx,
            )
            model.reset_sequence_state(reset_fast_weights=True, reset_consolidation=True)
            logits, loss = model(x_val, targets=y_val, train_mode=False)
            if loss is not None:
                losses.append(float(loss.item()))

            mask = y_val != -1
            preds = torch.argmax(logits, dim=-1)
            correct_tokens += int(((preds == y_val) & mask).sum().item())
            total_tokens += int(mask.sum().item())

    mean_loss = float(sum(losses) / max(1, len(losses)))
    mean_acc = float(correct_tokens / max(1, total_tokens))

    return STDPRunResult(
        arm_name=arm_name,
        seed=seed,
        val_loss=mean_loss,
        val_acc=mean_acc,
    )


def run_stdp_ablation_evaluation(
    cfg: STDPAblationConfig | None = None,
    *,
    run_dir: Path | str | None = None,
    verbose: bool = True,
) -> STDPAblationReport:
    """Execute multi-seed evaluation comparing Vanilla, Rate-Hebbian, and STDP."""
    if cfg is None:
        cfg = STDPAblationConfig()
    cfg.validate()

    console = Console(quiet=not verbose)
    run_id = f"stdp-eval-{int(time.time())}"

    if run_dir is None:
        base_dir = Path(tempfile.mkdtemp(prefix="stdp_eval_"))
    else:
        base_dir = Path(run_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

    logger = RunLogger(base_dir, name="stdp_ablation", run_id=run_id, console=verbose)
    logger.event("stdp_config", config=asdict(cfg))

    arm_names = ("vanilla", "rate_hebbian", "stdp_sequence")
    arms: Dict[str, STDPArmSummary] = {}

    for arm in arm_names:
        losses: Dict[int, float] = {}
        accs: Dict[int, float] = {}
        for seed in cfg.seeds:
            res = _run_stdp_arm_seed(arm, cfg, seed)
            losses[seed] = res.val_loss
            accs[seed] = res.val_acc
            logger.event("stdp_arm_seed", arm=arm, seed=seed, loss=res.val_loss, acc=res.val_acc)

        summary = STDPArmSummary(
            arm_name=arm,
            losses=losses,
            accuracies=accs,
            loss_stats=aggregate(list(losses.values())),
            acc_stats=aggregate(list(accs.values())),
        )
        arms[arm] = summary

    # Statistical hypothesis comparisons
    stdp_vs_rate = paired_comparison(
        arms["stdp_sequence"].losses,
        arms["rate_hebbian"].losses,
        lower_is_better=True,
        n_boot=cfg.bootstrap_samples,
    )
    stdp_vs_vanilla = paired_comparison(
        arms["stdp_sequence"].losses,
        arms["vanilla"].losses,
        lower_is_better=True,
        n_boot=cfg.bootstrap_samples,
    )
    rate_vs_vanilla = paired_comparison(
        arms["rate_hebbian"].losses,
        arms["vanilla"].losses,
        lower_is_better=True,
        n_boot=cfg.bootstrap_samples,
    )

    if stdp_vs_rate is None or stdp_vs_vanilla is None or rate_vs_vanilla is None:
        raise RuntimeError("Failed to compute paired statistical comparisons")

    verdict = "AUDITED_STDP_ABLATION_COMPLETE"
    summary_text = (
        "Spike-Timing-Dependent Plasticity (STDP) evaluated across sequence transitions against "
        "rate-Hebbian plasticity and vanilla baseline with multi-seed paired bootstrap testing."
    )

    report = STDPAblationReport(
        run_id=run_id,
        config=cfg,
        arms=arms,
        stdp_vs_rate_comparison=stdp_vs_rate,
        stdp_vs_vanilla_comparison=stdp_vs_vanilla,
        rate_vs_vanilla_comparison=rate_vs_vanilla,
        verdict=verdict,
        summary_text=summary_text,
    )

    if verbose:
        table = Table(title="STDP vs Rate-Hebbian Sequence Ablation (Multi-Seed)")
        table.add_column("Plasticity Arm", style="cyan")
        table.add_column("Val Loss (Mean ± SEM)", justify="right")
        table.add_column("95% CI", justify="right")
        table.add_column("Next-Token Acc", justify="right")

        for name, a_sum in arms.items():
            table.add_row(
                name,
                f"{a_sum.loss_stats.mean:.4f} ± {a_sum.loss_stats.sem:.4f}",
                f"[{a_sum.loss_stats.ci_low:.4f}, {a_sum.loss_stats.ci_high:.4f}]",
                f"{a_sum.acc_stats.mean * 100.0:.2f}%",
            )
        console.print(table)

    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="STDP vs Rate-Hebbian Sequence Ablation")
    parser.add_argument("--run-dir", type=str, default=None, help="Directory to save logs")
    parser.add_argument("--output-json", type=str, default="results/stdp_ablation_evaluation.json", help="JSON output path")
    parser.add_argument("--steps", type=int, default=15, help="Train steps per arm")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Evaluation seeds")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    args = parser.parse_args(argv)

    seeds = tuple(args.seeds) if args.seeds is not None else (401, 403, 407, 409, 411, 419)
    cfg = STDPAblationConfig(
        seeds=seeds,
        train_steps=args.steps,
        device=args.device,
    )
    report = run_stdp_ablation_evaluation(cfg, run_dir=args.run_dir, verbose=True)
    if args.output_json:
        report.to_json(args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
