"""Synthetic kinetics comparison smoke harness (bead `yw9.6`).

Executes a shared-schedule, multi-seed comparison across:
1. `default`: Static hand-tuned biophysical kinetics constants
2. `candidate`: Hand-entered candidate kinetics constants with no optimizer provenance
3. `learned`: End-to-end SGD-learned differentiable synaptic kinetics

The task is a deterministic synthetic delayed-copy proxy. It exercises the model and
statistics plumbing, but it is not the real-data evaluation or reproduced CMA-ES artifact
required for the headline `yw9.6` comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
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


# Unverified candidates retained only to exercise a distinct static-kinetics arm. No
# committed optimizer artifact establishes these values as a CMA-ES result.
UNVERIFIED_CANDIDATE_KINETICS = {
    "tau_c": 6.2,
    "alpha_ca": 0.65,
    "syt_fast_kd": 0.35,
    "syt_slow_kd": 1.10,
    "doc2_gain": 0.12,
    "complexin_bias": 0.05,
    "prime_rate": 0.085,
    "unprime_per_release": 0.045,
}


@dataclass(frozen=True)
class KineticsAblationConfig:
    """Predeclared task, model, and statistical configuration."""

    seeds: Tuple[int, ...] = (101, 103, 107, 109, 113, 127)
    train_steps: int = 20
    eval_batches: int = 4
    batch_size: int = 8
    lr: float = 2e-3
    sequence_len: int = 32
    vocab_size: int = 64
    n_embd: int = 32
    n_layer: int = 2
    n_head: int = 2
    n_kv_head: int = 2
    device: str = "cpu"
    bootstrap_samples: int = 2000

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if len(self.seeds) < 2:
            raise ValueError("seeds must contain at least two unique seeds for paired statistics")
        if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be non-negative integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        for name in (
            "train_steps",
            "eval_batches",
            "batch_size",
            "sequence_len",
            "vocab_size",
            "n_embd",
            "n_layer",
            "n_head",
            "n_kv_head",
            "bootstrap_samples",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.sequence_len % 2 != 0:
            raise ValueError("sequence_len must be even for the delayed-copy task")
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_kv_head > self.n_head or self.n_head % self.n_kv_head != 0:
            raise ValueError("n_kv_head must not exceed n_head and must divide it exactly")
        if isinstance(self.lr, bool) or self.lr <= 0.0 or not math.isfinite(self.lr):
            raise ValueError("lr must be positive and finite")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string")


@dataclass
class SingleSeedOutcome:
    seed: int
    mode: str
    train_loss: float
    val_loss: float
    val_acc: float
    learned_kinetics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ArmResult:
    mode: str
    outcomes: list[SingleSeedOutcome]
    losses: Dict[int, float]
    accuracies: Dict[int, float]
    loss_stats: Aggregate
    acc_stats: Aggregate


@dataclass
class KineticsAblationReport:
    run_id: str
    config: KineticsAblationConfig
    arms: Dict[str, ArmResult]
    comparisons: Dict[str, PairedResult]
    verdict: str
    summary_text: str
    supports_headline_claim: bool = False

    def to_json(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "run_id": self.run_id,
            "verdict": self.verdict,
            "summary": self.summary_text,
            "supports_headline_claim": self.supports_headline_claim,
            "config": asdict(self.config),
            "arms": {
                name: {
                    "mode": arm.mode,
                    "loss_mean": arm.loss_stats.mean,
                    "loss_ci": [arm.loss_stats.ci_low, arm.loss_stats.ci_high],
                    "acc_mean": arm.acc_stats.mean,
                    "acc_ci": [arm.acc_stats.ci_low, arm.acc_stats.ci_high],
                    "losses": arm.losses,
                    "accuracies": arm.accuracies,
                }
                for name, arm in self.arms.items()
            },
            "comparisons": {
                name: asdict(comp) for name, comp in self.comparisons.items()
            },
        }
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _generate_associative_recall_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: str,
    seed: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate the deterministic delayed-copy proxy task.

    Inputs: [c_1, c_2, ..., c_k, c_1, c_2, ..., c_k]
    Targets: Predict the second repeated half from working-memory traces.
    """
    half = seq_len // 2
    if seed is None:
        data = torch.randint(0, vocab_size, (batch_size, half), device=device)
    else:
        gen = torch.Generator(device=device).manual_seed(int(seed))
        data = torch.randint(0, vocab_size, (batch_size, half), generator=gen, device=device)

    x = torch.cat([data, data], dim=1)
    y = torch.full_like(x, -1)
    # Target loss is scored strictly on the second half prediction
    y[:, half - 1 : seq_len - 1] = x[:, half:seq_len]
    return x, y


def _build_model(mode: str, cfg: KineticsAblationConfig, seed: int) -> GPTSynaptic:
    """Build a model configured for default, candidate, or learned kinetics."""
    torch.manual_seed(seed)

    if mode == "default":
        syn_cfg = SynapticConfig(
            enable_presyn=True,
            learnable_kinetics=False,
            stochastic_train_frac=0.0,
        )
    elif mode == "candidate":
        syn_cfg = SynapticConfig(
            enable_presyn=True,
            learnable_kinetics=False,
            stochastic_train_frac=0.0,
            tau_c=UNVERIFIED_CANDIDATE_KINETICS["tau_c"],
            alpha_ca=UNVERIFIED_CANDIDATE_KINETICS["alpha_ca"],
            syt_fast_kd=UNVERIFIED_CANDIDATE_KINETICS["syt_fast_kd"],
            syt_slow_kd=UNVERIFIED_CANDIDATE_KINETICS["syt_slow_kd"],
            doc2_gain=UNVERIFIED_CANDIDATE_KINETICS["doc2_gain"],
            complexin_bias=UNVERIFIED_CANDIDATE_KINETICS["complexin_bias"],
            prime_rate=UNVERIFIED_CANDIDATE_KINETICS["prime_rate"],
            unprime_per_release=UNVERIFIED_CANDIDATE_KINETICS["unprime_per_release"],
        )
    elif mode == "learned":
        syn_cfg = SynapticConfig(
            enable_presyn=True,
            learnable_kinetics=True,
            stochastic_train_frac=0.0,
        )
    else:
        raise ValueError(f"Unknown kinetics mode: {mode!r}")

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
    return model


def _evaluate_model(
    model: GPTSynaptic,
    cfg: KineticsAblationConfig,
    eval_seed: int,
) -> Tuple[float, float]:
    """Evaluate validation cross-entropy loss and recall accuracy on held-out data."""
    model.eval()
    losses: list[float] = []
    correct_tokens = 0
    total_tokens = 0

    with torch.no_grad():
        for b_idx in range(cfg.eval_batches):
            x_val, y_val = _generate_associative_recall_batch(
                cfg.batch_size,
                cfg.sequence_len,
                cfg.vocab_size,
                cfg.device,
                seed=eval_seed + 1000 + b_idx,
            )
            # Keep trained fast Parameters intact while clearing only per-sequence traces.
            # reset_fast_weights=True would erase part of the trained model before scoring.
            model.reset_sequence_state(reset_fast_weights=False, reset_consolidation=True)
            logits, loss = model(x_val, targets=y_val, train_mode=False)
            if loss is not None:
                losses.append(float(loss.item()))

            # Accuracy on scored positions (y_val != -1)
            mask = y_val != -1
            preds = torch.argmax(logits, dim=-1)
            correct_tokens += int(((preds == y_val) & mask).sum().item())
            total_tokens += int(mask.sum().item())

    mean_val_loss = float(sum(losses) / max(1, len(losses)))
    mean_val_acc = float(correct_tokens / max(1, total_tokens))
    return mean_val_loss, mean_val_acc


def _run_single_arm(
    mode: str,
    cfg: KineticsAblationConfig,
    seed: int,
    logger: RunLogger | None = None,
) -> SingleSeedOutcome:
    """Train and evaluate an individual arm for a given seed."""
    model = _build_model(mode, cfg, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    train_losses: list[float] = []
    model.train()

    for step in range(cfg.train_steps):
        optimizer.zero_grad()
        x_tr, y_tr = _generate_associative_recall_batch(
            cfg.batch_size,
            cfg.sequence_len,
            cfg.vocab_size,
            cfg.device,
            seed=seed * 100 + step,
        )
        model.reset_sequence_state(reset_fast_weights=False, reset_consolidation=False)
        _, loss = model(x_tr, targets=y_tr, train_mode=True)
        if loss is not None:
            train_losses.append(float(loss.item()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # Extract learned kinetic parameter values if learnable
    learned_kinetics: Dict[str, float] = {}
    if mode == "learned":
        for name, param in model.named_parameters():
            if "kinetics_" in name:
                learned_kinetics[name] = float(param.detach().norm().item())

    final_train_loss = float(sum(train_losses[-3:]) / max(1, len(train_losses[-3:])))
    val_loss, val_acc = _evaluate_model(model, cfg, eval_seed=seed)

    outcome = SingleSeedOutcome(
        seed=seed,
        mode=mode,
        train_loss=final_train_loss,
        val_loss=val_loss,
        val_acc=val_acc,
        learned_kinetics=learned_kinetics,
    )

    if logger is not None:
        logger.event("kinetics_arm_outcome", mode=mode, seed=seed, val_loss=val_loss, val_acc=val_acc)

    return outcome


def run_kinetics_ablation(
    cfg: KineticsAblationConfig | None = None,
    *,
    run_dir: Path | str | None = None,
    verbose: bool = True,
) -> KineticsAblationReport:
    """Run the synthetic multi-seed comparison across default, candidate, and learned."""
    if cfg is None:
        cfg = KineticsAblationConfig()
    cfg.validate()

    console = Console(quiet=not verbose)
    run_id = f"kinetics-ablation-{int(time.time())}"

    if run_dir is None:
        base_dir = Path(tempfile.mkdtemp(prefix="kinetics_ablation_"))
    else:
        base_dir = Path(run_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

    logger = RunLogger(base_dir, name="kinetics_ablation", run_id=run_id, console=verbose)
    logger.event("kinetics_config", config=asdict(cfg))

    modes = ("default", "candidate", "learned")
    arms: Dict[str, ArmResult] = {}

    for mode in modes:
        outcomes: list[SingleSeedOutcome] = []
        losses: Dict[int, float] = {}
        accs: Dict[int, float] = {}

        for seed in cfg.seeds:
            out = _run_single_arm(mode, cfg, seed, logger=logger)
            outcomes.append(out)
            losses[seed] = out.val_loss
            accs[seed] = out.val_acc

        arm_res = ArmResult(
            mode=mode,
            outcomes=outcomes,
            losses=losses,
            accuracies=accs,
            loss_stats=aggregate(list(losses.values())),
            acc_stats=aggregate(list(accs.values())),
        )
        arms[mode] = arm_res

    # Paired statistical comparisons
    comp_learned_vs_default = paired_comparison(
        arms["learned"].losses,
        arms["default"].losses,
        lower_is_better=True,
        n_boot=cfg.bootstrap_samples,
    )
    comp_learned_vs_candidate = paired_comparison(
        arms["learned"].losses,
        arms["candidate"].losses,
        lower_is_better=True,
        n_boot=cfg.bootstrap_samples,
    )
    comp_candidate_vs_default = paired_comparison(
        arms["candidate"].losses,
        arms["default"].losses,
        lower_is_better=True,
        n_boot=cfg.bootstrap_samples,
    )

    if (
        comp_learned_vs_default is None
        or comp_learned_vs_candidate is None
        or comp_candidate_vs_default is None
    ):
        raise RuntimeError("Failed to compute paired comparisons across shared seeds")

    comparisons: Dict[str, PairedResult] = {
        "learned_vs_default": comp_learned_vs_default,
        "learned_vs_candidate": comp_learned_vs_candidate,
        "candidate_vs_default": comp_candidate_vs_default,
    }

    # These labels describe only this synthetic run. A CI spanning zero is inconclusive;
    # it is not evidence of parity or statistical equivalence.
    if (
        comp_learned_vs_default.delta_ci_high < 0.0
        and comp_learned_vs_candidate.delta_ci_high < 0.0
    ):
        verdict = "OBSERVED_GAIN"
        summary_text = (
            "The learned arm had lower loss than both static arms in this synthetic "
            "delayed-copy run; this does not establish the yw9.6 headline claim."
        )
    elif (
        comp_learned_vs_default.delta_ci_low > 0.0
        or comp_learned_vs_candidate.delta_ci_low > 0.0
    ):
        verdict = "OBSERVED_REGRESSION"
        summary_text = (
            "The learned arm had higher loss than at least one static arm in this synthetic "
            "delayed-copy run; this does not establish the yw9.6 headline claim."
        )
    else:
        verdict = "INCONCLUSIVE"
        summary_text = (
            "This synthetic delayed-copy run did not distinguish the learned arm from both "
            "static arms; a confidence interval spanning zero is not evidence of equivalence."
        )

    report = KineticsAblationReport(
        run_id=run_id,
        config=cfg,
        arms=arms,
        comparisons=comparisons,
        verdict=verdict,
        summary_text=summary_text,
    )

    if verbose:
        table = Table(title="Synthetic Kinetics Comparison (Shared Training Schedule)")
        table.add_column("Mode", style="cyan")
        table.add_column("Val Loss (Mean ± SEM)", justify="right")
        table.add_column("95% Bootstrap CI", justify="right")
        table.add_column("Val Accuracy", justify="right")

        for mode, arm in arms.items():
            table.add_row(
                mode,
                f"{arm.loss_stats.mean:.4f} ± {arm.loss_stats.sem:.4f}",
                f"[{arm.loss_stats.ci_low:.4f}, {arm.loss_stats.ci_high:.4f}]",
                f"{arm.acc_stats.mean * 100.0:.1f}%",
            )
        console.print(table)

        comp_table = Table(title="Paired Statistical Comparisons (Loss Δ, Lower is Better)")
        comp_table.add_column("Comparison", style="cyan")
        comp_table.add_column("Mean Δ", justify="right")
        comp_table.add_column("95% CI Δ", justify="right")
        comp_table.add_column("t-test p", justify="right")
        comp_table.add_column("Wilcoxon p", justify="right")
        comp_table.add_column("Cohen d_z", justify="right")

        for name, comp in comparisons.items():
            comp_table.add_row(
                name,
                f"{comp.mean_delta:+.4f}",
                f"[{comp.delta_ci_low:+.4f}, {comp.delta_ci_high:+.4f}]",
                f"{comp.t_p_value:.4e}" if comp.t_p_value is not None else "N/A",
                f"{comp.wilcoxon_p_value:.4e}" if comp.wilcoxon_p_value is not None else "N/A",
                f"{comp.cohen_dz:.3f}" if comp.cohen_dz is not None else "N/A",
            )
        console.print(comp_table)
        console.print(f"[bold]Verdict:[/bold] {verdict} — {summary_text}")

    return report


def main(argv: Sequence[str] | None = None) -> int:
    defaults = KineticsAblationConfig()
    parser = argparse.ArgumentParser(description="Synthetic kinetics multi-seed comparison")
    parser.add_argument("--run-dir", type=str, default=None, help="Directory to save logs")
    parser.add_argument("--output-json", type=str, default="results/kinetics_ablation_evaluation.json", help="JSON output path")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(defaults.seeds))
    parser.add_argument("--steps", type=int, default=defaults.train_steps, help="Train steps")
    parser.add_argument("--eval-batches", type=int, default=defaults.eval_batches)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--sequence-len", type=int, default=defaults.sequence_len)
    parser.add_argument("--vocab-size", type=int, default=defaults.vocab_size)
    parser.add_argument("--n-embd", type=int, default=defaults.n_embd)
    parser.add_argument("--n-layer", type=int, default=defaults.n_layer)
    parser.add_argument("--n-head", type=int, default=defaults.n_head)
    parser.add_argument("--n-kv-head", type=int, default=defaults.n_kv_head)
    parser.add_argument("--bootstrap-samples", type=int, default=defaults.bootstrap_samples)
    parser.add_argument("--device", type=str, default=defaults.device, help="Device: cpu or cuda")
    args = parser.parse_args(argv)

    cfg = KineticsAblationConfig(
        seeds=tuple(args.seeds),
        train_steps=args.steps,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        sequence_len=args.sequence_len,
        vocab_size=args.vocab_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        bootstrap_samples=args.bootstrap_samples,
        device=args.device,
    )
    report = run_kinetics_ablation(cfg, run_dir=args.run_dir, verbose=True)
    if args.output_json:
        report.to_json(args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
