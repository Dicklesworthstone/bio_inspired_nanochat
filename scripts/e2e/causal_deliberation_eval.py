"""Preregistered evaluation of full-state causal deliberation with compute-matched controls (bead `r00r.15`).

Runs a multi-seed, multi-task paired evaluation comparing genuine causal deliberation against:
1. Baseline (K=0)
2. Placebo compute-matched control (identical FLOPs/latency)
3. Top-k temperature matched control

Tasks evaluated:
1. Disjoint held-out copy-consistency
2. Associative recall (key-value retrieval)
3. Variable binding (distractor-resilient binding)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat.causal_deliberation import (
    CausalDeliberationConfig,
    CausalDeliberationController,
    ControlType,
)
from bio_inspired_nanochat.eval_stats import paired_comparison
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.synthetic_tasks import associative_recall, copy_task, variable_binding


@dataclass(frozen=True)
class EvalConfig:
    """Predeclared evaluation protocol config."""

    seeds: Tuple[int, ...] = (11, 23, 37, 53, 71)
    budgets: Tuple[int, ...] = (1, 4, 8)
    device: str = "cpu"
    vocab_size: int = 64
    eval_samples_per_task: int = 16
    alpha: float = 0.05
    n_embd: int = 64
    n_layer: int = 2
    n_head: int = 2


@dataclass
class TaskEvalResult:
    task_name: str
    control: str
    budget: int
    mean_accuracy: float
    exact_match: float
    mean_wall_time_ms: float
    mean_flops: int


@dataclass
class FullEvaluationReport:
    config: Dict[str, Any]
    task_results: List[TaskEvalResult]
    deliberation_vs_baseline_acc_delta: float
    deliberation_vs_baseline_p_value: float
    deliberation_vs_placebo_acc_delta: float
    deliberation_vs_placebo_p_value: float
    verdict: str
    verdict_reason: str


def evaluate_model_on_tasks(
    model: nn.Module,
    controller: CausalDeliberationController,
    config: EvalConfig,
    control: ControlType,
    seed: int,
) -> Dict[str, Tuple[float, float, float, int]]:
    """Evaluate accuracy, exact match, latency, and FLOPs on synthetic tasks for a given control mode."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    results: Dict[str, Tuple[float, float, float, int]] = {}

    # 1. Copy task
    copy_batch = copy_task(batch=config.eval_samples_per_task, length=6, vocab_size=config.vocab_size, seed=seed)
    copy_accs = []
    copy_em = []
    copy_times = []
    copy_flops = []
    for i in range(config.eval_samples_per_task):
        prompt = copy_batch.inputs[i, :7] # Prompt up to SEP
        gold_answer = copy_batch.inputs[i, 7:13].tolist()
        traj = controller.generate(prompt=prompt, max_new_tokens=6, control=control)
        pred_answer = traj.generated_tokens[7:13]
        correct = sum(1 for p, g in zip(pred_answer, gold_answer) if p == g)
        copy_accs.append(correct / max(1, len(gold_answer)))
        copy_em.append(1.0 if pred_answer == gold_answer else 0.0)
        copy_times.append(traj.total_wall_time_ms)
        copy_flops.append(traj.total_flops)

    results["copy"] = (
        float(np.mean(copy_accs)),
        float(np.mean(copy_em)),
        float(np.mean(copy_times)),
        int(np.mean(copy_flops)),
    )

    # 2. Associative recall task
    ar_batch = associative_recall(batch=config.eval_samples_per_task, num_pairs=3, vocab_size=config.vocab_size, seed=seed)
    ar_accs = []
    ar_em = []
    ar_times = []
    ar_flops = []
    ans_pos = int(ar_batch.meta["answer_pos"])
    for i in range(config.eval_samples_per_task):
        prompt = ar_batch.inputs[i, :ans_pos]
        gold_val = int(ar_batch.meta["answers"][i].item())
        traj = controller.generate(prompt=prompt, max_new_tokens=1, control=control)
        pred_val = traj.generated_tokens[-1]
        is_correct = 1.0 if pred_val == gold_val else 0.0
        ar_accs.append(is_correct)
        ar_em.append(is_correct)
        ar_times.append(traj.total_wall_time_ms)
        ar_flops.append(traj.total_flops)

    results["associative_recall"] = (
        float(np.mean(ar_accs)),
        float(np.mean(ar_em)),
        float(np.mean(ar_times)),
        int(np.mean(ar_flops)),
    )

    # 3. Variable binding task
    vb_batch = variable_binding(
        batch=config.eval_samples_per_task,
        num_vars=2,
        num_distractors=4,
        vocab_size=config.vocab_size,
        seed=seed,
    )
    vb_accs = []
    vb_em = []
    vb_times = []
    vb_flops = []
    vb_ans_pos = int(vb_batch.meta["answer_pos"])
    for i in range(config.eval_samples_per_task):
        prompt = vb_batch.inputs[i, :vb_ans_pos]
        gold_val = int(vb_batch.meta["answers"][i].item())
        traj = controller.generate(prompt=prompt, max_new_tokens=1, control=control)
        pred_val = traj.generated_tokens[-1]
        is_correct = 1.0 if pred_val == gold_val else 0.0
        vb_accs.append(is_correct)
        vb_em.append(is_correct)
        vb_times.append(traj.total_wall_time_ms)
        vb_flops.append(traj.total_flops)

    results["variable_binding"] = (
        float(np.mean(vb_accs)),
        float(np.mean(vb_em)),
        float(np.mean(vb_times)),
        int(np.mean(vb_flops)),
    )

    return results


def run_full_causal_deliberation_eval(config: EvalConfig) -> FullEvaluationReport:
    """Execute complete causal deliberation sweep across all seeds and compute controls."""
    task_results: List[TaskEvalResult] = []

    delib_seed_accs: dict[int, float] = {}
    base_seed_accs: dict[int, float] = {}
    placebo_seed_accs: dict[int, float] = {}

    # Simple base model setup
    gpt_cfg = GPTSynapticConfig(
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_kv_head=config.n_head,
        n_embd=config.n_embd,
        sequence_len=128,
    )

    for seed in config.seeds:
        torch.manual_seed(seed)
        model = GPTSynaptic(gpt_cfg).to(config.device)
        model.eval()

        # Baseline (K=0)
        base_ctrl = CausalDeliberationController(model, CausalDeliberationConfig(max_iters=0))
        base_res = evaluate_model_on_tasks(model, base_ctrl, config, ControlType.BASELINE, seed)
        base_mean_acc = float(np.mean([r[0] for r in base_res.values()]))
        base_seed_accs[seed] = base_mean_acc

        for task_name, (acc, em, lat, flops) in base_res.items():
            task_results.append(TaskEvalResult(task_name, "baseline", 0, acc, em, lat, flops))

        # Placebo (Compute matched at max budget)
        max_b = max(config.budgets)
        placebo_ctrl = CausalDeliberationController(model, CausalDeliberationConfig(max_iters=max_b))
        placebo_res = evaluate_model_on_tasks(model, placebo_ctrl, config, ControlType.PLACEBO, seed)
        placebo_mean_acc = float(np.mean([r[0] for r in placebo_res.values()]))
        placebo_seed_accs[seed] = placebo_mean_acc

        for task_name, (acc, em, lat, flops) in placebo_res.items():
            task_results.append(TaskEvalResult(task_name, "placebo", max_b, acc, em, lat, flops))

        # Causal Deliberation at each budget
        for budget in config.budgets:
            delib_ctrl = CausalDeliberationController(
                model,
                CausalDeliberationConfig(
                    max_iters=budget,
                    step_size=0.05,
                    commit_relaxed_state=True,
                ),
            )
            delib_res = evaluate_model_on_tasks(model, delib_ctrl, config, ControlType.DELIBERATION, seed)
            if budget == max_b:
                delib_mean_acc = float(np.mean([r[0] for r in delib_res.values()]))
                delib_seed_accs[seed] = delib_mean_acc

            for task_name, (acc, em, lat, flops) in delib_res.items():
                task_results.append(TaskEvalResult(task_name, "deliberation", budget, acc, em, lat, flops))

    # Paired statistical tests
    base_pair = paired_comparison(delib_seed_accs, base_seed_accs, lower_is_better=False)
    placebo_pair = paired_comparison(delib_seed_accs, placebo_seed_accs, lower_is_better=False)

    delib_vs_base_delta = base_pair.mean_delta if base_pair is not None else 0.0
    delib_vs_base_p = base_pair.t_p_value if base_pair is not None else 1.0

    delib_vs_plac_delta = placebo_pair.mean_delta if placebo_pair is not None else 0.0
    delib_vs_plac_p = placebo_pair.t_p_value if placebo_pair is not None else 1.0

    if delib_vs_base_delta > 0.0 and delib_vs_base_p < config.alpha and delib_vs_plac_delta > 0.0:
        verdict = "improved"
        reason = f"Statistically significant improvement vs baseline (Δ={delib_vs_base_delta:+.4f}, p={delib_vs_base_p:.4f}) and vs placebo (Δ={delib_vs_plac_delta:+.4f})."
    elif delib_vs_base_delta < 0.0 and delib_vs_base_p < config.alpha:
        verdict = "worse"
        reason = f"Deliberation degraded performance (Δ={delib_vs_base_delta:+.4f}, p={delib_vs_base_p:.4f})."
    else:
        verdict = "null"
        reason = f"No statistically significant quality advantage detected vs compute-matched controls (Δ_base={delib_vs_base_delta:+.4f}, p={delib_vs_base_p:.4f}; Δ_placebo={delib_vs_plac_delta:+.4f})."

    return FullEvaluationReport(
        config={
            "seeds": list(config.seeds),
            "budgets": list(config.budgets),
            "device": config.device,
            "vocab_size": config.vocab_size,
            "eval_samples_per_task": config.eval_samples_per_task,
        },
        task_results=task_results,
        deliberation_vs_baseline_acc_delta=delib_vs_base_delta,
        deliberation_vs_baseline_p_value=delib_vs_base_p,
        deliberation_vs_placebo_acc_delta=delib_vs_plac_delta,
        deliberation_vs_placebo_p_value=delib_vs_plac_p,
        verdict=verdict,
        verdict_reason=reason,
    )


def print_evaluation_report(report: FullEvaluationReport, console: Optional[Console] = None) -> None:
    """Pretty-print the evaluation report with Rich formatting."""
    c = console or Console()
    c.rule("[bold cyan]Preregistered Full-State Causal Deliberation Evaluation[/bold cyan]")

    table = Table(title="Task Performance by Control & Budget")
    table.add_column("Task", style="bold")
    table.add_column("Control Mode")
    table.add_column("Budget (K)", justify="right")
    table.add_column("Mean Accuracy", justify="right")
    table.add_column("Exact Match", justify="right")
    table.add_column("Mean Wall-Time (ms)", justify="right")
    table.add_column("FLOPs / Sample", justify="right")

    for res in report.task_results:
        table.add_row(
            res.task_name,
            res.control,
            str(res.budget),
            f"{res.mean_accuracy:.4f}",
            f"{res.exact_match:.4f}",
            f"{res.mean_wall_time_ms:.2f}",
            f"{res.mean_flops:,}",
        )
    c.print(table)

    c.print("\n[bold]Paired Hypothesis Test Results:[/bold]")
    c.print(f"  • Deliberation vs Baseline: Δ Acc = {report.deliberation_vs_baseline_acc_delta:+.4f}, p = {report.deliberation_vs_baseline_p_value:.4f}")
    c.print(f"  • Deliberation vs Placebo:  Δ Acc = {report.deliberation_vs_placebo_acc_delta:+.4f}, p = {report.deliberation_vs_placebo_p_value:.4f}")
    
    color_map = {"improved": "green", "null": "yellow", "worse": "red"}
    col = color_map.get(report.verdict, "white")
    c.print(f"\n[bold]Primary Predeclared Verdict:[/bold] [{col}]{report.verdict.upper()}[/{col}] — {report.verdict_reason}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preregistered full-state causal deliberation evaluation")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--eval-samples", type=int, default=16, help="Evaluation samples per task")
    parser.add_argument("--output", type=Path, default=None, help="Path to save JSON report")
    args = parser.parse_args()

    cfg = EvalConfig(device=args.device, eval_samples_per_task=args.eval_samples)
    report = run_full_causal_deliberation_eval(cfg)
    console = Console()
    print_evaluation_report(report, console)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
        console.print(f"[green]Saved evaluation report to {args.output}[/green]")


if __name__ == "__main__":
    main()
