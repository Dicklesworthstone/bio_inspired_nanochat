"""Synaptic granularity evaluation harness and benchmark (bead vap.2).

Performs an executable, apples-to-apples empirical comparison across architectural
synaptic granularities:
  1. Fine (L1)   — Per-Connection (`per_connection`): per-head/key presynaptic state and
                   full configured-rank projection eligibility.
  2. Medium (L2) — Per-Neuron (`per_neuron`): head-pooled presynaptic state and rank<=4
                   projection eligibility.
  3. Coarse (L3) — Per-Expert (`per_expert`): layer-pooled presynaptic/molecular state and
                   rank-1 projection eligibility.

Measures memory allocation, state footprint, token throughput, training loss, and
validation bpb across matched seeds.

Usage:
    python -m scripts.eval_synaptic_granularity
    pytest tests/test_synaptic_granularity.py -v
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.synthetic_tasks import associative_recall
from bio_inspired_nanochat.synaptic import (
    SynapticConfig,
    SynapticGranularity,
    build_presyn_state,
)


@dataclass
class GranularityBenchConfig:
    """Benchmark configuration for apples-to-apples granularity comparison."""

    vocab_size: int = 128
    n_layer: int = 2
    n_head: int = 4
    n_kv_head: int = 4
    n_embd: int = 64
    sequence_len: int = 32
    batch_size: int = 4
    num_steps: int = 12
    learning_rate: float = 1e-3
    seeds: tuple[int, ...] = (42, 1337)
    granularities: tuple[str, ...] = (
        SynapticGranularity.PER_CONNECTION.value,
        SynapticGranularity.PER_NEURON.value,
        SynapticGranularity.PER_EXPERT.value,
    )


@dataclass
class GranularityArmResult:
    """Empirical measurements for one granularity arm on one seed."""

    granularity: str
    seed: int
    train_losses: list[float]
    final_train_loss: float
    val_loss: float
    val_bpb: float
    val_accuracy: float
    total_time_sec: float
    tokens_per_sec: float
    persistent_buffers_bytes: int
    plastic_parameters_bytes: int
    presyn_runtime_bytes: int
    total_state_bytes: int
    num_state_tensors: int
    peak_memory_bytes: int
    passed: bool


@dataclass
class GranularityAggregateStats:
    """Multi-seed summary for a single granularity arm."""

    granularity: str
    num_seeds: int
    mean_val_loss: float
    std_val_loss: float
    mean_val_bpb: float
    std_val_bpb: float
    mean_val_accuracy: float
    std_val_accuracy: float
    mean_throughput: float
    std_throughput: float
    mean_state_bytes: float
    mean_peak_memory_bytes: float


@dataclass
class GranularityBenchReport:
    """Complete granularity benchmark report with raw and aggregated results."""

    config: GranularityBenchConfig
    arm_results: list[GranularityArmResult]
    aggregates: list[GranularityAggregateStats]
    passed: bool
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "arm_results": [asdict(r) for r in self.arm_results],
            "aggregates": [asdict(a) for a in self.aggregates],
            "passed": self.passed,
            "summary": self.summary,
        }

    def print_summary(self, console: Console | None = None) -> None:
        c = console or Console()
        c.rule("[bold cyan]Synaptic Granularity Benchmark Summary (vap.2)[/bold cyan]")
        t = Table(title="Granularity Quality & Efficiency Comparison")
        t.add_column("Granularity", style="cyan")
        t.add_column("State Footprint", style="magenta", justify="right")
        t.add_column("Throughput (tok/s)", style="green", justify="right")
        t.add_column("Val Loss (mean±std)", style="yellow", justify="right")
        t.add_column("Val BPB (mean±std)", style="bold", justify="right")
        t.add_column("Recall Acc (mean±std)", style="blue", justify="right")

        for agg in self.aggregates:
            state_kb = agg.mean_state_bytes / 1024.0
            t.add_row(
                agg.granularity,
                f"{state_kb:.1f} KB",
                f"{agg.mean_throughput:.0f} ± {agg.std_throughput:.0f}",
                f"{agg.mean_val_loss:.4f} ± {agg.std_val_loss:.4f}",
                f"{agg.mean_val_bpb:.4f} ± {agg.std_val_bpb:.4f}",
                f"{agg.mean_val_accuracy:.3f} ± {agg.std_val_accuracy:.3f}",
            )
        c.print(t)


def _generate_associative_batches(
    num_batches: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if seq_len < 4 or seq_len % 2 != 0:
        raise ValueError(
            "sequence_len must be an even integer >= 4 for the associative-recall protocol"
        )
    num_pairs = (seq_len - 2) // 2
    return [
        (
            batch.inputs,
            batch.targets,
        )
        for offset in range(num_batches)
        for batch in (
            associative_recall(
                batch=batch_size,
                num_pairs=num_pairs,
                vocab_size=vocab_size,
                seed=seed + offset,
            ),
        )
    ]


def _synaptic_config(granularity: SynapticGranularity) -> SynapticConfig:
    return SynapticConfig(
        granularity=granularity,
        enable_presyn=True,
        enable_hebbian=True,
        enable_metabolism=True,
    )


def _gpt_config(cfg: GranularityBenchConfig, syn_cfg: SynapticConfig) -> GPTSynapticConfig:
    return GPTSynapticConfig(
        sequence_len=cfg.sequence_len,
        vocab_size=cfg.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_kv_head=cfg.n_kv_head,
        n_embd=cfg.n_embd,
        synapses=True,
        syn_cfg=syn_cfg,
    )


def _build_matched_model(
    granularity: SynapticGranularity,
    seed: int,
    cfg: GranularityBenchConfig,
) -> GPTSynaptic:
    """Build an arm with identical shape-compatible backbone initialization.

    Different state shapes consume different amounts of RNG during construction. Building the
    per-connection reference first and copying every shape-compatible tensor prevents those RNG
    offsets from silently changing embeddings/backbone weights between arms.
    """
    torch.manual_seed(seed)
    reference = GPTSynaptic(
        _gpt_config(cfg, _synaptic_config(SynapticGranularity.PER_CONNECTION))
    )
    if granularity is SynapticGranularity.PER_CONNECTION:
        model = reference
    else:
        torch.manual_seed(seed)
        model = GPTSynaptic(_gpt_config(cfg, _synaptic_config(granularity)))
        reference_state = reference.state_dict()
        with torch.no_grad():
            for name, target in model.state_dict().items():
                source = reference_state.get(name)
                if source is not None and source.shape == target.shape:
                    target.copy_(source)
    # Training-time stochastic release receives the same lazy seed in every arm.
    torch.manual_seed(seed + 10_000)
    return model


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.nelement() * tensor.element_size())


def _measure_state_footprint(
    model: GPTSynaptic,
    syn_cfg: SynapticConfig,
    cfg: GranularityBenchConfig,
) -> tuple[int, int, int, int, int]:
    persistent_buffers = list(model.buffers())
    persistent_buffer_bytes = sum(_tensor_bytes(tensor) for tensor in persistent_buffers)
    plastic_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.endswith("w_fast") or ".post." in name
    ]
    plastic_parameter_bytes = sum(_tensor_bytes(tensor) for tensor in plastic_parameters)

    sample_state = build_presyn_state(
        cfg.batch_size,
        cfg.sequence_len,
        cfg.n_head,
        device="cpu",
        dtype=torch.float32,
        cfg=syn_cfg,
    )
    runtime_tensors = [
        tensor
        for value in sample_state.values()
        for tensor in (value if isinstance(value, list) else [value])
        if torch.is_tensor(tensor)
    ]
    presyn_runtime_bytes = cfg.n_layer * sum(
        _tensor_bytes(tensor) for tensor in runtime_tensors
    )
    tensor_count = (
        len(persistent_buffers) + len(plastic_parameters) + cfg.n_layer * len(runtime_tensors)
    )
    total_state_bytes = (
        persistent_buffer_bytes + plastic_parameter_bytes + presyn_runtime_bytes
    )
    return (
        persistent_buffer_bytes,
        plastic_parameter_bytes,
        presyn_runtime_bytes,
        total_state_bytes,
        tensor_count,
    )


def run_granularity_arm(
    granularity: str,
    seed: int,
    cfg: GranularityBenchConfig,
) -> GranularityArmResult:
    """Run a single training & evaluation trajectory for one granularity mode."""
    resolved_granularity = cast_granularity(granularity)
    syn_cfg = _synaptic_config(resolved_granularity)
    model = _build_matched_model(resolved_granularity, seed, cfg)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    train_data = _generate_associative_batches(
        cfg.num_steps, cfg.batch_size, cfg.sequence_len, cfg.vocab_size, seed=seed
    )
    val_data = _generate_associative_batches(
        4, cfg.batch_size, cfg.sequence_len, cfg.vocab_size, seed=seed + 1000
    )

    (
        persistent_buffer_bytes,
        plastic_parameter_bytes,
        presyn_runtime_bytes,
        total_state_bytes,
        state_tensor_count,
    ) = _measure_state_footprint(model, syn_cfg, cfg)

    train_losses: list[float] = []
    model_device = next(model.parameters()).device
    if model_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model_device)
    t0 = time.perf_counter()
    tokens_processed = 0

    for x, y in train_data:
        optimizer.zero_grad()
        _logits, loss = model(x, targets=y, train_mode=True)
        if loss is None:
            raise RuntimeError("Model returned None loss during training")
        loss.backward()
        optimizer.step()

        val = float(loss.item())
        if not math.isfinite(val):
            raise FloatingPointError(f"Divergent loss {val} under granularity {granularity}")
        train_losses.append(val)
        tokens_processed += int(x.numel())

    dt = time.perf_counter() - t0
    tok_per_sec = tokens_processed / max(dt, 1e-6)

    # Validation pass
    model.eval()
    val_losses: list[float] = []
    correct = 0
    supervised = 0
    with torch.no_grad():
        for vx, vy in val_data:
            logits, vloss = model(vx, targets=vy, train_mode=False)
            if vloss is None:
                raise RuntimeError("Model returned None loss during evaluation")
            val_losses.append(float(vloss.item()))
            mask = vy.ne(-1)
            correct += int(logits.argmax(dim=-1)[mask].eq(vy[mask]).sum().item())
            supervised += int(mask.sum().item())

    mean_vloss = sum(val_losses) / len(val_losses) if val_losses else 0.0
    val_bpb = mean_vloss / math.log(2.0)
    val_accuracy = correct / supervised if supervised else 0.0

    peak_memory = (
        torch.cuda.max_memory_allocated(model_device)
        if model_device.type == "cuda"
        else total_state_bytes
    )

    return GranularityArmResult(
        granularity=granularity,
        seed=seed,
        train_losses=train_losses,
        final_train_loss=train_losses[-1] if train_losses else 0.0,
        val_loss=mean_vloss,
        val_bpb=val_bpb,
        val_accuracy=val_accuracy,
        total_time_sec=dt,
        tokens_per_sec=tok_per_sec,
        persistent_buffers_bytes=persistent_buffer_bytes,
        plastic_parameters_bytes=plastic_parameter_bytes,
        presyn_runtime_bytes=presyn_runtime_bytes,
        total_state_bytes=total_state_bytes,
        num_state_tensors=state_tensor_count,
        peak_memory_bytes=peak_memory,
        passed=True,
    )


def cast_granularity(g: str) -> SynapticGranularity:
    if g == SynapticGranularity.PER_CONNECTION.value:
        return SynapticGranularity.PER_CONNECTION
    if g == SynapticGranularity.PER_NEURON.value:
        return SynapticGranularity.PER_NEURON
    if g == SynapticGranularity.PER_EXPERT.value:
        return SynapticGranularity.PER_EXPERT
    raise ValueError(f"Unknown granularity: {g}")


def run_granularity_benchmark(
    cfg: GranularityBenchConfig | None = None,
) -> GranularityBenchReport:
    """Execute the full apples-to-apples granularity benchmark across all seeds."""
    config = cfg or GranularityBenchConfig()
    results: list[GranularityArmResult] = []

    for gran in config.granularities:
        for seed in config.seeds:
            arm_res = run_granularity_arm(gran, seed, config)
            results.append(arm_res)

    # Compute per-granularity aggregate statistics
    aggregates: list[GranularityAggregateStats] = []
    for gran in config.granularities:
        gran_runs = [r for r in results if r.granularity == gran]
        n = len(gran_runs)
        if n == 0:
            continue

        losses = [r.val_loss for r in gran_runs]
        bpbs = [r.val_bpb for r in gran_runs]
        accuracies = [r.val_accuracy for r in gran_runs]
        tps = [r.tokens_per_sec for r in gran_runs]
        mem = [r.total_state_bytes for r in gran_runs]
        pmem = [r.peak_memory_bytes for r in gran_runs]

        m_loss = sum(losses) / n
        s_loss = math.sqrt(sum((x - m_loss) ** 2 for x in losses) / max(n - 1, 1)) if n > 1 else 0.0

        m_bpb = sum(bpbs) / n
        s_bpb = math.sqrt(sum((x - m_bpb) ** 2 for x in bpbs) / max(n - 1, 1)) if n > 1 else 0.0

        m_accuracy = sum(accuracies) / n
        s_accuracy = (
            math.sqrt(
                sum((x - m_accuracy) ** 2 for x in accuracies) / max(n - 1, 1)
            )
            if n > 1
            else 0.0
        )

        m_tp = sum(tps) / n
        s_tp = math.sqrt(sum((x - m_tp) ** 2 for x in tps) / max(n - 1, 1)) if n > 1 else 0.0

        m_mem = sum(mem) / n
        m_pmem = sum(pmem) / n

        aggregates.append(
            GranularityAggregateStats(
                granularity=gran,
                num_seeds=n,
                mean_val_loss=m_loss,
                std_val_loss=s_loss,
                mean_val_bpb=m_bpb,
                std_val_bpb=s_bpb,
                mean_val_accuracy=m_accuracy,
                std_val_accuracy=s_accuracy,
                mean_throughput=m_tp,
                std_throughput=s_tp,
                mean_state_bytes=m_mem,
                mean_peak_memory_bytes=m_pmem,
            )
        )

    all_passed = all(r.passed for r in results)
    report = GranularityBenchReport(
        config=config,
        arm_results=results,
        aggregates=aggregates,
        passed=all_passed,
        summary={
            "total_runs": len(results),
            "granularities_tested": list(config.granularities),
            "seeds_tested": list(config.seeds),
            "task": "associative_recall",
            "state_footprint_definition": (
                "persistent model buffers + plastic parameters + one presynaptic runtime "
                "state per attention layer"
            ),
            "torch_version": torch.__version__,
            "device": "cpu",
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Synaptic Granularity Benchmark (vap.2)")
    parser.add_argument("--quick", action="store_true", help="Run fast 1-seed smoke test")
    parser.add_argument(
        "--output",
        type=str,
        default="results/granularity_comparison.json",
        help="Path to output JSON results",
    )
    args = parser.parse_args()

    cfg = GranularityBenchConfig()
    if args.quick:
        cfg.seeds = (42,)
        cfg.num_steps = 4

    report = run_granularity_benchmark(cfg)
    console = Console()
    report.print_summary(console)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    console.print(f"[green]Results archived to[/green] {out_path}")


if __name__ == "__main__":
    main()
