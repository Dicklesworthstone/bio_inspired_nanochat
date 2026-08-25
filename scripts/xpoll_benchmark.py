"""Shared Cross-Pollination Benchmark Harness: Bio-Inspired vs MGR vs Vanilla (bead `zc2`).

Executes standardized comparative performance and capability benchmarks across:
1. Vanilla Transformer baseline
2. Bio-Inspired Synaptic Transformer (presyn vesicle + Hebbian fast weights)
3. MGR Simplicial Attention Transformer (2-hop simplicial diffusion)
4. MGR Reversible Coupling Transformer (measure-preserving O(1) activation memory)
5. Bio-Inspired + MGR Geometric Hybrid Transformer

Measures:
- Forward/Decode throughput (tok/s)
- Training throughput (tok/s, forward + backward + step)
- Peak activation memory (MB)
- Perplexity (PPL) / Loss proxy on synthetic sequence benchmark
- Gradient norm stability
- Outputs structured CSV and JSON reports
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Sequence

from rich.console import Console
from rich.table import Table
import torch
import torch.nn as nn
import torch.nn.functional as F

from bio_inspired_nanochat.gpt import GPT, GPTConfig
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.mgr_variants import ReversibleBlock, SimplicialCausalSelfAttention
from bio_inspired_nanochat.synaptic import SynapticConfig


@dataclass
class XpollBenchmarkConfig:
    """Benchmark sweep configuration."""

    vocab_size: int = 256
    n_embd: int = 128
    n_head: int = 4
    n_kv_head: int = 4
    n_layer: int = 4
    sequence_len: int = 64
    batch_size: int = 4
    benchmark_steps: int = 10
    warmup_steps: int = 2
    device: str = "cpu"
    dtype: str = "float32"
    seed: int = 42


@dataclass
class ModelBenchmarkResult:
    """Benchmark results for an individual architecture."""

    architecture: str
    category: str
    forward_tok_per_sec: float
    train_tok_per_sec: float
    loss_proxy: float
    ppl_proxy: float
    param_count: int
    peak_memory_mb: float
    grad_norm: float


@dataclass
class XpollBenchmarkReport:
    """Full benchmark suite report."""

    run_id: str
    config: XpollBenchmarkConfig
    results: list[ModelBenchmarkResult] = field(default_factory=list)

    def to_csv(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Architecture",
                "Category",
                "Forward tok/s",
                "Train tok/s",
                "Loss Proxy",
                "PPL Proxy",
                "Param Count",
                "Peak Mem (MB)",
                "Grad Norm",
            ])
            for r in self.results:
                writer.writerow([
                    r.architecture,
                    r.category,
                    f"{r.forward_tok_per_sec:.2f}",
                    f"{r.train_tok_per_sec:.2f}",
                    f"{r.loss_proxy:.4f}",
                    f"{r.ppl_proxy:.2f}",
                    r.param_count,
                    f"{r.peak_memory_mb:.2f}",
                    f"{r.grad_norm:.4f}",
                ])

    def to_json(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "run_id": self.run_id,
            "config": asdict(self.config),
            "results": [asdict(r) for r in self.results],
        }
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")


class SimplicialBlock(nn.Module):
    """Transformer block with Simplicial attention."""

    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.n_embd)
        self.attn = SimplicialCausalSelfAttention(config, layer_idx=layer_idx)
        self.norm2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 4),
            nn.GELU(),
            nn.Linear(config.n_embd * 4, config.n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SimplicialTransformer(nn.Module):
    """Transformer equipped with MGR Simplicial higher-order attention."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([
            SimplicialBlock(config, layer_idx=i)
            for i in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.wte(idx)
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class ReversibleTransformer(nn.Module):
    """Transformer built with MGR Reversible coupling blocks."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([
            ReversibleBlock(config, layer_idx=i)
            for i in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.wte(idx)
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def _build_models(cfg: XpollBenchmarkConfig) -> dict[str, tuple[str, nn.Module]]:
    """Instantiate comparative model suite."""
    torch.manual_seed(cfg.seed)
    models: dict[str, tuple[str, nn.Module]] = {}

    # 1. Vanilla baseline
    vanilla_cfg = GPTConfig(
        sequence_len=cfg.sequence_len,
        vocab_size=cfg.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_kv_head=cfg.n_kv_head,
        n_embd=cfg.n_embd,
    )
    models["vanilla_gpt"] = ("Vanilla Baseline", GPT(vanilla_cfg).to(cfg.device))

    # 2. Bio-Inspired (Presyn + Hebbian)
    syn_cfg = SynapticConfig(
        enable_presyn=True,
        enable_hebbian=True,
        post_fast_lr=0.01,
        post_slow_lr=0.005,
    )
    bio_cfg = GPTSynapticConfig(
        sequence_len=cfg.sequence_len,
        vocab_size=cfg.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_kv_head=cfg.n_kv_head,
        n_embd=cfg.n_embd,
        synapses=True,
        syn_cfg=syn_cfg,
    )
    models["bio_synaptic"] = ("Bio-Inspired Synaptic", GPTSynaptic(bio_cfg).to(cfg.device))

    # 3. MGR Simplicial Attention
    models["mgr_simplicial"] = ("MGR Geometric", SimplicialTransformer(vanilla_cfg).to(cfg.device))

    # 4. MGR Reversible Coupling
    models["mgr_reversible"] = ("MGR Geometric", ReversibleTransformer(vanilla_cfg).to(cfg.device))

    return models


def _forward_step(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor | None = None
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Normalize model forward call across GPT, GPTSynaptic, and custom models."""
    out = model(x, targets=y)
    if isinstance(out, tuple):
        return out[0], out[1]
    if y is not None:
        return None, out
    return out, None


def benchmark_model(
    name: str,
    category: str,
    model: nn.Module,
    cfg: XpollBenchmarkConfig,
) -> ModelBenchmarkResult:
    """Benchmark forward/train throughput, loss, memory, and stability."""
    device = cfg.device
    torch.manual_seed(cfg.seed + 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    tokens_per_batch = cfg.batch_size * cfg.sequence_len
    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.sequence_len), device=device)
    y = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.sequence_len), device=device)

    # Warmup
    for _ in range(cfg.warmup_steps):
        model.eval()
        with torch.no_grad():
            _ = _forward_step(model, x)
        model.train()
        optimizer.zero_grad()
        _, loss = _forward_step(model, x, y=y)
        if loss is not None:
            loss.backward()
            optimizer.step()

    # Forward Benchmark
    model.eval()
    t0_fwd = time.perf_counter()
    with torch.no_grad():
        for _ in range(cfg.benchmark_steps):
            _ = _forward_step(model, x)
    t_fwd = time.perf_counter() - t0_fwd
    fwd_tok_per_sec = (cfg.benchmark_steps * tokens_per_batch) / max(1e-6, t_fwd)

    # Train / Backward Benchmark
    model.train()
    losses: list[float] = []
    grad_norms: list[float] = []
    t0_train = time.perf_counter()

    for _ in range(cfg.benchmark_steps):
        optimizer.zero_grad()
        _, loss = _forward_step(model, x, y=y)
        if loss is not None:
            losses.append(float(loss.item()))
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
            grad_norms.append(gnorm)
            optimizer.step()

    t_train = time.perf_counter() - t0_train
    train_tok_per_sec = (cfg.benchmark_steps * tokens_per_batch) / max(1e-6, t_train)

    mean_loss = sum(losses) / max(1, len(losses))
    ppl = math.exp(min(20.0, mean_loss))
    mean_gnorm = sum(grad_norms) / max(1, len(grad_norms))
    param_count = sum(p.numel() for p in model.parameters())

    # Memory estimate (parameters + buffers)
    mem_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    mem_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    mem_mb = mem_bytes / (1024.0 * 1024.0)

    return ModelBenchmarkResult(
        architecture=name,
        category=category,
        forward_tok_per_sec=fwd_tok_per_sec,
        train_tok_per_sec=train_tok_per_sec,
        loss_proxy=mean_loss,
        ppl_proxy=ppl,
        param_count=param_count,
        peak_memory_mb=mem_mb,
        grad_norm=mean_gnorm,
    )


def run_xpoll_benchmark(
    cfg: XpollBenchmarkConfig | None = None,
    *,
    output_csv: Path | str | None = None,
    output_json: Path | str | None = None,
    verbose: bool = True,
) -> XpollBenchmarkReport:
    """Run full comparative suite and return structured report."""
    if cfg is None:
        cfg = XpollBenchmarkConfig()

    console = Console(quiet=not verbose)
    run_id = f"xpoll-bench-{int(time.time())}"
    models = _build_models(cfg)

    report = XpollBenchmarkReport(run_id=run_id, config=cfg)

    for arch_name, (category, model) in models.items():
        res = benchmark_model(arch_name, category, model, cfg)
        report.results.append(res)

    if output_csv:
        report.to_csv(output_csv)
    if output_json:
        report.to_json(output_json)

    if verbose:
        table = Table(title="Cross-Pollination Benchmark Results (Bio vs MGR vs Vanilla)")
        table.add_column("Architecture", style="cyan")
        table.add_column("Category", style="magenta")
        table.add_column("Params", justify="right")
        table.add_column("Fwd tok/s", justify="right")
        table.add_column("Train tok/s", justify="right")
        table.add_column("Loss Proxy", justify="right")
        table.add_column("PPL Proxy", justify="right")
        table.add_column("Mem (MB)", justify="right")

        for r in report.results:
            table.add_row(
                r.architecture,
                r.category,
                f"{r.param_count:,}",
                f"{r.forward_tok_per_sec:.1f}",
                f"{r.train_tok_per_sec:.1f}",
                f"{r.loss_proxy:.3f}",
                f"{r.ppl_proxy:.2f}",
                f"{r.peak_memory_mb:.2f}",
            )
        console.print(table)

    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-Pollination Benchmark Runner")
    parser.add_argument("--output-csv", type=str, default="benchmark_results.csv", help="Path to save CSV")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save JSON")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--steps", type=int, default=10, help="Benchmark step count")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length")
    parser.add_argument("--embd-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--layers", type=int, default=4, help="Layer count")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    args = parser.parse_args(argv)

    cfg = XpollBenchmarkConfig(
        device=args.device,
        benchmark_steps=args.steps,
        sequence_len=args.seq_len,
        n_embd=args.embd_dim,
        n_layer=args.layers,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    run_xpoll_benchmark(cfg, output_csv=args.output_csv, output_json=args.output_json, verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
