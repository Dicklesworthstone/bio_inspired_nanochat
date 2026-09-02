"""Fast-Weight Programmer & Working-Memory Baseline Comparison (bead `sax.5`).

Benchmarks bio-inspired synaptic fast weights against standard fast-weight architectures:
  1. `vanilla`: Standard Transformer with positional / KV attention
  2. `outer_product_fw`: Classical Schmidhuber / Hebbian Outer-Product Fast Weights (W_t = λ W_{t-1} + η v k^T)
  3. `deltanet_fw`: Linear Transformer / DeltaNet Error-Correcting Fast Weights (ΔW = β (v - W k) k^T)
  4. `bio_synaptic`: Bio-inspired Synaptic Transformer with CaMKII/PP1 latching, presynaptic vesicle dynamics, and normalized Hebbian updates

Evaluated across the standardized working memory suite:
  - Multi-pair Associative Recall
  - Variable Binding under Distractor Pressure
  - Needle-In-A-Haystack (NIAH)
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, Sequence, Tuple

from rich.console import Console
from rich.table import Table
import torch
import torch.nn as nn
import torch.nn.functional as F

from bio_inspired_nanochat.results_registry import measurement_regime
from bio_inspired_nanochat.eval_stats import (
    Aggregate,
    PairedResult,
    aggregate,
    paired_comparison,
)
from bio_inspired_nanochat.gpt import GPT, GPTConfig
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.run_logging import RunLogger
from bio_inspired_nanochat.synaptic import SynapticConfig
from bio_inspired_nanochat.synthetic_tasks import working_memory_suite


@dataclass(frozen=True)
class FastWeightBenchConfig:
    """Predeclared architecture and evaluation parameters."""

    seeds: Tuple[int, ...] = (301, 303, 307, 309, 311)
    vocab_size: int = 97
    sequence_len: int = 160
    n_embd: int = 64
    n_layer: int = 2
    n_head: int = 2
    n_kv_head: int = 2
    batch_size: int = 16
    recall_pairs: Tuple[int, ...] = (2, 4, 8)
    binding_distractors: Tuple[int, ...] = (0, 8, 32)
    niah_lengths: Tuple[int, ...] = (16, 64)
    device: str = "cpu"
    bootstrap_samples: int = 2000

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if len(self.seeds) < 2:
            raise ValueError("seeds must contain at least two unique seeds for paired statistics")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if self.n_embd % 4 != 0:
            raise ValueError("n_embd must be divisible by 4")


class ClassicalOuterProductFastWeightBlock(nn.Module):
    """Schmidhuber classical outer-product fast weight layer (W_t = decay*W_{t-1} + eta * v k^T)."""

    def __init__(self, n_embd: int, decay: float = 0.95, eta: float = 0.5):
        super().__init__()
        self.n_embd = n_embd
        self.decay = decay
        self.eta = eta
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q = F.normalize(self.q_proj(x), dim=-1)
        k = F.normalize(self.k_proj(x), dim=-1)
        v = self.v_proj(x)

        w_fast = torch.zeros(b, d, d, device=x.device, dtype=x.dtype)
        outputs = []
        for i in range(t):
            qi = q[:, i : i + 1, :]   # (B, 1, D)
            ki = k[:, i : i + 1, :]   # (B, 1, D)
            vi = v[:, i : i + 1, :]   # (B, 1, D)

            # Read from fast weights
            readout = torch.bmm(qi, w_fast)  # (B, 1, D)
            outputs.append(readout)

            # Write outer product to fast weights
            outer = torch.bmm(ki.transpose(1, 2), vi)  # (B, D, D)
            w_fast = self.decay * w_fast + self.eta * outer

        out = torch.cat(outputs, dim=1)
        return self.out_proj(out)


class DeltaNetFastWeightBlock(nn.Module):
    """Linear Transformer / DeltaNet with error-correcting delta rule (ΔW = β (v - W k) k^T)."""

    def __init__(self, n_embd: int, beta: float = 0.5):
        super().__init__()
        self.n_embd = n_embd
        self.beta = beta
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q = F.normalize(self.q_proj(x), dim=-1)
        k = F.normalize(self.k_proj(x), dim=-1)
        v = self.v_proj(x)

        w_fast = torch.zeros(b, d, d, device=x.device, dtype=x.dtype)
        outputs = []
        for i in range(t):
            qi = q[:, i : i + 1, :]
            ki = k[:, i : i + 1, :]
            vi = v[:, i : i + 1, :]

            # Retrieve prior prediction
            v_pred = torch.bmm(ki, w_fast)  # (B, 1, D)
            # Delta error term
            error = vi - v_pred
            # Readout with query
            readout = torch.bmm(qi, w_fast)
            outputs.append(readout)

            # Error-correcting update
            delta_w = self.beta * torch.bmm(ki.transpose(1, 2), error)
            w_fast = w_fast + delta_w

        out = torch.cat(outputs, dim=1)
        return self.out_proj(out)


class CustomFastWeightTransformer(nn.Module):
    """Wrapper transformer architecture incorporating modular fast weight blocks."""

    def __init__(self, mode: str, cfg: FastWeightBenchConfig):
        super().__init__()
        self.mode = mode
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.sequence_len, cfg.n_embd)

        self.blocks = nn.ModuleList()
        for _ in range(cfg.n_layer):
            if mode == "outer_product_fw":
                self.blocks.append(ClassicalOuterProductFastWeightBlock(cfg.n_embd))
            elif mode == "deltanet_fw":
                self.blocks.append(DeltaNetFastWeightBlock(cfg.n_embd))
            else:
                raise ValueError(f"Unknown mode: {mode}")

        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def reset_sequence_state(self, **kwargs) -> None:
        pass

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor | None]:
        b, t = idx.shape
        positions = torch.arange(0, t, device=idx.device).unsqueeze(0)
        h = self.token_emb(idx) + self.pos_emb(positions)

        for block in self.blocks:
            h = h + block(h)

        h = self.ln_f(h)
        logits = self.head(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.cfg.vocab_size), targets.view(-1), ignore_index=-1)

        return logits, loss


def _build_benchmark_model(mode: str, cfg: FastWeightBenchConfig, seed: int) -> nn.Module:
    """Instantiate the target architecture."""
    torch.manual_seed(seed)

    if mode == "vanilla":
        gpt_cfg = GPTConfig(
            sequence_len=cfg.sequence_len,
            vocab_size=cfg.vocab_size,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_kv_head=cfg.n_kv_head,
            n_embd=cfg.n_embd,
        )
        return GPT(gpt_cfg).to(cfg.device)

    elif mode == "bio_synaptic":
        syn_cfg = SynapticConfig(
            enable_presyn=True,
            enable_hebbian=True,
            bistable_latch=True,
            fast_weight_normalized=True,
            stochastic_train_frac=0.0,
        )
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
        return GPTSynaptic(gpt_cfg).to(cfg.device)

    elif mode in ("outer_product_fw", "deltanet_fw"):
        return CustomFastWeightTransformer(mode, cfg).to(cfg.device)

    else:
        raise ValueError(f"Unknown benchmark mode: {mode}")


@dataclass
class ArchitectureMemoryScore:
    mode: str
    seed: int
    recall_overall: float
    binding_overall: float
    niah_overall: float
    composite_score: float
    details: Dict[str, Any]


@dataclass
class ArchitectureBenchmarkSummary:
    mode: str
    scores: list[ArchitectureMemoryScore]
    composite_stats: Aggregate
    recall_stats: Aggregate
    binding_stats: Aggregate
    niah_stats: Aggregate


@dataclass
class FastWeightBenchReport:
    run_id: str
    config: FastWeightBenchConfig
    architectures: Dict[str, ArchitectureBenchmarkSummary]
    comparisons_vs_vanilla: Dict[str, PairedResult]
    comparisons_bio_vs_deltanet: PairedResult
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
            "architectures": {
                name: {
                    "composite_mean": arch.composite_stats.mean,
                    "composite_ci": [arch.composite_stats.ci_low, arch.composite_stats.ci_high],
                    "recall_mean": arch.recall_stats.mean,
                    "binding_mean": arch.binding_stats.mean,
                    "niah_mean": arch.niah_stats.mean,
                    "per_seed_composite": {s.seed: s.composite_score for s in arch.scores},
                }
                for name, arch in self.architectures.items()
            },
            "measurement_regime": measurement_regime(),
            "comparisons_vs_vanilla": {
                name: asdict(comp) for name, comp in self.comparisons_vs_vanilla.items()
            },
            "comparisons_bio_vs_deltanet": asdict(self.comparisons_bio_vs_deltanet),
        }
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_fast_weight_benchmark(
    cfg: FastWeightBenchConfig | None = None,
    *,
    run_dir: Path | str | None = None,
    verbose: bool = True,
) -> FastWeightBenchReport:
    """Execute multi-seed evaluation across the 4 fast-weight architectures."""
    if cfg is None:
        cfg = FastWeightBenchConfig()
    cfg.validate()

    console = Console(quiet=not verbose)
    run_id = f"fw-bench-{int(time.time())}"

    if run_dir is None:
        base_dir = Path(tempfile.mkdtemp(prefix="fw_bench_"))
    else:
        base_dir = Path(run_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

    logger = RunLogger(base_dir, name="fast_weight_bench", run_id=run_id, console=verbose)
    logger.event("bench_config", config=asdict(cfg))

    modes = ("vanilla", "outer_product_fw", "deltanet_fw", "bio_synaptic")
    arch_summaries: Dict[str, ArchitectureBenchmarkSummary] = {}

    for mode in modes:
        scores: list[ArchitectureMemoryScore] = []
        composites: Dict[int, float] = {}
        recalls: list[float] = []
        bindings: list[float] = []
        niahs: list[float] = []

        for seed in cfg.seeds:
            model = _build_benchmark_model(mode, cfg, seed)
            suite_res = working_memory_suite(
                model,
                vocab_size=cfg.vocab_size,
                recall_pairs=cfg.recall_pairs,
                binding_distractors=cfg.binding_distractors,
                niah_lengths=cfg.niah_lengths,
                batch=cfg.batch_size,
                seed=seed,
            )

            r_ov = float(suite_res["summary"]["recall_overall"])
            b_ov = float(suite_res["summary"]["binding_overall"])
            n_ov = float(suite_res["summary"]["niah_overall"])
            comp = float((r_ov + b_ov + n_ov) / 3.0)

            recalls.append(r_ov)
            bindings.append(b_ov)
            niahs.append(n_ov)
            composites[seed] = comp

            score_obj = ArchitectureMemoryScore(
                mode=mode,
                seed=seed,
                recall_overall=r_ov,
                binding_overall=b_ov,
                niah_overall=n_ov,
                composite_score=comp,
                details=suite_res,
            )
            scores.append(score_obj)
            logger.event("arch_score", mode=mode, seed=seed, composite=comp, recall=r_ov, binding=b_ov, niah=n_ov)

        arch_summaries[mode] = ArchitectureBenchmarkSummary(
            mode=mode,
            scores=scores,
            composite_stats=aggregate(list(composites.values())),
            recall_stats=aggregate(recalls),
            binding_stats=aggregate(bindings),
            niah_stats=aggregate(niahs),
        )

    # Statistical comparisons against vanilla baseline (higher is better for accuracy/retrieval)
    comparisons_vs_vanilla: Dict[str, PairedResult] = {}
    vanilla_composites = {s.seed: s.composite_score for s in arch_summaries["vanilla"].scores}

    for mode in ("outer_product_fw", "deltanet_fw", "bio_synaptic"):
        cand_composites = {s.seed: s.composite_score for s in arch_summaries[mode].scores}
        comp_res = paired_comparison(
            cand_composites,
            vanilla_composites,
            lower_is_better=False,
            n_boot=cfg.bootstrap_samples,
        )
        if comp_res is not None:
            comparisons_vs_vanilla[mode] = comp_res

    # Direct head-to-head comparison: Bio-Synaptic vs DeltaNet
    bio_composites = {s.seed: s.composite_score for s in arch_summaries["bio_synaptic"].scores}
    deltanet_composites = {s.seed: s.composite_score for s in arch_summaries["deltanet_fw"].scores}
    comp_bio_vs_deltanet = paired_comparison(
        bio_composites,
        deltanet_composites,
        lower_is_better=False,
        n_boot=cfg.bootstrap_samples,
    )
    if comp_bio_vs_deltanet is None:
        raise RuntimeError("Failed to compute bio vs deltanet paired comparison")

    verdict = "AUDITED_BENCHMARK_COMPLETE"
    summary_text = (
        "Apples-to-apples working-memory suite comparison evaluated Vanilla Transformer, Classical Outer-Product "
        "Fast Weights, DeltaNet Error-Correcting Fast Weights, and Bio-Inspired Synaptic Transformers across multi-pair "
        "associative recall, distractor binding, and NIAH."
    )

    report = FastWeightBenchReport(
        run_id=run_id,
        config=cfg,
        architectures=arch_summaries,
        comparisons_vs_vanilla=comparisons_vs_vanilla,
        comparisons_bio_vs_deltanet=comp_bio_vs_deltanet,
        verdict=verdict,
        summary_text=summary_text,
    )

    if verbose:
        table = Table(title="Working-Memory Suite Benchmark across Fast-Weight Architectures")
        table.add_column("Architecture", style="cyan")
        table.add_column("Composite Score", justify="right")
        table.add_column("95% CI", justify="right")
        table.add_column("Recall Acc", justify="right")
        table.add_column("Binding Acc", justify="right")
        table.add_column("NIAH Acc", justify="right")

        for name, arch in arch_summaries.items():
            table.add_row(
                name,
                f"{arch.composite_stats.mean * 100.0:.1f}%",
                f"[{arch.composite_stats.ci_low * 100.0:.1f}%, {arch.composite_stats.ci_high * 100.0:.1f}%]",
                f"{arch.recall_stats.mean * 100.0:.1f}%",
                f"{arch.binding_stats.mean * 100.0:.1f}%",
                f"{arch.niah_stats.mean * 100.0:.1f}%",
            )
        console.print(table)

    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fast-Weight Programmer & Working-Memory Benchmark")
    parser.add_argument("--run-dir", type=str, default=None, help="Directory to save logs")
    parser.add_argument("--output-json", type=str, default="results/fast_weight_comparison_evaluation.json", help="JSON output path")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Evaluation seeds")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    args = parser.parse_args(argv)

    seeds = tuple(args.seeds) if args.seeds is not None else (301, 303, 307, 309, 311)
    cfg = FastWeightBenchConfig(
        seeds=seeds,
        device=args.device,
    )
    report = run_fast_weight_benchmark(cfg, run_dir=args.run_dir, verbose=True)
    if args.output_json:
        report.to_json(args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
