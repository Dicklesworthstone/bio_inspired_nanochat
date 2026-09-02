r"""Performance-Regression Test Harness & Committed Baselines (beads eqyk.15, r2d).

Measures training & inference throughput (tokens/second), latency (ms/step),
and peak memory consumption across representative model configurations,
evaluating measurements against committed baseline thresholds.
"""

from __future__ import annotations

from typing import Any, cast

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from bio_inspired_nanochat.gpt import GPT, GPTConfig
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.engine import Engine, KVCache
from bio_inspired_nanochat.results_registry import RunRecord, append_record
from bio_inspired_nanochat.synaptic import SynapticConfig

DEFAULT_BASELINES_PATH = Path("results") / "perf_baselines.json"


@dataclass
class PerfBenchmarkConfig:
    name: str
    mode: str = "train"  # "train" | cached one-token "decode"
    synaptic: bool = False
    batch_size: int = 2
    seq_len: int = 32  # training length, or the untimed decode prompt length
    vocab_size: int = 128
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    warmup_steps: int = 2
    measure_steps: int = 5
    device: str = "cpu"


@dataclass
class BenchmarkResult:
    name: str
    mode: str
    synaptic: bool
    tok_per_sec: float
    latency_ms: float
    peak_memory_mb: float
    steps_measured: int
    batch_size: int
    seq_len: int
    device: str
    benchmark_config: dict[str, bool | int | str]
    timestamp: float = field(default_factory=time.time)


@dataclass
class BaselineEntry:
    name: str
    mode: str
    tok_per_sec: float
    peak_memory_mb: float
    device: str
    benchmark_config: dict[str, bool | int | str]
    tolerance: float = 0.25  # Allowable degradation (e.g. 25% tolerance for CPU/CI variance)


@dataclass
class GateComparison:
    name: str
    mode: str
    observed_tok_per_sec: float
    baseline_tok_per_sec: float
    speed_ratio: float
    passed: bool
    detail: str


STANDARD_BENCHMARK_CONFIGS = (
    PerfBenchmarkConfig(
        name="standard_transformer_train",
        mode="train",
        synaptic=False,
        batch_size=2,
        seq_len=32,
        n_layer=2,
        n_head=4,
        n_embd=64,
    ),
    PerfBenchmarkConfig(
        name="synaptic_transformer_train",
        mode="train",
        synaptic=True,
        batch_size=2,
        seq_len=32,
        n_layer=2,
        n_head=4,
        n_embd=64,
    ),
    PerfBenchmarkConfig(
        name="standard_transformer_cached_tq1_decode",
        mode="decode",
        synaptic=False,
        batch_size=1,
        seq_len=16,
        n_layer=2,
        n_head=4,
        n_embd=64,
        warmup_steps=20,
        measure_steps=200,
    ),
    PerfBenchmarkConfig(
        name="synaptic_transformer_cached_tq1_decode",
        mode="decode",
        synaptic=True,
        batch_size=1,
        seq_len=16,
        n_layer=2,
        n_head=4,
        n_embd=64,
        warmup_steps=20,
        measure_steps=200,
    ),
)


def _synchronize_for_timing(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA benchmark requested on a host without CUDA: {requested}")
    return device


def _config_snapshot(
    cfg: PerfBenchmarkConfig,
    device: torch.device,
) -> dict[str, bool | int | str]:
    snapshot = asdict(cfg)
    snapshot["device"] = str(device)
    return snapshot


class PerfRegressionHarness:
    """Benchmark harness executing throughput measurements and evaluating gates."""

    def __init__(self, baselines_path: Path | str | None = None) -> None:
        self.baselines_path = Path(baselines_path or DEFAULT_BASELINES_PATH)

    def run_benchmark(self, cfg: PerfBenchmarkConfig) -> BenchmarkResult:
        device = _resolve_device(cfg.device)
        model_sequence_len = cfg.seq_len + (
            cfg.warmup_steps + cfg.measure_steps if cfg.mode == "decode" else 0
        )

        if cfg.synaptic:
            gpt_cfg = GPTSynapticConfig(
                sequence_len=model_sequence_len,
                vocab_size=cfg.vocab_size,
                n_layer=cfg.n_layer,
                n_head=cfg.n_head,
                n_kv_head=cfg.n_head,
                n_embd=cfg.n_embd,
                synapses=True,
                syn_cfg=SynapticConfig(),
            )
            model: nn.Module = GPTSynaptic(gpt_cfg).to(device)
        else:
            gpt_cfg_std = GPTConfig(
                sequence_len=model_sequence_len,
                vocab_size=cfg.vocab_size,
                n_layer=cfg.n_layer,
                n_head=cfg.n_head,
                n_kv_head=cfg.n_head,
                n_embd=cfg.n_embd,
            )
            model = GPT(gpt_cfg_std).to(device)

        inputs = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device=device)
        targets = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device=device)

        if cfg.mode == "train":
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            # Warmup
            for _ in range(cfg.warmup_steps):
                optimizer.zero_grad()
                out = model(inputs, targets=targets)
                loss = out[1] if isinstance(out, tuple) else out
                loss.backward()
                optimizer.step()

            t0 = time.perf_counter()
            for _ in range(cfg.measure_steps):
                optimizer.zero_grad()
                out = model(inputs, targets=targets)
                loss = out[1] if isinstance(out, tuple) else out
                loss.backward()
                optimizer.step()
            elapsed = time.perf_counter() - t0

        else:  # decode mode
            model.eval()
            reset_sequence_state = getattr(model, "reset_sequence_state", None)
            if callable(reset_sequence_state):
                reset_sequence_state()
            engine = Engine(model, tokenizer=None)
            model_cfg = cast(Any, model).config  # nn.Module attribute access types as Tensor | Module
            continuation = torch.randint(
                0,
                cfg.vocab_size,
                (cfg.batch_size, cfg.warmup_steps + cfg.measure_steps),
                device=device,
            )
            cache = KVCache(
                batch_size=cfg.batch_size,
                num_heads=model_cfg.n_kv_head,
                seq_len=cfg.seq_len + cfg.warmup_steps + cfg.measure_steps,
                head_dim=model_cfg.n_embd // model_cfg.n_head,
                num_layers=model_cfg.n_layer,
            )
            with torch.inference_mode():
                # Prompt prefill establishes the representative cache length but is deliberately
                # outside the timed window and token count. Every later call is true Tq=1 decode
                # through the same serving seam as Engine.generate.
                engine._forward(inputs, cache)
                for step in range(cfg.warmup_steps):
                    engine._forward(continuation[:, step : step + 1], cache)
                _synchronize_for_timing(device)
                t0 = time.perf_counter()
                for step in range(cfg.warmup_steps, continuation.size(1)):
                    engine._forward(continuation[:, step : step + 1], cache)
                _synchronize_for_timing(device)
                elapsed = time.perf_counter() - t0
            expected_cache_pos = cfg.seq_len + cfg.warmup_steps + cfg.measure_steps
            if cache.get_pos() != expected_cache_pos:
                raise RuntimeError(
                    "cached decode benchmark did not advance exactly one position per token; "
                    f"got pos={cache.get_pos()}, expected={expected_cache_pos}"
                )

        tokens_per_step = cfg.batch_size * (cfg.seq_len if cfg.mode == "train" else 1)
        total_tokens = tokens_per_step * cfg.measure_steps
        tok_per_sec = total_tokens / max(1e-6, elapsed)
        latency_ms = (elapsed / cfg.measure_steps) * 1000.0

        # Memory estimate in MB
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        mem_mb = param_bytes / (1024.0 * 1024.0)

        return BenchmarkResult(
            name=cfg.name,
            mode=cfg.mode,
            synaptic=cfg.synaptic,
            tok_per_sec=tok_per_sec,
            latency_ms=latency_ms,
            peak_memory_mb=mem_mb,
            steps_measured=cfg.measure_steps,
            batch_size=cfg.batch_size,
            seq_len=cfg.seq_len,
            device=str(device),
            benchmark_config=_config_snapshot(cfg, device),
        )

    def run_all(self, configs: Sequence[PerfBenchmarkConfig] | None = None) -> list[BenchmarkResult]:
        cfgs = configs or STANDARD_BENCHMARK_CONFIGS
        return [self.run_benchmark(c) for c in cfgs]

    def load_baselines(self) -> dict[str, BaselineEntry]:
        if not self.baselines_path.exists():
            return {}
        try:
            data = json.loads(self.baselines_path.read_text(encoding="utf-8"))
            return {k: BaselineEntry(**v) for k, v in data.items()}
        except Exception:
            return {}

    def save_baselines(self, results: list[BenchmarkResult], tolerance: float = 0.25) -> None:
        self.baselines_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {}
        for r in results:
            payload[r.name] = asdict(
                BaselineEntry(
                    name=r.name,
                    mode=r.mode,
                    tok_per_sec=r.tok_per_sec,
                    peak_memory_mb=r.peak_memory_mb,
                    device=r.device,
                    benchmark_config=r.benchmark_config,
                    tolerance=tolerance,
                )
            )
        self.baselines_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def evaluate_gates(
        self,
        results: list[BenchmarkResult],
        baselines: dict[str, BaselineEntry] | None = None,
        override_tolerance: float | None = None,
    ) -> list[GateComparison]:
        base_map = baselines if baselines is not None else self.load_baselines()
        comparisons: list[GateComparison] = []

        for res in results:
            base = base_map.get(res.name)
            if base is None:
                comparisons.append(
                    GateComparison(
                        name=res.name,
                        mode=res.mode,
                        observed_tok_per_sec=res.tok_per_sec,
                        baseline_tok_per_sec=0.0,
                        speed_ratio=0.0,
                        passed=False,
                        detail="No valid baseline found; update baselines explicitly before checking",
                    )
                )
                continue

            provenance_mismatches = []
            if res.mode != base.mode:
                provenance_mismatches.append(f"mode {res.mode!r} != {base.mode!r}")
            if res.device != base.device:
                provenance_mismatches.append(f"device {res.device!r} != {base.device!r}")
            if res.benchmark_config != base.benchmark_config:
                provenance_mismatches.append("benchmark configuration differs")
            if provenance_mismatches:
                comparisons.append(
                    GateComparison(
                        name=res.name,
                        mode=res.mode,
                        observed_tok_per_sec=res.tok_per_sec,
                        baseline_tok_per_sec=base.tok_per_sec,
                        speed_ratio=res.tok_per_sec / max(1e-6, base.tok_per_sec),
                        passed=False,
                        detail="Baseline provenance mismatch: " + "; ".join(provenance_mismatches),
                    )
                )
                continue

            tol = override_tolerance if override_tolerance is not None else base.tolerance
            threshold = base.tok_per_sec * (1.0 - tol)
            speed_ratio = res.tok_per_sec / max(1e-6, base.tok_per_sec)
            passed = res.tok_per_sec >= threshold

            detail = (
                f"Observed {res.tok_per_sec:.1f} tok/s vs baseline {base.tok_per_sec:.1f} tok/s "
                f"({speed_ratio:.2%}, tol={tol:.1%})"
            )
            comparisons.append(
                GateComparison(
                    name=res.name,
                    mode=res.mode,
                    observed_tok_per_sec=res.tok_per_sec,
                    baseline_tok_per_sec=base.tok_per_sec,
                    speed_ratio=speed_ratio,
                    passed=passed,
                    detail=detail,
                )
            )

        return comparisons

    def record_to_registry(
        self,
        results: list[BenchmarkResult],
        registry_path: str = "results/registry.jsonl",
    ) -> None:
        """Log performance run to the experiment registry."""
        for res in results:
            record = RunRecord(
                run_id=f"perf-gate-{res.name}-{int(res.timestamp)}",
                harness="eval",
                metrics={
                    "tok_per_sec": float(res.tok_per_sec),
                    "latency_ms": float(res.latency_ms),
                    "memory_mb": float(res.peak_memory_mb),
                },
                notes=f"Perf benchmark mode={res.mode} synaptic={res.synaptic}",
                verdict="positive",
            )
            try:
                append_record(record, path=registry_path)
            except Exception:
                pass
