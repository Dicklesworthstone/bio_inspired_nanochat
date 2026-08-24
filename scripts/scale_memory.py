"""Memory-budget estimator + throughput micro-benchmark for the scale-up (bead hwxb.2.2).

Predicts the VRAM footprint of a Phase-0 config BEFORE committing a multi-hour 2×4090 run,
so we know the headroom left for the synaptic per-key state once Phase 3 turns mechanisms ON.

What is EXACT (CPU-computable, unit-tested in tests/test_scaleup_memory.py):
  - param_bytes        : Σ numel·element_size over model.parameters()
  - buffer_bytes       : Σ over persistent registered buffers (eligibility traces, EMAs, …)
  - optimizer_bytes    : the moment state — AdamW keeps 2 (exp_avg, exp_avg_sq), Muon keeps 1
                         (momentum_buffer), each zeros_like(param). ZeRO-style Dist* optimizers
                         shard moment state, so per-rank ≈ total / world_size. (Excludes the
                         negligible per-param int64 `step` scalar.)

What is a ROUGH ESTIMATE (documented as such; depends on autocast / activation checkpointing /
the exact kernels, which only a real run on the 4090 pins down — that measured table is hwxb.2.2's
GPU-gated residual):
  - activation_bytes_est       : transformer activations ≈ B·T·d·L·bytes·k
  - synaptic_state_bytes_est   : per-key presyn buffers ≈ B·H·T·n_presyn·L·bytes (synaptic only)

Run it: `python -m scripts.scale_memory --depth 10 --tie-embeddings --batch 16 --seq 1024 --world-size 2`
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat.torch_imports import torch
from bio_inspired_nanochat.muon import DistMuon, Muon
from bio_inspired_nanochat.run_logging import gather_provenance

console = Console()

# Rough activation multiplier: residual stream + attention scores + MLP intermediates, in bf16.
# Deliberately conservative; real autocast/checkpointing shifts this. Tune against a measured run.
_ACT_MULT = 16
# Number of per-key presynaptic state buffers carried per layer (C/BUF/RRP/RES/PR/CL/E ≈ 7).
_N_PRESYN_BUFFERS = 7


@dataclass
class MemoryBudget:
    param_bytes: int
    buffer_bytes: int
    optimizer_bytes: int          # per-rank moment state
    activation_bytes_est: int
    synaptic_state_bytes_est: int
    world_size: int
    batch: int
    seq: int

    @property
    def persistent_bytes(self) -> int:
        """Resident model + optimizer state on one rank (params + buffers + moments)."""
        return self.param_bytes + self.buffer_bytes + self.optimizer_bytes

    @property
    def total_est_bytes(self) -> int:
        return self.persistent_bytes + self.activation_bytes_est + self.synaptic_state_bytes_est

    def as_gb(self) -> dict[str, float]:
        gb = 1024**3
        return {
            "param_gb": self.param_bytes / gb,
            "buffer_gb": self.buffer_bytes / gb,
            "optimizer_gb": self.optimizer_bytes / gb,
            "activation_est_gb": self.activation_bytes_est / gb,
            "synaptic_state_est_gb": self.synaptic_state_bytes_est / gb,
            "persistent_gb": self.persistent_bytes / gb,
            "total_est_gb": self.total_est_bytes / gb,
        }

    def headroom_gb(self, vram_gb: float) -> float:
        return vram_gb - self.total_est_bytes / 1024**3


@dataclass(frozen=True)
class BenchmarkResult:
    """One reproducible train/inference measurement row."""

    variant: str
    mode: str
    batch: int
    seq: int
    steps: int
    warmup: int
    seed: int
    device: str
    world_size: int
    tok_per_sec: float
    step_ms: float
    peak_allocated_gb: float | None
    peak_reserved_gb: float | None
    gpu_utilization_pct: float | None
    profile_path: str | None
    provenance: dict[str, Any]


# --------------------------------------------------------------------------- #
# Exact (CPU-computable) terms
# --------------------------------------------------------------------------- #
def param_bytes(model: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def buffer_bytes(model: torch.nn.Module) -> int:
    """Persistent registered buffers only (RoPE cos/sin are persistent=False, so excluded)."""
    return sum(b.numel() * b.element_size() for b in model.buffers())


def optimizer_moment_bytes(model: torch.nn.Module, *, world_size: int = 1) -> int:
    """Per-rank optimizer moment-state bytes from the model's real param grouping.

    AdamW → 2 moments/param; Muon → 1/param; each the same dtype/shape as the param.
    Dist* optimizers shard moment state ZeRO-style, so we divide by world_size (the
    tiny non-shardable 0-D/odd params are not sharded, but they are negligible).
    """
    total = 0
    for opt in model.setup_optimizers():
        per_param = 1 if isinstance(opt, (Muon, DistMuon)) else 2
        for group in opt.param_groups:
            for p in group["params"]:
                total += per_param * p.numel() * p.element_size()
    return total // max(1, world_size)


# --------------------------------------------------------------------------- #
# Rough estimates
# --------------------------------------------------------------------------- #
def activation_bytes_est(config, *, batch: int, seq: int, bytes_per: int = 2) -> int:
    d = int(config.n_embd)
    layers = int(config.n_layer)
    return _ACT_MULT * batch * seq * d * layers * bytes_per


def synaptic_state_bytes_est(config, *, batch: int, seq: int, bytes_per: int = 4) -> int:
    """Per-key presyn buffers ≈ B·H·T·n_buffers·L. Zero for the vanilla model."""
    if not getattr(config, "synapses", False):
        return 0
    heads = int(config.n_head)
    layers = int(config.n_layer)
    return _N_PRESYN_BUFFERS * batch * heads * seq * layers * bytes_per


def estimate(model: torch.nn.Module, config, *, batch: int, seq: int, world_size: int = 1) -> MemoryBudget:
    return MemoryBudget(
        param_bytes=param_bytes(model),
        buffer_bytes=buffer_bytes(model),
        optimizer_bytes=optimizer_moment_bytes(model, world_size=world_size),
        activation_bytes_est=activation_bytes_est(config, batch=batch, seq=seq),
        synaptic_state_bytes_est=synaptic_state_bytes_est(config, batch=batch, seq=seq),
        world_size=world_size,
        batch=batch,
        seq=seq,
    )


# --------------------------------------------------------------------------- #
# Throughput micro-benchmark (runs on whatever device; ready for the 4090)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _noop():  # pragma: no cover
    pass


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _gpu_utilization(device: torch.device) -> float | None:
    if device.type != "cuda" or not hasattr(torch.cuda, "utilization"):
        return None
    try:
        return float(torch.cuda.utilization(device))
    except (ImportError, OSError, RuntimeError):
        return None


def measure_throughput(
    model,
    config,
    *,
    batch: int,
    seq: int,
    steps: int = 20,
    warmup: int = 5,
    device: str = "cpu",
    mode: str = "train",
    seed: int = 0,
    profile_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Measure synchronized tok/s, latency, memory, utilization, and an optional profiler trace.

    ``mode='train'`` performs forward/backward/AdamW steps; ``mode='infer'`` measures full-prompt
    forward latency without KV-cache decode.  CUDA timing is explicitly synchronized.  A requested
    profile is a separate post-measurement step so profiler overhead never contaminates throughput.
    """
    if mode not in {"train", "infer"}:
        raise ValueError(f"mode must be 'train' or 'infer', got {mode!r}")
    if min(batch, seq, steps) < 1 or warmup < 0:
        raise ValueError("batch, seq, and steps must be positive; warmup must be non-negative")
    synaptic = getattr(config, "synapses", False)
    torch_device = torch.device(device)
    model.to(torch_device).train(mode == "train")
    optimizers = model.setup_optimizers() if mode == "train" else []
    vocab = int(config.vocab_size)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randint(0, vocab, (batch, seq), generator=gen).to(torch_device)
    y = torch.randint(0, vocab, (batch, seq), generator=gen).to(torch_device)

    def _step():
        if mode == "infer":
            with torch.inference_mode():
                return model(x, None, None, train_mode=False) if synaptic else model(x)
        if synaptic:
            _, loss = model(x, y, None, train_mode=True)
        else:
            loss = model(x, y)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        return loss

    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch_device)

    for _ in range(warmup):
        _step()
    _synchronize(torch_device)
    t0 = time.monotonic()
    utilization: list[float] = []
    utilization_interval = max(1, steps // 10)
    for step_index in range(steps):
        _step()
        if step_index % utilization_interval == 0:
            sample = _gpu_utilization(torch_device)
            if sample is not None:
                utilization.append(sample)
    _synchronize(torch_device)
    dt = time.monotonic() - t0
    world_size = 1
    dist = torch.distributed
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        elapsed = torch.tensor(dt, dtype=torch.float64, device=torch_device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        dt = float(elapsed.item())
    tok = batch * seq * steps * world_size
    peak_allocated = peak_reserved = None
    if torch_device.type == "cuda":
        gb = 1024**3
        peak_allocated = torch.cuda.max_memory_allocated(torch_device) / gb
        peak_reserved = torch.cuda.max_memory_reserved(torch_device) / gb

    trace = None
    if profile_path is not None:
        trace_path = Path(profile_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch_device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
        ) as profiler:
            _step()
            _synchronize(torch_device)
        profiler.export_chrome_trace(str(trace_path))
        trace = str(trace_path)

    return {
        "tok_per_sec": tok / dt if dt > 0 else float("inf"),
        "step_ms": 1000.0 * dt / steps,
        "steps": steps,
        "peak_allocated_gb": peak_allocated,
        "peak_reserved_gb": peak_reserved,
        "gpu_utilization_pct": sum(utilization) / len(utilization) if utilization else None,
        "profile_path": trace,
    }


def _parse_int_list(value: str, *, name: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated integer list") from exc
    if not values or any(item < 1 for item in values):
        raise ValueError(f"{name} must contain positive integers")
    return values


def run_benchmark_matrix(
    *,
    depth: int,
    variants: list[str],
    modes: list[str],
    batches: list[int],
    seqs: list[int],
    steps: int,
    warmup: int,
    seed: int,
    device: str,
    tie_embeddings: bool,
    profile_dir: str | None = None,
) -> list[BenchmarkResult]:
    """Run a fixed-seed variant × mode × batch × sequence sweep."""
    unknown_variants = sorted(set(variants) - {"vanilla", "bio"})
    unknown_modes = sorted(set(modes) - {"train", "infer"})
    if not variants or not modes:
        raise ValueError("variants and modes must each contain at least one entry")
    if unknown_variants or unknown_modes:
        raise ValueError(f"unknown variants={unknown_variants}, modes={unknown_modes}")
    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if resolved_device == "auto":
        resolved_device = "cpu"
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but torch.cuda.is_available() is false")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    rows: list[BenchmarkResult] = []
    cleanup = False
    if world_size > 1:
        if resolved_device != "cuda":
            raise RuntimeError("multi-process benchmark requires CUDA/NCCL")
        from bio_inspired_nanochat.common import compute_init

        _, _, _, _, initialized_device = compute_init("cuda")
        resolved_device = str(initialized_device)
        cleanup = True
    gpu_name = None
    if torch.device(resolved_device).type == "cuda":
        gpu_name = torch.cuda.get_device_name(torch.device(resolved_device))
    provenance = gather_provenance(
        {
            "seed": seed,
            "gpu_name": gpu_name,
            "cuda_version": torch.version.cuda,
            "world_size": world_size,
            "rank": rank,
            "nccl_p2p_level": os.environ.get("NCCL_P2P_LEVEL"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
    )
    try:
        for variant in variants:
            for mode in modes:
                for batch in batches:
                    for seq in seqs:
                        torch.manual_seed(seed)
                        model, config = _build_model(depth, seq, variant == "bio", tie_embeddings)
                        profile_path = None
                        if profile_dir is not None:
                            profile_path = str(
                                Path(profile_dir)
                                / f"{variant}_{mode}_b{batch}_t{seq}_rank{rank}.json"
                            )
                        measured = measure_throughput(
                            model,
                            config,
                            batch=batch,
                            seq=seq,
                            steps=steps,
                            warmup=warmup,
                            device=resolved_device,
                            mode=mode,
                            seed=seed,
                            profile_path=profile_path,
                        )
                        rows.append(
                            BenchmarkResult(
                                variant=variant,
                                mode=mode,
                                batch=batch,
                                seq=seq,
                                steps=steps,
                                warmup=warmup,
                                seed=seed,
                                device=resolved_device,
                                world_size=world_size,
                                tok_per_sec=float(measured["tok_per_sec"]),
                                step_ms=float(measured["step_ms"]),
                                peak_allocated_gb=measured["peak_allocated_gb"],
                                peak_reserved_gb=measured["peak_reserved_gb"],
                                gpu_utilization_pct=measured["gpu_utilization_pct"],
                                profile_path=measured["profile_path"],
                                provenance=provenance,
                            )
                        )
    finally:
        if cleanup:
            from bio_inspired_nanochat.common import compute_cleanup

            compute_cleanup()
    return rows


def write_benchmark_jsonl(rows: list[BenchmarkResult], path: str | os.PathLike[str]) -> Path:
    """Write finite, schema-stable benchmark rows for regression analysis."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for row in rows:
        record = asdict(row)
        if not math.isfinite(row.tok_per_sec) or not math.isfinite(row.step_ms):
            raise ValueError("benchmark timing must be finite before it can be recorded")
        records.append(json.dumps(record, sort_keys=True))
    output.write_text("\n".join(records) + ("\n" if records else ""), encoding="utf-8")
    return output


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_model(depth: int, seq: int, synapses: bool, tie: bool):
    model_dim = depth * 64
    n_head = max(1, (model_dim + 127) // 128)
    common = dict(sequence_len=seq, vocab_size=65536, n_layer=depth, n_head=n_head,
                  n_kv_head=n_head, n_embd=model_dim, tie_embeddings=tie)
    if synapses:
        from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig

        cfg = GPTSynapticConfig(synapses=True, **common)  # ty: ignore[invalid-argument-type]
        model = GPTSynaptic(cfg)
    else:
        from bio_inspired_nanochat.gpt import GPT, GPTConfig

        cfg = GPTConfig(**common)  # ty: ignore[invalid-argument-type]
        model = GPT(cfg)
    model.init_weights()
    return model, cfg


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Scale-up memory-budget estimate + throughput")
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--synapses", action="store_true")
    p.add_argument("--tie-embeddings", action="store_true")
    p.add_argument("--vram-gb", type=float, default=24.0)
    p.add_argument("--throughput", action="store_true", help="also run the throughput micro-bench")
    p.add_argument("--matrix", action="store_true", help="run the reproducible perf sweep")
    p.add_argument("--variants", default="vanilla,bio", help="matrix variants: vanilla,bio")
    p.add_argument("--modes", default="train,infer", help="matrix modes: train,infer")
    p.add_argument("--batches", default="1,2,4,8", help="matrix batch-size CSV")
    p.add_argument("--seqs", default="512,1024,2048", help="matrix sequence-length CSV")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--output-jsonl", default="runs/perf_4090/benchmark.jsonl")
    p.add_argument("--profile-dir", default=None, help="optional Chrome-trace directory")
    args = p.parse_args(argv)

    if args.matrix:
        rows = run_benchmark_matrix(
            depth=args.depth,
            variants=[part.strip() for part in args.variants.split(",") if part.strip()],
            modes=[part.strip() for part in args.modes.split(",") if part.strip()],
            batches=_parse_int_list(args.batches, name="batches"),
            seqs=_parse_int_list(args.seqs, name="seqs"),
            steps=args.steps,
            warmup=args.warmup,
            seed=args.seed,
            device=args.device,
            tie_embeddings=args.tie_embeddings,
            profile_dir=args.profile_dir,
        )
        if int(os.environ.get("RANK", "0")) != 0:
            return 0
        path = write_benchmark_jsonl(rows, args.output_jsonl)
        table = Table(title=f"Performance sweep → {path}")
        for heading in ("variant", "mode", "batch", "seq", "tok/s", "step ms", "peak GB", "GPU %"):
            table.add_column(heading, justify="right" if heading not in {"variant", "mode"} else "left")
        for row in rows:
            table.add_row(
                row.variant,
                row.mode,
                str(row.batch),
                str(row.seq),
                f"{row.tok_per_sec:,.1f}",
                f"{row.step_ms:.2f}",
                "n/a" if row.peak_allocated_gb is None else f"{row.peak_allocated_gb:.2f}",
                "n/a" if row.gpu_utilization_pct is None else f"{row.gpu_utilization_pct:.1f}",
            )
        console.print(table)
        return 0

    model, cfg = _build_model(args.depth, args.seq, args.synapses, args.tie_embeddings)
    b = estimate(model, cfg, batch=args.batch, seq=args.seq, world_size=args.world_size)
    gb = b.as_gb()
    console.print(
        f"[bold]memory budget[/bold] depth={args.depth} synapses={args.synapses} "
        f"tie={args.tie_embeddings} batch={args.batch} seq={args.seq} "
        f"world_size={args.world_size}"
    )
    for k, v in gb.items():
        console.print(f"  {k:24s} {v:8.3f} GB")
    console.print(
        f"  {'headroom_vs_' + str(args.vram_gb) + 'GB':24s} "
        f"{b.headroom_gb(args.vram_gb):8.3f} GB"
    )
    if args.throughput:
        resolved_device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if resolved_device == "auto":
            resolved_device = "cpu"
        tp = measure_throughput(
            model,
            cfg,
            batch=args.batch,
            seq=args.seq,
            steps=args.steps,
            warmup=args.warmup,
            device=resolved_device,
            seed=args.seed,
        )
        console.print(
            f"  throughput: {tp['tok_per_sec']:.1f} tok/s  ({tp['step_ms']:.1f} ms/step)"
        )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
