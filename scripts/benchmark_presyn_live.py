"""RTX 4090 benchmark gate for the jyb.2 live presynaptic decode kernel.

This benchmark deliberately targets the only exact one-physical-kernel slice: deterministic
``Tq == 1`` decode under ``torch.inference_mode``. It compares both the release primitive and a
real cached GPTSynaptic decode loop against the eager canonical Python path. No result is written
unless ``--output`` is supplied, and an existing result requires ``--overwrite``.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import triton
from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat.engine import KVCache
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.synaptic import (
    SynapticConfig,
    SynapticPresyn,
    build_presyn_state,
)


console = Console()


@dataclass(frozen=True)
class TimingSummary:
    median_ms: float
    p95_ms: float
    minimum_ms: float
    trials: int


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _speedup(eager_ms: float, fused_ms: float) -> float:
    if fused_ms <= 0.0:
        raise RuntimeError(f"invalid non-positive fused CUDA timing: {fused_ms}")
    return eager_ms / fused_ms


def _summarize(samples: list[float]) -> TimingSummary:
    return TimingSummary(
        median_ms=statistics.median(samples),
        p95_ms=_percentile(samples, 0.95),
        minimum_ms=min(samples),
        trials=len(samples),
    )


def _timed_cuda_call(call: Callable[[], None]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    call()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _time_cuda_pair(
    eager_call: Callable[[], None],
    fused_call: Callable[[], None],
    *,
    warmup: int,
    trials: int,
) -> tuple[TimingSummary, TimingSummary]:
    """Alternate eager/fused order to reduce drift and thermal-order bias."""
    for iteration in range(warmup):
        ordered = (
            (eager_call, fused_call) if iteration % 2 == 0 else (fused_call, eager_call)
        )
        for call in ordered:
            call()
    torch.cuda.synchronize()
    eager_samples: list[float] = []
    fused_samples: list[float] = []
    for iteration in range(trials):
        ordered = (
            ((eager_call, eager_samples), (fused_call, fused_samples))
            if iteration % 2 == 0
            else ((fused_call, fused_samples), (eager_call, eager_samples))
        )
        for call, samples in ordered:
            samples.append(_timed_cuda_call(call))
    return _summarize(eager_samples), _summarize(fused_samples)


def _time_cuda_prepared_pair(
    eager_prepare: Callable[[], Callable[[], None]],
    fused_prepare: Callable[[], Callable[[], None]],
    *,
    warmup: int,
    trials: int,
) -> tuple[TimingSummary, TimingSummary]:
    """Alternate prepared decode trials while keeping cache cloning outside timed regions."""
    for iteration in range(warmup):
        ordered = (
            (eager_prepare, fused_prepare)
            if iteration % 2 == 0
            else (fused_prepare, eager_prepare)
        )
        for prepare in ordered:
            prepare()()
    torch.cuda.synchronize()
    eager_samples: list[float] = []
    fused_samples: list[float] = []
    for iteration in range(trials):
        ordered = (
            ((eager_prepare, eager_samples), (fused_prepare, fused_samples))
            if iteration % 2 == 0
            else ((fused_prepare, fused_samples), (eager_prepare, eager_samples))
        )
        for prepare, samples in ordered:
            samples.append(_timed_cuda_call(prepare()))
    return _summarize(eager_samples), _summarize(fused_samples)


def _clone_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        name: [item.clone() for item in value]
        if isinstance(value, list)
        else value.clone()
        for name, value in state.items()
    }


def _state_max_abs(actual: dict[str, Any], expected: dict[str, Any]) -> float:
    maximum = 0.0
    for name in actual:
        actual_items = (
            actual[name] if isinstance(actual[name], list) else [actual[name]]
        )
        expected_items = (
            expected[name] if isinstance(expected[name], list) else [expected[name]]
        )
        if len(actual_items) != len(expected_items):
            raise AssertionError(f"state queue length mismatch for {name}")
        for actual_item, expected_item in zip(actual_items, expected_items):
            torch.testing.assert_close(actual_item, expected_item, rtol=1e-5, atol=1e-6)
            maximum = max(maximum, float((actual_item - expected_item).abs().max()))
    return maximum


def _release_benchmark(
    *,
    batch: int,
    heads: int,
    t_key: int,
    topk: int,
    dtype: torch.dtype,
    warmup: int,
    trials: int,
) -> dict[str, Any]:
    device = torch.device("cuda")
    torch.manual_seed(1337)
    base_cfg = SynapticConfig(
        stochastic_train_frac=0.0, native_presyn=False, attn_topk=topk
    )
    fused_cfg = SynapticConfig(
        stochastic_train_frac=0.0, native_presyn=True, attn_topk=topk
    )
    base_pre = SynapticPresyn(128, base_cfg).to(device)
    fused_pre = SynapticPresyn(128, fused_cfg).to(device)
    fused_pre.load_state_dict(base_pre.state_dict())
    initial = build_presyn_state(batch, t_key, heads, device, dtype, base_cfg)
    drive = torch.randn(batch, heads, 1, topk, device=device, dtype=dtype)
    idx = torch.topk(
        torch.randn(batch, heads, 1, t_key, device=device), topk, dim=-1
    ).indices
    valid = torch.ones_like(drive, dtype=torch.bool)
    initial_logits = torch.randn(batch, heads, 1, t_key, device=device, dtype=dtype)

    def eager_step(
        state: dict[str, Any], logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        release = base_pre.release_canonical(
            state, drive, idx, train=False, valid=valid
        )
        augmentation = torch.zeros_like(logits)
        source = base_cfg.lambda_loge * torch.log(base_cfg.epsilon + release).to(dtype)
        if base_cfg.loge_bias_clamp > 0.0:
            source = source.clamp(-base_cfg.loge_bias_clamp, base_cfg.loge_bias_clamp)
        augmentation.scatter_add_(-1, idx, source * valid)
        return release, logits + augmentation

    def make_call(
        pre: SynapticPresyn, state: dict[str, Any], *, native: bool
    ) -> Callable[[], None]:
        logits = initial_logits.clone()

        def call() -> None:
            if native:
                pre.release_canonical(
                    state, drive, idx, train=False, valid=valid, logits=logits
                )
                return
            _ = eager_step(state, logits)

        return call

    with torch.inference_mode():
        reference_state = _clone_state(initial)
        kernel_state = _clone_state(initial)
        expected_release, expected_logits = eager_step(reference_state, initial_logits)
        kernel_logits = initial_logits.clone()
        torch.cuda.synchronize()
        cold_start = time.perf_counter()
        actual_release = fused_pre.release_canonical(
            kernel_state,
            drive,
            idx,
            train=False,
            valid=valid,
            logits=kernel_logits,
        )
        torch.cuda.synchronize()
        cold_start_ms = 1000.0 * (time.perf_counter() - cold_start)
        torch.testing.assert_close(
            actual_release, expected_release, rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(kernel_logits, expected_logits, rtol=1e-5, atol=1e-6)
        release_max_abs = float((actual_release - expected_release).abs().max())
        logits_max_abs = float((kernel_logits - expected_logits).abs().max())
        state_max_abs = _state_max_abs(kernel_state, reference_state)

        eager_state = _clone_state(initial)
        fused_state = _clone_state(initial)
        eager, fused = _time_cuda_pair(
            make_call(base_pre, eager_state, native=False),
            make_call(fused_pre, fused_state, native=True),
            warmup=warmup,
            trials=trials,
        )
    return {
        "shape": {"batch": batch, "heads": heads, "t_key": t_key, "topk": topk},
        "scope": "canonical release plus top-k log-bias injection",
        "correctness": {
            "passed": True,
            "release_max_abs": release_max_abs,
            "logits_max_abs": logits_max_abs,
            "state_max_abs": state_max_abs,
        },
        "cold_start_ms": cold_start_ms,
        "eager": asdict(eager),
        "fused": asdict(fused),
        "speedup": _speedup(eager.median_ms, fused.median_ms),
    }


def _make_model(
    *, native_presyn: bool, sequence_len: int, dtype: torch.dtype
) -> GPTSynaptic:
    syn_cfg = SynapticConfig(
        native_presyn=native_presyn,
        stochastic_train_frac=0.0,
        attn_topk=32,
    )
    cfg = GPTSynapticConfig(
        sequence_len=sequence_len,
        vocab_size=1024,
        n_layer=2,
        n_head=8,
        n_kv_head=4,
        n_embd=512,
        syn_cfg=syn_cfg,
        dropout=0.0,
    )
    return GPTSynaptic(cfg).to(device="cuda", dtype=dtype).eval()


def _new_cache(model: GPTSynaptic, batch: int) -> KVCache:
    cfg = model.config
    return KVCache(
        batch_size=batch,
        num_heads=cfg.n_kv_head,
        seq_len=cfg.sequence_len,
        head_dim=cfg.n_embd // cfg.n_head,
        num_layers=cfg.n_layer,
    )


def _model_correctness(
    eager_model: GPTSynaptic,
    fused_model: GPTSynaptic,
    prefix: torch.Tensor,
    tokens: torch.Tensor,
) -> dict[str, Any]:
    eager_cache = _new_cache(eager_model, int(prefix.shape[0]))
    fused_cache = _new_cache(fused_model, int(prefix.shape[0]))
    eager_prefix, _ = eager_model(prefix, kv_cache=eager_cache, train_mode=False)
    fused_prefix, _ = fused_model(prefix, kv_cache=fused_cache, train_mode=False)
    torch.testing.assert_close(fused_prefix, eager_prefix, rtol=1e-5, atol=1e-6)
    logits_max_abs = float((fused_prefix - eager_prefix).abs().max())

    for position in range(tokens.shape[1]):
        token = tokens[:, position : position + 1]
        eager_logits, _ = eager_model(token, kv_cache=eager_cache, train_mode=False)
        fused_logits, _ = fused_model(token, kv_cache=fused_cache, train_mode=False)
        torch.testing.assert_close(fused_logits, eager_logits, rtol=1e-5, atol=1e-6)
        logits_max_abs = max(
            logits_max_abs, float((fused_logits - eager_logits).abs().max())
        )

    if not isinstance(eager_cache.presyn_state, list) or not isinstance(
        fused_cache.presyn_state, list
    ):
        raise AssertionError("expected per-layer presynaptic state lists")
    if len(eager_cache.presyn_state) != len(fused_cache.presyn_state):
        raise AssertionError("per-layer presynaptic state count mismatch")
    state_max_abs = 0.0
    for fused_state, eager_state in zip(
        fused_cache.presyn_state, eager_cache.presyn_state
    ):
        state_max_abs = max(state_max_abs, _state_max_abs(fused_state, eager_state))
    if eager_cache.kv_cache is None or fused_cache.kv_cache is None:
        raise AssertionError("expected initialized KV caches")
    if eager_cache.pos != fused_cache.pos:
        raise AssertionError("KV cache positions differ")
    eager_kv = eager_cache.kv_cache[..., : eager_cache.pos, :]
    fused_kv = fused_cache.kv_cache[..., : fused_cache.pos, :]
    torch.testing.assert_close(fused_kv, eager_kv, rtol=1e-5, atol=1e-6)
    kv_max_abs = float((fused_kv - eager_kv).abs().max())
    return {
        "passed": True,
        "logits_max_abs": logits_max_abs,
        "state_max_abs": state_max_abs,
        "kv_max_abs": kv_max_abs,
    }


def _decode_trial(
    model: GPTSynaptic, prefix: torch.Tensor, tokens: torch.Tensor
) -> Callable[[], Callable[[], None]]:
    base_cache = _new_cache(model, int(prefix.shape[0]))
    model(prefix, kv_cache=base_cache, train_mode=False)

    def prepare() -> Callable[[], None]:
        cache = _new_cache(model, int(prefix.shape[0]))
        cache.prefill(base_cache)

        def call() -> None:
            for position in range(tokens.shape[1]):
                model(
                    tokens[:, position : position + 1], kv_cache=cache, train_mode=False
                )

        return call

    return prepare


def _model_benchmark(
    *,
    batch: int,
    prefix_len: int,
    decode_tokens: int,
    dtype: torch.dtype,
    warmup: int,
    trials: int,
) -> dict[str, Any]:
    torch.manual_seed(1337)
    eager_model = _make_model(
        native_presyn=False, sequence_len=prefix_len + decode_tokens + 8, dtype=dtype
    )
    fused_model = _make_model(
        native_presyn=True, sequence_len=prefix_len + decode_tokens + 8, dtype=dtype
    )
    fused_model.load_state_dict(eager_model.state_dict())
    prefix = torch.randint(0, 1024, (batch, prefix_len), device="cuda")
    tokens = torch.randint(0, 1024, (batch, decode_tokens), device="cuda")
    with torch.inference_mode():
        correctness = _model_correctness(eager_model, fused_model, prefix, tokens)
        eager, fused = _time_cuda_prepared_pair(
            _decode_trial(eager_model, prefix, tokens),
            _decode_trial(fused_model, prefix, tokens),
            warmup=warmup,
            trials=trials,
        )
    eager_per_token = eager.median_ms / decode_tokens
    fused_per_token = fused.median_ms / decode_tokens
    return {
        "shape": {
            "batch": batch,
            "prefix_len": prefix_len,
            "decode_tokens": decode_tokens,
        },
        "correctness": correctness,
        "eager": asdict(eager),
        "fused": asdict(fused),
        "eager_ms_per_token": eager_per_token,
        "fused_ms_per_token": fused_per_token,
        "speedup": _speedup(eager_per_token, fused_per_token),
    }


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32}[name]


def _git_provenance() -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return completed.stdout.strip()

    benchmark_path = str(Path(__file__).resolve().relative_to(repo))
    source_tracked = (
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", benchmark_path],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        ).returncode
        == 0
    )
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "tracked_worktree_clean": not bool(
            git("status", "--porcelain", "--untracked-files=no")
        ),
        "benchmark_source_tracked": source_tracked,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float32",), default="float32")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--t-key", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--prefix-len", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "jyb.2 benchmark requires a CUDA GPU; no device is available"
        )
    if args.topk < 1 or args.topk > args.t_key:
        raise ValueError("topk must be in [1, t_key]")
    if min(args.batch, args.heads, args.t_key, args.prefix_len, args.decode_tokens) < 1:
        raise ValueError("all benchmark dimensions must be positive")
    if min(args.warmup, args.trials) < 1:
        raise ValueError("warmup and trials must be positive")
    if args.output is not None and args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing benchmark: {args.output}"
        )

    dtype = _dtype(args.dtype)
    release = _release_benchmark(
        batch=args.batch,
        heads=args.heads,
        t_key=args.t_key,
        topk=args.topk,
        dtype=dtype,
        warmup=args.warmup,
        trials=args.trials,
    )
    model = _model_benchmark(
        batch=args.batch,
        prefix_len=args.prefix_len,
        decode_tokens=args.decode_tokens,
        dtype=dtype,
        warmup=args.warmup,
        trials=args.trials,
    )
    release_passed = bool(release["speedup"] >= 1.20)
    model_passed = bool(model["speedup"] >= 1.05)
    gpu_name = torch.cuda.get_device_name()
    hardware_passed = "RTX 4090" in gpu_name.upper()
    properties = torch.cuda.get_device_properties(0)
    provenance = _git_provenance()
    acceptance_eligible = bool(
        hardware_passed
        and release_passed
        and model_passed
        and provenance["tracked_worktree_clean"]
        and provenance["benchmark_source_tracked"]
    )
    result = {
        "schema_version": 1,
        "seed": 1337,
        "arguments": {
            "dtype": args.dtype,
            "batch": args.batch,
            "heads": args.heads,
            "t_key": args.t_key,
            "topk": args.topk,
            "prefix_len": args.prefix_len,
            "decode_tokens": args.decode_tokens,
            "warmup": args.warmup,
            "trials": args.trials,
        },
        "git": provenance,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "triton": triton.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "gpu": {
            "name": gpu_name,
            "capability": list(torch.cuda.get_device_capability()),
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
            "rtx_4090_eligible": hardware_passed,
        },
        "dtype": args.dtype,
        "release": release,
        "model_decode": model,
        "gates": {
            "release_speedup_min": 1.20,
            "model_speedup_min": 1.05,
            "release_passed": release_passed,
            "model_passed": model_passed,
            "hardware_passed": hardware_passed,
            "acceptance_eligible": acceptance_eligible,
        },
    }

    table = Table(title="jyb.2 live presyn decode benchmark")
    table.add_column("Path")
    table.add_column("Eager median")
    table.add_column("Fused median")
    table.add_column("Speedup")
    table.add_row(
        "release",
        f"{release['eager']['median_ms']:.4f} ms",
        f"{release['fused']['median_ms']:.4f} ms",
        f"{release['speedup']:.3f}×",
    )
    table.add_row(
        "model decode / token",
        f"{model['eager_ms_per_token']:.4f} ms",
        f"{model['fused_ms_per_token']:.4f} ms",
        f"{model['speedup']:.3f}×",
    )
    console.print(table)
    console.print_json(data=result)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        console.print(f"[green]Wrote[/green] {args.output}")
    return 0 if acceptance_eligible else 1


if __name__ == "__main__":
    raise SystemExit(main())
