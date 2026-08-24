"""
Benchmark utilities.

Modes:
1) FlexAttention benchmark (default): throughput/VRAM on CUDA.
2) CA init micro-benchmark: compare `init_type` settings on a deterministic synthetic task,
   logging loss + throughput + weight stats, and saving CSV/plots under `runs/`.

Examples:
- Flex benchmark (CUDA):
  python -m scripts.benchmark_flex

- CA init micro-benchmark (CPU):
  python -m scripts.benchmark_flex --mode=ca_init --steps=400 --device_type=cpu

- CA init micro-benchmark (CUDA, if available):
  python -m scripts.benchmark_flex --mode=ca_init --device_type=cuda --dtype=bfloat16
"""

from __future__ import annotations

import csv
import json
import os
import platform
import runpy
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from torch.nn.attention import SDPBackend, sdpa_kernel

from bio_inspired_nanochat.common import (
    autodetect_device_type,
    compute_cleanup,
    compute_init,
)
from bio_inspired_nanochat.gpt import GPT, GPTConfig
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.synaptic import SynapticConfig

console = Console()


# -----------------------------------------------------------------------------
# User settings (override via bio_inspired_nanochat/configurator.py)

mode = "flex"  # flex | ca_init

# Flex benchmark settings
batch_size = 4
seq_len = 2048
n_layer = 12
n_head = 12
n_embd = 768
attention_cases = (
    "vanilla_sdpa_auto,vanilla_sdpa_flash,vanilla_sdpa_math,"
    "synaptic_dense,synaptic_flex"
)
attention_warmup_steps = 5
attention_benchmark_steps = 20
attention_compile = True
attention_require_4090 = True
attention_seed = 42
attention_dtype = "bfloat16"
attention_out_dir = "runs/attention_backends"

# CA init micro-benchmark settings
synapses = 0  # 0=GPT, 1=GPTSynaptic
init_types = "baseline,ca_rule30,ca_rule116"
init_seed = 42
train_seed = 123
steps = 2000
log_every = 10
spectrum_every = 200
out_dir = "runs/ca_init_microbench"

# Tiny model defaults (override as needed)
micro_depth = 2
micro_seq_len = 128
micro_vocab_size = 256
micro_n_head = 4
micro_n_embd = 128
lr = 3e-4

# Runtime
device_type = ""  # cuda|cpu|mps (empty => autodetect)
dtype = "float32"  # float32|bfloat16|float16 (used for ca_init mode)

config_keys = [
    k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
_configured = runpy.run_path(
    os.path.join("bio_inspired_nanochat", "configurator.py"),
    init_globals=globals(),
    run_name="_benchmark_configurator",
)
for _config_key in config_keys:
    globals()[_config_key] = _configured[_config_key]


def _parse_dtype(name: str) -> torch.dtype:
    n = name.strip().lower()
    if n in ("float32", "fp32"):
        return torch.float32
    if n in ("bfloat16", "bf16"):
        return torch.bfloat16
    if n in ("float16", "fp16"):
        return torch.float16
    raise ValueError(f"Unknown dtype {name!r}")


def _synthetic_next_plus_one(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    start = torch.randint(0, vocab_size, (batch_size, 1), generator=generator, device=device)
    ar = torch.arange(seq_len + 1, device=device).view(1, -1)
    toks = (start + ar) % vocab_size
    return toks[:, :-1].to(torch.long), toks[:, 1:].to(torch.long)


def _cosine_to_init(cur: torch.Tensor, init: torch.Tensor, eps: float = 1e-12) -> float:
    a = cur.reshape(-1).to(torch.float32)
    b = init.reshape(-1).to(torch.float32)
    denom = (a.norm() * b.norm()).clamp_min(float(eps))
    return float((a @ b / denom).item())


def _svd_topk_stats(w: torch.Tensor, k: int = 8) -> dict[str, float]:
    s = torch.linalg.svdvals(w.to(torch.float32))
    k_eff = min(int(k), int(s.numel()))
    out: dict[str, float] = {"sv_mean": float(s.mean().item()), "sv_max": float(s.max().item())}
    for i in range(k_eff):
        out[f"sv_{i}"] = float(s[i].item())
    return out


@dataclass(frozen=True)
class _RunResult:
    init_type: str
    csv_path: str
    loss_png_path: str
    steps: int
    final_loss: float
    tok_per_sec: float
    sim_at_200: float


def _bench_ca_init() -> list[_RunResult]:
    dtp = _parse_dtype(dtype)
    dty = autodetect_device_type() if device_type == "" else device_type
    ddp, _ddp_rank, _, _, device = compute_init(dty)
    if ddp:
        raise RuntimeError("ca_init benchmark is intended for single-process runs (no torchrun/DDP).")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    run_dir = Path(out_dir) / datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]CA init micro-benchmark[/bold] → {run_dir}")

    types = [t.strip() for t in init_types.split(",") if t.strip()]
    if not types:
        raise ValueError("init_types must contain at least one entry")

    results: list[_RunResult] = []
    for itype in types:
        # Ensure baseline init is also deterministic with respect to init_seed.
        torch.manual_seed(int(init_seed))
        if device.type == "cuda":
            torch.cuda.manual_seed(int(init_seed))

        if int(synapses) == 1:
            syn_cfg = SynapticConfig(
                enable_presyn=True,
                enable_hebbian=True,
                enable_metabolism=False,
                use_flex_attention=False,
                native_genetics=False,
            )
            cfg = GPTSynapticConfig(
                sequence_len=int(micro_seq_len),
                vocab_size=int(micro_vocab_size),
                n_layer=int(micro_depth),
                n_head=int(micro_n_head),
                n_kv_head=int(micro_n_head),
                n_embd=int(micro_n_embd),
                synapses=True,
                syn_cfg=syn_cfg,
                init_type=str(itype),
                init_seed=int(init_seed),
            )
            with torch.device("meta"):
                model = GPTSynaptic(cfg)
            model.to_empty(device=device)
            model.init_weights()
        else:
            cfg = GPTConfig(
                sequence_len=int(micro_seq_len),
                vocab_size=int(micro_vocab_size),
                n_layer=int(micro_depth),
                n_head=int(micro_n_head),
                n_kv_head=int(micro_n_head),
                n_embd=int(micro_n_embd),
                init_type=str(itype),
                init_seed=int(init_seed),
            )
            with torch.device("meta"):
                model = GPT(cfg)
            model.to_empty(device=device)
            model.init_weights()

        model.train()
        model.to(dtype=dtp)
        if int(synapses) == 0:
            # GPT.forward asserts RoPE buffers are bfloat16.
            model.cos = model.cos.to(dtype=torch.bfloat16)
            model.sin = model.sin.to(dtype=torch.bfloat16)

        opt = torch.optim.AdamW(model.parameters(), lr=float(lr), betas=(0.9, 0.95), eps=1e-8)
        g = torch.Generator(device=device)
        g.manual_seed(int(train_seed))

        # Track a couple of representative matrices (fallback to first 2D params if names don't exist).
        preferred = [
            "transformer.h.0.attn.c_q.weight",
            "transformer.h.0.mlp.c_fc.weight",
        ]
        tracked: list[tuple[str, torch.Tensor]] = []
        init_snap: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.ndim == 2 and (name in preferred or not tracked):
                tracked.append((name, p))
            if len(tracked) >= 2 and all(n in dict(tracked) for n in preferred):
                break
        if not tracked:
            raise RuntimeError("No 2D parameters found to track for similarity/spectrum metrics.")
        for name, p in tracked:
            init_snap[name] = p.detach().to(torch.float32).cpu().clone()

        csv_path = run_dir / f"{itype}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "step",
                "loss",
                "dt_ms",
                "tok_per_sec",
                "sim_min",
                "w_norm_0",
                "w_norm_1",
            ]
            # SVD keys (only written on spectrum_every steps; blank otherwise).
            svd_keys = ["sv_mean", "sv_max", "sv_0", "sv_1", "sv_2", "sv_3"]
            for ki in svd_keys:
                fieldnames.append(f"{tracked[0][0]}:{ki}")
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            losses: list[float] = []
            tok_rates: list[float] = []
            sim_at_200 = float("nan")

            pbar = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                console=console,
            )
            task_id = pbar.add_task(f"[bold cyan]{itype}[/bold cyan]", total=int(steps))
            with pbar:
                for step_idx in range(int(steps)):
                    x, y = _synthetic_next_plus_one(
                        batch_size=int(batch_size),
                        seq_len=int(micro_seq_len),
                        vocab_size=int(micro_vocab_size),
                        generator=g,
                        device=device,
                    )

                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    opt.zero_grad(set_to_none=True)

                    out = model(x, y, train_mode=True) if int(synapses) == 1 else model(x, y)
                    if isinstance(out, tuple):
                        _, loss_t = out
                    else:
                        loss_t = out
                    if loss_t is None:
                        raise RuntimeError("Expected a loss tensor, got None")
                    loss_t.backward()
                    opt.step()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    dt = time.perf_counter() - t0

                    losses.append(float(loss_t.detach().to(torch.float32).cpu().item()))
                    tok_per_sec = (int(batch_size) * int(micro_seq_len)) / max(dt, 1e-9)
                    tok_rates.append(tok_per_sec)

                    if step_idx % int(log_every) == 0 or step_idx == int(steps) - 1:
                        sims: list[float] = []
                        norms: list[float] = []
                        for name, p in tracked:
                            cur = p.detach().to(torch.float32).cpu()
                            sims.append(_cosine_to_init(cur, init_snap[name]))
                            norms.append(float(cur.norm().item()))
                        sim_min = min(sims)
                        if step_idx == 200:
                            sim_at_200 = sim_min

                        row: dict[str, object] = {
                            "step": step_idx,
                            "loss": losses[-1],
                            "dt_ms": dt * 1000.0,
                            "tok_per_sec": tok_per_sec,
                            "sim_min": sim_min,
                            "w_norm_0": norms[0] if len(norms) > 0 else float("nan"),
                            "w_norm_1": norms[1] if len(norms) > 1 else float("nan"),
                        }

                        if int(spectrum_every) > 0 and (step_idx % int(spectrum_every) == 0 or step_idx == int(steps) - 1):
                            name0, p0 = tracked[0]
                            stats = _svd_topk_stats(p0.detach().to(torch.float32).cpu(), k=4)
                            for ki in svd_keys:
                                row[f"{name0}:{ki}"] = stats.get(ki, float("nan"))
                        else:
                            for ki in svd_keys:
                                row[f"{tracked[0][0]}:{ki}"] = ""

                        writer.writerow(row)
                        f.flush()

                    pbar.update(task_id, advance=1)

        # Plot loss curve
        loss_png = run_dir / f"{itype}_loss.png"
        plt.figure(figsize=(8, 4))
        plt.plot(losses)
        plt.title(f"Loss curve ({itype})")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.tight_layout()
        plt.savefig(loss_png)
        plt.close()

        res = _RunResult(
            init_type=str(itype),
            csv_path=str(csv_path),
            loss_png_path=str(loss_png),
            steps=int(steps),
            final_loss=float(losses[-1]) if losses else float("nan"),
            tok_per_sec=float(sum(tok_rates[-50:]) / max(len(tok_rates[-50:]), 1)),
            sim_at_200=float(sim_at_200),
        )
        results.append(res)

    # Combined plot
    if len(results) >= 2:
        plt.figure(figsize=(9, 4))
        for res in results:
            # Re-read loss from CSV (cheap; avoids storing all series in memory for long runs).
            steps_list: list[int] = []
            loss_list: list[float] = []
            with open(res.csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("loss") in (None, ""):
                        continue
                    steps_list.append(int(float(row["step"])))
                    loss_list.append(float(row["loss"]))
            plt.plot(steps_list, loss_list, label=res.init_type)
        plt.title("CA init micro-benchmark: loss comparison")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "loss_compare.png")
        plt.close()

    tbl = Table(title="CA init micro-benchmark summary")
    tbl.add_column("init_type")
    tbl.add_column("final_loss", justify="right")
    tbl.add_column("tok/s (avg last 50)", justify="right")
    tbl.add_column("sim_min@200", justify="right")
    tbl.add_column("csv")
    for r in results:
        tbl.add_row(
            r.init_type,
            f"{r.final_loss:.4f}",
            f"{r.tok_per_sec:,.0f}",
            f"{r.sim_at_200:.4f}" if r.sim_at_200 == r.sim_at_200 else "n/a",
            r.csv_path,
        )
    console.print(tbl)
    compute_cleanup()
    return results


@dataclass(frozen=True)
class _AttentionCase:
    name: str
    model_path: str
    sdpa_backend: str | None = None
    use_flex: bool = False


_ATTENTION_CASES = {
    "vanilla_sdpa_auto": _AttentionCase("vanilla_sdpa_auto", "vanilla", "auto"),
    "vanilla_sdpa_flash": _AttentionCase("vanilla_sdpa_flash", "vanilla", "flash"),
    "vanilla_sdpa_math": _AttentionCase("vanilla_sdpa_math", "vanilla", "math"),
    # The canonical synaptic path materializes its biological logit bias, so it cannot use
    # stock SDPA/FlashAttention without changing the function being measured.
    "synaptic_dense": _AttentionCase("synaptic_dense", "synaptic"),
    "synaptic_flex": _AttentionCase("synaptic_flex", "synaptic", use_flex=True),
}


def _parse_attention_cases(spec: str) -> list[_AttentionCase]:
    names = [name.strip() for name in spec.split(",") if name.strip()]
    if not names:
        raise ValueError("attention_cases must contain at least one case")
    unknown = sorted(set(names) - set(_ATTENTION_CASES))
    if unknown:
        allowed = ", ".join(sorted(_ATTENTION_CASES))
        raise ValueError(f"Unknown attention case(s) {unknown}; allowed: {allowed}")
    if len(names) != len(set(names)):
        raise ValueError("attention_cases must not contain duplicates")
    return [_ATTENTION_CASES[name] for name in names]


def _attention_preflight(
    *, cuda_available: bool, device_name: str | None, require_4090: bool
) -> str | None:
    if not cuda_available:
        return "CUDA is unavailable; attention performance evidence requires a CUDA GPU."
    if require_4090 and (device_name is None or "4090" not in device_name.casefold()):
        return (
            "This run requires an RTX 4090 so its results satisfy bead zsi; "
            f"detected {device_name or 'an unknown CUDA device'}."
        )
    return None


def _sdpa_context(backend: str | None):
    if backend in (None, "auto"):
        return nullcontext()
    if backend == "flash":
        # A single backend makes this a strict measurement: PyTorch must fail instead of silently
        # falling through to math or memory-efficient attention.
        return sdpa_kernel(SDPBackend.FLASH_ATTENTION)
    if backend == "math":
        return sdpa_kernel(SDPBackend.MATH)
    raise ValueError(f"Unsupported SDPA backend {backend!r}")


def _build_attention_model(
    case: _AttentionCase, *, device: torch.device, dtp: torch.dtype
) -> torch.nn.Module:
    torch.manual_seed(int(attention_seed))
    torch.cuda.manual_seed_all(int(attention_seed))
    if case.model_path == "synaptic":
        syn_cfg = SynapticConfig(
            enable_presyn=True,
            enable_hebbian=True,
            enable_metabolism=True,
            use_flex_attention=case.use_flex,
        )
        config = GPTSynapticConfig(
            sequence_len=int(seq_len),
            vocab_size=50257,
            n_layer=int(n_layer),
            n_head=int(n_head),
            n_kv_head=int(n_head),
            n_embd=int(n_embd),
            synapses=True,
            syn_cfg=syn_cfg,
        )
        model: torch.nn.Module = GPTSynaptic(config)
    else:
        config = GPTConfig(
            sequence_len=int(seq_len),
            vocab_size=50257,
            n_layer=int(n_layer),
            n_head=int(n_head),
            n_kv_head=int(n_head),
            n_embd=int(n_embd),
        )
        model = GPT(config)

    model.to(device=device, dtype=dtp)
    if isinstance(model, GPT):
        # GPT intentionally keeps RoPE caches in bf16 regardless of the activation dtype.
        model.cos = model.cos.to(dtype=torch.bfloat16)
        model.sin = model.sin.to(dtype=torch.bfloat16)
    model.train()
    if bool(attention_compile):
        model = cast(torch.nn.Module, torch.compile(model, dynamic=False))
    return model


def _attention_loss(
    model: torch.nn.Module,
    case: _AttentionCase,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    dtp: torch.dtype,
) -> torch.Tensor:
    with (
        _sdpa_context(case.sdpa_backend),
        torch.amp.autocast(device_type="cuda", dtype=dtp),
    ):
        output = model(tokens, targets)
    if case.model_path == "synaptic":
        _logits, loss = output
    else:
        loss = output
    if loss is None or not isinstance(loss, torch.Tensor):
        raise RuntimeError(f"{case.name} returned no loss tensor")
    return loss


def _benchmark_attention_case(
    case: _AttentionCase,
    *,
    device: torch.device,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    dtp: torch.dtype,
) -> dict[str, object]:
    row: dict[str, object] = {
        "case": case.name,
        "model_path": case.model_path,
        "sdpa_backend": case.sdpa_backend,
        "use_flex_attention": case.use_flex,
        "status": "failed",
        "reason": None,
        "latency_ms": None,
        "tokens_per_second": None,
        "peak_allocated_gib": None,
        "peak_reserved_gib": None,
    }
    model: torch.nn.Module | None = None
    try:
        console.print(f"[bold cyan]Benchmarking {case.name}[/bold cyan]")
        model = _build_attention_model(case, device=device, dtp=dtp)

        for _ in range(int(attention_warmup_steps)):
            model.zero_grad(set_to_none=True)
            _attention_loss(model, case, tokens, targets, dtp).backward()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        start = time.perf_counter()
        for _ in range(int(attention_benchmark_steps)):
            model.zero_grad(set_to_none=True)
            _attention_loss(model, case, tokens, targets, dtp).backward()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

        measured_steps = int(attention_benchmark_steps)
        measured_tokens = measured_steps * int(batch_size) * int(seq_len)
        row.update(
            status="passed",
            latency_ms=1000.0 * elapsed / measured_steps,
            tokens_per_second=measured_tokens / elapsed,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 1024**3,
        )
    except torch.OutOfMemoryError as exc:
        row.update(status="oom", reason=str(exc))
    except Exception as exc:  # noqa: BLE001 - one unsupported backend must not erase the matrix
        row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
    finally:
        del model
        torch.cuda.empty_cache()
    return row


def _recommend_backend(
    rows: list[dict[str, object]], *, model_path: str, baseline: str
) -> dict[str, object] | None:
    successful = [
        row
        for row in rows
        if row["model_path"] == model_path
        and row["status"] == "passed"
        and isinstance(row["tokens_per_second"], float)
    ]
    if not successful:
        return None
    best = max(successful, key=lambda row: float(row["tokens_per_second"]))
    baseline_row = next((row for row in successful if row["case"] == baseline), None)
    delta_pct = None
    if baseline_row is not None:
        baseline_rate = float(baseline_row["tokens_per_second"])
        delta_pct = 100.0 * (float(best["tokens_per_second"]) / baseline_rate - 1.0)
    return {
        "model_path": model_path,
        "recommended_case": best["case"],
        "baseline_case": baseline if baseline_row is not None else None,
        "throughput_delta_percent": delta_pct,
    }


def _format_metric(value: object, format_spec: str) -> str:
    if isinstance(value, (int, float)):
        return format(float(value), format_spec)
    return "—"


def _bench_flex() -> Path:
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    blocker = _attention_preflight(
        cuda_available=torch.cuda.is_available(),
        device_name=device_name,
        require_4090=bool(attention_require_4090),
    )
    if blocker is not None:
        raise RuntimeError(blocker)
    if int(attention_warmup_steps) < 1 or int(attention_benchmark_steps) < 1:
        raise ValueError("attention_warmup_steps and attention_benchmark_steps must be positive")

    cases = _parse_attention_cases(str(attention_cases))
    device = torch.device("cuda", 0)
    dtp = _parse_dtype(str(attention_dtype))
    if dtp not in (torch.float16, torch.bfloat16):
        raise ValueError("attention_dtype must be float16 or bfloat16 for CUDA benchmarking")
    torch.manual_seed(int(attention_seed))
    torch.cuda.manual_seed_all(int(attention_seed))
    tokens = torch.randint(0, 50257, (int(batch_size), int(seq_len)), device=device)
    targets = torch.randint(0, 50257, (int(batch_size), int(seq_len)), device=device)

    rows = [
        _benchmark_attention_case(
            case,
            device=device,
            tokens=tokens,
            targets=targets,
            dtp=dtp,
        )
        for case in cases
    ]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = Path(str(attention_out_dir)) / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": device_name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "batch_size": int(batch_size),
        "sequence_length": int(seq_len),
        "n_layer": int(n_layer),
        "n_head": int(n_head),
        "n_embd": int(n_embd),
        "warmup_steps": int(attention_warmup_steps),
        "benchmark_steps": int(attention_benchmark_steps),
        "compiled": bool(attention_compile),
        "dtype": str(attention_dtype),
        "seed": int(attention_seed),
        "flash_note": (
            "vanilla_sdpa_flash strictly forces PyTorch's FLASH_ATTENTION SDPA backend. "
            "PyTorch does not expose a FlashAttention-2/3 generation label here; do not infer "
            "one from this artifact. The synaptic dense path needs an additive biological score "
            "bias, so stock SDPA is not functionally equivalent; synaptic_flex is its fused case."
        ),
    }
    artifact = {
        "metadata": metadata,
        "results": rows,
        "recommendations": [
            recommendation
            for recommendation in (
                _recommend_backend(rows, model_path="vanilla", baseline="vanilla_sdpa_auto"),
                _recommend_backend(rows, model_path="synaptic", baseline="synaptic_dense"),
            )
            if recommendation is not None
        ],
    }
    json_path = run_dir / "attention_backends.json"
    with json_path.open("x", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")

    csv_path = run_dir / "attention_backends.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    table = Table(title=f"Attention backend benchmark — {device_name}")
    table.add_column("Case")
    table.add_column("Status")
    table.add_column("ms/step", justify="right")
    table.add_column("tokens/s", justify="right")
    table.add_column("peak GiB", justify="right")
    for row in rows:
        latency = row["latency_ms"]
        rate = row["tokens_per_second"]
        peak = row["peak_allocated_gib"]
        table.add_row(
            str(row["case"]),
            str(row["status"]),
            _format_metric(latency, ".2f"),
            _format_metric(rate, ",.0f"),
            _format_metric(peak, ".2f"),
        )
    console.print(table)
    console.print(f"[green]Wrote immutable benchmark artifacts to {run_dir}[/green]")
    return run_dir


if __name__ == "__main__":
    try:
        if mode == "ca_init":
            _bench_ca_init()
        else:
            _bench_flex()
    except (RuntimeError, ValueError) as exc:
        console.print(f"[bold red]Benchmark aborted:[/bold red] {exc}")
        raise SystemExit(2) from exc
    finally:
        # Ensure we don't leave process groups around in case compute_init detected DDP env vars.
        compute_cleanup()
