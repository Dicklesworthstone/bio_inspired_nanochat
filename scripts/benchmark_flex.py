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
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from bio_inspired_nanochat.common import autodetect_device_type, compute_cleanup, compute_init
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
with open(os.path.join("bio_inspired_nanochat", "configurator.py")) as f:
    exec(f.read())  # nosec B102 # CLI overrides


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
    ddp, ddp_rank, _, _, device = compute_init(dty)
    if ddp:
        raise RuntimeError("ca_init benchmark is intended for single-process runs (no torchrun/DDP).")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    run_dir = Path(out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
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


def _bench_flex() -> None:
    def benchmark(use_flex: bool) -> tuple[float, float]:
        print(f"\n--- Benchmarking with use_flex_attention={use_flex} ---")

        # Reset memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Config
        syn_cfg = SynapticConfig(
            enable_presyn=True,
            enable_hebbian=True,
            enable_metabolism=True,
            use_flex_attention=use_flex,
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

        device = torch.device("cuda")
        # Use float16 to avoid Triton atomic_add bf16 issues
        dtp = torch.float16

        print("Initializing model...")
        # Avoid meta device for benchmark to prevent buffer init issues
        model = GPTSynaptic(config).to(device).to(dtp)
        model.train()

        # Compile is REQUIRED for FlexAttention
        print("Compiling model...")
        compiled_model = torch.compile(model)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Dummy Data
        x = torch.randint(0, 50257, (int(batch_size), int(seq_len)), device=device)
        y = torch.randint(0, 50257, (int(batch_size), int(seq_len)), device=device)

        # Warmup
        print("Warmup steps...")
        for _ in range(5):
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=dtp):
                _, loss = compiled_model(x, y)
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()

        # Benchmark
        print("Benchmarking steps...")
        t0 = time.time()
        bench_steps = 20
        for _ in range(bench_steps):
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=dtp):
                _, loss = compiled_model(x, y)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()

        dt = t1 - t0
        tokens_per_sec = (bench_steps * int(batch_size) * int(seq_len)) / dt
        max_mem = torch.cuda.max_memory_allocated() / 1024**3

        print(f"Time: {dt:.4f}s")
        print(f"Throughput: {tokens_per_sec:.2f} tokens/sec")
        print(f"Peak VRAM: {max_mem:.2f} GB")

        return tokens_per_sec, max_mem

    # Standard Baseline
    try:
        perf_std, mem_std = benchmark(use_flex=False)
        res_std = f"{perf_std:.2f} t/s | {mem_std:.2f} GB"
    except torch.OutOfMemoryError:
        print("Standard: OOM")
        res_std = "OOM"
        perf_std = 0
        mem_std = float("inf")
    except Exception as e:  # noqa: BLE001
        print(f"Standard Failed: {e}")
        res_std = "Failed"
        perf_std = 0
        mem_std = float("inf")

    # Flex Attention
    try:
        perf_flex, mem_flex = benchmark(use_flex=True)
        res_flex = f"{perf_flex:.2f} t/s | {mem_flex:.2f} GB"
    except Exception as e:  # noqa: BLE001
        print(f"Flex Failed: {e}")
        import traceback

        traceback.print_exc()
        res_flex = "Failed"
        perf_flex = 0
        mem_flex = 0

    print("\n=== Results Summary ===")
    print(f"Standard: {res_std}")
    print(f"Flex:     {res_flex}")
    if perf_std > 0:
        print(f"Speedup:  {perf_flex / perf_std:.2f}x")
        print(f"Mem Red:  {mem_flex / mem_std:.2f}x (Lower is better)")
    else:
        print("Speedup:  Infinite (Standard OOM)")


if __name__ == "__main__":
    try:
        if mode == "ca_init":
            _bench_ca_init()
        else:
            _bench_flex()
    finally:
        # Ensure we don't leave process groups around in case compute_init detected DDP env vars.
        compute_cleanup()
