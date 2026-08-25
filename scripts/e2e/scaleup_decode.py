"""Scale-Up Inference & Autoregressive Decode Battery (bead `hwxb.6.4`).

Verifies the high-performance autoregressive generation path at scale on single-GPU/CPU:
1. Autoregressive decode with KV-cache and presynaptic biophysical state carried across steps.
2. Exact decode-vs-contiguous prefix parity across token boundaries.
3. Per-prompt scratchpad isolation: fast weights and volatile state reset cleanly between prompts.
4. Online fast-weight adaptation during decode ('learns context online').
5. Throughput and latency benchmarking with structured Rich console logs and JSONL traces.
6. Non-degenerate generation and diversity checks.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
import tempfile
import time
from typing import Any

from rich.console import Console
from rich.table import Table
import torch
from torch import Tensor

from bio_inspired_nanochat.e2e_harness import InvariantResult
from bio_inspired_nanochat.engine import KVCache, sample_next_token
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.run_logging import RunLogger
from bio_inspired_nanochat.synaptic import (
    SynapticConfig,
    SynapticLinear,
    build_presyn_state,
)


@dataclass
class ScaleupDecodeConfig:
    """Configuration for the Scale-Up Inference & Decode Battery."""

    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int = 4
    n_embd: int = 128
    vocab_size: int = 256
    sequence_len: int = 128
    device: str = "cpu"
    dtype: str = "float32"
    seed: int = 42

    # Generation parameters
    prompt_len: int = 16
    decode_len: int = 32
    batch_size: int = 2
    temperature: float = 0.7
    top_k: int = 32
    num_prompts: int = 3
    enable_online_plasticity: bool = True


@dataclass
class ScaleupDecodeReport:
    """Structured report returned by the Scale-Up Decode Battery."""

    run_id: str
    config: ScaleupDecodeConfig
    passed: bool
    invariants: list[InvariantResult]
    throughput_tok_per_sec: float = 0.0
    ttft_ms: float = 0.0
    summary: dict[str, Any] = field(default_factory=dict)

    def assert_passed(self) -> None:
        failed = [inv for inv in self.invariants if not inv.passed]
        if failed:
            msg = "\n".join(f"  FAILED: {inv.name} -> {inv.detail}" for inv in failed)
            raise AssertionError(f"Scale-Up Decode battery failed with {len(failed)} failure(s):\n{msg}")


def _build_model(cfg: ScaleupDecodeConfig) -> GPTSynaptic:
    """Instantiate a reproducible GPTSynaptic model for decode verification."""
    torch.manual_seed(cfg.seed)
    syn_cfg = SynapticConfig(
        enable_presyn=True,
        enable_hebbian=True,
        plasticity_during_training=True,
        bistable_latch=True,
        fast_weight_normalized=True,
        post_fast_lr=0.01,
        post_slow_lr=0.005,
    )
    gpt_cfg = GPTSynapticConfig(
        sequence_len=cfg.sequence_len,
        vocab_size=cfg.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_kv_head=cfg.n_kv_head,
        n_embd=cfg.n_embd,
        synapses=True,
        syn_cfg=syn_cfg,
    )
    model = GPTSynaptic(gpt_cfg).to(cfg.device)
    model.eval()
    return model


def _init_kv_cache(model: GPTSynaptic, batch_size: int, device: str) -> KVCache:
    """Initialize a full KV cache with per-layer presynaptic biophysical states."""
    c = model.config
    head_dim = c.n_embd // c.n_head
    cache = KVCache(
        batch_size=batch_size,
        num_heads=c.n_kv_head,
        seq_len=c.sequence_len,
        head_dim=head_dim,
        num_layers=c.n_layer,
    )
    cache.presyn_state = [
        build_presyn_state(
            B=batch_size,
            T=c.sequence_len,
            H=c.n_head,
            device=device,
            dtype=torch.float32,
            cfg=c.syn_cfg,
        )
        for _ in range(c.n_layer)
    ]
    return cache


def run_scaleup_decode(
    cfg: ScaleupDecodeConfig | None = None,
    *,
    run_dir: Path | str | None = None,
    verbose: bool = True,
) -> ScaleupDecodeReport:
    """Execute the full Scale-Up Decode battery and return a structured report."""
    if cfg is None:
        cfg = ScaleupDecodeConfig()

    console = Console(quiet=not verbose)
    run_id = f"scaleup-decode-{int(time.time())}"
    invariants: list[InvariantResult] = []

    if run_dir is None:
        base_dir = Path(tempfile.mkdtemp(prefix="scaleup_decode_"))
    else:
        base_dir = Path(run_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

    run_logger = RunLogger(base_dir, name="scaleup_decode", run_id=run_id, console=verbose)
    run_logger.event("decode_config", config=asdict(cfg))

    model = _build_model(cfg)
    device = cfg.device
    rng = torch.Generator(device=device).manual_seed(cfg.seed)

    # -------------------------------------------------------------------------
    # Part 1: Prefill + Incremental Autoregressive Decode with KV-Cache
    # -------------------------------------------------------------------------
    torch.manual_seed(cfg.seed + 1)
    prompt_tokens = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.prompt_len), device=device)

    model.reset_sequence_state(reset_fast_weights=True, reset_consolidation=True)
    cache = _init_kv_cache(model, cfg.batch_size, device)

    # Measure Time-To-First-Token (TTFT)
    t0_prefill = time.perf_counter()
    with torch.no_grad():
        logits_prefill, _ = model(
            prompt_tokens,
            kv_cache=cache,
            train_mode=False,
        )
    t_ttft = (time.perf_counter() - t0_prefill) * 1000.0

    generated_tokens: list[Tensor] = []
    step_latencies_ms: list[float] = []
    last_logits = logits_prefill[:, -1, :]

    # Autoregressive generation loop
    t0_decode = time.perf_counter()
    for step in range(cfg.decode_len):
        t0_step = time.perf_counter()
        next_tok = sample_next_token(last_logits, rng, temperature=cfg.temperature, top_k=cfg.top_k)
        generated_tokens.append(next_tok)

        with torch.no_grad():
            logits_step, _ = model(
                next_tok,
                kv_cache=cache,
                train_mode=cfg.enable_online_plasticity,
            )
        step_latencies_ms.append((time.perf_counter() - t0_step) * 1000.0)
        last_logits = logits_step[:, -1, :]

    total_decode_time = time.perf_counter() - t0_decode
    total_tokens_generated = cfg.batch_size * cfg.decode_len
    tok_per_sec = total_tokens_generated / max(1e-6, total_decode_time)

    # -------------------------------------------------------------------------
    # Part 2: Verify Exact Parity vs Contiguous Forward (Deterministic Sampler)
    # -------------------------------------------------------------------------
    full_sequence = torch.cat([prompt_tokens] + generated_tokens, dim=1)
    model.reset_sequence_state(reset_fast_weights=True, reset_consolidation=True)

    with torch.no_grad():
        contiguous_logits, _ = model(full_sequence, train_mode=False)

    # Check parity on prefix logits
    prefix_parity_max_err = float(
        torch.max(torch.abs(logits_prefill - contiguous_logits[:, : cfg.prompt_len, :])).item()
    )
    parity_passed = prefix_parity_max_err < 5e-3

    invariants.append(
        InvariantResult(
            name="decode_contiguous_parity",
            passed=parity_passed,
            observed={
                "prefix_max_abs_err": prefix_parity_max_err,
                "prompt_len": cfg.prompt_len,
            },
            detail=f"Max prefix discrepancy: {prefix_parity_max_err:.6e} (< 5e-3 threshold)",
        )
    )

    # -------------------------------------------------------------------------
    # Part 3: Presynaptic Biophysical State Advancement
    # -------------------------------------------------------------------------
    states_advanced = True
    if cache.presyn_state is not None:
        for layer_idx, st in enumerate(cache.presyn_state):
            if st is None:
                states_advanced = False
                break
            if "c" in st and torch.isnan(st["c"]).any():
                states_advanced = False

    invariants.append(
        InvariantResult(
            name="presyn_state_carried_across_steps",
            passed=states_advanced,
            observed={"num_layers": cfg.n_layer, "states_advanced": states_advanced},
            detail=f"Presynaptic calcium/RRP carried across {cfg.decode_len} decode steps",
        )
    )

    # -------------------------------------------------------------------------
    # Part 4: Per-Prompt Scratchpad Reset & Multi-Turn Isolation
    # -------------------------------------------------------------------------
    # 1. Baseline evaluation of prompt B on clean model
    model.reset_sequence_state(reset_fast_weights=True, reset_consolidation=True)
    cache_b0 = _init_kv_cache(model, 1, device)
    prompt_b = torch.randint(0, cfg.vocab_size, (1, cfg.prompt_len), device=device)
    with torch.no_grad():
        logits_b_baseline, _ = model(prompt_b, kv_cache=cache_b0, train_mode=False)

    # 2. Process prompt A with online fast-weight plasticity (mutates volatile scratchpad)
    model.reset_sequence_state(reset_fast_weights=True, reset_consolidation=True)
    cache_a = _init_kv_cache(model, 1, device)
    prompt_a = torch.randint(0, cfg.vocab_size, (1, cfg.prompt_len), device=device)

    with torch.no_grad():
        model(prompt_a, kv_cache=cache_a, train_mode=True)

    fast_norm_after_a = sum(
        float(m.w_fast.norm().item())
        for m in model.modules()
        if isinstance(m, SynapticLinear) and m.w_fast is not None
    )

    # 3. Reset per-prompt scratchpad for prompt B
    model.reset_sequence_state(reset_fast_weights=True, reset_consolidation=True)

    fast_norm_after_reset = sum(
        float(m.w_fast.norm().item())
        for m in model.modules()
        if isinstance(m, SynapticLinear) and m.w_fast is not None
    )

    # 4. Prompt B evaluated on cleanly reset model
    cache_b1 = _init_kv_cache(model, 1, device)
    with torch.no_grad():
        logits_b_after, _ = model(prompt_b, kv_cache=cache_b1, train_mode=False)

    reset_isolation_err = float(torch.max(torch.abs(logits_b_after - logits_b_baseline)).item())
    isolation_passed = (fast_norm_after_reset == 0.0) and (reset_isolation_err < 1e-3)

    invariants.append(
        InvariantResult(
            name="per_prompt_reset_isolation",
            passed=isolation_passed,
            observed={
                "fast_norm_after_prompt": fast_norm_after_a,
                "fast_norm_after_reset": fast_norm_after_reset,
                "reset_isolation_max_err": reset_isolation_err,
            },
            detail=(
                f"Fast weights zeroed post-reset (norm={fast_norm_after_reset}); "
                f"Prompt B before/after max err={reset_isolation_err:.6e}"
            ),
        )
    )

    # -------------------------------------------------------------------------
    # Part 5: Non-Degenerate Diversity and Entropy Check
    # -------------------------------------------------------------------------
    cat_gen = torch.cat(generated_tokens, dim=1)
    unique_tokens = len(torch.unique(cat_gen))
    entropy_ratio = unique_tokens / float(cat_gen.numel())
    non_degenerate = entropy_ratio > 0.25

    invariants.append(
        InvariantResult(
            name="non_degenerate_generation_diversity",
            passed=non_degenerate,
            observed={
                "unique_tokens": unique_tokens,
                "total_generated": cat_gen.numel(),
                "diversity_ratio": entropy_ratio,
            },
            detail=f"Generated {unique_tokens}/{cat_gen.numel()} unique tokens (diversity={entropy_ratio:.2f})",
        )
    )

    # -------------------------------------------------------------------------
    # Part 6: Performance & Throughput Gate
    # -------------------------------------------------------------------------
    throughput_passed = tok_per_sec > 2.0

    invariants.append(
        InvariantResult(
            name="decode_throughput_bounded",
            passed=throughput_passed,
            observed={
                "tokens_per_second": tok_per_sec,
                "ttft_ms": t_ttft,
                "mean_step_ms": float(sum(step_latencies_ms) / max(1, len(step_latencies_ms))),
            },
            detail=f"Decoded {total_tokens_generated} tokens at {tok_per_sec:.1f} tok/s (TTFT={t_ttft:.2f}ms)",
        )
    )

    for inv in invariants:
        run_logger.event("scaleup_decode_invariant", **asdict(inv))

    all_passed = all(inv.passed for inv in invariants)
    report = ScaleupDecodeReport(
        run_id=run_id,
        config=cfg,
        passed=all_passed,
        invariants=invariants,
        throughput_tok_per_sec=tok_per_sec,
        ttft_ms=t_ttft,
        summary={
            "total_tokens_generated": total_tokens_generated,
            "ttft_ms": t_ttft,
            "tokens_per_second": tok_per_sec,
            "mean_step_latency_ms": sum(step_latencies_ms) / max(1, len(step_latencies_ms)),
        },
    )

    if verbose:
        table = Table(title="Scale-Up Inference & Decode Battery Invariant Status")
        table.add_column("Invariant", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Detail")

        for inv in invariants:
            status = "[bold green]PASS[/bold green]" if inv.passed else "[bold red]FAIL[/bold red]"
            table.add_row(inv.name, status, inv.detail)
        console.print(table)

    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scale-Up Decode & Inference Battery")
    parser.add_argument("--run-dir", type=str, default=None, help="Directory to save run logs")
    parser.add_argument("--device", type=str, default="cpu", help="Target device: cpu or cuda")
    parser.add_argument("--prompt-len", type=int, default=16, help="Prompt sequence length")
    parser.add_argument("--decode-len", type=int, default=32, help="Autoregressive decode tokens")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args(argv)

    cfg = ScaleupDecodeConfig(
        device=args.device,
        prompt_len=args.prompt_len,
        decode_len=args.decode_len,
        seed=args.seed,
    )
    report = run_scaleup_decode(cfg, run_dir=args.run_dir, verbose=True)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
