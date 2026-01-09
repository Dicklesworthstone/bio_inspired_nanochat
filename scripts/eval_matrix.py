"""
Standardized evaluation harness (bio vs vanilla).

Run:
  python -m scripts.eval_matrix run --preset vanilla --train-tokens 1000000 --seed 1337 --data synthetic
  python -m scripts.eval_matrix matrix --presets vanilla,bio_all --train-tokens 1000000 --seeds 1337,1338 --data synthetic

Design reference:
  docs/eval_benchmark_matrix.md
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal, Optional, cast

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from bio_inspired_nanochat.common import autodetect_device_type, compute_cleanup, compute_init
from bio_inspired_nanochat.dataloader import tokenizing_distributed_data_loader
from bio_inspired_nanochat.loss_eval import evaluate_bpb
from bio_inspired_nanochat.torch_imports import F, Tensor, torch
from bio_inspired_nanochat.tokenizer import get_token_bytes, get_tokenizer

from bio_inspired_nanochat.gpt import GPT, GPTConfig
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.synaptic import SynapticConfig

from scripts.base_eval import evaluate_model

console = Console()

PresetId = Literal[
    "vanilla",
    "bio_all",
    "bio_no_presyn",
    "bio_no_hebbian",
    "bio_no_metabolism",
    "bio_no_stochastic_release",
    "bio_no_doc2",
    "bio_no_bdnf",
    "bio_no_septin_barrier",
]

DEFAULT_ABLATION_PRESETS: tuple[PresetId, ...] = (
    "vanilla",
    "bio_all",
    "bio_no_presyn",
    "bio_no_hebbian",
    "bio_no_metabolism",
    "bio_no_stochastic_release",
    "bio_no_doc2",
    "bio_no_bdnf",
    "bio_no_septin_barrier",
)

SUMMARY_FIELDS: tuple[str, ...] = (
    "status",
    "error",
    "run_id",
    "run_dir",
    "preset",
    "seed",
    "data",
    "device_type",
    "init_type",
    "sequence_len",
    "vocab_size",
    "n_layer",
    "n_head",
    "n_embd",
    "use_moe",
    "num_experts",
    "moe_top_k",
    "device_batch_size",
    "total_batch_size_tokens",
    "grad_accum_steps",
    "train_tokens_requested",
    "train_tokens_processed",
    "steps",
    "eval_tokens",
    "eval_steps",
    "eval_bpb",
    "core_eval",
    "core_max_per_task",
    "ece_bins",
    "walltime_sec",
    "tok_per_sec",
    "train_loss_final",
    "val_loss",
    "val_ppl",
    "val_bpb",
    "core_metric",
    "ece",
    "niah_acc",
)


@dataclass(frozen=True)
class HarnessRunSummary:
    run_id: str
    preset: str
    seed: int
    train_tokens_requested: int
    train_tokens_processed: int
    walltime_sec: float
    tok_per_sec: float
    train_loss_final: float
    val_loss: float
    val_ppl: float
    val_bpb: Optional[float]
    core_metric: Optional[float]
    ece: Optional[float]
    niah_acc: Optional[float]


def _parse_int_list(csv_str: str) -> list[int]:
    out: list[int] = []
    for part in csv_str.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    if not out:
        raise ValueError("Expected a non-empty comma-separated list of ints")
    return out


def _parse_str_list(csv_str: str) -> list[str]:
    out: list[str] = []
    for part in csv_str.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(p)
    if not out:
        raise ValueError("Expected a non-empty comma-separated list of strings")
    return out


def _set_seed(seed: int, *, device_type: str) -> torch.Generator:
    torch.manual_seed(seed)
    if device_type == "cuda":
        torch.cuda.manual_seed_all(seed)
    g = torch.Generator(device=device_type)
    g.manual_seed(seed)
    return g


def _synthetic_loader(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> Iterator[tuple[Tensor, Tensor]]:
    while True:
        start = torch.randint(0, vocab_size, (batch_size, 1), generator=generator, device=device)
        ar = torch.arange(seq_len + 1, device=device).view(1, -1)
        toks = (start + ar) % vocab_size
        yield toks[:, :-1].to(torch.long), toks[:, 1:].to(torch.long)


def _get_logits(model: Any, idx: Tensor) -> Tensor:
    out = model(idx)
    if isinstance(out, tuple):
        logits = out[0]
    elif hasattr(out, "logits"):
        logits = out.logits
    else:
        logits = out
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"Expected logits Tensor, got {type(logits)}")
    return logits


def _val_loss_ppl_ece(
    model: Any,
    batches: Iterator[tuple[Tensor, Tensor]],
    *,
    steps: int,
    device_type: str,
    ddp: bool,
    ece_bins: int = 15,
) -> tuple[float, float, Optional[float]]:
    model.eval()
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )
    # ECE accumulator: bins on max prob of predicted token.
    conf_sum = torch.zeros(ece_bins, dtype=torch.float64, device="cpu")
    acc_sum = torch.zeros(ece_bins, dtype=torch.float64, device="cpu")
    count = torch.zeros(ece_bins, dtype=torch.float64, device="cpu")

    losses: list[Tensor] = []
    for _ in range(steps):
        x, y = next(batches)
        with torch.no_grad(), autocast_ctx:
            logits = _get_logits(model, x).to(torch.float32)
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = y.reshape(-1)
            loss_flat = F.cross_entropy(
                logits_flat,
                targets_flat,
                reduction="none",
                ignore_index=-1,
            )
            valid = targets_flat >= 0
            if valid.any():
                losses.append(loss_flat[valid].mean())
            else:
                losses.append(torch.tensor(float("nan"), device=logits.device))

            # ECE (optional): use max prob and correctness per token.
            probs = torch.softmax(logits_flat, dim=-1)
            conf, pred = probs.max(dim=-1)
            correct = (pred == targets_flat) & valid
            # Bin by confidence in [0,1]
            bins = torch.clamp((conf * ece_bins).to(torch.int64), 0, ece_bins - 1)
            for b in range(ece_bins):
                mask = (bins == b) & valid
                if mask.any():
                    conf_sum[b] += float(conf[mask].sum().item())
                    acc_sum[b] += float(correct[mask].float().sum().item())
                    count[b] += float(mask.sum().item())

    val_loss = torch.stack(losses).mean()
    if ddp and torch.distributed.is_initialized():
        torch.distributed.all_reduce(val_loss, op=torch.distributed.ReduceOp.AVG)
    val_loss_f = float(val_loss.item())
    val_ppl = float(math.exp(val_loss_f)) if math.isfinite(val_loss_f) else float("inf")

    ece: Optional[float]
    if float(count.sum().item()) == 0.0:
        ece = None
    else:
        conf_mean = conf_sum / count.clamp_min(1.0)
        acc_mean = acc_sum / count.clamp_min(1.0)
        weights = count / count.sum()
        ece = float((weights * (conf_mean - acc_mean).abs()).sum().item())
    return val_loss_f, val_ppl, ece


def _apply_syn_preset(preset: str, syn_cfg: SynapticConfig) -> None:
    if preset == "bio_all":
        return
    if preset == "bio_no_presyn":
        syn_cfg.enable_presyn = False
        return
    if preset == "bio_no_hebbian":
        syn_cfg.enable_hebbian = False
        return
    if preset == "bio_no_metabolism":
        syn_cfg.enable_metabolism = False
        return
    if preset == "bio_no_stochastic_release":
        syn_cfg.stochastic_train_frac = 0.0
        return
    if preset == "bio_no_doc2":
        syn_cfg.doc2_gain = 0.0
        return
    if preset == "bio_no_bdnf":
        syn_cfg.bdnf_scale = 0.0
        return
    if preset == "bio_no_septin_barrier":
        syn_cfg.barrier_strength = 0.0
        return
    raise ValueError(f"Unknown synaptic preset: {preset!r}")


def _build_model(
    *,
    preset: PresetId,
    seed: int,
    device: torch.device,
    sequence_len: int,
    vocab_size: int,
    n_layer: int,
    n_head: int,
    n_embd: int,
    init_type: str,
    use_moe: bool,
    num_experts: int,
    moe_top_k: int,
) -> Any:
    if preset == "vanilla":
        with torch.device("meta"):
            cfg = GPTConfig(
                sequence_len=sequence_len,
                vocab_size=vocab_size,
                n_layer=n_layer,
                n_head=n_head,
                n_kv_head=n_head,
                n_embd=n_embd,
                init_type=init_type,
                init_seed=seed,
            )
            model = GPT(cfg)
        model.to_empty(device=device)
        model.init_weights()
        return model

    syn_cfg = SynapticConfig()
    _apply_syn_preset(preset, syn_cfg)
    with torch.device("meta"):
        cfg = GPTSynapticConfig(
            sequence_len=sequence_len,
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_kv_head=n_head,
            n_embd=n_embd,
            synapses=True,
            syn_cfg=syn_cfg,
            use_moe=bool(use_moe),
            num_experts=int(num_experts),
            moe_top_k=int(moe_top_k),
            init_type=init_type,
            init_seed=seed,
        )
        model = GPTSynaptic(cfg)
    model.to_empty(device=device)
    model.init_weights()
    return model


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _append_csv(path: Path, *, fieldnames: tuple[str, ...], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)


def _normalize_row_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in SUMMARY_FIELDS:
        v = row.get(k)
        out[k] = "" if v is None else v
    return out


def _write_summary(out_dir: Path, row: dict[str, Any]) -> None:
    csv_row = _normalize_row_for_csv(row)
    _append_csv(out_dir / "summary.csv", fieldnames=SUMMARY_FIELDS, row=csv_row)
    _write_jsonl(out_dir / "summary.jsonl", row)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _error_row(
    *,
    run_id: str,
    preset: str,
    seed: int,
    data: str,
    device_type: str,
    init_type: str,
    sequence_len: int,
    vocab_size: int,
    n_layer: int,
    n_head: int,
    n_embd: int,
    use_moe: bool,
    num_experts: int,
    moe_top_k: int,
    device_batch_size: int,
    total_batch_size_tokens: int,
    train_tokens_requested: int,
    eval_tokens: int,
    eval_bpb: bool,
    core_eval: bool,
    core_max_per_task: int,
    ece_bins: int,
    error: str,
) -> dict[str, Any]:
    tokens_per_micro = device_batch_size * sequence_len
    grad_accum_steps = (
        total_batch_size_tokens // tokens_per_micro if tokens_per_micro > 0 else None
    )
    steps = (
        int(math.ceil(train_tokens_requested / total_batch_size_tokens))
        if total_batch_size_tokens > 0
        else None
    )
    eval_steps = (
        int(eval_tokens // tokens_per_micro) if tokens_per_micro > 0 else None
    )

    return {
        "status": "error",
        "error": error,
        "run_id": run_id,
        "run_dir": "",
        "preset": preset,
        "seed": seed,
        "data": data,
        "device_type": device_type,
        "init_type": init_type,
        "sequence_len": sequence_len,
        "vocab_size": vocab_size,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "use_moe": use_moe,
        "num_experts": num_experts,
        "moe_top_k": moe_top_k,
        "device_batch_size": device_batch_size,
        "total_batch_size_tokens": total_batch_size_tokens,
        "grad_accum_steps": grad_accum_steps,
        "train_tokens_requested": train_tokens_requested,
        "train_tokens_processed": 0,
        "steps": steps,
        "eval_tokens": eval_tokens,
        "eval_steps": eval_steps,
        "eval_bpb": eval_bpb,
        "core_eval": core_eval,
        "core_max_per_task": core_max_per_task,
        "ece_bins": ece_bins,
        "walltime_sec": None,
        "tok_per_sec": None,
        "train_loss_final": None,
        "val_loss": None,
        "val_ppl": None,
        "val_bpb": None,
        "core_metric": None,
        "ece": None,
        "niah_acc": None,
    }


def _run_one(
    *,
    preset: PresetId,
    train_tokens: int,
    seed: int,
    device_type: str,
    data: str,
    out_dir: Path,
    # model arch
    sequence_len: int,
    vocab_size: int,
    n_layer: int,
    n_head: int,
    n_embd: int,
    # optim / batch
    device_batch_size: int,
    total_batch_size_tokens: int,
    embedding_lr: float,
    unembedding_lr: float,
    matrix_lr: float,
    weight_decay: float,
    # eval
    eval_tokens: int,
    eval_bpb: bool,
    core_eval: bool,
    core_max_per_task: int,
    ece_bins: int,
    init_type: str,
    use_moe: bool,
    num_experts: int,
    moe_top_k: int,
) -> HarnessRunSummary:
    ddp, ddp_rank, _, ddp_world_size, device = compute_init(device_type)
    if ddp:
        raise RuntimeError("eval_matrix harness currently supports single-process runs (no torchrun).")

    g = _set_seed(seed, device_type=device_type)
    run_id = f"Q-{preset}-t{train_tokens}-s{seed}-{_utc_stamp()}"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Data loaders
    if data == "synthetic":
        train_iter = _synthetic_loader(
            batch_size=device_batch_size,
            seq_len=sequence_len,
            vocab_size=vocab_size,
            device=device,
            generator=g,
        )
        val_iter = _synthetic_loader(
            batch_size=device_batch_size,
            seq_len=sequence_len,
            vocab_size=vocab_size,
            device=device,
            generator=g,
        )
        tokenizer = None
    elif data == "fineweb":
        train_iter = iter(
            tokenizing_distributed_data_loader(
                device_batch_size,
                sequence_len,
                split="train",
                device=device,
            )
        )
        val_iter = iter(
            tokenizing_distributed_data_loader(
                device_batch_size,
                sequence_len,
                split="val",
                device=device,
            )
        )
        tokenizer = get_tokenizer()
    else:
        raise ValueError(f"Unknown data source {data!r} (expected 'synthetic' or 'fineweb')")

    # Model
    model = _build_model(
        preset=preset,
        seed=seed,
        device=device,
        sequence_len=sequence_len,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        init_type=init_type,
        use_moe=use_moe,
        num_experts=num_experts,
        moe_top_k=moe_top_k,
    )
    model.train()

    # Batch math
    tokens_per_micro = device_batch_size * sequence_len
    if total_batch_size_tokens % tokens_per_micro != 0:
        raise ValueError(
            f"total_batch_size_tokens={total_batch_size_tokens} must be divisible by "
            f"device_batch_size*sequence_len={tokens_per_micro}"
        )
    grad_accum_steps = total_batch_size_tokens // tokens_per_micro
    steps = max(1, int(math.ceil(train_tokens / total_batch_size_tokens)))

    # Optimizers
    optimizers = model.setup_optimizers(
        unembedding_lr=unembedding_lr,
        embedding_lr=embedding_lr,
        matrix_lr=matrix_lr,
        weight_decay=weight_decay,
    )

    # Write config snapshot
    _write_jsonl(
        run_dir / "run_config.jsonl",
        {
            "run_id": run_id,
            "preset": preset,
            "seed": seed,
            "data": data,
            "train_tokens_requested": train_tokens,
            "sequence_len": sequence_len,
            "vocab_size": vocab_size,
            "n_layer": n_layer,
            "n_head": n_head,
            "n_embd": n_embd,
            "device_batch_size": device_batch_size,
            "total_batch_size_tokens": total_batch_size_tokens,
            "grad_accum_steps": grad_accum_steps,
            "steps": steps,
            "init_type": init_type,
            "use_moe": use_moe,
            "num_experts": num_experts,
            "moe_top_k": moe_top_k,
        },
    )

    # Training loop
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}[/bold]"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )
    train_task = progress.add_task(f"train {run_id}", total=steps)
    losses: list[float] = []
    t_start = time.perf_counter()
    supports_train_mode = "train_mode" in inspect.signature(model.forward).parameters
    with progress:
        for step in range(steps):
            t0 = time.perf_counter()
            for _ in range(grad_accum_steps):
                x, y = next(train_iter)
                result = model(x, y, train_mode=True) if supports_train_mode else model(x, y)
                if isinstance(result, tuple):
                    _, loss = result
                else:
                    loss = result
                if loss is None:
                    raise RuntimeError("Model returned loss=None during training")
                (loss / grad_accum_steps).backward()

            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)
            t1 = time.perf_counter()

            dt = t1 - t0
            tok_per_sec = total_batch_size_tokens / max(dt, 1e-12)
            loss_f = float(loss.detach().float().item())
            losses.append(loss_f)

            _write_jsonl(
                run_dir / "train_metrics.jsonl",
                {
                    "step": step,
                    "loss": loss_f,
                    "dt_sec": dt,
                    "tok_per_sec": tok_per_sec,
                },
            )
            progress.update(train_task, advance=1)

    walltime_sec = time.perf_counter() - t_start
    tokens_processed = steps * total_batch_size_tokens
    tok_per_sec_avg = tokens_processed / max(walltime_sec, 1e-12)
    train_loss_final = float(losses[-1]) if losses else float("nan")

    # Evaluation
    eval_steps = max(1, int(eval_tokens // (device_batch_size * sequence_len)))
    val_loss, val_ppl, ece = _val_loss_ppl_ece(
        model, val_iter, steps=eval_steps, device_type=device_type, ddp=ddp, ece_bins=ece_bins
    )

    val_bpb: Optional[float] = None
    if eval_bpb:
        if tokenizer is None:
            val_bpb = None
        else:
            token_bytes = get_token_bytes(device=device)
            if "loss_reduction" in inspect.signature(model.forward).parameters:
                val_bpb = float(evaluate_bpb(model, val_iter, eval_steps, token_bytes))
            else:
                orig_forward = model.forward

                def _syn_forward_wrapper(
                    idx: Tensor,
                    targets: Optional[Tensor] = None,
                    kv_cache=None,
                    loss_reduction: str = "mean",
                    **kwargs: Any,
                ):
                    if targets is None:
                        logits, _ = orig_forward(idx, None, kv_cache, train_mode=False)
                        return logits
                    logits, loss = orig_forward(idx, targets, kv_cache, train_mode=False)
                    if loss_reduction == "none":
                        logits_flat = logits.reshape(-1, logits.size(-1))
                        targets_flat = targets.reshape(-1)
                        loss_per_token = F.cross_entropy(
                            logits_flat,
                            targets_flat,
                            reduction="none",
                            ignore_index=-1,
                        )
                        return loss_per_token.reshape(targets.shape)
                    return loss

                model.forward = _syn_forward_wrapper
                try:
                    val_bpb = float(evaluate_bpb(model, val_iter, eval_steps, token_bytes))
                finally:
                    model.forward = orig_forward

    core_metric: Optional[float] = None
    if core_eval:
        if tokenizer is None:
            core_metric = None
        else:
            out = evaluate_model(model, tokenizer, device, max_per_task=core_max_per_task)
            core_metric = float(out["core_metric"])

    summary = HarnessRunSummary(
        run_id=run_id,
        preset=preset,
        seed=seed,
        train_tokens_requested=train_tokens,
        train_tokens_processed=tokens_processed,
        walltime_sec=walltime_sec,
        tok_per_sec=tok_per_sec_avg,
        train_loss_final=train_loss_final,
        val_loss=val_loss,
        val_ppl=val_ppl,
        val_bpb=val_bpb,
        core_metric=core_metric,
        ece=ece,
        niah_acc=None,
    )

    # Persist summary
    row: dict[str, Any] = {
        "status": "ok",
        "error": "",
        "run_id": summary.run_id,
        "run_dir": str(run_dir),
        "preset": summary.preset,
        "seed": summary.seed,
        "data": data,
        "device_type": device_type,
        "init_type": init_type,
        "sequence_len": sequence_len,
        "vocab_size": vocab_size,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "use_moe": use_moe,
        "num_experts": num_experts,
        "moe_top_k": moe_top_k,
        "device_batch_size": device_batch_size,
        "total_batch_size_tokens": total_batch_size_tokens,
        "grad_accum_steps": grad_accum_steps,
        "train_tokens_requested": train_tokens,
        "train_tokens_processed": tokens_processed,
        "steps": steps,
        "eval_tokens": eval_tokens,
        "eval_steps": eval_steps,
        "eval_bpb": eval_bpb,
        "core_eval": core_eval,
        "core_max_per_task": core_max_per_task,
        "ece_bins": ece_bins,
        "walltime_sec": summary.walltime_sec,
        "tok_per_sec": summary.tok_per_sec,
        "train_loss_final": summary.train_loss_final,
        "val_loss": summary.val_loss,
        "val_ppl": summary.val_ppl,
        "val_bpb": summary.val_bpb,
        "core_metric": summary.core_metric,
        "ece": summary.ece,
        "niah_acc": summary.niah_acc,
    }
    _write_summary(out_dir, row)

    # Pretty print
    if ddp_rank == 0:
        tbl = Table(title=f"Eval Matrix Summary: {run_id}")
        tbl.add_column("key")
        tbl.add_column("value", justify="right")
        for k, v in row.items():
            tbl.add_row(k, "" if v is None else str(v))
        console.print(tbl)

    compute_cleanup()
    return summary


def _cmd_run(args: argparse.Namespace) -> int:
    _run_one(
        preset=cast(PresetId, args.preset),
        train_tokens=args.train_tokens,
        seed=args.seed,
        device_type=args.device_type,
        data=args.data,
        out_dir=Path(args.out_dir),
        sequence_len=args.sequence_len,
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        device_batch_size=args.device_batch_size,
        total_batch_size_tokens=args.total_batch_size_tokens,
        embedding_lr=args.embedding_lr,
        unembedding_lr=args.unembedding_lr,
        matrix_lr=args.matrix_lr,
        weight_decay=args.weight_decay,
        eval_tokens=args.eval_tokens,
        eval_bpb=args.eval_bpb,
        core_eval=args.core_eval,
        core_max_per_task=args.core_max_per_task,
        ece_bins=args.ece_bins,
        init_type=args.init_type,
        use_moe=args.use_moe,
        num_experts=args.num_experts,
        moe_top_k=args.moe_top_k,
    )
    return 0


def _run_batch(
    *,
    batch_kind: str,
    presets: list[PresetId],
    seeds: list[int],
    args: argparse.Namespace,
) -> int:
    batch_id = args.batch_id or f"{batch_kind}_{_utc_stamp()}"
    batch_out_dir = Path(args.out_dir) / batch_id
    batch_out_dir.mkdir(parents=True, exist_ok=True)

    tbl = Table(title=f"Eval Matrix Batch: {batch_id}")
    tbl.add_column("preset")
    tbl.add_column("seed", justify="right")
    for preset in presets:
        for seed in seeds:
            tbl.add_row(preset, str(seed))
    console.print(tbl)

    for preset in presets:
        for seed in seeds:
            try:
                _run_one(
                    preset=preset,
                    train_tokens=args.train_tokens,
                    seed=seed,
                    device_type=args.device_type,
                    data=args.data,
                    out_dir=batch_out_dir,
                    sequence_len=args.sequence_len,
                    vocab_size=args.vocab_size,
                    n_layer=args.n_layer,
                    n_head=args.n_head,
                    n_embd=args.n_embd,
                    device_batch_size=args.device_batch_size,
                    total_batch_size_tokens=args.total_batch_size_tokens,
                    embedding_lr=args.embedding_lr,
                    unembedding_lr=args.unembedding_lr,
                    matrix_lr=args.matrix_lr,
                    weight_decay=args.weight_decay,
                    eval_tokens=args.eval_tokens,
                    eval_bpb=args.eval_bpb,
                    core_eval=args.core_eval,
                    core_max_per_task=args.core_max_per_task,
                    ece_bins=args.ece_bins,
                    init_type=args.init_type,
                    use_moe=args.use_moe,
                    num_experts=args.num_experts,
                    moe_top_k=args.moe_top_k,
                )
            except Exception as e:
                err_id = f"ERR-{preset}-t{args.train_tokens}-s{seed}-{_utc_stamp()}"
                row = _error_row(
                    run_id=err_id,
                    preset=preset,
                    seed=seed,
                    data=args.data,
                    device_type=args.device_type,
                    init_type=args.init_type,
                    sequence_len=args.sequence_len,
                    vocab_size=args.vocab_size,
                    n_layer=args.n_layer,
                    n_head=args.n_head,
                    n_embd=args.n_embd,
                    use_moe=args.use_moe,
                    num_experts=args.num_experts,
                    moe_top_k=args.moe_top_k,
                    device_batch_size=args.device_batch_size,
                    total_batch_size_tokens=args.total_batch_size_tokens,
                    train_tokens_requested=args.train_tokens,
                    eval_tokens=args.eval_tokens,
                    eval_bpb=args.eval_bpb,
                    core_eval=args.core_eval,
                    core_max_per_task=args.core_max_per_task,
                    ece_bins=args.ece_bins,
                    error=repr(e),
                )
                _write_summary(batch_out_dir, row)
                console.print(f"[bold red]Run failed:[/bold red] preset={preset} seed={seed} error={e!r}")
                if args.fail_fast:
                    raise

    console.print(f"Batch outputs: {batch_out_dir}")
    return 0


def _cmd_matrix(args: argparse.Namespace) -> int:
    allowed = set(PresetId.__args__)  # type: ignore[attr-defined]
    presets_raw = _parse_str_list(args.presets)
    presets: list[PresetId] = []
    for preset in presets_raw:
        if preset not in allowed:
            raise ValueError(f"Unknown preset {preset!r}. Allowed: {sorted(allowed)}")
        presets.append(cast(PresetId, preset))
    seeds = _parse_int_list(args.seeds)
    return _run_batch(batch_kind="matrix", presets=presets, seeds=seeds, args=args)


def _cmd_ablation(args: argparse.Namespace) -> int:
    seeds = _parse_int_list(args.seeds)
    return _run_batch(
        batch_kind="ablation",
        presets=list(DEFAULT_ABLATION_PRESETS),
        seeds=seeds,
        args=args,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Bio vs vanilla eval matrix harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--device-type", default="", help="cuda|cpu|mps (default: autodetect)")
        p.add_argument("--data", default="fineweb", choices=["fineweb", "synthetic"])
        p.add_argument("--out-dir", default="runs/eval_matrix")
        p.add_argument("--train-tokens", type=int, default=10_000_000)
        p.add_argument("--eval-tokens", type=int, default=1_000_000)
        p.add_argument("--eval-bpb", action="store_true", help="Also compute val bpb (requires tokenizer artifacts)")
        p.add_argument("--core-eval", action="store_true", help="Also compute CORE metric (requires eval bundle)")
        p.add_argument("--core-max-per-task", type=int, default=200)
        p.add_argument("--ece-bins", type=int, default=15)
        p.add_argument("--batch-id", default=None, help="Optional subdirectory name under --out-dir")
        p.add_argument("--fail-fast", action="store_true", help="Stop the batch on the first failure")

        # model arch
        p.add_argument("--sequence-len", type=int, default=2048)
        p.add_argument("--vocab-size", type=int, default=50304)
        p.add_argument("--n-layer", type=int, default=12)
        p.add_argument("--n-head", type=int, default=12)
        p.add_argument("--n-embd", type=int, default=768)

        # init
        p.add_argument("--init-type", default="baseline", choices=["baseline", "ca_rule30", "ca_rule116"])

        # bio/moe
        p.add_argument("--use-moe", action="store_true")
        p.add_argument("--num-experts", type=int, default=8)
        p.add_argument("--moe-top-k", type=int, default=2)

        # batch / opt
        p.add_argument("--device-batch-size", type=int, default=8)
        p.add_argument("--total-batch-size-tokens", type=int, default=131072)
        p.add_argument("--embedding-lr", type=float, default=0.2)
        p.add_argument("--unembedding-lr", type=float, default=0.004)
        p.add_argument("--matrix-lr", type=float, default=0.02)
        p.add_argument("--weight-decay", type=float, default=0.0)

    p_run = sub.add_parser("run", help="Run a single preset/seed")
    add_common(p_run)
    p_run.add_argument("--preset", required=True, choices=list(PresetId.__args__))  # type: ignore[attr-defined]
    p_run.add_argument("--seed", type=int, default=1337)
    p_run.set_defaults(func=_cmd_run)

    p_matrix = sub.add_parser("matrix", help="Run presets × seeds")
    add_common(p_matrix)
    p_matrix.add_argument("--presets", required=True, help="Comma-separated presets")
    p_matrix.add_argument("--seeds", required=True, help="Comma-separated seeds")
    p_matrix.set_defaults(func=_cmd_matrix)

    p_ablation = sub.add_parser("ablation", help="Run the standard feature-ablation sweep")
    add_common(p_ablation)
    p_ablation.add_argument("--seeds", required=True, help="Comma-separated seeds")
    p_ablation.set_defaults(func=_cmd_ablation)

    args = parser.parse_args()
    if args.device_type == "":
        args.device_type = autodetect_device_type()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
