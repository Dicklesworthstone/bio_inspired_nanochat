"""Compare CUDA arithmetic on the same trained checkpoint, training blocks and held-out tokens."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console

from bio_inspired_nanochat.checkpoint_manager import load_checkpoint_model
from bio_inspired_nanochat.dataloader import tokenizing_distributed_data_loader
from bio_inspired_nanochat.engine import Engine, KVCache
from bio_inspired_nanochat.matrix_manifest import file_digest


FORMATS = ("fp32", "tf32", "bf16", "fp16")
PROMPT = 32
CACHED = 8


def _autocast(mode: str):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(mode)
    return nullcontext() if dtype is None else torch.autocast("cuda", dtype)


def _data_hash(tokens: torch.Tensor, targets: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in (tokens, targets):
        digest.update(tensor.cpu().numpy().astype("<i8", copy=False).tobytes())
    return digest.hexdigest()


def _measure(stock: Any, train, valid, mode: str, seed: int):
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("highest" if mode == "fp32" else "high")
    model = copy.deepcopy(stock)
    model.eval()
    x, y = valid
    cfg = model.config
    cache = KVCache(1, cfg.n_kv_head, PROMPT + CACHED, cfg.n_embd // cfg.n_head, cfg.n_layer)
    engine = Engine(model, tokenizer=None)
    outputs = []
    with torch.inference_mode(), _autocast(mode):
        engine._forward(x[:, :PROMPT], cache)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for token in x[:, PROMPT:].split(1, dim=1):
            output = engine._forward(token, cache)
            outputs.append(output[0] if isinstance(output, tuple) else output)
        torch.cuda.synchronize()
        decode_s = time.perf_counter() - start
    logits = torch.cat(outputs, dim=1).float()
    if cache.pos != PROMPT + CACHED or not torch.isfinite(logits).all():
        raise ValueError("invalid cache position or nonfinite cached logits")
    losses = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y[:, PROMPT:].flatten(), reduction="none")

    # Restore the pristine state so inference cannot change the starting point of the update.
    model = copy.deepcopy(stock)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with _autocast(mode):
        output = model(train[0], targets=train[1])
        loss = output[1] if isinstance(output, tuple) else output
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    train_s = time.perf_counter() - start
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    finite = torch.stack([torch.isfinite(t).all() for t in (*grads, *model.parameters(), loss)])
    if not grads or not finite.all():
        raise ValueError("absent gradients or nonfinite loss, gradients or updated weights")
    return {
        "status": "passed", "decode_seconds": decode_s, "train_seconds": train_s,
        "train_loss": float(loss.detach()), "gradient_tensors": len(grads),
        "target_nll": losses.tolist(), "argmax": logits.argmax(-1).flatten().tolist(),
    }


def _summarize(rows: list[dict[str, Any]], warmup: dict[str, Any], mode: str) -> dict[str, Any]:
    pairs = [(r["formats"]["fp32"], r["formats"][mode]) for r in rows]
    if any(warmup["formats"][m]["status"] != "passed" for m in ("fp32", mode)) or any(
        a["status"] != "passed" or b["status"] != "passed" for a, b in pairs
    ):
        return {"eligible": False, "reason": "incomplete or nonfinite pair"}
    agreement = np.mean([np.equal(a["argmax"], b["argmax"]).mean() for a, b in pairs])
    delta = np.mean([abs(np.mean(a["target_nll"]) - np.mean(b["target_nll"])) for a, b in pairs])
    eligible = agreement >= 0.99 and delta <= 0.01
    result: dict[str, Any] = {"eligible": bool(eligible), "argmax_agreement": float(agreement),
                              "mean_abs_block_nll_delta": float(delta)}
    indices = np.random.default_rng(1951).integers(len(pairs), size=(20000, len(pairs)))
    for timing in ("decode_seconds", "train_seconds"):
        ratios = np.array([a[timing] / b[timing] for a, b in pairs])
        lo, hi = np.quantile(np.median(ratios[indices], axis=1), [0.025, 0.975])
        median = float(np.median(ratios))
        result[timing] = {"median_speedup": median, "paired_bootstrap_95": [float(lo), float(hi)],
                          "passes_rule": bool(eligible and median >= 1.05 and lo > 1.0)}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-blocks", type=int, default=0,
                        help="Skip this many batches in both streams before warmup and measurement")
    args = parser.parse_args()
    if args.skip_blocks < 0:
        parser.error("--skip-blocks must be nonnegative")
    device = torch.device("cuda", 0)
    stock, _ = load_checkpoint_model(str(args.checkpoint_dir), args.step, device, "eval")
    # Promote stored matrix weights once for all arms. Rotary lookup tables and
    # persistent biological state retain their checkpoint-loader dtypes.
    for module in stock.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.float()
    stock.tie_weights()
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in stock.parameters()):
        raise ValueError("checkpoint has unexpected non-FP32 parameters outside matrix layers")
    train_stream = tokenizing_distributed_data_loader(1, PROMPT, "train", device="cuda")
    val_stream = tokenizing_distributed_data_loader(1, PROMPT + CACHED, "val", device="cuda")
    for _ in range(args.skip_blocks):
        next(train_stream)
        next(val_stream)
    observations = []
    failed = False
    for block in range(-1, 4):
        train, valid = next(train_stream), next(val_stream)
        order = FORMATS[block % 4:] + FORMATS[:block % 4]
        row: dict[str, Any] = {"block": block, "order": order, "train_sha256": _data_hash(*train),
                               "validation_sha256": _data_hash(*valid), "formats": {}}
        for mode in order:
            try:
                row["formats"][mode] = _measure(stock, train, valid, mode, 1337 + block)
            except (AssertionError, RuntimeError, ValueError) as error:
                row["formats"][mode] = {"status": "failed", "reason": str(error)}
                failed = True
        observations.append(row)
    warmup, *rows = observations
    report = {
        "scope": "Four paired text blocks per checkpoint after one excluded warmup block. Single-device "
                 "cached inference and one unscaled AdamW update per block. No training-quality estimate.",
        "checkpoint": args.checkpoint_dir.name, "step": args.step, "source_sha256": file_digest(Path(__file__)),
        "checkpoint_sha256": file_digest(args.checkpoint_dir / f"model_{args.step:06d}.pt"),
        "model_source_sha256": {
            name: file_digest(Path(__file__).resolve().parents[2] / "bio_inspired_nanochat" / name)
            for name in ("common.py", "gpt.py", "gpt_synaptic.py")
        },
        "skip_blocks": args.skip_blocks,
        "torch": str(torch.__version__), "device": torch.cuda.get_device_name(device),
        "prompt": PROMPT, "cached_steps": CACHED, "warmup": warmup, "blocks": rows,
        "comparisons_with_fp32": {m: _summarize(rows, warmup, m) for m in FORMATS[1:]},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    Console().print(report["comparisons_with_fp32"])
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
