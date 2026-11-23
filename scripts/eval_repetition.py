"""Acute presynaptic repetition ablation on a frozen dense synaptic checkpoint.

The primary contrast is presyn-on minus presyn-off at greedy decoding. Fixed
24-token horizons give each of 16 held-out documents 22 potential trigram slots.
Additional sampling and calibrated count-penalty arms are descriptive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from rich.console import Console

from bio_inspired_nanochat.checkpoint_manager import load_model
from bio_inspired_nanochat.common import get_base_dir
from bio_inspired_nanochat.dataset import parquet_paths_for_split
from bio_inspired_nanochat.engine import Engine, KVCache, sample_next_token
from bio_inspired_nanochat.eval_stats import bootstrap_ci
from bio_inspired_nanochat.matrix_manifest import file_digest
from bio_inspired_nanochat.synaptic import SynapticCausalSelfAttention, SynapticPresyn

PREFIX, HORIZON = 16, 24
PENALTIES = (0.0, 0.5, 1.0)


def repetition(tokens: list[int], horizon: int = HORIZON) -> dict:
    """Count repetitions after the first occurrence, including overlapping n-grams."""
    if horizon < 3 or len(tokens) > horizon:
        raise ValueError("generated sequence must fit the registered horizon of at least three tokens")
    grams = list(zip(tokens, tokens[1:], tokens[2:]))
    repeated = len(grams) - len(set(grams))
    return {"length": len(tokens), "trigram_opportunities": len(grams), "repeated_trigrams": repeated,
            "repeated_per_slot": repeated / (horizon - 2),
            "repeated_per_emitted_trigram": repeated / len(grams) if grams else None}


def _documents(path: Path, tokenizer) -> tuple[list[dict], list[dict]]:
    # Independent uniform selections from disjoint 32-document strata. The first
    # 384 rows were used by a different study. No prediction enters eligibility.
    docs = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=["text"]):
        docs.extend(batch.column(0).to_pylist())
        if len(docs) >= 1024:
            break
    if len(docs) < 1024:
        raise ValueError("native validation shard is too short for the registered strata")
    rng = random.Random(20261122)
    samples = []
    seen = set()
    for start in range(384, 1024, 32):
        order = list(range(start, start + 32))
        rng.shuffle(order)
        for row_id in order:
            ids = tokenizer.encode(docs[row_id])
            if len(ids) >= PREFIX + HORIZON:
                prefix = tuple(ids[:PREFIX])
                if prefix in seen:
                    raise ValueError("duplicate selected document prefix")
                seen.add(prefix)
                samples.append({"row": row_id, "prefix": list(prefix),
                                "reference": ids[PREFIX:PREFIX + HORIZON], "seed": 62000 + row_id})
                break
        else:
            raise ValueError(f"no eligible document in stratum beginning at {start}")
    return samples[:4], samples[4:]


def _weight_digest(model) -> str:
    h = hashlib.sha256()
    for name, tensor in model.named_parameters():
        h.update(name.encode() + b"\x00")
        h.update(str((tuple(tensor.shape), tensor.dtype)).encode() + b"\x00")
        h.update(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@contextmanager
def _presyn(model, enabled: bool):
    cfgs = {id(m.cfg): m.cfg for m in model.modules()
            if isinstance(m, (SynapticCausalSelfAttention, SynapticPresyn))}
    if not cfgs or not all(cfg.enable_presyn and not cfg.enable_hebbian for cfg in cfgs.values()):
        raise ValueError("requires a trained presyn-on checkpoint with Hebbian plasticity disabled")
    try:
        for cfg in cfgs.values():
            cfg.enable_presyn = enabled
        model.reset_sequence_state()
        yield
    finally:
        for cfg in cfgs.values():
            cfg.enable_presyn = True


@torch.inference_mode()
def _decode(model, tokenizer, docs: list[dict], temperature: float, penalty: float) -> list[dict]:
    """Batched raw-LM decoding with serving forward, native KV state and per-document RNGs."""
    cfg = model.config
    cache = KVCache(batch_size=len(docs), num_heads=cfg.n_kv_head, head_dim=cfg.n_embd // cfg.n_head,
                    seq_len=PREFIX + HORIZON, num_layers=cfg.n_layer)
    engine = Engine(model, tokenizer)
    generators = [torch.Generator().manual_seed(doc["seed"]) for doc in docs]
    tokens = torch.tensor([d["prefix"] for d in docs], dtype=torch.long)
    counts = torch.zeros((len(docs), cfg.vocab_size))
    generated: list[list[int]] = [[] for _ in docs]
    stops: list[int | None] = [None for _ in docs]
    stop_ids = {tokenizer.get_bos_token_id(), tokenizer.encode_special("<|assistant_end|>")}
    for _ in range(HORIZON):
        logits, _loss = engine._forward(tokens, cache)
        scores = logits[:, -1, :].float()
        if penalty:
            counts.scatter_add_(1, tokens, torch.ones_like(tokens, dtype=counts.dtype))
            scores = scores - penalty * counts
        draws = []
        for j, row in enumerate(scores):
            if stops[j] is None:
                token = int(sample_next_token(row[None, :], generators[j], temperature, None).item())
                if token in stop_ids:
                    stops[j] = token
                else:
                    generated[j].append(token)
            else:
                token = stops[j]
            draws.append(token)
        if all(s is not None for s in stops):
            break
        tokens = torch.tensor(draws, dtype=torch.long).unsqueeze(1)
    return [{"output_ids": ids, "stop_token": stop, "terminated_by": "stop_token" if stop is not None else "length",
             **repetition(ids)} for ids, stop in zip(generated, stops, strict=True)]


@torch.inference_mode()
def _continuation_nll(model, docs: list[dict], penalties: set[float]) -> dict[float, list[float]]:
    """Mean next-token NLL on each of the 24 authentic following tokens."""
    ids = torch.tensor([d["prefix"] + d["reference"] for d in docs])
    logits, _ = model(ids[:, :-1], train_mode=False)
    logits = logits[:, PREFIX - 1:, :].float()
    counts = torch.zeros((len(docs), logits.size(-1)))
    counts.scatter_add_(1, ids[:, :PREFIX - 1], torch.ones_like(ids[:, :PREFIX - 1], dtype=counts.dtype))
    losses = {p: torch.zeros(len(docs)) for p in penalties}
    for offset in range(HORIZON):
        context_token = ids[:, PREFIX - 1 + offset : PREFIX + offset]
        counts.scatter_add_(1, context_token, torch.ones_like(context_token, dtype=counts.dtype))
        for p in penalties:
            scores = logits[:, offset, :] - p * counts if p else logits[:, offset, :]
            losses[p] += F.cross_entropy(scores, ids[:, PREFIX + offset], reduction="none")
    return {p: (loss / HORIZON).tolist() for p, loss in losses.items()}


def _contrast(treatment, control, field) -> dict:
    deltas = [a[field] - b[field] for a, b in zip(treatment, control, strict=True)]
    low, high = bootstrap_ci(np.asarray(deltas), n_boot=10000, seed=0)
    return {"document_deltas": deltas, "mean": float(np.mean(deltas)), "paired_bootstrap_95": [low, high],
            "lower": sum(v < 0 for v in deltas), "higher": sum(v > 0 for v in deltas),
            "ties": sum(v == 0 for v in deltas)}


def run(tag: str, step: int) -> dict:
    console = Console()
    root = Path(get_base_dir())
    model, tokenizer, meta = load_model("base", torch.device("cpu"), "eval", model_tag=tag, step=step)
    if model.training or not meta.get("synapses") or model.config.use_moe or meta["step"] != step:
        raise ValueError("requires an eval-mode dense synaptic checkpoint at the requested step")
    val, = parquet_paths_for_split("val", data_dir=str(root / "base_data"))
    if val in parquet_paths_for_split("train", data_dir=str(root / "base_data")):
        raise ValueError("training and validation share the same shard")
    cal, held = _documents(Path(val), tokenizer)
    before = _weight_digest(model)
    console.print(f"Selected {len(cal)} calibration and {len(held)} held-out documents.")
    with _presyn(model, True):
        cal_on = _decode(model, tokenizer, cal, 0.0, 0.0)
    with _presyn(model, False):
        cal_off = {p: _decode(model, tokenizer, cal, 0.0, p) for p in PENALTIES}
    def mean_rep(rows):
        return sum(row["repeated_per_slot"] for row in rows) / len(rows)

    chosen = min(PENALTIES, key=lambda p: (abs(mean_rep(cal_off[p]) - mean_rep(cal_on)), p))
    aliases = {"on": (True, 0.0), "off": (False, 0.0),
               "off_penalty": (False, chosen), "on_penalty": (True, chosen)}
    samples = {"greedy": {}, "sampled": {}}
    computed = {}
    for on in (False, True):
        with _presyn(model, on):
            quality = _continuation_nll(model, held, {0.0, chosen})
            for policy, temperature in (("greedy", 0.0), ("sampled", 1.0)):
                for p in sorted({0.0, chosen}):
                    console.print(f"Decoding {len(held)} documents, presyn={on}, {policy}, penalty={p}.")
                    rows = _decode(model, tokenizer, held, temperature, p)
                    for row, nll in zip(rows, quality[p], strict=True):
                        row["reference_nll"] = nll
                    computed[on, policy, p] = rows
    contrasts = {}
    for policy, arms in samples.items():
        for name, (on, p) in aliases.items():
            arms[name] = computed[on, policy, p]
        contrasts[policy] = {}
        for treatment, control in (("on", "off"), ("on", "off_penalty"), ("on_penalty", "off_penalty")):
            contrasts[policy][f"{treatment}_minus_{control}"] = {
                field: _contrast(arms[treatment], arms[control], field)
                for field in ("repeated_per_slot", "length", "reference_nll")}
    after = _weight_digest(model)
    if after != before:
        raise ValueError("model parameters changed during an inference ablation")
    ckpt = root / "base_checkpoints" / tag
    encoding = root / "tokenizer" / "tokenizer.json"
    if not encoding.exists():
        encoding = root / "tokenizer" / "tokenizer.pkl"
    files = [ckpt / f"model_{step:06d}.pt", ckpt / f"meta_{step:06d}.json", encoding, Path(val)]
    return {
        "scope": "Exploratory acute inference switch on one trained checkpoint. No retraining or D1 selection.",
        "protocol": {"prefix_tokens": PREFIX, "horizon": HORIZON, "calibration_strata": "4 x 32 rows from 384",
                     "heldout_strata": "16 x 32 rows from 512", "document_selection_seed": 20261122,
                     "sampling_seed": "62000 + document row", "sampling": "temperature=1, full vocabulary",
                     "primary": "greedy on-minus-off mean repeated generated trigram occurrences per 22 slots",
                     "frequency_penalty": "subtract lambda times token count in prefix plus emitted history",
                     "calibration": "match ON mean greedy repetition with OFF; minimize absolute delta, ties smaller lambda",
                     "uncertainty": "paired document bootstrap with 10,000 resamples, seed 0; conditional on this checkpoint, shard, sampling seed and prompt selection",
                     "stops": "BOS or assistant_end stops before further content; no tools or abstention policy"},
        "checkpoint": {"tag": tag, "step": step, "architecture": meta["model_config"],
                       "training_tokens": step * meta["user_config"]["total_batch_size"],
                       "source_provenance": meta.get("provenance"),
                       "training_tokenizer_hash": None,
                       "weight_sha256_before": before, "weight_sha256_after": after,
                       "ablation": "enable_presyn only; the shared causal distance barrier remains enabled"},
        "file_sha256": {str(p.relative_to(root)): file_digest(p) for p in files},
        "source_sha256": file_digest(Path(__file__)), "torch_version": str(torch.__version__),
        "calibration_docs": cal, "calibration_on": cal_on, "calibration_off_grid": cal_off,
        "selected_lambda": chosen, "heldout_docs": held, "arms": samples, "contrasts": contrasts,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    opts = parser.parse_args()
    result = run(opts.model_tag, opts.step)
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    opts.output.write_text(json.dumps(result, allow_nan=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
