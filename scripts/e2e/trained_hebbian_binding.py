"""Matched training and read interventions on Nanochat's normalized Hebbian write."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from rich.console import Console
from torch.nn.functional import log_softmax

from bio_inspired_nanochat.common import get_base_dir
from bio_inspired_nanochat.dataloader import tokenizing_distributed_data_loader
from bio_inspired_nanochat.dataset import parquet_paths_for_split
from bio_inspired_nanochat.engine import KVCache
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.matrix_manifest import file_digest
from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticLinear
from bio_inspired_nanochat.synthetic_tasks import associative_recall
from bio_inspired_nanochat.tokenizer import get_token_bytes, get_tokenizer


CONTEXT, CHUNK, STEPS, DOCUMENTS = 1024, 128, 768, 128
SEEDS = (6311, 6312, 6313, 6314)
EXPOSURES = (STEPS, 128, 64)
CONSOLE = Console()


class TraceRead:
    """Apply one deterministic random row permutation to both trace factors."""

    def __init__(self, model: GPTSynaptic, seed: int | None):
        self.stack = ExitStack()
        self.seed = seed
        self.sites = [module for module in model.modules() if isinstance(module, SynapticLinear)]
        self.calls = 0
        self.indices: torch.Tensor | None = None
        self.generator = torch.Generator().manual_seed(seed) if seed is not None else None

    def __enter__(self):
        for site in self.sites:
            original = site._update_hebb_traces

            def observe(x, y, genes, original=original):
                if self.generator is not None:
                    if self.calls % len(self.sites) == 0:
                        self.indices = torch.stack(
                            [torch.randperm(y.size(1), generator=self.generator) for _ in range(y.size(0))]
                        ).to(y.device)
                    if self.indices is None or tuple(self.indices.shape) != tuple(y.shape[:2]):
                        raise RuntimeError("Trace sites did not see the same dense chunk")
                    y = y.gather(1, self.indices.unsqueeze(-1).expand_as(y))
                self.calls += 1
                return original(x, y, genes)

            self.stack.callback(delattr, site, "_update_hebb_traces")
            setattr(site, "_update_hebb_traces", observe)
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)


def model_for(seed: int, vocab: int, device: torch.device) -> GPTSynaptic:
    torch.manual_seed(seed)
    config = GPTSynapticConfig(
        sequence_len=CONTEXT, vocab_size=vocab, n_layer=2, n_head=3, n_kv_head=3,
        n_embd=192, tie_embeddings=True, init_seed=seed,
        syn_cfg=SynapticConfig(enable_hebbian=True, enable_stdp=False, enable_presyn=False,
                               enable_metabolism=False, post_slow_lr=0.0, fast_weight_normalized=True),
    )
    model = GPTSynaptic(config)
    model.init_weights()
    torch.nn.init.normal_(model.wte.weight, std=0.02)
    return model.to(device)


def tensor_digest(state) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def inputs_identity() -> dict:
    root = Path(get_base_dir())
    reference = json.loads(Path("results/vanilla_matched_counts_2026-11-23.json").read_text())["dataset_identity"]
    train = list(map(Path, parquet_paths_for_split("train")))
    val = list(map(Path, parquet_paths_for_split("val")))
    if len(train) != 27 or len(val) != 1 or not train[0].name.endswith("00000.parquet"):
        raise ValueError("Expected the archived 27 training shards and one separate validation shard")
    paths = [train[0], val[0], *(root / "tokenizer" / name for name in reference["tokenizer"])]
    expected = [reference["train"][0], reference["val"][0], *reference["tokenizer"].values()]
    digests = {p.name: file_digest(p) for p in paths}
    if list(digests.values()) != expected:
        raise ValueError("Staged dataset/tokenizer input digest mismatch")
    return digests


def train(seed: int, mode: str, steps: int, vocab: int, device: torch.device, out: Path, *, save: bool) -> dict:
    model = model_for(seed, vocab, device).train()
    initial = tensor_digest(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0, fused=True)
    loader = tokenizing_distributed_data_loader(1, CONTEXT, "train", device="cpu")
    stream = hashlib.sha256()
    losses, durations = [], []
    activity = []
    torch.cuda.reset_peak_memory_stats(device)
    with TraceRead(model, None if mode == "paired" else seed + 200_000) as intervention:
        for step in range(steps):
            x, y = next(loader)
            stream.update(torch.stack([x, y]).numpy().astype("<u4", copy=False).tobytes())
            start = time.perf_counter()
            model.reset_sequence_state(reset_fast_weights=True)
            loss = model.chunked_train_step(x.to(device), y.to(device), chunk_len=CHUNK)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss))
            if not math.isfinite(losses[-1]) or not math.isfinite(float(norm)):
                raise FloatingPointError(f"Non-finite loss or gradient at {seed}/{mode}/{step}")
            durations.append(time.perf_counter() - start)
            if step == 0 or (step + 1) % 64 == 0 or step + 1 == steps:
                norms = [float(site.w_fast.detach().norm()) for site in intervention.sites if site.w_fast is not None]
                if not norms or not all(math.isfinite(n) for n in norms) or max(norms) <= 1e-5:
                    raise RuntimeError("Expected finite, nonzero fast weights")
                activity.append({"step": step + 1, "norms": norms})
                CONSOLE.log(f"{seed} {mode} step {step + 1}/{steps} loss={losses[-1]:.5f}")
    if intervention.calls != len(intervention.sites) * steps * (CONTEXT // CHUNK):
        raise RuntimeError("Wrong number of trace observations")
    model.reset_sequence_state(reset_fast_weights=True)
    state = model.state_dict()
    if any(not torch.isfinite(t).all() for t in state.values() if t.is_floating_point()):
        raise FloatingPointError("Non-finite checkpoint tensor")
    summary = {"initial_sha256": initial, "train_stream_sha256": stream.hexdigest(), "losses": losses,
               "seconds": durations, "activity": activity, "trace_calls": intervention.calls,
               "state_sha256": tensor_digest(state), "peak_gpu_bytes": torch.cuda.max_memory_allocated(device)}
    if save:
        path = out / f"seed{seed}-{mode}.pt"
        with path.open("xb") as file:
            torch.save(state, file)
        summary.update({"checkpoint_sha256": file_digest(path), "checkpoint_bytes": path.stat().st_size})
    return summary


def validation(tokenizer: Any, sizes: torch.Tensor, count: int):
    path, = parquet_paths_for_split("val")
    selected = {}
    with pq.ParquetFile(path) as parquet:
        for row_group in range(parquet.num_row_groups // 2, parquet.num_row_groups):
            for batch in parquet.iter_batches(row_groups=[row_group], columns=["text"], batch_size=64):
                for content in batch.column(0).to_pylist():
                    ids = tokenizer.encode(content)[:512]
                    if len(ids) < 256 or int(sizes[ids].min()) <= 0:
                        continue
                    doc_id = hashlib.sha256(np.asarray(ids, dtype="<u8").tobytes()).hexdigest()
                    if doc_id not in selected:
                        selected[doc_id] = (row_group, ids, int(sizes[ids[CHUNK:]].sum()))
                    if len(selected) == count:
                        return selected
    raise ValueError("Insufficient eligible validation documents")


@torch.inference_mode()
def read(model: GPTSynaptic, inputs: torch.Tensor, targets: torch.Tensor, seed: int | None,
         *, active: bool = True, last_only: bool = False) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    model.reset_sequence_state(reset_fast_weights=True)
    cfg = model.config
    cache = KVCache(inputs.size(0), cfg.n_kv_head, inputs.size(1), cfg.n_embd // cfg.n_head, cfg.n_layer)
    logps, hits = [], 0
    prefix = None
    with TraceRead(model, seed) as control:
        for start in range(0, inputs.size(1), CHUNK):
            stop = start + CHUNK
            logits, _ = model(inputs[:, start:stop], kv_cache=cache, train_mode=False, update_mem=active)
            if start == 0:
                prefix = logits.detach().clone()
            if last_only:
                if stop < inputs.size(1):
                    continue
                logits = logits[:, -1:]
                gold = targets
                hits = int((logits.argmax(-1) == gold).sum())
            else:
                gold = targets[:, start:stop]
            logps.append(log_softmax(logits.float(), dim=-1).gather(-1, gold.unsqueeze(-1)).squeeze(-1).double().cpu())
    if prefix is None:
        raise ValueError("Read requires a nonempty sequence")
    expected = math.ceil(inputs.size(1) / CHUNK) * len(control.sites) if active else 0
    if control.calls != expected:
        raise RuntimeError("Unmatched trace-call counts")
    return torch.cat(logps, dim=-1), prefix, hits, control.calls


def modes(model, inputs, targets, seed, last_only=False):
    identity, prefix, correct, writes = read(model, inputs, targets, None, last_only=last_only)
    frozen, anchor, frozen_correct, _ = read(model, inputs, targets, None, active=False, last_only=last_only)
    first_error = float((prefix - anchor).abs().max())
    shuffled, shuffled_correct = [], []
    for draw in range(4):
        logp, first, hits, count = read(model, inputs, targets, seed + draw, last_only=last_only)
        first_error = max(first_error, float((prefix - first).abs().max()))
        if count != writes:
            raise RuntimeError("Write counts differ across active read arms")
        shuffled.append(logp)
        shuffled_correct.append(hits)
    if first_error != 0.0:
        raise RuntimeError(f"First-chunk no-effect control failed: {first_error}")
    return identity, frozen, torch.stack(shuffled), correct, frozen_correct, shuffled_correct, writes


def score(seed: int, vocab: int, bos: int, device: torch.device, documents, out: Path) -> None:
    report = {}
    training = json.loads((out / f"train-seed{seed}.json").read_text())["model"]
    with gzip.open(out / f"seed{seed}-docs.jsonl.gz", "xb") as stream:
        for arm in ("initial", "paired", "shuffle"):
            model = model_for(seed, vocab, device).eval()
            if arm == "initial":
                expected = training["paired"]["initial_sha256"]
            else:
                path = out / f"seed{seed}-{arm}.pt"
                if file_digest(path) != training[arm]["checkpoint_sha256"]:
                    raise ValueError(f"Checkpoint archive digest changed: {seed}/{arm}")
                model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
                expected = training[arm]["state_sha256"]
            if tensor_digest(model.state_dict()) != expected:
                raise ValueError(f"Loaded checkpoint tensor identity changed: {seed}/{arm}")
            biggest_change, write_count = 0.0, 0
            for doc_id, (row_group, ids, nbytes) in documents.items():
                x = torch.tensor([[bos, *ids[:-1]]], device=device)
                y = torch.tensor([ids], device=device)
                rng = int(doc_id[:14], 16) + 400_000
                identity, frozen, shuffled, _, _, _, writes = modes(model, x, y, rng)
                identity, frozen, shuffled = identity[:, CHUNK:], frozen[:, CHUNK:], shuffled[:, :, CHUNK:]
                biggest_change = max(biggest_change, float((identity - shuffled).abs().max()))
                write_count += writes
                mix = torch.logsumexp(shuffled, dim=0) - math.log(4.0)
                row = {"seed": seed, "arm": arm, "doc_sha256": doc_id, "row_group": row_group,
                       "bytes": nbytes, "tokens": len(ids) - CHUNK, "writes_per_active_read": writes,
                       "identity_bits": -float(identity.sum()) / math.log(2.0),
                       "frozen_bits": -float(frozen.sum()) / math.log(2.0),
                       "shuffle_bits": -float(mix.sum()) / math.log(2.0),
                       "per_draw_bits": (-shuffled.sum(dim=(1, 2)) / math.log(2.0)).tolist()}
                stream.write((json.dumps(row, allow_nan=False) + "\n").encode())
            if biggest_change < 1e-5:
                raise RuntimeError("The paired and shuffled writes did not change later predictions")
            probes = {}
            for pairs in (4, 8):
                recall = associative_recall(batch=64, num_pairs=pairs, vocab_size=vocab, seed=17900 + pairs)
                counts = np.zeros(6, dtype=np.int64)
                gold_losses = np.zeros(6, dtype=np.float64)
                for example in range(64):
                    row = recall.inputs[example].tolist()
                    x = torch.tensor([[*row[:-2], *([bos] * (CHUNK - len(row) + 2)), *row[-2:]]], device=device)
                    y = recall.targets[example, -1:].view(1, 1).to(device)
                    ident, frozen, shuffled, a, b, draws, _ = modes(model, x, y, 59000 + example, last_only=True)
                    counts += [a, b, *draws]
                    gold_losses -= [float(ident), float(frozen), *shuffled.flatten().tolist()]
                probes[pairs] = {"examples": 64, "hits": counts.tolist(), "gold_nll": gold_losses.tolist()}
            report[arm] = {"max_later_target_logprob_difference": biggest_change,
                           "active_trace_calls_per_mode": write_count,
                           "recall_order": ["paired", "frozen", "shuffle0", "shuffle1", "shuffle2", "shuffle3"],
                           "recall": probes}
            CONSOLE.log(f"{seed} {arm} held-out evaluation completed")
    with (out / f"seed{seed}-evaluation.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


def decision(rows, seeds=SEEDS):
    by_seed = {seed: {} for seed in seeds}
    for row in rows:
        if row["seed"] in by_seed:
            key = row["doc_sha256"], row["arm"]
            if key in by_seed[row["seed"]]:
                raise ValueError(f"Duplicate scored document: {row['seed']}/{key}")
            by_seed[row["seed"]][key] = row
    suffixes = sorted({key[0] for key in by_seed[seeds[0]]})
    required = {(doc, arm) for doc in suffixes for arm in ("initial", "paired", "shuffle")}
    if len(suffixes) != DOCUMENTS or any(set(by_seed[seed]) != required for seed in seeds):
        raise ValueError("Incomplete paired design")
    byte_counts = np.array([by_seed[seeds[0]][doc, "paired"]["bytes"] for doc in suffixes])
    observations = []
    bpb = {}
    for seed, data in by_seed.items():
        if any(row["bytes"] != byte_counts[i] for i, doc in enumerate(suffixes)
               for row in (data[doc, "initial"], data[doc, "paired"], data[doc, "shuffle"])):
            raise ValueError("Paired byte denominators differ")
        bpb[seed] = {arm: {mode: math.fsum(data[doc, arm][f"{mode}_bits"] for doc in suffixes) / int(byte_counts.sum())
                           for mode in ("identity", "shuffle", "frozen")} for arm in ("initial", "paired", "shuffle")}
        matrix = []
        for doc in suffixes:
            initial, paired, shuffle = (data[doc, arm] for arm in ("initial", "paired", "shuffle"))
            gap = paired["shuffle_bits"] - paired["identity_bits"]
            matrix.append((gap, paired["frozen_bits"] - paired["identity_bits"],
                           gap - (initial["shuffle_bits"] - initial["identity_bits"]),
                           gap - (shuffle["shuffle_bits"] - shuffle["identity_bits"])))
        observations.append(matrix)
    effects = np.array(observations)
    per_seed = effects.sum(axis=1) / byte_counts.sum()
    avg = effects.mean(axis=0)
    rng = np.random.default_rng(19610)
    idx = rng.integers(len(suffixes), size=(32768, len(suffixes)))
    lower = np.quantile(avg[idx].sum(axis=1) / byte_counts[idx].sum(axis=1)[:, None], .0125, axis=0)
    contrasts = {}
    for index, name in enumerate(("over_shuffle", "over_frozen", "trained_minus_initial", "paired_minus_shuffle_training")):
        mean = float(per_seed[:, index].mean())
        contrasts[name] = {"delta_bpb": mean, "paired_lower": float(lower[index]),
                           "seed_deltas": per_seed[:, index].tolist(),
                           "pass": mean >= .005 and lower[index] > 0 and bool((per_seed[:, index] > 0).all())}
        contrasts[name]["pass"] = bool(contrasts[name]["pass"])
    return {"n_docs": len(suffixes), "bytes": int(byte_counts.sum()), "bpb": bpb,
            "contrasts": contrasts, "advance": all(c["pass"] for c in contrasts.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("probe", "train", "score", "decision"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--steps", type=int, choices=EXPOSURES)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "decision":
        rows = []
        for seed in SEEDS:
            with gzip.open(args.out / f"seed{seed}-docs.jsonl.gz", "rt") as stream:
                rows.extend(json.loads(line) for line in stream)
        with (args.out / "decision.json").open("x") as stream:
            json.dump(decision(rows), stream, indent=2, allow_nan=False)
        return
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")
    provenance = {"inputs": inputs_identity(), "source_sha256": file_digest(Path(__file__)),
                  "torch": str(torch.__version__), "device": torch.cuda.get_device_name(),
                  "seeds": args.seeds, "stage": args.stage}
    tokenizer = get_tokenizer()
    vocab, bos = tokenizer.get_vocab_size(), tokenizer.get_bos_token_id()
    if args.stage == "score":
        docs = validation(tokenizer, get_token_bytes("cpu"), DOCUMENTS)
        for seed in args.seeds:
            score(seed, vocab, bos, device, docs, args.out)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    steps = 16 if args.stage == "probe" else args.steps
    if steps is None:
        parser.error("train requires an explicitly admitted --steps")
    for seed in args.seeds:
        arms = {arm: train(seed, arm, steps, vocab, device, args.out, save=args.stage == "train")
                for arm in ("paired", "shuffle")}
        for field in ("initial_sha256", "train_stream_sha256", "trace_calls"):
            if arms["paired"][field] != arms["shuffle"][field]:
                raise RuntimeError(f"Training pair mismatch: {field}")
        if arms["paired"]["state_sha256"] == arms["shuffle"]["state_sha256"]:
            raise RuntimeError("The training intervention did not affect the checkpoint")
        per_update = max(statistics.fmean(a["seconds"][4:16]) for a in arms.values())
        projections = {exposure: per_update * 4 * exposure for exposure in EXPOSURES}
        recommended = next((exposure for exposure, seconds in projections.items() if seconds <= 2400), None)
        with (args.out / f"{args.stage}-seed{seed}.json").open("x") as stream:
            json.dump({"provenance": provenance, "model": arms, "steps": steps, "tokens": steps * CONTEXT,
                       "projected_seconds_per_card": projections, "largest_admitted_steps": recommended},
                      stream, indent=2, allow_nan=False)
        CONSOLE.log(f"{seed} projected per-card seconds: {projections}")


if __name__ == "__main__":
    main()
