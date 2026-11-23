"""Isolate cue-value pairing in document memory on a frozen pretrained language model."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import cast

import numpy as np
import pyarrow.parquet as pq
import torch
from rich.console import Console
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

from bio_inspired_nanochat.matrix_manifest import file_digest
from bio_inspired_nanochat.tokenizer import get_tokenizer


MIXES = (0.0, 0.01, 0.03, 0.1, 0.3, 0.6, 1.0)
TEMPS = (1, 5, 15, 50)
BLOCK = 32
WEIGHT_SHA = "29d2e457a664e41c12c735f20a36dc0956a665f614a54ce5db21a32e75965270"
FINEWEB_SHA = "4bbe362eb1ab719547b26608ca3f81d8b39262f31285dfb28c3d15a4c01db017"
WIKI_SHA = "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91"


def components(x, y, p0, hidden, block=BLOCK):
    """Target probabilities using only fully committed blocks. Hidden rows predict y at the same index."""
    result = {k: p0.copy() for k in ("binding", "exchangeable", "lexical", *(f"cosine_{t}" for t in TEMPS))}
    for i in range(block, len(y), block):
        a, b = max(0, i - 2 * block), min(i + block, len(y))
        same_x = x[i:b, None] == x[None, a:i]
        same_y = y[i:b, None] == y[None, a:i]
        nx = same_x.sum(axis=1)
        nxy = (same_x & same_y).sum(axis=1)
        # Average over all value permutations independently within each memory block.
        mean_nxy = np.zeros(b - i)
        for j in range(0, i - a, block):
            span = slice(j, j + block)
            mean_nxy += same_x[:, span].sum(axis=1) * same_y[:, span].sum(axis=1) / block
        result["binding"][i:b] = (nxy + 4 * p0[i:b]) / (nx + 4)
        result["exchangeable"][i:b] = (mean_nxy + 4 * p0[i:b]) / (nx + 4)
        result["lexical"][i:b] = (same_y.sum(axis=1) + 4 * p0[i:b]) / (i - a + 4)
        cosine = hidden[i:b] @ hidden[a:i].T
        cosine -= cosine.max(axis=1, keepdims=True)
        for t in TEMPS:
            kernel = np.exp(t * cosine)
            result[f"cosine_{t}"][i:b] = (kernel * same_y).sum(axis=1) / kernel.sum(axis=1)
    return result


def curves(p0, parts):
    # A pure continuous cache assigns zero probability to absent tokens. Represent
    # its infinite codelength explicitly, including if a whole article contains a miss.
    values = {}
    for arm, probs in parts.items():
        series = []
        for w in MIXES:
            p = (1 - w) * p0 + w * probs
            if not np.isfinite(p).all() or (p < 0).any() or (p > 1 + 1e-12).any():
                raise ValueError(f"Invalid probability in {arm}")
            series.append(float(-np.log2(p).sum()) if np.all(p > 0) else None)
        values[arm] = series
    return values


def wiki_articles(path: Path):
    lines = []
    with pq.ParquetFile(path) as parquet:
        for batch in parquet.iter_batches(columns=["text"]):
            for line in batch.column(0).to_pylist():
                stripped = line.strip()
                if re.fullmatch(r"= [^=].* =", stripped):
                    if lines:
                        yield "\n".join(lines).strip()
                    lines = [line]
                elif lines:
                    lines.append(line)
    if lines:
        yield "\n".join(lines).strip()


def fineweb_articles(path: Path):
    with pq.ParquetFile(path) as parquet:
        if parquet.num_row_groups < 2:
            raise ValueError("Two distinct terminal row groups required")
        for i in (parquet.num_row_groups - 1, parquet.num_row_groups - 2):
            for batch in parquet.iter_batches(columns=["text"], row_groups=[i], batch_size=64):
                yield from batch.column(0).to_pylist()


def panel(articles, tokenizer, prior_hashes, project_tokenizer=None, limit=None):
    selected = []
    seen = set()
    counts = {"considered": 0, "short": 0, "roundtrip": 0, "duplicate": 0, "prior_overlap": 0}
    for text in articles:
        counts["considered"] += 1
        prefix = text[:2048]
        ids = tokenizer.encode(prefix, add_special_tokens=False).ids
        while len(ids) > 512:
            prefix = prefix[:max(1, len(prefix) * 512 // len(ids))]
            ids = tokenizer.encode(prefix, add_special_tokens=False).ids
        if len(ids) < 128:
            counts["short"] += 1
            continue
        if 0 in ids or tokenizer.decode(ids, skip_special_tokens=False) != prefix:
            counts["roundtrip"] += 1
            continue
        digest = hashlib.sha256(prefix.encode()).hexdigest()
        if digest in seen:
            counts["duplicate"] += 1
            continue
        if project_tokenizer is not None:
            project_ids = np.asarray(project_tokenizer.encode(text), dtype="<u8")
            if hashlib.sha256(project_ids.tobytes()).hexdigest() in prior_hashes:
                counts["prior_overlap"] += 1
                continue
        selected.append((digest, len(prefix.encode()), ids))
        seen.add(digest)
        if limit is not None and len(selected) == limit:
            break
    if len(selected) < 16 or (limit is not None and len(selected) < limit):
        raise ValueError(f"Insufficient eligible documents: {counts}")
    return sorted(selected), counts


@torch.inference_mode()
def neural_inputs(model, ids, device, check_prefix=False):
    y = np.asarray(ids)
    x = np.asarray([0, *ids[:-1]])
    inputs = torch.tensor(x, device=device)[None, :]
    target = torch.tensor(y, device=device)
    output = model(inputs, use_cache=False, output_hidden_states=True)
    logp = output.logits[0].float().log_softmax(-1).gather(1, target[:, None]).squeeze(1)
    prefix_error = None
    if check_prefix:
        early = model(inputs[:, :64], use_cache=False).logits[0, -1].float().log_softmax(-1)
        prefix_error = float((early[target[63]] - logp[63]).abs())
        if prefix_error > 0.001:
            raise ValueError(f"Prefix invariance error: {prefix_error} nats")
    h = torch.nn.functional.normalize(output.hidden_states[-1][0].float(), dim=-1)
    return x, y, np.exp(logp.cpu().double().numpy()), h.cpu().double().numpy(), prefix_error


def choose(rows, arms):
    """One fixed setting per corpus/arm, learned only from the calibration articles."""
    candidates = []
    for arm in arms:
        temp = int(arm.split("_")[-1]) if arm.startswith("cosine_") else 0
        for j, w in enumerate(MIXES):
            values = [r["bits"][arm][j] for r in rows]
            if all(v is not None for v in values):
                candidates.append((sum(values), w, temp, arm, j))
    _, w, temp, arm, j = min(candidates)
    return {"arm": arm, "index": j, "mixture": w, "temperature": temp}


def calibrate(cal):
    settings = {a: choose(cal, [a]) for a in ("binding", "exchangeable", "lexical")}
    settings["continuous"] = choose(cal, [f"cosine_{t}" for t in TEMPS])
    settings["frozen"] = {"arm": "binding", "index": 0, "mixture": 0.0}
    settings["exchangeable_same"] = {**settings["binding"], "arm": "exchangeable"}
    return settings


def analyze(test, settings, seed):
    weights = np.array([r["bytes"] for r in test])
    sums = {}
    for k, s in settings.items():
        values = [r["bits"][s["arm"]][s["index"]] for r in test]
        if any(v is None for v in values):
            raise ValueError(f"Calibrated setting gave zero probability in test: {k}")
        sums[k] = np.array(values)
    bpb = {k: float(v.sum() / weights.sum()) for k, v in sums.items()}
    controls = ("frozen", "exchangeable_same", "exchangeable", "lexical", "continuous")
    diff = np.stack([sums[k] - sums["binding"] for k in controls], axis=1)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(128):
        idx = rng.integers(len(test), size=(256, len(test)))
        draws.append(diff[idx].sum(axis=1) / weights[idx].sum(axis=1)[:, None])
    lower = np.quantile(np.concatenate(draws), 0.005, axis=0)
    contrasts = {}
    for k, lo in zip(controls, lower.tolist(), strict=True):
        delta = bpb[k] - bpb["binding"]
        contrasts[k] = {"saving_bpb": delta, "lower_bpb": lo, "pass": delta >= 0.005 and lo > 0}
    return {"test_docs": len(test), "test_bytes": int(weights.sum()),
            "test_tokens": sum(r["tokens"] for r in test), "settings": settings, "bpb": bpb,
            "contrasts": contrasts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "wiki", "fineweb", "previous", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    pinned = {args.model / "model.safetensors": WEIGHT_SHA,
              args.model / "config.json": "76eb275107220e450d31258f792a2efcbee109d8b62ae0088260057dec06362f",
              args.model / "tokenizer.json": "c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624",
              args.fineweb: FINEWEB_SHA, args.wiki: WIKI_SHA,
              args.previous: "7239144ae469366ca2b9f5b46798d958372ca1f3f63c0a4f771f9986d2f28578"}
    for path, sha in pinned.items():
        if file_digest(path) != sha:
            raise ValueError(f"Input digest mismatch for {path.name}")
    with gzip.open(args.previous, "rt") as stream:
        previous = {json.loads(line)["document_sha256"] for line in stream}
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")
    pretrained = AutoModelForCausalLM.from_pretrained(str(args.model), local_files_only=True,
                                                      dtype=torch.float32, attn_implementation="eager")
    model = cast(torch.nn.Module, pretrained).to(device).eval()
    candidates = {"fineweb": panel(fineweb_articles(args.fineweb), tokenizer, previous, get_tokenizer(), 128),
                  "wikitext": panel(wiki_articles(args.wiki), tokenizer, set())}
    if {row[0] for row in candidates["fineweb"][0]} & {row[0] for row in candidates["wikitext"][0]}:
        raise ValueError("Duplicate selected text across corpora")
    documents = []
    summary = {"source_sha256": file_digest(Path(__file__)), "torch": str(torch.__version__),
               "device": torch.cuda.get_device_name(), "prior_documents_sha256": file_digest(args.previous),
               "inputs": {p.name: sha for p, sha in pinned.items()}, "corpora": {}}
    for number, (corpus, (samples, intake)) in enumerate(candidates.items()):
        def score(split):
            rows = []
            for i, (digest, nbytes, ids) in enumerate(samples):
                if (i % 4 == 0) != (split == "calibration"):
                    continue
                x, y, base, h, error = neural_inputs(model, ids, device, i == 0)
                rows.append({"corpus": corpus, "sha256": digest, "bytes": nbytes, "tokens": len(ids),
                             "split": split, "prefix_error_nats": error,
                             "bits": curves(base, components(x, y, base, h))})
            return rows

        calibration = score("calibration")
        settings = calibrate(calibration)
        test = score("test")
        summary["corpora"][corpus] = {**analyze(test, settings, 8911 + number),
                                      "calibration_docs": len(calibration), "intake": intake}
        documents.extend(sorted([*calibration, *test], key=lambda r: r["sha256"]))
    contrasts = [v["contrasts"] for v in summary["corpora"].values()]
    summary["mechanism_pass"] = all(c[k]["pass"] for c in contrasts for k in ("frozen", "exchangeable_same"))
    summary["competitive_pass"] = all(v["pass"] for c in contrasts for v in c.values())
    args.out.mkdir(parents=True, exist_ok=False)
    raw = args.out / "documents.jsonl.gz"
    with gzip.GzipFile(filename=str(raw), mode="xb", mtime=0) as f:
        for row in documents:
            f.write((json.dumps(row, allow_nan=False) + "\n").encode())
    summary["documents_sha256"] = file_digest(raw)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    Console().print("Association gate", summary["mechanism_pass"], "competitive gate", summary["competitive_pass"])


if __name__ == "__main__":
    main()
