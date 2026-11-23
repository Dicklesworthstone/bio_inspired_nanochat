"""Compare causal episodic write policies using archived train counts and real validation documents."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from rich.console import Console

from bio_inspired_nanochat.matrix_manifest import file_digest


BLOCK, WRITES, CAPACITY = 32, 4, 64
MASS = 4.0
CONTROLS = ("frozen", "uniform", "recency", "rarity", "surprise", "frequency_surprise",
            "uniform_order2", "frequency_surprise_order2")
ARMS = ("gain", *CONTROLS, "dense")
ROOT = Path(__file__).resolve().parents[2]
Event = tuple[int, int, int]


def write_gain(p0: float, block_hits: int, block_cues: int, hits: int, cues: int) -> float:
    """Expected marginal log-score improvement, weighted by empirical cue frequency."""
    posterior = (block_hits + MASS * p0) / (block_cues + MASS)
    return block_cues * (posterior * math.log2(1 + 1 / (hits + MASS * p0))
                         - math.log2(1 + 1 / (cues + MASS)))


class Memory:
    def __init__(self, order2: bool = False) -> None:
        self.order2 = order2
        self.slots: deque[Event] = deque()
        self.cues: Counter[int] = Counter()
        self.pairs: Counter[tuple[int, int]] = Counter()
        self.contexts: Counter[tuple[int, int]] = Counter()
        self.triples: Counter[Event] = Counter()
        self.writes = 0

    @staticmethod
    def _adjust[K](table: Counter[K], key: K, change: int) -> None:
        table[key] += change
        if not table[key]:
            del table[key]

    def _count(self, event: Event, change: int) -> None:
        z, x, y = event
        self._adjust(self.cues, x, change)
        self._adjust(self.pairs, (x, y), change)
        if self.order2:
            self._adjust(self.contexts, (z, x), change)
            self._adjust(self.triples, event, change)

    def write(self, event: Event) -> None:
        if len(self.slots) == CAPACITY:
            self._count(self.slots.popleft(), -1)
        self.slots.append(event)
        self._count(event, 1)
        self.writes += 1

    def predict(self, event: Event, base: float) -> float:
        z, x, y = event
        prob = (self.pairs[x, y] + MASS * base) / (self.cues[x] + MASS)
        if self.order2:
            prob = (self.triples[event] + MASS * prob) / (self.contexts[z, x] + MASS)
        return prob


def complete_documents(raw: np.ndarray, vocab: int, bos: int):
    """Yield offsets of whole nonempty documents; neither end can cross a stream gap."""
    x, y = raw // vocab, raw % vocab
    start = None
    previous = -1
    for i, (cue, target) in enumerate(zip(x.tolist(), y.tolist(), strict=True)):
        if cue != previous:
            start = None
        if cue == bos:
            start = i
        if target == bos:
            if start is not None and start < i:
                yield start, i
            start = None
        previous = target


def score_document(events: list[Event], bigram: list[float], unigram: list[float], seed: int):
    """All writes are deferred until their block has been fully predicted."""
    memories = {arm: Memory(arm.endswith("_order2")) for arm in ARMS if arm != "frozen"}
    bits = dict.fromkeys(ARMS, 0.0)
    rng = random.Random(seed)
    for start in range(0, len(events), BLOCK):
        block = events[start:start + BLOCK]
        prior = bigram[start:start + BLOCK]
        for event, p in zip(block, prior, strict=True):
            bits["frozen"] -= math.log2(p)
            for arm, mem in memories.items():
                bits[arm] -= math.log2(mem.predict(event, p))
        if len(block) < BLOCK:
            continue
        cues = Counter(x for _, x, _ in block)
        pairs = Counter((x, y) for _, x, y in block)
        surprise = [-math.log2(p) for p in prior]
        freq_surprise = [pairs[x, y] * s for (_, x, y), s in zip(block, surprise, strict=True)]
        mem = memories["gain"]
        gain = [write_gain(p, pairs[x, y], cues[x], mem.pairs[x, y], mem.cues[x])
                for (_, x, y), p in zip(block, prior, strict=True)]
        def top(scores):
            return sorted(range(BLOCK), key=scores.__getitem__, reverse=True)[:WRITES]

        uniform = rng.sample(range(BLOCK), WRITES)
        frequent = top(freq_surprise)
        selection = {"gain": top(gain), "uniform": uniform, "recency": range(BLOCK-WRITES, BLOCK),
                     "rarity": top([-p for p in unigram[start:start + BLOCK]]),
                     "surprise": top(surprise), "frequency_surprise": frequent,
                     "uniform_order2": uniform, "frequency_surprise_order2": frequent,
                     "dense": range(BLOCK)}
        for arm, indices in selection.items():
            for idx in sorted(indices):
                memories[arm].write(block[idx])
    writes = {arm: mem.writes for arm, mem in memories.items()}
    if any(n != len(events) // BLOCK * (BLOCK if arm == "dense" else WRITES)
           for arm, n in writes.items()):
        raise ValueError("write budget mismatch")
    return bits, writes


@dataclass
class Document:
    start: int
    end: int
    digest: str
    panel: int
    nbytes: int


def document_index(raw, vocab, bos, sizes):
    y = raw % vocab
    seen = set()
    docs = []
    duplicates = 0
    for start, end in complete_documents(raw, vocab, bos):
        targets = y[start:end]
        if np.any(sizes[targets] <= 0):
            raise ValueError("Unexpected zero-byte token inside a bounded document")
        digest = hashlib.sha256(targets.astype("<u8", copy=False).tobytes()).hexdigest()
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        docs.append(Document(start, end, digest, int(digest[:2], 16) % 2, int(sizes[targets].sum())))
    return docs, duplicates


def load_prior(root, raw, sizes, reference, prefix):
    """Reconstruct probabilities and independently reproduce the archived whole-stream score."""
    files = [root / f"{prefix}counts-rank{rank}.npz" for rank in range(2)]
    if [file_digest(f) for f in files] != reference["rank_count_archive_sha256"]:
        raise ValueError("Training count archive digest mismatch")
    field = "nxy" if prefix else "counts"
    with np.load(files[0], allow_pickle=False) as a, np.load(files[1], allow_pickle=False) as b:
        ux, uy, nxy = (a[k] + b[k] for k in ("ux", "uy", field))
    vocab = len(sizes)
    x, y = raw // vocab, raw % vocab
    valid = sizes[y] > 0
    keys, index = np.unique(raw[valid], return_inverse=True)
    if len(keys) != len(nxy) or int(ux.sum()) != int(uy.sum()):
        raise ValueError("Invalid count dimensions or training totals")
    uni = (uy + 0.5) / (uy.sum() + 0.5 * vocab)
    prob = np.zeros(len(raw), dtype=np.float64)
    prob[valid] = (nxy[index] + uni[y[valid]]) / (ux[x[valid]] + 1)
    total_bits = float(-np.log2(prob[valid]).sum())
    denominator = int(sizes[y].sum())
    if (not math.isclose(total_bits, reference["bigram_cross_entropy_bits"], rel_tol=1e-12)
            or not math.isclose(total_bits / denominator, reference["bigram_bpb"], rel_tol=1e-12)):
        raise ValueError("Archived held-out bigram numerator or BPB does not reproduce")
    return uni, prob, {"train_pairs": int(ux.sum()), "whole_stream_bits": total_bits,
                       "whole_stream_bytes": denominator, "whole_stream_bpb": total_bits / denominator}


def compare(rows, panel):
    subset = [r for r in rows if r["panel"] == panel]
    sizes = np.array([r["bytes"] for r in subset])
    totals = {arm: sum(r["bits"][arm] for r in subset) / int(sizes.sum()) for arm in ARMS}
    diffs = np.array([[r["bits"][arm] - r["bits"]["gain"] for arm in CONTROLS] for r in subset])
    rng = np.random.default_rng(7291 + panel)
    draws = []
    for _ in range(128):
        idx = rng.integers(len(subset), size=(256, len(subset)))
        draws.append(diffs[idx].sum(axis=1) / sizes[idx].sum(axis=1)[:, None])
    lower = np.quantile(np.concatenate(draws), 0.05 / 32, axis=0)
    contrasts = {}
    for arm, bound in zip(CONTROLS, lower.tolist(), strict=True):
        saving = totals[arm] - totals["gain"]
        contrasts[arm] = {"saving_bpb": saving, "lower_bpb": bound,
                          "pass": saving >= 0.005 and bound > 0}
    return {"documents": len(subset), "bytes": int(sizes.sum()), "bpb": totals, "contrasts": contrasts}


def study(root: Path):
    refs = [json.loads((ROOT / "results" / name).read_text()) for name in
            ("vanilla_matched_counts_2026-11-23.json", "vanilla_ngram_2026-11-22.json")]
    tokenizer_ids = refs[0]["dataset_identity"]["tokenizer"]
    if any(file_digest(root / name) != sha for name, sha in tokenizer_ids.items()):
        raise ValueError("Tokenizer identity mismatch")
    sizes = torch.load(root / "token_bytes.pt", weights_only=True, map_location="cpu").numpy()
    bos = json.loads((root / "tokenizer.json").read_text())["special_tokens"]["<|bos|>"]
    raw = np.load(root / "eval-pairs.npy", allow_pickle=False)
    digest = hashlib.sha256(raw.astype("<u8", copy=False).tobytes()).hexdigest()
    if digest != refs[0]["validation_ordered_pairs_sha256"] or digest != refs[1]["eval_pair_sha256"]:
        raise ValueError("Evaluation stream identity mismatch")
    docs, duplicates = document_index(raw, len(sizes), bos, sizes)
    summary = {"source_sha256": file_digest(Path(__file__)), "stream_sha256": digest,
               "complete_unique_documents": len(docs), "duplicate_documents": duplicates,
               "scored_positions": sum(d.end - d.start for d in docs), "priors": {}}
    records = []
    for prefix, name, ref in zip(("prefix-", ""), ("pilot", "full"), refs, strict=True):
        uni, prior, archived = load_prior(root, raw, sizes, ref, prefix)
        rows = []
        for doc in docs:
            pairs = raw[doc.start:doc.end]
            x, y = (pairs // len(sizes)).tolist(), (pairs % len(sizes)).tolist()
            events = list(zip([-1, *x[:-1]], x, y, strict=True))
            bits, writes = score_document(events, prior[doc.start:doc.end].tolist(), uni[y].tolist(),
                                          int(doc.digest, 16))
            rows.append({"prior": name, "document_sha256": doc.digest, "panel": doc.panel,
                         "start": doc.start, "end": doc.end, "bytes": doc.nbytes,
                         "bits": bits, "writes": writes})
        summary["priors"][name] = {"archive": archived, "panels": [compare(rows, panel) for panel in (0, 1)]}
        records.extend(rows)
    outcomes = [c["pass"] for prior in summary["priors"].values() for panel in prior["panels"]
                for c in panel["contrasts"].values()]
    summary["passed_contrasts"] = sum(outcomes)
    summary["total_contrasts"] = len(outcomes)
    summary["decision"] = "advance" if all(outcomes) else "stop"
    return summary, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    summary, records = study(args.inputs)
    args.out.mkdir(parents=True, exist_ok=False)
    raw = args.out / "documents.jsonl.gz"
    with gzip.GzipFile(filename=str(raw), mode="xb", mtime=0) as stream:
        for record in records:
            stream.write((json.dumps(record, allow_nan=False) + "\n").encode())
    summary["documents_sha256"] = file_digest(raw)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    Console().print(summary["decision"], summary["passed_contrasts"], "/", summary["total_contrasts"])


if __name__ == "__main__":
    main()
