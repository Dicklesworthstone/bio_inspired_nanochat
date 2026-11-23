"""Paired CPU experiment on a synthetic SCAN-style compositional split."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from rich.console import Console
from torch import nn

from bio_inspired_nanochat.eval_stats import bootstrap_ci, paired_t_test, wilcoxon_signed_rank
from bio_inspired_nanochat.results_registry import RunRecord, append_record
from bio_inspired_nanochat.sheaf_binding import SheafConsistencyMonitor, SheafDiffusionLayer

ACTIONS = ("walk", "run", "jump", "look")
REPEATS = ("once", "twice", "thrice")
WORDS = (*ACTIONS, *REPEATS, "and", "after", "end", "empty")
HOLDOUT = {frozenset(("walk", "jump")), frozenset(("run", "look"))}
SEEDS = (11, 23, 37, 53, 71, 89, 107, 131)
BUDGETS = (100, 400)
BLANK = len(ACTIONS)
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Example:
    prompt: str
    tokens: tuple[int, ...]
    target: tuple[int, ...]
    parts: tuple[int, ...]


def example(words: tuple[str, ...], actions: tuple[int, ...], parts: tuple[int, ...]) -> Example:
    return Example(
        " ".join(w for w in words if w not in ("end", "empty")),
        tuple(WORDS.index(w) for w in words),
        actions + (BLANK,) * (6 - len(actions)),
        parts,
    )


def corpus() -> tuple[list[Example], list[Example], list[Example]]:
    atoms = [
        example((verb, repeat, "end", "empty", "empty"), (action,) * count, ())
        for action, verb in enumerate(ACTIONS) for count, repeat in enumerate(REPEATS, 1)
    ]
    train, heldout = list(atoms), []
    for (i, left), (j, right), operator in product(enumerate(atoms), enumerate(atoms), ("and", "after")):
        words = (*left.prompt.split(), operator, *right.prompt.split())
        first, second = (i, j) if operator == "and" else (j, i)
        actions = tuple(a for k in (first, second) for a in atoms[k].target if a != BLANK)
        split = heldout if frozenset((words[0], words[3])) in HOLDOUT else train
        split.append(example(words, actions, (first, second)))
    return atoms, train, heldout


class Transducer(nn.Module):
    """The trained arms differ only in whether the parameter-free diffusion is enabled."""

    def __init__(self, diffusion: bool):
        super().__init__()
        self.tokens = nn.Embedding(len(WORDS), 32)
        self.positions = nn.Parameter(torch.randn(5, 32) * 0.02)
        self.encoder = nn.TransformerEncoderLayer(32, 4, 64, dropout=0.0, batch_first=True)
        self.diffusion = SheafDiffusionLayer(32, 3, 0.1, enabled=diffusion)
        self.register_buffer("path", SheafConsistencyMonitor.build_sheaf_laplacian(
            [(0, 1), (1, 2), (2, 3), (3, 4)], num_nodes=5, stalk_dim=1
        ))
        self.readout = nn.Sequential(nn.Flatten(), nn.Linear(160, 64), nn.ReLU(), nn.Linear(64, 30))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.tokens(tokens) + self.positions)
        return self.readout(self.diffusion(encoded, self.path)).reshape(-1, 6, 5)


def action_counts(sequence: list[int] | tuple[int, ...]) -> list[int]:
    counts = Counter(sequence)
    return [counts[action] for action in range(len(ACTIONS))]


def cycle_sum(first: list[int], second: list[int], whole: list[int]) -> list[int]:
    """Exact oriented cycle coordinate for three flat identity restriction maps."""
    if not (len(first) == len(second) == len(whole) == len(ACTIONS)):
        raise ValueError("three fully observed four-component edges required")
    return [x + y - z for x, y, z in zip(first, second, whole, strict=True)]


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(data: object) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _inputs(items: list[Example]) -> torch.Tensor:
    return torch.tensor([e.tokens for e in items])


def observations(model: Transducer, atoms: list[Example], heldout: list[Example]) -> list[dict]:
    # Each edge is a separate model prediction. Gold labels only score the compound.
    with torch.no_grad():
        atom_predictions = model(_inputs(atoms)).argmax(-1).tolist()
        compounds = model(_inputs(heldout)).argmax(-1).tolist()
    rows = []
    for query, whole in zip(heldout, compounds, strict=True):
        first, second = query.parts
        sub = [atom_predictions[first], atom_predictions[second]]
        edges = [action_counts(pred) for pred in (*sub, whole)]
        imbalance = cycle_sum(*edges)
        rows.append({
            "prompt": query.prompt, "gold": query.target, "prediction": whole,
            "subcommands": [atoms[first].prompt, atoms[second].prompt],
            "subcommand_predictions": sub, "edge_counts": edges, "cycle_sum": imbalance,
            "correct": list(query.target) == whole, "zero_class": not any(imbalance),
        })
    return rows


def summary(rows: list[dict]) -> dict:
    correct = [row for row in rows if row["correct"]]
    errors = [row for row in rows if not row["correct"]]
    return {
        "total": len(rows), "exact": len(correct),
        "correct_zero": sum(row["zero_class"] for row in correct),
        "correct_nonzero": sum(not row["zero_class"] for row in correct),
        "error_zero": sum(row["zero_class"] for row in errors),
        "error_nonzero": sum(not row["zero_class"] for row in errors),
        "error_auroc_cycle_l1": SheafConsistencyMonitor.compute_binding_auroc(
            [sum(map(abs, r["cycle_sum"])) for r in correct],
            [sum(map(abs, r["cycle_sum"])) for r in errors]
        ) if correct and errors else None,
    }


def controls(atoms: list[Example], heldout: list[Example]) -> dict:
    """Analytic negative controls are kept separate from model observations."""
    results = {name: [] for name in ("oracle", "reverse", "relabel", "delete", "wrong_subcommand")}
    for item in heldout:
        left, right = ([a for a in atoms[i].target if a != BLANK] for i in item.parts)
        gold = left + right
        cases = {
            "oracle": (left, right, gold), "reverse": (left, right, right + left),
            "relabel": tuple([(a + 1) % len(ACTIONS) for a in seq] for seq in (left, right, gold)),
            "delete": (left, right, gold[:-1]), "wrong_subcommand": (left[1:], right, gold),
        }
        for name, (first, second, whole) in cases.items():
            imbalance = cycle_sum(action_counts(first), action_counts(second), action_counts(whole))
            results[name].append({"correct": whole == gold, "zero_class": not any(imbalance), "cycle_sum": imbalance})
    return {key: summary(rows) for key, rows in results.items()}


def seed_inference(cells: list[dict], budget: int) -> dict:
    chosen = {arm: {c["seed"]: c for c in cells if c["step"] == budget and c["arm"] == arm}
              for arm in ("plain", "sheaf", "sheaf_disabled_at_eval")}
    difference = [chosen["sheaf"][s]["exact"] - chosen["plain"][s]["exact"] for s in SEEDS]
    n = chosen["plain"][SEEDS[0]]["total"]
    observed = abs(sum(difference))
    p = mean(abs(sum(x*y for x, y in zip(difference, signs, strict=True))) >= observed
             for signs in product((-1, 1), repeat=len(difference)))
    deltas = np.asarray(difference, dtype=np.float64) / n
    low, high = bootstrap_ci(deltas, n_boot=10000, seed=3101)
    arms = {}
    for name, runs in chosen.items():
        scores = [r["exact"] / r["total"] for r in runs.values()]
        counts = {k: sum(r[k] for r in runs.values()) for k in
                  ("correct_zero", "correct_nonzero", "error_zero", "error_nonzero")}
        aucs = [r["error_auroc_cycle_l1"] for r in runs.values() if r["error_auroc_cycle_l1"] is not None]
        arms[name] = {"mean_exact_match": mean(scores), "seed_bootstrap_95": bootstrap_ci(np.asarray(scores), seed=3101),
                      "contingency": counts, "auc_eligible_seeds": len(aucs),
                      "mean_error_auroc": mean(aucs) if aucs else None,
                      "auc_seed_bootstrap_95": bootstrap_ci(np.asarray(aucs), seed=3101) if aucs else None,
                      "perfect_error_certificate": len(aucs) == len(SEEDS)
                      and counts["error_zero"] == 0 and counts["correct_nonzero"] == 0}
    return {"arms": arms, "paired_exact_match_delta": mean(difference) / n,
            "seed_deltas_in_correct_prompts": difference, "paired_bootstrap_95": [low, high],
            "paired_t_p": paired_t_test(deltas)[1], "wilcoxon_p": wilcoxon_signed_rank(deltas),
            "two_sided_paired_sign_flip_p": p, "superiority": low > 0 and p < .05}


def run(summary_path: Path, raw_path: Path, registry: Path | None) -> dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    atoms, train, heldout = corpus()
    x, y = _inputs(train), torch.tensor([e.target for e in train])
    cells, raw = [], []
    parameter_count = 0
    for seed in SEEDS:
        generator = torch.Generator().manual_seed(10000 + seed)
        batches = torch.randint(len(train), (max(BUDGETS), 32), generator=generator)
        initial_hash = None
        for arm in ("plain", "sheaf"):
            torch.manual_seed(seed)
            model = Transducer(diffusion=arm == "sheaf")
            initial = _state_hash(model)
            if initial_hash is not None and initial_hash != initial:
                raise RuntimeError("paired models have different initial states")
            initial_hash = initial
            parameter_count = sum(p.numel() for p in model.parameters())
            optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
            for step, batch in enumerate(batches, 1):
                model.train()
                optimizer.zero_grad()
                logits = model(x[batch])
                loss = nn.functional.cross_entropy(logits.flatten(0, 1), y[batch].flatten())
                loss.backward()
                optimizer.step()
                if step not in BUDGETS:
                    continue
                model.eval()
                state = _state_hash(model)
                eval_arms = (arm,) if arm == "plain" else (arm, "sheaf_disabled_at_eval")
                for eval_arm in eval_arms:
                    model.diffusion.enabled = eval_arm == "sheaf"
                    rows = observations(model, atoms, heldout)
                    with torch.no_grad():
                        train_exact = int((model(x).argmax(-1) == y).all(-1).sum())
                    key = {"seed": seed, "step": step, "arm": eval_arm}
                    cells.append({**key, **summary(rows), "train_exact": train_exact,
                                  "init_sha256": initial, "state_sha256": state,
                                  "batch_schedule_sha256": _hash(batches.numpy().tobytes())})
                    raw.extend({**key, **r} for r in rows)
                model.diffusion.enabled = arm == "sheaf"

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_bytes = b"".join(_json(row) + b"\n" for row in raw)
    compressed = gzip.compress(raw_bytes, mtime=0)
    raw_path.write_bytes(compressed)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    report = {
        "study": "synthetic SCAN-style held-out primitive pairs", "source_base_revision": revision,
        "source_sha256": {str(p.relative_to(ROOT)): _hash(p.read_bytes()) for p in
                          (Path(__file__).resolve(), ROOT / "bio_inspired_nanochat/sheaf_binding.py")},
        "torch": str(torch.__version__), "python": platform.python_version(), "device": "cpu",
        "cpu_threads": 1, "seeds": SEEDS, "budgets": BUDGETS, "primary_step": max(BUDGETS),
        "paired_parameter_count": parameter_count, "optimizer": {"name":"AdamW", "lr": .003, "weight_decay": .01, "batch": 32},
        "action_vocabulary": [*ACTIONS, "blank"], "input_vocabulary": WORDS,
        "train_commands": len(train), "test_commands": len(heldout),
        "training_data_sha256": _hash(_json([asdict(e) for e in train])),
        "heldout_data_sha256": _hash(_json([asdict(e) for e in heldout])),
        "heldout_unordered_pairs": sorted(sorted(pair) for pair in HOLDOUT),
        "h1_dimension": len(ACTIONS), "raw_rows": len(raw), "raw_file": raw_path.name,
        "raw_sha256": _hash(compressed), "uncompressed_sha256": _hash(raw_bytes),
        "controls": controls(atoms, heldout), "cells": cells,
        "comparisons": {str(b): seed_inference(cells, b) for b in BUDGETS},
    }
    primary = report["comparisons"][str(max(BUDGETS))]
    report["passed_joint_endpoint"] = bool(primary["superiority"] and primary["arms"]["sheaf"]["perfect_error_certificate"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if registry is not None:
        append_record(RunRecord(
            run_id="sheaf-compositional-" + _hash(raw_bytes)[:16], harness="eval",
            metrics={"eval_accuracy": primary["arms"]["sheaf"]["mean_exact_match"]}, git_sha=revision,
            config_hash=_hash(_json({k: report[k] for k in (
                "source_sha256", "seeds", "budgets", "optimizer", "training_data_sha256",
                "heldout_data_sha256", "torch", "cpu_threads"
            )})), timestamp=time.time(),
            hardware="CPU, one thread", dataset_shards=[report["training_data_sha256"], report["heldout_data_sha256"]],
            notes=f"Synthetic systematic split, aggregate over {len(SEEDS)} seeds. {summary_path.name}",
            verdict="positive" if report["passed_joint_endpoint"] else "null", eligible_for_best=False,
        ), str(registry))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--registry", type=Path)
    args = parser.parse_args()
    result = run(args.summary, args.raw, args.registry)
    Console().print(result["comparisons"])
