"""Measure error detection on real outputs from GPTs trained on synthetic variable binding."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console

from bio_inspired_nanochat.eval_stats import bootstrap_ci, holm_adjust, paired_comparison
from bio_inspired_nanochat.gpt import GPT, GPTConfig
from bio_inspired_nanochat.sheaf_detector import SheafHallucinationDetector
from bio_inspired_nanochat.synthetic_tasks import variable_binding


METHODS = ("obstruction", "entropy", "self_consistency", "cosine", "random")


class ScoredSplit(TypedDict):
    data_seed: int
    sampling_seed: int
    rows: list[dict]


class PairedComparison(TypedDict):
    pairs: int
    obstruction_minus_baseline: float
    paired_bootstrap_95: list[float]
    t_p: float
    wilcoxon_p: float
    seeds_favorable: int
    descriptive_tpr_difference: dict | None
    holm_t_p: float
    holm_wilcoxon_p: float


@dataclass(frozen=True)
class SheafEvalConfig:
    seeds: tuple[int, ...] = (11, 23, 37, 53, 71, 89, 107, 127)
    samples_per_seed: int = 64  # Separate calibration and held-out sets of this size.
    train_steps: int = 64
    train_batch: int = 32
    d_model: int = 32
    vocab_size: int = 36
    fpr_target: float = 0.05


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _weights(model: GPT) -> str:
    digest = hashlib.sha256()
    for name, p in model.named_parameters():
        digest.update(name.encode())
        digest.update(p.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _ids(inputs: torch.Tensor) -> list[str]:
    return [_sha(json.dumps(row, separators=(",", ":")).encode()) for row in inputs.tolist()]


def _batch(n: int, vocab: int, seed: int):
    return variable_binding(batch=n, num_vars=3, num_distractors=4, vocab_size=vocab, seed=seed)


@torch.inference_mode()
def _score(model: GPT, detector: SheafHallucinationDetector, inputs: torch.Tensor,
           gold: torch.Tensor, seed: int) -> list[dict]:
    hidden = model.get_hidden_states(inputs)
    logits = model.hidden_to_logits(hidden[:, -1]).float()
    log_probs = logits.log_softmax(-1)
    probabilities = log_probs.exp()
    predictions = logits.argmax(-1)
    # Only input positions determine this relation (the query key and its earlier occurrence).
    matches = inputs[:, :-2] == inputs[:, -1:]
    if not torch.all(matches.sum(-1) == 1):
        raise ValueError("variable-binding input must contain one earlier occurrence of its query key")
    positions = matches.long().argmax(-1)
    indices = torch.arange(inputs.size(0))
    antecedents = hidden[indices, positions]
    queries = hidden[:, -1]
    laplacian = hidden.new_tensor([[1, -1], [-1, 1]])
    entropy = -(probabilities * log_probs).sum(-1)
    cosine = 1 - F.cosine_similarity(antecedents, queries, dim=-1)
    draws = torch.multinomial(probabilities, 8, replacement=True, generator=torch.Generator().manual_seed(seed))
    random_scores = torch.rand(inputs.size(0), generator=torch.Generator().manual_seed(seed + 1))
    rows = []
    for i, source_id in enumerate(_ids(inputs)):
        binding = torch.stack([antecedents[i], queries[i]])
        scores = {"obstruction": detector.compute_obstruction_score(binding, laplacian),
                  "entropy": float(entropy[i]), "cosine": float(cosine[i]), "random": float(random_scores[i]),
                  "self_consistency": float((draws[i] != predictions[i]).float().mean())}
        rows.append({"source_id": source_id, "input_ids": inputs[i].tolist(), "answer": int(gold[i]),
                     "prediction": int(predictions[i]), "sampled_answers": draws[i].tolist(), "scores": scores})
    return rows


def _metrics(cal: list[dict], held: list[dict], method: str, target_fpr: float) -> dict:
    negatives = sorted((r["scores"][method] for r in cal if r["answer"] == r["prediction"]), reverse=True)
    # Strictly exceed the (k+1)-th largest calibration negative; ties remain unflagged.
    threshold = negatives[math.floor(target_fpr * len(negatives))] if negatives else None
    scores = np.asarray([r["scores"][method] for r in held])
    error = np.asarray([r["answer"] != r["prediction"] for r in held], dtype=bool)
    pos, neg = scores[error], scores[~error]
    auroc = None
    if len(pos) and len(neg):
        pairs = pos[:, None] - neg[None, :]
        auroc = float(np.mean((pairs > 0) + 0.5 * (pairs == 0)))
    flagged = scores > threshold if threshold is not None else None
    return {"auroc": auroc, "calibration_correct": len(negatives), "threshold": threshold,
            "calibration_fpr": None if threshold is None else sum(s > threshold for s in negatives) / len(negatives),
            "errors": len(pos), "correct": len(neg), "flagged": None if flagged is None else int(flagged.sum()),
            "true_positives": None if flagged is None else int(flagged[error].sum()),
            "false_positives": None if flagged is None else int(flagged[~error].sum()),
            "coverage_flagged": None if flagged is None else float(flagged.mean()),
            "tpr": None if flagged is None or not len(pos) else float(flagged[error].mean()),
            "fpr": None if flagged is None or not len(neg) else float(flagged[~error].mean())}


def _train_and_score(config: SheafEvalConfig, seed: int) -> dict:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = GPT(GPTConfig(sequence_len=32, vocab_size=config.vocab_size, n_layer=2, n_head=2,
                              n_kv_head=2, n_embd=config.d_model, init_seed=seed))
        model.init_weights()
        initial = _weights(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        generator = np.random.default_rng(seed)
        seen = set()
        losses = []
        for _ in range(config.train_steps):
            batch = _batch(config.train_batch, config.vocab_size, int(generator.integers(2**63 - 1)))
            seen.update(_ids(batch.inputs))
            logits = model(batch.inputs)[:, -1]
            loss = F.cross_entropy(logits, batch.targets[:, -1])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        fitted = _weights(model)
        if fitted == initial or not all(map(math.isfinite, losses)):
            raise ValueError("training did not produce changed weights and finite losses")
        model.eval()
        detector = SheafHallucinationDetector(d_model=config.d_model).eval()
        splits: dict[str, ScoredSplit] = {}
        for fold in ("calibration", "heldout"):
            data_seed = int(generator.integers(2**63 - 1))
            sample_seed = int(generator.integers(2**63 - 1))
            batch = _batch(config.samples_per_seed, config.vocab_size, data_seed)
            rows = _score(model, detector, batch.inputs, batch.targets[:, -1], sample_seed)
            ids = {r["source_id"] for r in rows}
            if len(ids) != len(rows) or seen & ids:
                raise ValueError(f"{fold} contains a duplicate or reused input")
            seen.update(ids)
            splits[fold] = {"data_seed": data_seed, "sampling_seed": sample_seed, "rows": rows}
        metrics = {m: _metrics(splits["calibration"]["rows"], splits["heldout"]["rows"], m, config.fpr_target)
                   for m in METHODS}
        return {"seed": seed, "initial_weight_sha256": initial, "trained_weight_sha256": fitted,
                "first_batch_loss": losses[0], "last_batch_loss": losses[-1], "metrics": metrics, **splits}


def _summarize(runs: Sequence[Mapping]) -> dict:
    def seed_values(method: str, metric: str) -> dict[int, float]:
        return {r["seed"]: r["metrics"][method][metric] for r in runs
                if r["metrics"][method][metric] is not None}

    means = {}
    for method in METHODS:
        means[method] = {}
        for key in ("auroc", "tpr", "fpr", "coverage_flagged"):
            valid = list(seed_values(method, key).values())
            means[method][key] = {"mean": float(np.mean(valid)) if valid else None, "seeds": len(valid)}
    paired_aucs = {}
    obstruction_auc = seed_values("obstruction", "auroc")
    obstruction_tpr = seed_values("obstruction", "tpr")
    for name in METHODS[1:]:
        paired = paired_comparison(obstruction_auc, seed_values(name, "auroc"), lower_is_better=False)
        if paired is not None:
            paired_aucs[name] = paired
    adjusted_t = holm_adjust({name: p.t_p_value for name, p in paired_aucs.items()})
    adjusted_w = holm_adjust({name: p.wilcoxon_p_value for name, p in paired_aucs.items()})
    comparisons: dict[str, PairedComparison] = {}
    for name, paired in paired_aucs.items():
        tpr = paired_comparison(obstruction_tpr, seed_values(name, "tpr"), lower_is_better=False)
        comparisons[name] = {
            "pairs": paired.n_pairs, "obstruction_minus_baseline": paired.mean_delta,
            "paired_bootstrap_95": [paired.delta_ci_low, paired.delta_ci_high],
            "t_p": paired.t_p_value, "wilcoxon_p": paired.wilcoxon_p_value, "seeds_favorable": paired.n_favorable,
            "descriptive_tpr_difference": None if tpr is None else {
                "pairs": tpr.n_pairs, "mean": tpr.mean_delta, "paired_bootstrap_95": [tpr.delta_ci_low, tpr.delta_ci_high]},
            "holm_t_p": adjusted_t[name], "holm_wilcoxon_p": adjusted_w[name]}
    calibrated = all(r["metrics"][m]["tpr"] is not None for r in runs for m in METHODS)
    complete = (len(runs) >= 8 and calibrated and len(obstruction_auc) == len(runs)
                and len(comparisons) == len(METHODS) - 1)
    if not complete:
        verdict = "inconclusive_missing_classes_or_replications"
    elif means["obstruction"]["auroc"]["mean"] >= 0.75 and all(
            r["paired_bootstrap_95"][0] > 0 and r["holm_t_p"] <= 0.05 and r["holm_wilcoxon_p"] <= 0.05
            for r in comparisons.values()):
        verdict = "superiority_on_synthetic_binding_control"
    elif any(r["obstruction_minus_baseline"] <= 0 for r in comparisons.values()):
        verdict = "unfavorable_observed_comparison"
    else:
        verdict = "inconclusive_superiority_thresholds_unmet"
    return {"mean_by_seed": means, "paired_comparisons": comparisons, "verdict": verdict,
            "passed_statistical_advantage": verdict == "superiority_on_synthetic_binding_control"}


def _operating_points(report: dict) -> dict:
    """Verify per-seed results against recorded queries before deriving pooled FPR counts."""
    runs = report["runs"]
    operating_points = {}
    for method in METHODS:
        false_positives = 0
        correct = 0
        fprs = []
        for run in runs:
            measured = _metrics(run["calibration"]["rows"], run["heldout"]["rows"], method, report["config"]["fpr_target"])
            if measured != run["metrics"][method]:
                raise ValueError(f"stored metrics disagree with {method} scores for seed {run['seed']}")
            if measured["fpr"] is not None:
                false_positives += measured["false_positives"]
                correct += measured["correct"]
                fprs.append(measured["fpr"])
        operating_points[method] = {
            "heldout_false_positives": false_positives, "correct_heldout_answers_with_threshold": correct,
            "contributing_seeds": len(fprs), "pooled_fpr": false_positives / correct if correct else None,
            "mean_seed_fpr_bootstrap_95": list(bootstrap_ci(np.asarray(fprs), seed=0)) if len(fprs) >= 2 else None}
    vocab = report["config"]["vocab_size"]
    content = vocab - 2
    third = content // 3
    return {"heldout_fpr": operating_points, "fpr_uncertainty": "The interval resamples training seeds with replacement (10,000 draws) for the equally weighted mean held-out FPR. Pooled FP/correct counts weight queries equally and have a different estimand.",
            "query_distribution": (
                f"Vocabulary {vocab}. Three distinct variable IDs are sampled uniformly without replacement from 0..{third-1}; "
                f"three distinct values from {third}..{2*third-1} are paired in order. Four distractors are sampled "
                f"with replacement from {2*third}..{content-1}. The seven items (three adjacent variable/value pairs and four "
                f"singletons) are shuffled, followed by QUERY ({content}) and one uniformly selected bound variable. "
                "The gold is the paired value. Each input has 12 tokens. Training, calibration and tests draw from this same "
                "generator. Held-out prompts are checked against training and calibration prompts within each seed.")}


def run_sheaf_detector_evaluation(config: SheafEvalConfig) -> dict:
    if not config.seeds or len(set(config.seeds)) != len(config.seeds) or min(config.train_steps, config.train_batch, config.samples_per_seed) < 1:
        raise ValueError("distinct seeds, training and evaluation samples required")
    if not 0 < config.fpr_target < 1:
        raise ValueError("target FPR must be strictly between 0 and 1")
    runs = []
    for seed in config.seeds:
        runs.append(_train_and_score(config, seed))
        Console().print(f"Scored trained seed {seed} on {config.samples_per_seed} held-out queries.")
    from bio_inspired_nanochat import eval_stats, gpt, sheaf_binding, sheaf_detector, synthetic_tasks
    sources = [Path(__file__), *(Path(m.__file__) for m in (eval_stats, gpt, sheaf_binding, sheaf_detector, synthetic_tasks))]
    report = {"config": asdict(config), "benchmark": "Synthetic variable binding with three variables and four distractors",
            "recipe": "CPU vanilla GPT, two layers, two heads, AdamW lr 0.003, cross entropy on answer token",
            "scope": "Error detection for one-token greedy answers on independently generated binding queries from the training distribution. No natural-language factuality labels.",
            "scores": "Higher scores predict greedy errors. Identity-edge normalized Dirichlet energy and cosine use final states of the two visible occurrences of the query key; entropy uses the entire next-token distribution; self_consistency is disagreement of eight independent next-token draws with greedy.",
            "graph_limit": "For equal-norm features at the two endpoints, normalized identity-edge energy equals half the cosine dissimilarity. The GPT final RMS normalization makes these rankings nearly equivalent.",
            "uncertainty": "Training seeds are the paired unit. Percentile bootstrap of the mean AUROC difference (10,000 resamples, seed 0); two-sided paired t and Wilcoxon tests, each Holm-corrected across four baselines. Thresholds fit only on independent calibration negatives; held-out TPR, FPR and flagged fraction reported.",
            "registered_rule": "At least eight complete seeds, mean obstruction AUROC >= 0.75 and all four lower paired 95% limits > 0, Holm t and Holm Wilcoxon p <= 0.05.",
            "source_sha256": {p.name: _sha(p.read_bytes()) for p in sources}, "torch_version": str(torch.__version__),
            "runs": runs, **_summarize(runs)}
    return {**report, **_operating_points(report)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=64, help="Number of queries in each calibration and held-out split per seed")
    parser.add_argument("--output", type=Path, help="Retain per-query predictions and scores as JSON")
    parser.add_argument("--summarize-existing", type=Path, help="Verify saved scores and derive operating points without training")
    args = parser.parse_args()
    if args.summarize_existing:
        report = json.loads(args.summarize_existing.read_text(encoding="utf-8"))
        for key, value in _summarize(report["runs"]).items():
            if report[key] != value:
                raise ValueError(f"saved {key} differs from the recomputed decision")
        report.update(_operating_points(report))
        report["summary_source_sha256"] = _sha(Path(__file__).read_bytes())
    else:
        report = run_sheaf_detector_evaluation(SheafEvalConfig(samples_per_seed=args.samples))
    Console().print(report["mean_by_seed"], report["verdict"])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, allow_nan=False, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
