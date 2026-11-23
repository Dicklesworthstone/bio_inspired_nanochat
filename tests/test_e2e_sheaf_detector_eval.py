import hashlib
import json
from typing import TypedDict

import pytest
import torch

from bio_inspired_nanochat.gpt import GPT, GPTConfig
from bio_inspired_nanochat.sheaf_detector import SheafHallucinationDetector
from scripts.e2e.sheaf_detector_eval import METHODS, SheafEvalConfig, _batch, _metrics, _score, _summarize, run_sheaf_detector_evaluation


def test_calibrated_threshold_uses_correct_calibration_and_test_labels_with_ties():
    def row(score, error):
        return {"scores": {"entropy": score}, "answer": 0, "prediction": int(error)}

    calibration = [row(0.1, False), row(0.7, False), row(0.7, False), row(0.99, True)]
    held = [row(0.7, False), row(0.9, True), row(0.7, True), row(0.8, False)]
    metrics = _metrics(calibration, held, "entropy", 0.05)
    assert metrics["threshold"] == 0.7
    assert metrics["tpr"] == metrics["fpr"] == metrics["coverage_flagged"] == 0.5
    assert metrics["auroc"] == pytest.approx(0.625)
    assert _metrics(calibration, held[::-1], "entropy", 0.05) == metrics
    unseen_class = _metrics(calibration, [row(0.4, True)], "entropy", 0.05)
    assert unseen_class["auroc"] is None and unseen_class["fpr"] is None
    uncalibrated = _metrics([row(0.4, True)], held, "entropy", 0.05)
    assert uncalibrated["tpr"] is None and uncalibrated["threshold"] is None


def test_scores_share_actual_model_prediction_and_are_blind_to_gold():
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(42)
        model = GPT(GPTConfig(sequence_len=32, vocab_size=36, n_layer=1, n_head=2, n_kv_head=2, n_embd=32)).eval()
        model.init_weights()
        model.lm_head.weight.normal_(std=0.1)
        detector = SheafHallucinationDetector(32)
        data = _batch(5, 36, 19)
        gold = data.targets[:, -1]
        logits = model(data.inputs)[:, -1]
        actual = torch.distributions.Categorical(logits=logits)
        rows = _score(model, detector, data.inputs, gold, 47)
        changed_answers = _score(model, detector, data.inputs, gold + 1, 47)
    for i, (row, changed) in enumerate(zip(rows, changed_answers, strict=True)):
        assert row["prediction"] == int(logits[i].argmax())
        assert row["scores"]["entropy"] == pytest.approx(float(actual.entropy()[i]), abs=1e-6)
        assert row["scores"]["self_consistency"] == sum(v != row["prediction"] for v in row["sampled_answers"]) / 8
        assert row["scores"] == changed["scores"] and row["answer"] != changed["answer"]
        assert abs(2 * row["scores"]["obstruction"] - row["scores"]["cosine"]) < 2e-5


def test_trained_models_have_measured_joined_labels_disjoint_splits_and_no_two_seed_verdict():
    config = SheafEvalConfig(seeds=(11, 23), samples_per_seed=12, train_steps=3, train_batch=4)
    report = run_sheaf_detector_evaluation(config)
    assert report["verdict"] == "inconclusive_missing_classes_or_replications"
    assert not report["passed_statistical_advantage"]
    assert len({r["trained_weight_sha256"] for r in report["runs"]}) == 2
    for run in report["runs"]:
        assert run["trained_weight_sha256"] != run["initial_weight_sha256"]
        cal, held = run["calibration"]["rows"], run["heldout"]["rows"]
        assert len({r["source_id"] for r in cal + held}) == 24
        for row in held:
            inputs = row["input_ids"]
            assert row["answer"] == inputs[inputs.index(inputs[-1]) + 1]
            assert row["source_id"] == hashlib.sha256(json.dumps(inputs, separators=(",", ":")).encode()).hexdigest()
            assert set(row["scores"]) == set(METHODS)
        for method in METHODS:
            assert run["metrics"][method] == _metrics(cal, held, method, config.fpr_target)
    for m in METHODS:
        eligible = [r["metrics"][m] for r in report["runs"] if r["metrics"][m]["fpr"] is not None]
        assert report["heldout_fpr"][m]["heldout_false_positives"] == sum(r["false_positives"] for r in eligible)
        assert report["heldout_fpr"][m]["correct_heldout_answers_with_threshold"] == sum(r["correct"] for r in eligible)


def test_all_four_paired_comparisons_and_accuracy_floor_are_required():
    class FixtureRun(TypedDict):
        seed: int
        metrics: dict[str, dict[str, float | None]]

    metrics = {"tpr": 0.5, "fpr": 0.05, "coverage_flagged": 0.1}
    runs: list[FixtureRun] = [{"seed": seed, "metrics": {name: {**metrics, "auroc": 0.55 if name != "obstruction" else 0.80 + seed / 100}
                                                       for name in METHODS}} for seed in range(8)]
    assert _summarize(runs)["passed_statistical_advantage"]
    for r in runs:
        r["metrics"]["cosine"]["auroc"] = r["metrics"]["obstruction"]["auroc"]
    result = _summarize(runs)
    assert result["verdict"] == "unfavorable_observed_comparison"
    assert result["paired_comparisons"]["cosine"]["holm_wilcoxon_p"] == 1.0
    runs[2]["metrics"]["obstruction"]["auroc"] = None
    assert _summarize(runs)["verdict"] == "inconclusive_missing_classes_or_replications"
