from __future__ import annotations

import math
import json
import sys

import numpy as np

from bio_inspired_nanochat.results_registry import read_records
from scripts.tune_bio_params import CandidateEvalResult, _lce_predict_from_points


def test_lce_predict_from_points_recovers_powerlaw() -> None:
    a = 2.0
    b = 10.0
    exponent = 0.5

    points = []
    for step in range(1, 101):
        loss = a + b * (step ** (-exponent))
        points.append((step, loss))

    pred = _lce_predict_from_points(points[-50:], target_step=400, exponent=exponent)
    assert pred is not None

    expected = a + b * (400 ** (-exponent))
    assert math.isfinite(pred)
    assert abs(pred - expected) < 1e-6


def test_lce_predict_from_points_rejects_increasing_curve() -> None:
    points = [(step, 1.0 + 0.1 * step) for step in range(1, 20)]
    pred = _lce_predict_from_points(points, target_step=40, exponent=0.5)
    assert pred is None


def test_lce_predict_from_points_requires_valid_exponent() -> None:
    points = [(1, 1.0), (2, 0.9), (3, 0.85), (4, 0.83)]
    assert _lce_predict_from_points(points, target_step=10, exponent=0.0) is None
    assert _lce_predict_from_points(points, target_step=10, exponent=-0.5) is None


def test_lce_predict_from_points_requires_enough_points() -> None:
    points = [(1, 1.0), (2, 0.9), (3, 0.85)]
    assert _lce_predict_from_points(points, target_step=10, exponent=0.5) is None


def test_optimize_emits_registry_record_joined_to_progress(tmp_path, monkeypatch) -> None:
    import scripts.tune_bio_params as tune

    def fake_evaluate(solution_vector, **_kwargs):
        objective = float(np.square(np.asarray(solution_vector, dtype=np.float64)).sum())
        return CandidateEvalResult(mean_last_loss=objective, steps_run=1)

    run_dir = tmp_path / "run"
    registry_path = tmp_path / "registry.jsonl"
    monkeypatch.setattr(tune, "evaluate_candidate_detailed", fake_evaluate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tune_bio_params",
            "optimize",
            "--device",
            "cpu",
            "--seed",
            "17",
            "--generations",
            "1",
            "--popsize",
            "4",
            "--steps",
            "1",
            "--run-dir",
            str(run_dir),
            "--registry-path",
            str(registry_path),
            "--no-checkpoints",
            "--no-tensorboard",
            "--stagnation-action",
            "none",
        ],
    )

    assert tune.main() == 0

    records = read_records(str(registry_path))
    assert len(records) == 1
    record = records[0]
    assert record.harness == "tune"
    assert record.seed == 17
    assert record.git_sha and record.config_hash
    assert record.metrics["tune_generation"] == 1.0
    assert math.isfinite(record.metrics["tune_objective"])

    progress = [
        json.loads(line)
        for line in (run_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    best_params = json.loads((run_dir / "best_params.json").read_text(encoding="utf-8"))
    assert {row["run_id"] for row in progress} == {record.run_id}
    assert best_params["run_id"] == record.run_id
