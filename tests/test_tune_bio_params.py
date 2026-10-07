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


def _bigram_lm_task(vocab: int = 64, seq_len: int = 32, n_train: int = 8192, n_val: int = 2048):
    """A structured stream (each token's successor is one of two fixed choices), 2 bytes/token."""
    import torch

    from scripts.tune_bio_params import LMTask

    g = torch.Generator().manual_seed(0)
    table = torch.randint(0, vocab, (vocab, 2), generator=g)

    def stream(n: int) -> torch.Tensor:
        out = torch.empty(n, dtype=torch.long)
        out[0] = 0
        choice = torch.randint(0, 2, (n,), generator=g)
        for i in range(1, n):
            out[i] = table[out[i - 1], choice[i]]
        return out

    return LMTask(
        train_tokens=stream(n_train + 1),
        val_tokens=stream(n_val + 1),
        token_bytes=torch.full((vocab,), 2, dtype=torch.int64),
        vocab_size=vocab,
        seq_len=seq_len,
    )


def test_lm_train_batch_is_next_token_windows_cycling_through_the_stream() -> None:
    import torch

    from scripts.tune_bio_params import lm_train_batch

    task = _bigram_lm_task(n_train=320, seq_len=32)  # (321 - 1) // 32 = 10 windows
    x, y = lm_train_batch(task, step=0, batch_size=4)
    assert x.shape == y.shape == (4, 32)
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert torch.equal(x[1], task.train_tokens[32:64])
    x_wrap, _ = lm_train_batch(task, step=2, batch_size=4)  # windows 8, 9, 0, 1
    assert torch.equal(x_wrap[2], x[0]) and torch.equal(x_wrap[3], x[1])


def test_lm_objective_learns_real_structure_deterministically() -> None:
    """idh4: the copy task never trained inside the proxy budget, so its fitness was noise.
    The LM objective must move: 40 steps of the base_train recipe on a learnable stream cut the
    held-out bits/byte well below the untrained model, and a rerun reproduces it exactly."""
    from dataclasses import replace

    from scripts.tune_bio_params import (
        MODEL_CONFIG,
        TOP10_PARAM_SPECS,
        encode_params,
        evaluate_candidate_detailed,
    )
    from bio_inspired_nanochat.synaptic import SynapticConfig

    task = _bigram_lm_task()
    x0 = encode_params(SynapticConfig(), TOP10_PARAM_SPECS)
    kwargs = dict(
        specs=TOP10_PARAM_SPECS, seed=3, batch_size=4, device="cpu", lr=0.0, weight_decay=0.0,
        timeout_seconds=None, max_retries=0, raise_on_error=True, lm_task=task,
        model_config=replace(MODEL_CONFIG, n_layer=1, n_embd=64, n_head=2, n_kv_head=2),
    )
    untrained = evaluate_candidate_detailed(x0, steps=0, **kwargs).objective
    trained = evaluate_candidate_detailed(x0, steps=40, **kwargs).objective
    again = evaluate_candidate_detailed(x0, steps=40, **kwargs).objective
    # Uniform over 64 tokens is 6 bits = 3 bits/byte; the stream carries 1 bit/token = 0.5 bits/byte.
    assert untrained > 2.5, untrained
    assert trained < untrained - 1.0, (untrained, trained)
    assert trained == again
