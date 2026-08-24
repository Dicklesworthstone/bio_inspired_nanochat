"""
CI smoke tests for the three flagship harnesses — bead hm4.3.

The audit found CI runs lint/type/rust/pytest but NO train/eval/HPO smoke, so the flagship
scripts are exercised only by incidental imports and can silently rot. This module runs the
CORE of each harness in-process on tiny/synthetic configs, in seconds:

- base_train  -> a few real forward/backward/optimizer steps on a tiny synaptic model, plus the
                 run-record emission path (hm4.1).
- base_eval   -> evaluate_bpb on a tiny model + synthetic batches.
- tune_bio_params -> the CMA-ES code path (Rosenbrock toy), synthetic data, config encode/decode.

The full script entry points need a tokenizer (rustbpe) + real data; here we exercise their
importable core functions instead, and skip cma-dependent parts if cma is absent.

Run the fast smoke subset:  pytest -m smoke
"""

from __future__ import annotations

import json

import pytest
import torch

from _bio_testkit import make_tiny_synaptic, make_tiny_vanilla, random_tokens


# --------------------------------------------------------------------------- #
# 1. base_train
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.smoke
def test_train_harness_loop_smoke():
    m = make_tiny_synaptic(seed=0, train=True).train()
    opt = torch.optim.SGD(m.parameters(), lr=1e-3)
    for _ in range(5):
        x = random_tokens(2, 16, vocab=m.config.vocab_size)
        _logits, loss = m(x, targets=x)
        assert torch.isfinite(loss), "training step produced a non-finite loss"
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)


@pytest.mark.unit
@pytest.mark.smoke
def test_train_run_record_emission(tmp_path):
    # The base_train end-of-run record shares the telemetry correlation id (hm4.1).
    from bio_inspired_nanochat.results_registry import append_record, make_record, read_records
    from bio_inspired_nanochat.run_logging import TrainingTelemetry

    telemetry_dir = tmp_path / "telemetry"
    registry_path = str(tmp_path / "registry.jsonl")
    with TrainingTelemetry(
        telemetry_dir, tensorboard=False, console=False
    ) as telemetry:
        rec = make_record(
            "train",
            {"val_bpb": 1.5, "smooth_train_loss": 4.2, "mfu": 0.1, "step": 5.0},
            run_id=str(telemetry.run_id),
            config={"model": {"depth": 2}, "training": {"seed": 0}},
            seed=0,
            dataset_shards=["synthetic:train", "synthetic:val"],
            timestamp=1.0,
        )
        append_record(rec, registry_path)

    run_start = json.loads(
        (telemetry_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    persisted = read_records(registry_path)[0]
    assert persisted.run_id == run_start["run_id"]
    assert persisted.metrics["val_bpb"] == 1.5 and persisted.config_hash


# --------------------------------------------------------------------------- #
# 2. base_eval
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.smoke
def test_eval_harness_bpb_smoke():
    from bio_inspired_nanochat.loss_eval import evaluate_bpb

    m = make_tiny_vanilla(seed=0).eval()
    vocab = m.config.vocab_size
    batches = [(random_tokens(2, 16, vocab=vocab), random_tokens(2, 16, vocab=vocab)) for _ in range(4)]
    token_bytes = torch.ones(vocab, dtype=torch.long)  # 1 byte per token (synthetic; integer counts)
    with torch.no_grad():
        bpb = float(evaluate_bpb(m, iter(batches), steps=3, token_bytes=token_bytes))
    assert bpb == bpb and bpb > 0, f"evaluate_bpb must return a finite positive bpb, got {bpb}"


# --------------------------------------------------------------------------- #
# 3. tune_bio_params
# --------------------------------------------------------------------------- #
@pytest.mark.smoke
def test_tune_cmaes_code_path_smoke():
    pytest.importorskip("cma")
    from scripts.tune_bio_params import run_rosenbrock_2d_cmaes

    best_x, best_f = run_rosenbrock_2d_cmaes(seed=1, iterations=8, popsize=4)
    assert best_x.shape == (2,)
    assert best_f >= 0.0, "Rosenbrock is non-negative; the CMA-ES path must run"


@pytest.mark.unit
@pytest.mark.smoke
def test_tune_synthetic_data_and_config_roundtrip():
    from scripts.tune_bio_params import (
        TOP10_PARAM_SPECS,
        _build_synaptic_config,
        decode_params,
        encode_params,
        generate_batch,
    )

    x, y = generate_batch(2, 16, 64, "cpu")
    assert x.shape == (2, 16) and y.shape == (2, 16)
    cfg = _build_synaptic_config({"tau_c": 7.0})
    assert cfg.tau_c == 7.0
    decoded = decode_params(encode_params(cfg, TOP10_PARAM_SPECS), TOP10_PARAM_SPECS)
    assert abs(decoded["tau_c"] - 7.0) < 1e-3, "tune config encode/decode must round-trip"
