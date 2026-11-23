"""Tests for CMA-ES tuning loop E2E harness (bead eqyk.7).

Verifies the full CMA-ES optimization lifecycle in CI time:
  1. 2-generation optimization on the synthetic associative recall task.
  2. Structured artifacts produced: progress.jsonl, best_params.json, inert JSON replay states.
  3. Resume contract: resuming from checkpoint continues without loss of state.
  4. Stagnation policy: triggers early-stopping or sigma-reset under stalled loss.
  5. Detailed per-generation logging in events.jsonl.

Run:
    pytest tests/test_e2e_cmaes.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.e2e.cmaes_tune import CmaesE2EConfig, main as e2e_main, run_cmaes_e2e

pytestmark = pytest.mark.e2e

EXPECTED_INVARIANTS = (
    "initial_optimization_exits_clean",
    "progress_jsonl_written",
    "best_params_json_written",
    "checkpoints_written",
    "checkpoint_resume_contract",
    "stagnation_policy_fires",
    "results_registry_appended",
)


def test_cmaes_e2e_full_battery(tmp_path: Path):
    """The entire CMA-ES battery runs end-to-end and every invariant holds."""
    cfg = CmaesE2EConfig(
        generations=2,
        popsize=4,
        steps=2,
        batch_size=4,
        seed=1337,
        device="cpu",
    )
    report = run_cmaes_e2e(cfg, run_dir=tmp_path, verbose=False)
    report.assert_passed()

    inv_map = {inv.name: inv for inv in report.invariants}
    for name in EXPECTED_INVARIANTS:
        assert name in inv_map, f"Missing invariant: {name}"
        assert inv_map[name].passed, f"Invariant {name} failed: {inv_map[name].detail}"

    # Verify events.jsonl was logged
    events_path = tmp_path / "events.jsonl"
    assert events_path.exists(), "events.jsonl trace must be created"
    events = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        line_s = line.strip()
        if line_s:
            try:
                events.append(json.loads(line_s))
            except json.JSONDecodeError:
                pass
    assert len(events) >= len(EXPECTED_INVARIANTS) + 1


def test_cmaes_e2e_cli_entrypoint(tmp_path: Path):
    """The main entrypoint can be executed directly with CLI arguments."""
    ret = e2e_main(["--run-dir", str(tmp_path), "--device", "cpu", "--seed", "42"])
    assert ret == 0, f"CLI entrypoint exited with non-zero code {ret}"


def _fixed_population(rank, world_size, work):
    import argparse
    from datetime import timedelta
    import math
    import os
    import torch
    import torch.distributed as dist
    from scripts import tune_bio_params as tune

    torch.set_num_threads(1)
    parallel = world_size > 1
    if parallel:
        os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size))
        dist.init_process_group("gloo", init_method=(work / "rendezvous").as_uri(),
                                rank=rank, world_size=world_size, timeout=timedelta(seconds=90))
    info = tune.DistInfo(parallel, rank, rank, world_size, "gloo" if parallel else "", "cpu", torch.device("cpu"))
    args = argparse.Namespace(seed=729, batch_size=1, lr=0.01, weight_decay=0., retries=0,
                              lce_exponent=.5, lce_tail_points=4, lce_stride=1, timeout_seconds=None,
                              record_weight_digests=True)
    spec = tune.TOP10_PARAM_SPECS[0]
    # Byte-tokenized text, with distinct deterministic train and held-out streams.
    task = tune.LMTask(torch.tensor(list(b"apples ripen in autumn. " * 4)),
                       torch.tensor(list(b"winter follows autumn. ")), torch.ones(256, dtype=torch.long), 256, 8)
    cfg = tune.GPTSynapticConfig(n_layer=1, n_head=1, n_kv_head=1, n_embd=16)
    vectors = torch.tensor([[math.log(2)], [math.log(6)], [math.log(20)]], dtype=torch.float64)
    populations = [vectors]
    if parallel:
        # Change every healthy candidate's position, swap rank ownership, and add a
        # finite encoded value whose exponential overflows during hyperparameter decoding.
        populations.extend([vectors[[1, 2, 0]], torch.cat((vectors[[2]], torch.tensor([[1000.]], dtype=vectors.dtype), vectors[:2]))])
    try:
        for idx, population in enumerate(populations):
            scores, records = tune._evaluate_population(
                population, args=args, specs=(spec,), dist_info=info, steps=3,
                lce_target=None, lm_task=task, model_config=cfg,
            )
            if rank == 0:
                (work / f"population-{world_size}-{idx}.json").write_text(
                    json.dumps({"scores": scores, "records": records}, allow_nan=False), encoding="utf-8",
                )
    finally:
        if parallel:
            dist.destroy_process_group()


def test_distributed_lm_population_matches_serial_after_reordering_and_failure(tmp_path):
    import multiprocessing
    import time
    from rich.console import Console
    from scripts.tune_bio_params import PENALTY_LOSS

    _fixed_population(0, 1, tmp_path)
    ctx = multiprocessing.get_context("spawn")
    processes = [ctx.Process(target=_fixed_population, args=(rank, 2, tmp_path)) for rank in range(2)]
    for process in processes:
        process.start()
    deadline = time.monotonic() + 240
    try:
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        assert [p.exitcode for p in processes] == [0, 0], "both population ranks must finish"
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(10)
    reference = json.loads((tmp_path / "population-1-0.json").read_text())["records"]
    assert all(r["error"] is None for r in reference), [(r["index"], r["error"]) for r in reference]
    assert len({r["final_weights_sha256"] for r in reference}) == 3, "solutions must train different weights"
    expected = {r["candidate_sha256"]: (r["score"], r["final_weights_sha256"]) for r in reference}
    assert all(r["error"] is None and r["score"] > 0 and r["optimizer_names"] == ["AdamW", "Muon"] for r in reference)
    for idx, size in enumerate((3, 3, 4)):
        report = json.loads((tmp_path / f"population-2-{idx}.json").read_text())
        records = report["records"]
        assert len(records) == size
        healthy = {r["candidate_sha256"]: (r["score"], r["final_weights_sha256"]) for r in records if r["error"] is None}
        assert healthy == expected, "CPU scores and all final parameter bytes must match serial execution exactly"
        for r in records:
            assert (r["group_size"], r["group_backend"], r["rank"]) == (2, "gloo", r["index"] % 2)
            if r["error"] is None:
                assert r["optimizer_names"] == ["AdamW", "Muon"] and r["steps_run"] == 3
            else:
                assert idx == 2 and "OverflowError" in r["error"] and r["steps_run"] == 0
                assert report["scores"][r["index"]] == PENALTY_LOSS
        Console().print_json(data=report)


def test_torchrun_lm_cma_replays_serial_population(tmp_path):
    """The launcher, rank-0 controller, worker loop and on-disk replay must agree."""
    import os
    import subprocess
    import sys
    import pyarrow as pa
    import pyarrow.parquet as pq
    import tiktoken
    import torch
    from bio_inspired_nanochat.tokenizer import RustBPETokenizer

    shards = tmp_path / "base_data"
    shards.mkdir()
    for i, text in enumerate(("apples ripen in autumn. ", "some birds nest in spring. ")):
        pq.write_table(pa.table({"text": [text * 4]}), shards / f"shard_{i:05d}.parquet")
    encoding = tiktoken.Encoding(name="bytes", pat_str="(?s).",
                                 mergeable_ranks={bytes([i]): i for i in range(256)}, special_tokens={"<|bos|>": 256})
    RustBPETokenizer(encoding, "<|bos|>").save(str(tmp_path / "tokenizer"))
    torch.save(torch.tensor([1] * 256 + [0]), tmp_path / "tokenizer" / "token_bytes.pt")
    env = {**os.environ, "NANOCHAT_BASE_DIR": str(tmp_path), "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "RAYON_NUM_THREADS": "1"}
    cli = ["-m", "scripts.tune_bio_params", "optimize", "--objective", "lm", "--device", "cpu",
           "--seed", "23", "--steps", "2", "--batch-size", "1", "--popsize", "3", "--generations", "1",
           "--lm-train-tokens", "64", "--lm-eval-tokens", "32", "--lm-seq-len", "8",
           "--record-weight-digests", "--no-tensorboard"]
    rows = {}
    for size in (1, 2):
        directory = tmp_path / f"run-{size}"
        runner = [sys.executable]
        if size == 2:
            runner.extend(["-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2"])
        cmd = [*runner, *cli, "--run-dir", str(directory), "--registry-path", str(tmp_path / f"registry-{size}.jsonl")]
        if size == 2:
            cmd.append("--distributed")
        run = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], env=env,
                             capture_output=True, text=True, timeout=240)
        assert run.returncode == 0, (run.stdout[-1800:], run.stderr[-1800:])
        progress = json.loads((directory / "progress.jsonl").read_text())
        candidates = progress["candidates"]
        assert len(candidates) == 3 and all(c["error"] is None for c in candidates)
        assert [c["rank"] for c in candidates] == [i % size for i in range(3)]
        rows[size] = [(c["candidate_sha256"], c["score"], c["final_weights_sha256"]) for c in candidates]
        replay = json.loads((directory / "es_state.json").read_text())
        assert replay["generation_records"][0]["fitnesses"] == [c["score"] for c in candidates]
    assert rows[1] == rows[2]
