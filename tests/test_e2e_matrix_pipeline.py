"""The whole D1 pipeline at toy scale: spec -> base_train -> eval_matrix -> eval_stats (bridge plan G1).

Until 2026-09-02 nothing exercised the path a GPU run will take: `scripts.matrix_launch` deriving
each cell's `base_train` command from `ablation_matrix`, `base_train` writing the checkpoint under the
tag `eval_matrix` resolves, `eval_matrix matrix --checkpoint-dir` loading and scoring those
checkpoints on the FineWeb-shaped validation split, and `eval_stats` pairing the columns. This test
runs all of it as subprocesses on two synthetic parquet shards, two columns (the `vanilla` and
`synaptic_off` anchors) and two seeds, so a broken link fails here before it burns GPU hours.

Marked e2e and slow (about ten minutes on a loaded CPU); the nightly validation runs it.

Run:  pytest tests/test_e2e_matrix_pipeline.py -v -m "slow"
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_e2e_quick_start import _write_shards  # noqa: E402

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

REPO = Path(__file__).resolve().parents[1]
COLUMNS = ("vanilla", "synaptic_off")
SEEDS = (1337, 1338)
RECIPE = (
    "--depth=2 --max_seq_len=64 --device_batch_size=2 --total_batch_size=128 --num_iterations=2 "
    "--eval_every=2 --eval_tokens=128 --core_metric_every=-1 --sample_every=-1 --device_type=cpu"
)


def _run(module_args: list[str], env: dict[str, str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", *module_args], cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    base_dir = tmp_path_factory.mktemp("matrix_base")
    _write_shards(base_dir)
    env = {**os.environ, "NANOCHAT_BASE_DIR": str(base_dir), "BIO_RESULTS_REGISTRY": str(base_dir / "registry.jsonl"), "PYTHONWARNINGS": "ignore"}
    tok = _run(["scripts.tok_train", "--max_chars=200000", "--vocab_size=1024"], env, timeout=600)
    assert tok.returncode == 0, tok.stderr[-3000:]

    launch = _run(
        ["scripts.matrix_launch", "--columns", ",".join(COLUMNS), "--seeds", ",".join(map(str, SEEDS)), f"--recipe={RECIPE}", "--execute"],
        env, timeout=2400,
    )
    assert launch.returncode == 0, (launch.stdout[-2000:], launch.stderr[-4000:])

    out_dir = base_dir / "matrix_out"
    template = str(base_dir / "base_checkpoints" / "matrix_{preset}_s{seed}")
    scored = _run(
        [
            "scripts.eval_matrix", "matrix",
            "--presets", ",".join(COLUMNS), "--seeds", ",".join(map(str, SEEDS)),
            "--checkpoint-dir", template, "--data", "fineweb", "--device-type", "cpu",
            "--eval-bpb", "--eval-tokens", "128", "--device-batch-size", "2",
            "--niah-lengths", "8", "--continual-exposures", "2", "--ece-bins", "5",
            "--out-dir", str(out_dir), "--batch-id", "pipeline", "--registry-path", str(base_dir / "eval_registry.jsonl"),
        ],
        env, timeout=2400,
    )
    assert scored.returncode == 0, (scored.stdout[-2000:], scored.stderr[-4000:])
    summary = next(out_dir.rglob("summary.csv"))
    stats = _run(["bio_inspired_nanochat.eval_stats", str(summary), "--min-pairs", "2", "--json-out", str(base_dir / "verdict.json")], env, timeout=600)
    assert stats.returncode == 0, stats.stderr[-3000:]
    return {"base_dir": base_dir, "launch": launch, "summary": summary, "verdict": base_dir / "verdict.json"}


def test_launcher_trained_every_cell_under_the_tag_eval_matrix_resolves(pipeline):
    for column in COLUMNS:
        for seed in SEEDS:
            ckpt = pipeline["base_dir"] / "base_checkpoints" / f"matrix_{column}_s{seed}"
            assert (ckpt / "model_000002.pt").exists(), sorted(p.name for p in ckpt.iterdir()) if ckpt.exists() else f"missing {ckpt}"
            meta = json.loads((ckpt / "meta_000002.json").read_text(encoding="utf-8"))
            assert int(meta["model_config"]["init_seed"]) == seed
            assert bool(meta.get("synapses", False)) == (column != "vanilla")


def test_eval_matrix_scored_every_checkpoint_from_its_own_metadata(pipeline):
    rows = list(csv.DictReader(pipeline["summary"].open(encoding="utf-8")))
    cells = {(r["preset"], int(r["seed"])) for r in rows}
    assert cells == {(c, s) for c in COLUMNS for s in SEEDS}, cells
    for r in rows:
        assert math.isfinite(float(r["val_bpb"])), r
        assert r["recipe_source"] == "base_train_checkpoint", r["recipe_source"]


def test_eval_stats_pairs_the_synaptic_off_anchor_against_vanilla(pipeline):
    verdict = json.loads(pipeline["verdict"].read_text(encoding="utf-8"))
    presets = verdict.get("presets") or verdict
    assert "synaptic_off" in presets, list(presets)[:5]
    entry = presets["synaptic_off"]
    assert "paired_vs_baseline" in entry and entry["paired_vs_baseline"]["n_pairs"] == len(SEEDS)
