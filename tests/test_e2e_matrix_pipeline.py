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
import shlex
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


def _score(launch, env):
    argv = shlex.split(launch.stdout.splitlines()[-1])
    assert argv[:4] == [sys.executable, "-m", "scripts.eval_matrix", "matrix"]
    assert "--eval-bpb" in argv
    scored = subprocess.run([sys.executable, *argv[1:]], cwd=REPO, env=env,
                            capture_output=True, text=True, timeout=2400)
    assert scored.returncode == 0, (scored.stdout[-3000:], scored.stderr[-4000:])
    manifest_path = Path(argv[argv.index("--matrix-manifest") + 1])
    manifest = json.loads(manifest_path.read_text())
    summary = next(Path(argv[argv.index("--out-dir") + 1]).rglob("summary.csv"))
    return summary, manifest


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    base_dir = tmp_path_factory.mktemp("matrix_base")
    _write_shards(base_dir)
    env = {**os.environ, "NANOCHAT_BASE_DIR": str(base_dir), "BIO_RESULTS_REGISTRY": str(base_dir / "registry.jsonl"), "PYTHONWARNINGS": "ignore"}
    tok = _run(["scripts.tok_train", "--max_chars=200000", "--vocab_size=1024"], env, timeout=600)
    assert tok.returncode == 0, tok.stderr[-3000:]

    argv = ["scripts.matrix_launch", "--exploratory", "--columns", ",".join(COLUMNS),
            "--seeds", ",".join(map(str, SEEDS)), f"--recipe={RECIPE}"]
    plan = _run([*argv, "--plan"], env, timeout=300)
    assert plan.returncode == 0, plan.stderr
    score_argv = shlex.split(plan.stdout.splitlines()[-1])
    path = Path(score_argv[score_argv.index("--matrix-manifest") + 1])
    assert path.is_file() and not (path.parent / "attempts.jsonl").exists()
    launch = _run([*argv, "--execute"], env, timeout=2400)
    assert launch.returncode == 0, (launch.stdout[-2000:], launch.stderr[-4000:])
    assert launch.stdout.splitlines()[-1] == plan.stdout.splitlines()[-1]

    summary, manifest = _score(launch, env)
    stats = _run(["bio_inspired_nanochat.eval_stats", str(summary), "--min-pairs", "2", "--json-out", str(base_dir / "verdict.json")], env, timeout=600)
    assert stats.returncode == 0, stats.stderr[-3000:]
    return {"base_dir": base_dir, "launch": launch, "manifest": manifest, "summary": summary, "verdict": base_dir / "verdict.json", "env": env}


def test_launcher_trained_every_cell_under_the_tag_eval_matrix_resolves(pipeline):
    for column in COLUMNS:
        for seed in SEEDS:
            tag = pipeline["manifest"]["checkpoint_template"].format(preset=column, seed=seed)
            ckpt = pipeline["base_dir"] / "base_checkpoints" / tag
            assert (ckpt / "model_000002.pt").exists(), sorted(p.name for p in ckpt.iterdir()) if ckpt.exists() else f"missing {ckpt}"
            meta = json.loads((ckpt / "meta_000002.json").read_text(encoding="utf-8"))
            assert int(meta["model_config"]["init_seed"]) == seed
            assert bool(meta.get("synapses", False)) == (column != "vanilla")


def test_eval_matrix_scored_every_checkpoint_from_its_own_metadata(pipeline):
    with pipeline["summary"].open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    cells = {(r["preset"], int(r["seed"])) for r in rows}
    assert cells == {(c, s) for c in COLUMNS for s in SEEDS}, cells
    for r in rows:
        assert math.isfinite(float(r["val_bpb"])), r
        assert r["recipe_source"] == "base_train_checkpoint", r["recipe_source"]
        assert r["matrix_manifest_digest"] == pipeline["manifest"]["run_id"]
        assert int(r["trainable_params"]) > 0 and int(r["active_params"]) > 0


def test_restart_requires_matching_run_and_preserves_complete_checkpoints(pipeline):
    manifest = pipeline["manifest"]
    timing = pipeline["base_dir"] / "matrix_runs" / manifest["run_id"] / "attempts.jsonl"
    before = timing.read_bytes()
    argv = ["scripts.matrix_launch", "--exploratory", "--columns", ",".join(COLUMNS),
            "--seeds", ",".join(map(str, SEEDS)), f"--recipe={RECIPE}", "--execute"]
    refused = _run(argv, pipeline["env"], timeout=300)
    assert refused.returncode != 0 and "run already exists" in refused.stderr
    resumed = _run([*argv, "--restart"], pipeline["env"], timeout=300)
    assert resumed.returncode == 0, resumed.stderr
    changed = [a.replace("--num_iterations=2", "--num_iterations=3") for a in argv]
    rejected = _run([*changed, "--restart"], pipeline["env"], timeout=300)
    assert rejected.returncode != 0 and "existing matching matrix manifest" in rejected.stderr
    assert timing.read_bytes() == before


def test_eval_stats_pairs_the_synaptic_off_anchor_against_vanilla(pipeline):
    verdict = json.loads(pipeline["verdict"].read_text(encoding="utf-8"))
    presets = verdict.get("presets") or verdict
    assert "synaptic_off" in presets, list(presets)[:5]
    entry = presets["synaptic_off"]
    assert "paired_vs_baseline" in entry and entry["paired_vs_baseline"]["n_pairs"] == len(SEEDS)


def test_structural_stage_pipeline_trains_and_scores_moe_columns(pipeline):
    """MoE structural stage (moe_fixed vs moe_splitmerge) trains, scores, and pairs cleanly (uta.10)."""
    base_dir: Path = pipeline["base_dir"]
    struct_cols = ("moe_fixed", "moe_splitmerge")
    env = {
        **os.environ,
        "NANOCHAT_BASE_DIR": str(base_dir),
        "BIO_RESULTS_REGISTRY": str(base_dir / "registry.jsonl"),
        "PYTHONWARNINGS": "ignore",
    }
    launch = _run(
        [
            "scripts.matrix_launch",
            "--stage",
            "structural",
            "--exploratory",
            "--columns",
            ",".join(struct_cols),
            "--seeds",
            ",".join(map(str, SEEDS)),
            f"--recipe={RECIPE}",
            "--execute",
        ],
        env,
        timeout=2400,
    )
    assert launch.returncode == 0, (launch.stdout[-2000:], launch.stderr[-4000:])

    summary, manifest = _score(launch, env)
    assert manifest["run_id"] != pipeline["manifest"]["run_id"]
    with summary.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {(r["preset"], int(r["seed"])) for r in rows} == {
        (c, s) for c in struct_cols for s in SEEDS
    }
    for r in rows:
        assert math.isfinite(float(r["val_bpb"])), r
        assert r["matrix_manifest_digest"] == manifest["run_id"]
        assert 0 < int(r["active_params"]) < int(r["trainable_params"])
        assert math.isfinite(float(r["moe_gini"])), r
        assert math.isfinite(float(r["dead_expert_frac"])), r

    verdict_path = base_dir / "structural_verdict.json"
    stats = _run(
        [
            "bio_inspired_nanochat.eval_stats",
            str(summary),
            "--baseline",
            "moe_fixed",
            "--min-pairs",
            "2",
            "--json-out",
            str(verdict_path),
        ],
        env,
        timeout=600,
    )
    assert stats.returncode == 0, stats.stderr[-3000:]
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    presets = verdict.get("presets") or verdict
    assert "moe_splitmerge" in presets
    assert presets["moe_splitmerge"]["paired_vs_baseline"]["n_pairs"] == len(SEEDS)

