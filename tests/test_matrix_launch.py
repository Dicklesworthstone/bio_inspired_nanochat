"""The matrix spec is the only source of the base_train commands that produce its checkpoints.

Bridge plan G1/G4 (hwxb.5.2). Locks:

* every screening column round-trips: the ``--syn_cfg.*`` flags ``base_train_argv`` emits, fed
  through the same override parser ``base_train`` uses, rebuild exactly the column's
  ``SynapticConfig`` — so a checkpoint trained from the printed command passes eval_matrix's
  bio-flag check;
* vanilla gets ``--synapses=0`` and no synaptic flags; the seed travels as ``--init_seed`` and the
  model tag is the one eval_matrix's ``--checkpoint-dir`` template resolves;
* the structural pair is opt-in (not in ``screening_columns``), carries the lifecycle globals, and
  is a preset eval_matrix knows;
* ``scripts.matrix_launch`` prints one command per (column, seed) and the scoring command.

Run:  pytest tests/test_matrix_launch.py -v
"""

from __future__ import annotations

import subprocess
import sys
import shlex
from pathlib import Path

import pytest

from bio_inspired_nanochat import ablation_matrix as am
from bio_inspired_nanochat.ablation_registry import MECHANISMS
from bio_inspired_nanochat.cmaes_params import (
    apply_syn_cfg_overrides,
    extract_syn_cfg_cli_overrides,
)
from bio_inspired_nanochat.synaptic import SynapticConfig

pytestmark = pytest.mark.unit

RECIPE = ("--depth=10", "--total_batch_size=524288")


@pytest.mark.parametrize("column", am.screening_columns() + am.d1_screening_columns()[-2:] + am.structural_columns(), ids=lambda c: c.config_id)
def test_every_column_round_trips_through_the_base_train_override_parser(column):
    argv = am.base_train_argv(column, seed=1337, recipe=RECIPE)
    assert argv[: len(RECIPE)] == list(RECIPE)
    remaining, raw_overrides = extract_syn_cfg_cli_overrides(argv)
    expected = column.build_syn_cfg()
    if column.base is am.Base.VANILLA:
        assert "--synapses=0" in remaining and not raw_overrides
        assert expected is None
        return
    assert "--synapses=1" in remaining
    rebuilt = apply_syn_cfg_overrides(SynapticConfig(native_presyn=True, native_genetics=True), raw_overrides)
    for mechanism in MECHANISMS:
        assert getattr(rebuilt, mechanism.field) == getattr(expected, mechanism.field), (
            column.config_id, mechanism.field
        )
    assert "--init_seed=1337" in remaining
    assert f"--model_tag={am.matrix_model_tag(column.config_id, 1337)}" in remaining


def test_synaptic_off_anchor_neutralises_every_default_on_mechanism_on_the_command_line():
    (synaptic_off,) = [c for c in am.anchors() if c.config_id == "synaptic_off"]
    argv = am.base_train_argv(synaptic_off, seed=1)
    flags = {a for a in argv if a.startswith("--syn_cfg.")}
    for mechanism in MECHANISMS:
        if mechanism.default_on:
            assert any(a.startswith(f"--syn_cfg.{mechanism.field}=") for a in flags), mechanism.field


def test_model_tags_are_unique_and_match_the_eval_matrix_template():
    cols = am.screening_columns() + am.structural_columns()
    tags = {am.matrix_model_tag(c.config_id, s) for c in cols for s in am.CONFIRMATION_SEEDS}
    assert len(tags) == len(cols) * len(am.CONFIRMATION_SEEDS)
    assert am.matrix_model_tag("bio_all", 1337) == am.MATRIX_CHECKPOINT_TEMPLATE.format(preset="bio_all", seed=1337)


def test_structural_pair_is_opt_in_and_carries_the_lifecycle_globals():
    screening_ids = {c.config_id for c in am.screening_columns()}
    structural = am.structural_columns()
    assert [c.config_id for c in structural[:2]] == ["moe_fixed", "moe_splitmerge"]
    assert [c.config_id for c in structural[2:]] == [c.config_id for c in am.moe_mechanism_columns()]
    assert not screening_ids & {c.config_id for c in structural}, "the MoE stage is not screening"
    assert len(am.screening_columns()) == 17
    fixed_argv = am.base_train_argv(structural[0], seed=7)
    sm_argv = am.base_train_argv(structural[1], seed=7)
    assert "--use_moe=1" in fixed_argv and not any(a.startswith("--splitmerge_every") for a in fixed_argv)
    assert f"--splitmerge_every={am.STRUCTURAL_SPLITMERGE_EVERY}" in sm_argv
    assert "--sm_health_mode=credit" in sm_argv
    assert "--split_health_min=1.5" in sm_argv and "--merge_health_max=0.35" in sm_argv
    # Both share bio_all's SynapticConfig: the contrast is purely structural.
    assert structural[0].build_syn_cfg() == structural[1].build_syn_cfg() == SynapticConfig()


def test_eval_matrix_knows_the_structural_presets():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.eval_matrix import MATRIX_COLUMNS

    assert {"moe_fixed", "moe_splitmerge", "moe_no_genome", "moe_add_glial_homeostasis"} <= set(MATRIX_COLUMNS)
    # Leave-one-out columns are named registry presets and live in PRESETS, not here.
    assert "synaptic_off" in MATRIX_COLUMNS and "bio_no_presyn" not in MATRIX_COLUMNS


def test_vanilla_column_refuses_synaptic_overrides():
    bad = am.AblationConfig("vanilla_x", am.Base.VANILLA, {"enable_presyn": False}, "anchor", "planted")
    with pytest.raises(ValueError, match="vanilla column"):
        am.base_train_argv(bad, seed=0)


def test_unbound_preview_prints_each_cell_and_directs_user_to_the_bound_plan():
    proc = subprocess.run(
        # --recipe=... with the '=': the value starts with '--', and a single flag such as
        # --depth=4 would otherwise be read by argparse as an unknown option.
        [sys.executable, "-m", "scripts.matrix_launch", "--stage", "structural", "--seeds", "1,2", "--recipe=--depth=4 --num_iterations=3"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=300,
        env={**__import__("os").environ, "COLUMNS": "400"},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    lines = [ln for ln in proc.stdout.splitlines() if "scripts.base_train" in ln and "--model_tag=" in ln]
    stage = [c.config_id for c in am.structural_columns()]
    assert len(lines) == 2 * len(stage), proc.stdout
    assert all("--depth=4" in ln and "--num_iterations=3" in ln for ln in lines)
    assert "Unbound recipe preview" in proc.stdout and "--plan" in proc.stdout
    assert "scripts.eval_matrix matrix" not in proc.stdout
    # A one-flag recipe must work too (the documented --recipe= form).
    one = subprocess.run(
        [sys.executable, "-m", "scripts.matrix_launch", "--stage", "structural", "--seeds", "1", "--recipe=--depth=4"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=300,
    )
    assert one.returncode == 0, one.stderr[-1000:]
    assert sum("--depth=4" in ln and "--model_tag=" in ln for ln in one.stdout.splitlines()) == len(stage)


def test_confirmation_stage_defaults_to_statistically_feasible_seed_count():
    """Each confirmation column prints every fresh seed prescribed by the fixed Holm family."""
    from bio_inspired_nanochat.eval_stats import d1_confirmation_plan

    plan = d1_confirmation_plan(["bio_no_presyn"])
    expected_seeds = plan["seeds"]
    assert not set(expected_seeds).intersection(am.SCREENING_SEEDS)
    assert 2 ** (1 - len(expected_seeds)) <= 0.05 / 3  # two anchor contrasts and one mechanism
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.matrix_launch",
            "--stage",
            "confirmation",
            "--survivors",
            "bio_no_presyn",
            "--recipe=--depth=4",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=300,
        env={**__import__("os").environ, "COLUMNS": "400"},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    tags = [arg for line in proc.stdout.splitlines() if "scripts.base_train" in line
            for arg in shlex.split(line) if arg.startswith("--model_tag=")]
    assert tags == [f"--model_tag={am.matrix_model_tag(column, seed)}"
                    for column in plan["columns"] for seed in expected_seeds]

    # Structural survivors include moe_fixed anchor; unknown survivor IDs raise ValueError
    from scripts.matrix_launch import columns_for_stage

    struct_cols = [c.config_id for c in columns_for_stage("confirmation", ["moe_splitmerge"])]
    assert "moe_fixed" in struct_cols
    assert "moe_splitmerge" in struct_cols
    with pytest.raises(ValueError, match="unknown survivor column id"):
        columns_for_stage("confirmation", ["not_a_real_column"])


def test_identical_data_shards_refused_even_under_different_names(tmp_path, monkeypatch):
    from bio_inspired_nanochat import matrix_manifest as mm

    monkeypatch.setattr(mm, "get_base_dir", lambda: str(tmp_path))
    data = tmp_path / "base_data"
    monkeypatch.setattr("bio_inspired_nanochat.dataset.DATA_DIR", str(data))
    data.mkdir()
    (data / "01.parquet").write_bytes(b"one shard")
    with pytest.raises(ValueError, match="share a shard"):
        mm.dataset_identity()
    (data / "02.parquet").write_bytes(b"one shard")
    with pytest.raises(ValueError, match="share a shard"):
        mm.dataset_identity()


def test_manifest_changes_with_recipe_and_rejects_tag_tampering(tmp_path):
    import json
    from bio_inspired_nanochat.matrix_manifest import load_manifest, seal_manifest

    fields = {"schema": 1, "stage": "screening", "recipe": {"num_iterations": "7"}}
    sealed = seal_manifest(fields)
    other = seal_manifest({**fields, "recipe": {"num_iterations": "8"}})
    assert sealed["checkpoint_template"] != other["checkpoint_template"]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(sealed))
    assert load_manifest(path) == sealed
    path.write_text(json.dumps({**sealed, "checkpoint_template": other["checkpoint_template"]}))
    with pytest.raises(ValueError, match="identity mismatch"):
        load_manifest(path)


def test_device_cost_and_invalid_rates():
    hours = am.estimate_gpu_hours(am.anchors()[:1], (11,), 144_000_000, 1000, world_size=2)
    assert hours == 80 > am.DEFAULT_GPU_HOUR_CAP
    for rate in (float("inf"), float("nan"), -1):
        with pytest.raises(ValueError, match="tok_per_sec"):
            am.estimate_gpu_hours(am.anchors(), (11,), 10_000_000, rate)


def test_screen_rejects_40_wall_hours_on_two_devices_before_spawning(tmp_path, monkeypatch, capsys):
    from scripts import matrix_launch

    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(matrix_launch.mm, "dataset_identity", lambda: {"fixture": "budget-only"})
    monkeypatch.setattr(matrix_launch.mm, "source_identity", lambda: ("revision", "source"))
    batch = 65_536
    steps = matrix_launch.mm.registered_tokens("screening", batch) // batch
    recipe = (f"--depth=10 --device_type=cuda --max_seq_len=1024 --tie_embeddings=1 --eval_tokens=1000000 "
              f"--total_batch_size={batch} --num_iterations={steps}")
    aggregate_rate = len(am.d1_screening_columns()) * len(am.SCREENING_SEEDS) * steps * batch / (40 * 3600)
    with pytest.raises(SystemExit):
        matrix_launch.main(["--execute", "--nproc", "2", f"--recipe={recipe}", "--tok-per-sec", str(aggregate_rate),
                            "--score-seconds-per-cell", "1"])
    assert "exceeds the aggregate device-hour cap" in capsys.readouterr().err
    assert not (tmp_path / "matrix_runs").exists()


def test_scoring_shares_physical_device_budget_with_training(tmp_path, monkeypatch):
    from bio_inspired_nanochat import matrix_manifest as mm

    monkeypatch.setattr(mm, "get_base_dir", lambda: str(tmp_path))
    manifest = {"run_id": "cost-fixture", "checkpoint_template": "{preset}-{seed}",
                "world_size": 2, "cap_gpu_hours": 61 / 3600}
    directory = tmp_path / "matrix_runs" / "cost-fixture"
    directory.mkdir(parents=True)
    for filename, seconds in [("attempts.jsonl", 30), ("scoring_attempts.jsonl", 2)]:
        mm.append_event(directory / filename, event="start", tag="vanilla-1")
        mm.append_event(directory / filename, event="end", tag="vanilla-1", seconds=seconds, exit=0)
    with pytest.raises(ValueError, match="exhausted before scoring"), mm.score_budget(manifest, "vanilla", 1):
        pytest.fail("over-budget scorer was admitted")
    manifest["cap_gpu_hours"] = 1
    with mm.score_budget(manifest, "vanilla", 1):
        pass
    assert mm.attempt_seconds(directory / "scoring_attempts.jsonl")["vanilla-1"] > 2


def test_confirmation_admits_complete_scored_screen_and_refuses_an_edited_pair(tmp_path, monkeypatch):
    import json
    from test_eval_stats import _fixture_screen, _write_d1_fixture
    from scripts import matrix_launch

    mm = matrix_launch.mm
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    assets = {"fixture": "synthetic admission example"}
    monkeypatch.setattr(mm, "dataset_identity", lambda: assets)
    monkeypatch.setattr(mm, "source_identity", lambda: ("2" * 40, "1" * 64))
    common = ("--device_type=cuda --depth=10 --max_seq_len=1024 --tie_embeddings=1 --eval_tokens=1048576 "
              "--device_batch_size=8 --total_batch_size=32768")
    assert matrix_launch.main(["--stage", "screening", "--plan", "--nproc", "2", "--tok-per-sec", "30000",
                               "--score-seconds-per-cell", "8", f"--recipe={common} --num_iterations=305"]) == 0
    screen_path, = (tmp_path / "matrix_runs").glob("*/manifest.json")
    screen = mm.load_manifest(screen_path)
    data = _fixture_screen()
    data["add_bdnf"] = {1337: 3.5, 1338: 7.0}
    csv_path = tmp_path / "screen.csv"
    _write_d1_fixture(csv_path, data, manifest_digest=screen["run_id"])
    for column in screen["columns"]:
        for seed in screen["seeds"]:
            tag = screen["checkpoint_template"].format(preset=column, seed=seed)
            for filename, seconds in (("attempts.jsonl", 50), ("scoring_attempts.jsonl", 8)):
                mm.append_event(screen_path.parent / filename, event="start", tag=tag)
                mm.append_event(screen_path.parent / filename, event="end", tag=tag, seconds=seconds, exit=0)
    mm.seal_scores(screen, csv_path)
    argv = ["--stage", "confirmation", "--plan", "--nproc", "2", "--screen-csv", str(csv_path),
            f"--recipe={common} --num_iterations=3051"]
    assert matrix_launch.main(argv) == 0
    plans = [json.loads(p.read_text()) for p in (tmp_path / "matrix_runs").glob("*/manifest.json")]
    confirmation, = (p for p in plans if p["stage"] == "confirmation")
    assert confirmation["seeds"] == list(range(2337, 2344))
    assert confirmation["screen_gpu_hours"] == pytest.approx(36 * (2 * 50 + 8) / 3600)
    assert len(confirmation["per_arm_gpu_hours"]) == 5
    data["add_bdnf"][1337] -= 0.25
    _write_d1_fixture(csv_path, data, manifest_digest=screen["run_id"])
    with pytest.raises(ValueError, match="completed scorer receipt"):
        matrix_launch.main(argv)


@pytest.mark.parametrize("flags", [
    ["--stage", "confirmation", "--survivors", "bio_no_presyn"],
    ["--stage", "screening", "--seeds", "1337,1337"],
    ["--stage", "screening", "--columns", "vanilla"],
    ["--stage", "screening", "--cap-gpu-hours", "73"],
])
def test_inadmissible_execution_fails_before_training(flags, tmp_path, monkeypatch):
    from scripts.matrix_launch import main

    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    with pytest.raises(SystemExit) as exc:
        main(["--execute", *flags])
    assert exc.value.code == 2
    assert not (tmp_path / "base_checkpoints").exists()
