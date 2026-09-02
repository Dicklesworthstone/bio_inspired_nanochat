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


@pytest.mark.parametrize("column", am.screening_columns() + am.structural_columns(), ids=lambda c: c.config_id)
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
    rebuilt = apply_syn_cfg_overrides(SynapticConfig(), raw_overrides)
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
    assert [c.config_id for c in structural] == ["moe_fixed", "moe_splitmerge"]
    assert not screening_ids & {c.config_id for c in structural}, "the pre-registered set is unchanged"
    assert len(am.screening_columns()) == 20
    fixed_argv = am.base_train_argv(structural[0], seed=7)
    sm_argv = am.base_train_argv(structural[1], seed=7)
    assert "--use_moe=1" in fixed_argv and not any(a.startswith("--splitmerge_every") for a in fixed_argv)
    assert f"--splitmerge_every={am.STRUCTURAL_SPLITMERGE_EVERY}" in sm_argv
    assert "--sm_health_mode=relative" in sm_argv
    assert "--split_health_min=1.5" in sm_argv and "--merge_health_max=0.35" in sm_argv
    # Both share bio_all's SynapticConfig: the contrast is purely structural.
    assert structural[0].build_syn_cfg() == structural[1].build_syn_cfg() == SynapticConfig()


def test_eval_matrix_knows_the_structural_presets():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.eval_matrix import MATRIX_COLUMNS

    assert {"moe_fixed", "moe_splitmerge"} <= set(MATRIX_COLUMNS)
    # Leave-one-out columns are named registry presets and live in PRESETS, not here.
    assert "synaptic_off" in MATRIX_COLUMNS and "bio_no_presyn" not in MATRIX_COLUMNS


def test_vanilla_column_refuses_synaptic_overrides():
    bad = am.AblationConfig("vanilla_x", am.Base.VANILLA, {"enable_presyn": False}, "anchor", "planted")
    with pytest.raises(ValueError, match="vanilla column"):
        am.base_train_argv(bad, seed=0)


def test_launcher_prints_one_command_per_cell_and_the_scoring_command():
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
    assert len(lines) == 4, proc.stdout
    assert all("--depth=4" in ln and "--num_iterations=3" in ln for ln in lines)
    assert "scripts.eval_matrix batch --presets moe_fixed,moe_splitmerge --seeds 1,2" in proc.stdout
    assert "matrix_{preset}_s{seed}" in proc.stdout
    # A one-flag recipe must work too (the documented --recipe= form).
    one = subprocess.run(
        [sys.executable, "-m", "scripts.matrix_launch", "--stage", "structural", "--seeds", "1", "--recipe=--depth=4"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=300,
    )
    assert one.returncode == 0, one.stderr[-1000:]
    assert sum("--depth=4" in ln and "--model_tag=" in ln for ln in one.stdout.splitlines()) == 2
