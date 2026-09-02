"""
Launch (or print) the base_train runs behind the pre-registered ablation matrix (hwxb.5.2).

The matrix in ``bio_inspired_nanochat.ablation_matrix`` says WHAT each column is; ``eval_matrix``
loads the finished ``base_train`` checkpoint of each (column, seed) and scores it. Until this script
existed nothing turned a column into a training command, so the headline experiment had a spec and
a scorer but no producer. Every command below is derived from the spec through
``ablation_matrix.base_train_argv`` — the ``--syn_cfg.<field>=<value>`` overrides, the seed as
``--init_seed`` (what eval_matrix checks the checkpoint against) and the model tag eval_matrix
resolves — so the spec, the runs and the scoring cannot drift apart.

Examples
--------
Print the screening pass for the D1 recipe (nothing runs):

    python -m scripts.matrix_launch --stage screening \
        --recipe="--depth=10 --tie_embeddings=1 --device_batch_size=32 --total_batch_size=524288 --num_iterations=950"

(Write ``--recipe=...`` with the ``=``: the value starts with ``--``, and argparse would otherwise
read a single flag such as ``--depth=10`` as an unknown option.)

Run it on two GPUs, then score it:

    python -m scripts.matrix_launch --stage screening --recipe="..." --nproc 2 --execute
    python -m scripts.eval_matrix batch --presets <printed list> --seeds 1337,1338 \
        --checkpoint-dir "<base_dir>/base_checkpoints/matrix_{preset}_s{seed}"

The structural pair (MoE fixed vs split/merge) is ``--stage structural``; it is not part of the
pre-registered screening set.
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import subprocess
import sys
from typing import Sequence

from rich.console import Console
from rich.table import Table

from bio_inspired_nanochat import ablation_matrix as am
from bio_inspired_nanochat.common import get_base_dir

logger = logging.getLogger("bio_inspired_nanochat.matrix_launch")
console = Console()

STAGES = ("screening", "confirmation", "structural", "all")


def columns_for_stage(stage: str, survivors: Sequence[str] = ()) -> list[am.AblationConfig]:
    if stage == "screening":
        return am.screening_columns()
    if stage == "confirmation":
        if not survivors:
            raise ValueError("--stage confirmation needs --survivors (the columns screening kept)")
        return am.confirmation_columns(list(survivors))
    if stage == "structural":
        return am.structural_columns()
    if stage == "all":
        return am.screening_columns() + am.structural_columns()
    raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")


def launcher_prefix(nproc: int) -> list[str]:
    """``python -m scripts.base_train`` on one process; ``torchrun`` for ``nproc`` GPUs."""
    if nproc <= 1:
        return [sys.executable, "-m", "scripts.base_train"]
    return [
        "torchrun", "--standalone", f"--nproc_per_node={int(nproc)}",
        "-m", "scripts.base_train", "--",
    ]


def build_commands(
    columns: Sequence[am.AblationConfig], seeds: Sequence[int], recipe: Sequence[str], nproc: int
) -> list[tuple[am.AblationConfig, int, list[str]]]:
    out: list[tuple[am.AblationConfig, int, list[str]]] = []
    for column in columns:
        for seed in seeds:
            argv = am.base_train_argv(column, seed=seed, recipe=tuple(recipe))
            out.append((column, seed, launcher_prefix(nproc) + argv))
    return out


def eval_matrix_hint(columns: Sequence[am.AblationConfig], seeds: Sequence[int]) -> str:
    template = os.path.join(get_base_dir(), "base_checkpoints", am.MATRIX_CHECKPOINT_TEMPLATE)
    presets = ",".join(c.config_id for c in columns)
    seed_list = ",".join(str(s) for s in seeds)
    return (
        f"{sys.executable} -m scripts.eval_matrix batch --presets {presets} --seeds {seed_list} "
        f"--checkpoint-dir {shlex.quote(template)}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=STAGES, default="screening")
    parser.add_argument("--survivors", default="", help="Comma-separated column ids kept after screening")
    parser.add_argument(
        "--seeds", default="",
        help="Comma-separated init seeds (default: the stage's pre-registered seeds)",
    )
    parser.add_argument(
        "--recipe", default="",
        help="Shared base_train flags for every cell, as one shell-quoted string "
        "(depth, batch, tokens, data, eval cadence, ...). Write it as --recipe=\"--depth=10 ...\" "
        "(with the =), since the value itself starts with --",
    )
    parser.add_argument("--nproc", type=int, default=1, help="GPUs per run; >1 uses torchrun")
    parser.add_argument("--execute", action="store_true", help="Run the commands sequentially (default: print only)")
    args = parser.parse_args(argv)

    survivors = [s for s in args.survivors.split(",") if s]
    columns = columns_for_stage(args.stage, survivors)
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s]
    else:
        seeds = list(am.CONFIRMATION_SEEDS if args.stage == "confirmation" else am.SCREENING_SEEDS)
    recipe = shlex.split(args.recipe)
    commands = build_commands(columns, seeds, recipe, args.nproc)

    table = Table(title=f"matrix {args.stage}: {len(columns)} columns x {len(seeds)} seeds = {len(commands)} runs")
    table.add_column("column")
    table.add_column("seed", justify="right")
    table.add_column("base_train arguments beyond the recipe")
    for column, seed, cmd in commands:
        table.add_row(column.config_id, str(seed), " ".join(cmd[len(launcher_prefix(args.nproc)) + len(recipe):]))
    console.print(table)
    for _column, _seed, cmd in commands:
        # Plain print on purpose: these lines are meant to be copied or piped, so they must stay
        # one physical line each regardless of terminal width (rich re-wraps at the console width).
        print(shlex.join(cmd))
    console.print("\n[bold]score with:[/bold]")
    print(eval_matrix_hint(columns, seeds))

    if not args.execute:
        return 0
    for column, seed, cmd in commands:
        logger.info("[matrix_launch] start column=%s seed=%d", column.config_id, seed)
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            logger.error(
                "[matrix_launch] column=%s seed=%d failed with exit %d; stopping",
                column.config_id, seed, result.returncode,
            )
            return int(result.returncode)
        logger.info("[matrix_launch] done column=%s seed=%d", column.config_id, seed)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    raise SystemExit(main())
