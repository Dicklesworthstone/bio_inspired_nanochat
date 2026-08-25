"""Security and integrity tests for inert CMA-ES checkpoint replay (bead zrzy)."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from rich.console import Console

from bio_inspired_nanochat.synaptic import SynapticConfig
from scripts.tune_bio_params import (
    TOP10_PARAM_SPECS,
    _cma_bounds,
    _load_cma_checkpoint,
    _new_cma_strategy,
    _prepare_run_artifacts,
    _save_cma_checkpoint,
    encode_params,
    main as tune_main,
)

pytestmark = pytest.mark.unit


def _artifacts(run_dir: Path, *, seed: int = 42):
    return _prepare_run_artifacts(
        argparse.Namespace(
            run_dir=str(run_dir),
            resume=True,
            seed=seed,
            no_tensorboard=True,
        )
    )


def _create_safe_checkpoint(
    run_dir: Path,
    *,
    generations: int = 2,
    seed: int = 42,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = _artifacts(run_dir, seed=seed)
    assert artifacts is not None
    specs = TOP10_PARAM_SPECS
    defaults = SynapticConfig()
    x0 = encode_params(defaults, specs)
    lbs, ubs = _cma_bounds(specs)
    sigma0 = 0.2
    popsize = 4
    es = _new_cma_strategy(
        x0,
        sigma0,
        popsize=popsize,
        lower_bounds=lbs,
        upper_bounds=ubs,
        seed=seed,
    )
    records: list[dict[str, object]] = []
    best_history: list[float] = []
    best = float("inf")
    for generation in range(1, generations + 1):
        solutions = es.ask()
        fitnesses = [float(np.square(solution).sum()) for solution in solutions]
        es.tell(solutions, fitnesses)
        best = min(best, min(fitnesses))
        best_history.append(best)
        records.append(
            {
                "generation": generation,
                "solutions": np.asarray(solutions, dtype=np.float64).tolist(),
                "fitnesses": fitnesses,
                "sigma_after": None,
            }
        )
    _save_cma_checkpoint(
        artifacts.es_state_json,
        run_id=artifacts.run_id,
        specs=specs,
        x0=x0,
        sigma0=sigma0,
        popsize=popsize,
        seed=seed,
        strategy=es,
        generation_records=records,
        best_loss_history=best_history,
        restart_events=0,
    )
    return artifacts, es


def test_verified_checkpoint_roundtrip(tmp_path: Path):
    """A valid JSON checkpoint reproduces the exact next CMA population."""
    run_dir = tmp_path / "valid_run"
    artifacts, original = _create_safe_checkpoint(run_dir, generations=3)
    expected_next_population = np.asarray(original.ask(), dtype=np.float64)

    loaded = _load_cma_checkpoint(
        artifacts,
        specs=TOP10_PARAM_SPECS,
        console=Console(quiet=True),
    )
    assert loaded is not None
    assert loaded.strategy.countiter == 3
    assert loaded.strategy.N == len(TOP10_PARAM_SPECS)
    assert len(loaded.generation_records) == 3
    actual_next_population = np.asarray(loaded.strategy.ask(), dtype=np.float64)
    np.testing.assert_array_equal(actual_next_population, expected_next_population)


def _write_marker(marker_path: str) -> None:
    Path(marker_path).write_text("pickle payload executed", encoding="utf-8")


class _MaliciousCheckpoint:
    def __init__(self, marker_path: Path):
        self.marker_path = marker_path

    def __reduce__(self):
        return (_write_marker, (str(self.marker_path),))


def test_legacy_pickle_fails_closed(tmp_path: Path):
    """A malicious legacy pickle is refused without executing its payload."""
    run_dir = tmp_path / "legacy_pickle_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = _artifacts(run_dir)
    assert artifacts is not None
    marker_path = tmp_path / "payload_was_executed"

    artifacts.legacy_es_latest_pkl.write_bytes(
        pickle.dumps(_MaliciousCheckpoint(marker_path))
    )

    with pytest.raises(ValueError) as ei:
        _load_cma_checkpoint(
            artifacts,
            specs=TOP10_PARAM_SPECS,
            console=Console(quiet=True),
        )
    assert "Legacy CMA pickle checkpoints are not executable resume inputs" in str(
        ei.value
    )
    assert not marker_path.exists()


def test_corrupted_json_fails_closed(tmp_path: Path):
    """Malformed state JSON fails closed."""
    run_dir = tmp_path / "corrupt_json_run"
    artifacts, _ = _create_safe_checkpoint(run_dir, generations=2)

    # Corrupt JSON state
    artifacts.es_state_json.write_text("{corrupt: json", encoding="utf-8")

    with pytest.raises(ValueError) as ei:
        _load_cma_checkpoint(
            artifacts,
            specs=TOP10_PARAM_SPECS,
            console=Console(quiet=True),
        )
    assert "Corrupt CMA checkpoint" in str(ei.value)


def test_dimension_mismatch_fails_closed(tmp_path: Path):
    """Checkpoint with mismatched search space dimension fails closed."""
    run_dir = tmp_path / "mismatch_run"
    artifacts, _ = _create_safe_checkpoint(run_dir, generations=2)

    # Attempt to load with subset of specs (dim 5 vs 10)
    with pytest.raises(ValueError) as ei:
        _load_cma_checkpoint(
            artifacts,
            specs=TOP10_PARAM_SPECS[:5],
            console=Console(quiet=True),
        )
    assert "search space" in str(ei.value) or "match" in str(ei.value)


def test_tampered_numeric_history_fails_replay_validation(tmp_path: Path):
    """A modified generation record cannot silently change the resumed state."""
    run_dir = tmp_path / "tampered_history_run"
    artifacts, _ = _create_safe_checkpoint(run_dir, generations=2)
    document = json.loads(artifacts.es_state_json.read_text(encoding="utf-8"))
    document["generation_records"][0]["solutions"][0][0] += 0.25
    artifacts.es_state_json.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="candidate solutions do not match|checkpoint replay did not reproduce",
    ):
        _load_cma_checkpoint(
            artifacts,
            specs=TOP10_PARAM_SPECS,
            console=Console(quiet=True),
        )


def test_missing_checkpoint_raises_file_not_found(tmp_path: Path):
    """Missing state JSON raises FileNotFoundError."""
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = _artifacts(run_dir)
    assert artifacts is not None

    with pytest.raises(FileNotFoundError):
        _load_cma_checkpoint(
            artifacts,
            specs=TOP10_PARAM_SPECS,
            console=Console(quiet=True),
        )


def test_e2e_resume_command_roundtrip(tmp_path: Path):
    """Running optimize then resuming through CLI works with JSON state format."""
    run_dir = tmp_path / "cli_resume_run"
    cmd1 = [
        "optimize",
        "--seed",
        "42",
        "--device",
        "cpu",
        "--generations",
        "1",
        "--popsize",
        "4",
        "--steps",
        "2",
        "--batch-size",
        "4",
        "--run-dir",
        str(run_dir),
        "--no-tensorboard",
    ]
    ret1 = tune_main(cmd1)
    assert ret1 == 0
    assert (run_dir / "es_state.json").exists(), "es_state.json must be written"

    cmd2 = [
        "optimize",
        "--seed",
        "42",
        "--device",
        "cpu",
        "--generations",
        "2",
        "--popsize",
        "4",
        "--steps",
        "2",
        "--batch-size",
        "4",
        "--run-dir",
        str(run_dir),
        "--resume",
        "--no-tensorboard",
    ]
    ret2 = tune_main(cmd2)
    assert ret2 == 0
