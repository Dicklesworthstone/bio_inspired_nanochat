"""Security and integrity tests for CMA-ES checkpoint resumption (bead zrzy).

Validates:
1. Checkpoints write cryptographic manifest (es_manifest.json) alongside serialized state.
2. Valid checkpoints resume smoothly and match search state.
3. Tampered checkpoint pickle bytes fail closed with SHA-256 mismatch and Rich error.
4. Corrupted manifest JSON fails closed.
5. Mismatched search space dimensions fail closed.
6. Legacy unverified checkpoints (missing manifest) are refused by default.
7. Legacy unverified checkpoints load only when --allow-unverified-checkpoint is explicitly passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cma as cma_module
import pytest
from rich.console import Console

from bio_inspired_nanochat.synaptic import SynapticConfig
from scripts.tune_bio_params import (
    TOP10_PARAM_SPECS,
    _cma_bounds,
    _load_verified_cma_checkpoint,
    _prepare_run_artifacts,
    encode_params,
    main as tune_main,
)

pytestmark = pytest.mark.unit


def _create_mock_checkpoint(
    run_dir: Path,
    *,
    dim: int = len(TOP10_PARAM_SPECS),
    countiter: int = 2,
    write_manifest: bool = True,
    tamper_manifest: bool = False,
    corrupt_manifest: bool = False,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    specs = TOP10_PARAM_SPECS[:dim]
    defaults = SynapticConfig()
    x0 = encode_params(defaults, specs)
    lbs, ubs = _cma_bounds(specs)
    es = cma_module.CMAEvolutionStrategy(
        x0,
        0.2,
        {"popsize": 4, "bounds": [lbs, ubs], "verbose": -1, "seed": 42},
    )
    # Advance iterations
    es.countiter = countiter

    es_bytes = es.pickle_dumps()
    es_path = run_dir / "es_latest.pkl"
    es_path.write_bytes(es_bytes)

    if write_manifest:
        manifest_path = run_dir / "es_manifest.json"
        if corrupt_manifest:
            manifest_path.write_text("{ corrupt json ", encoding="utf-8")
        else:
            sha = "0000000000000000000000000000000000000000000000000000000000000000" if tamper_manifest else hashlib.sha256(es_bytes).hexdigest()
            doc = {
                "format": "cmaes-checkpoint-manifest-v1",
                "run_id": "test-run",
                "generation": countiter,
                "countiter": countiter,
                "best_loss": 1.234,
                "dim": dim,
                "sha256": sha,
                "saved_at_unix": 1000.0,
            }
            manifest_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    return es_path


def test_verified_checkpoint_roundtrip(tmp_path: Path):
    """A valid checkpoint with genuine manifest resumes with exact generation count."""
    run_dir = tmp_path / "valid_run"
    _create_mock_checkpoint(run_dir, countiter=5)

    args = argparse.Namespace(
        run_dir=str(run_dir),
        resume=True,
        seed=42,
        no_tensorboard=True,
    )

    artifacts = _prepare_run_artifacts(args)
    assert artifacts is not None

    es = _load_verified_cma_checkpoint(
        artifacts,
        specs=TOP10_PARAM_SPECS,
        allow_unverified=False,
        console=Console(quiet=True),
    )
    assert es is not None
    assert es.countiter == 5
    assert es.N == len(TOP10_PARAM_SPECS)


def test_tampered_pickle_fails_closed(tmp_path: Path):
    """Mutating the pickle bytes triggers a SHA-256 mismatch and fails closed."""
    run_dir = tmp_path / "tampered_run"
    es_path = _create_mock_checkpoint(run_dir, countiter=3)

    # Tamper with the pickle bytes
    raw = bytearray(es_path.read_bytes())
    raw[-1] ^= 0xFF
    es_path.write_bytes(bytes(raw))

    args = argparse.Namespace(
        run_dir=str(run_dir),
        resume=True,
        seed=42,
        no_tensorboard=True,
    )

    artifacts = _prepare_run_artifacts(args)
    assert artifacts is not None

    with pytest.raises(ValueError) as ei:
        _load_verified_cma_checkpoint(
            artifacts,
            specs=TOP10_PARAM_SPECS,
            allow_unverified=False,
            console=Console(quiet=True),
        )
    assert "SHA-256 digest mismatch" in str(ei.value)


def test_corrupted_manifest_fails_closed(tmp_path: Path):
    """Malformed manifest JSON fails closed."""
    run_dir = tmp_path / "corrupt_manifest_run"
    _create_mock_checkpoint(run_dir, corrupt_manifest=True)

    args = argparse.Namespace(
        run_dir=str(run_dir),
        resume=True,
        seed=42,
        no_tensorboard=True,
    )

    artifacts = _prepare_run_artifacts(args)
    assert artifacts is not None

    with pytest.raises(ValueError) as ei:
        _load_verified_cma_checkpoint(
            artifacts,
            specs=TOP10_PARAM_SPECS,
            allow_unverified=False,
            console=Console(quiet=True),
        )
    assert "Corrupt checkpoint manifest" in str(ei.value)


def test_dimension_mismatch_fails_closed(tmp_path: Path):
    """Manifest with mismatched search space dimension fails closed."""
    run_dir = tmp_path / "mismatch_run"
    # Create checkpoint with dim=5 instead of 10
    _create_mock_checkpoint(run_dir, dim=5)

    args = argparse.Namespace(
        run_dir=str(run_dir),
        resume=True,
        seed=42,
        no_tensorboard=True,
    )

    artifacts = _prepare_run_artifacts(args)
    assert artifacts is not None

    with pytest.raises(ValueError) as ei:
        _load_verified_cma_checkpoint(
            artifacts,
            specs=TOP10_PARAM_SPECS,  # dim=10
            allow_unverified=False,
            console=Console(quiet=True),
        )
    assert "Checkpoint search space mismatch" in str(ei.value)


def test_legacy_unverified_refused_by_default(tmp_path: Path):
    """Checkpoints without manifest fail closed unless --allow-unverified-checkpoint is used."""
    run_dir = tmp_path / "legacy_run"
    _create_mock_checkpoint(run_dir, write_manifest=False)

    args = argparse.Namespace(
        run_dir=str(run_dir),
        resume=True,
        seed=42,
        no_tensorboard=True,
    )

    artifacts = _prepare_run_artifacts(args)
    assert artifacts is not None

    with pytest.raises(ValueError) as ei:
        _load_verified_cma_checkpoint(
            artifacts,
            specs=TOP10_PARAM_SPECS,
            allow_unverified=False,
            console=Console(quiet=True),
        )
    assert "Unverified checkpoint" in str(ei.value)
    assert "--allow-unverified-checkpoint" in str(ei.value)


def test_legacy_unverified_allowed_with_explicit_opt_in(tmp_path: Path):
    """Checkpoints without manifest load when allow_unverified=True is passed."""
    run_dir = tmp_path / "legacy_optin_run"
    _create_mock_checkpoint(run_dir, countiter=4, write_manifest=False)

    args = argparse.Namespace(
        run_dir=str(run_dir),
        resume=True,
        seed=42,
        no_tensorboard=True,
    )

    artifacts = _prepare_run_artifacts(args)
    assert artifacts is not None

    es = _load_verified_cma_checkpoint(
        artifacts,
        specs=TOP10_PARAM_SPECS,
        allow_unverified=True,
        console=Console(quiet=True),
    )
    assert es is not None
    assert es.countiter == 4


def test_e2e_resume_command_uses_verified_manifest(tmp_path: Path):
    """Running optimize then resuming through CLI uses manifest verification."""
    run_dir = tmp_path / "cli_resume_run"
    cmd1 = [
        "optimize",
        "--seed", "42",
        "--device", "cpu",
        "--generations", "1",
        "--popsize", "4",
        "--steps", "2",
        "--batch-size", "4",
        "--run-dir", str(run_dir),
        "--no-tensorboard",
    ]
    ret1 = tune_main(cmd1)
    assert ret1 == 0
    assert (run_dir / "es_manifest.json").exists(), "es_manifest.json must be written"

    cmd2 = [
        "optimize",
        "--seed", "42",
        "--device", "cpu",
        "--generations", "2",
        "--popsize", "4",
        "--steps", "2",
        "--batch-size", "4",
        "--run-dir", str(run_dir),
        "--resume",
        "--no-tensorboard",
    ]
    ret2 = tune_main(cmd2)
    assert ret2 == 0
