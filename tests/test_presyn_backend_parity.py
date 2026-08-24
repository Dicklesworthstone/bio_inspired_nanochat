"""Golden-driven parity for the canonical Python, Triton, and Rust presyn backends."""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from bio_inspired_nanochat.synaptic import SynapticPresyn
from test_presyn_golden import (
    ATOL,
    RTOL,
    STATE_KEYS,
    load_decode_golden,
    run_decode_canonical,
)

pytestmark = [pytest.mark.unit, pytest.mark.golden]


def _clone_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        name: [entry.clone() for entry in value]
        if isinstance(value, list)
        else value.clone()
        for name, value in state.items()
    }


def _assert_state_close(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    assert actual.keys() == expected.keys()
    for name in STATE_KEYS:
        torch.testing.assert_close(actual[name], expected[name], rtol=RTOL, atol=ATOL)
    assert len(actual["DELAY"]) == len(expected["DELAY"])
    for actual_delay, expected_delay in zip(
        actual["DELAY"], expected["DELAY"], strict=True
    ):
        torch.testing.assert_close(actual_delay, expected_delay, rtol=RTOL, atol=ATOL)


def _numpy_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        name: [entry.detach().cpu().numpy() for entry in value]
        if isinstance(value, list)
        else value.detach().cpu().numpy()
        for name, value in state.items()
    }


def _torch_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        name: [torch.from_numpy(np.asarray(entry)) for entry in value]
        if isinstance(value, list)
        else torch.from_numpy(np.asarray(value))
        for name, value in state.items()
    }


@pytest.fixture(scope="module")
def decode_golden() -> dict[str, Any]:
    golden = load_decode_golden()
    assert golden["provenance"]["schema_version"] == 2
    assert golden["expected_ema"] == 1.0
    return golden


def test_python_canonical_matches_frozen_decode_trajectory(
    decode_golden: dict[str, Any],
) -> None:
    releases, logits, states, ema = run_decode_canonical(
        decode_golden["cfg"], decode_golden["init_state"], decode_golden["steps"]
    )
    assert ema == decode_golden["expected_ema"]
    for step, (release, expected_release) in enumerate(
        zip(releases, decode_golden["expected_e"], strict=True)
    ):
        torch.testing.assert_close(
            release,
            expected_release,
            rtol=RTOL,
            atol=ATOL,
            msg=lambda message: f"step {step}: {message}",
        )
        torch.testing.assert_close(
            logits[step], decode_golden["expected_logits"][step], rtol=RTOL, atol=ATOL
        )
        _assert_state_close(states[step], decode_golden["expected_states"][step])


def test_triton_interpreter_matches_frozen_decode_trajectory() -> None:
    """Execute the real Triton kernel in CPU interpreter mode on every CI host."""
    program = textwrap.dedent(
        """
        import sys
        from pathlib import Path

        import torch

        sys.path.insert(0, str(Path.cwd() / "tests"))
        from test_presyn_golden import ATOL, RTOL, STATE_KEYS, load_decode_golden
        from bio_inspired_nanochat.kernels.presyn_fused import presyn_live_decode_step

        golden = load_decode_golden()
        state = {
            name: [entry.clone() for entry in value] if isinstance(value, list) else value.clone()
            for name, value in golden["init_state"].items()
        }
        ema = torch.tensor([golden["expected_ema"]], dtype=torch.float32)
        for step, (drive, idx, valid, logits) in enumerate(golden["steps"]):
            actual_logits = logits.clone()
            actual = presyn_live_decode_step(
                state,
                drive,
                idx,
                golden["cfg"],
                ema_e=ema,
                valid=valid,
                logits=actual_logits,
                _interpret=True,
            )
            torch.testing.assert_close(actual, golden["expected_e"][step], rtol=RTOL, atol=ATOL)
            torch.testing.assert_close(
                actual_logits, golden["expected_logits"][step], rtol=RTOL, atol=ATOL
            )
            expected = golden["expected_states"][step]
            for name in STATE_KEYS:
                torch.testing.assert_close(state[name], expected[name], rtol=RTOL, atol=ATOL)
            for got, want in zip(state["DELAY"], expected["DELAY"], strict=True):
                torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "TRITON_INTERPRET": "1"},
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA backend unavailable")
def test_cuda_triton_dispatch_matches_frozen_decode_trajectory(
    decode_golden: dict[str, Any],
) -> None:
    device = torch.device("cuda")
    cfg = dataclasses.replace(decode_golden["cfg"], native_presyn=True)
    pre = SynapticPresyn(16, cfg).to(device)
    pre.ema_e.fill_(decode_golden["expected_ema"])
    state = {
        name: [entry.to(device) for entry in value]
        if isinstance(value, list)
        else value.to(device)
        for name, value in decode_golden["init_state"].items()
    }
    with torch.inference_mode():
        for step, (drive, idx, valid, logits) in enumerate(decode_golden["steps"]):
            actual_logits = logits.to(device)
            actual = pre.release_canonical(
                state,
                drive.to(device),
                idx.to(device),
                train=False,
                valid=valid.to(device),
                logits=actual_logits,
            )
            torch.testing.assert_close(
                actual.cpu(), decode_golden["expected_e"][step], rtol=RTOL, atol=ATOL
            )
            torch.testing.assert_close(
                actual_logits.cpu(),
                decode_golden["expected_logits"][step],
                rtol=RTOL,
                atol=ATOL,
            )
            expected = decode_golden["expected_states"][step]
            _assert_state_close(
                {
                    name: [entry.cpu() for entry in value]
                    if isinstance(value, list)
                    else value.cpu()
                    for name, value in state.items()
                },
                expected,
            )


def test_rust_backend_matches_frozen_decode_trajectory(
    decode_golden: dict[str, Any],
) -> None:
    if importlib.util.find_spec("rustbpe") is None:
        pytest.skip(
            "rustbpe extension not built; run `uv run maturin develop --release`"
        )
    rustbpe = importlib.import_module("rustbpe")
    assert hasattr(rustbpe, "presyn_release_canonical_cpu"), (
        "built rustbpe is missing presyn_release_canonical_cpu; rebuild the extension"
    )

    state = _clone_state(decode_golden["init_state"])
    for step, (drive, idx, valid, _logits) in enumerate(decode_golden["steps"]):
        release, next_state = rustbpe.presyn_release_canonical_cpu(
            drive.numpy(),
            idx.numpy(),
            valid.numpy(),
            _numpy_state(state),
            decode_golden["cfg"],
            decode_golden["expected_ema"],
        )
        state = _torch_state(dict(next_state))
        torch.testing.assert_close(
            torch.from_numpy(np.asarray(release)),
            decode_golden["expected_e"][step],
            rtol=RTOL,
            atol=ATOL,
        )
        _assert_state_close(state, decode_golden["expected_states"][step])
