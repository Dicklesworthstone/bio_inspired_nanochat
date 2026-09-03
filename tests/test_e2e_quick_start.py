"""The README Quick Start, run as a test (bridge plan G5).

Every other test imports pieces of the training stack; none ran ``scripts.base_train`` the way a
user does, which is how an ungated ``torch.compile`` call kept the flagship script from starting
on Python 3.14 for months without a red test. This test runs the documented commands as
subprocesses against a throwaway base directory:

1. ``scripts.tok_train`` on two tiny synthetic FineWeb-shaped parquet shards;
2. ``scripts.base_train --synapses=1 --syn_cfg.<field>=…`` for two CPU steps, which must write a
   checkpoint and exactly one ``harness == "train"`` registry row;
3. ``scripts.chat_cli -i base -g <tag>`` on that checkpoint, which must generate text;
4. the planted negative: an unknown ``--syn_cfg.<field>`` must fail before training and name it.

Green here does not say the model is any good (two steps); it says the Quick Start is wired.

Run:  pytest tests/test_e2e_quick_start.py -v
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

REPO = Path(__file__).resolve().parents[1]
MODEL_TAG = "e2e_quick_start"
WORDS = (
    "synapse vesicle calcium release neuron dendrite axon spike plasticity memory "
    "learning network gradient token model layer attention expert energy fatigue "
    "the a of and to in is that for with as on by from it this be are was were"
).split()


def _write_shards(base_dir: Path) -> None:
    """Two shards so the loader has a train shard and a distinct val shard (the last file)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    data_dir = base_dir / "base_data"
    data_dir.mkdir(parents=True)
    rng = random.Random(0)
    for shard in range(2):
        docs = []
        for _ in range(160):
            n = rng.randint(40, 90)
            words = [rng.choice(WORDS) for _ in range(n)]
            words[0] = words[0].capitalize()
            docs.append(" ".join(words) + ".")
        table = pa.table({"text": docs})
        pq.write_table(table, data_dir / f"shard_{shard:05d}.parquet", row_group_size=40)


def _run(module_args: list[str], env: dict[str, str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", *module_args],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.fixture(scope="module")
def quick_start(tmp_path_factory):
    base_dir = tmp_path_factory.mktemp("quick_start_base")
    _write_shards(base_dir)
    env = {
        **os.environ,
        "NANOCHAT_BASE_DIR": str(base_dir),
        "BIO_RESULTS_REGISTRY": str(base_dir / "registry.jsonl"),
        "PYTHONWARNINGS": "ignore",
    }
    tok = _run(["scripts.tok_train", "--max_chars=200000", "--vocab_size=1024"], env, timeout=600)
    assert tok.returncode == 0, tok.stderr[-3000:]
    train = _run(
        [
            "scripts.base_train",
            "--synapses=1",
            "--depth=2",
            "--max_seq_len=64",
            "--device_batch_size=2",
            "--total_batch_size=128",
            "--num_iterations=2",
            "--eval_every=2",
            "--eval_tokens=128",
            "--core_metric_every=-1",
            "--sample_every=-1",
            "--device_type=cpu",
            f"--model_tag={MODEL_TAG}",
            "--syn_cfg.tau_rrp=60.0",
            "--syn_cfg.bistable_latch=1",
        ],
        env,
        timeout=1500,
    )
    assert train.returncode == 0, train.stderr[-4000:]
    return {"base_dir": base_dir, "env": env, "train": train}


def test_tokenizer_is_trained_into_the_base_dir(quick_start):
    tok_dir = quick_start["base_dir"] / "tokenizer"
    assert (tok_dir / "tokenizer.json").exists()
    assert (tok_dir / "token_bytes.pt").exists()


def test_base_train_applies_the_syn_cfg_overrides_and_writes_a_checkpoint(quick_start):
    out = quick_start["train"].stdout
    assert "[config] syn_cfg.tau_rrp = 60.0" in out
    assert "[config] syn_cfg.bistable_latch = True" in out
    ckpt = quick_start["base_dir"] / "base_checkpoints" / MODEL_TAG
    assert (ckpt / "model_000002.pt").exists(), sorted(p.name for p in ckpt.iterdir())
    meta = json.loads((ckpt / "meta_000002.json").read_text(encoding="utf-8"))
    assert meta["model_config"]["n_layer"] == 2


def test_base_train_records_exactly_one_train_row_in_the_registry(quick_start):
    rows = [
        json.loads(line)
        for line in (quick_start["base_dir"] / "registry.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_rows = [r for r in rows if r.get("harness") == "train"]
    assert len(train_rows) == 1, rows
    assert train_rows[0]["hardware"].startswith("cpu")
    assert math.isfinite(float(train_rows[0]["metrics"]["val_bpb"]))


def test_chat_cli_generates_from_the_base_checkpoint(quick_start):
    chat = _run(
        [
            "scripts.chat_cli",
            "-i", "base",
            "-g", MODEL_TAG,
            "-p", "The synapse",
            "--device-type", "cpu",
            "-d", "float32",
        ],
        quick_start["env"],
        timeout=600,
    )
    assert chat.returncode == 0, chat.stderr[-3000:]
    assert "Assistant:" in chat.stdout
    reply = chat.stdout.split("Assistant:", 1)[1].strip()
    assert reply, "the base checkpoint must generate at least one token"


def test_chat_cli_selective_decoding_abstains_at_a_zero_threshold(quick_start):
    """--selective (wmel.2) reaches the engine's abstention path from the real CLI."""
    chat = _run(
        ["scripts.chat_cli", "-i", "base", "-g", MODEL_TAG, "-p", "The synapse", "--device-type", "cpu", "-d", "float32",
         "--selective", "--max-entropy", "0.0"],
        quick_start["env"], timeout=600,
    )
    assert chat.returncode == 0, chat.stderr[-3000:]
    assert "[abstain: predictive entropy" in chat.stdout, chat.stdout[-500:]


def test_chunked_regime_trains_and_is_recorded(quick_start):
    """--hebb_chunk_len (bead hwxb.8) runs through the real script and lands in the registry row."""
    run = _run(
        [
            "scripts.base_train",
            "--synapses=1",
            "--depth=2",
            "--max_seq_len=64",
            "--device_batch_size=2",
            "--total_batch_size=128",
            "--num_iterations=2",
            "--eval_every=2",
            "--eval_tokens=128",
            "--core_metric_every=-1",
            "--sample_every=-1",
            "--device_type=cpu",
            f"--model_tag={MODEL_TAG}_chunked",
            "--hebb_chunk_len=8",
        ],
        quick_start["env"],
        timeout=1500,
    )
    assert run.returncode == 0, run.stderr[-4000:]
    # The registry row hashes the config; the checkpoint metadata carries it verbatim.
    meta = json.loads(
        (quick_start["base_dir"] / "base_checkpoints" / f"{MODEL_TAG}_chunked" / "meta_000002.json").read_text(encoding="utf-8")
    )
    assert meta["user_config"]["hebb_chunk_len"] == 8, meta.get("user_config")
    rows = [
        json.loads(line)
        for line in (quick_start["base_dir"] / "registry.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_rows = [r for r in rows if r.get("harness") == "train"]
    assert len(train_rows) == 2, "the plain and the chunked run each leave one registry row"
    assert all(math.isfinite(float(r["metrics"]["val_bpb"])) for r in train_rows)


def test_unknown_syn_cfg_field_is_refused_before_training(quick_start):
    bad = _run(
        [
            "scripts.base_train",
            "--synapses=1",
            "--depth=2",
            "--device_type=cpu",
            "--num_iterations=1",
            "--syn_cfg.bogus_field=1",
        ],
        quick_start["env"],
        timeout=600,
    )
    assert bad.returncode != 0
    assert "bogus_field" in bad.stderr
