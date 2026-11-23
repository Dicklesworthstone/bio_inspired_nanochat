"""Identity shared by the matrix launcher, trainer and checkpoint evaluator."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from bio_inspired_nanochat.common import get_base_dir
from bio_inspired_nanochat.dataset import parquet_paths_for_split


def digest(value: object) -> str:
    payload = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dataset_identity() -> dict:
    """Fingerprint the actual loader order and the encoding and byte table used for BPB."""
    root = Path(get_base_dir())
    splits = {}
    for split in ("train", "val"):
        paths = parquet_paths_for_split(split)
        if not paths:
            raise ValueError(f"matrix has no {split} shards")
        splits[split] = [file_digest(Path(path)) for path in paths]
    if set(splits["train"]).intersection(splits["val"]):
        raise ValueError("train and validation share a shard by content")
    directory = root / "tokenizer"
    encoding = directory / "tokenizer.json"
    if not encoding.is_file():
        encoding = directory / "tokenizer.pkl"
    tokenizer = {p.name: file_digest(p) for p in (encoding, directory / "token_bytes.pt")}
    return {**splits, "tokenizer": tokenizer, "train_sha256": digest(splits["train"]),
            "val_sha256": digest(splits["val"]), "tokenizer_sha256": digest(tokenizer)}


def source_identity() -> tuple[str, str]:
    """Include uncommitted source, while excluding generated runs and machine-specific paths."""
    root = Path(__file__).resolve().parent.parent
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=30).strip()
    files = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--",
         "bio_inspired_nanochat/*.py", "scripts/*.py", "rust_src/src/*.rs",
         "rust_src/Cargo.toml", "rust_src/Cargo.lock", "Cargo.toml", "Cargo.lock", "pyproject.toml", "uv.lock"],
        cwd=root, timeout=30,
    ).decode().split("\0")
    return revision, digest({f: file_digest(root / f) for f in files if f})


def registered_tokens(stage: str, batch_size: int) -> int:
    from bio_inspired_nanochat.ablation_matrix import CONFIRMATION_TOKENS, SCREENING_TOKENS

    target = {"screening": SCREENING_TOKENS, "confirmation": CONFIRMATION_TOKENS}[stage]
    if batch_size <= 0:
        raise ValueError("total_batch_size must be positive")
    actual = target // batch_size * batch_size
    if actual < target * 0.95:
        raise ValueError("batch size leaves more than 5% of the registered budget unused")
    return actual


def seal_manifest(fields: dict) -> dict:
    run_id = digest(fields)
    return {**fields, "run_id": run_id,
            "checkpoint_template": f"matrix_{fields['stage']}_{run_id[:20]}_{{preset}}_s{{seed}}"}


def load_manifest(path: str | Path) -> dict:
    try:
        with Path(path).open(encoding="utf-8") as stream:
            data = json.load(stream)
    except json.JSONDecodeError as exc:
        exc.add_note(f"while reading matrix manifest {path}")
        raise
    if not isinstance(data, dict) or data.get("schema") != 1:
        raise ValueError("unsupported matrix manifest schema")
    fields = {k: v for k, v in data.items() if k not in {"run_id", "checkpoint_template"}}
    if data != seal_manifest(fields):
        raise ValueError("matrix manifest identity mismatch")
    return data


def append_event(path: Path, **event) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def score_fingerprint(csv_path: Path) -> dict[str, str]:
    return {"csv_sha256": file_digest(csv_path), "jsonl_sha256": file_digest(csv_path.with_suffix(".jsonl"))}


def seal_scores(manifest: dict, csv_path: Path) -> None:
    import fcntl

    directory = Path(get_base_dir()) / "matrix_runs" / manifest["run_id"]
    with (directory / "launch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt = directory / "scored.json"
        if receipt.exists():
            raise ValueError("registered matrix already has a completed score receipt")
        staged = directory / "scored.pending"
        with staged.open("w", encoding="utf-8") as stream:
            json.dump(score_fingerprint(csv_path), stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, receipt)


def attempt_seconds(path: Path, *, recover: bool = False) -> dict[str, float]:
    """Aggregate real wall time, including retries. Recover only under the run lock."""
    events = [json.loads(row) for row in path.read_text().splitlines()] if path.is_file() else []
    if len(events) % 2:
        last = events[-1]
        elapsed = time.time() - last.get("started_at", math.inf)
        if not recover or last.get("event") != "start" or not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("previous matrix attempt time cannot be reconstructed")
        end = {"event": "end", "tag": last["tag"], "seconds": elapsed, "exit": None}
        append_event(path, **end)
        events.append(end)
    wall: dict[str, float] = {}
    for start, end in zip(events[::2], events[1::2], strict=True):
        seconds = end.get("seconds", 0)
        if (start.get("event") != "start" or end.get("event") != "end" or end.get("tag") != start["tag"]
            or not math.isfinite(seconds) or seconds <= 0):
            raise ValueError("invalid matrix attempt ledger")
        wall[start["tag"]] = wall.get(start["tag"], 0.0) + seconds
    return wall


@contextmanager
def score_budget(manifest: dict, preset: str, seed: int):
    """Charge the published single-GPU scorer against the same allocation as training."""
    import fcntl
    import signal

    directory = Path(get_base_dir()) / "matrix_runs" / manifest["run_id"]
    tag = manifest["checkpoint_template"].format(preset=preset, seed=seed)
    path = directory / "scoring_attempts.jsonl"
    with (directory / "launch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "scored.json").exists():
            raise ValueError("registered matrix already has a completed score receipt")
        training = attempt_seconds(directory / "attempts.jsonl", recover=True)
        if tag not in training:
            raise ValueError("registered scoring requires a charged training attempt")
        previous = attempt_seconds(path, recover=True)
        seconds = manifest["cap_gpu_hours"] * 3600 - sum(training.values()) * manifest["world_size"] - sum(previous.values())
        if seconds <= 0:
            raise ValueError("device-hour cap exhausted before scoring")
        append_event(path, event="start", tag=tag, started_at=time.time())
        started = time.monotonic()
        code = 1
        # This single-process CLI uses the process deadline even inside a long CUDA call.
        # On expiry, the next run conservatively recovers the unfinished ledger entry.
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield
            if time.monotonic() - started > seconds:
                raise ValueError("measured scoring time exceeded the remaining device-hour allocation")
            code = 0
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            append_event(path, event="end", tag=tag, seconds=time.monotonic() - started, exit=code)
