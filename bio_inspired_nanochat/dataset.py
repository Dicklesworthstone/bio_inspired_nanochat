"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.

Download CLI:
  uv run python -m bio_inspired_nanochat.dataset -n 8 --num-workers 4 --verify-size --verify-existing

Checksum verification (optional):
  - Provide `--checksum-file` as either:
    - JSON: {"shard_00000.parquet": "<sha256>", ...}
    - sha256sum-style text: "<sha256>  shard_00000.parquet"
"""

import argparse
import functools
import hashlib
import json
import os
import time
from multiprocessing import Pool

import pyarrow.parquet as pq
import requests

from bio_inspired_nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The URL on the internet where the data is hosted and downloaded from on demand
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822 # the last datashard is shard_01822.parquet
def index_to_filename(index):
    return f"shard_{index:05d}.parquet" # format of the filenames
# Resolved lazily via ensure_data_dir() so that merely importing this module
# never creates directories under HOME (read-only-HOME CI containers crashed).
DATA_DIR: str | None = None


def ensure_data_dir() -> str:
    """Return (and ensure) the base_data directory; safe to call repeatedly."""
    global DATA_DIR
    if DATA_DIR is None:
        DATA_DIR = os.path.join(get_base_dir(), "base_data")
        os.makedirs(DATA_DIR, exist_ok=True)
    return DATA_DIR

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(data_dir: str | None = None) -> list[str]:
    """Looks into a data dir and returns full paths to all parquet files."""
    resolved = ensure_data_dir() if data_dir is None else data_dir
    parquet_files = sorted([
        f for f in os.listdir(resolved)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(resolved, f) for f in parquet_files]
    return parquet_paths


def parquet_paths_for_split(split: str, *, data_dir: str | None = None) -> list[str]:
    """Return parquet paths for a given split.

    Convention:
    - If there are 2+ shards: last shard is validation, the rest are training.
    - If there is 1 shard: it is used for both train and val (useful for smoke runs).
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files(data_dir=data_dir)
    if split == "val":
        return parquet_paths[-1:]
    return parquet_paths if len(parquet_paths) <= 1 else parquet_paths[:-1]


def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = parquet_paths_for_split(split)
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

# -----------------------------------------------------------------------------
def _load_sha256_map(path: str | None) -> dict[str, str]:
    if path is None:
        return {}

    p = os.path.expanduser(path)
    if p.endswith(".json"):
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise TypeError("checksum JSON must be an object mapping filename -> sha256")
        out: dict[str, str] = {}
        for k, v in data.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise TypeError("checksum JSON must map strings to strings")
            out[k] = v.lower()
        return out

    out = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            sha = parts[0].lower()
            filename = parts[-1]
            out[filename] = sha
    return out


def _sha256_file(path: str, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _head_content_length(url: str, *, timeout: float) -> int | None:
    try:
        with requests.head(url, allow_redirects=True, timeout=timeout) as resp:
            resp.raise_for_status()
            cl = resp.headers.get("Content-Length")
    except requests.RequestException:
        return None
    if cl is None:
        return None
    try:
        return int(cl)
    except ValueError:
        return None


def _verify_download(
    path: str,
    *,
    expected_size: int | None,
    expected_sha256: str | None,
) -> str | None:
    if expected_size is not None:
        actual = os.path.getsize(path)
        if actual != expected_size:
            return f"size mismatch: expected {expected_size} bytes, got {actual} bytes"
    if expected_sha256 is not None:
        actual_sha = _sha256_file(path)
        if actual_sha.lower() != expected_sha256.lower():
            return f"sha256 mismatch: expected {expected_sha256}, got {actual_sha}"
    return None


def download_single_file(
    index: int,
    *,
    verify_size: bool = False,
    sha256_map: dict[str, str] | None = None,
    verify_existing: bool = False,
    timeout_sec: float = 30.0,
    chunk_size: int = 1024 * 1024,
    max_attempts: int = 5,
) -> bool:
    """Downloads a single file index, with backoff and optional validation."""

    # Construct the local filepath for this file and skip if it already exists
    filename = index_to_filename(index)
    filepath = os.path.join(ensure_data_dir(), filename)
    temp_path = filepath + ".tmp"
    # Construct the remote URL for this file
    url = f"{BASE_URL}/{filename}"
    had_existing = os.path.exists(filepath)
    expected_sha256 = (sha256_map or {}).get(filename)

    if had_existing and not verify_existing:
        print(f"Skipping {filepath} (already exists)")
        return True

    expected_size = _head_content_length(url, timeout=timeout_sec) if verify_size else None

    if had_existing:
        if verify_existing and (expected_size is not None or expected_sha256 is not None):
            err = _verify_download(
                filepath,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
            )
            if err is None:
                print(f"Skipping {filepath} (already exists; verified)")
                return True
            print(f"Existing shard failed validation; redownloading {filename}: {err}")
        else:
            print(f"Skipping {filepath} (already exists; no verification data)")
            return True

    print(f"Downloading {filename}...")

    # Download with retries
    for attempt in range(1, max_attempts + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout_sec) as response:
                response.raise_for_status()
                # Write to temporary file first
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
            err = _verify_download(
                temp_path,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
            )
            if err is not None:
                raise IOError(err)
            # Move temp file to final location
            os.replace(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            if not had_existing and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except OSError:
                    pass
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu 100BT dataset shards")
    parser.add_argument(
        "-n",
        "--num-files",
        type=int,
        default=-1,
        help="Number of shards to download (default: -1), -1 = all shards",
    )
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    parser.add_argument(
        "--verify-size",
        action="store_true",
        help="Validate downloaded shard size against Content-Length (HEAD request).",
    )
    parser.add_argument(
        "--checksum-file",
        default=None,
        help="Optional sha256 map (sha256sum format or JSON mapping filename->sha256).",
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="If a shard already exists, verify it (size/hash) and redownload on mismatch.",
    )
    args = parser.parse_args()

    num = MAX_SHARD + 1 if args.num_files == -1 else min(args.num_files, MAX_SHARD + 1)
    ids_to_download = list(range(num))
    sha256_map = _load_sha256_map(args.checksum_file)
    print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    print(f"Target directory: {ensure_data_dir()}")
    print(
        f"Integrity: verify_size={bool(args.verify_size)} "
        f"verify_existing={bool(args.verify_existing)} "
        f"sha256_entries={len(sha256_map)}"
    )
    print()
    worker = functools.partial(
        download_single_file,
        verify_size=bool(args.verify_size),
        sha256_map=sha256_map if sha256_map else None,
        verify_existing=bool(args.verify_existing),
    )
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(worker, ids_to_download)

    # Report results
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {ensure_data_dir()}")
