"""Unit tests for the distributed streaming dataloader (bead hwxb.2.3)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from bio_inspired_nanochat.dataloader import tokenizing_distributed_data_loader_with_state


class MockTokenizer:
    """Mock tokenizer providing deterministic token sequences for dataloader testing."""

    def get_bos_token_id(self) -> int:
        return 1

    def encode(
        self,
        doc_batch: list[str],
        prepend: int | None = None,
        num_threads: int = 1,
    ) -> list[list[int]]:
        del num_threads
        out = []
        for doc in doc_batch:
            tokens = [prepend] if prepend is not None else []
            tokens.extend([ord(c) % 256 for c in doc])
            out.append(tokens)
        return out


@pytest.fixture
def sample_parquet_dir(tmp_path: Path) -> Path:
    """Create two sample parquet shards containing text columns."""
    for shard_idx in range(2):
        texts = [
            f"The quick brown fox jumps over the lazy dog document {shard_idx}_{row}."
            for row in range(30)
        ]
        table = pa.Table.from_arrays([pa.array(texts)], names=["text"])
        pq_path = tmp_path / f"shard_{shard_idx:05d}.parquet"
        pq.write_table(table, str(pq_path), row_group_size=10)
    return tmp_path


def test_dataloader_yields_correct_shapes_and_state(sample_parquet_dir: Path) -> None:
    """Dataloader yields batches of (B, T) with valid state_dict."""
    with (
        patch("bio_inspired_nanochat.dataset.DATA_DIR", str(sample_parquet_dir)),
        patch("bio_inspired_nanochat.dataloader.get_tokenizer", return_value=MockTokenizer()),
    ):
        B, T = 2, 8
        loader = tokenizing_distributed_data_loader_with_state(
            B=B,
            T=T,
            split="train",
            tokenizer_threads=1,
            tokenizer_batch_size=8,
            device="cpu",
        )

        inputs, targets, state_dict = next(loader)
        assert inputs.shape == (B, T)
        assert targets.shape == (B, T)
        assert inputs.dtype == torch.long
        assert targets.dtype == torch.long
        assert "pq_idx" in state_dict
        assert "rg_idx" in state_dict


def test_dataloader_resumes_from_state_dict(sample_parquet_dir: Path) -> None:
    """Dataloader accepts resume_state_dict and continues streaming."""
    with (
        patch("bio_inspired_nanochat.dataset.DATA_DIR", str(sample_parquet_dir)),
        patch("bio_inspired_nanochat.dataloader.get_tokenizer", return_value=MockTokenizer()),
    ):
        B, T = 2, 8
        loader1 = tokenizing_distributed_data_loader_with_state(
            B=B,
            T=T,
            split="train",
            tokenizer_threads=1,
            tokenizer_batch_size=8,
            device="cpu",
        )
        _, _, state1 = next(loader1)

        loader_resumed = tokenizing_distributed_data_loader_with_state(
            B=B,
            T=T,
            split="train",
            tokenizer_threads=1,
            tokenizer_batch_size=8,
            device="cpu",
            resume_state_dict=state1,
        )
        inputs_res, targets_res, state_res = next(loader_resumed)
        assert inputs_res.shape == (B, T)
        assert targets_res.shape == (B, T)
        assert state_res["pq_idx"] >= state1["pq_idx"]
