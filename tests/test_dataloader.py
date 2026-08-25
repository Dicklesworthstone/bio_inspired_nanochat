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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"B": 0, "T": 8, "split": "train"},
        {"B": 2, "T": -1, "split": "train"},
        {"B": 2, "T": 8, "split": "test"},
        {"B": 2, "T": 8, "split": "train", "tokenizer_threads": 0},
        {"B": 2, "T": 8, "split": "train", "tokenizer_batch_size": 0},
        {
            "B": 2,
            "T": 8,
            "split": "train",
            "resume_state_dict": {"pq_idx": -1, "rg_idx": 0},
        },
        {
            "B": 2,
            "T": 8,
            "split": "train",
            "resume_state_dict": {"pq_idx": 0},
        },
        {
            "B": 2,
            "T": 8,
            "split": "train",
            "resume_state_dict": [],
        },
    ],
)
def test_dataloader_rejects_invalid_boundaries_before_io(kwargs) -> None:
    loader = tokenizing_distributed_data_loader_with_state(device="cpu", **kwargs)
    with pytest.raises(ValueError):
        next(loader)


def test_dataloader_rejects_resume_shard_outside_dataset(sample_parquet_dir: Path) -> None:
    with (
        patch("bio_inspired_nanochat.dataset.DATA_DIR", str(sample_parquet_dir)),
        patch("bio_inspired_nanochat.dataloader.get_tokenizer", return_value=MockTokenizer()),
    ):
        loader = tokenizing_distributed_data_loader_with_state(
            B=2,
            T=8,
            split="train",
            tokenizer_threads=1,
            tokenizer_batch_size=8,
            device="cpu",
            resume_state_dict={"pq_idx": 2, "rg_idx": 0},
        )
        with pytest.raises(ValueError, match="outside the available"):
            next(loader)

        row_group_loader = tokenizing_distributed_data_loader_with_state(
            B=2,
            T=8,
            split="train",
            tokenizer_threads=1,
            tokenizer_batch_size=8,
            device="cpu",
            resume_state_dict={"pq_idx": 0, "rg_idx": 99},
        )
        with pytest.raises(ValueError, match="outside the parquet row-group"):
            next(row_group_loader)


def test_multi_rank_lossless_exact_resume(sample_parquet_dir: Path) -> None:
    """Multi-rank regression: each rank resumes from its exact position with zero repeats/skips."""
    B, T = 2, 8
    world_size = 2
    total_steps = 10
    resume_step = 4

    uninterrupted_batches: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {0: [], 1: []}
    saved_states: dict[int, dict] = {}

    # 1. Run uninterrupted stream for both ranks
    for rank in range(world_size):
        with (
            patch("bio_inspired_nanochat.dataset.DATA_DIR", str(sample_parquet_dir)),
            patch("bio_inspired_nanochat.dataloader.get_tokenizer", return_value=MockTokenizer()),
            patch("bio_inspired_nanochat.dataloader.get_dist_info", return_value=(True, rank, rank, world_size)),
        ):
            loader = tokenizing_distributed_data_loader_with_state(
                B=B,
                T=T,
                split="train",
                tokenizer_threads=1,
                tokenizer_batch_size=4,
                device="cpu",
            )
            for step in range(total_steps):
                inputs, targets, state = next(loader)
                uninterrupted_batches[rank].append((inputs.clone(), targets.clone()))
                if step == resume_step - 1:
                    saved_states[rank] = state

    # 2. Resume each rank independently from saved rank-local state
    for rank in range(world_size):
        with (
            patch("bio_inspired_nanochat.dataset.DATA_DIR", str(sample_parquet_dir)),
            patch("bio_inspired_nanochat.dataloader.get_tokenizer", return_value=MockTokenizer()),
            patch("bio_inspired_nanochat.dataloader.get_dist_info", return_value=(True, rank, rank, world_size)),
        ):
            resumed_loader = tokenizing_distributed_data_loader_with_state(
                B=B,
                T=T,
                split="train",
                tokenizer_threads=1,
                tokenizer_batch_size=4,
                device="cpu",
                resume_state_dict=saved_states[rank],
            )
            for step in range(resume_step, total_steps):
                inputs_res, targets_res, _ = next(resumed_loader)
                expected_inputs, expected_targets = uninterrupted_batches[rank][step]
                torch.testing.assert_close(
                    inputs_res,
                    expected_inputs,
                    msg=f"Rank {rank} inputs mismatch at step {step}",
                )
                torch.testing.assert_close(
                    targets_res,
                    expected_targets,
                    msg=f"Rank {rank} targets mismatch at step {step}",
                )


def test_collated_multi_rank_state_resume(sample_parquet_dir: Path) -> None:
    """Collated multi-rank state dictionary allows each rank to extract its own state."""
    from bio_inspired_nanochat.dataloader import collate_dataloader_state_dicts

    B, T = 2, 8
    world_size = 2
    steps = 6
    cut_step = 2

    rank_states = {}
    ground_truth = {0: [], 1: []}

    for rank in range(world_size):
        with (
            patch("bio_inspired_nanochat.dataset.DATA_DIR", str(sample_parquet_dir)),
            patch("bio_inspired_nanochat.dataloader.get_tokenizer", return_value=MockTokenizer()),
            patch("bio_inspired_nanochat.dataloader.get_dist_info", return_value=(True, rank, rank, world_size)),
        ):
            loader = tokenizing_distributed_data_loader_with_state(
                B=B,
                T=T,
                split="train",
                tokenizer_threads=1,
                tokenizer_batch_size=4,
                device="cpu",
            )
            for step in range(steps):
                inp, tgt, st = next(loader)
                ground_truth[rank].append((inp.clone(), tgt.clone()))
                if step == cut_step - 1:
                    rank_states[rank] = st

    collated = collate_dataloader_state_dicts(rank_states, world_size=world_size)
    assert collated["version"] == 2
    assert collated["world_size"] == world_size
    assert "0" in collated["ranks"]
    assert "1" in collated["ranks"]

    for rank in range(world_size):
        with (
            patch("bio_inspired_nanochat.dataset.DATA_DIR", str(sample_parquet_dir)),
            patch("bio_inspired_nanochat.dataloader.get_tokenizer", return_value=MockTokenizer()),
            patch("bio_inspired_nanochat.dataloader.get_dist_info", return_value=(True, rank, rank, world_size)),
        ):
            res_loader = tokenizing_distributed_data_loader_with_state(
                B=B,
                T=T,
                split="train",
                tokenizer_threads=1,
                tokenizer_batch_size=4,
                device="cpu",
                resume_state_dict=collated,
            )
            for step in range(cut_step, steps):
                inp_res, tgt_res, _ = next(res_loader)
                expected_inp, expected_tgt = ground_truth[rank][step]
                torch.testing.assert_close(inp_res, expected_inp)
                torch.testing.assert_close(tgt_res, expected_tgt)


def test_incompatible_world_size_and_rank_fail_closed(sample_parquet_dir: Path) -> None:
    """Loader rejects state recorded for a different world_size or rank."""
    from bio_inspired_nanochat.dataloader import collate_dataloader_state_dicts

    with (
        patch("bio_inspired_nanochat.dataset.DATA_DIR", str(sample_parquet_dir)),
        patch("bio_inspired_nanochat.dataloader.get_tokenizer", return_value=MockTokenizer()),
        patch("bio_inspired_nanochat.dataloader.get_dist_info", return_value=(True, 0, 0, 4)),
    ):
        # Saved for world_size=2, currently running on world_size=4
        collated_ws2 = collate_dataloader_state_dicts(
            {"0": {"pq_idx": 0, "rg_idx": 0}, "1": {"pq_idx": 0, "rg_idx": 1}},
            world_size=2,
        )
        with pytest.raises(ValueError, match="checkpoint world_size \\(2\\) does not match current world_size \\(4\\)"):
            next(tokenizing_distributed_data_loader_with_state(
                B=2, T=8, split="train", device="cpu", resume_state_dict=collated_ws2
            ))

        # Single-rank state recorded for rank 1, running on rank 0
        state_rank1 = {"pq_idx": 0, "rg_idx": 1, "rank": 1, "world_size": 4}
        with pytest.raises(ValueError, match="recorded for rank 1"):
            next(tokenizing_distributed_data_loader_with_state(
                B=2, T=8, split="train", device="cpu", resume_state_dict=state_rank1
            ))


def test_malformed_resume_state_fails_closed(sample_parquet_dir: Path) -> None:
    """Loader rejects non-conforming or corrupted state dict payloads."""
    with (
        patch("bio_inspired_nanochat.dataset.DATA_DIR", str(sample_parquet_dir)),
        patch("bio_inspired_nanochat.dataloader.get_tokenizer", return_value=MockTokenizer()),
    ):
        # Non-integer doc_idx
        with pytest.raises(ValueError, match="doc_idx"):
            next(tokenizing_distributed_data_loader_with_state(
                B=2, T=8, split="train", device="cpu",
                resume_state_dict={"pq_idx": 0, "rg_idx": 0, "doc_idx": "five"}
            ))

        # Negative doc_idx
        with pytest.raises(ValueError, match="doc_idx"):
            next(tokenizing_distributed_data_loader_with_state(
                B=2, T=8, split="train", device="cpu",
                resume_state_dict={"pq_idx": 0, "rg_idx": 0, "doc_idx": -1}
            ))

        # Invalid token_buffer type
        with pytest.raises(ValueError, match="token_buffer"):
            next(tokenizing_distributed_data_loader_with_state(
                B=2, T=8, split="train", device="cpu",
                resume_state_dict={"pq_idx": 0, "rg_idx": 0, "token_buffer": "invalid_payload"}
            ))

        # Invalid token in token_buffer (e.g. negative or bool)
        with pytest.raises(ValueError, match="token_buffer"):
            next(tokenizing_distributed_data_loader_with_state(
                B=2, T=8, split="train", device="cpu",
                resume_state_dict={"pq_idx": 0, "rg_idx": 0, "token_buffer": [1, 2, -5]}
            ))

