from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

from bio_inspired_nanochat.torch_imports import torch
import pyarrow.parquet as pq

from bio_inspired_nanochat.common import get_dist_info
from bio_inspired_nanochat.dataset import parquet_paths_for_split
from bio_inspired_nanochat.tokenizer import get_tokenizer


def collate_dataloader_state_dicts(
    rank_states: Sequence[Mapping[str, Any]] | Mapping[int | str, Mapping[str, Any]],
    world_size: int | None = None,
) -> dict[str, Any]:
    """Collate per-rank dataloader state dicts into a multi-rank checkpoint structure."""
    if isinstance(rank_states, Sequence):
        ws = world_size or len(rank_states)
        ranks_dict = {str(i): dict(st) for i, st in enumerate(rank_states)}
    elif isinstance(rank_states, Mapping):
        ws = world_size or len(rank_states)
        ranks_dict = {str(k): dict(v) for k, v in rank_states.items()}
    else:
        raise ValueError("rank_states must be a sequence or mapping")

    return {
        "version": 2,
        "world_size": ws,
        "ranks": ranks_dict,
    }


def extract_rank_dataloader_state(
    resume_state_dict: Mapping[str, Any] | None,
    rank: int,
    world_size: int,
) -> dict[str, Any] | None:
    """Extract and validate the rank-local dataloader state from a single-rank or multi-rank dict.

    Supports:
    1. Multi-rank container dict with 'ranks' or 'rank_states' mapping/sequence.
    2. Single-rank state dict with optional 'rank' and 'world_size'.
    3. Legacy state dict (version 1: {'pq_idx': int, 'rg_idx': int}).

    Fails closed if the checkpoint world_size does not match current world_size,
    if the rank is missing from a multi-rank container, or if a single-rank state was recorded
    for a different rank.
    """
    if resume_state_dict is None:
        return None
    if not isinstance(resume_state_dict, Mapping):
        raise ValueError("resume_state_dict must be a mapping")

    # Check for multi-rank container
    ranks_container = resume_state_dict.get("ranks")
    if ranks_container is None:
        ranks_container = resume_state_dict.get("rank_states")

    if ranks_container is not None:
        if not isinstance(ranks_container, (Mapping, Sequence)):
            raise ValueError("resume_state_dict['ranks'] must be a mapping or sequence")

        saved_ws = resume_state_dict.get("world_size")
        if saved_ws is not None:
            if isinstance(saved_ws, bool) or not isinstance(saved_ws, int) or saved_ws <= 0:
                raise ValueError("resume_state_dict['world_size'] must be a positive integer")
            if saved_ws != world_size:
                raise ValueError(
                    f"Cannot resume dataloader: checkpoint world_size ({saved_ws}) "
                    f"does not match current world_size ({world_size})"
                )

        if isinstance(ranks_container, Mapping):
            rank_state = ranks_container.get(rank)
            if rank_state is None:
                rank_state = ranks_container.get(str(rank))
            if rank_state is None:
                raise ValueError(
                    f"Rank {rank} not found in resume_state_dict['ranks'] (available ranks: {list(ranks_container.keys())})"
                )
        else:  # Sequence
            if rank < 0 or rank >= len(ranks_container):
                raise ValueError(
                    f"Rank {rank} out of bounds for resume_state_dict['ranks'] with length {len(ranks_container)}"
                )
            rank_state = ranks_container[rank]

        if not isinstance(rank_state, Mapping):
            raise ValueError(f"State for rank {rank} must be a mapping")
        return dict(rank_state)

    # Single-rank state dict
    saved_ws = resume_state_dict.get("world_size")
    if saved_ws is not None:
        if isinstance(saved_ws, bool) or not isinstance(saved_ws, int) or saved_ws <= 0:
            raise ValueError("resume_state_dict['world_size'] must be a positive integer")
        if saved_ws != world_size:
            raise ValueError(
                f"Cannot resume dataloader: checkpoint world_size ({saved_ws}) "
                f"does not match current world_size ({world_size})"
            )

    saved_rank = resume_state_dict.get("rank")
    if saved_rank is not None:
        if isinstance(saved_rank, bool) or not isinstance(saved_rank, int) or saved_rank < 0:
            raise ValueError("resume_state_dict['rank'] must be a non-negative integer")
        if saved_rank != rank:
            raise ValueError(
                f"Cannot resume dataloader on rank {rank}: state was recorded for rank {saved_rank}"
            )

    return dict(resume_state_dict)


def tokenizing_distributed_data_loader_with_state(
    B: int,
    T: int,
    split: str,
    tokenizer_threads: int = 4,
    tokenizer_batch_size: int = 128,
    device: str = "cuda",
    resume_state_dict: Mapping[str, Any] | None = None,
):
    """
    Stream pretraining text from parquet files, tokenize, yield training batches.

    Provides exact, rank-local lossless resumption in distributed training:
    - Saves shard index (pq_idx), rank-local row-group index (rg_idx), in-group document offset (doc_idx),
      and unconsumed token buffer (token_buffer).
    - Supports both single-rank and multi-rank collated state dicts.
    - Validates world_size and rank bounds to fail closed on incompatible resume.
    - Maintains backward compatibility with legacy v1 state dicts.
    """
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    for name, value in (
        ("B", B),
        ("T", T),
        ("tokenizer_threads", tokenizer_threads),
        ("tokenizer_batch_size", tokenizer_batch_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    rank_state = extract_rank_dataloader_state(resume_state_dict, rank=ddp_rank, world_size=ddp_world_size)

    if rank_state is not None:
        for key in ("pq_idx", "rg_idx"):
            value = rank_state.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"resume_state_dict[{key!r}] must be a non-negative integer")

        if "doc_idx" in rank_state:
            val = rank_state["doc_idx"]
            if isinstance(val, bool) or not isinstance(val, int) or val < 0:
                raise ValueError("resume_state_dict['doc_idx'] must be a non-negative integer")

        if "token_buffer" in rank_state:
            val = rank_state["token_buffer"]
            if not isinstance(val, (list, tuple, deque)):
                raise ValueError("resume_state_dict['token_buffer'] must be a sequence of token IDs")
            for tok in val:
                if isinstance(tok, bool) or not isinstance(tok, int) or tok < 0:
                    raise ValueError("elements of resume_state_dict['token_buffer'] must be non-negative integers")

    is_v2 = rank_state is not None and (
        "doc_idx" in rank_state or "token_buffer" in rank_state or rank_state.get("version", 1) >= 2
    )

    resume_pq_idx = rank_state.get("pq_idx", 0) if rank_state is not None else 0
    resume_rg_idx = rank_state.get("rg_idx") if rank_state is not None else None
    resume_doc_idx = rank_state.get("doc_idx", 0) if rank_state is not None else 0
    resume_tokens = list(rank_state.get("token_buffer", [])) if rank_state is not None else []

    def document_batches():
        parquet_paths = parquet_paths_for_split(split)
        if parquet_paths and resume_pq_idx >= len(parquet_paths):
            raise ValueError(
                "resume_state_dict['pq_idx'] is outside the available parquet shard range"
            )
        pq_idx = resume_pq_idx
        first_shard = True
        while True:  # iterate infinitely (multi-epoch)
            if not parquet_paths:
                raise RuntimeError("No parquet files found for split: " + split)
            while pq_idx < len(parquet_paths):  # iterate over all parquet files
                filepath = parquet_paths[pq_idx]
                pf = pq.ParquetFile(filepath)

                if first_shard and resume_rg_idx is not None:
                    if resume_rg_idx >= pf.num_row_groups:
                        raise ValueError(
                            "resume_state_dict['rg_idx'] is outside the parquet row-group range"
                        )
                    if is_v2:
                        rg_idx = resume_rg_idx
                        doc_start = resume_doc_idx
                    else:
                        base_idx = resume_rg_idx // ddp_world_size
                        base_idx += 1
                        rg_idx = base_idx * ddp_world_size + ddp_rank
                        doc_start = 0
                else:
                    rg_idx = ddp_rank
                    doc_start = 0

                first_shard = False

                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    batch = rg.column("text").to_pylist()
                    num_docs = len(batch)

                    for i in range(doc_start, num_docs, tokenizer_batch_size):
                        chunk = batch[i : i + tokenizer_batch_size]
                        next_doc_idx = i + len(chunk)
                        yield chunk, (pq_idx, rg_idx, next_doc_idx)

                    doc_start = 0
                    rg_idx += ddp_world_size
                pq_idx += 1
            pq_idx = 0

    batches = document_batches()

    needed_tokens = B * T + 1
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    token_buffer = deque(resume_tokens)

    curr_pq_idx = resume_pq_idx
    curr_rg_idx = resume_rg_idx if resume_rg_idx is not None else ddp_rank
    curr_doc_idx = resume_doc_idx

    while True:
        while len(token_buffer) < needed_tokens:
            doc_batch, (curr_pq_idx, curr_rg_idx, curr_doc_idx) = next(batches)
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
            for tokens in token_lists:
                token_buffer.extend(tokens)

        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        device_obj = torch.device(device)
        use_cuda_optimizations = device_obj.type == "cuda"
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations)
        inputs_cpu = scratch[:-1]
        targets_cpu = scratch[1:]
        inputs = inputs_cpu.view(B, T).to(device=device_obj, non_blocking=use_cuda_optimizations)
        targets = targets_cpu.view(B, T).to(device=device_obj, non_blocking=use_cuda_optimizations)
        state_dict = {
            "version": 2,
            "pq_idx": curr_pq_idx,
            "rg_idx": curr_rg_idx,
            "doc_idx": curr_doc_idx,
            "token_buffer": list(token_buffer),
            "rank": ddp_rank,
            "world_size": ddp_world_size,
        }
        yield inputs, targets, state_dict


def tokenizing_distributed_data_loader(*args, **kwargs):
    # helper function that only emits the inputs/targets and not the state_dict
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets
