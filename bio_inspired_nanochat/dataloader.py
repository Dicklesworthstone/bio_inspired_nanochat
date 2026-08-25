from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any, cast

from bio_inspired_nanochat.torch_imports import torch
import pyarrow.parquet as pq

from bio_inspired_nanochat.common import get_dist_info
from bio_inspired_nanochat.dataset import parquet_paths_for_split
from bio_inspired_nanochat.tokenizer import get_tokenizer


def _validate_rank_state_ownership(
    state: Mapping[str, Any],
    *,
    rank: int,
    world_size: int,
) -> None:
    saved_rank = state.get("rank")
    if saved_rank is not None:
        if isinstance(saved_rank, bool) or not isinstance(saved_rank, int) or saved_rank < 0:
            raise ValueError("resume_state_dict['rank'] must be a non-negative integer")
        if saved_rank != rank:
            raise ValueError(
                f"Cannot resume dataloader on rank {rank}: state was recorded for rank {saved_rank}"
            )
    saved_world_size = state.get("world_size")
    if saved_world_size is not None:
        if (
            isinstance(saved_world_size, bool)
            or not isinstance(saved_world_size, int)
            or saved_world_size <= 0
        ):
            raise ValueError("resume_state_dict['world_size'] must be a positive integer")
        if saved_world_size != world_size:
            raise ValueError(
                f"Cannot resume dataloader: checkpoint world_size ({saved_world_size}) "
                f"does not match current world_size ({world_size})"
            )


def collate_dataloader_state_dicts(
    rank_states: Sequence[Mapping[str, Any]] | Mapping[int | str, Mapping[str, Any]],
    world_size: int | None = None,
) -> dict[str, Any]:
    """Collate per-rank dataloader state dicts into a multi-rank checkpoint structure."""
    if isinstance(world_size, bool) or (
        world_size is not None and (not isinstance(world_size, int) or world_size <= 0)
    ):
        raise ValueError("world_size must be a positive integer")
    if isinstance(rank_states, Sequence) and not isinstance(
        rank_states, (str, bytes, bytearray)
    ):
        ws = len(rank_states) if world_size is None else world_size
        if len(rank_states) != ws:
            raise ValueError(
                f"rank state count {len(rank_states)} does not match world_size {ws}"
            )
        ranks_dict: dict[str, dict[str, Any]] = {}
        for rank, state in enumerate(rank_states):
            if not isinstance(state, Mapping):
                raise ValueError(f"State for rank {rank} must be a mapping")
            typed_state = cast(Mapping[str, Any], state)
            _validate_rank_state_ownership(typed_state, rank=rank, world_size=ws)
            ranks_dict[str(rank)] = dict(typed_state)
    elif isinstance(rank_states, Mapping):
        ws = len(rank_states) if world_size is None else world_size
        ranks_dict = {}
        for raw_rank, state in rank_states.items():
            if isinstance(raw_rank, bool) or not isinstance(raw_rank, (int, str)):
                raise ValueError("rank state keys must be non-negative integer ranks")
            try:
                rank = int(raw_rank)
            except ValueError as error:
                raise ValueError(
                    "rank state keys must be non-negative integer ranks"
                ) from error
            if rank < 0 or str(rank) in ranks_dict:
                raise ValueError(f"duplicate or invalid rank state key: {raw_rank!r}")
            if not isinstance(state, Mapping):
                raise ValueError(f"State for rank {rank} must be a mapping")
            typed_state = cast(Mapping[str, Any], state)
            _validate_rank_state_ownership(typed_state, rank=rank, world_size=ws)
            ranks_dict[str(rank)] = dict(typed_state)
    else:
        raise ValueError("rank_states must be a sequence or mapping")

    if ws <= 0:
        raise ValueError("rank_states must contain at least one rank")
    expected_ranks = {str(rank) for rank in range(ws)}
    if set(ranks_dict) != expected_ranks:
        raise ValueError(
            "rank states must contain exactly one state for every rank in "
            f"[0, {ws}); got {sorted(ranks_dict)}"
        )

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
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError("rank must be a non-negative integer")
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size <= 0
    ):
        raise ValueError("world_size must be a positive integer")
    if rank >= world_size:
        raise ValueError(f"rank {rank} must be smaller than world_size {world_size}")
    if resume_state_dict is None:
        return None
    if not isinstance(resume_state_dict, Mapping):
        raise ValueError("resume_state_dict must be a mapping")

    # Check for multi-rank container
    ranks_container = resume_state_dict.get("ranks")
    if ranks_container is None:
        ranks_container = resume_state_dict.get("rank_states")

    if ranks_container is not None:
        if resume_state_dict.get("version") != 2:
            raise ValueError("multi-rank resume_state_dict['version'] must be 2")
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
        _validate_rank_state_ownership(rank_state, rank=rank, world_size=world_size)
        return dict(rank_state)

    # Single-rank state dict
    _validate_rank_state_ownership(resume_state_dict, rank=rank, world_size=world_size)
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
        version = rank_state.get("version", 1)
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version not in {1, 2}
        ):
            raise ValueError("resume_state_dict['version'] must be 1 or 2")
        if version == 2:
            required_v2_fields = {
                "pq_idx",
                "rg_idx",
                "doc_idx",
                "token_buffer",
                "rank",
                "world_size",
            }
            missing_v2_fields = sorted(required_v2_fields - rank_state.keys())
            if missing_v2_fields:
                raise ValueError(
                    "version 2 resume_state_dict is missing exact-resume fields: "
                    f"{missing_v2_fields}"
                )
        elif "doc_idx" in rank_state or "token_buffer" in rank_state:
            raise ValueError("doc_idx/token_buffer resume state requires version 2")
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

    is_v2 = rank_state is not None and rank_state.get("version", 1) == 2

    resume_pq_idx = rank_state.get("pq_idx", 0) if rank_state is not None else 0
    resume_rg_idx = rank_state.get("rg_idx") if rank_state is not None else None
    resume_doc_idx = rank_state.get("doc_idx", 0) if rank_state is not None else 0
    resume_tokens = list(rank_state.get("token_buffer", [])) if rank_state is not None else []
    if is_v2 and resume_rg_idx is not None and resume_rg_idx % ddp_world_size != ddp_rank:
        raise ValueError(
            "resume_state_dict['rg_idx'] is not owned by the saved rank/world_size"
        )

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
                    if doc_start > num_docs:
                        raise ValueError(
                            "resume_state_dict['doc_idx'] is outside the parquet row-group range"
                        )

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
