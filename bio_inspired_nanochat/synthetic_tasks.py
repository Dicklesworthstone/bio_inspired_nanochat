"""
Shared synthetic-task generators (bead eqyk.3).

ONE deterministic library of small, token-level synthetic tasks, used by BOTH the test
harness and the science evals so they stay consistent and CI-cheap. Today the CMA-ES
proxy, the working-memory suite, NIAH, and the continual benchmark each imply their own
generators; this unifies them.

Conventions
-----------
- All tasks are next-token-prediction over integer token ids in ``[0, vocab_size)``.
- ``targets[i]`` is the token the model should predict at position ``i`` (i.e. it equals
  ``inputs[i+1]`` at supervised positions). Non-supervised positions use ``IGNORE_INDEX``
  (= -1), matching the model's ``F.cross_entropy(..., ignore_index=-1)``.
- The TOP ids of the vocab are reserved as control tokens (SEP, QUERY); content tokens are
  drawn from ``[0, content_vocab)``. So a task needs ``vocab_size`` a bit larger than its
  content alphabet.
- Determinism: every generator takes ``seed`` (or a ``torch.Generator``) and uses ONLY
  that generator — never the global RNG. Same seed -> identical batch.

Quick use
---------
    from bio_inspired_nanochat.synthetic_tasks import make_task, associative_recall
    b = associative_recall(batch=8, num_pairs=4, vocab_size=64, seed=0)
    logits, loss = model(b.inputs, targets=b.targets)        # loss only at b.answer_pos
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable

import torch
from torch import Tensor

IGNORE_INDEX = -1  # matches F.cross_entropy(ignore_index=-1) used across the models


# --------------------------------------------------------------------------- #
# Batch container + helpers
# --------------------------------------------------------------------------- #
@dataclass
class SyntheticBatch:
    """A synthetic next-token batch. ``meta`` carries task-specific structure
    (answer positions, the gold answer ids, query keys, ...) for evaluation."""

    inputs: Tensor                 # (B, T) long
    targets: Tensor                # (B, T) long, IGNORE_INDEX where unsupervised
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return self.inputs.shape[0]

    @property
    def seq_len(self) -> int:
        return self.inputs.shape[1]

    def to(self, device) -> "SyntheticBatch":
        return SyntheticBatch(self.inputs.to(device), self.targets.to(device), dict(self.meta))


def _generator(seed: int | torch.Generator | None) -> torch.Generator:
    if isinstance(seed, torch.Generator):
        return seed
    g = torch.Generator()
    g.manual_seed(0 if seed is None else int(seed))
    return g


def control_tokens(vocab_size: int, n_special: int = 2) -> SimpleNamespace:
    """Reserve the top ``n_special`` ids as control tokens; content lives below them."""
    assert vocab_size > n_special + 2, f"vocab_size {vocab_size} too small for {n_special} control tokens"
    return SimpleNamespace(
        SEP=vocab_size - 1,
        QUERY=vocab_size - 2,
        content_vocab=vocab_size - n_special,
    )


def _randint(gen: torch.Generator, high: int, shape: tuple[int, ...]) -> Tensor:
    return torch.randint(0, high, shape, generator=gen, dtype=torch.long)


def _unique_per_row(gen: torch.Generator, batch: int, k: int, high: int) -> Tensor:
    """``(batch, k)`` of distinct ids per row, sampled from ``[0, high)`` (k <= high)."""
    assert k <= high, f"cannot draw {k} distinct ids from [0,{high})"
    rows = [torch.randperm(high, generator=gen)[:k] for _ in range(batch)]
    return torch.stack(rows, dim=0)


# --------------------------------------------------------------------------- #
# 1. Copy / echo
# --------------------------------------------------------------------------- #
def copy_task(*, batch: int = 8, length: int = 8, vocab_size: int = 64, seed=0) -> SyntheticBatch:
    """``[data, SEP, data]`` — predict the second copy. Tests verbatim short-term recall.

    Supervised only on the second copy (positions ``[length, 2*length)``), so a model that
    can't copy scores chance there.
    """
    ct = control_tokens(vocab_size)
    gen = _generator(seed)
    data = _randint(gen, ct.content_vocab, (batch, length))
    sep = torch.full((batch, 1), ct.SEP, dtype=torch.long)
    inputs = torch.cat([data, sep, data], dim=1)            # (B, 2L+1)
    targets = torch.full_like(inputs, IGNORE_INDEX)
    # position i predicts inputs[i+1]; supervise the second copy.
    targets[:, length:2 * length] = data
    return SyntheticBatch(inputs, targets, {
        "task": "copy", "answer_start": length, "answer_len": length,
    })


# --------------------------------------------------------------------------- #
# 2. Associative recall
# --------------------------------------------------------------------------- #
def associative_recall(*, batch: int = 8, num_pairs: int = 4, vocab_size: int = 64, seed=0) -> SyntheticBatch:
    """``[k1 v1 k2 v2 ... QUERY kq]`` — predict ``vq`` (the value paired with the queried key).

    Keys are distinct within a sample; the answer is supervised at the final position.
    """
    ct = control_tokens(vocab_size)
    gen = _generator(seed)
    half = ct.content_vocab // 2
    keys = _unique_per_row(gen, batch, num_pairs, half)              # keys in [0, half)
    vals = _unique_per_row(gen, batch, num_pairs, half) + half       # values in [half, content)
    pairs = torch.stack([keys, vals], dim=2).reshape(batch, 2 * num_pairs)  # k1 v1 k2 v2 ...
    q_idx = _randint(gen, num_pairs, (batch, 1))                     # which pair to query
    q_key = torch.gather(keys, 1, q_idx)                            # (B,1)
    q_val = torch.gather(vals, 1, q_idx)                            # (B,1) gold answer
    query = torch.full((batch, 1), ct.QUERY, dtype=torch.long)
    inputs = torch.cat([pairs, query, q_key], dim=1)                # (B, 2P+2)
    targets = torch.full_like(inputs, IGNORE_INDEX)
    answer_pos = inputs.shape[1] - 1                                # predict vq after kq
    targets[:, answer_pos] = q_val.squeeze(1)
    return SyntheticBatch(inputs, targets, {
        "task": "associative_recall", "answer_pos": answer_pos,
        "answers": q_val.squeeze(1), "query_keys": q_key.squeeze(1),
    })


# --------------------------------------------------------------------------- #
# 3. Variable binding (recall with distractors): "X has property Y"
# --------------------------------------------------------------------------- #
def variable_binding(
    *, batch: int = 8, num_vars: int = 3, num_distractors: int = 8, vocab_size: int = 96, seed=0,
) -> SyntheticBatch:
    """Var/value pairs interspersed with DISTRACTOR tokens, then query a var -> predict its value.

    Harder than plain recall: the binding must survive irrelevant tokens in between, the
    core of compositional binding (and the needle-with-distractors flavor of long context).
    """
    ct = control_tokens(vocab_size)
    gen = _generator(seed)
    third = ct.content_vocab // 3
    var_ids = _unique_per_row(gen, batch, num_vars, third)                  # vars in [0, third)
    val_ids = _unique_per_row(gen, batch, num_vars, third) + third          # values in [third, 2third)
    # distractors come from the top content band so they never collide with vars/values.
    distractors = _randint(gen, ct.content_vocab - 2 * third, (batch, num_distractors)) + 2 * third

    rows_in, rows_meta = [], []
    for b in range(batch):
        # Build ITEMS: each binding is a contiguous 2-token [var, value] unit; each
        # distractor is a 1-token unit. We shuffle the ITEM ORDER (not tokens), so a
        # binding pair is never split — keeping the task well-posed (the value always
        # immediately follows its var) while distractors sit BETWEEN pairs.
        items: list[list[int]] = [[int(var_ids[b, j]), int(val_ids[b, j])] for j in range(num_vars)]
        items += [[int(d)] for d in distractors[b].tolist()]
        perm = torch.randperm(len(items), generator=gen).tolist()
        toks: list[int] = []
        for idx in perm:
            toks += items[idx]
        qj = int(torch.randint(0, num_vars, (1,), generator=gen).item())
        toks += [ct.QUERY, int(var_ids[b, qj].item())]
        rows_in.append(toks)
        rows_meta.append(int(val_ids[b, qj].item()))
    T = len(rows_in[0])
    inputs = torch.tensor(rows_in, dtype=torch.long)
    targets = torch.full((batch, T), IGNORE_INDEX, dtype=torch.long)
    answers = torch.tensor(rows_meta, dtype=torch.long)
    targets[:, T - 1] = answers
    return SyntheticBatch(inputs, targets, {
        "task": "variable_binding", "answer_pos": T - 1, "answers": answers,
    })


# --------------------------------------------------------------------------- #
# 4. Needle in a haystack (length-parametrized)
# --------------------------------------------------------------------------- #
def needle_in_haystack(
    *, batch: int = 8, haystack_len: int = 64, vocab_size: int = 64, depth_frac: float = 0.5, seed=0,
) -> SyntheticBatch:
    """A ``key value`` needle buried at ``depth_frac`` in filler, then ``QUERY key`` -> predict value.

    The canonical long-context retrieval probe; sweep ``haystack_len``/``depth_frac`` for
    an accuracy-by-length curve (consumed by NIAH eval 74f.2).
    """
    ct = control_tokens(vocab_size)
    gen = _generator(seed)
    half = ct.content_vocab // 2
    key = _randint(gen, half, (batch, 1))                     # key in [0, half)
    val = _randint(gen, half, (batch, 1)) + half              # value in [half, content)
    # filler tokens avoid the key id so the needle key is unambiguous.
    filler = _randint(gen, ct.content_vocab, (batch, haystack_len))
    depth = max(0, min(haystack_len - 2, int(round(depth_frac * (haystack_len - 2)))))
    filler[:, depth:depth + 1] = key                         # place the needle key
    filler[:, depth + 1:depth + 2] = val                     # ...followed by its value
    query = torch.full((batch, 1), ct.QUERY, dtype=torch.long)
    inputs = torch.cat([filler, query, key], dim=1)          # (B, H+2)
    targets = torch.full_like(inputs, IGNORE_INDEX)
    answer_pos = inputs.shape[1] - 1
    targets[:, answer_pos] = val.squeeze(1)
    return SyntheticBatch(inputs, targets, {
        "task": "niah", "answer_pos": answer_pos, "answers": val.squeeze(1),
        "needle_depth": depth, "haystack_len": haystack_len, "depth_frac": depth_frac,
    })


# --------------------------------------------------------------------------- #
# 5. Continual-learning task sequence
# --------------------------------------------------------------------------- #
def continual_task_sequence(
    *, num_tasks: int = 3, batch: int = 8, length: int = 8, vocab_size: int = 96, seed=0,
) -> list[SyntheticBatch]:
    """A list of ``num_tasks`` DISTINCT copy tasks over DISJOINT vocab bands.

    Disjoint alphabets make the tasks genuinely different (learn A then B, then re-test A to
    measure forgetting). Consumed by the continual benchmark (cel.4).
    """
    ct = control_tokens(vocab_size)
    band = ct.content_vocab // num_tasks
    assert band >= length, f"vocab too small: band {band} < length {length}"
    gen = _generator(seed)
    out: list[SyntheticBatch] = []
    for t in range(num_tasks):
        lo = t * band
        data = _randint(gen, band, (batch, length)) + lo     # tokens in task t's band
        sep = torch.full((batch, 1), ct.SEP, dtype=torch.long)
        inputs = torch.cat([data, sep, data], dim=1)
        targets = torch.full_like(inputs, IGNORE_INDEX)
        targets[:, length:2 * length] = data
        out.append(SyntheticBatch(inputs, targets, {
            "task": "continual_copy", "task_id": t, "vocab_band": (lo, lo + band),
            "answer_start": length, "answer_len": length,
        }))
    return out


# --------------------------------------------------------------------------- #
# 6. Reward task (for RL / intrinsic-motivation experiments)
# --------------------------------------------------------------------------- #
def reward_task(
    *, batch: int = 8, context_len: int = 4, vocab_size: int = 64, seed=0,
) -> tuple[SyntheticBatch, Callable[[Tensor], Tensor]]:
    """A minimal contextual-bandit: predict the gold answer (here: the FIRST context token,
    a 'copy-first' rule) given a short context. Returns the batch (answer supervised at the
    last position) and a ``reward_fn(pred_token_ids) -> (B,) float`` (1.0 if correct).

    Used by the neuromodulated-RL / curiosity beads (hy8.3, M12); the deterministic rule
    keeps tests cheap while exercising a reward signal.
    """
    ct = control_tokens(vocab_size)
    gen = _generator(seed)
    context = _randint(gen, ct.content_vocab, (batch, context_len))
    gold = context[:, 0].clone()                              # rule: answer = first token
    query = torch.full((batch, 1), ct.QUERY, dtype=torch.long)
    inputs = torch.cat([context, query], dim=1)               # (B, C+1)
    targets = torch.full_like(inputs, IGNORE_INDEX)
    answer_pos = inputs.shape[1] - 1
    targets[:, answer_pos] = gold

    def reward_fn(pred_token_ids: Tensor) -> Tensor:
        """``pred_token_ids`` is ``(B,)`` predicted token ids; reward 1.0 when == gold."""
        return (pred_token_ids.to(gold.device) == gold).float()

    return SyntheticBatch(inputs, targets, {
        "task": "reward", "rule": "copy_first", "answer_pos": answer_pos, "answers": gold,
    }), reward_fn


# --------------------------------------------------------------------------- #
# Registry (lets the eval matrix request tasks by name)
# --------------------------------------------------------------------------- #
TASKS: dict[str, Callable[..., Any]] = {
    "copy": copy_task,
    "associative_recall": associative_recall,
    "variable_binding": variable_binding,
    "niah": needle_in_haystack,
    "continual": continual_task_sequence,
    "reward": reward_task,
}


def make_task(name: str, **kwargs: Any):
    """Dispatch to a generator by name (for config-driven evals). Raises on unknown name."""
    if name not in TASKS:
        raise KeyError(f"unknown synthetic task '{name}'; choices: {sorted(TASKS)}")
    return TASKS[name](**kwargs)
