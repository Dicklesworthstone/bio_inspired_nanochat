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
        # Move tensor-valued meta (e.g. gold ``answers``) too, so GPU eval doesn't hit a
        # CPU/GPU device mismatch when comparing predictions to the gold answers.
        meta = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in self.meta.items()}
        return SyntheticBatch(self.inputs.to(device), self.targets.to(device), meta)


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
    # Three DISJOINT token bands so the needle key/value can never appear in the filler —
    # otherwise the retrieval would be ambiguous and a perfect model couldn't score 100%.
    band = max(1, ct.content_vocab // 4)
    key = _randint(gen, band, (batch, 1))                     # key   in [0, band)
    val = _randint(gen, band, (batch, 1)) + band              # value in [band, 2*band)
    filler_lo = 2 * band
    filler = _randint(gen, ct.content_vocab - filler_lo, (batch, haystack_len)) + filler_lo  # [2*band, content)
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


# --------------------------------------------------------------------------- #
# Retrieval evaluation (bead 74f.2) — run a model on a task and score the answer
# --------------------------------------------------------------------------- #
@torch.no_grad()
def retrieval_accuracy(model: Any, batch: SyntheticBatch) -> float:
    """Fraction of rows where the model's argmax at ``answer_pos`` equals the gold answer.

    Works for any task whose ``meta`` carries ``answer_pos`` + ``answers`` (niah,
    associative_recall, variable_binding). The models are no-shift — ``logits[:, t]``
    predicts ``targets[:, t]`` — so the prediction is read directly at ``answer_pos``.
    """
    out = model(batch.inputs)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    ap = int(batch.meta["answer_pos"])
    pred = logits[:, ap, :].argmax(dim=-1)
    gold = batch.meta["answers"].to(pred.device)
    return float((pred == gold).float().mean().item())


@torch.no_grad()
def niah_accuracy_by_length(
    model: Any,
    *,
    vocab_size: int,
    lengths: tuple[int, ...] = (16, 64, 128),
    depth_fracs: tuple[float, ...] = (0.1, 0.5, 0.9),
    batch: int = 32,
    seed: int = 0,
    device: Any = None,
) -> dict[str, Any]:
    """Needle-in-a-haystack accuracy swept over haystack length × needle depth.

    For each haystack length the needle is placed at every ``depth_frac`` (begin/middle/
    end) and accuracies are averaged, giving the canonical accuracy-by-length curve.
    Fast-weight state is reset per batch so each retrieval is independent. Returns
    ``{"by_length": {L: acc}, "overall": mean_acc}``.
    """
    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()
    by_length: dict[int, float] = {}
    for length in lengths:
        accs = []
        for i, df in enumerate(depth_fracs):
            b = needle_in_haystack(
                batch=batch,
                haystack_len=length,
                vocab_size=vocab_size,
                depth_frac=df,
                seed=seed + 1000 * length + i,
            )
            if device is not None:
                b = b.to(device)
            # Fresh fast-weights per batch so the needle can't leak across evals.
            if hasattr(model, "reset_sequence_state"):
                model.reset_sequence_state(reset_fast_weights=True)
            accs.append(retrieval_accuracy(model, b))
        by_length[length] = sum(accs) / len(accs)
    if was_training and hasattr(model, "train"):
        model.train()
    overall = sum(by_length.values()) / len(by_length) if by_length else float("nan")
    return {"by_length": by_length, "overall": overall}


# --------------------------------------------------------------------------- #
# Working-memory evaluation suite (sax.4)
# --------------------------------------------------------------------------- #


def _reset_fast_state(model: Any) -> None:
    """Wipe per-sequence fast-weight/eligibility state so each eval batch is independent."""
    if hasattr(model, "reset_sequence_state"):
        model.reset_sequence_state(reset_fast_weights=True)


def _model_context(model: Any) -> Any:
    """The model's max context length (``config.sequence_len``), or None if unknown."""
    return getattr(getattr(model, "config", None), "sequence_len", None)


def _fits(batch: SyntheticBatch, context: Any) -> bool:
    """Whether ``batch`` fits the model context. Unknown context => assume it fits (proceed)."""
    return context is None or int(batch.inputs.shape[1]) <= int(context)


@torch.no_grad()
def recall_accuracy_by_pairs(
    model: Any,
    *,
    vocab_size: int,
    num_pairs: tuple[int, ...] = (2, 4, 8, 16),
    batch: int = 32,
    seed: int = 0,
    device: Any = None,
) -> dict[str, Any]:
    """Associative-recall accuracy swept over the number of key→value pairs (memory load).

    More pairs ⇒ more to hold in working memory before the query, so the accuracy-by-load curve
    is a direct probe of recall capacity. Fast-weight state is reset per batch so retrievals are
    independent. Points whose generated sequence exceeds the model context are skipped (omitted
    from the curve) rather than crashing. Returns ``{"by_pairs": {n: acc}, "overall": mean_acc}``.
    """
    context = _model_context(model)
    by_pairs: dict[int, float] = {}
    for n in num_pairs:
        b = associative_recall(batch=batch, num_pairs=n, vocab_size=vocab_size, seed=seed + n)
        if not _fits(b, context):
            continue
        if device is not None:
            b = b.to(device)
        _reset_fast_state(model)
        by_pairs[n] = retrieval_accuracy(model, b)
    overall = sum(by_pairs.values()) / len(by_pairs) if by_pairs else float("nan")
    return {"by_pairs": by_pairs, "overall": overall}


@torch.no_grad()
def binding_accuracy_by_distractors(
    model: Any,
    *,
    vocab_size: int,
    num_distractors: tuple[int, ...] = (0, 4, 16, 64),
    num_vars: int = 3,
    batch: int = 32,
    seed: int = 0,
    device: Any = None,
) -> dict[str, Any]:
    """Variable-binding accuracy swept over the number of DISTRACTOR tokens between the binding
    and the query — the "infinite local context" stress test.

    A binding ("zibzab has purple fur") must survive an increasing number of irrelevant tokens
    before the query. The accuracy-vs-distractors curve measures how far back the model can bind;
    a model with genuine working memory degrades gracefully, a context-window model falls off a
    cliff. Points whose generated sequence exceeds the model context are skipped (omitted from the
    curve) rather than crashing. Returns ``{"by_distractors": {n: acc}, "overall": mean_acc}``.
    """
    context = _model_context(model)
    by_distractors: dict[int, float] = {}
    for n in num_distractors:
        b = variable_binding(
            batch=batch, num_vars=num_vars, num_distractors=n, vocab_size=vocab_size, seed=seed + n
        )
        if not _fits(b, context):
            continue
        if device is not None:
            b = b.to(device)
        _reset_fast_state(model)
        by_distractors[n] = retrieval_accuracy(model, b)
    overall = sum(by_distractors.values()) / len(by_distractors) if by_distractors else float("nan")
    return {"by_distractors": by_distractors, "overall": overall}


@torch.no_grad()
def working_memory_suite(
    model: Any,
    *,
    vocab_size: int,
    recall_pairs: tuple[int, ...] = (2, 4, 8, 16),
    binding_distractors: tuple[int, ...] = (0, 4, 16, 64),
    niah_lengths: tuple[int, ...] = (16, 64, 128),
    batch: int = 32,
    seed: int = 0,
    device: Any = None,
) -> dict[str, Any]:
    """The working-memory evaluation suite (sax.4): a single entry point that tests the
    "infinite local context" claim with clean, by-difficulty curves.

    Runs three complementary probes — associative recall by memory load, variable binding by
    distractor count, and needle-in-a-haystack by length — each with per-batch fast-weight resets
    so retrievals are independent. Model-agnostic: works on bio (synaptic) and vanilla models, so
    it drives bio-vs-vanilla and fast-weight-baseline comparisons. Returns the three curves plus a
    flat ``summary`` of overall accuracies (the headline numbers).

    Eval points whose generated sequence would exceed the model context are skipped (omitted from
    their curve) rather than crashing, so the suite runs on any model size — only the points that
    fit are measured (the by-difficulty dict keys show exactly which).
    """
    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()

    recall = recall_accuracy_by_pairs(
        model, vocab_size=vocab_size, num_pairs=recall_pairs, batch=batch, seed=seed, device=device
    )
    binding = binding_accuracy_by_distractors(
        model, vocab_size=vocab_size, num_distractors=binding_distractors,
        batch=batch, seed=seed, device=device,
    )
    # NIAH generates a needle+query around the haystack (sequence length = L + 2), so keep only the
    # lengths that fit the model context.
    context = _model_context(model)
    niah_fit = tuple(L for L in niah_lengths if context is None or L + 2 <= int(context))
    niah = niah_accuracy_by_length(
        model, vocab_size=vocab_size, lengths=niah_fit, batch=batch, seed=seed, device=device
    )

    if was_training and hasattr(model, "train"):
        model.train()

    return {
        "recall": recall,
        "binding": binding,
        "niah": niah,
        "summary": {
            "recall_overall": recall["overall"],
            "binding_overall": binding["overall"],
            "niah_overall": niah["overall"],
        },
    }
