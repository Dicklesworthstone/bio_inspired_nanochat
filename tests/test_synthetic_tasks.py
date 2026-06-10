"""
Tests for the shared synthetic-task generators (bead eqyk.3).

Beyond shapes/determinism, these verify the tasks are *well-posed*: the supervised gold
answer is actually recoverable from the input (so a perfect model could score 100%), and
loss is masked to exactly the answer positions. A tiny model consumes each batch to prove
the whole library is model-compatible.

Run:  pytest tests/test_synthetic_tasks.py -v
"""

from __future__ import annotations

import pytest
import torch

from bio_inspired_nanochat.synthetic_tasks import (
    IGNORE_INDEX,
    SyntheticBatch,
    associative_recall,
    continual_task_sequence,
    copy_task,
    make_task,
    needle_in_haystack,
    reward_task,
    variable_binding,
)

from _bio_testkit import make_tiny_synaptic

VOCAB = 64


def _all_tokens_in_range(b: SyntheticBatch, vocab: int):
    assert b.inputs.min().item() >= 0 and b.inputs.max().item() < vocab
    # targets are token ids OR IGNORE_INDEX
    valid = b.targets[b.targets != IGNORE_INDEX]
    if valid.numel():
        assert valid.min().item() >= 0 and valid.max().item() < vocab


# ----------------------------------------------------------------------- #
# Determinism + token-range (parametrized over every task)
# ----------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize("name,kwargs", [
    ("copy", dict(batch=4, length=6, vocab_size=VOCAB)),
    ("associative_recall", dict(batch=4, num_pairs=3, vocab_size=VOCAB)),
    ("variable_binding", dict(batch=4, num_vars=3, num_distractors=5, vocab_size=96)),
    ("niah", dict(batch=4, haystack_len=20, vocab_size=VOCAB)),
    ("reward", dict(batch=4, context_len=3, vocab_size=VOCAB)),
])
def test_deterministic_and_in_range(name, kwargs):
    r1 = make_task(name, seed=7, **kwargs)
    r2 = make_task(name, seed=7, **kwargs)
    b1 = r1[0] if isinstance(r1, tuple) else r1
    b2 = r2[0] if isinstance(r2, tuple) else r2
    assert torch.equal(b1.inputs, b2.inputs) and torch.equal(b1.targets, b2.targets)
    # a different seed should (almost surely) differ
    b3 = make_task(name, seed=8, **kwargs)
    b3 = b3[0] if isinstance(b3, tuple) else b3
    assert not torch.equal(b1.inputs, b3.inputs)
    _all_tokens_in_range(b1, kwargs["vocab_size"])
    # exactly one supervised position per row for the QA-style tasks; copy supervises a span
    sup_per_row = (b1.targets != IGNORE_INDEX).sum(dim=1)
    assert (sup_per_row > 0).all()


# ----------------------------------------------------------------------- #
# Per-task correctness: gold answer recoverable from the input
# ----------------------------------------------------------------------- #
@pytest.mark.unit
def test_copy_second_half_equals_first_and_is_supervised():
    b = copy_task(batch=4, length=6, vocab_size=VOCAB, seed=0)
    assert b.seq_len == 2 * 6 + 1
    first, second = b.inputs[:, :6], b.inputs[:, 7:13]
    assert torch.equal(first, second), "the two copies must match"
    # supervised exactly on positions [6, 12); targets there equal the data
    assert torch.equal(b.targets[:, 6:12], first)
    mask = torch.ones(b.seq_len, dtype=torch.bool)
    mask[6:12] = False
    assert (b.targets[:, mask] == IGNORE_INDEX).all()


@pytest.mark.unit
def test_associative_recall_answer_is_the_paired_value():
    b = associative_recall(batch=16, num_pairs=4, vocab_size=VOCAB, seed=1)
    ap = b.meta["answer_pos"]
    pairs = b.inputs[:, :2 * 4]               # k1 v1 k2 v2 ...
    keys, vals = pairs[:, 0::2], pairs[:, 1::2]
    for i in range(b.batch_size):
        qk = b.meta["query_keys"][i]
        j = (keys[i] == qk).nonzero(as_tuple=True)[0]
        assert j.numel() == 1, "queried key must be unique"
        assert vals[i, j].item() == b.meta["answers"][i].item()
        assert b.targets[i, ap].item() == b.meta["answers"][i].item()


@pytest.mark.unit
def test_variable_binding_answer_recoverable_despite_distractors():
    b = variable_binding(batch=16, num_vars=3, num_distractors=8, vocab_size=96, seed=2)
    ap = b.meta["answer_pos"]                      # queried var sits at the final position
    for i in range(b.batch_size):
        seq = b.inputs[i].tolist()
        qvar, ans = seq[ap], b.meta["answers"][i].item()
        # the var appears exactly once in its (contiguous) binding pair, followed by its
        # value — distractors never split it. Exclude the query occurrence at ap.
        binding = [p for p in range(len(seq) - 1) if seq[p] == qvar and p != ap]
        assert len(binding) == 1, "var must appear once in a binding pair (plus the query)"
        assert seq[binding[0] + 1] == ans, "the value immediately follows its var"


@pytest.mark.unit
def test_niah_needle_placed_and_answerable():
    b = needle_in_haystack(batch=8, haystack_len=24, vocab_size=VOCAB, depth_frac=0.5, seed=3)
    depth = b.meta["needle_depth"]
    key = b.inputs[:, -1]                        # query key is the last token
    needle_key = b.inputs[:, depth]
    needle_val = b.inputs[:, depth + 1]
    assert torch.equal(key, needle_key), "query key must equal the buried needle key"
    assert torch.equal(needle_val, b.meta["answers"]), "answer is the needle value"
    assert b.targets[:, b.meta["answer_pos"]].tolist() == b.meta["answers"].tolist()


@pytest.mark.unit
def test_niah_key_is_unique_so_retrieval_is_unambiguous():
    # The needle key must appear ONLY at the needle position and the query — never in the
    # filler — so a perfect model can score 100%. Guards the disjoint-band design.
    b = needle_in_haystack(batch=16, haystack_len=40, vocab_size=VOCAB, seed=11)
    depth = b.meta["needle_depth"]
    for i in range(b.batch_size):
        key = b.inputs[i, -1].item()
        assert (b.inputs[i] == key).sum().item() == 2, "key appears exactly twice (needle + query)"
        assert b.inputs[i, depth].item() == key


@pytest.mark.unit
def test_niah_depth_tracks_depth_frac():
    shallow = needle_in_haystack(batch=2, haystack_len=40, depth_frac=0.1, vocab_size=VOCAB, seed=0)
    deep = needle_in_haystack(batch=2, haystack_len=40, depth_frac=0.9, vocab_size=VOCAB, seed=0)
    assert shallow.meta["needle_depth"] < deep.meta["needle_depth"]


@pytest.mark.unit
def test_continual_tasks_use_disjoint_vocab_bands():
    tasks = continual_task_sequence(num_tasks=3, batch=4, length=6, vocab_size=96, seed=0)
    assert len(tasks) == 3
    bands = [t.meta["vocab_band"] for t in tasks]
    # bands are disjoint and the content tokens fall within them
    for t in tasks:
        lo, hi = t.meta["vocab_band"]
        content = t.inputs[t.inputs < 96 - 2]    # exclude SEP control token
        assert content.min().item() >= lo and content.max().item() < hi
    assert bands[0][1] <= bands[1][0] and bands[1][1] <= bands[2][0]


@pytest.mark.unit
def test_reward_task_rule_and_reward_fn():
    b, reward_fn = reward_task(batch=8, context_len=4, vocab_size=VOCAB, seed=5)
    gold = b.meta["answers"]
    assert torch.equal(gold, b.inputs[:, 0]), "rule is copy-first"
    assert torch.equal(reward_fn(gold), torch.ones(8))                 # correct -> 1
    assert torch.equal(reward_fn((gold + 1) % VOCAB), torch.zeros(8))  # wrong -> 0


@pytest.mark.unit
def test_make_task_unknown_raises():
    with pytest.raises(KeyError, match="unknown synthetic task"):
        make_task("nope")


# ----------------------------------------------------------------------- #
# Model-compatibility: a tiny model consumes every task -> finite loss
# ----------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize("name,kwargs", [
    ("copy", dict(batch=2, length=6, vocab_size=VOCAB)),
    ("associative_recall", dict(batch=2, num_pairs=3, vocab_size=VOCAB)),
    ("niah", dict(batch=2, haystack_len=20, vocab_size=VOCAB)),
    ("reward", dict(batch=2, context_len=4, vocab_size=VOCAB)),
])
def test_tiny_model_consumes_task(name, kwargs):
    m = make_tiny_synaptic(seed=0, vocab_size=VOCAB, train=True)
    r = make_task(name, seed=0, **kwargs)
    b = r[0] if isinstance(r, tuple) else r
    logits, loss = m(b.inputs, targets=b.targets)
    assert torch.isfinite(loss), f"{name}: loss must be finite"
    assert logits.shape[:2] == b.inputs.shape
