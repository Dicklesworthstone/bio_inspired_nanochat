"""
Working-memory evaluation suite (bead sax.4).

A targeted suite for the "infinite local context" claim: associative recall by memory load,
variable binding by distractor count (the needle-with-distractors stress), and needle-in-a-haystack
by length — each producing a clean by-difficulty curve with per-batch fast-weight resets. It is
model-agnostic (bio + vanilla), so it drives bio-vs-vanilla and fast-weight-baseline comparisons.

These tests lock: the suite produces all three curves with valid metrics; it runs on both bio and
vanilla models; the standalone recall/binding sweeps expose the requested difficulty axes; and the
results are reproducible (the per-batch reset makes retrievals independent and deterministic).

Run:  pytest tests/test_working_memory_suite.py -v
"""

from __future__ import annotations

import pytest

from bio_inspired_nanochat.synthetic_tasks import (
    working_memory_suite,
    recall_accuracy_by_pairs,
    binding_accuracy_by_distractors,
)

from _bio_testkit import make_tiny_synaptic, make_tiny_vanilla

VOCAB = 97
RECALL_PAIRS = (2, 4, 8)
BINDING_DISTRACTORS = (0, 8, 32)
NIAH_LENGTHS = (16, 64)


def _model(make, seq_len=160):
    # context must fit the longest eval sequence (binding distractors / NIAH lengths)
    return make(seed=0, sequence_len=seq_len)


def _all_accuracies(res):
    return (
        list(res["recall"]["by_pairs"].values())
        + list(res["binding"]["by_distractors"].values())
        + list(res["niah"]["by_length"].values())
    )


# --------------------------------------------------------------------------- #
# 1. The suite produces all three by-difficulty curves with valid metrics
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_suite_produces_three_curves_and_summary():
    res = working_memory_suite(
        _model(make_tiny_synaptic), vocab_size=VOCAB, recall_pairs=RECALL_PAIRS,
        binding_distractors=BINDING_DISTRACTORS, niah_lengths=NIAH_LENGTHS, batch=16, seed=1,
    )
    assert set(res) == {"recall", "binding", "niah", "summary"}
    assert tuple(res["recall"]["by_pairs"]) == RECALL_PAIRS
    assert tuple(res["binding"]["by_distractors"]) == BINDING_DISTRACTORS
    assert tuple(res["niah"]["by_length"]) == NIAH_LENGTHS
    assert set(res["summary"]) == {"recall_overall", "binding_overall", "niah_overall"}
    assert all(0.0 <= v <= 1.0 for v in _all_accuracies(res)), "accuracies must be in [0,1]"


# --------------------------------------------------------------------------- #
# 2. Model-agnostic: runs on bio AND vanilla (the bio-vs-vanilla comparison)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize("make", [make_tiny_synaptic, make_tiny_vanilla])
def test_suite_runs_on_bio_and_vanilla(make):
    res = working_memory_suite(
        _model(make), vocab_size=VOCAB, recall_pairs=RECALL_PAIRS,
        binding_distractors=BINDING_DISTRACTORS, niah_lengths=NIAH_LENGTHS, batch=16, seed=2,
    )
    assert all(0.0 <= v <= 1.0 for v in _all_accuracies(res))
    for k in ("recall_overall", "binding_overall", "niah_overall"):
        assert 0.0 <= res["summary"][k] <= 1.0


# --------------------------------------------------------------------------- #
# 3. Standalone sweeps expose the requested difficulty axes
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_recall_and_binding_sweeps_have_requested_axes():
    m = _model(make_tiny_synaptic)
    r = recall_accuracy_by_pairs(m, vocab_size=VOCAB, num_pairs=(2, 4, 8, 16), batch=8, seed=0)
    assert tuple(r["by_pairs"]) == (2, 4, 8, 16) and 0.0 <= r["overall"] <= 1.0

    b = binding_accuracy_by_distractors(
        m, vocab_size=VOCAB, num_distractors=(0, 4, 16, 64), batch=8, seed=0
    )
    assert tuple(b["by_distractors"]) == (0, 4, 16, 64) and 0.0 <= b["overall"] <= 1.0


# --------------------------------------------------------------------------- #
# 4. Per-batch reset ⇒ reproducible, independent retrievals
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_suite_skips_oversized_points_instead_of_crashing():
    # The DEFAULT difficulty params generate sequences (recall up to 34, binding up to 72, NIAH up
    # to 130 tokens) that exceed a small model's context — the suite must SKIP those points, not
    # crash the model's `T <= sequence_len` assertion. Use the default tiny model (context 32).
    model = make_tiny_synaptic(seed=0)  # sequence_len = 32
    ctx = model.config.sequence_len
    res = working_memory_suite(model, vocab_size=VOCAB, batch=8, seed=1)  # all-default params
    # every measured point fits the context; at least one fits (so the curves aren't all empty)
    assert all(2 * n + 2 <= ctx for n in res["recall"]["by_pairs"]), "recall kept an oversized point"
    assert all(n + 8 <= ctx for n in res["binding"]["by_distractors"]), "binding kept an oversized point"
    assert all(L + 2 <= ctx for L in res["niah"]["by_length"]), "niah kept an oversized point"
    assert res["recall"]["by_pairs"] and res["niah"]["by_length"], "some points must still fit"
    assert all(0.0 <= v <= 1.0 for v in _all_accuracies(res))


@pytest.mark.unit
def test_suite_is_reproducible():
    kw = dict(
        vocab_size=VOCAB, recall_pairs=RECALL_PAIRS, binding_distractors=BINDING_DISTRACTORS,
        niah_lengths=NIAH_LENGTHS, batch=16, seed=7,
    )
    a = working_memory_suite(_model(make_tiny_synaptic), **kw)
    b = working_memory_suite(_model(make_tiny_synaptic), **kw)
    assert a["summary"] == b["summary"], "fixed seed + per-batch reset must be deterministic"
