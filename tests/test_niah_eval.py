"""Needle-in-a-haystack retrieval evaluation — bead 74f.2.

NIAH was a hardcoded ``None`` in eval_matrix; this wires the existing
``needle_in_haystack`` generator into a real accuracy-by-length eval. These tests
validate the scorer end-to-end with a perfect-retrieval oracle (must score 1.0), a
wrong constant model (must score ~chance), determinism, and that the harness runs on
both real model types (synaptic returns ``(logits, None)``, vanilla returns logits).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from bio_inspired_nanochat.synthetic_tasks import (
    IGNORE_INDEX,
    SyntheticBatch,
    niah_accuracy_by_length,
    retrieval_accuracy,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bio_testkit import make_tiny_synaptic, make_tiny_vanilla  # noqa: E402

pytestmark = pytest.mark.unit

VOCAB = 64


class _ConstModel(nn.Module):
    """Predicts a fixed token id at every position (an intentionally bad retriever)."""

    def __init__(self, pred: int, vocab: int = VOCAB):
        super().__init__()
        self.pred, self.vocab = pred, vocab

    def forward(self, inputs):
        b, t = inputs.shape
        logits = torch.full((b, t, self.vocab), -10.0)
        logits[:, :, self.pred] = 10.0
        return logits, None


class _PerfectNeedleModel(nn.Module):
    """A genuine retriever: find the query-key in the filler, predict the next token.

    The needle layout is ``[filler(H), QUERY, key]``; the key is planted once in the
    filler (disjoint bands guarantee uniqueness), followed by its value.
    """

    def __init__(self, vocab: int = VOCAB):
        super().__init__()
        self.vocab = vocab

    def forward(self, inputs):
        b, t = inputs.shape
        logits = torch.full((b, t, self.vocab), -10.0)
        key = inputs[:, -1]
        for row in range(b):
            filler = inputs[row, : t - 2]  # exclude QUERY and the trailing query-key
            hit = (filler == key[row]).nonzero(as_tuple=False)
            if hit.numel() > 0:
                val = int(inputs[row, int(hit[0, 0]) + 1])
                logits[row, t - 1, val] = 10.0
        return logits, None


def test_retrieval_accuracy_reads_answer_pos():
    batch = SyntheticBatch(
        inputs=torch.tensor([[1, 2, 3], [4, 5, 6]]),
        targets=torch.full((2, 3), IGNORE_INDEX),
        meta={"answer_pos": 2, "answers": torch.tensor([7, 7])},
    )
    assert retrieval_accuracy(_ConstModel(7), batch) == 1.0
    assert retrieval_accuracy(_ConstModel(0), batch) == 0.0


def test_perfect_oracle_scores_one():
    res = niah_accuracy_by_length(
        _PerfectNeedleModel(), vocab_size=VOCAB, lengths=(16, 32, 64), batch=16, seed=1
    )
    assert set(res["by_length"]) == {16, 32, 64}
    assert res["overall"] == pytest.approx(1.0)
    assert all(v == pytest.approx(1.0) for v in res["by_length"].values())


def test_wrong_constant_model_scores_near_chance():
    # A constant prediction in the filler band essentially never equals the value band.
    res = niah_accuracy_by_length(
        _ConstModel(VOCAB - 3), vocab_size=VOCAB, lengths=(16, 64), batch=32, seed=2
    )
    assert res["overall"] < 0.1


def test_deterministic_given_seed():
    kw = dict(vocab_size=VOCAB, lengths=(16, 32), batch=8, seed=3)
    a = niah_accuracy_by_length(_PerfectNeedleModel(), **kw)
    b = niah_accuracy_by_length(_PerfectNeedleModel(), **kw)
    assert a == b


@pytest.mark.parametrize("make", [make_tiny_synaptic, make_tiny_vanilla])
def test_runs_on_real_models_structure_and_range(make):
    # Exercises both return conventions (synaptic tuple, vanilla bare logits), the
    # reset_sequence_state hook, and logits-over-all-positions. Untrained -> low acc,
    # but must be finite and in [0,1].
    model = make(seed=0, vocab_size=VOCAB, sequence_len=130)
    res = niah_accuracy_by_length(
        model, vocab_size=VOCAB, lengths=(16, 64, 128), batch=8, seed=0
    )
    assert set(res["by_length"]) == {16, 64, 128}
    assert all(0.0 <= v <= 1.0 for v in res["by_length"].values())
    assert 0.0 <= res["overall"] <= 1.0
