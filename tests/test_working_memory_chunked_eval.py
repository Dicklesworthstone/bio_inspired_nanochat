"""Chunked (KV-cache) evaluation lets the working-memory probes SEE online fast-weight writes.

Beads sax.1 / hwxb.4.4 / sx1m. Every earlier measurement of the README's "infinite local
context" claim ran the probe sequence through ONE full teacher-forced forward. In that regime the
online Hebbian writes are deferred (training) or applied after the matmuls (inference), so nothing
written while the key/value pairs are read can influence the query position of the same forward:
the probes were structurally blind to the mechanism they were scoring, and the recorded null
(bio == vanilla, p = 1.0) says nothing about the mechanism.

``retrieval_accuracy(..., chunk_len=k)`` feeds the sequence through a KV cache k tokens at a
time, so writes made on earlier chunks are in force for later ones — exactly how token-by-token
generation behaves. These tests lock the contract:

* vanilla GPT: chunked logits == full-forward logits (the cache path is exact);
* GPTSynaptic with Hebbian OFF: chunked == full (presyn state carries exactly through the cache);
* GPTSynaptic with Hebbian ON: the fast-weight write routine runs once per chunk per synaptic
  linear (instead of once per sequence), and the chunked logits differ from the single-forward
  logits — the writes are now visible to the probe;
* the suite accepts ``chunk_len``, and a chunk longer than the sequence degrades to the full
  forward.

Green here does NOT show that fast weights help retrieval; it shows the probe can now measure it.

Run:  pytest tests/test_working_memory_chunked_eval.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bio_testkit import make_tiny_synaptic, make_tiny_vanilla  # noqa: E402

from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticLinear  # noqa: E402
from bio_inspired_nanochat.synthetic_tasks import (  # noqa: E402
    _logits_for,
    associative_recall,
    retrieval_accuracy,
    working_memory_suite,
)

pytestmark = pytest.mark.unit

VOCAB = 97  # matches the tiny testkit models


def _batch(seed: int = 0, num_pairs: int = 4, batch: int = 4):
    return associative_recall(batch=batch, num_pairs=num_pairs, vocab_size=VOCAB, seed=seed)


def _reset(model) -> None:
    if hasattr(model, "reset_sequence_state"):
        model.reset_sequence_state(reset_fast_weights=True)


@torch.no_grad()
def _full_and_chunked(model, inputs: torch.Tensor, chunk_len: int):
    _reset(model)
    full = _logits_for(model, inputs, chunk_len=None).clone()
    _reset(model)
    chunked = _logits_for(model, inputs, chunk_len=chunk_len).clone()
    return full, chunked


@pytest.mark.parametrize("chunk_len", [1, 3])
def test_vanilla_chunked_matches_full_forward(chunk_len):
    model = make_tiny_vanilla(seed=0)
    b = _batch()
    full, chunked = _full_and_chunked(model, b.inputs, chunk_len)
    assert chunked.shape == full.shape == (b.inputs.shape[0], b.inputs.shape[1], VOCAB)
    torch.testing.assert_close(chunked, full, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("chunk_len", [1, 3])
def test_synaptic_without_hebbian_chunked_matches_full_forward(chunk_len):
    model = make_tiny_synaptic(seed=0, syn_cfg=SynapticConfig(enable_hebbian=False))
    b = _batch()
    full, chunked = _full_and_chunked(model, b.inputs, chunk_len)
    # The presynaptic state (calcium, RRP, buffers) is carried through kv_cache.presyn_state, so
    # the incremental path must reproduce the full causal forward.
    torch.testing.assert_close(chunked, full, rtol=1e-4, atol=1e-4)


def test_hebbian_writes_land_once_per_chunk_and_become_visible(monkeypatch):
    model = make_tiny_synaptic(seed=0)
    assert model.config.syn_cfg.enable_hebbian, "the default config must keep Hebbian on"

    calls = {"n": 0}
    original = SynapticLinear._apply_hebb_weight_writes

    def counting(self, gate_scale):
        calls["n"] += 1
        return original(self, gate_scale)

    monkeypatch.setattr(SynapticLinear, "_apply_hebb_weight_writes", counting)

    b = _batch()
    seq_len = int(b.inputs.shape[1])
    with torch.no_grad():
        _reset(model)
        full = _logits_for(model, b.inputs, chunk_len=None).clone()
        full_calls = calls["n"]
        calls["n"] = 0
        _reset(model)
        chunked = _logits_for(model, b.inputs, chunk_len=1).clone()
        chunk_calls = calls["n"]

    n_syn_linear = sum(isinstance(m, SynapticLinear) for m in model.modules())
    # One full forward applies the writes exactly once per synaptic linear, AFTER its matmuls:
    # nothing written can influence that forward's own logits.
    assert full_calls == n_syn_linear > 0
    # Token-by-token reading applies them after every token, so later tokens see them.
    assert chunk_calls == seq_len * n_syn_linear
    diff = float((chunked - full).abs().max())
    assert diff > 0.0, "chunked reading must expose the fast-weight writes to later positions"


def test_suite_accepts_chunk_len_and_oversized_chunk_degrades_to_full_forward():
    model = make_tiny_synaptic(seed=0)
    out = working_memory_suite(
        model,
        vocab_size=VOCAB,
        recall_pairs=(2,),
        binding_distractors=(0,),
        niah_lengths=(8,),
        batch=4,
        seed=0,
        chunk_len=1,
    )
    assert set(out) >= {"recall", "binding", "niah", "summary"}
    assert 0.0 <= out["recall"]["overall"] <= 1.0

    b = _batch()
    _reset(model)
    full_acc = retrieval_accuracy(model, b)
    _reset(model)
    degenerate_acc = retrieval_accuracy(model, b, chunk_len=10_000)
    assert degenerate_acc == full_acc
