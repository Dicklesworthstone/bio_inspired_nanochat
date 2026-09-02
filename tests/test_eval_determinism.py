"""An eval-mode GPTSynaptic is deterministic by default (bridge plan G3).

Until 2026-09-01 ``GPTSynaptic.forward`` defaulted to ``train_mode=True``, which switched on
stochastic vesicle sampling (``stochastic_train_frac=0.12``) and online plasticity even for a
model in eval mode. Every evaluator that did not pass ``train_mode=False`` explicitly (CORE eval,
``base_eval``, the e2e comparison scripts, the CMA-ES proxy) therefore scored a noisy,
self-modifying model: identical forwards differed by up to 0.067 in logits.

These tests pin the fixed contract from the outside, through the entry points evaluators use:

* two eval-mode forwards on the same batch are bit-identical, with no flag passed;
* ``core_eval.forward_model`` (the CORE metric path) is idempotent;
* the working-memory suite returns identical numbers on repeat;
* a *training*-mode forward with stochastic sampling on still differs between calls, so the
  default is a genuine mode switch and not a global disabling of stochastic release;
* an eval forward leaves the fast weights untouched, while an explicit ``update_mem=True``
  read is allowed to move them.

Run:  pytest tests/test_eval_determinism.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bio_testkit import make_tiny_synaptic, random_tokens  # noqa: E402

from bio_inspired_nanochat.core_eval import forward_model  # noqa: E402
from bio_inspired_nanochat.synaptic import SynapticLinear  # noqa: E402
from bio_inspired_nanochat.synthetic_tasks import working_memory_suite  # noqa: E402

pytestmark = pytest.mark.unit

VOCAB = 97


def _tokens(seed: int = 0) -> torch.Tensor:
    return random_tokens(2, 24, VOCAB, seed=seed)


@torch.no_grad()
def _logits(model, x: torch.Tensor, **kw) -> torch.Tensor:
    out = model(x, **kw)
    return (out[0] if isinstance(out, tuple) else out).clone()


def test_eval_mode_forward_is_bit_identical_without_any_flag():
    model = make_tiny_synaptic(seed=0)
    assert model.config.syn_cfg.stochastic_train_frac > 0, "the default must keep stochastic release on for TRAINING"
    x = _tokens()
    a, b = _logits(model, x), _logits(model, x)
    assert torch.equal(a, b)


def test_core_eval_forward_path_is_idempotent():
    model = make_tiny_synaptic(seed=1)
    x = _tokens(1)
    losses_a, preds_a = forward_model(model, x)
    losses_b, preds_b = forward_model(model, x)
    # core_eval leaves the unsupervised final position as NaN by construction; NaN != NaN under
    # torch.equal, so compare with equal_nan (zero tolerance otherwise).
    torch.testing.assert_close(losses_a, losses_b, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(preds_a, preds_b)


def test_working_memory_suite_repeats_exactly():
    model = make_tiny_synaptic(seed=2)
    kw: dict[str, Any] = dict(vocab_size=VOCAB, recall_pairs=(2,), binding_distractors=(0,), niah_lengths=(8,), batch=8, seed=3)
    first = working_memory_suite(model, **kw)
    second = working_memory_suite(model, **kw)
    assert first["recall"] == second["recall"]
    assert first["binding"] == second["binding"]
    assert first["niah"] == second["niah"]


def test_training_mode_forward_still_samples_stochastically():
    model = make_tiny_synaptic(seed=0, train=True)
    x = _tokens()
    a, b = _logits(model, x), _logits(model, x)
    assert not torch.equal(a, b), "training-mode stochastic vesicle release must still be live"
    # And the old behaviour is still one keyword away for callers who want it in eval mode.
    model.eval()
    c, d = _logits(model, x, train_mode=True), _logits(model, x, train_mode=True)
    assert not torch.equal(c, d)


def test_eval_forward_leaves_fast_weights_alone_but_explicit_update_mem_may_move_them():
    model = make_tiny_synaptic(seed=4)
    layer = next(m for m in model.modules() if isinstance(m, SynapticLinear) and m.w_fast is not None)
    x = _tokens(4)
    before = layer.w_fast.detach().clone()
    _logits(model, x)
    assert torch.equal(layer.w_fast.detach(), before), "a plain eval forward must not adapt"
    _logits(model, x, train_mode=False, update_mem=True)
    assert not torch.equal(layer.w_fast.detach(), before), "an explicit update_mem read applies the writes"
