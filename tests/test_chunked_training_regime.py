"""The chunked training regime for online fast weights (bead hwxb.8, bridge plan G2).

Under ordinary training every batch is one full teacher-forced forward and the online Hebbian
writes are deferred to the top of the NEXT forward, so nothing written while reading a sequence can
influence that sequence: the slow weights never learn to use the scratchpad, and the working-memory
probes could not see it either (fixed for reading in synthetic_tasks; this is the training half).

``GPTSynaptic.chunked_train_step`` forwards the batch ``chunk_len`` tokens at a time through a KV
cache, back-propagating each chunk's token-weighted loss immediately and detaching the cache after,
so chunk ``c``'s writes land before chunk ``c+1``'s matmuls. These tests lock:

* Hebbian OFF, deterministic release: the chunked loss equals the full-forward loss for any chunk
  length, and a single chunk gives the full forward's gradients;
* Hebbian ON: the write routine runs once per synaptic linear per chunk boundary inside ONE
  sequence and moves ``w_fast``; a plain full forward applies no writes within the sequence;
* 20 optimizer steps under the default config stay finite;
* the step refuses eval mode and bad chunk lengths; ``base_train`` exposes ``--hebb_chunk_len``
  with its refusals; ``eval_matrix`` exposes ``--hebb-chunk-len``.

Green here makes the regime available and correct; it does not show fast weights help (hwxb.9).

Run:  pytest tests/test_chunked_training_regime.py -v
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bio_testkit import make_tiny_synaptic, random_tokens  # noqa: E402

from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticLinear  # noqa: E402

pytestmark = pytest.mark.unit

VOCAB = 97
REPO = Path(__file__).resolve().parents[1]


def _batch(seed: int, batch: int = 2, seq: int = 20):
    x = random_tokens(batch, seq, VOCAB, seed=seed)
    y = random_tokens(batch, seq, VOCAB, seed=seed + 1000)
    y[:, :3] = -1  # a few unsupervised positions, so the token weighting is exercised
    return x, y


def _deterministic_model(seed: int = 0, *, hebbian: bool):
    cfg = SynapticConfig(enable_hebbian=hebbian, stochastic_train_frac=0.0)
    return make_tiny_synaptic(seed=seed, train=True, sequence_len=32, syn_cfg=cfg)


def _grads(model) -> dict[str, torch.Tensor]:
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("chunk_len", [1, 5, 20])
def test_hebbian_off_chunked_loss_equals_full_forward_loss(chunk_len):
    x, y = _batch(0)
    full_model = _deterministic_model(hebbian=False)
    _, full = full_model(x, y, train_mode=True)
    chunk_model = _deterministic_model(hebbian=False)
    chunked = chunk_model.chunked_train_step(x, y, chunk_len=chunk_len)
    torch.testing.assert_close(chunked, full.detach().float(), rtol=1e-5, atol=1e-5)
    assert chunk_model.wte.weight.grad is not None, "the chunked step must back-propagate"


def test_single_chunk_reproduces_the_full_forward_gradients():
    x, y = _batch(1)
    full_model = _deterministic_model(hebbian=False)
    _, full = full_model(x, y, train_mode=True)
    full.backward()
    chunk_model = _deterministic_model(hebbian=False)
    chunk_model.chunked_train_step(x, y, chunk_len=x.shape[1])
    g_full, g_chunk = _grads(full_model), _grads(chunk_model)
    assert set(g_full) == set(g_chunk)
    for name in g_full:
        torch.testing.assert_close(g_chunk[name], g_full[name], rtol=1e-5, atol=1e-6, msg=name)


def test_hebbian_writes_land_between_chunks_within_one_sequence(monkeypatch):
    model = make_tiny_synaptic(seed=0, train=True, sequence_len=32)  # default: Hebbian on
    lins = [m for m in model.modules() if isinstance(m, SynapticLinear) and m.w_fast is not None]
    assert lins
    calls = {"n": 0}
    original = SynapticLinear._apply_hebb_weight_writes

    def counting(self, gate_scale):
        calls["n"] += 1
        return original(self, gate_scale)

    monkeypatch.setattr(SynapticLinear, "_apply_hebb_weight_writes", counting)
    x, y = _batch(2)
    model.reset_sequence_state(reset_fast_weights=True)
    norm_before = max(lin.w_fast.norm().item() for lin in lins)

    # The contrast: one full forward applies no write inside the sequence (all stay pending).
    model(x, y, train_mode=True)
    assert calls["n"] == 0
    assert all(lin._plasticity_pending for lin in lins)
    model.reset_sequence_state(reset_fast_weights=True)
    model.zero_grad(set_to_none=True)

    n_chunks = 4
    model.chunked_train_step(x, y, chunk_len=x.shape[1] // n_chunks)
    n_lin = sum(isinstance(m, SynapticLinear) for m in model.modules())
    assert calls["n"] == (n_chunks - 1) * n_lin, "one write pass per synaptic linear per chunk boundary"
    assert all(lin._plasticity_pending for lin in lins), "the last chunk's write waits for the next forward"
    norm_after = max(lin.w_fast.norm().item() for lin in lins)
    assert norm_before == 0.0 and norm_after > 0.0, "fast weights moved inside the sequence"


def test_twenty_optimizer_steps_stay_finite_under_the_default_config():
    model = make_tiny_synaptic(seed=3, train=True, sequence_len=32)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for step in range(20):
        x, y = _batch(100 + step, seq=32)
        model.reset_sequence_state(reset_fast_weights=True)
        loss = model.chunked_train_step(x, y, chunk_len=8)
        assert torch.isfinite(loss), f"step {step}: loss {loss}"
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss))
    assert all(torch.isfinite(torch.tensor(losses)))


def test_refuses_eval_mode_and_bad_chunk_lengths():
    model = make_tiny_synaptic(seed=0, train=True)
    x, y = _batch(4, seq=16)
    with pytest.raises(ValueError, match="chunk_len"):
        model.chunked_train_step(x, y, chunk_len=0)
    model.eval()
    with pytest.raises(RuntimeError, match="training step"):
        model.chunked_train_step(x, y, chunk_len=4)


def test_base_train_exposes_the_flag_and_its_refusals():
    src = (REPO / "scripts" / "base_train.py").read_text(encoding="utf-8")
    assert "hebb_chunk_len = 0" in src
    assert "hebb_chunk_len > 0 requires synapses=1" in src
    assert "hebb_chunk_len > 0 is single-process for now" in src
    assert "orig_model.chunked_train_step(" in src


def test_eval_matrix_exposes_the_flag():
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.eval_matrix", "run", "--help"],
        cwd=REPO, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert "--hebb-chunk-len" in proc.stdout
