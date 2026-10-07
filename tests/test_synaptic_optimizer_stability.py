"""GPTSynaptic's training recipe must not blow up its multiplicative 1-D gains.

``setup_optimizers`` used to give every 1-D block parameter (LayerNorm gains and biases, linear
biases, ``PostsynapticHebb.fast``/``slow`` in ``y = v * (1 + fast + slow)``) the embedding LR,
0.2 * (d/768)^-0.5 = 0.49 at d=128. Every gain then moved ~0.5 per step: bio_all went NaN within
20 steps on real text (results/scalar_lr_sweep_2026-10-07.json) and the 2026-09-02 toy screening's
seed 1338 ended at train loss 26.9. These tests pin the separate scalar LR from the outside: the
optimizer group it lands in, and a short base_train-recipe run whose loss must never rise above
its starting value, with the legacy LR as the planted negative that does.

Run:  pytest tests/test_synaptic_optimizer_stability.py -v
"""

from __future__ import annotations

import pytest
import torch

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.synaptic import SynapticConfig

pytestmark = pytest.mark.unit

VOCAB, SEQ, BATCH, STEPS = 512, 128, 4, 10
LEGACY_SCALAR_LR = 0.2 * (128 / 768) ** -0.5


def _model() -> GPTSynaptic:
    torch.manual_seed(1338)
    cfg = GPTSynapticConfig(
        sequence_len=SEQ, vocab_size=VOCAB, n_layer=2, n_head=1, n_kv_head=1, n_embd=128,
        # The full stack, Hebbian included (opt-in since hwxb.9): its post.fast/slow gains are
        # the multiplicative 1-D parameters most exposed to the scalar LR.
        synapses=True, syn_cfg=SynapticConfig(enable_hebbian=True),
    )
    model = GPTSynaptic(cfg)
    model.init_weights()
    return model


def _bigram_batches() -> list[torch.Tensor]:
    """Structured text: each token's successor is one of four fixed choices."""
    g = torch.Generator().manual_seed(0)
    table = torch.randint(0, VOCAB, (VOCAB, 4), generator=g)
    batches = []
    for _ in range(STEPS):
        x = torch.empty(BATCH, SEQ + 1, dtype=torch.long)
        x[:, 0] = torch.randint(0, VOCAB, (BATCH,), generator=g)
        for t in range(SEQ):
            x[:, t + 1] = table[x[:, t], torch.randint(0, 4, (BATCH,), generator=g)]
        batches.append(x)
    return batches


def _losses(**optimizer_kwargs) -> list[float]:
    model = _model()
    model.train()
    optimizers = model.setup_optimizers(**optimizer_kwargs)
    losses = []
    for x in _bigram_batches():
        _, loss = model(x[:, :-1], targets=x[:, 1:])
        loss.backward()
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
        losses.append(loss.item())
    return losses


def test_one_dimensional_block_params_get_the_scalar_lr_not_the_embedding_lr():
    model = _model()
    adamw, _muon = model.setup_optimizers()
    gains = {id(p) for n, p in model.h.named_parameters() if p.ndim < 2}
    assert any(n.endswith("post.fast") for n, p in model.h.named_parameters() if p.ndim < 2)
    groups = [g for g in adamw.param_groups if gains & {id(p) for p in g["params"]}]
    assert len(groups) == 1 and {id(p) for p in groups[0]["params"]} == gains
    assert groups[0]["lr"] == pytest.approx(0.01)


def test_default_recipe_never_rises_above_its_initial_loss():
    losses = _losses()
    assert all(torch.isfinite(torch.tensor(losses)))
    assert max(losses[1:]) <= losses[0], losses
    assert losses[-1] < losses[0] - 1.0, losses


def test_planted_negative_legacy_scalar_lr_spikes_above_the_initial_loss():
    losses = _losses(scalar_lr=LEGACY_SCALAR_LR)
    assert max(losses[1:]) > losses[0] + 1.0, losses
