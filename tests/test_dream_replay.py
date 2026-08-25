"""Tests for Generative Dream Replay (bead `r00r.6`)."""

import torch

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.sleep_consolidation import SleepConsolidationController
from bio_inspired_nanochat.synaptic import SynapticLinear


def _make_model() -> GPTSynaptic:
    cfg = GPTSynapticConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        synapses=True,
        use_moe=False,
    )
    return GPTSynaptic(cfg)


def test_generate_dreams_shape_and_range():
    """Verify that generate_dreams outputs valid sequences within vocabulary bounds."""
    model = _make_model()
    controller = SleepConsolidationController()
    state_before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    dreams = controller.generate_dreams(model, num_dreams=3, seq_len=8)

    assert dreams.shape == (3, 8)
    assert (dreams >= 0).all()
    assert (dreams < 32).all()
    assert all(
        torch.equal(value, state_before[name])
        for name, value in model.state_dict().items()
    )


def test_dream_replay_sleep_consolidation():
    """Verify that dream replay consolidates fast weights without an external buffer."""
    model = _make_model()
    controller = SleepConsolidationController(consolidation_lr=0.1, latch_threshold=0.0)

    syn_lin = next(mod for mod in model.modules() if isinstance(mod, SynapticLinear))
    w_slow_orig = syn_lin.w_slow.data.clone()
    assert syn_lin.w_fast is not None
    syn_lin.w_fast.data.fill_(0.4)

    report = controller.run_sleep_phase(
        model=model,
        replay_buffer=None,
        sleep_steps=2,
        batch_size=2,
        use_dream_replay=True,
    )

    assert report["status"] == "consolidated"
    assert report["total_transferred_norm"] > 0.0
    assert syn_lin.w_fast.norm().item() > 0.0
    assert not torch.equal(syn_lin.w_slow.data, w_slow_orig)
