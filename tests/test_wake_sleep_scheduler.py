"""Tests for Wake/Sleep Scheduler in training and inference loops (bead `cel.5`)."""

import torch

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.sleep_consolidation import (
    WakeSleepConfig,
    WakeSleepScheduler,
)


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


def test_scheduler_training_interleaving():
    """Verify that scheduler triggers sleep consolidation at scheduled training intervals."""
    model = _make_model()
    cfg = WakeSleepConfig(enabled=True, sleep_every_n_steps=5, sleep_duration_steps=2, surprise_threshold=0.5)
    scheduler = WakeSleepScheduler(model=model, cfg=cfg)

    reports = []
    tokens = torch.randint(0, 32, (2, 8))

    for step in range(1, 11):
        rep = scheduler.step_training(step_idx=step, tokens=tokens, step_loss=1.2)
        if rep is not None:
            reports.append(rep)

    assert len(reports) == 2
    assert scheduler.total_sleep_phases == 2
    assert reports[0]["status"] == "consolidated"


def test_scheduler_inference_session_consolidation():
    """Verify that scheduler executes inter-session consolidation at session conclusion."""
    model = _make_model()
    cfg = WakeSleepConfig(enabled=True, consolidate_on_session_end=True, sleep_duration_steps=2)
    scheduler = WakeSleepScheduler(model=model, cfg=cfg)

    session_tok = torch.randint(0, 32, (8,))
    report = scheduler.on_session_end(session_tok)

    assert report is not None
    assert report["status"] == "consolidated"
    assert scheduler.total_sleep_phases == 1


def test_scheduler_disabled_flag():
    """Verify that disabled scheduler yields no actions during training or session end."""
    model = _make_model()
    cfg = WakeSleepConfig(enabled=False)
    scheduler = WakeSleepScheduler(model=model, cfg=cfg)

    rep = scheduler.step_training(step_idx=10, tokens=torch.randint(0, 32, (2, 8)), step_loss=2.0)
    assert rep is None
    assert scheduler.on_session_end() is None
