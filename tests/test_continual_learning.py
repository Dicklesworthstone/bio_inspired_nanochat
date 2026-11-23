import io

import pytest
import torch
from rich.console import Console

from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic
from bio_inspired_nanochat.sleep_consolidation import get_synaptic_layers
from scripts.e2e.continual_learning_eval import (
    GAUGE, RULE, SLEEP, ContinualEvalConfig, _compare, _inputs, _labels, _model, _score, _weights, print_continual_summary, run_continual_benchmark,
)


def test_scores_come_from_raw_predictions_with_identical_outputs_across_task_labels():
    cfg = ContinualEvalConfig()
    model = _model(cfg, 7, False).eval()
    probe, streams = _inputs(cfg, 7)
    with torch.no_grad():
        model.lm_head.weight.normal_(std=0.2)
        logits = model(probe)
    result = _score(model, probe, cfg.vocab_size)
    for task in RULE:
        gold = _labels(probe, task, cfg.vocab_size)
        actual = result['measurements'][task]
        assert actual['correct'] == int((logits.argmax(-1) == gold).sum())
        assert actual['nll'] == pytest.approx(float(torch.nn.functional.cross_entropy(logits.flatten(0, 1), gold.flatten())), abs=1e-6)
    held = {tuple(v) for v in probe.tolist()}
    assert len(held) == cfg.probe_rows
    assert all(tuple(v) not in held for batch in streams.values() for v in batch.flatten(0, 1).tolist())


def test_real_continual_benchmark_checks_task_acquisition_and_exposes_identical_candidate():
    cfg = ContinualEvalConfig(seeds=(1, 2), steps_per_task=2, return_steps=1, sleep_steps=1, probe_rows=8)
    report = run_continual_benchmark(cfg)
    for row in report['rows']:
        if row['mode'].startswith('bio'):
            assert row['wake_layers'] > 0 and row['wake_layer_writes'] == 3 * cfg.steps_per_task * row['wake_layers']
            assert row['offline_optimizer_steps'] == 0
        p = row['phase_scores']
        end = 'C' if row['mode'] in SLEEP else 'return'
        gold = _labels(torch.tensor(report['data'][row['seed']]['probe_inputs']), 'A', cfg.vocab_size)
        for phase in ('A', 'B', end):
            observed = int((torch.tensor(p[phase]['predictions']) == gold).sum())
            assert p[phase]['measurements']['A']['correct'] == observed
        delta = (p['A']['measurements']['A']['correct'] - p[end]['measurements']['A']['correct']) / gold.numel()
        assert row['a_accuracy_forgetting'] == delta
        assert row['forward_tokens']['optimizer'] == cfg.batch_size * cfg.sequence_len * (
            cfg.steps_per_task * 3 if row['mode'] in SLEEP else cfg.steps_per_task * 2 + cfg.return_steps)
    compare = report['comparisons']['gauge_penalty_minus_ewc']
    assert report['gauge_equals_independent_ewc_final_parameters']
    assert compare['a_nll_forgetting']['paired_bootstrap_95'] == [0.0, 0.0]
    assert report['holonomy'] is None and report['holonomy_forgetting_correlation'] is None
    assert len(report['rows']) == len(cfg.seeds) * (len(SLEEP) + len(GAUGE))
    if not all(report['learned_all_required_phases'][m] for m in SLEEP):
        assert not report['sleep_superiority']
    rendered = io.StringIO()
    print_continual_summary(report, Console(file=rendered))
    assert 'gauge' in rendered.getvalue()


def test_heldout_scoring_never_commits_a_pending_wake_write():
    cfg = ContinualEvalConfig()
    bio = _model(cfg, 17, True)
    assert isinstance(bio, GPTSynaptic)
    probe, streams = _inputs(cfg, 17)
    train = streams['A'][0]
    logits, _ = bio(train, train_mode=True)
    torch.nn.functional.cross_entropy(logits.flatten(0, 1), _labels(train, 'A', cfg.vocab_size).flatten()).backward()
    layers = get_synaptic_layers(bio)
    assert layers and all(layer._plasticity_pending for layer in layers)
    before = _weights(bio)
    _score(bio, probe, cfg.vocab_size)
    assert all(layer._plasticity_pending for layer in layers)
    assert _weights(bio) == before


def test_invalid_predictions_and_missing_pairs_cannot_become_successful_scores():
    cfg = ContinualEvalConfig()
    model = _model(cfg, 29, False)
    with torch.no_grad():
        model.lm_head.weight.fill_(float('nan'))
    probe, _ = _inputs(cfg, 29)
    with pytest.raises(FloatingPointError, match='non-finite'):
        _score(model, probe, cfg.vocab_size)
    result = _compare([{'mode': 'sleep', 'seed': 1, 'forgetting': 0.0}], 'sleep', 'control', 'forgetting')
    assert result['seeds'] == 0 and result['mean_difference'] is None and result['paired_t_p'] is None
