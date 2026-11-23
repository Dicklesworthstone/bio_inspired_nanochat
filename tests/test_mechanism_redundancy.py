"""Counterexamples and real training/score checks for the conditional mechanism interaction."""
import json
import math

import pytest
import torch
import torch.nn.functional as F

from bio_inspired_nanochat.synaptic import SynapticMoE
from scripts.e2e.mechanism_redundancy_eval import (
    ARMS, BOTH, GLIA, MET, OFF, RedundancyConfig, _batch, _build, _reset_witness, arm_configs, interaction, main,
)


def test_interaction_uses_matched_differences_and_rejects_missing_conditions():
    offsets = {OFF: 0, MET: -0.2, GLIA: -0.3, BOTH: -0.5}
    cells = {a: {s: s / 3 + o for s in range(6)} for a, o in offsets.items()}
    result = interaction(cells, margin=0.01, signals_active=True, learned=True)
    assert result['verdict'] == 'within_practical_margin_on_this_task'
    assert result['mean'] == pytest.approx(0, abs=1e-12)
    for s in cells[BOTH]:
        cells[BOTH][s] += 0.15
    result = interaction(cells, margin=0.01, signals_active=True, learned=True)
    assert result['verdict'] == 'positive_interaction_on_this_task'
    assert result['paired_bootstrap_95'] == pytest.approx([0.15, 0.15])
    assert interaction(cells, margin=.01, signals_active=False, learned=True)['verdict'].startswith('inconclusive')
    assert interaction(cells, margin=.01, signals_active=True, learned=False)['verdict'].startswith('inconclusive')
    cells[OFF].pop(2)
    with pytest.raises(ValueError, match='paired seeds'):
        interaction(cells, margin=.01, signals_active=True, learned=True)


def test_historical_single_forward_reset_discards_pending_hebbian_writes():
    witness = _reset_witness(34)
    old = witness['historical_reset']
    control = witness['successive_forwards']
    assert old['pending_before_forward_or_reset'] == control['pending_before_forward_or_reset'] == [False, True, True]
    assert old['bdnf_trace_l1'] == old['slow_consolidation_l1'] == 0
    assert control['bdnf_trace_l1'] > 0 and control['slow_consolidation_l1'] > 0


def test_dense_inert_glial_toggle_preserves_nonuniform_model_predictions():
    config = RedundancyConfig()
    arms = arm_configs()
    x, y = _batch(torch.tensor([0, 4, 8, 12]), 8)
    logits = []
    for arm in (OFF, GLIA):
        model = _build(config, arms[arm], 47, moe=False).eval()
        assert not any(isinstance(m, SynapticMoE) for m in model.modules())
        with torch.no_grad():
            model.lm_head.weight.normal_(std=0.1)
            logits.append(model(x, train_mode=False)[0])
    assert float(logits[0].std()) > 0.05
    assert torch.equal(logits[0], logits[1])
    assert F.cross_entropy(logits[0].flatten(0, 1), y.flatten()) == F.cross_entropy(logits[1].flatten(0, 1), y.flatten())


def test_real_training_cli_reports_observed_nll_without_arm_offsets(tmp_path):
    output = tmp_path / 'interaction.json'
    assert main(['--seeds', '41', '43', '--steps', '2', '--eval-batches', '1', '--output-json', str(output)]) == 0
    result = json.loads(output.read_text())
    assert len(result['rows']) == 8
    assert result['interaction']['verdict'].startswith('inconclusive')
    for seed in (41, 43):
        quartet = [r for r in result['rows'] if r['seed'] == seed]
        assert {r['arm'] for r in quartet} == set(ARMS)
        assert len({r['initial_parameter_sha256'] for r in quartet}) == 1
        for row in quartet:
            logits = torch.tensor(row['heldout_logits'])
            gold = torch.tensor(row['heldout_targets'])
            assert row['nll'] == pytest.approx(float(F.cross_entropy(logits.flatten(0, 1), gold.flatten())), abs=1e-7)
            assert row['accuracy'] == float((logits.argmax(-1) == gold).float().mean())
            assert row['trained_parameter_sha256'] != row['initial_parameter_sha256']
            assert not set(row['heldout_unique_starts']).intersection(sum(row['training_starts'], []))
            assert row['glial_observations'] == (2 if row['arm'] in (GLIA, BOTH) else 0)


def test_redundancy_config_validation():
    with pytest.raises(ValueError, match='unique'):
        RedundancyConfig(seeds=(1,))
    with pytest.raises(ValueError, match='unique'):
        RedundancyConfig(seeds=(2, 2))
    with pytest.raises(ValueError, match='positive'):
        RedundancyConfig(train_steps=0)
    with pytest.raises(ValueError, match='divisible'):
        RedundancyConfig(n_embd=6)
    with pytest.raises(ValueError, match='finite'):
        RedundancyConfig(lr=math.nan)
