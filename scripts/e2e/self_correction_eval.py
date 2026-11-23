"""Evaluate bounded self-correction on exact, two-token held-out binding answers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console

from bio_inspired_nanochat import causal_deliberation, self_correcting_generator, sheaf_obstruction
from bio_inspired_nanochat.causal_deliberation import CausalDeliberationConfig, CausalDeliberationController
from bio_inspired_nanochat.eval_stats import paired_comparison
from bio_inspired_nanochat.gpt import GPT, GPTConfig
from bio_inspired_nanochat.self_correcting_generator import (
    SelfCorrectingGenerator, SelfCorrectionConfig, SelfCorrectionEvalSample, evaluate_self_correction_benchmark,
)
from bio_inspired_nanochat.sheaf_obstruction import ObstructionAction, SheafDetectorConfig, SheafObstructionDetector
from bio_inspired_nanochat.synthetic_tasks import control_tokens
from scripts.e2e.sheaf_detector_eval import SheafEvalConfig, _batch, _ids, _weights


POLICY = SelfCorrectionConfig(enabled=True, max_repair_attempts=2, obstruction_threshold=0.40,
                              deliberation_budget=4, max_repair_span=2, abstain_on_exhaustion=True)
REGISTERED = replace(SheafEvalConfig(), samples_per_seed=32)


class ObservedDetector(SheafObstructionDetector):
    """Retain real inspection outcomes, including below-threshold terminal checks."""

    def __init__(self, threshold: float):
        super().__init__(SheafDetectorConfig(enabled=True, action=ObstructionAction.DELIBERATE, threshold=threshold))
        self.checks: list[dict] = []

    def inspect(self, stalks, edge_index, **kwargs):
        result = super().inspect(stalks, edge_index, **kwargs)
        self.checks.append({'score': result.score, 'threshold': result.threshold, 'available': result.available,
                            'flagged': result.flagged, 'reason': result.fallback_reason,
                            'edge_residual_norms': list(result.edge_residual_norms)})
        return result


def _fit(cfg: SheafEvalConfig, seed: int):
    torch.manual_seed(seed)
    model = GPT(GPTConfig(sequence_len=32, vocab_size=cfg.vocab_size, n_layer=2, n_head=2,
                          n_kv_head=2, n_embd=cfg.d_model, init_seed=seed))
    model.init_weights()
    initial = _weights(model)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    gen = np.random.default_rng(seed)
    sep = control_tokens(cfg.vocab_size).SEP
    train_ids, losses = set(), []
    for _ in range(cfg.train_steps):
        batch = _batch(cfg.train_batch, cfg.vocab_size, int(gen.integers(2**63 - 1)))
        train_ids.update(_ids(batch.inputs))
        value = batch.targets[:, -1:]
        targets = torch.cat((value, torch.full_like(value, sep)), dim=1)
        logits = model(torch.cat((batch.inputs, value), dim=1))[:, -2:]
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    fitted = _weights(model)
    if initial == fitted or not all(map(math.isfinite, losses)):
        raise ValueError('binding training requires finite losses and changed parameters')
    model.eval()
    return model, train_ids, {'initial_weight_sha256': initial, 'fitted_weight_sha256': fitted,
                              'training_prompt_ids_sha256': hashlib.sha256(''.join(sorted(train_ids)).encode()).hexdigest(),
                              'distinct_training_prompts': len(train_ids), 'training_losses': losses}


def _evaluate_seed(config: SheafEvalConfig, seed: int):
    model, train_ids, training = _fit(config, seed)
    batch = _batch(config.samples_per_seed, config.vocab_size, 990000 + seed)
    source_ids = _ids(batch.inputs)
    if len(set(source_ids)) != len(source_ids) or set(source_ids) & train_ids:
        raise ValueError('held-out prompts must be distinct and excluded from training')
    # The production relaxation gate is seed-initialized independently of the trained backbone.
    torch.manual_seed(310000 + seed)
    controller = CausalDeliberationController(model, CausalDeliberationConfig(max_iters=POLICY.deliberation_budget,
                                                                               commit_relaxed_state=True))
    detector = ObservedDetector(POLICY.obstruction_threshold)
    generator = SelfCorrectingGenerator(model, POLICY, detector, controller)
    contexts: list[list[int]] = []
    operations = Counter()

    def record_backbone(_module, inputs):
        contexts.append(inputs[0][0].tolist())

    def record_head(_module, _inputs):
        operations['head'] += 1

    def record_relaxation(_module, _inputs):
        operations['relaxer_gate'] += 1

    handles = [model.wte.register_forward_pre_hook(record_backbone), model.lm_head.register_forward_pre_hook(record_head),
               controller.relaxer.recurrent_gate.register_forward_pre_hook(record_relaxation)]
    rows = []
    try:
        for i, source_id in enumerate(source_ids):
            prompt = batch.inputs[i].tolist()
            gold = prompt + [int(batch.targets[i, -1]), control_tokens(config.vocab_size).SEP]
            costs: list[dict] = []
            draft_checks = []
            detector.checks.clear()

            def oracle(tokens):
                # The canonical harness scores each arm immediately after generation.
                costs.append({'backbone_calls': len(contexts), 'backbone_input_tokens': sum(map(len, contexts)),
                              'context_lengths': list(map(len, contexts)), 'head_calls': operations['head'],
                              'relaxer_gate_calls': operations['relaxer_gate']})
                full_context = next((seq for seq in contexts if len(seq) == len(gold)), None)
                draft_checks.append(full_context)
                contexts.clear()
                operations.clear()
                return list(tokens) != gold

            sample = SelfCorrectionEvalSample(batch.inputs[i], 2, oracle, False, name=source_id)
            generation_seed = 700000 + seed * 1024 + i
            report = evaluate_self_correction_benchmark(generator, [sample], temperature=0.8, top_k=None, seed=generation_seed)
            pair = report.sample_results[0]
            trajectory = report.trajectories[0]
            if len(costs) != 2 or draft_checks[1] != pair.baseline_tokens:
                raise ValueError('both arms must generate the identical initial draft')
            candidate = list(pair.baseline_tokens)
            attempts = []
            for event in trajectory.events:
                if candidate[event.span_start:event.span_end] != event.corrupted_tokens:
                    raise ValueError('repair lineage disagrees with the generated draft')
                candidate[event.span_start:event.span_end] = event.repaired_tokens
                attempts.append({**event.to_dict(), 'candidate_completion': candidate[len(prompt):],
                                 'candidate_exact_match': candidate == gold})
            if not trajectory.is_abstention and candidate != trajectory.final_tokens:
                raise ValueError('repair lineage disagrees with returned answer')
            rows.append({'source_id': source_id, 'prompt': prompt, 'gold': gold[len(prompt):], 'rng_seed': generation_seed,
                         'baseline': pair.baseline_tokens[len(prompt):], 'corrected': pair.self_correcting_tokens[len(prompt):],
                         'baseline_failure': pair.baseline_error, 'corrected_failure': pair.self_correcting_failure,
                         'abstained': pair.abstained, 'outcome': trajectory.outcome.value, 'attempts': attempts,
                         'checks': detector.checks.copy(), 'baseline_cost': costs[0], 'corrected_cost': costs[1],
                         'baseline_ms': pair.baseline_latency_ms, 'corrected_ms': pair.self_correcting_latency_ms})
    finally:
        for handle in handles:
            handle.remove()
    if _weights(model) != training['fitted_weight_sha256']:
        raise ValueError('paired evaluation changed fitted model weights')
    return {'seed': seed, 'training': training, 'samples': rows}, train_ids


def run_binding_correction(config: SheafEvalConfig = REGISTERED) -> dict:
    if len(set(config.seeds)) != len(config.seeds) or len(config.seeds) < 2 or config.samples_per_seed < 1:
        raise ValueError('need at least two independent seeds and a positive number of held-out samples')
    runs, training_ids, held_ids = [], set(), set()
    with torch.random.fork_rng(devices=[]):
        for seed in config.seeds:
            run, used = _evaluate_seed(config, seed)
            ids = {row['source_id'] for row in run['samples']}
            if ids & held_ids:
                raise ValueError('a held-out prompt was reused across seeds')
            held_ids.update(ids)
            training_ids.update(used)
            runs.append(run)
    if training_ids & held_ids:
        raise ValueError('a held-out prompt occurred in another seed training stream')
    rows = [row for run in runs for row in run['samples']]
    errors = {arm: {run['seed']: sum(row[f'{arm}_failure'] for row in run['samples']) / len(run['samples'])
                     for run in runs} for arm in ('baseline', 'corrected')}
    stats = paired_comparison(errors['corrected'], errors['baseline'], lower_is_better=True)
    if stats is None:
        raise ValueError('paired uncertainty requires matched training seeds')
    cost = {arm: {name: sum(row[f'{arm}_cost'][name] for row in rows)
                  for name in ('backbone_calls', 'backbone_input_tokens', 'head_calls', 'relaxer_gate_calls')}
            for arm in ('baseline', 'corrected')}
    times = {arm: sum(row[f'{arm}_ms'] for row in rows) for arm in cost}
    calls_ratio = cost['corrected']['backbone_calls'] / cost['baseline']['backbone_calls']
    wall_ratio = times['corrected'] / times['baseline']
    answered = [r for r in rows if not r['abstained']]
    win = (stats.delta_ci_high < 0 and stats.t_p_value < .05 and stats.wilcoxon_p_value < .05
           and calls_ratio <= 3 and wall_ratio <= 3)
    curve = {}
    for arm in cost:
        failures = np.cumsum([r[f'{arm}_failure'] for r in rows])
        curve[arm] = (failures / np.arange(1, len(rows) + 1)).tolist()
    return {'scope': 'Synthetic variable binding. Both value and trailing separator must match. Clean prompts, native model samples, no planted errors. Every abstention counts as a failure. Instrumented CPU timings exclude model-state restoration and oracle scoring.',
            'protocol': {'training': asdict(config), 'generation': asdict(POLICY), 'temperature': .8, 'top_k': None,
                         'optimizer': 'AdamW lr=.003 betas=(.9,.999) eps=1e-8 weight_decay=.01', 'completion_tokens': 2,
                         'heldout_seed': '990000+training_seed', 'relaxer_seed': '310000+training_seed',
                         'win_rule': 'paired seed bootstrap 95% upper bound <0, paired t and Wilcoxon p<.05, backbone and wall ratios <=3'},
            'pairs': len(rows), 'seeds': stats.n_pairs, 'failures': {arm: sum(r[f'{arm}_failure'] for r in rows) for arm in cost},
            'coverage': len(answered) / len(rows), 'answered_only_failures': sum(r['corrected_failure'] for r in answered),
            'answered_only_risk': sum(r['corrected_failure'] for r in answered) / len(answered) if answered else None,
            'outcomes': dict(Counter(r['outcome'] for r in rows)), 'per_seed_error_rates': errors,
            'mean_error_difference': stats.mean_delta, 'seed_bootstrap_95': [stats.delta_ci_low, stats.delta_ci_high],
            'paired_t_p': stats.t_p_value, 'wilcoxon_p': stats.wilcoxon_p_value, 'bounded_superiority': win,
            'costs': cost, 'measured_total_ms': times, 'backbone_call_ratio': calls_ratio, 'measured_wall_ratio': wall_ratio,
            'cumulative_error_rates': curve, 'runs': runs, 'torch_version': str(torch.__version__),
            'source_sha256': {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                              for p in (__file__, self_correcting_generator.__file__, causal_deliberation.__file__, sheaf_obstruction.__file__)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = run_binding_correction()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, allow_nan=False, separators=(',', ':')) + '\n', encoding='utf-8')
    Console().print({k: report[k] for k in ('pairs', 'failures', 'coverage', 'bounded_superiority', 'measured_wall_ratio')})


if __name__ == '__main__':
    main()
