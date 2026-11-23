"""Continual-learning controls on learned, interfering token rules and held-out sequences."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console

from bio_inspired_nanochat.eval_stats import paired_comparison
from bio_inspired_nanochat.gauge_consolidation import FisherNaturalConsolidator
from bio_inspired_nanochat.gpt import GPT, GPTConfig
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.sleep_consolidation import PrioritizedReplayBuffer, SleepConsolidationController, get_synaptic_layers
from bio_inspired_nanochat.synaptic import SynapticConfig

SLEEP = ('vanilla', 'bio_no_sleep', 'bio_with_sleep')
GAUGE = ('gauge_penalty', 'ewc', 'supervised_replay')
RULE = {'A': 1, 'B': -1, 'C': 3}


@dataclass(frozen=True)
class ContinualEvalConfig:
    seeds: tuple[int, ...] = (10, 20, 30, 40, 50, 60)
    steps_per_task: int = 32
    return_steps: int = 4
    batch_size: int = 8
    sequence_len: int = 4
    vocab_size: int = 8
    n_embd: int = 16
    sleep_steps: int = 2
    probe_rows: int = 32
    lr: float = 0.005

    def __post_init__(self):
        if len(set(self.seeds)) != len(self.seeds) or len(self.seeds) < 2:
            raise ValueError('need at least two distinct training seeds')
        if min(self.steps_per_task, self.return_steps, self.sleep_steps, self.probe_rows, self.sequence_len) < 1:
            raise ValueError('phase and sample sizes must be positive')
        if self.batch_size < 4 or self.batch_size % 2 or self.n_embd % 4 or self.vocab_size < 5:
            raise ValueError('need an even batch >=4, width divisible by four, and distinct A/B/C rules')
        if self.vocab_size ** self.sequence_len < 2 * self.probe_rows or not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError('need distinct held-out sequences and a finite positive learning rate')


def _inputs(cfg: ContinualEvalConfig, seed: int):
    rng = torch.Generator().manual_seed(10000 + seed)
    held = set()
    probe = []
    while len(probe) < cfg.probe_rows:
        row = tuple(torch.randint(cfg.vocab_size, (cfg.sequence_len,), generator=rng).tolist())
        if row not in held:
            held.add(row)
            probe.append(row)
    streams = {}
    for phase, count in (('A', cfg.steps_per_task), ('B', cfg.steps_per_task), ('C', cfg.steps_per_task), ('return', cfg.return_steps)):
        rows = []
        while len(rows) < cfg.batch_size * count:
            draw = torch.randint(cfg.vocab_size, (cfg.batch_size, cfg.sequence_len), generator=rng).tolist()
            rows.extend(v for v in draw if tuple(v) not in held)
        streams[phase] = torch.tensor(rows[:cfg.batch_size * count]).reshape(count, cfg.batch_size, cfg.sequence_len)
    return torch.tensor(probe), streams


def _weights(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode() + b'\0')
        digest.update(parameter.detach().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _model(cfg, seed, bio):
    torch.manual_seed(seed)
    if bio:
        model = GPTSynaptic(GPTSynapticConfig(sequence_len=cfg.sequence_len, vocab_size=cfg.vocab_size, n_layer=1,
            n_head=2, n_kv_head=2, n_embd=cfg.n_embd, init_seed=seed, syn_cfg=SynapticConfig(
                enable_presyn=True, enable_hebbian=True, stochastic_train_frac=0.0, native_presyn=False)))
    else:
        model = GPT(GPTConfig(sequence_len=cfg.sequence_len, vocab_size=cfg.vocab_size, n_layer=1,
                              n_head=2, n_kv_head=2, n_embd=cfg.n_embd, init_seed=seed))
    model.init_weights()
    return model


def _logits(model, x, *, training):
    if isinstance(model, GPTSynaptic):
        return model(x, train_mode=training)[0]
    return model(x)


def _labels(x, task, vocab):
    return (x + RULE[task]) % vocab


@torch.inference_mode()
def _score(model, x, vocab) -> dict:
    model.eval()
    log_probs = _logits(model, x, training=False).float().log_softmax(-1)
    if not torch.isfinite(log_probs).all():
        raise FloatingPointError('held-out model log probabilities are non-finite')
    predictions = log_probs.argmax(-1)
    measurements = {}
    for task in RULE:
        y = _labels(x, task, vocab)
        true_logp = log_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        measurements[task] = {'correct': int((predictions == y).sum()), 'tokens': y.numel(),
                              'nll': float(-true_logp.mean()), 'gold_log_probability': true_logp.flatten().tolist()}
    return {'predictions': predictions.tolist(), 'measurements': measurements}


def _reference_fisher(model, batches):
    """Diagonal empirical Fisher of per-sequence mean target log likelihood, independent of the candidate."""
    model.eval()
    references = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    diagonal = {n: torch.zeros_like(p) for n, p in references.items()}
    for x, y in batches:
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(x).flatten(0, 1), y.flatten()).backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                diagonal[n].add_(p.grad.detach().square())
    for value in diagonal.values():
        value.div_(len(batches))
    return references, diagonal


def _run_arm(cfg, seed, mode, probe, streams) -> dict:
    started = time.perf_counter()
    bio = mode.startswith('bio')
    family = 'sleep' if mode in SLEEP else 'gauge'
    model = _model(cfg, seed, bio)
    initial_sha = _weights(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    replay = PrioritizedReplayBuffer(capacity=32)
    controller = SleepConsolidationController(consolidation_lr=0.1)
    exposures = {k: 0 for k in ('optimizer', 'probes', 'offline', 'fisher')}
    current = 'optimizer'

    def count_tokens(_module, inputs):
        exposures[current] += inputs[0].numel()

    handle = model.register_forward_pre_hook(count_tokens)
    sleep_reports = []
    scores = {}
    fisher = None
    ewc_state = None
    fisher_sum = None
    penalties = []
    replayed_a = 0
    wake_writes = 0
    layers = get_synaptic_layers(model)
    offline_seconds = 0.0
    phases = ('A', 'B', 'C') if family == 'sleep' else ('A', 'B', 'return')
    rng = torch.Generator().manual_seed(50000 + seed)
    memory_a = streams['A'].flatten(0, 1)
    last_finite_loss = None

    def failed(phase, step, reason):
        handle.remove()
        return {'status': 'failed', 'mode': mode, 'seed': seed, 'phase': phase, 'step_in_phase': step,
                'reason': reason, 'last_finite_training_loss': last_finite_loss, 'completed_phase_scores': scores,
                'forward_tokens_at_stop': exposures, 'wake_layer_writes_at_stop': wake_writes,
                'initial_parameter_sha256': initial_sha, 'parameter_sha256_at_stop': _weights(model),
                'elapsed_seconds': time.perf_counter() - started}

    for index, phase in enumerate(phases):
        task = 'A' if phase == 'return' else phase
        current = 'optimizer'
        model.train()
        for step, original in enumerate(streams[phase], start=1):
            x = original
            y = _labels(x, task, cfg.vocab_size)
            if mode == 'supervised_replay' and phase == 'B':
                n = cfg.batch_size // 2
                old = memory_a[torch.randint(len(memory_a), (n,), generator=rng)]
                x = torch.cat([x[n:], old])
                y = torch.cat([y[n:], _labels(old, 'A', cfg.vocab_size)])
                replayed_a += old.numel()
            if isinstance(model, GPTSynaptic):
                model.reset_sequence_state(reset_fast_weights=False, reset_consolidation=False)
            optimizer.zero_grad(set_to_none=True)
            logits = _logits(model, x, training=True)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
            observed_loss = float(loss.detach())
            if not math.isfinite(observed_loss):
                return failed(phase, step, 'non-finite training cross entropy')
            last_finite_loss = observed_loss
            if family == 'sleep':
                replay.push(x, surprise_score=observed_loss)
            if phase != 'A' and mode in ('gauge_penalty', 'ewc'):
                if fisher is not None:
                    penalty = fisher.compute_penalty()
                else:
                    assert ewc_state is not None
                    refs, diag = ewc_state
                    penalty = 0.5 * sum((diag[n] * (p - refs[n]).square()).sum()
                                        for n, p in model.named_parameters() if n in refs)
                observed_penalty = float(penalty.detach())
                if not math.isfinite(observed_penalty):
                    return failed(phase, step, 'non-finite consolidation penalty')
                penalties.append(observed_penalty)
                loss = loss + penalty  # registered EWC coefficient 1
            loss.backward()
            optimizer.step()
            # A new independent batch resets eligibility. Commit the just-finished wake
            # step first, at the same safe optimizer boundary for both biological arms.
            landed = 0
            with torch.no_grad():
                for layer in layers:
                    if layer._plasticity_pending:
                        layer._apply_hebb_weight_writes(layer._last_gate_scale)
                        layer._plasticity_pending = False
                        landed += 1
            if landed != len(layers):
                raise ValueError('every dense Hebbian layer must commit its one pending wake write')
            wake_writes += landed
        current = 'probes'
        try:
            scores[phase] = _score(model, probe, cfg.vocab_size)
        except FloatingPointError as exc:
            return failed(phase, len(streams[phase]), str(exc))
        if phase == 'A' and family == 'gauge':
            current = 'fisher'
            batches = [(v[None], _labels(v[None], 'A', cfg.vocab_size)) for v in memory_a[:8]]
            if mode == 'gauge_penalty':
                fisher = FisherNaturalConsolidator(model)
                fisher.update_fisher_estimates(batches, num_samples=len(batches))
                diagonal = fisher.fisher_diagonal
            else:
                ewc_state = _reference_fisher(model, batches)
                diagonal = ewc_state[1]
            fisher_sum = sum(float(v.sum()) for v in diagonal.values())
        if family == 'sleep' and index < 2:
            current = 'offline'
            offline_start = time.perf_counter()
            # All controls pay for the same count of frozen replay observations; update only the sleep arm.
            np_state = np.random.get_state()
            np.random.seed(70000 + seed * 2 + index)
            try:
                if mode == 'bio_with_sleep':
                    try:
                        rep = controller.run_sleep_phase(model, replay, sleep_steps=cfg.sleep_steps, batch_size=4)
                    except FloatingPointError as exc:
                        return failed(phase, len(streams[phase]), str(exc))
                    sleep_reports.append(rep)
                else:
                    with torch.inference_mode():
                        for _ in range(cfg.sleep_steps):
                            items, _indices = replay.sample(4)
                            _logits(model, torch.stack([item.tokens for item in items]), training=False)
            finally:
                np.random.set_state(np_state)
                offline_seconds += time.perf_counter() - offline_start
    handle.remove()
    last = phases[-1]
    before, after = (scores[p]['measurements']['A'] for p in ('A', last))
    return {'status': 'completed', 'mode': mode, 'seed': seed, 'initial_parameter_sha256': initial_sha, 'final_parameter_sha256': _weights(model),
            'phase_scores': scores, 'forward_tokens': exposures, 'supervised_A_replay_tokens_during_B': replayed_a,
            'a_accuracy_forgetting': (before['correct'] - after['correct']) / before['tokens'],
            'a_nll_forgetting': after['nll'] - before['nll'], 'fisher_trace': fisher_sum,
            'wake_layer_writes': wake_writes, 'wake_layers': len(layers), 'offline_seconds': offline_seconds,
            'offline_optimizer_steps': 0,
            'max_applied_penalty': max(penalties) if penalties else None, 'sleep_updates': sleep_reports,
            'elapsed_seconds': time.perf_counter() - started}


def _compare(rows, name, control, metric):
    cells = {mode: {r['seed']: r[metric] for r in rows if r['mode'] == mode} for mode in (name, control)}
    stat = paired_comparison(cells[name], cells[control], lower_is_better=True)
    if stat is None:
        return {'seeds': len(set(cells[name]) & set(cells[control])), 'mean_difference': None,
                'paired_bootstrap_95': None, 'paired_t_p': None, 'wilcoxon_p': None}
    return {'seeds': stat.n_pairs, 'mean_difference': stat.mean_delta, 'paired_bootstrap_95': [stat.delta_ci_low, stat.delta_ci_high],
            'paired_t_p': stat.t_p_value, 'wilcoxon_p': stat.wilcoxon_p_value}


def run_continual_benchmark(config: ContinualEvalConfig) -> dict:
    rows, failures, data = [], [], {}
    for seed in config.seeds:
        probe, streams = _inputs(config, seed)
        data[seed] = {'probe_inputs': probe.tolist(), 'training_sha256': {
            k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k, v in streams.items()}}
        with torch.random.fork_rng(devices=[]):
            cohort = [_run_arm(config, seed, mode, probe, streams) for mode in SLEEP + GAUGE]
        completed = [r for r in cohort if r['status'] == 'completed']
        failures.extend(r for r in cohort if r['status'] == 'failed')
        for group in (SLEEP, GAUGE):
            budgets = [r['forward_tokens'] for r in completed if r['mode'] in group]
            if budgets and any(b != budgets[0] for b in budgets):
                raise ValueError('forward token budgets differ within comparison family')
        gauge, ewc, replay = (next(r for r in cohort if r['mode'] == mode) for mode in GAUGE)
        bio_writes = [r['wake_layer_writes'] for r in completed if r['mode'].startswith('bio')]
        if any(v <= 0 for v in bio_writes) or len(set(bio_writes)) > 1:
            raise ValueError('biological treatments require identical nonzero numbers of committed wake writes')
        if len({r['initial_parameter_sha256'] for r in (gauge, ewc, replay)}) != 1:
            raise ValueError('gauge comparison requires identical starting parameters')
        rows.extend(completed)
        Console().print(f'Finished {len(completed)} of {len(cohort)} continual-learning arms for seed {seed}.')
    contrasts = {f'{t}_minus_{c}': {m: _compare(rows, t, c, m) for m in ('a_accuracy_forgetting', 'a_nll_forgetting')}
                 for t, c in ((SLEEP[2], SLEEP[0]), (SLEEP[2], SLEEP[1]), (GAUGE[0], GAUGE[1]), (GAUGE[0], GAUGE[2]))}
    learned = {}
    for mode in SLEEP + GAUGE:
        phases = ('A', 'B', 'C') if mode in SLEEP else ('A', 'B')
        mode_rows = [r for r in rows if r['mode'] == mode]
        learned[mode] = len(mode_rows) == len(config.seeds) and all(
            m['correct'] >= .75 * m['tokens'] for r in mode_rows
            for p in phases for m in [r['phase_scores'][p]['measurements'][p]])
    # Preserve the original favorable final-A accuracy and paired t-test vs vanilla, and also demand a matched sleep effect.
    retest = {}
    for r in rows:
        end = 'C' if r['mode'] in SLEEP else 'return'
        measurement = r['phase_scores'][end]['measurements']['A']
        r['final_a_error'] = 1 - measurement['correct'] / measurement['tokens']
    for control in SLEEP[:2]:
        retest[control] = _compare(rows, SLEEP[2], control, 'final_a_error')
    passed = all(learned[m] for m in SLEEP) and all(
        p['mean_difference'] < 0 and p['paired_bootstrap_95'][1] < 0 and p['paired_t_p'] < .05 and p['wilcoxon_p'] < .05
        for p in retest.values())
    def end_hashes(mode):
        return {r['seed']: r['final_parameter_sha256'] for r in rows if r['mode'] == mode}

    gauge_hashes, ewc_hashes = end_hashes('gauge_penalty'), end_hashes('ewc')
    equivalent = gauge_hashes == ewc_hashes if set(gauge_hashes) == set(ewc_hashes) == set(config.seeds) else None
    return {'config': asdict(config), 'task_rules': RULE, 'data': data, 'rows': rows, 'failed_arms': failures, 'comparisons': contrasts,
            'learned_all_required_phases': learned, 'sleep_retest_error_comparisons': retest, 'sleep_superiority': passed,
            'gauge_equals_independent_ewc_final_parameters': equivalent, 'holonomy': None, 'holonomy_forgetting_correlation': None,
            'gauge_verdict': 'unidentifiable_connection_and_transport',
            'scope': 'Synthetic learned token rules on the same input support, fresh training draws, disjoint held-out sequences. Sleep A-B-C and gauge A-B-A have separate matched forward/backward-example budgets. The supervised replay arm replaces half of B examples by A; reported token counts do not imply equal wall time or FLOPs.',
            'holonomy_limit': 'The candidate exposes only a diagonal Fisher reference penalty. CurvatureMonitor accepts a caller-supplied holonomy number and supplies no learned connection or path transport.',
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'torch_version': str(torch.__version__)}


def print_continual_summary(summary: dict, console: Console | None = None) -> None:
    c = console or Console()
    c.print(summary['comparisons'], {'sleep_superiority': summary['sleep_superiority'], 'gauge_verdict': summary['gauge_verdict']})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run_continual_benchmark(ContinualEvalConfig())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, allow_nan=False, separators=(',', ':')) + '\n', encoding='utf-8')
    print_continual_summary(report)


if __name__ == '__main__':
    main()
