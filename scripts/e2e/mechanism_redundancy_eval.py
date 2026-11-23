"""A conditional two-factor study of direct metabolic and glial bias in trained MoE routing."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console

from bio_inspired_nanochat.ablation_matrix import structural_columns
from bio_inspired_nanochat.eval_stats import bootstrap_ci
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticLinear, SynapticMoE


OFF, MET, GLIA, BOTH = 'moe_no_metabolism', 'moe_fixed', 'moe_glial_no_metabolism', 'moe_add_glial_homeostasis'
ARMS = (OFF, MET, GLIA, BOTH)
HOLDOUT_STARTS = (0, 4, 8, 12)
TRAIN_STARTS = [i for i in range(16) if i not in HOLDOUT_STARTS]


@dataclass(frozen=True)
class RedundancyConfig:
    seeds: tuple[int, ...] = (401, 409, 419, 421)
    train_steps: int = 16
    eval_batches: int = 4
    batch_size: int = 8
    sequence_len: int = 8
    n_embd: int = 16
    lr: float = 0.003

    def __post_init__(self):
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError('need at least two unique training seeds')
        if min(self.train_steps, self.eval_batches, self.batch_size, self.sequence_len, self.n_embd) < 1:
            raise ValueError('training and evaluation dimensions must be positive')
        if self.batch_size % 4 or self.n_embd % 4 or not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError('batch size and width must be divisible by 4, and lr must be positive and finite')


def arm_configs() -> dict[str, SynapticConfig]:
    columns = {c.config_id: c for c in structural_columns()}
    configs = {}
    for name in (OFF, MET, BOTH):
        cfg = columns[name].build_syn_cfg()
        if cfg is None or columns[name].train_overrides.get('use_moe') != 1:
            raise ValueError(f'{name} requires a synaptic MoE')
        configs[name] = cfg
    configs[GLIA] = replace(configs[BOTH], enable_metabolism=False)
    for name, cfg in configs.items():
        expected = replace(configs[OFF], enable_metabolism=name in (MET, BOTH), glial_homeostasis=name in (GLIA, BOTH))
        if cfg != expected:
            raise ValueError(f'{name} differs from its matched control beyond the two registered factors')
    return configs


def _batch(starts: torch.Tensor, length: int):
    ids = (starts[:, None] + torch.arange(length + 1)) % 16
    return ids[:, :-1], ids[:, 1:]


def _weight_hash(model) -> str:
    digest = hashlib.sha256()
    for name, p in model.named_parameters():
        digest.update(name.encode() + b'\0')
        digest.update(p.detach().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _build(cfg: RedundancyConfig, syn: SynapticConfig, seed: int, *, moe: bool = True):
    torch.manual_seed(seed)
    model = GPTSynaptic(GPTSynapticConfig(sequence_len=cfg.sequence_len, vocab_size=16, n_layer=1, n_head=2,
                                         n_kv_head=2, n_embd=cfg.n_embd, syn_cfg=syn, init_seed=seed,
                                         use_moe=moe, num_experts=8, moe_top_k=2))
    model.init_weights()
    return model


def _run_single_seed_arm(name: str, syn: SynapticConfig, cfg: RedundancyConfig, seed: int) -> dict:
    model = _build(cfg, syn, seed)
    router, = [m for m in model.modules() if isinstance(m, SynapticMoE)]
    initial = _weight_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    generator = torch.Generator().manual_seed(100000 + seed)
    training_starts = []
    for _ in range(cfg.train_steps):
        positions = torch.randint(len(TRAIN_STARTS), (cfg.batch_size,), generator=generator)
        starts = torch.tensor(TRAIN_STARTS)[positions]
        training_starts.append(starts.tolist())
        x, y = _batch(starts, cfg.sequence_len)
        model.reset_sequence_state()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y, train_mode=True)
        loss.backward()
        optimizer.step()
    # Frozen validation uses the ordinary evaluation path; the training auxiliary loss is excluded.
    model.eval()
    logits_by_batch, route_indices = [], []
    starts = torch.tensor(HOLDOUT_STARTS * (cfg.batch_size // 4))
    x, y = _batch(starts, cfg.sequence_len)
    with torch.inference_mode():
        for _ in range(cfg.eval_batches):
            model.reset_sequence_state()
            logits, _ = model(x, train_mode=False)
            logits_by_batch.append(logits.float())
            route_indices.append(router.last_ctx['indices'].clone())
    if not all(torch.equal(logits_by_batch[0], item) for item in logits_by_batch[1:]):
        raise ValueError('duplicate frozen validation input produced different predictions')
    if not all(torch.equal(route_indices[0], item) for item in route_indices[1:]):
        raise ValueError('duplicate frozen validation input produced different routes')
    logits = logits_by_batch[0][:4]
    direct = 0.1 * (router.energy - router.fatigue)
    glia = router.glial
    return {'arm': name, 'seed': seed, 'initial_parameter_sha256': initial, 'trained_parameter_sha256': _weight_hash(model),
            'training_starts': training_starts, 'nll': float(F.cross_entropy(logits.reshape(-1, 16), y[:4].reshape(-1))),
            'accuracy': float((logits.argmax(-1) == y[:4]).float().mean()),
            'heldout_unique_starts': list(HOLDOUT_STARTS), 'heldout_logits': logits.tolist(),
            'heldout_targets': y[:4].tolist(), 'heldout_top_two': route_indices[0][:4].tolist(),
            'direct_bias_applied': syn.enable_metabolism, 'energy': router.energy.tolist(), 'fatigue': router.fatigue.tolist(),
            'metabolic_bias_range': float(direct.max() - direct.min()) if syn.enable_metabolism else 0.0,
            'glial_observations': int(glia.steps) if glia is not None else 0,
            'glial_bias_range': float(glia.routing_bias.max() - glia.routing_bias.min()) if glia is not None else 0.0}


def interaction(arms: dict[str, dict[int, float]], *, margin: float, signals_active: bool, learned: bool) -> dict:
    if set(arms) != set(ARMS) or margin <= 0 or not math.isfinite(margin):
        raise ValueError('four factorial conditions and a finite positive margin are required')
    seeds = sorted(arms[OFF])
    if len(seeds) < 2 or any(set(arms[a]) != set(seeds) for a in ARMS):
        raise ValueError('every condition must have the same complete set of paired seeds')
    effects = {s: arms[BOTH][s] - arms[MET][s] - arms[GLIA][s] + arms[OFF][s] for s in seeds}
    deltas = np.asarray(list(effects.values()))
    low, high = bootstrap_ci(deltas, seed=0)
    conditional = {}
    for treatment, control in ((MET, OFF), (GLIA, OFF), (BOTH, MET), (BOTH, GLIA)):
        differences = [arms[treatment][s] - arms[control][s] for s in seeds]
        conditional[f'{treatment}_minus_{control}'] = {
            'per_seed': dict(zip(seeds, differences, strict=True)), 'mean': float(np.mean(differences)),
            'paired_bootstrap_95': list(bootstrap_ci(np.asarray(differences), seed=0))}
    if not signals_active or not learned:
        verdict = 'inconclusive_activity_or_learning_control'
    elif len(seeds) < 4:
        verdict = 'inconclusive_too_few_seeds'
    elif low > margin:
        verdict = 'positive_interaction_on_this_task'
    elif high < -margin:
        verdict = 'negative_interaction_on_this_task'
    elif low > -margin and high < margin:
        verdict = 'within_practical_margin_on_this_task'
    else:
        verdict = 'inconclusive_precision'
    return {'per_seed': effects, 'mean': float(deltas.mean()), 'paired_bootstrap_95': [low, high],
            'conditional_effects': conditional,
            'margin_nats': margin, 'verdict': verdict}


def _reset_witness(seed: int) -> dict:
    """Replay the old single-forward/reset pattern against two successive real Hebbian writes."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 8, 16, generator=generator)
    outcomes = {}
    for reset in (True, False):
        torch.manual_seed(seed)
        layer = SynapticLinear(16, 16, SynapticConfig(enable_hebbian=True, bistable_latch=True, bdnf_scale=1.0))
        pending_before_reset = []
        for _ in range(3):
            pending_before_reset.append(layer._plasticity_pending)
            if reset:
                layer.reset_sequence_state(reset_fast_weights=False, reset_consolidation=False)
            y = layer(x, torch.tensor(1.0), torch.tensor(1.0), update_mem=True)
            y.square().mean().backward()
            layer.zero_grad(set_to_none=True)
        assert layer.post is not None
        outcomes['historical_reset' if reset else 'successive_forwards'] = {
            'pending_before_forward_or_reset': pending_before_reset,
            'slow_consolidation_l1': float(layer.post.slow.detach().abs().sum()),
            'bdnf_trace_l1': float(layer.post.bdnf_hebb_accum.abs().sum())}
    return outcomes


def run_mechanism_redundancy_evaluation(cfg: RedundancyConfig, *, verbose: bool = True) -> dict:
    configs = arm_configs()
    rows = []
    for seed in cfg.seeds:
        quartet = [_run_single_seed_arm(name, configs[name], cfg, seed) for name in ARMS]
        if len({r['initial_parameter_sha256'] for r in quartet}) != 1 or any(
                r['training_starts'] != quartet[0]['training_starts'] for r in quartet):
            raise ValueError('factorial quartet differs in initial parameters or training input')
        rows.extend(quartet)
        if verbose:
            Console().print(f'Completed 4 matched MoE arms for training seed {seed}.')
    losses = {a: {r['seed']: r['nll'] for r in rows if r['arm'] == a} for a in ARMS}
    learned = all(v < math.log(16) - 0.20 for v in losses[OFF].values())
    signals_active = all((not r['direct_bias_applied'] or r['metabolic_bias_range'] > 0) and
                         (r['arm'] not in (GLIA, BOTH) or (r['glial_bias_range'] > 0 and r['glial_observations'] == cfg.train_steps))
                         for r in rows)
    return {'config': asdict(cfg), 'task': 'next token equals (current token + 1) modulo 16; four reserved evaluation starts; repeated identical validation batches; only training seeds are replicates',
            'protocol': 'Fixed eight-expert top-two MoE, reference backends. Training includes usual auxiliary balance loss; evaluation is pure next-token cross entropy. Interaction = both - direct - glial + neither. Paired percentile bootstrap over training seeds (10,000, seed 0). Practical margin 0.01 nats; require live nonuniform bias for both toggles and baseline at least 0.20 nats below log(16) on every seed.',
            'limits': 'The metabolism flag gates the direct energy/fatigue router bias. The EMAs continue evolving in all arms and supply glial feedback when enabled. This experiment does not ablate all metabolism; it has no lifecycle or NeuroScore treatment. BDNF and the latch require a separate matched Hebbian factorial.',
            'learned_baseline': learned, 'signals_active': signals_active, 'rows': rows,
            'interaction': interaction(losses, margin=0.01, learned=learned, signals_active=signals_active),
            'historical_reset_witness': {s: _reset_witness(s) for s in cfg.seeds},
            'mechanism_configs': {k: asdict(v) for k, v in configs.items()},
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'torch_version': str(torch.__version__)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--seeds', nargs='+', type=int, default=[401, 409, 419, 421])
    parser.add_argument('--eval-batches', type=int, default=4)
    parser.add_argument('--device', choices=['cpu'], default='cpu')
    args = parser.parse_args(argv)
    cfg = RedundancyConfig(seeds=tuple(args.seeds), train_steps=args.steps, eval_batches=args.eval_batches)
    report = run_mechanism_redundancy_evaluation(cfg)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, allow_nan=False, separators=(',', ':')) + '\n', encoding='utf-8')
    Console().print(report['interaction'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
