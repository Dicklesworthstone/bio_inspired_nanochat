"""
Standardized evaluation harness (bio vs vanilla).

Run:
  python -m scripts.eval_matrix run --preset vanilla --seed 1337 --checkpoint-dir /checkpoints/vanilla_s1337
  python -m scripts.eval_matrix matrix --presets vanilla,bio_all --seeds 1337,1338 \
    --checkpoint-dir '/checkpoints/{preset}_s{seed}'

Design reference:
  docs/eval_benchmark_matrix.md
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal, Optional, Sequence, TypeVar, cast

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from bio_inspired_nanochat.common import autodetect_device_type, compute_cleanup, compute_init
from bio_inspired_nanochat.dataloader import tokenizing_distributed_data_loader
from bio_inspired_nanochat.loss_eval import evaluate_bpb
from bio_inspired_nanochat.torch_imports import F, Tensor, torch
from bio_inspired_nanochat.tokenizer import get_token_bytes, get_tokenizer

from bio_inspired_nanochat.ablation_registry import MECHANISMS, apply_preset
from bio_inspired_nanochat.checkpoint_manager import (
    list_checkpoint_steps,
    load_checkpoint,
    synaptic_config_from_meta,
)
from bio_inspired_nanochat.gpt import GPT, GPTConfig
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.results_registry import (
    DEFAULT_REGISTRY,
    RunRecord,
    append_record,
    make_record,
)
from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticMoE

from scripts.base_eval import evaluate_model

console = Console()

PresetId = Literal[
    "vanilla",
    "bio_all",
    "bio_no_presyn",
    "bio_no_hebbian",
    "bio_no_metabolism",
    "bio_no_genome",
    "bio_no_stochastic_release",
    "bio_no_doc2",
    "bio_no_bdnf",
    "bio_no_septin_barrier",
]

DEFAULT_ABLATION_PRESETS: tuple[PresetId, ...] = (
    "vanilla",
    "bio_all",
    "bio_no_presyn",
    "bio_no_hebbian",
    "bio_no_metabolism",
    "bio_no_genome",
    "bio_no_stochastic_release",
    "bio_no_doc2",
    "bio_no_bdnf",
    "bio_no_septin_barrier",
)

SUMMARY_FIELDS: tuple[str, ...] = (
    "status",
    "error",
    "run_id",
    "run_dir",
    "preset",
    "seed",
    "recipe_source",
    "checkpoint_dir",
    "checkpoint_step",
    "checkpoint_git_sha",
    "checkpoint_config_hash",
    "data",
    "device_type",
    "world_size",
    "init_type",
    "sequence_len",
    "vocab_size",
    "n_layer",
    "n_head",
    "n_embd",
    "use_moe",
    "num_experts",
    "moe_top_k",
    "device_batch_size",
    "total_batch_size_tokens",
    "grad_accum_steps",
    "train_tokens_requested",
    "train_tokens_processed",
    "steps",
    "eval_tokens",
    "eval_steps",
    "eval_bpb",
    "core_eval",
    "core_max_per_task",
    "ece_bins",
    "dead_expert_threshold",
    "continual_tasks",
    "continual_exposures",
    "walltime_sec",
    "tok_per_sec",
    "train_loss_final",
    "val_loss",
    "val_ppl",
    "val_bpb",
    "core_metric",
    "id_ece",
    "ood_auroc",
    "forgetting_rate",
    "moe_gini",
    "dead_expert_frac",
    "niah_acc",
    "recall_by_length",
    "forgetting_by_task",
    "capability_metric_status",
)


@dataclass(frozen=True)
class HarnessRunSummary:
    run_id: str
    preset: str
    seed: int
    train_tokens_requested: int
    train_tokens_processed: int
    walltime_sec: float
    tok_per_sec: float
    train_loss_final: Optional[float]
    val_loss: float
    val_ppl: float
    val_bpb: Optional[float]
    core_metric: Optional[float]
    id_ece: Optional[float]
    ood_auroc: Optional[float]
    forgetting_rate: Optional[float]
    moe_gini: Optional[float]
    dead_expert_frac: Optional[float]
    niah_acc: Optional[float]
    recall_by_length: dict[str, float]
    forgetting_by_task: dict[str, dict[str, float]]


@dataclass(frozen=True)
class RoutingMetricSummary:
    moe_gini: Optional[float]
    dead_expert_frac: Optional[float]
    layers: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ContinualMetricSummary:
    forgetting_rate: Optional[float]
    by_task: dict[str, dict[str, float]]
    accuracy_matrix: list[list[Optional[float]]]
    status: str
    reason: Optional[str]


ComputeRuntime = tuple[bool, int, int, int, torch.device]
_ListEntry = TypeVar("_ListEntry", int, str)


def _require_unique(values: Sequence[_ListEntry], *, name: str) -> None:
    seen: set[_ListEntry] = set()
    duplicates: list[_ListEntry] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f"{name} must not contain duplicates: {duplicates}")


def _parse_int_list(csv_str: str) -> list[int]:
    out: list[int] = []
    for part in csv_str.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    if not out:
        raise ValueError("Expected a non-empty comma-separated list of ints")
    _require_unique(out, name="integer list")
    return out


def _parse_str_list(csv_str: str) -> list[str]:
    out: list[str] = []
    for part in csv_str.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(p)
    if not out:
        raise ValueError("Expected a non-empty comma-separated list of strings")
    _require_unique(out, name="string list")
    return out


def _batch_output_dir(out_dir: Path, batch_id: str) -> Path:
    """Return one safe batch child without allowing the ID to escape its root."""
    if not isinstance(batch_id, str) or not batch_id or Path(batch_id).is_absolute():
        raise ValueError("batch_id must be a non-empty relative subdirectory name")
    if "\\" in batch_id:
        raise ValueError("batch_id must not contain path separators")
    candidate = out_dir / batch_id
    if candidate.resolve().parent != out_dir.resolve():
        raise ValueError("batch_id must name one direct child beneath out_dir")
    return candidate


def _set_seed(seed: int, *, device_type: str) -> torch.Generator:
    torch.manual_seed(seed)
    if device_type == "cuda":
        torch.cuda.manual_seed_all(seed)
    g = torch.Generator(device=device_type)
    g.manual_seed(seed)
    return g


def _synthetic_loader(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> Iterator[tuple[Tensor, Tensor]]:
    while True:
        start = torch.randint(0, vocab_size, (batch_size, 1), generator=generator, device=device)
        ar = torch.arange(seq_len + 1, device=device).view(1, -1)
        toks = (start + ar) % vocab_size
        yield toks[:, :-1].to(torch.long), toks[:, 1:].to(torch.long)


def _forward_logits(model: Any, idx: Tensor, *, train_mode: bool) -> Tensor:
    out = (
        model(idx, train_mode=train_mode)
        if "train_mode" in inspect.signature(model.forward).parameters
        else model(idx)
    )
    if isinstance(out, tuple):
        logits = out[0]
    elif hasattr(out, "logits"):
        logits = out.logits
    else:
        logits = out
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"Expected logits Tensor, got {type(logits)}")
    return logits


def _get_logits(model: Any, idx: Tensor) -> Tensor:
    """Return inference logits without updating synaptic state."""
    return _forward_logits(model, idx, train_mode=False)


def _deterministic_ood_tokens(tokens: Tensor, *, vocab_size: int, seed: int) -> Tensor:
    """Destroy sequence structure with a deterministic token/position hash.

    The transform is deliberately private-RNG-free: computing the OOD arm cannot advance model or
    data-loader RNG state. Every token changes while remaining inside the checkpoint vocabulary.
    """
    if vocab_size < 2:
        raise ValueError("OOD corruption requires vocab_size >= 2")
    batch = torch.arange(tokens.shape[0], device=tokens.device, dtype=torch.int64).view(-1, 1)
    position = torch.arange(tokens.shape[1], device=tokens.device, dtype=torch.int64).view(1, -1)
    values = tokens.to(torch.int64)
    mixed = (
        31 * values.square()
        + 17 * values
        + 13 * position.square()
        + 7 * batch
        + 19 * int(seed)
        + 23
    )
    corrupted = torch.remainder(mixed, vocab_size)
    return torch.where(corrupted == values, torch.remainder(corrupted + 1, vocab_size), corrupted)


def _binary_auroc(id_scores: Sequence[float], ood_scores: Sequence[float]) -> float:
    """Exact tie-aware AUROC, treating OOD examples as the positive class."""
    if not id_scores or not ood_scores:
        raise ValueError("AUROC requires at least one ID and one OOD score")
    labelled = [(float(score), 0) for score in id_scores]
    labelled.extend((float(score), 1) for score in ood_scores)
    if not all(math.isfinite(score) for score, _ in labelled):
        raise ValueError("AUROC scores must be finite")
    labelled.sort(key=lambda item: item[0])
    positive_rank_sum = 0.0
    start = 0
    while start < len(labelled):
        end = start + 1
        while end < len(labelled) and labelled[end][0] == labelled[start][0]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(label for _, label in labelled[start:end])
        start = end
    n_positive = len(ood_scores)
    n_negative = len(id_scores)
    return (
        positive_rank_sum - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)


def _gini_coefficient(values: Tensor) -> float:
    """Population Gini for non-negative routing counts (zero for an all-zero vector)."""
    x = values.detach().to(dtype=torch.float64).reshape(-1)
    if x.numel() == 0 or bool((x < 0).any()):
        raise ValueError("Gini requires a non-empty non-negative vector")
    total = float(x.sum().item())
    if total == 0.0:
        return 0.0
    ordered = x.sort().values
    n = ordered.numel()
    ranks = torch.arange(1, n + 1, device=ordered.device, dtype=torch.float64)
    return float((((2.0 * ranks - n - 1.0) * ordered).sum() / (n * ordered.sum())).item())


def _routing_counts(model: Any) -> dict[str, Tensor]:
    return {
        name: torch.zeros(module.num_experts, dtype=torch.float64, device=module.energy.device)
        for name, module in model.named_modules()
        if isinstance(module, SynapticMoE)
    }


def _accumulate_routing_counts(model: Any, counts: dict[str, Tensor]) -> None:
    modules = dict(model.named_modules())
    for name, total in counts.items():
        module = modules[name]
        if not isinstance(module, SynapticMoE):
            raise TypeError(f"routing module {name!r} changed type during evaluation")
        indices = module.last_ctx.get("indices")
        if indices is None:
            continue
        total.add_(torch.bincount(indices.reshape(-1), minlength=module.num_experts).to(total))


def _summarize_routing_counts(
    counts: dict[str, Tensor], *, dead_expert_threshold: float
) -> RoutingMetricSummary:
    if not 0.0 <= dead_expert_threshold < 1.0:
        raise ValueError("dead_expert_threshold must be in [0, 1)")
    if not counts:
        return RoutingMetricSummary(None, None, {})
    layers: dict[str, dict[str, Any]] = {}
    ginis: list[float] = []
    dead = 0
    experts = 0
    for name, raw_counts in counts.items():
        total = float(raw_counts.sum().item())
        if total <= 0.0:
            raise ValueError(f"MoE layer {name!r} emitted no routing assignments")
        shares = raw_counts / total
        gini = _gini_coefficient(raw_counts)
        layer_dead = int((shares < dead_expert_threshold).sum().item())
        layer_experts = raw_counts.numel()
        ginis.append(gini)
        dead += layer_dead
        experts += layer_experts
        layers[name] = {
            "assignments": [int(value) for value in raw_counts.cpu().tolist()],
            "routing_share": [float(value) for value in shares.cpu().tolist()],
            "gini": gini,
            "dead_experts": layer_dead,
            "num_experts": layer_experts,
            "dead_expert_threshold": dead_expert_threshold,
        }
    return RoutingMetricSummary(
        moe_gini=sum(ginis) / len(ginis),
        dead_expert_frac=dead / experts,
        layers=layers,
    )


def _distributed_scores(scores: list[float], *, ddp: bool) -> list[float]:
    if not ddp or not torch.distributed.is_initialized():
        return scores
    gathered: list[Optional[list[float]]] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, scores)
    return [score for rank_scores in gathered if rank_scores is not None for score in rank_scores]


@contextmanager
def _force_synaptic_eval_forward(model: Any):
    """Make generic evaluators call a synaptic model with ``train_mode=False``."""
    if "train_mode" not in inspect.signature(model.forward).parameters:
        yield model
        return
    original_forward = model.forward

    def eval_forward(idx: Tensor, targets: Optional[Tensor] = None, kv_cache=None, **kwargs: Any):
        kwargs.pop("train_mode", None)
        return original_forward(idx, targets, kv_cache, train_mode=False, **kwargs)

    model.forward = eval_forward
    try:
        yield model
    finally:
        model.forward = original_forward


def _val_loss_ppl_ece(
    model: Any,
    batches: Iterator[tuple[Tensor, Tensor]],
    *,
    steps: int,
    device_type: str,
    ddp: bool,
    ece_bins: int = 15,
    ood_seed: int = 0,
    dead_expert_threshold: float = 0.01,
) -> tuple[float, float, Optional[float], float, RoutingMetricSummary]:
    if ece_bins < 2:
        raise ValueError("ece_bins must be >= 2")
    model.train(False)
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )
    # ECE accumulator: bins on max prob of predicted token.
    accumulator_device = next(model.parameters()).device
    conf_sum = torch.zeros(ece_bins, dtype=torch.float64, device=accumulator_device)
    acc_sum = torch.zeros(ece_bins, dtype=torch.float64, device=accumulator_device)
    count = torch.zeros(ece_bins, dtype=torch.float64, device=accumulator_device)
    routing_counts = _routing_counts(model)
    id_uncertainty: list[float] = []
    ood_uncertainty: list[float] = []

    loss_sum = torch.zeros((), dtype=torch.float64, device=accumulator_device)
    valid_token_count = torch.zeros((), dtype=torch.int64, device=accumulator_device)
    for _ in range(steps):
        x, y = next(batches)
        with torch.no_grad(), autocast_ctx:
            logits = _get_logits(model, x).to(torch.float32)
            _accumulate_routing_counts(model, routing_counts)
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = y.reshape(-1)
            loss_flat = F.cross_entropy(
                logits_flat,
                targets_flat,
                reduction="none",
                ignore_index=-1,
            )
            valid = targets_flat >= 0
            if valid.any():
                loss_sum += loss_flat[valid].sum(dtype=torch.float64)
                valid_token_count += valid.sum()
            # All-masked batch (every target is ignore_index): contribute NOTHING
            # rather than a NaN or an artificial zero-weight observation.

            # ECE (optional): use max prob and correctness per token.
            probs = torch.softmax(logits_flat, dim=-1)
            conf, pred = probs.max(dim=-1)
            correct = (pred == targets_flat) & valid
            # Bin by confidence in [0,1]
            bins = torch.clamp((conf * ece_bins).to(torch.int64), 0, ece_bins - 1)
            for b in range(ece_bins):
                mask = (bins == b) & valid
                if mask.any():
                    conf_sum[b] += float(conf[mask].sum().item())
                    acc_sum[b] += float(correct[mask].float().sum().item())
                    count[b] += float(mask.sum().item())

            id_log_probs = torch.log_softmax(logits, dim=-1)
            id_probs = id_log_probs.exp()
            id_uncertainty.extend(
                float(value)
                for value in (-(id_probs * id_log_probs).sum(dim=-1).mean(dim=-1)).cpu().tolist()
            )
            ood_x = _deterministic_ood_tokens(
                x,
                vocab_size=logits.size(-1),
                seed=ood_seed,
            )
            ood_logits = _get_logits(model, ood_x).to(torch.float32)
            ood_log_probs = torch.log_softmax(ood_logits, dim=-1)
            ood_probs = ood_log_probs.exp()
            ood_uncertainty.extend(
                float(value)
                for value in (-(ood_probs * ood_log_probs).sum(dim=-1).mean(dim=-1)).cpu().tolist()
            )

    if ddp and torch.distributed.is_initialized():
        # Pool the numerator and denominator across ranks. Averaging rank-local
        # means biases the result whenever ranks observe different valid-token
        # counts (for example, masked or uneven final batches).
        torch.distributed.all_reduce(loss_sum, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(valid_token_count, op=torch.distributed.ReduceOp.SUM)
        for accumulator in (conf_sum, acc_sum, count):
            torch.distributed.all_reduce(accumulator, op=torch.distributed.ReduceOp.SUM)
        for layer_counts in routing_counts.values():
            torch.distributed.all_reduce(layer_counts, op=torch.distributed.ReduceOp.SUM)
    if int(valid_token_count.item()) == 0:
        raise ValueError("validation produced no valid target tokens")
    val_loss = loss_sum / valid_token_count
    val_loss_f = float(val_loss.item())
    val_ppl = float(math.exp(val_loss_f)) if math.isfinite(val_loss_f) else float("inf")

    ece: Optional[float]
    if float(count.sum().item()) == 0.0:
        ece = None
    else:
        conf_mean = conf_sum / count.clamp_min(1.0)
        acc_mean = acc_sum / count.clamp_min(1.0)
        weights = count / count.sum()
        ece = float((weights * (conf_mean - acc_mean).abs()).sum().item())
    id_uncertainty = _distributed_scores(id_uncertainty, ddp=ddp)
    ood_uncertainty = _distributed_scores(ood_uncertainty, ddp=ddp)
    ood_auroc = _binary_auroc(id_uncertainty, ood_uncertainty)
    routing = _summarize_routing_counts(
        routing_counts,
        dead_expert_threshold=dead_expert_threshold,
    )
    return val_loss_f, val_ppl, ece, ood_auroc, routing


def _apply_syn_preset(preset: str, syn_cfg: SynapticConfig) -> None:
    # Single source of truth lives in the ablation registry (hm4.7).
    apply_preset(preset, syn_cfg)


def _build_model(
    *,
    preset: PresetId,
    seed: int,
    device: torch.device,
    sequence_len: int,
    vocab_size: int,
    n_layer: int,
    n_head: int,
    n_embd: int,
    init_type: str,
    use_moe: bool,
    num_experts: int,
    moe_top_k: int,
) -> Any:
    if preset in {"vanilla"}:
        with torch.device("meta"):
            cfg = GPTConfig(
                sequence_len=sequence_len,
                vocab_size=vocab_size,
                n_layer=n_layer,
                n_head=n_head,
                n_kv_head=n_head,
                n_embd=n_embd,
                init_type=init_type,
                init_seed=seed,
            )
            model = GPT(cfg)
        model.to_empty(device=device)
        model.init_weights()
        return model

    syn_cfg = SynapticConfig()
    _apply_syn_preset(preset, syn_cfg)
    with torch.device("meta"):
        cfg = GPTSynapticConfig(
            sequence_len=sequence_len,
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_kv_head=n_head,
            n_embd=n_embd,
            synapses=True,
            syn_cfg=syn_cfg,
            use_moe=bool(use_moe),
            num_experts=int(num_experts),
            moe_top_k=int(moe_top_k),
            init_type=init_type,
            init_seed=seed,
        )
        model = GPTSynaptic(cfg)
    model.to_empty(device=device)
    model.init_weights()
    return model


def _resolve_checkpoint_dir(template: str, *, preset: str, seed: int) -> Path:
    """Resolve a matrix checkpoint template without silently accepting unknown fields."""
    try:
        rendered = template.format(preset=preset, seed=seed)
    except KeyError as exc:
        raise ValueError(
            "--checkpoint-dir only supports the {preset} and {seed} template fields"
        ) from exc
    if not rendered.strip():
        raise ValueError("resolved checkpoint directory is empty")
    return Path(rendered).expanduser().resolve()


def _load_base_train_checkpoint(
    checkpoint_dir: Path,
    *,
    step: int,
    preset: PresetId,
    seed: int,
    device: torch.device,
) -> tuple[Any, dict[str, Any], int]:
    """Rebuild and load one real ``base_train`` checkpoint without tokenizer side effects.

    Architecture, bio configuration, and learned tensors come exclusively from checkpoint
    metadata/state.  CLI architecture flags therefore cannot turn evaluation into a subtly
    different model.  The requested matrix cell is also checked against the checkpoint's model
    family, seed, and canonical ablation flags so a mislabeled row fails closed.
    """
    available_steps = list_checkpoint_steps(str(checkpoint_dir))
    if not available_steps:
        raise FileNotFoundError(f"no model_*.pt checkpoints found in {checkpoint_dir}")
    resolved_step = available_steps[-1] if step < 0 else step
    if resolved_step not in available_steps:
        raise FileNotFoundError(
            f"checkpoint step {resolved_step} not found in {checkpoint_dir}; "
            f"available={available_steps}"
        )

    model_data, _, meta_data = load_checkpoint(
        str(checkpoint_dir), resolved_step, device, load_optimizer=False
    )
    model_config = meta_data.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("base_train checkpoint metadata must contain a model_config object")

    is_synaptic = bool(meta_data.get("synapses", False))
    expects_synaptic = preset not in {"vanilla"}
    if is_synaptic != expects_synaptic:
        family = "synaptic" if is_synaptic else "vanilla"
        raise ValueError(f"preset {preset!r} does not match {family} checkpoint")

    checkpoint_seed = model_config.get(
        "init_seed", (meta_data.get("user_config") or {}).get("init_seed")
    )
    if checkpoint_seed is not None and int(checkpoint_seed) != seed:
        raise ValueError(
            f"matrix seed {seed} does not match checkpoint init_seed {checkpoint_seed}"
        )

    if is_synaptic:
        syn_cfg = synaptic_config_from_meta(meta_data)
        expected_syn_cfg = apply_preset(preset, SynapticConfig())
        mismatches = [
            mechanism.field
            for mechanism in MECHANISMS
            if getattr(syn_cfg, mechanism.field) != getattr(expected_syn_cfg, mechanism.field)
        ]
        if mismatches:
            raise ValueError(
                f"checkpoint bio flags do not match preset {preset!r}: {sorted(mismatches)}"
            )
        cfg = GPTSynapticConfig(
            sequence_len=int(model_config["sequence_len"]),
            vocab_size=int(model_config["vocab_size"]),
            n_layer=int(model_config["n_layer"]),
            n_head=int(model_config["n_head"]),
            n_kv_head=int(model_config.get("n_kv_head", model_config["n_head"])),
            n_embd=int(model_config["n_embd"]),
            synapses=True,
            syn_cfg=syn_cfg,
            dropout=float(model_config.get("dropout", 0.0)),
            use_moe=bool(model_config.get("use_moe", False)),
            num_experts=int(model_config.get("num_experts", 8)),
            moe_experts_per_layer=(
                tuple(int(value) for value in model_config["moe_experts_per_layer"])
                if model_config.get("moe_experts_per_layer") is not None
                else None
            ),
            moe_top_k=int(model_config.get("moe_top_k", 2)),
            moe_hidden_mult=int(model_config.get("moe_hidden_mult", 4)),
            moe_balance_loss=float(model_config.get("moe_balance_loss", 0.01)),
            structural_every=int(model_config.get("structural_every", 0)),
            init_type=str(model_config.get("init_type", "baseline")),
            init_seed=int(model_config.get("init_seed", seed)),
            tie_embeddings=bool(model_config.get("tie_embeddings", False)),
        )
        with torch.device("meta"):
            model = GPTSynaptic(cfg)
    else:
        cfg = GPTConfig(**model_config)
        with torch.device("meta"):
            model = GPT(cfg)

    if device.type in {"cpu", "mps"}:
        model_data = {
            name: value.float() if value.dtype == torch.bfloat16 else value
            for name, value in model_data.items()
        }
    model_data = {name.removeprefix("_orig_mod."): value for name, value in model_data.items()}
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.tie_weights()
    model.train(False)
    return model, meta_data, resolved_step


def _checkpoint_recipe_stats(
    meta_data: dict[str, Any], checkpoint_step: int
) -> tuple[int, int, float, float, Optional[float]]:
    """Extract truthful training facts from ``base_train`` JSON metadata."""
    user_config = meta_data.get("user_config") or {}
    loop_state = meta_data.get("loop_state") or {}
    total_batch = int(user_config.get("total_batch_size", 0))
    configured_steps = int(user_config.get("num_iterations", checkpoint_step))
    if configured_steps < 0:
        configured_steps = checkpoint_step
    requested = max(0, configured_steps) * total_batch
    processed = max(0, checkpoint_step) * total_batch
    walltime = float(loop_state.get("total_training_time", 0.0))
    throughput = processed / walltime if processed > 0 and walltime > 0 else 0.0
    smooth_loss = loop_state.get("smooth_train_loss")
    train_loss = float(smooth_loss) if smooth_loss is not None else None
    return requested, processed, walltime, throughput, train_loss

def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Strict JSON: non-finite floats (NaN/Inf from degraded rows) become null so
    # the artifact stays parseable by jq/strict parsers exactly when it is most
    # needed (debugging a degraded run). Mirrors eval_stats' CLI writer.
    def _strict(v: Any) -> Any:
        if isinstance(v, float) and not math.isfinite(v):
            return None
        if isinstance(v, dict):
            return {k: _strict(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_strict(x) for x in v]
        return v

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_strict(record), sort_keys=True, allow_nan=False) + "\n")


def _append_csv(path: Path, *, fieldnames: tuple[str, ...], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)


def _normalize_row_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in SUMMARY_FIELDS:
        v = row.get(k)
        if isinstance(v, (dict, list, tuple)):
            out[k] = json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False)
        else:
            out[k] = "" if v is None else v
    return out


def _write_summary(out_dir: Path, row: dict[str, Any]) -> None:
    csv_row = _normalize_row_for_csv(row)
    _append_csv(out_dir / "summary.csv", fieldnames=SUMMARY_FIELDS, row=csv_row)
    _write_jsonl(out_dir / "summary.jsonl", row)


def _publish_success_summary(
    out_dir: Path,
    row: dict[str, Any],
    registry_record: RunRecord,
    registry_path: str,
) -> None:
    """Publish one successful cell, with CSV as the final commit index.

    Statistical readers consume ``summary.csv``. Keeping that append last means
    registry validation or JSONL-mirror failures cannot expose unaudited success
    evidence, even though earlier diagnostic/audit artifacts may remain for repair.
    """
    csv_row = _normalize_row_for_csv(row)
    append_record(registry_record, registry_path)
    _write_jsonl(out_dir / "summary.jsonl", row)
    _append_csv(out_dir / "summary.csv", fieldnames=SUMMARY_FIELDS, row=csv_row)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _error_row(
    *,
    run_id: str,
    preset: str,
    seed: int,
    recipe_source: str,
    checkpoint_dir: str,
    checkpoint_step: int | None,
    data: str,
    device_type: str,
    world_size: int,
    init_type: str,
    sequence_len: int,
    vocab_size: int,
    n_layer: int,
    n_head: int,
    n_embd: int,
    use_moe: bool,
    num_experts: int,
    moe_top_k: int,
    device_batch_size: int,
    total_batch_size_tokens: int,
    train_tokens_requested: int,
    eval_tokens: int,
    eval_bpb: bool,
    core_eval: bool,
    core_max_per_task: int,
    ece_bins: int,
    dead_expert_threshold: float,
    continual_tasks: int,
    continual_exposures: int,
    error: str,
) -> dict[str, Any]:
    tokens_per_micro = device_batch_size * sequence_len
    grad_accum_steps = (
        total_batch_size_tokens // tokens_per_micro if tokens_per_micro > 0 else None
    )
    steps = (
        int(math.ceil(train_tokens_requested / total_batch_size_tokens))
        if total_batch_size_tokens > 0
        else None
    )
    eval_steps = (
        int(eval_tokens // tokens_per_micro) if tokens_per_micro > 0 else None
    )

    return {
        "status": "error",
        "error": error,
        "run_id": run_id,
        "run_dir": "",
        "preset": preset,
        "seed": seed,
        "recipe_source": recipe_source,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_step": checkpoint_step,
        "checkpoint_git_sha": None,
        "checkpoint_config_hash": None,
        "data": data,
        "device_type": device_type,
        "world_size": world_size,
        "init_type": init_type,
        "sequence_len": sequence_len,
        "vocab_size": vocab_size,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "use_moe": use_moe,
        "num_experts": num_experts,
        "moe_top_k": moe_top_k,
        "device_batch_size": device_batch_size,
        "total_batch_size_tokens": total_batch_size_tokens,
        "grad_accum_steps": grad_accum_steps,
        "train_tokens_requested": train_tokens_requested,
        "train_tokens_processed": 0,
        "steps": steps,
        "eval_tokens": eval_tokens,
        "eval_steps": eval_steps,
        "eval_bpb": eval_bpb,
        "core_eval": core_eval,
        "core_max_per_task": core_max_per_task,
        "ece_bins": ece_bins,
        "dead_expert_threshold": dead_expert_threshold,
        "continual_tasks": continual_tasks,
        "continual_exposures": continual_exposures,
        "walltime_sec": None,
        "tok_per_sec": None,
        "train_loss_final": None,
        "val_loss": None,
        "val_ppl": None,
        "val_bpb": None,
        "core_metric": None,
        "id_ece": None,
        "ood_auroc": None,
        "forgetting_rate": None,
        "moe_gini": None,
        "dead_expert_frac": None,
        "niah_acc": None,
        "recall_by_length": {},
        "forgetting_by_task": {},
        "capability_metric_status": {"run": "error"},
    }


def _resolve_niah_lengths(niah_lengths: str, max_len: int) -> tuple[int, ...]:
    """Resolve the NIAH context lengths for an eval run (v7c).

    A non-empty ``niah_lengths`` ("16,64,128") is parsed and each length kept only if it fits the
    model context (``8 <= L <= max_len``); an empty string defaults to ``(16, 64, max_len)``.
    Returns a de-duplicated, sorted, clamped tuple (possibly empty if nothing fits).
    """
    if niah_lengths.strip():
        requested = [int(x) for x in niah_lengths.split(",") if x.strip()]
    else:
        requested = [16, 64, max_len]
    kept = sorted({length for length in requested if 8 <= length <= max_len})
    return tuple(kept)


def _masked_token_accuracy(model: Any, batch: Any) -> float:
    logits = _get_logits(model, batch.inputs)
    valid = batch.targets >= 0
    if not bool(valid.any()):
        raise ValueError("continual task contains no supervised targets")
    predictions = logits.argmax(dim=-1)
    return float((predictions[valid] == batch.targets[valid]).float().mean().item())


def _forgetting_rate_from_accuracy_matrix(
    accuracy_matrix: Sequence[Sequence[Optional[float]]],
) -> tuple[float, dict[str, dict[str, float]]]:
    """Standard continual-learning forgetting over a lower-triangular accuracy matrix.

    Row ``t`` is measured after learning task ``t``. For every task except the last (which has no
    subsequent interference), forgetting is its best accuracy from acquisition through the final
    phase minus its final accuracy. Including the final phase in the maximum makes the rate
    non-negative while still crediting later positive transfer.
    """
    task_count = len(accuracy_matrix)
    if task_count < 2 or any(len(row) != task_count for row in accuracy_matrix):
        raise ValueError("forgetting requires a square accuracy matrix with at least two tasks")
    by_task: dict[str, dict[str, float]] = {}
    drops: list[float] = []
    for task in range(task_count - 1):
        observed = [accuracy_matrix[phase][task] for phase in range(task, task_count)]
        if any(value is None for value in observed):
            raise ValueError(f"task {task} is missing an accuracy after acquisition")
        values = [float(cast(float, value)) for value in observed]
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"task {task} accuracies must be finite fractions")
        peak = max(values)
        final = values[-1]
        drop = peak - final
        drops.append(drop)
        by_task[str(task)] = {"peak_accuracy": peak, "final_accuracy": final, "forgetting": drop}
    return sum(drops) / len(drops), by_task


def _continual_forgetting_metric(
    model: Any,
    *,
    sequence_len: int,
    vocab_size: int,
    task_count: int,
    exposures: int,
    seed: int,
    device: torch.device,
) -> ContinualMetricSummary:
    if task_count < 2:
        raise ValueError("continual_tasks must be >= 2")
    if exposures < 1:
        raise ValueError("continual_exposures must be >= 1")
    task_length = min(8, (vocab_size - 2) // task_count, (sequence_len - 1) // 2)
    if task_length < 1:
        return ContinualMetricSummary(None, {}, [], "not_applicable", "model context/vocabulary too small")

    from bio_inspired_nanochat.synthetic_tasks import continual_task_sequence

    tasks = [
        batch.to(device)
        for batch in continual_task_sequence(
            num_tasks=task_count,
            batch=8,
            length=task_length,
            vocab_size=vocab_size,
            seed=seed,
        )
    ]
    reset = getattr(model, "reset_sequence_state", None)
    if callable(reset):
        reset(reset_fast_weights=True, reset_consolidation=True)
    matrix: list[list[Optional[float]]] = [
        [None for _ in range(task_count)] for _ in range(task_count)
    ]
    try:
        with torch.no_grad():
            for phase, task in enumerate(tasks):
                for _ in range(exposures):
                    _forward_logits(model, task.inputs, train_mode=True)
                for learned_task in range(phase + 1):
                    matrix[phase][learned_task] = _masked_token_accuracy(model, tasks[learned_task])
        rate, by_task = _forgetting_rate_from_accuracy_matrix(matrix)
        return ContinualMetricSummary(rate, by_task, matrix, "ok", None)
    finally:
        if callable(reset):
            reset(reset_fast_weights=True, reset_consolidation=True)


def _run_one(
    *,
    preset: PresetId,
    train_tokens: int,
    seed: int,
    device_type: str,
    data: str,
    out_dir: Path,
    # model arch
    sequence_len: int,
    vocab_size: int,
    n_layer: int,
    n_head: int,
    n_embd: int,
    # optim / batch
    device_batch_size: int,
    total_batch_size_tokens: int,
    embedding_lr: float,
    unembedding_lr: float,
    matrix_lr: float,
    weight_decay: float,
    # eval
    eval_tokens: int,
    eval_bpb: bool,
    core_eval: bool,
    core_max_per_task: int,
    ece_bins: int,
    niah_lengths: str = "",
    dead_expert_threshold: float = 0.01,
    continual_tasks: int = 3,
    continual_exposures: int = 4,
    init_type: str,
    use_moe: bool,
    num_experts: int,
    moe_top_k: int,
    checkpoint_dir: str,
    checkpoint_step: int,
    registry_path: str,
    runtime: ComputeRuntime,
) -> HarnessRunSummary:
    ddp, ddp_rank, _, ddp_world_size, device = runtime
    g = _set_seed(seed, device_type=device_type)
    if ddp:
        # The model seed stays identical on every rank; only synthetic data is rank-partitioned.
        g.manual_seed(seed + ddp_rank)

    checkpoint_path: Path | None = None
    checkpoint_meta: dict[str, Any] = {}
    resolved_checkpoint_step: int | None = None
    recipe_source = "inline_smoke_noncanonical"
    if checkpoint_dir:
        checkpoint_path = _resolve_checkpoint_dir(checkpoint_dir, preset=preset, seed=seed)
        model, checkpoint_meta, resolved_checkpoint_step = _load_base_train_checkpoint(
            checkpoint_path,
            step=checkpoint_step,
            preset=preset,
            seed=seed,
            device=device,
        )
        recipe_source = "base_train_checkpoint"
        config = model.config
        sequence_len = int(config.sequence_len)
        vocab_size = int(config.vocab_size)
        n_layer = int(config.n_layer)
        n_head = int(config.n_head)
        n_embd = int(config.n_embd)
        init_type = str(getattr(config, "init_type", "baseline"))
        use_moe = bool(getattr(config, "use_moe", False))
        num_experts = int(getattr(config, "num_experts", 0))
        moe_top_k = int(getattr(config, "moe_top_k", 0))
    else:
        model = _build_model(
            preset=preset,
            seed=seed,
            device=device,
            sequence_len=sequence_len,
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            init_type=init_type,
            use_moe=use_moe,
            num_experts=num_experts,
            moe_top_k=moe_top_k,
        )
        model.train()
        if preset == "vanilla":
            use_moe = False
            num_experts = 0
            moe_top_k = 0

    stamp_payload = [_utc_stamp() if ddp_rank == 0 else ""]
    if ddp and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(stamp_payload, src=0, device=device)
    stamp = str(stamp_payload[0])
    source_tag = (
        f"ckpt{resolved_checkpoint_step}"
        if resolved_checkpoint_step is not None
        else f"t{train_tokens}"
    )
    run_id = f"Q-{preset}-{source_tag}-s{seed}-{stamp}"
    run_dir = out_dir / run_id
    if ddp_rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    if ddp and torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Data loaders
    if data == "synthetic":
        train_iter = _synthetic_loader(
            batch_size=device_batch_size,
            seq_len=sequence_len,
            vocab_size=vocab_size,
            device=device,
            generator=g,
        )
        val_iter = _synthetic_loader(
            batch_size=device_batch_size,
            seq_len=sequence_len,
            vocab_size=vocab_size,
            device=device,
            generator=g,
        )
        tokenizer = None
    elif data == "fineweb":
        train_iter = iter(
            tokenizing_distributed_data_loader(
                device_batch_size,
                sequence_len,
                split="train",
                device=device,
            )
        )
        val_iter = iter(
            tokenizing_distributed_data_loader(
                device_batch_size,
                sequence_len,
                split="val",
                device=device,
            )
        )
        tokenizer = get_tokenizer()
        if tokenizer.get_vocab_size() != vocab_size:
            raise ValueError(
                f"tokenizer vocab {tokenizer.get_vocab_size()} does not match model vocab "
                f"{vocab_size}"
            )
    else:
        raise ValueError(f"Unknown data source {data!r} (expected 'synthetic' or 'fineweb')")

    # Batch math
    world_tokens_per_micro = device_batch_size * sequence_len * ddp_world_size
    if checkpoint_path is None:
        if total_batch_size_tokens % world_tokens_per_micro != 0:
            raise ValueError(
                f"total_batch_size_tokens={total_batch_size_tokens} must be divisible by "
                "device_batch_size*sequence_len*world_size="
                f"{world_tokens_per_micro}"
            )
        grad_accum_steps: int | None = total_batch_size_tokens // world_tokens_per_micro
        steps = max(1, int(math.ceil(train_tokens / total_batch_size_tokens)))
        tokens_requested = train_tokens
        optimizers = model.setup_optimizers(
            unembedding_lr=unembedding_lr,
            embedding_lr=embedding_lr,
            matrix_lr=matrix_lr,
            weight_decay=weight_decay,
        )
    else:
        if resolved_checkpoint_step is None:
            raise RuntimeError("checkpoint loader did not resolve a checkpoint step")
        tokens_requested, tokens_processed, walltime_sec, tok_per_sec_avg, train_loss_final = (
            _checkpoint_recipe_stats(checkpoint_meta, resolved_checkpoint_step)
        )
        steps = resolved_checkpoint_step
        grad_accum_steps = None
        optimizers = []
    checkpoint_user_config = checkpoint_meta.get("user_config") or {}
    recorded_device_batch_size = int(
        checkpoint_meta.get("device_batch_size", device_batch_size)
        if checkpoint_path is not None
        else device_batch_size
    )
    recorded_total_batch_size = int(
        checkpoint_user_config.get("total_batch_size", total_batch_size_tokens)
        if checkpoint_path is not None
        else total_batch_size_tokens
    )

    run_config = {
        "preset": preset,
        "seed": seed,
        "recipe_source": recipe_source,
        "checkpoint_dir": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_step": resolved_checkpoint_step,
        "checkpoint_metadata": checkpoint_meta or None,
        "data": data,
        "world_size": ddp_world_size,
        "train_tokens_requested": tokens_requested,
        "eval_tokens": eval_tokens,
        "eval_bpb": eval_bpb,
        "core_eval": core_eval,
        "core_max_per_task": core_max_per_task,
        "ece_bins": ece_bins,
        "niah_lengths": niah_lengths,
        "dead_expert_threshold": dead_expert_threshold,
        "continual_tasks": continual_tasks,
        "continual_exposures": continual_exposures,
        "sequence_len": sequence_len,
        "vocab_size": vocab_size,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "device_batch_size": recorded_device_batch_size,
        "eval_device_batch_size": device_batch_size,
        "total_batch_size_tokens": recorded_total_batch_size,
        "grad_accum_steps": grad_accum_steps,
        "steps": steps,
        "embedding_lr": embedding_lr,
        "unembedding_lr": unembedding_lr,
        "matrix_lr": matrix_lr,
        "weight_decay": weight_decay,
        "init_type": init_type,
        "use_moe": use_moe,
        "num_experts": num_experts,
        "moe_top_k": moe_top_k,
    }
    registry_config = {
        key: value for key, value in run_config.items() if key != "checkpoint_metadata"
    }
    registry_config["checkpoint_provenance"] = (
        checkpoint_meta.get("provenance") if checkpoint_meta else None
    )

    # Write config snapshot
    if ddp_rank == 0:
        _write_jsonl(
            run_dir / "run_config.jsonl",
            {"run_id": run_id, **run_config},
        )

    # The historical inline loop is retained only for explicit CI/smoke use. Scientific rows load
    # the completed base_train model above and never retrain it under a divergent recipe.
    if checkpoint_path is None:
        if grad_accum_steps is None:
            raise RuntimeError("inline training did not resolve gradient accumulation")
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}[/bold]"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            disable=ddp_rank != 0,
        )
        train_task = progress.add_task(f"train {run_id}", total=steps)
        losses: list[float] = []
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_start = time.perf_counter()
        supports_train_mode = "train_mode" in inspect.signature(model.forward).parameters
        with progress:
            for train_step in range(steps):
                t0 = time.perf_counter()
                for _ in range(grad_accum_steps):
                    x, y = next(train_iter)
                    result = model(x, y, train_mode=True) if supports_train_mode else model(x, y)
                    if isinstance(result, tuple):
                        _, loss = result
                    else:
                        loss = result
                    if loss is None:
                        raise RuntimeError("Model returned loss=None during training")
                    (loss / grad_accum_steps).backward()

                for opt in optimizers:
                    opt.step()
                model.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                dt = time.perf_counter() - t0
                tok_per_sec = total_batch_size_tokens / max(dt, 1e-12)
                loss_f = float(loss.detach().float().item())
                losses.append(loss_f)

                if ddp_rank == 0:
                    _write_jsonl(
                        run_dir / "train_metrics.jsonl",
                        {
                            "step": train_step,
                            "loss": loss_f,
                            "dt_sec": dt,
                            "tok_per_sec": tok_per_sec,
                        },
                    )
                progress.update(train_task, advance=1)

        walltime_sec = time.perf_counter() - t_start
        if ddp and torch.distributed.is_initialized():
            walltime_tensor = torch.tensor(walltime_sec, dtype=torch.float64, device=device)
            torch.distributed.all_reduce(walltime_tensor, op=torch.distributed.ReduceOp.MAX)
            walltime_sec = float(walltime_tensor.item())
            final_loss_tensor = torch.tensor(losses[-1], dtype=torch.float64, device=device)
            torch.distributed.all_reduce(final_loss_tensor, op=torch.distributed.ReduceOp.AVG)
            losses[-1] = float(final_loss_tensor.item())
        tokens_processed = steps * total_batch_size_tokens
        tok_per_sec_avg = tokens_processed / max(walltime_sec, 1e-12)
        train_loss_final = float(losses[-1]) if losses else None
        model.train(False)

    # Evaluation
    eval_steps = max(1, int(eval_tokens // world_tokens_per_micro))
    val_loss, val_ppl, id_ece, ood_auroc, routing = _val_loss_ppl_ece(
        model,
        val_iter,
        steps=eval_steps,
        device_type=device_type,
        ddp=ddp,
        ece_bins=ece_bins,
        ood_seed=seed + 74_007,
        dead_expert_threshold=dead_expert_threshold,
    )

    val_bpb: Optional[float] = None
    if eval_bpb:
        if tokenizer is None:
            val_bpb = None
        else:
            token_bytes = get_token_bytes(device=device)
            if "loss_reduction" in inspect.signature(model.forward).parameters:
                val_bpb = float(evaluate_bpb(model, val_iter, eval_steps, token_bytes))
            else:
                orig_forward = model.forward

                def _syn_forward_wrapper(
                    idx: Tensor,
                    targets: Optional[Tensor] = None,
                    kv_cache=None,
                    loss_reduction: str = "mean",
                    **kwargs: Any,
                ):
                    if targets is None:
                        logits, _ = orig_forward(idx, None, kv_cache, train_mode=False)
                        return logits
                    logits, loss = orig_forward(idx, targets, kv_cache, train_mode=False)
                    if loss_reduction == "none":
                        logits_flat = logits.reshape(-1, logits.size(-1))
                        targets_flat = targets.reshape(-1)
                        loss_per_token = F.cross_entropy(
                            logits_flat,
                            targets_flat,
                            reduction="none",
                            ignore_index=-1,
                        )
                        return loss_per_token.reshape(targets.shape)
                    return loss

                model.forward = _syn_forward_wrapper
                try:
                    val_bpb = float(evaluate_bpb(model, val_iter, eval_steps, token_bytes))
                finally:
                    model.forward = orig_forward

    core_metric: Optional[float] = None
    if core_eval and ddp_rank == 0:
        if tokenizer is None:
            core_metric = None
        else:
            with _force_synaptic_eval_forward(model) as eval_model:
                out = evaluate_model(
                    eval_model, tokenizer, device, max_per_task=core_max_per_task
                )
            core_metric = float(out["core_metric"])

    # Needle-in-a-haystack long-context retrieval accuracy (74f.2): the key probe of
    # the fast-weight / long-context claim. Swept over length × needle depth.
    niah_acc: Optional[float] = None
    recall_by_length: dict[str, float] = {}
    memory_status = "not_applicable"
    memory_reason: Optional[str] = "no supported context lengths"
    try:
        from bio_inspired_nanochat.synthetic_tasks import niah_accuracy_by_length

        context_max = int(sequence_len) - 2
        max_len = context_max if niah_lengths.strip() else min(context_max, 256)
        lengths_used = _resolve_niah_lengths(niah_lengths, max_len)
        if lengths_used and ddp_rank == 0:
            with _force_synaptic_eval_forward(model) as eval_model:
                niah_result = niah_accuracy_by_length(
                    eval_model,
                    vocab_size=min(64, int(vocab_size)),
                    lengths=lengths_used,
                    batch=32,
                    seed=int(seed),
                    device=device,
                )
            niah_acc = float(niah_result["overall"])
            recall_by_length = {
                str(length): float(accuracy)
                for length, accuracy in niah_result["by_length"].items()
            }
            memory_status = "ok"
            memory_reason = None
    except Exception as e:  # eval is best-effort; never fail a run on the probe
        if ddp_rank == 0:
            console.print(f"[yellow][niah] eval skipped:[/yellow] {e}")
            memory_status = "error"
            memory_reason = repr(e)

    continual = ContinualMetricSummary(None, {}, [], "not_run", "non-zero DDP rank")
    if ddp_rank == 0:
        try:
            continual = _continual_forgetting_metric(
                model,
                sequence_len=sequence_len,
                vocab_size=vocab_size,
                task_count=continual_tasks,
                exposures=continual_exposures,
                seed=seed + 74_700,
                device=device,
            )
        except Exception as e:
            continual = ContinualMetricSummary(None, {}, [], "error", repr(e))
            console.print(f"[yellow][continual] eval skipped:[/yellow] {e}")

    summary = HarnessRunSummary(
        run_id=run_id,
        preset=preset,
        seed=seed,
        train_tokens_requested=tokens_requested,
        train_tokens_processed=tokens_processed,
        walltime_sec=walltime_sec,
        tok_per_sec=tok_per_sec_avg,
        train_loss_final=train_loss_final,
        val_loss=val_loss,
        val_ppl=val_ppl,
        val_bpb=val_bpb,
        core_metric=core_metric,
        id_ece=id_ece,
        ood_auroc=ood_auroc,
        forgetting_rate=continual.forgetting_rate,
        moe_gini=routing.moe_gini,
        dead_expert_frac=routing.dead_expert_frac,
        niah_acc=niah_acc,
        recall_by_length=recall_by_length,
        forgetting_by_task=continual.by_task,
    )

    capability_metric_status = {
        "uncertainty": "ok",
        "continual": continual.status,
        "routing": "ok" if routing.layers else "not_applicable",
        "memory": memory_status,
    }

    # Persist summary
    row: dict[str, Any] = {
        "status": "ok",
        "error": "",
        "run_id": summary.run_id,
        "run_dir": str(run_dir),
        "preset": summary.preset,
        "seed": summary.seed,
        "recipe_source": recipe_source,
        "checkpoint_dir": str(checkpoint_path) if checkpoint_path is not None else "",
        "checkpoint_step": resolved_checkpoint_step,
        "checkpoint_git_sha": (checkpoint_meta.get("provenance") or {}).get("git_sha"),
        "checkpoint_config_hash": (checkpoint_meta.get("provenance") or {}).get(
            "synaptic_config_hash"
        ),
        "data": data,
        "device_type": device_type,
        "world_size": ddp_world_size,
        "init_type": init_type,
        "sequence_len": sequence_len,
        "vocab_size": vocab_size,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "use_moe": use_moe,
        "num_experts": num_experts,
        "moe_top_k": moe_top_k,
        "device_batch_size": recorded_device_batch_size,
        "total_batch_size_tokens": recorded_total_batch_size,
        "grad_accum_steps": grad_accum_steps,
        "train_tokens_requested": tokens_requested,
        "train_tokens_processed": tokens_processed,
        "steps": steps,
        "eval_tokens": eval_tokens,
        "eval_steps": eval_steps,
        "eval_bpb": eval_bpb,
        "core_eval": core_eval,
        "core_max_per_task": core_max_per_task,
        "ece_bins": ece_bins,
        "dead_expert_threshold": dead_expert_threshold,
        "continual_tasks": continual_tasks,
        "continual_exposures": continual_exposures,
        "walltime_sec": summary.walltime_sec,
        "tok_per_sec": summary.tok_per_sec,
        "train_loss_final": summary.train_loss_final,
        "val_loss": summary.val_loss,
        "val_ppl": summary.val_ppl,
        "val_bpb": summary.val_bpb,
        "core_metric": summary.core_metric,
        "id_ece": summary.id_ece,
        "ood_auroc": summary.ood_auroc,
        "forgetting_rate": summary.forgetting_rate,
        "moe_gini": summary.moe_gini,
        "dead_expert_frac": summary.dead_expert_frac,
        "niah_acc": summary.niah_acc,
        "recall_by_length": summary.recall_by_length,
        "forgetting_by_task": summary.forgetting_by_task,
        "capability_metric_status": capability_metric_status,
    }
    if ddp_rank == 0:
        registry_metrics = {"tok_per_sec": float(summary.tok_per_sec)}
        if summary.train_loss_final is not None and math.isfinite(summary.train_loss_final):
            registry_metrics["train_loss"] = float(summary.train_loss_final)
        if summary.val_bpb is not None and math.isfinite(summary.val_bpb):
            registry_metrics["eval_bpb"] = float(summary.val_bpb)
        if summary.niah_acc is not None and math.isfinite(summary.niah_acc):
            registry_metrics["niah_accuracy"] = float(summary.niah_acc)
        for metric_name, value in (
            ("id_ece", summary.id_ece),
            ("ood_auroc", summary.ood_auroc),
            ("forgetting_rate", summary.forgetting_rate),
            ("moe_gini", summary.moe_gini),
            ("dead_expert_frac", summary.dead_expert_frac),
        ):
            if value is not None and math.isfinite(value):
                registry_metrics[metric_name] = float(value)
        registry_record = make_record(
            "eval",
            registry_metrics,
            run_id=run_id,
            config=registry_config,
            seed=seed,
            dataset_shards=[f"{data}:val"],
            timestamp=time.time(),
            notes=f"artifact_dir={run_dir}; preset={preset}; recipe={recipe_source}",
        )
        capability_log = run_dir / "capability_metrics.jsonl"
        _write_jsonl(
            capability_log,
            {
                "run_id": run_id,
                "capability": "uncertainty",
                "status": "ok",
                "id_ece": summary.id_ece,
                "ood_auroc": summary.ood_auroc,
                "ece_bins": ece_bins,
                "ood_protocol": "deterministic_token_position_hash",
                "ood_score": "mean_predictive_entropy_per_sequence",
            },
        )
        _write_jsonl(
            capability_log,
            {
                "run_id": run_id,
                "capability": "routing",
                "status": capability_metric_status["routing"],
                "moe_gini": summary.moe_gini,
                "dead_expert_frac": summary.dead_expert_frac,
                "dead_expert_threshold": dead_expert_threshold,
                "layers": routing.layers,
            },
        )
        _write_jsonl(
            capability_log,
            {
                "run_id": run_id,
                "capability": "memory",
                "status": memory_status,
                "reason": memory_reason,
                "niah_acc": summary.niah_acc,
                "recall_by_length": summary.recall_by_length,
            },
        )
        _write_jsonl(
            capability_log,
            {
                "run_id": run_id,
                "capability": "continual",
                "status": continual.status,
                "reason": continual.reason,
                "forgetting_rate": continual.forgetting_rate,
                "forgetting_by_task": continual.by_task,
                "accuracy_matrix": continual.accuracy_matrix,
                "task_count": continual_tasks,
                "exposures_per_task": continual_exposures,
            },
        )
        _publish_success_summary(out_dir, row, registry_record, registry_path)

    # Pretty print
    if ddp_rank == 0:
        tbl = Table(title=f"Eval Matrix Summary: {run_id}")
        tbl.add_column("key")
        tbl.add_column("value", justify="right")
        for k, v in row.items():
            tbl.add_row(k, "" if v is None else str(v))
        console.print(tbl)

    return summary


def _cmd_run(args: argparse.Namespace) -> int:
    runtime = compute_init(args.device_type)
    try:
        _run_one(
            preset=cast(PresetId, args.preset),
            train_tokens=args.train_tokens,
            seed=args.seed,
            device_type=args.device_type,
            data=args.data,
            out_dir=Path(args.out_dir),
            sequence_len=args.sequence_len,
            vocab_size=args.vocab_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            device_batch_size=args.device_batch_size,
            total_batch_size_tokens=args.total_batch_size_tokens,
            embedding_lr=args.embedding_lr,
            unembedding_lr=args.unembedding_lr,
            matrix_lr=args.matrix_lr,
            weight_decay=args.weight_decay,
            eval_tokens=args.eval_tokens,
            eval_bpb=args.eval_bpb,
            core_eval=args.core_eval,
            core_max_per_task=args.core_max_per_task,
            ece_bins=args.ece_bins,
            niah_lengths=args.niah_lengths,
            dead_expert_threshold=args.dead_expert_threshold,
            continual_tasks=args.continual_tasks,
            continual_exposures=args.continual_exposures,
            init_type=args.init_type,
            use_moe=args.use_moe,
            num_experts=args.num_experts,
            moe_top_k=args.moe_top_k,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_step=args.checkpoint_step,
            registry_path=args.registry_path,
            runtime=runtime,
        )
    finally:
        compute_cleanup()
    return 0


def _run_batch(
    *,
    batch_kind: str,
    presets: list[PresetId],
    seeds: list[int],
    args: argparse.Namespace,
) -> int:
    explicit_out_dir = (
        _batch_output_dir(Path(args.out_dir), args.batch_id)
        if args.batch_id is not None
        else None
    )
    runtime = compute_init(args.device_type)
    ddp, ddp_rank, _, _, device = runtime
    stamp_payload = [_utc_stamp() if ddp_rank == 0 else ""]
    if ddp and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(stamp_payload, src=0, device=device)
    batch_id = args.batch_id or f"{batch_kind}_{stamp_payload[0]}"
    batch_out_dir = (
        explicit_out_dir
        if explicit_out_dir is not None
        else _batch_output_dir(Path(args.out_dir), batch_id)
    )
    if ddp_rank == 0:
        batch_out_dir.mkdir(parents=True, exist_ok=True)
    if ddp and torch.distributed.is_initialized():
        torch.distributed.barrier()

    if ddp_rank == 0:
        tbl = Table(title=f"Eval Matrix Batch: {batch_id}")
        tbl.add_column("preset")
        tbl.add_column("seed", justify="right")
        for preset in presets:
            for seed in seeds:
                tbl.add_row(preset, str(seed))
        console.print(tbl)

    had_failure = False
    try:
        for preset in presets:
            for seed in seeds:
                try:
                    _run_one(
                        preset=preset,
                        train_tokens=args.train_tokens,
                        seed=seed,
                        device_type=args.device_type,
                        data=args.data,
                        out_dir=batch_out_dir,
                        sequence_len=args.sequence_len,
                        vocab_size=args.vocab_size,
                        n_layer=args.n_layer,
                        n_head=args.n_head,
                        n_embd=args.n_embd,
                        device_batch_size=args.device_batch_size,
                        total_batch_size_tokens=args.total_batch_size_tokens,
                        embedding_lr=args.embedding_lr,
                        unembedding_lr=args.unembedding_lr,
                        matrix_lr=args.matrix_lr,
                        weight_decay=args.weight_decay,
                        eval_tokens=args.eval_tokens,
                        eval_bpb=args.eval_bpb,
                        core_eval=args.core_eval,
                        core_max_per_task=args.core_max_per_task,
                        ece_bins=args.ece_bins,
                        niah_lengths=args.niah_lengths,
                        dead_expert_threshold=args.dead_expert_threshold,
                        continual_tasks=args.continual_tasks,
                        continual_exposures=args.continual_exposures,
                        init_type=args.init_type,
                        use_moe=args.use_moe,
                        num_experts=args.num_experts,
                        moe_top_k=args.moe_top_k,
                        checkpoint_dir=args.checkpoint_dir,
                        checkpoint_step=args.checkpoint_step,
                        registry_path=args.registry_path,
                        runtime=runtime,
                    )
                except Exception as e:
                    had_failure = True
                    err_id = f"ERR-{preset}-t{args.train_tokens}-s{seed}-{_utc_stamp()}"
                    resolved_checkpoint = (
                        str(_resolve_checkpoint_dir(args.checkpoint_dir, preset=preset, seed=seed))
                        if args.checkpoint_dir
                        else ""
                    )
                    row = _error_row(
                        run_id=err_id,
                        preset=preset,
                        seed=seed,
                        recipe_source=(
                            "base_train_checkpoint"
                            if args.checkpoint_dir
                            else "inline_smoke_noncanonical"
                        ),
                        checkpoint_dir=resolved_checkpoint,
                        checkpoint_step=(args.checkpoint_step if args.checkpoint_step >= 0 else None),
                        data=args.data,
                        device_type=args.device_type,
                        world_size=runtime[3],
                        init_type=args.init_type,
                        sequence_len=args.sequence_len,
                        vocab_size=args.vocab_size,
                        n_layer=args.n_layer,
                        n_head=args.n_head,
                        n_embd=args.n_embd,
                        use_moe=args.use_moe,
                        num_experts=args.num_experts,
                        moe_top_k=args.moe_top_k,
                        device_batch_size=args.device_batch_size,
                        total_batch_size_tokens=args.total_batch_size_tokens,
                        train_tokens_requested=args.train_tokens,
                        eval_tokens=args.eval_tokens,
                        eval_bpb=args.eval_bpb,
                        core_eval=args.core_eval,
                        core_max_per_task=args.core_max_per_task,
                        ece_bins=args.ece_bins,
                        dead_expert_threshold=args.dead_expert_threshold,
                        continual_tasks=args.continual_tasks,
                        continual_exposures=args.continual_exposures,
                        error=repr(e),
                    )
                    if ddp_rank == 0:
                        _write_summary(batch_out_dir, row)
                        console.print(
                            f"[bold red]Run failed:[/bold red] preset={preset} "
                            f"seed={seed} error={e!r}"
                        )
                    if args.fail_fast:
                        raise
        if ddp and torch.distributed.is_initialized():
            failure_flag = torch.tensor(
                int(had_failure), dtype=torch.int32, device=device
            )
            torch.distributed.all_reduce(failure_flag, op=torch.distributed.ReduceOp.SUM)
            had_failure = int(failure_flag.item()) > 0
    finally:
        compute_cleanup()

    if ddp_rank == 0:
        console.print(f"Batch outputs: {batch_out_dir}")
    return int(had_failure)


def _cmd_matrix(args: argparse.Namespace) -> int:
    allowed = set(PresetId.__args__)
    presets_raw = _parse_str_list(args.presets)
    presets: list[PresetId] = []
    for preset in presets_raw:
        if preset not in allowed:
            raise ValueError(f"Unknown preset {preset!r}. Allowed: {sorted(allowed)}")
        presets.append(cast(PresetId, preset))
    seeds = _parse_int_list(args.seeds)
    return _run_batch(batch_kind="matrix", presets=presets, seeds=seeds, args=args)


def _cmd_ablation(args: argparse.Namespace) -> int:
    seeds = _parse_int_list(args.seeds)
    return _run_batch(
        batch_kind="ablation",
        presets=list(DEFAULT_ABLATION_PRESETS),
        seeds=seeds,
        args=args,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Bio vs vanilla eval matrix harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--device-type", default="", help="cuda|cpu|mps (default: autodetect)")
        p.add_argument("--data", default="fineweb", choices=["fineweb", "synthetic"])
        p.add_argument("--out-dir", default="runs/eval_matrix")
        p.add_argument(
            "--registry-path",
            default=DEFAULT_REGISTRY,
            help="Committed JSONL results registry path",
        )
        p.add_argument(
            "--checkpoint-dir",
            default="",
            help=(
                "base_train checkpoint directory; matrix runs may use {preset} and {seed} "
                "template fields"
            ),
        )
        p.add_argument(
            "--checkpoint-step",
            type=int,
            default=-1,
            help="base_train checkpoint step (-1 selects the latest complete model checkpoint)",
        )
        p.add_argument(
            "--inline-smoke-training",
            action="store_true",
            help="explicitly allow the noncanonical tiny inline loop for CI/smoke only",
        )
        p.add_argument(
            "--train-tokens",
            type=int,
            default=10_000_000,
            help="token budget for --inline-smoke-training (ignored for checkpoint evaluation)",
        )
        p.add_argument("--eval-tokens", type=int, default=1_000_000)
        p.add_argument("--eval-bpb", action="store_true", help="Also compute val bpb (requires tokenizer artifacts)")
        p.add_argument("--core-eval", action="store_true", help="Also compute CORE metric (requires eval bundle)")
        p.add_argument("--core-max-per-task", type=int, default=200)
        p.add_argument("--ece-bins", type=int, default=15)
        p.add_argument(
            "--dead-expert-threshold",
            type=float,
            default=0.01,
            help="Routing-share floor below which an MoE expert is classified as dead",
        )
        p.add_argument(
            "--continual-tasks",
            type=int,
            default=3,
            help="Number of disjoint online-copy tasks in the forgetting probe (>=2)",
        )
        p.add_argument(
            "--continual-exposures",
            type=int,
            default=4,
            help="State-updating exposures per task in the forgetting probe (>=1)",
        )
        p.add_argument(
            "--niah-lengths", default="",
            help="Comma-separated NIAH context lengths, e.g. '16,64,4096' (default: 16,64,min(model max,256)); "
            "explicit values are clamped only to the model context. Use fixed --seed for reproducibility.",
        )
        p.add_argument("--batch-id", default=None, help="Optional subdirectory name under --out-dir")
        p.add_argument("--fail-fast", action="store_true", help="Stop the batch on the first failure")

        # model arch
        p.add_argument("--sequence-len", type=int, default=2048)
        p.add_argument("--vocab-size", type=int, default=50304)
        p.add_argument("--n-layer", type=int, default=12)
        p.add_argument("--n-head", type=int, default=12)
        p.add_argument("--n-embd", type=int, default=768)

        # init
        p.add_argument("--init-type", default="baseline", choices=["baseline", "ca_rule30", "ca_rule116"])

        # bio/moe
        p.add_argument("--use-moe", action="store_true")
        p.add_argument("--num-experts", type=int, default=8)
        p.add_argument("--moe-top-k", type=int, default=2)

        # batch / opt
        p.add_argument("--device-batch-size", type=int, default=8)
        p.add_argument("--total-batch-size-tokens", type=int, default=131072)
        p.add_argument("--embedding-lr", type=float, default=0.2)
        p.add_argument("--unembedding-lr", type=float, default=0.004)
        p.add_argument("--matrix-lr", type=float, default=0.02)
        p.add_argument("--weight-decay", type=float, default=0.0)

    p_run = sub.add_parser("run", help="Run a single preset/seed")
    add_common(p_run)
    p_run.add_argument("--preset", required=True, choices=list(PresetId.__args__))
    p_run.add_argument("--seed", type=int, default=1337)
    p_run.set_defaults(func=_cmd_run)

    p_matrix = sub.add_parser("matrix", help="Run presets × seeds")
    add_common(p_matrix)
    p_matrix.add_argument("--presets", required=True, help="Comma-separated presets")
    p_matrix.add_argument("--seeds", required=True, help="Comma-separated seeds")
    p_matrix.set_defaults(func=_cmd_matrix)

    p_ablation = sub.add_parser("ablation", help="Run the standard feature-ablation sweep")
    add_common(p_ablation)
    p_ablation.add_argument("--seeds", required=True, help="Comma-separated seeds")
    p_ablation.set_defaults(func=_cmd_ablation)

    args = parser.parse_args()
    if args.device_type == "":
        args.device_type = autodetect_device_type()
    if args.ece_bins < 2:
        parser.error("--ece-bins must be >= 2")
    if not 0.0 <= args.dead_expert_threshold < 1.0:
        parser.error("--dead-expert-threshold must be in [0, 1)")
    if args.continual_tasks < 2:
        parser.error("--continual-tasks must be >= 2")
    if args.continual_exposures < 1:
        parser.error("--continual-exposures must be >= 1")
    if not args.checkpoint_dir and not args.inline_smoke_training:
        parser.error(
            "scientific evaluation requires --checkpoint-dir from base_train; "
            "use --inline-smoke-training only for CI plumbing smoke tests"
        )
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
