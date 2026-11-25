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
from ast import literal_eval
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from bio_inspired_nanochat.ablation_matrix import (
    AblationConfig,
    d1_screening_columns,
    exploratory_columns,
    screening_columns,
    structural_columns,
)
from bio_inspired_nanochat.ablation_registry import MECHANISMS, apply_preset
from bio_inspired_nanochat.checkpoint_manager import (
    config_hash,
    list_checkpoint_steps,
    load_checkpoint_model,
    synaptic_config_from_meta,
    synaptic_config_to_meta,
)
from bio_inspired_nanochat.common import (
    autodetect_device_type,
    compute_cleanup,
    compute_init,
)
from bio_inspired_nanochat.dataloader import tokenizing_distributed_data_loader
from bio_inspired_nanochat.gpt import GPT, GPTConfig
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.loss_eval import evaluate_bpb
from bio_inspired_nanochat.matrix_manifest import dataset_identity, digest, load_manifest, score_budget, seal_scores, source_identity
from bio_inspired_nanochat.results_registry import (
    DEFAULT_REGISTRY,
    RunRecord,
    append_record,
    make_record,
)
from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticMoE
from bio_inspired_nanochat.tokenizer import get_token_bytes, get_tokenizer
from bio_inspired_nanochat.torch_imports import F, Tensor, torch
from scripts.base_eval import evaluate_model

console = Console()

# The dense leave-one-out presets. The MoE-only mechanisms' ablations (metabolism, genome) are MoE
# stage columns (moe_no_*), materialised through MATRIX_COLUMNS like every non-preset column.
PresetId = Literal[
    "vanilla",
    "bio_all",
    "bio_no_presyn",
    "bio_no_stochastic_release",
    "bio_no_doc2",
    "bio_no_septin_barrier",
]

DEFAULT_ABLATION_PRESETS: tuple[PresetId, ...] = (
    "vanilla",
    "bio_all",
    "bio_no_presyn",
    "bio_no_stochastic_release",
    "bio_no_doc2",
    "bio_no_septin_barrier",
)

# Matrix columns outside the named registry presets include the synaptic_off anchor,
# add-one-in columns, and explicitly selected exploratory columns. They are materialised through
# ablation_matrix.AblationConfig.build_syn_cfg(), so this runner cannot drift from the spec.
ALL_MATRIX_COLUMNS: dict[str, AblationConfig] = {
    column.config_id: column
    for column in screening_columns() + d1_screening_columns() + structural_columns() + exploratory_columns()
}
MATRIX_COLUMNS = {k: v for k, v in ALL_MATRIX_COLUMNS.items() if k not in set(PresetId.__args__)}

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
    "matrix_stage",
    "protocol_id",
    "matrix_manifest_digest",
    "matrix_source_sha256",
    "model_file_sha256",
    "trained_cell_id",
    "train_data_digest",
    "eval_data_digest",
    "tokenizer_digest",
    "screen_summary_sha256",
    "eval_device_batch_size",
    "eval_tokens_processed",
    "train_device_type",
    "train_world_size",
    "optimizer_recipe_digest",
    "optimizer_names",
    "trainable_params",
    "active_params",
    "parameter_match_rule",
)


@dataclass(frozen=True)
class HarnessRunSummary:
    run_id: str
    preset: str
    seed: int
    train_tokens_requested: int
    train_tokens_processed: int
    walltime_sec: float | None
    tok_per_sec: float | None
    train_loss_final: float | None
    val_loss: float | None
    val_ppl: float | None
    val_bpb: float | None
    core_metric: float | None
    id_ece: float | None
    ood_auroc: float | None
    forgetting_rate: float | None
    moe_gini: float | None
    dead_expert_frac: float | None
    niah_acc: float | None
    recall_by_length: dict[str, float]
    forgetting_by_task: dict[str, dict[str, float]]


@dataclass(frozen=True)
class RoutingMetricSummary:
    moe_gini: float | None
    dead_expert_frac: float | None
    layers: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ContinualMetricSummary:
    forgetting_rate: float | None
    by_task: dict[str, dict[str, float]]
    accuracy_matrix: list[list[float | None]]
    status: str
    reason: str | None


ComputeRuntime = tuple[bool, int, int, int, torch.device]


def _require_unique[ListEntry: (int, str)](values: Sequence[ListEntry], *, name: str) -> None:
    seen: set[ListEntry] = set()
    duplicates: list[ListEntry] = []
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
    gathered: list[list[float] | None] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, scores)
    return [score for rank_scores in gathered if rank_scores is not None for score in rank_scores]


@contextmanager
def _force_synaptic_eval_forward(model: Any):
    """Make generic evaluators call a synaptic model with ``train_mode=False``."""
    if "train_mode" not in inspect.signature(model.forward).parameters:
        yield model
        return
    original_forward = model.forward

    def eval_forward(idx: Tensor, targets: Tensor | None = None, kv_cache=None, **kwargs: Any):
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
) -> tuple[float, float, float | None, float, RoutingMetricSummary]:
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

            # Convert logits to log-probabilities in-place after cross_entropy so
            # large-vocab checkpoints (e.g. B=16, T=2048, V=65536 => 8 GiB per fp32
            # tensor) never hold logits + probs + id_log_probs + id_probs + ood_logits
            # simultaneously in VRAM.
            logits.sub_(torch.logsumexp(logits, dim=-1, keepdim=True))
            max_log_prob, pred = logits_flat.max(dim=-1)
            conf = max_log_prob.exp()
            correct = (pred == targets_flat) & valid
            # Bin by confidence in [0,1]
            bins = torch.clamp((conf * ece_bins).to(torch.int64), 0, ece_bins - 1)
            for b in range(ece_bins):
                mask = (bins == b) & valid
                if mask.any():
                    conf_sum[b] += float(conf[mask].sum().item())
                    acc_sum[b] += float(correct[mask].float().sum().item())
                    count[b] += float(mask.sum().item())

            vocab_dim = logits.size(-1)
            for row_lp in logits:
                id_uncertainty.append(
                    float((-(row_lp.exp() * row_lp).sum(dim=-1).mean(dim=-1)).item())
                )
            del logits, logits_flat

            ood_x = _deterministic_ood_tokens(
                x,
                vocab_size=vocab_dim,
                seed=ood_seed,
            )
            ood_logits = _get_logits(model, ood_x).to(torch.float32)
            ood_logits.sub_(torch.logsumexp(ood_logits, dim=-1, keepdim=True))
            for row_lp in ood_logits:
                ood_uncertainty.append(
                    float((-(row_lp.exp() * row_lp).sum(dim=-1).mean(dim=-1)).item())
                )
            del ood_logits

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

    ece: float | None
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


def _syn_cfg_for_preset(preset: str) -> SynapticConfig:
    """The SynapticConfig for a named registry preset or a pre-registered matrix column."""
    column = MATRIX_COLUMNS.get(preset)
    if column is not None:
        built = column.build_syn_cfg()
        if built is None:
            raise ValueError(f"matrix column {preset!r} has no synaptic configuration")
        return built
    syn_cfg = SynapticConfig()
    _apply_syn_preset(preset, syn_cfg)
    return syn_cfg


def _build_model(
    *,
    preset: str,  # a PresetId or a MATRIX_COLUMNS id
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

    syn_cfg = _syn_cfg_for_preset(preset)
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
    manifest: dict[str, Any] | None = None,
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

    model, meta_data = load_checkpoint_model(str(checkpoint_dir), resolved_step, device, "eval")
    model_config = meta_data.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("base_train checkpoint metadata must contain a model_config object")

    training = meta_data.get("user_config") or {}
    if checkpoint_dir.name.startswith("matrix_") and training.get("model_tag") != checkpoint_dir.name:
        raise ValueError("matrix checkpoint directory does not match training model_tag")
    if training.get("matrix_run_id") and manifest is None:
        raise ValueError("this training checkpoint requires its --matrix-manifest")
    if manifest is not None:
        _verify_matrix_cell(manifest, meta_data, checkpoint_dir, preset, seed, resolved_step)
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
        expected_syn_cfg = ALL_MATRIX_COLUMNS[preset].build_syn_cfg() if manifest else _syn_cfg_for_preset(preset)
        if expected_syn_cfg is None:
            raise ValueError(f"synaptic checkpoint cannot represent vanilla column {preset}")
        if manifest:
            # Training starts with these shared switches, then applies the frozen
            # column's explicit overrides. Check every kinetic constant as well as flags.
            from bio_inspired_nanochat.cmaes_params import coerce_syn_cfg_override

            expected_syn_cfg.use_flex_attention = bool(training.get("use_flex_attention", False))
            flags = {**manifest["recipe"], **manifest["column_flags"][preset]}
            for key, raw in flags.items():
                if key.startswith("syn_cfg."):
                    field = key.removeprefix("syn_cfg.")
                    setattr(expected_syn_cfg, field, coerce_syn_cfg_override(field, raw))
            expected = synaptic_config_to_meta(expected_syn_cfg)
            saved = meta_data.get("synaptic_config") or {}
            mismatches = [k for k in set(expected) | set(saved) if expected.get(k) != saved.get(k) or k not in expected or k not in saved]
        else:
            mismatches = [
                mechanism.field for mechanism in MECHANISMS
                if getattr(syn_cfg, mechanism.field) != getattr(expected_syn_cfg, mechanism.field)
            ]
        if mismatches:
            raise ValueError(
                f"checkpoint bio flags do not match preset {preset!r}: {sorted(mismatches)}"
            )
    return model, meta_data, resolved_step


def _verify_matrix_cell(
    manifest: dict[str, Any], meta: dict[str, Any], directory: Path, preset: str, seed: int,
    step: int,
) -> None:
    """Bind the scored weights to the training run, selected arm, topology and budget."""
    if preset not in manifest["columns"] or seed not in manifest["seeds"]:
        raise ValueError(f"cell ({preset}, {seed}) is outside the matrix manifest")
    tag = manifest["checkpoint_template"].format(preset=preset, seed=seed)
    training = meta["user_config"]
    arch = meta["model_config"]

    def require(field: str, observed: Any, expected: Any) -> None:
        if observed != expected:
            raise ValueError(f"matrix checkpoint {field} mismatch: observed={observed!r}, expected={expected!r}")

    require("directory", directory.name, tag)
    require("model_tag", training.get("model_tag"), tag)
    require("matrix_run_id", training.get("matrix_run_id"), manifest["run_id"])
    require("training data/split/tokenizer", meta.get("matrix_data"), manifest["data"])
    require("training source", meta.get("matrix_source_sha256"), manifest["source_sha256"])
    require("git revision", meta.get("matrix_git_revision"), manifest["git_revision"])
    require("training world size", meta.get("matrix_train_world_size"), manifest["world_size"])
    if not meta.get("matrix_train_device_type") or not meta.get("matrix_optimizer_names"):
        raise ValueError("matrix checkpoint lacks its training device or optimizers")
    require("initialization seed", arch.get("init_seed"), seed)
    require("training seed", training.get("init_seed"), seed)
    require("checkpoint step", meta.get("step"), step)
    require("final step", step, manifest["num_iterations"])
    require("configured steps", training.get("num_iterations"), step)
    require("total batch size", training.get("total_batch_size"), manifest["total_batch_size"])
    require("training tokens", step * training["total_batch_size"], manifest["tokens_per_cell"])
    column = ALL_MATRIX_COLUMNS[preset]
    overrides = column.train_overrides
    # Fixed and lifecycle arms have identical SynapticConfig mechanism flags.
    # The default for a fixed population must be checked even when no CLI flag was emitted.
    controls = {"use_moe": 0, "splitmerge_every": 0, **overrides}
    for name, expected in controls.items():
        require(name, training.get(name), expected)
    require("use_moe topology", bool(arch.get("use_moe")), bool(controls["use_moe"]))
    require("saved lifecycle", (meta.get("splitmerge") or {}).get("every", 0), controls["splitmerge_every"])
    if controls["use_moe"]:
        for name in ("num_experts", "moe_top_k", "moe_hidden_mult"):
            require(name, arch[name], training[name])
        if not training.get("uta4_variable_experts"):
            require("per-layer expert counts", arch.get("moe_experts_per_layer"), [arch["num_experts"]] * arch["n_layer"])
    require("training depth", arch["n_layer"], training["depth"])
    require("training context length", arch["sequence_len"], training["max_seq_len"])
    require("model dimension", arch["n_embd"], training["depth"] * 64)
    expected_heads = max(1, (arch["n_embd"] + 127) // 128)
    require("attention head count", arch["n_head"], expected_heads)
    require("KV head count", arch["n_kv_head"], expected_heads)
    require("initialization", arch["init_type"], training["init_type"])
    require("tied embeddings", arch["tie_embeddings"], bool(training["tie_embeddings"]))
    if meta["synapses"]:
        require("output normalization", arch["final_norm"], bool(training["final_norm"]))
    flags = {**manifest["recipe"], **manifest["column_flags"][preset]}
    for key, raw in flags.items():
        if key.startswith("syn_cfg."):
            from bio_inspired_nanochat.cmaes_params import coerce_syn_cfg_override

            name = key.removeprefix("syn_cfg.")
            require(key, training["syn_cfg_overrides"].get(name), raw)
            require(key, (meta.get("synaptic_config") or {}).get(name), coerce_syn_cfg_override(name, raw))
        else:
            try:
                expected = literal_eval(raw)
            except (ValueError, SyntaxError):
                expected = raw
            require(key, training.get(key), expected)


def _parameter_counts(model: Any) -> tuple[int, int]:
    """Count unique trainable weights and nominal per-token top-k active capacity.

    Shared weights, including tied embeddings, count once. Active capacity includes
    frozen parameters (e.g. the MoE router embeddings) and excludes registered buffers.
    On MoE layers it includes the router and top-k experts on a full-depth token route.
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    active = sum(p.numel() for p in model.parameters())
    for module in model.modules():
        if isinstance(module, SynapticMoE):
            sizes = {sum(p.numel() for p in expert.parameters()) for expert in module.experts}
            if len(sizes) != 1:
                raise ValueError("active parameter count depends on route for unequal-sized experts")
            inactive_experts = max(len(module.experts) - module.top_k, 0)
            active -= inactive_experts * sizes.pop()
    return total, active


def _checkpoint_recipe_stats(
    meta_data: dict[str, Any], checkpoint_step: int
) -> tuple[int, int, float | None, float | None, float | None]:
    """Extract truthful training facts from ``base_train`` JSON metadata."""
    user_config = meta_data.get("user_config") or {}
    loop_state = meta_data.get("loop_state") or {}
    total_batch = int(user_config.get("total_batch_size", 0))
    configured_steps = int(user_config.get("num_iterations", checkpoint_step))
    if configured_steps < 0:
        configured_steps = checkpoint_step
    requested = max(0, configured_steps) * total_batch
    processed = max(0, checkpoint_step) * total_batch
    if user_config.get("matrix_run_id"):
        # Cumulative process time through the start of checkpoint publication, including
        # warmup and evaluation. The launcher measures final write and process-exit time.
        seconds = float(meta_data["matrix_training_wall_seconds"])
        timed_tokens = processed
    else:
        # Historical base_train only accumulates dt on zero-indexed steps >10.
        seconds = float(loop_state.get("total_training_time", 0.0))
        timed_tokens = max(0, checkpoint_step - 11) * total_batch
    walltime = seconds if timed_tokens > 0 and seconds > 0 and math.isfinite(seconds) else None
    throughput = timed_tokens / walltime if walltime is not None else None
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
    if exists:
        with path.open(encoding="utf-8", newline="") as f:
            if next(csv.reader(f), None) != list(fieldnames):
                raise ValueError(f"existing summary schema differs at {path}; choose a fresh batch directory")
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
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


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
        math.ceil(train_tokens_requested / total_batch_size_tokens)
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
    accuracy_matrix: Sequence[Sequence[float | None]],
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
    matrix: list[list[float | None]] = [
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
    bpb_only: bool = False,
    niah_lengths: str = "",
    dead_expert_threshold: float = 0.01,
    continual_tasks: int = 3,
    continual_exposures: int = 4,
    hebb_chunk_len: int = 0,
    init_type: str,
    use_moe: bool,
    num_experts: int,
    moe_top_k: int,
    checkpoint_dir: str,
    checkpoint_step: int,
    manifest: dict[str, Any] | None = None,
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
            manifest=manifest,
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

    live_syn_cfg = getattr(getattr(model, "config", None), "syn_cfg", None)
    neuromod_on = bool(
        live_syn_cfg is not None and getattr(live_syn_cfg, "neuromod_enabled", False)
    )

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
        steps = max(1, math.ceil(train_tokens / total_batch_size_tokens))
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

    trainable, active = _parameter_counts(model)
    identity: dict[str, Any] = {"trainable_params": trainable, "active_params": active}
    if manifest is not None:
        from bio_inspired_nanochat.eval_stats import D1_PARAMETER_MATCH_RULE

        optimizer_keys = (
            "embedding_lr", "unembedding_lr", "matrix_lr", "scalar_lr", "weight_decay",
            "grad_clip", "warmup_ratio", "warmdown_ratio", "final_lr_frac",
        )
        identity.update({
            "matrix_stage": manifest["stage"],
            "protocol_id": manifest["protocol_id"],
            "matrix_manifest_digest": manifest["run_id"],
            "matrix_source_sha256": checkpoint_meta["matrix_source_sha256"],
            "model_file_sha256": checkpoint_meta["model_file_sha256"],
            "checkpoint_git_sha": checkpoint_meta["matrix_git_revision"],
            "checkpoint_config_hash": config_hash({
                "model": {k: v for k, v in checkpoint_meta["model_config"].items() if k != "init_seed"},
                "synaptic": checkpoint_meta.get("synaptic_config"),
            }),
            "trained_cell_id": checkpoint_user_config["model_tag"],
            "train_data_digest": manifest["data"]["train_sha256"],
            "eval_data_digest": manifest["data"]["val_sha256"],
            "tokenizer_digest": manifest["data"]["tokenizer_sha256"],
            "screen_summary_sha256": manifest.get("screen_summary_sha256"),
            "train_device_type": checkpoint_meta["matrix_train_device_type"],
            "train_world_size": checkpoint_meta["matrix_train_world_size"],
            "optimizer_recipe_digest": digest({key: checkpoint_user_config[key] for key in optimizer_keys}),
            "optimizer_names": checkpoint_meta["matrix_optimizer_names"],
            "parameter_match_rule": D1_PARAMETER_MATCH_RULE,
        })
    run_config = {
        **identity,
        **({"bpb_only": True} if bpb_only else {}),
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
        "neuromod_enabled": neuromod_on,
        "hebb_chunk_len": hebb_chunk_len,
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

    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        if device_type == "cuda"
        else nullcontext()
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
        # hy8.1: the neuromodulatory bus is a training-harness object (as in base_train). It
        # gates every synaptic layer after each optimizer step from this step's loss and
        # predictive entropy; without it the neuromod_enabled flag would be a silent no-op here.
        nm_bus = None
        if live_syn_cfg is not None:
            from bio_inspired_nanochat.neuromod import bus_for_config

            nm_bus = bus_for_config(live_syn_cfg)
        with progress:
            for train_step in range(steps):
                t0 = time.perf_counter()
                for _ in range(grad_accum_steps):
                    x, y = next(train_iter)
                    if supports_train_mode and hebb_chunk_len > 0:
                        # hwxb.8: chunked regime — forward+backward per chunk inside the model.
                        step_logits = None
                        with autocast_ctx:
                            loss = model.chunked_train_step(
                                x, y, chunk_len=hebb_chunk_len, loss_scale=1.0 / grad_accum_steps
                            )
                        continue
                    with autocast_ctx:
                        result = model(x, y, train_mode=True) if supports_train_mode else model(x, y)
                    if isinstance(result, tuple):
                        step_logits, loss = result
                    else:
                        step_logits, loss = None, result
                    if loss is None:
                        raise RuntimeError("Model returned loss=None during training")
                    (loss / grad_accum_steps).backward()

                for opt in optimizers:
                    opt.step()
                model.zero_grad(set_to_none=True)
                nm_telemetry: dict[str, float] = {}
                if nm_bus is not None:
                    entropy = (
                        nm_bus.entropy_from_logits(step_logits)
                        if step_logits is not None
                        else None
                    )
                    nm_bus.update(loss=float(loss.detach().float().item()), entropy=entropy)
                    nm_bus.broadcast(model)
                    nm_telemetry = nm_bus.telemetry()
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
                            **nm_telemetry,
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
    eval_steps = max(1, math.ceil(eval_tokens / world_tokens_per_micro) if manifest else eval_tokens // world_tokens_per_micro)
    if bpb_only:
        model.eval()
        # The full scorer reserves the first window for loss/PPL/ECE. Advance the
        # same loader so BPB still sees the exact second window of the manifest.
        for _ in range(eval_steps):
            next(val_iter)
        val_loss = val_ppl = id_ece = ood_auroc = None
        routing = RoutingMetricSummary(None, None, {})
    else:
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

    val_bpb: float | None = None
    if eval_bpb:
        if tokenizer is None:
            val_bpb = None
        else:
            token_bytes = get_token_bytes(device=device)
            if "loss_reduction" in inspect.signature(model.forward).parameters:
                with autocast_ctx:
                    val_bpb = float(evaluate_bpb(model, val_iter, eval_steps, token_bytes))
            else:
                orig_forward = model.forward

                def _syn_forward_wrapper(
                    idx: Tensor,
                    targets: Tensor | None = None,
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
                    with autocast_ctx:
                        val_bpb = float(evaluate_bpb(model, val_iter, eval_steps, token_bytes))
                finally:
                    model.forward = orig_forward
    if bpb_only and (val_bpb is None or not math.isfinite(val_bpb)):
        raise ValueError("BPB-only scoring requires a finite held-out BPB")

    core_metric: float | None = None
    if core_eval and ddp_rank == 0:
        if tokenizer is None:
            core_metric = None
        else:
            with autocast_ctx, _force_synaptic_eval_forward(model) as eval_model:
                out = evaluate_model(
                    eval_model, tokenizer, device, max_per_task=core_max_per_task
                )
            core_metric = float(out["core_metric"])

    # Needle-in-a-haystack long-context retrieval accuracy (74f.2): the key probe of
    # the fast-weight / long-context claim. Swept over length × needle depth.
    niah_acc: float | None = None
    recall_by_length: dict[str, float] = {}
    memory_status = "not_run" if bpb_only else "not_applicable"
    memory_reason: str | None = "no supported context lengths"
    if bpb_only:
        memory_reason = "BPB-only scoring"
    else:
        try:
            from bio_inspired_nanochat.synthetic_tasks import niah_accuracy_by_length

            context_max = int(sequence_len) - 2
            max_len = context_max if niah_lengths.strip() else min(context_max, 256)
            lengths_used = _resolve_niah_lengths(niah_lengths, max_len)
            if lengths_used and ddp_rank == 0:
                with autocast_ctx, _force_synaptic_eval_forward(model) as eval_model:
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

    continual_reason = "BPB-only scoring" if bpb_only else "non-zero DDP rank"
    continual = ContinualMetricSummary(None, {}, [], "not_run", continual_reason)
    if not bpb_only and ddp_rank == 0:
        try:
            with autocast_ctx:
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
        "uncertainty": "not_run" if bpb_only else "ok",
        "continual": continual.status,
        "routing": "not_run" if bpb_only else ("ok" if routing.layers else "not_applicable"),
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
        "eval_device_batch_size": device_batch_size,
        "eval_tokens_processed": eval_steps * world_tokens_per_micro,
        **identity,
    }
    if ddp_rank == 0:
        registry_metrics: dict[str, float] = {}
        if summary.tok_per_sec is not None:
            registry_metrics["tok_per_sec"] = summary.tok_per_sec
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
                "status": capability_metric_status["uncertainty"],
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
            hebb_chunk_len=args.hebb_chunk_len,
            init_type=args.init_type,
            use_moe=args.use_moe,
            num_experts=args.num_experts,
            moe_top_k=args.moe_top_k,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_step=args.checkpoint_step,
            manifest=args.matrix_manifest,
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
    registered = args.matrix_manifest and args.matrix_manifest["registered"]
    if registered and (ddp or device.type != "cuda"):
        raise ValueError("registered matrix scoring requires the single CUDA process in the launch plan")
    if registered and (presets != args.matrix_manifest["columns"] or seeds != args.matrix_manifest["seeds"]):
        raise ValueError("registered scoring requires the full ordered matrix roster")
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
        batch_out_dir.mkdir(parents=True, exist_ok=not registered)
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
                    score_cell = partial(_run_one,
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
                        bpb_only=args.bpb_only,
                        core_eval=args.core_eval,
                        core_max_per_task=args.core_max_per_task,
                        ece_bins=args.ece_bins,
                        niah_lengths=args.niah_lengths,
                        dead_expert_threshold=args.dead_expert_threshold,
                        continual_tasks=args.continual_tasks,
                        continual_exposures=args.continual_exposures,
                        hebb_chunk_len=args.hebb_chunk_len,
                        init_type=args.init_type,
                        use_moe=args.use_moe,
                        num_experts=args.num_experts,
                        moe_top_k=args.moe_top_k,
                        checkpoint_dir=args.checkpoint_dir,
                        checkpoint_step=args.checkpoint_step,
                        manifest=args.matrix_manifest,
                        registry_path=args.registry_path,
                        runtime=runtime,
                    )
                    with score_budget(args.matrix_manifest, preset, seed) if registered else nullcontext():
                        score_cell()
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
    if registered and not had_failure:
        seal_scores(args.matrix_manifest, batch_out_dir / "summary.csv")
    return int(had_failure)


def _cmd_matrix(args: argparse.Namespace) -> int:
    allowed = set(PresetId.__args__) | set(MATRIX_COLUMNS)
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
        p.add_argument("--matrix-manifest", help="Verify this frozen launcher manifest, checkpoint and live evaluation data")
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
        p.add_argument("--bpb-only", action="store_true",
                       help="Score the same held-out BPB window and skip auxiliary diagnostics")
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
        p.add_argument(
            "--hebb-chunk-len", type=int, default=0,
            help="hwxb.8: >0 trains synaptic presets by reading each batch this many tokens at a time "
            "through a KV cache so online Hebbian writes act within the sequence (0 = full forward)",
        )

        # batch / opt
        p.add_argument("--device-batch-size", type=int, default=8)
        p.add_argument("--total-batch-size-tokens", type=int, default=131072)
        p.add_argument("--embedding-lr", type=float, default=0.2)
        p.add_argument("--unembedding-lr", type=float, default=0.004)
        p.add_argument("--matrix-lr", type=float, default=0.02)
        p.add_argument("--weight-decay", type=float, default=0.0)

    p_run = sub.add_parser("run", help="Run a single preset/seed")
    add_common(p_run)
    p_run.add_argument("--preset", required=True, choices=[*PresetId.__args__, *MATRIX_COLUMNS])
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
    if args.bpb_only:
        if args.cmd != "matrix" or not args.matrix_manifest or not args.eval_bpb:
            parser.error("--bpb-only requires matrix, --matrix-manifest, and --eval-bpb")
        if args.core_eval or args.niah_lengths:
            parser.error("--bpb-only is incompatible with --core-eval or --niah-lengths")
        if args.eval_tokens < 1 or args.device_batch_size < 1:
            parser.error("--bpb-only requires positive evaluation token and batch sizes")
    if not args.checkpoint_dir and not args.inline_smoke_training:
        parser.error(
            "scientific evaluation requires --checkpoint-dir from base_train; "
            "use --inline-smoke-training only for CI plumbing smoke tests"
        )
    if args.matrix_manifest:
        if not args.checkpoint_dir or args.data != "fineweb" or not args.eval_bpb:
            parser.error("--matrix-manifest requires --checkpoint-dir, --data fineweb, and --eval-bpb")
        args.matrix_manifest = load_manifest(args.matrix_manifest)
        if args.matrix_manifest.get("registered") or "bpb_only" in args.matrix_manifest:
            if args.bpb_only is not args.matrix_manifest.get("bpb_only", False):
                parser.error("scoring mode differs from the matrix manifest")
        if args.bpb_only:
            recipe = args.matrix_manifest["recipe"]
            if (args.eval_tokens != int(recipe.get("eval_tokens", 1_000_000)) or
                args.device_batch_size != int(recipe.get("device_batch_size", 8))):
                parser.error("BPB token budget and batch size must match the matrix manifest")
        if dataset_identity() != args.matrix_manifest["data"]:
            raise ValueError("evaluation data/split/tokenizer do not match matrix manifest")
        if args.matrix_manifest["registered"]:
            from bio_inspired_nanochat.eval_stats import D1_PROTOCOL_VERSION

            if args.cmd != "matrix":
                parser.error("registered scoring requires the matrix command emitted by the launcher")
            if args.matrix_manifest["protocol_id"] != D1_PROTOCOL_VERSION:
                raise ValueError("evaluation protocol differs from the registered manifest")
            if source_identity() != (args.matrix_manifest["git_revision"], args.matrix_manifest["source_sha256"]):
                raise ValueError("evaluation source differs from the registered training manifest")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
