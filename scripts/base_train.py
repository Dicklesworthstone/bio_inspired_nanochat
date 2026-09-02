"""
Train model. Run as:

python base_train.py

or distributed as:

torchrun --nproc_per_node=8 base_train.py

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max_seq_len=512 --device_batch_size=1 --eval_tokens=512 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20
"""

import math
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from typing import Any, cast

import torch.distributed as torch_dist
import wandb

from bio_inspired_nanochat.checkpoint_manager import (
    capture_prefetched_batch,
    capture_rng_state,
    checkpoint_model_config,
    config_provenance,
    load_checkpoint,
    load_checkpoint_metadata,
    require_complete_checkpoint,
    restore_optimizer_states,
    restore_prefetched_batch,
    restore_rank_model_state,
    restore_rng_state,
    save_checkpoint,
    synaptic_config_from_meta,
    synaptic_config_to_meta,
    validate_exact_resume_payload_step,
)
from bio_inspired_nanochat.common import (
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    print0,
    print_banner,
)
from bio_inspired_nanochat.cmaes_params import extract_syn_cfg_cli_overrides
from bio_inspired_nanochat.dataloader import (
    collate_dataloader_state_dicts,
    tokenizing_distributed_data_loader,
    tokenizing_distributed_data_loader_with_state,
)
from bio_inspired_nanochat.divergence_guard import GuardAction, build_divergence_guard
from bio_inspired_nanochat.engine import Engine
from bio_inspired_nanochat.gpt import GPT, GPTConfig
from bio_inspired_nanochat.loss_eval import evaluate_bpb
from bio_inspired_nanochat.report import get_report
from bio_inspired_nanochat.run_logging import TrainingTelemetry
from bio_inspired_nanochat.tokenizer import get_token_bytes, get_tokenizer
from bio_inspired_nanochat.torch_imports import F, torch
from scripts.base_eval import evaluate_model

dist = cast(Any, torch_dist)

print_banner()

# -----------------------------------------------------------------------------
# User settings
run = "dummy"  # wandb run name default ("dummy" is special - we won't log to wandb)
# Runtime
device_type = ""  # cuda|cpu|mps (empty => autodetect good device type default, in order: CUDA > MPS > CPU)
# Model architecture
depth = (
    20  # the depth of the Transformer model to train, rest of the kwargs are derived
)
max_seq_len = 2048  # max context length
synapses = 0  # use synaptic model (GPTSynaptic) if 1, otherwise use standard GPT
use_flex_attention = 0 # use FlexAttention (requires torch>=2.5) if 1
load_cmaes_params = ""  # optional path to a CMA-ES params JSON overlaid onto SynapticConfig (bead c2l; see docs/cmaes_params.md)
init_type = "baseline"  # baseline | ca_rule30 | ca_rule116
init_seed = 42
tie_embeddings = 0  # hwxb.2.9: tie wte/lm_head into one shared matrix (1=on; recommended for small scale-up models)
# Split/merge controller (for MoE)
use_moe = 0  # enable SynapticMoE blocks (structural lifecycle enables this automatically)
num_experts = 8
moe_top_k = 2
moe_hidden_mult = 4
splitmerge_every = 0  # apply split/merge every N steps (0=off)
merge_cosine = 0.85  # merge cosine similarity threshold
merge_health_max = 0.25  # merge health threshold
splits_per_call = 1  # splits per step
merges_per_call = 1  # merges per step
split_health_min = 0.80  # split health threshold
sm_use_neuroscore = 0  # blend NeuroScore fitness into lifecycle health (needs NeuroViz/NeuroScore active)
sm_neuroscore_weight = 0.5  # blend weight in [0,1] when sm_use_neuroscore=1
sm_function_preserving = 1  # Net2Net/firefly: make split/merge output-preserving (uta.3); 0=legacy noisy clone
sm_fp_divergence_noise = 0.02  # relative (to weight RMS) antisymmetric fc1 noise for function-preserving split
sm_verbose = 0  # verbose split/merge logging
sm_homeostasis_guards = 0  # uta.6: routed-mass ramp + energy floor + row-wise moment warm restart after lifecycle events
sm_gate_ramp_forwards = 512  # uta.6: training forwards over which a freshly seeded expert ramps in (needs sm_homeostasis_guards=1)
sm_energy_floor = 0.05  # uta.6: per-expert energy floor after events (needs sm_homeostasis_guards=1)
topological_nas = 0  # 0642.5: certificate-driven lifecycle; default-off, falls back to UTA
uta4_variable_experts = 0  # uta.4: allow REAL expert-count growth/shrink under a budget
uta4_min_experts = 2  # hard floor on per-layer expert count
uta4_max_experts = 64  # hard cap on per-layer expert count
uta4_growth_budget_pct = 0.5  # max NET added experts, fraction of initial total (FLOP budget)
# Neuromodulatory bus (hy8.1): global DA/ACh/NE scalars gating plasticity/exploration/gain.
neuromod_enabled = 0  # 1=compute DA/ACh/NE from loss+entropy each step and broadcast to synapses
neuromod_log_every = 100  # print neuromodulator telemetry every N steps (0=never)
# Training horizon. Only one of these 3 will be used, in this order of precedence.
num_iterations = -1  # explicit number of steps of the optimization (-1 = disable)
target_flops = -1.0  # calculate num_iterations to reach target_flops. Useful for scaling laws experiments (-1 = disable)
target_param_data_ratio = 20  # calculate num_iterations to maintain fixed data:param ratio (Chinchilla=20) (-1 = disable)
# Optimization
device_batch_size = 32  # per-device batch size (set to not OOM)
total_batch_size = 524288  # total desired batch size, in #tokens
embedding_lr = 0.2  # learning rate for the embedding parameters (Adam)
unembedding_lr = 0.004  # learning rate for the unembedding parameters (Adam)
weight_decay = 0.0  # weight decay for the embedding/unembedding parameters (Adam)
matrix_lr = 0.02  # learning rate for the matrix parameters (Muon)
grad_clip = 1.0  # gradient clipping value (0.0 = disabled)
warmup_ratio = 0.0  # ratio of iterations for LR warmup
warmdown_ratio = 0.2  # ratio of iterations for LR warmdown
final_lr_frac = 0.0  # final LR is this fraction of the initial LR
resume_from_step = (
    -1
)  # resume training from this step of the optimization (-1 = disable)
# Evaluation
eval_every = 250  # every how many steps to evaluate the model for val bpb
eval_tokens = 20 * 524288  # number of tokens to evaluate val loss on
core_metric_every = (
    2000  # every how many steps to evaluate the core metric (-1 = disable)
)
core_metric_max_per_task = 500  # examples per task in estimating the core metric
sample_every = 2000  # every how many steps to sample from the model
save_every = -1  # every how many steps to save model checkpoints (-1 = disable, and save only at the end of the run)
# Output
model_tag = (
    ""  # optionally override the model tag for the output checkpoint directory name
)
# NeuroViz settings
neuroviz_dir = "runs/neuroviz"
neuroviz_image_every = 10000
neuroviz_tb_every = 1000
neuroviz_interactive_every = 25000

# ``--syn_cfg.<field>=<value>`` overrides any SynapticConfig field from the command line
# (the README's "Key Training Flags"). They are pulled out BEFORE the configurator runs
# because it only knows the module-level settings above and would reject the dotted keys.
# Values are typed and validated against the dataclass when the config is built below.
sys.argv, syn_cfg_overrides = extract_syn_cfg_cli_overrides(sys.argv)

# now allow CLI to override the settings via the configurator lol
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
with open(
    os.path.join("bio_inspired_nanochat", "configurator.py"),
    encoding="utf-8",
) as f:
    exec(f.read())  # noqa: S102  # nosec B102 # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys}  # will be useful for logging
user_config["syn_cfg_overrides"] = dict(syn_cfg_overrides)
if syn_cfg_overrides and not synapses:
    # Fail here, before the tokenizer/data/model setup, so a typo in the flags costs seconds.
    raise ValueError(
        f"--syn_cfg.* overrides ({sorted(syn_cfg_overrides)}) require synapses=1; "
        "the vanilla GPT has no SynapticConfig"
    )
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
autocast_ctx = (
    torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    if device_type == "cuda"
    else nullcontext()
)
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = (
    DummyWandb()
    if use_dummy_wandb
    else wandb.init(project="nanochat", name=run, config=user_config)
)

# Tokenizer will be useful for evaluation, also we need the vocab size
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# Resolve checkpoint metadata before deriving model dimensions or batch geometry.
# A resumed run must use the architecture and sequence length that produced the
# checkpoint, even when a model_tag lets the caller reach it with different CLI defaults.
base_dir = get_base_dir()
output_dirname = model_tag if model_tag else f"d{depth}"  # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = resume_from_step != -1
if resuming:
    require_complete_checkpoint(checkpoint_dir, resume_from_step)
resume_meta = (
    load_checkpoint_metadata(checkpoint_dir, resume_from_step) if resuming else None
)
if resuming:
    validate_exact_resume_payload_step(
        resume_meta,
        resume_from_step,
        payload_name="checkpoint metadata",
    )
resume_model_config = (resume_meta or {}).get("model_config", {})
if not isinstance(resume_model_config, dict):
    raise ValueError("checkpoint model_config metadata must be a mapping")
resume_splitmerge = (resume_meta or {}).get("splitmerge")
if resume_splitmerge is not None:
    if not isinstance(resume_splitmerge, dict):
        raise ValueError("checkpoint splitmerge metadata must be a mapping")
    saved_every = resume_splitmerge.get("every")
    saved_config = resume_splitmerge.get("config")
    if isinstance(saved_every, bool) or not isinstance(saved_every, int):
        raise ValueError("checkpoint splitmerge.every must be an integer")
    if saved_every <= 0 or not isinstance(saved_config, dict):
        raise ValueError("checkpoint splitmerge metadata is incomplete or disabled")
    splitmerge_every = saved_every

if resuming:
    core_fields = (
        "sequence_len",
        "vocab_size",
        "n_layer",
        "n_head",
        "n_embd",
    )
    missing_core = [name for name in core_fields if name not in resume_model_config]
    if missing_core:
        raise ValueError(
            f"checkpoint model_config is missing core fields: {missing_core}"
        )
    for name in core_fields:
        value = resume_model_config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"checkpoint model_config field {name!r} must be a positive integer"
            )
    saved_vocab_size = int(resume_model_config["vocab_size"])
    if saved_vocab_size != vocab_size:
        raise ValueError(
            "checkpoint vocabulary does not match the active tokenizer: "
            f"{saved_vocab_size} != {vocab_size}"
        )
    max_seq_len = int(resume_model_config["sequence_len"])
    num_layers = int(resume_model_config["n_layer"])
    num_heads = int(resume_model_config["n_head"])
    saved_num_kv_heads = resume_model_config.get("n_kv_head", num_heads)
    if (
        isinstance(saved_num_kv_heads, bool)
        or not isinstance(saved_num_kv_heads, int)
        or saved_num_kv_heads <= 0
    ):
        raise ValueError("checkpoint n_kv_head must be a positive integer")
    num_kv_heads = saved_num_kv_heads
    model_dim = int(resume_model_config["n_embd"])
    saved_meta_seq_len = (resume_meta or {}).get("max_seq_len", max_seq_len)
    if saved_meta_seq_len != max_seq_len:
        raise ValueError(
            "checkpoint max_seq_len disagrees with model_config.sequence_len"
        )
    saved_device_batch_size = (resume_meta or {}).get("device_batch_size")
    if (
        isinstance(saved_device_batch_size, bool)
        or not isinstance(saved_device_batch_size, int)
        or saved_device_batch_size <= 0
    ):
        raise ValueError("checkpoint device_batch_size must be a positive integer")
    device_batch_size = saved_device_batch_size
    saved_user_config = (resume_meta or {}).get("user_config", {})
    if not isinstance(saved_user_config, dict):
        raise ValueError("checkpoint user_config metadata must be a mapping")
    integer_trajectory_fields = (
        "num_iterations",
        "eval_every",
        "eval_tokens",
        "core_metric_every",
        "core_metric_max_per_task",
        "sample_every",
        "neuromod_log_every",
    )
    real_trajectory_fields = (
        "target_flops",
        "target_param_data_ratio",
        "grad_clip",
        "warmup_ratio",
        "warmdown_ratio",
        "final_lr_frac",
    )
    toggle_trajectory_fields = ("neuromod_enabled",)
    trajectory_config_fields = (
        *integer_trajectory_fields,
        *real_trajectory_fields,
        *toggle_trajectory_fields,
    )
    missing_trajectory_config = [
        name for name in trajectory_config_fields if name not in saved_user_config
    ]
    if missing_trajectory_config:
        raise ValueError(
            "checkpoint lacks trajectory-defining training config fields required "
            f"for exact resume: {missing_trajectory_config}"
        )
    for name in integer_trajectory_fields:
        value = saved_user_config[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"checkpoint trajectory config field {name!r} must be an integer"
            )
        globals()[name] = value
        user_config[name] = value
    for name in real_trajectory_fields:
        value = saved_user_config[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(
                f"checkpoint trajectory config field {name!r} must be numeric"
            )
        globals()[name] = value
        user_config[name] = value
    for name in toggle_trajectory_fields:
        value = saved_user_config[name]
        if not isinstance(value, (bool, int)) or int(value) not in (0, 1):
            raise ValueError(
                f"checkpoint trajectory config field {name!r} must be boolean-like"
            )
        globals()[name] = int(bool(value))
        user_config[name] = int(bool(value))
    saved_total_batch_size = (resume_meta or {}).get(
        "total_batch_size",
        saved_user_config.get("total_batch_size", total_batch_size),
    )
    if (
        isinstance(saved_total_batch_size, bool)
        or not isinstance(saved_total_batch_size, int)
        or saved_total_batch_size <= 0
    ):
        raise ValueError("checkpoint total_batch_size must be a positive integer")
    total_batch_size = saved_total_batch_size
    saved_init_seed = resume_model_config.get("init_seed", init_seed)
    if isinstance(saved_init_seed, bool) or not isinstance(saved_init_seed, int):
        raise ValueError("checkpoint init_seed must be an integer")
    init_seed = saved_init_seed
    init_type = str(resume_model_config.get("init_type", init_type))
    saved_tie_embeddings = resume_model_config.get("tie_embeddings", tie_embeddings)
    if (
        not isinstance(saved_tie_embeddings, (bool, int))
        or int(saved_tie_embeddings) not in (0, 1)
    ):
        raise ValueError("checkpoint tie_embeddings must be boolean-like")
    tie_embeddings = int(bool(saved_tie_embeddings))
    depth = num_layers
    user_config.update(
        depth=depth,
        max_seq_len=max_seq_len,
        device_batch_size=device_batch_size,
        total_batch_size=total_batch_size,
        init_type=init_type,
        init_seed=init_seed,
        tie_embeddings=tie_embeddings,
    )
else:
    # Model kwargs are derived from the desired depth for a fresh run.
    num_layers = depth
    model_dim = depth * 64
    num_heads = max(1, (model_dim + 127) // 128)
    num_kv_heads = num_heads
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")

# Optimizer / data / training length related hyperparameters
# figure out the needed gradient accumulation to reach the desired total batch size
tokens_per_fwdbwd = (
    device_batch_size * max_seq_len
)  # tokens per iteration for a single rank
world_tokens_per_fwdbwd = (
    tokens_per_fwdbwd * ddp_world_size
)  # total tokens per iteration for all ranks
if total_batch_size % world_tokens_per_fwdbwd != 0:
    raise ValueError(f"total_batch_size {total_batch_size} must be divisible by world_tokens_per_fwdbwd {world_tokens_per_fwdbwd}")
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(
    f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}"
)
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(
    f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}"
)

# -----------------------------------------------------------------------------
# Initialize the Model

# Create a new model with random weights
model_config_kwargs = {
    "sequence_len": max_seq_len,
    "vocab_size": vocab_size,
    "n_layer": num_layers,
    "n_head": num_heads,
    "n_kv_head": num_kv_heads,
    "n_embd": model_dim,
    "tie_embeddings": bool(resume_model_config.get("tie_embeddings", tie_embeddings)),
}
use_syn = bool((resume_meta or {}).get("synapses", synapses))
if splitmerge_every > 0 and not use_syn:
    raise ValueError("splitmerge_every > 0 requires synapses=1")
if use_syn:
    try:
        from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
        from bio_inspired_nanochat.synaptic import SynapticConfig
    except Exception as e:
        raise RuntimeError(
            "synapses=1 but synaptic model modules failed to import."
        ) from e

    syn_cfg = (
        synaptic_config_from_meta(resume_meta)
        if resume_meta is not None
        else SynapticConfig(
            use_flex_attention=bool(use_flex_attention),
            topological_nas=bool(topological_nas),
        )
    )
    if load_cmaes_params:
        if resume_meta is not None:
            raise ValueError(
                "load_cmaes_params cannot be combined with --resume: overlaying search "
                "parameters onto a resumed run would desync the model from its checkpoint "
                "config. Start a fresh run with synapses=1 instead."
            )
        from bio_inspired_nanochat.cmaes_params import apply_cmaes_params

        syn_cfg = apply_cmaes_params(syn_cfg, load_cmaes_params)
        print0(f"[config] overlaid CMA-ES params from {load_cmaes_params}")
    if syn_cfg_overrides:
        if resume_meta is not None:
            raise ValueError(
                "--syn_cfg.* overrides cannot be combined with --resume: the resumed run must "
                "keep the SynapticConfig its checkpoint was trained with. Start a fresh run."
            )
        from bio_inspired_nanochat.cmaes_params import apply_syn_cfg_overrides

        syn_cfg = apply_syn_cfg_overrides(syn_cfg, syn_cfg_overrides)
        for _name in sorted(syn_cfg_overrides):
            print0(f"[config] syn_cfg.{_name} = {getattr(syn_cfg, _name)!r}")
    if neuromod_enabled:
        # The bus is a training-loop object, but the model config records that it was on so
        # checkpoints and eval_matrix see the same flag (--syn_cfg.neuromod_enabled=1 also works).
        syn_cfg.neuromod_enabled = True
    model_config = GPTSynapticConfig(
        sequence_len=max_seq_len,
        vocab_size=vocab_size,
        n_layer=num_layers,
        n_head=num_heads,
        n_kv_head=num_kv_heads,
        n_embd=model_dim,
        synapses=True,
        syn_cfg=syn_cfg,
        use_moe=bool(
            resume_model_config.get("use_moe", use_moe or splitmerge_every > 0)
        ),
        num_experts=int(resume_model_config.get("num_experts", num_experts)),
        moe_experts_per_layer=(
            tuple(int(value) for value in resume_model_config["moe_experts_per_layer"])
            if resume_model_config.get("moe_experts_per_layer") is not None
            else None
        ),
        moe_top_k=int(resume_model_config.get("moe_top_k", moe_top_k)),
        moe_hidden_mult=int(
            resume_model_config.get("moe_hidden_mult", moe_hidden_mult)
        ),
        dropout=float(resume_model_config.get("dropout", 0.0)),
        moe_balance_loss=float(resume_model_config.get("moe_balance_loss", 0.01)),
        init_type=str(resume_model_config.get("init_type", init_type)),
        init_seed=int(resume_model_config.get("init_seed", init_seed)),
        tie_embeddings=bool(
            resume_model_config.get("tie_embeddings", tie_embeddings)
        ),
    )
    if syn_cfg.topological_nas and splitmerge_every <= 0:
        raise ValueError("topological_nas=1 requires splitmerge_every > 0")
    with torch.device("meta"):
        model = GPTSynaptic(model_config)
else:
    with torch.device("meta"):
        model_config = GPTConfig(
            sequence_len=max_seq_len,
            vocab_size=vocab_size,
            n_layer=num_layers,
            n_head=num_heads,
            n_kv_head=num_kv_heads,
            n_embd=model_dim,
            init_type=str(resume_model_config.get("init_type", init_type)),
            init_seed=int(resume_model_config.get("init_seed", init_seed)),
            tie_embeddings=bool(
                resume_model_config.get("tie_embeddings", tie_embeddings)
            ),
        )
        model = GPT(model_config)
model.to_empty(device=device)
model.init_weights()

# hwxb.7.1: unified telemetry — structured JSONL (queryable; consumed by the Phase-4
# ablation analysis) + TensorBoard scalars, rank-0 only (no-op on other ranks).
telemetry = TrainingTelemetry(
    os.path.join("runs", "telemetry", output_dirname),
    name=run if run != "dummy" else f"train-{output_dirname}",
    is_master=master_process,
    heavy_every=100,
    provenance={
        "synapses": bool(use_syn),
        "depth": int(depth),
        "world_size": int(ddp_world_size),
        "seed": int(init_seed),
    },
)
train_state: dict[str, Any] | None = None
if resuming:
    print0(f"Resuming optimization from step {resume_from_step}")
    model_data, optimizer_data, meta_data, train_state = load_checkpoint(
        checkpoint_dir, resume_from_step, device, load_optimizer=True, rank=ddp_rank,
        load_train_state=True,
    )
    validate_exact_resume_payload_step(
        train_state,
        resume_from_step,
        payload_name="rank-local train state",
    )
    model.load_state_dict(model_data, strict=True, assign=True)
    restore_rank_model_state(
        model,
        train_state,
        required=ddp_world_size > 1,
    )
    del model_data  # free up this memory after the copy
    # hwxb.2.9: load_state_dict(assign=True) replaces the param objects, breaking any
    # wte/lm_head tie — re-establish it so the shared weight trains as one on resume.
    model.tie_weights()
orig_model = model  # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
model = torch.compile(
    model, dynamic=False
)  # the inputs to model will never change shape so dynamic=False is safe
num_params = sum(p.numel() for p in model.parameters())
print0(f"Number of parameters: {num_params:,}")
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Calculate number of iterations. Either it is given, or from target flops, or from target data:param ratio (in that order)
if not (num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0):
    raise ValueError("No training horizon specified (num_iterations, target_param_data_ratio, or target_flops)")
if num_iterations > 0:
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif target_flops > 0:
    # calculate the number of iterations from the target flops
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif target_param_data_ratio > 0:
    # calculate the number of iterations from the target param data ratio
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print0(
        f"Calculated number of iterations from target data:param ratio: {num_iterations:,}"
    )
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(
    f"Tokens : Params ratio: {total_batch_size * num_iterations / num_params:.2f}"
)  # Chinchilla is ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr,
    embedding_lr=embedding_lr,
    matrix_lr=matrix_lr,
    weight_decay=weight_decay,
)
if len(optimizers) == 2:
    adamw_optimizer, muon_optimizer = optimizers
else:
    # GPTSynaptic returns single optimizer
    adamw_optimizer = optimizers[0]
    muon_optimizer = None

# Initialize NeuroVizManager
viz = None
if use_syn and master_process:
    try:
        from bio_inspired_nanochat.neuroviz import NeuroVizConfig, NeuroVizManager
    except Exception as e:
        print0(f"NeuroViz disabled (import failed): {e}")
    else:
        viz_cfg = NeuroVizConfig(
            log_dir=neuroviz_dir,
            image_every=neuroviz_image_every,
            tb_every=neuroviz_tb_every,
            interactive_every=neuroviz_interactive_every,
        )
        viz = NeuroVizManager(viz_cfg)
        # Use the original model (before compile) to avoid compile wrappers.
        viz.register_model(orig_model)

# Initialize split/merge controller if enabled
sm_ctrl = None
sm_cfg = None
if splitmerge_every > 0:
    try:
        from bio_inspired_nanochat.synaptic_splitmerge import (
            SplitMergeConfig,
            SplitMergeController,
        )
    except Exception as e:
        raise RuntimeError(
            "splitmerge_every > 0 but synaptic split/merge modules failed to import."
        ) from e
    if resume_splitmerge is not None:
        sm_cfg = SplitMergeConfig(**resume_splitmerge["config"])
        if not sm_cfg.enabled or sm_cfg.min_step_interval != splitmerge_every:
            raise ValueError(
                "checkpoint splitmerge schedule disagrees with its controller config"
            )
    else:
        sm_cfg = SplitMergeConfig(
            enabled=True,
            merge_cosine_threshold=merge_cosine,
            merge_health_max=merge_health_max,
            merges_per_call=merges_per_call,
            split_health_min=split_health_min,
            splits_per_call=splits_per_call,
            min_step_interval=splitmerge_every,
            use_neuroscore=bool(sm_use_neuroscore),
            neuroscore_weight=float(sm_neuroscore_weight),
            function_preserving=bool(sm_function_preserving),
            fp_divergence_noise=float(sm_fp_divergence_noise),
            variable_expert_count=bool(uta4_variable_experts),
            min_experts=int(uta4_min_experts),
            max_experts=int(uta4_max_experts),
            growth_budget_pct=float(uta4_growth_budget_pct),
            homeostasis_guards=bool(sm_homeostasis_guards),
            gate_ramp_forwards=int(sm_gate_ramp_forwards),
            energy_floor=float(sm_energy_floor),
            verbose=bool(sm_verbose),
            ddp_broadcast=True,
        )
    sm_ctrl = SplitMergeController(
        orig_model,
        sm_cfg,
        logger=viz,
        event_logger=telemetry,
    )
    if train_state is not None and train_state.get("splitmerge") is not None:
        sm_ctrl.load_state_dict(train_state["splitmerge"])
        print0("[checkpoint] restored split/merge schedule and growth-budget state")
    elif resuming:
        print0(
            "[checkpoint] WARNING: no split/merge controller state; "
            "lifecycle scheduling resumes from checkpoint step only"
        )

# Neuromodulatory bus (hy8.1): only for synaptic models, opt-in. Default-neutral when off.
nm_bus = None
if use_syn:
    from bio_inspired_nanochat.neuromod import bus_for_config

    # --neuromod_enabled=1 was folded into syn_cfg.neuromod_enabled above, so one flag rules.
    nm_bus = bus_for_config(syn_cfg)
    if nm_bus is not None and resuming:
        nm_state = train_state.get("neuromod") if train_state is not None else None
        if nm_state is None:
            raise ValueError("exact resume requires neuromodulator state when enabled")
        nm_bus.load_state_dict(nm_state, strict=True)
        nm_bus.broadcast(orig_model)
        print0("[checkpoint] restored neuromodulator EMA levels and broadcast gains")

if resuming:
    restore_optimizer_states(optimizers, optimizer_data)
    del optimizer_data  # free up the memory

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
tokens_dir = os.path.join(base_dir, "tokenized_data")
dataloader_resume_state_dict = None
if resuming:
    if (
        train_state is not None
        and isinstance(train_state, dict)
        and "dataloader_state_dict" in train_state
    ):
        dataloader_resume_state_dict = train_state["dataloader_state_dict"]
    elif meta_data is not None and "dataloader_state_dict" in meta_data:
        dataloader_resume_state_dict = meta_data["dataloader_state_dict"]

train_loader = tokenizing_distributed_data_loader_with_state(
    device_batch_size,
    max_seq_len,
    split="train",
    device=device,
    resume_state_dict=dataloader_resume_state_dict,
)


def build_val_loader():
    return tokenizing_distributed_data_loader(
        device_batch_size, max_seq_len, split="val", device=device
    )


if resuming:
    if dataloader_resume_state_dict is None:
        raise ValueError("exact resume requires a rank-local dataloader state")
    x, y = restore_prefetched_batch(
        train_state,
        device=device,
        expected_shape=(device_batch_size, max_seq_len),
    )
    # The restored batch is the loader's most recently yielded batch, so its saved
    # cursor already points immediately after x/y. The next loader advance after
    # backward will therefore produce the following batch without a skip.
    dataloader_state_dict = dataloader_resume_state_dict
else:
    x, y, dataloader_state_dict = next(
        train_loader
    )  # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Set up hyperparameter schedulers


# Learning rate scheduler
def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac


# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum


# -----------------------------------------------------------------------------
# Loop state (variables updated by the training loop)

if not resuming:
    step = 0
    min_val_bpb = float("inf")
    val_bpb = float("inf")
    smooth_train_loss = 0  # EMA of training loss
    total_training_time = 0  # total wall-clock time of training
else:
    step = meta_data["step"]
    val_bpb = meta_data.get("val_bpb")
    if (
        isinstance(val_bpb, bool)
        or not isinstance(val_bpb, (int, float))
        or not math.isfinite(float(val_bpb))
    ):
        raise ValueError("checkpoint val_bpb must be finite for exact resume")
    val_bpb = float(val_bpb)
    loop_state = meta_data["loop_state"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# Only used for end-of-run reporting; will be overwritten inside the loop.
mfu: float = 0.0

# vg9.7: training-loop divergence guard. Detects NaN/Inf and runaway dynamics on the loss and
# the bio buffers (CaMKII/BDNF/calcium/fast weights) and applies a configurable response
# (default: skip the bad step on NaN/Inf, back off the LR on a loss spike). Logs bio-buffer
# norms each step for early warning. Especially important with the stateful, positive-feedback
# bio mechanisms now live on the forward path.
divguard = build_divergence_guard()
if resuming:
    divguard_state = (
        train_state.get("divergence_guard") if train_state is not None else None
    )
    if divguard_state is None:
        raise ValueError("exact resume requires divergence-guard state")
    divguard.load_state_dict(divguard_state)
    print0("[checkpoint] restored divergence-guard EMA and rollback state")
    # Restore RNG last, after compile wrapping and every runtime/controller object has
    # been rebuilt. Initialization work must not consume draws from the saved future.
    restore_rng_state(train_state.get("rng"))
    print0(
        f"[checkpoint] restored RNG state for bit-comparable resume from step "
        f"{resume_from_step}"
    )

# -----------------------------------------------------------------------------
# Training loop
while True:
    last_step = (
        step == num_iterations
    )  # loop runs num_iterations+1 times so that we can eval/save at the end
    flops_so_far = num_flops_per_token * total_batch_size * step
    replaying_checkpoint_boundary = resuming and step == resume_from_step

    # once in a while: evaluate the val bpb (all ranks participate)
    if not replaying_checkpoint_boundary and (last_step or step % eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        # Ensure eval_steps is at least 1
        eval_steps = max(1, eval_steps)
        with autocast_ctx:
            # evaluate_bpb expects model(x, y, loss_reduction='none') which works for GPT
            # For GPTSynaptic, we need to wrap it
            if use_syn:
                orig_forward = model.forward

                def syn_forward_wrapper(
                    idx,
                    targets=None,
                    kv_cache=None,
                    loss_reduction="mean",
                    *,
                    orig_forward=orig_forward,  # bind now: this closure is rebuilt every eval
                    **kwargs,
                ):
                    if targets is not None:
                        logits, loss = orig_forward(
                            idx, targets, kv_cache, train_mode=False
                        )
                        if loss_reduction == "none":
                            # Need to compute per-token losses for evaluate_bpb
                            logits_flat = logits.view(-1, logits.size(-1))
                            targets_flat = targets.view(-1)
                            loss_per_token = F.cross_entropy(
                                logits_flat,
                                targets_flat,
                                reduction="none",
                                ignore_index=-1,
                            )
                            return loss_per_token.view(targets.shape)
                        return loss
                    else:
                        logits, _ = orig_forward(idx, None, kv_cache, train_mode=False)
                        return logits

                model.forward = syn_forward_wrapper
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
            if use_syn:
                model.forward = orig_forward
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        min_val_bpb = min(min_val_bpb, val_bpb)
        wandb_run.log(
            {
                "step": step,
                "total_training_flops": flops_so_far,
                "total_training_time": total_training_time,
                "val/bpb": val_bpb,
            }
        )
        telemetry.log_eval(step, val_bpb=float(val_bpb), min_val_bpb=float(min_val_bpb))
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    results = {}
    if not replaying_checkpoint_boundary and core_metric_every > 0 and (
        last_step or (step > 0 and step % core_metric_every == 0)
    ):
        model.eval()
        with autocast_ctx:
            results = evaluate_model(
                orig_model, tokenizer, device, max_per_task=core_metric_max_per_task
            )
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log(
            {
                "step": step,
                "total_training_flops": flops_so_far,
                "core_metric": results["core_metric"],
                "centered_results": results["centered_results"],
            }
        )
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if (
        not replaying_checkpoint_boundary
        and master_process
        and (last_step or (step > 0 and step % sample_every == 0))
    ):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer)  # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample, _ = engine.generate_batch(
                    tokens, num_samples=1, max_tokens=16, temperature=0
                )
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if not replaying_checkpoint_boundary and (
        last_step
        or (step > 0 and save_every > 0 and step % save_every == 0)
    ):
        if ddp_world_size > 1 and torch.distributed.is_initialized():
            gathered_loaders = [None for _ in range(ddp_world_size)]
            torch.distributed.all_gather_object(gathered_loaders, dataloader_state_dict)
            collated_loader_state = collate_dataloader_state_dicts(
                [g for g in gathered_loaders if g is not None],
                world_size=ddp_world_size,
            )
        else:
            collated_loader_state = dataloader_state_dict

        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),  # model parameters
            [opt.state_dict() for opt in optimizers],  # optimizer states
            {  # metadata saved as json
                "step": step,
                "val_bpb": val_bpb,  # loss at last step
                "model_config": checkpoint_model_config(orig_model, model_config_kwargs),
                "synapses": use_syn,  # mark if this is a synaptic model
                # vg9.6: persist the full bio kinetics + provenance so the model round-trips
                # exactly (build_model used to silently rebuild with SynapticConfig() defaults).
                "synaptic_config": synaptic_config_to_meta(syn_cfg) if use_syn else None,
                "splitmerge": (
                    {"every": int(splitmerge_every), "config": asdict(sm_cfg)}
                    if sm_ctrl is not None and sm_cfg is not None
                    else None
                ),
                "provenance": config_provenance(syn_cfg) if use_syn else None,
                "user_config": user_config,  # inputs to the training script
                "device_batch_size": device_batch_size,
                "total_batch_size": total_batch_size,
                "max_seq_len": max_seq_len,
                "dataloader_state_dict": collated_loader_state,
                "loop_state": {  # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
            # hwxb.2.6: per-rank RNG so a resumed run is bit-comparable (the synaptic
            # forward is stochastic during training; without this a resume diverges).
            train_state={
                "rng": capture_rng_state(),
                "step": step,
                # DDP synchronizes gradients, not rank-local forward mutations. Bio
                # buffers and online-updated weights may therefore legitimately differ
                # between ranks and must not all resume from rank 0's model artifact.
                "rank_model_state": (
                    orig_model.state_dict() if ddp_world_size > 1 else None
                ),
                "splitmerge": sm_ctrl.state_dict() if sm_ctrl is not None else None,
                "neuromod": nm_bus.state_dict() if nm_bus is not None else None,
                "divergence_guard": divguard.state_dict(),
                "dataloader_state_dict": dataloader_state_dict,
                "prefetched_batch": capture_prefetched_batch(x, y),
            },
        )

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            result = model(x, y, train_mode=True) if use_syn else model(x, y)
            if isinstance(result, tuple):
                logits, loss = result
            else:
                loss = result
        train_loss = loss.detach()  # for logging
        loss = (
            loss / grad_accum_steps
        )  # each .backward() is a grad sum => normalize loss here
        loss.backward()
        x, y, dataloader_state_dict = next(
            train_loader
        )  # prefetch the next batch while the GPU is busy with forward/backward
    # vg9.7: divergence guard — inspect the loss + bio buffers before applying the gradient.
    _guard = divguard.check(train_loss, orig_model, step=step)
    divguard.log(_guard, step)
    skip_optimizer_step = _guard.action in (GuardAction.SKIP, GuardAction.ROLLBACK)
    if _guard.action == GuardAction.ROLLBACK and divguard.can_rollback():
        divguard.rollback(orig_model, optimizers)
    # gradient clipping
    grad_clip_enabled = grad_clip > 0.0
    if grad_clip_enabled:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            orig_model.parameters(), grad_clip
        )
        grad_norm = (
            grad_norm_tensor.item()
        )  # GPU tensor -> CPU float (note: cpu-gpu sync point)
    # step the optimizers
    lrm = get_lr_multiplier(step)
    if _guard.action == GuardAction.BACKOFF:
        lrm = lrm * divguard.cfg.backoff_factor  # vg9.7: gentler step on a loss spike
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    if muon_optimizer is not None:
        muon_momentum = get_muon_momentum(step)
        for group in muon_optimizer.param_groups:
            group["momentum"] = muon_momentum
    if not skip_optimizer_step:
        for opt in optimizers:
            opt.step()
        divguard.maybe_snapshot(orig_model, optimizers, step)  # vg9.7: refresh last-good (opt-in)
    model.zero_grad(set_to_none=True)

    # Split/merge controller step
    if sm_ctrl is not None:
        # Pass ALL optimizers so split/merge resets stale momentum wherever the changed
        # expert/router weights live: AdamW (1D/embeddings) AND Muon (2D matrices). vg9.3.
        sm_ctrl.step(step, optimizer=optimizers)

    # Neuromodulatory bus (hy8.1): compute DA/ACh/NE from this step's loss + predictive entropy
    # and broadcast the gains so they gate the NEXT step's plasticity/exploration/gain.
    if nm_bus is not None:
        # use_syn guarantees the model returned (logits, loss), so logits is defined here.
        nm_entropy = nm_bus.entropy_from_logits(logits)
        nm_bus.update(loss=float(train_loss), entropy=nm_entropy)
        nm_bus.broadcast(orig_model)
        if neuromod_log_every and step % neuromod_log_every == 0:
            tel = nm_bus.telemetry()
            telemetry.log_bio(step, neuromod=tel)
            print0(
                f"  [neuromod] DA={tel['nm/da']:+.3f} ACh={tel['nm/ach']:.3f} NE={tel['nm/ne']:.3f} "
                f"| gains: plast={tel['nm/gain_plasticity']:.3f} explore={tel['nm/gain_explore']:.3f} "
                f"glob={tel['nm/gain_global']:.3f}"
            )
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging
    ema_beta = 0.9  # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = (
        ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    )  # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (
        1 - ema_beta ** (step + 1)
    )  # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec_h100 = (
        989e12 * ddp_world_size
    )  # bfloat16 H100 SXM and without 2:4 sparsity
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100  # in %
    if step > 10:
        total_training_time += dt  # only count the time after the first 10 steps
    print_grad_norm = f" grad norm: {grad_norm:.4f} |" if grad_clip_enabled else ""
    print0(
        f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} |{print_grad_norm} lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time / 60:.2f}m"
    )
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        }
        if grad_clip_enabled:
            log_data["train/grad_norm"] = grad_norm
        wandb_run.log(log_data)
        telemetry.log_train_step(
            step,
            loss=debiased_smooth_loss,
            lr=lrm,
            grad_norm=(grad_norm if grad_clip_enabled else None),
            tok_per_sec=tok_per_sec,
            step_ms=dt * 1000.0,
            mfu=mfu,
            vram_gb=get_max_memory() / 1024**3,
            divergence_action=_guard.action.name,
        )

    if viz is not None:
        # Use orig_model for visualization to avoid compilation artifacts if any
        viz.step(orig_model, step, loss=train_loss)

    # state update
    step += 1

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time / 60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# Log to report
get_report().log(
    section="Base model training",
    data=[
        user_config,  # CLI args
        {  # stats about the training setup
            "Number of parameters": num_params,
            "Number of FLOPs per token": f"{num_flops_per_token:e}",
            "Calculated number of iterations": num_iterations,
            "Number of training tokens": total_tokens,
            "Tokens : Params ratio": total_batch_size * num_iterations / num_params,
            "DDP world size": ddp_world_size,
            "warmup_ratio": warmup_ratio,
            "warmdown_ratio": warmdown_ratio,
            "final_lr_frac": final_lr_frac,
        },
        {  # stats about training outcomes
            "Minimum validation bpb": min_val_bpb,
            "Final validation bpb": val_bpb,
            "CORE metric estimate": results.get("core_metric", None),
            "MFU %": f"{mfu:.2f}%",
            "Total training flops": f"{flops_so_far:e}",
            "Total training time": f"{total_training_time / 60:.2f}m",
            "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
        },
    ],
)

# hm4.1: emit a provenance-stamped, schema-valid run record to the committed results registry
# (rank 0 only; non-fatal so a bad record can never break the run's cleanup).
if ddp_rank == 0:
    try:
        from bio_inspired_nanochat.results_registry import append_record, make_record

        _metric_candidates = {
            "val_bpb": val_bpb,
            "smooth_train_loss": smooth_train_loss,
            "mfu": mfu,
            "total_training_time": total_training_time,
            "step": step,
        }
        _metrics = {
            name: float(value)
            for name, value in _metric_candidates.items()
            if math.isfinite(float(value))
        }
        _dataset_shards = [
            f"fineweb:train:pq{dataloader_state_dict['pq_idx']}:rg{dataloader_state_dict['rg_idx']}",
            "fineweb:val",
        ]
        _rec = make_record(
            "train",
            _metrics,
            run_id=str(telemetry.run_id),
            config={
                "model": asdict(model_config),
                "training": user_config,
                "resolved_num_iterations": int(num_iterations),
                "world_size": int(ddp_world_size),
            },
            seed=int(init_seed),
            dataset_shards=_dataset_shards,
            timestamp=time.time(),
            notes=f"base_train; telemetry=runs/telemetry/{output_dirname}/events.jsonl",
        )
        append_record(_rec)
        print0(f"[results] appended run record {_rec.run_id} to the registry")
    except Exception as exc:  # pragma: no cover - depends on a full training run
        print0(f"[results] failed to record run (non-fatal): {exc}")

# cleanup
if viz is not None:
    viz.close()
telemetry.close()  # hwxb.7.1: flush + close the unified telemetry (TB + JSONL)
wandb_run.finish()  # wandb run finish
compute_cleanup()
