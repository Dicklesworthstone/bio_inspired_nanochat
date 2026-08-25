"""
Midtrain the model. Same as pretraining but simpler.
Run as:

python -m scripts.mid_train

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --device_batch_size=16
"""

import os
import time
from contextlib import nullcontext
from typing import Any, Protocol, cast

import torch
import torch.distributed as torch_dist
import torch.nn.functional as F
import wandb

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from bio_inspired_nanochat.checkpoint_manager import (
    capture_prefetched_batch,
    capture_rng_state,
    checkpoint_model_config,
    config_provenance,
    find_largest_model,
    load_model,
    load_rank_training_checkpoint,
    restore_optimizer_states,
    restore_prefetched_batch,
    restore_rank_model_state,
    restore_rng_state,
    save_checkpoint,
    synaptic_config_to_meta,
    validate_exact_resume_payload_step,
)
from bio_inspired_nanochat.dataloader import (
    collate_dataloader_state_dicts,
    extract_rank_dataloader_state,
    tokenizing_task_data_loader_with_state,
)
from bio_inspired_nanochat.common import (
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    print0,
)
from bio_inspired_nanochat.loss_eval import evaluate_bpb
from bio_inspired_nanochat.tokenizer import get_token_bytes
from tasks.common import TaskMixture
from tasks.customjson import CustomJSON
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.spellingbee import SimpleSpelling, SpellingBee


class _ReduceOpApi(Protocol):
    MAX: object


class _DistributedApi(Protocol):
    ReduceOp: _ReduceOpApi

    def is_initialized(self) -> bool: ...

    def all_reduce(self, tensor: torch.Tensor, *, op: object) -> None: ...

    def all_gather_object(self, object_list: list[Any], obj: object) -> None: ...


dist = cast(_DistributedApi, torch_dist)

# -----------------------------------------------------------------------------
run = "dummy" # wandb run name default ("dummy" is special - we won't log to wandb)
device_type = "" # cuda|cpu|mps (empty => autodetect)
model_tag = None # model tag to load the model from (base model or midtrained model)
step = None # step to load the model from (base model or midtrained model)
resume_model_tag = None # mid-checkpoint model tag to resume (None => infer largest)
resume_from_step = -1 # exact mid-training checkpoint step (-1 => fresh run)
dtype = "bfloat16"
synapses = 0 # use synaptic model (GPTSynaptic) if 1, otherwise use standard GPT (note: model loaded from checkpoint will auto-detect)
num_iterations = -1 # explicit number of steps of the optimization (-1 = disable)
max_seq_len = 2048
device_batch_size = 32
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
init_lr_frac = 1.0 # initial learning rate is this fraction of the base learning rate
weight_decay = 0.0
eval_every = 150 # -1 = disable
eval_tokens = 20*524288
total_batch_size = 524288
dry_run = 0 # dry_run=1 is for experiments: we will log to wandb but we won't write checkpoints or report
save_every = -1 # periodic optimizer-step checkpoint interval (-1 => final only)
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
with open(os.path.join("bio_inspired_nanochat", "configurator.py"), encoding="utf-8") as f:
    exec(f.read())  # nosec B102 # noqa: S102 # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys} # possibly useful for logging
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-mid", name=run, config=user_config)

# Load the source model or reconstruct the exact mid-training checkpoint.
if isinstance(resume_from_step, bool) or not isinstance(resume_from_step, int):
    raise TypeError("resume_from_step must be an integer")
resuming = resume_from_step >= 0
base_dir = get_base_dir()
mid_checkpoints_dir = os.path.join(base_dir, "mid_checkpoints")
resolved_resume_tag = resume_model_tag
dataloader_resume_state_dict: dict[str, Any] | None = None
checkpoint_meta: dict[str, Any] | None = None
optimizer_data = None
train_state = None
if resuming:
    if step is not None:
        raise ValueError("step selects a base checkpoint and cannot be combined with resume_from_step")
    if resolved_resume_tag is None:
        resolved_resume_tag = find_largest_model(mid_checkpoints_dir)
    model, tokenizer, meta = load_model(
        "mid",
        device,
        phase="train",
        model_tag=resolved_resume_tag,
        step=resume_from_step,
    )
else:
    model, tokenizer, meta = load_model(
        "base",
        device,
        phase="train",
        model_tag=model_tag,
        step=step,
    )

depth = int(model.config.n_layer)
output_dirname = str(resolved_resume_tag) if resuming else f"d{depth}"
checkpoint_dir = os.path.join(mid_checkpoints_dir, output_dirname)
if resuming:
    optimizer_data, checkpoint_meta, train_state = load_rank_training_checkpoint(
        checkpoint_dir,
        resume_from_step,
        device,
        rank=ddp_rank,
        expected_world_size=ddp_world_size,
    )
    validate_exact_resume_payload_step(
        checkpoint_meta,
        resume_from_step,
        payload_name="checkpoint metadata",
    )
    validate_exact_resume_payload_step(
        train_state,
        resume_from_step,
        payload_name="rank-local training state",
    )
    saved_user_config = checkpoint_meta.get("user_config")
    if not isinstance(saved_user_config, dict):
        raise ValueError("exact mid-training resume requires saved user_config metadata")
    trajectory_fields = (
        "max_seq_len",
        "device_batch_size",
        "total_batch_size",
        "num_iterations",
        "unembedding_lr",
        "embedding_lr",
        "matrix_lr",
        "init_lr_frac",
        "weight_decay",
        "eval_every",
        "eval_tokens",
        "save_every",
    )
    missing_fields = [name for name in trajectory_fields if name not in saved_user_config]
    if missing_fields:
        raise ValueError(
            "exact mid-training resume is missing trajectory configuration: "
            f"{missing_fields}"
        )
    max_seq_len = int(saved_user_config["max_seq_len"])
    device_batch_size = int(saved_user_config["device_batch_size"])
    total_batch_size = int(saved_user_config["total_batch_size"])
    num_iterations = int(saved_user_config["num_iterations"])
    unembedding_lr = float(saved_user_config["unembedding_lr"])
    embedding_lr = float(saved_user_config["embedding_lr"])
    matrix_lr = float(saved_user_config["matrix_lr"])
    init_lr_frac = float(saved_user_config["init_lr_frac"])
    weight_decay = float(saved_user_config["weight_decay"])
    eval_every = int(saved_user_config["eval_every"])
    eval_tokens = int(saved_user_config["eval_tokens"])
    save_every = int(saved_user_config["save_every"])
    dataloader_resume_state_dict = extract_rank_dataloader_state(
        checkpoint_meta.get("dataloader_state_dict"),
        rank=ddp_rank,
        world_size=ddp_world_size,
    )
    restore_rank_model_state(model, train_state, required=True)
else:
    pretrain_batch_size = meta.get("device_batch_size")
    if pretrain_batch_size is not None and device_batch_size > pretrain_batch_size:
        print0(
            "FOOTGUN WARNING: base model training used device_batch_size "
            f"{pretrain_batch_size}; verify --device_batch_size for mid-training"
        )

orig_model = model
model = torch.compile(model, dynamic=False)
use_syn = bool(getattr(model.config, "synapses", False))
num_flops_per_token = model.estimate_flops()
tokens_per_fwdbwd = device_batch_size * max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
if total_batch_size % world_tokens_per_fwdbwd != 0:
    raise ValueError(f"total_batch_size {total_batch_size} must be divisible by world_tokens_per_fwdbwd {world_tokens_per_fwdbwd}")
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
token_bytes = get_token_bytes(device=device)

# Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
optimizers = model.setup_optimizers(unembedding_lr=unembedding_lr, embedding_lr=embedding_lr, matrix_lr=matrix_lr, weight_decay=weight_decay)
if len(optimizers) == 2:
    adamw_optimizer, muon_optimizer = optimizers
else:
    adamw_optimizer = optimizers[0]
    muon_optimizer = None
# Override the initial learning rate as a fraction of the base learning rate
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * init_lr_frac
        group["initial_lr"] = group["lr"] # save the initial learning so we can decay easily later
if resuming:
    restore_optimizer_states(optimizers, optimizer_data)
    del optimizer_data

# Midtraining data mixture and DataLoader
identity_conversations_filepath = os.path.join(base_dir, "identity_conversations.jsonl")
train_dataset = TaskMixture([
    SmolTalk(split="train"), # 460K rows of general conversations
    MMLU(subset="auxiliary_train", split="train"), # 100K rows of multiple choice problems drawn from ARC, MC_TEST, OBQA, RACE
    GSM8K(subset="main", split="train"), # 8K rows teaching simple math and (calculator) tool use
    CustomJSON(filepath=identity_conversations_filepath), # 1000 rows of synthetic identity conversations
    CustomJSON(filepath=identity_conversations_filepath), # let's do 2 epochs of these
    SimpleSpelling(size=200000, split="train"), # 200K rows of Simple Spelling (e.g. spell the word 'apple')
    SpellingBee(size=80000, split="train"), # 80K rows of Spelling Bee (e.g. how many 'r' are in 'strawberry'?)
]) # total: 460K + 100K + 8K + 200K + 80K = 848K rows
val_dataset = TaskMixture([
    SmolTalk(split="test"), # 24K rows in test set
    MMLU(subset="all", split="test", stop=5200), # 14K rows in test set, use only 5.2K to match the train ratios
    GSM8K(subset="main", split="test", stop=420), # 1.32K rows in test set, use only 420 to match the train ratios
]) # total: 24K + 14K + 1.32K ~= 39K rows

# DataLoader setup using exact stateful TaskMixture loader
train_loader = tokenizing_task_data_loader_with_state(
    train_dataset,
    tokenizer,
    device_batch_size,
    max_seq_len,
    device=device,
    resume_state_dict=dataloader_resume_state_dict,
)


def build_val_loader():
    for val_inputs, val_targets, _state in tokenizing_task_data_loader_with_state(
        val_dataset,
        tokenizer,
        device_batch_size,
        max_seq_len,
        device=device,
    ):
        yield val_inputs, val_targets


if resuming:
    if dataloader_resume_state_dict is None or train_state is None:
        raise ValueError("exact resume requires rank-local dataloader state and train state")
    x, y = restore_prefetched_batch(
        train_state,
        device=device,
        expected_shape=(device_batch_size, max_seq_len),
    )
    dataloader_state_dict = dataloader_resume_state_dict
    restore_rng_state(train_state.get("rng"))
    print0(
        f"[checkpoint] restored RNG and prefetched batch for bit-comparable resume from step {resume_from_step}"
    )
else:
    x, y, dataloader_state_dict = next(train_loader)

if resuming and checkpoint_meta is not None:
    step = int(checkpoint_meta["step"])
    val_bpb = float(checkpoint_meta.get("val_bpb", float("inf")))
    loop_state = checkpoint_meta.get("loop_state", {})
    min_val_bpb = float(loop_state.get("min_val_bpb", val_bpb))
    smooth_train_loss = float(loop_state.get("smooth_train_loss", 0.0))
    total_training_time = float(loop_state.get("total_training_time", 0.0))
else:
    step = 0
    val_bpb = float("inf")
    min_val_bpb = float("inf")
    smooth_train_loss = 0.0
    total_training_time = 0.0

ema_beta = 0.9
train_dataset_size = len(train_dataset)
progress = 0.0


# Learning rate scheduler
def get_lr_multiplier(prog):
    # first 80% of training: no decay, then linearly ramp down to 0.
    return 1.0 if prog < 0.8 else max(0.0, 1.0 - (prog - 0.8) / 0.2)


# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum


# -----------------------------------------------------------------------------
# Training loop
while True:
    flops_so_far = num_flops_per_token * total_batch_size * step
    replaying_checkpoint_boundary = resuming and step == resume_from_step

    if num_iterations > 0:
        last_step = step == num_iterations
    else:
        last_step = bool(
            dataloader_state_dict is not None
            and int(dataloader_state_dict.get("epochs_completed", 0)) >= 1
        )

    # Synchronize last_step across all ranks to avoid hangs in the distributed setting
    if ddp and dist.is_initialized():
        last_step_tensor = torch.tensor(int(last_step), dtype=torch.int32, device=device)
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
        last_step = bool(last_step_tensor.item() > 0)

    # once in a while: evaluate the val bpb (all ranks participate)
    if not replaying_checkpoint_boundary and eval_every > 0 and (last_step or step % eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        eval_steps = max(1, eval_steps)
        with autocast_ctx:
            # Handle GPTSynaptic return signature for evaluate_bpb
            if hasattr(model, "config") and getattr(model.config, "synapses", False):
                orig_forward = model.forward

                def syn_forward_wrapper(
                    idx,
                    targets=None,
                    kv_cache=None,
                    loss_reduction="mean",
                    *,
                    _orig=orig_forward,
                    **kwargs,
                ):
                    if targets is not None:
                        logits, loss = _orig(idx, targets, kv_cache, train_mode=False)
                        if loss_reduction == "none":
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
                        logits, _ = _orig(idx, None, kv_cache, train_mode=False)
                        return logits

                model.forward = syn_forward_wrapper
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
            if hasattr(model, "config") and getattr(model.config, "synapses", False):
                model.forward = orig_forward
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        min_val_bpb = min(min_val_bpb, val_bpb)
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # save checkpoint at the end of the run or periodically (all ranks participate)
    if not replaying_checkpoint_boundary and not dry_run and (
        last_step or (step > 0 and save_every > 0 and step % save_every == 0)
    ):
        if ddp_world_size > 1 and dist.is_initialized():
            gathered_loaders: list[dict[str, Any] | None] = [
                None for _ in range(ddp_world_size)
            ]
            dist.all_gather_object(gathered_loaders, dataloader_state_dict)
            collated_loader_state = collate_dataloader_state_dicts(
                [g for g in gathered_loaders if g is not None],
                world_size=ddp_world_size,
            )
        else:
            collated_loader_state = dataloader_state_dict

        syn_cfg = getattr(orig_model.config, "syn_cfg", None)
        model_config_kwargs = {
            "sequence_len": max_seq_len,
            "vocab_size": tokenizer.get_vocab_size(),
            "n_layer": depth,
            "n_head": model.config.n_head,
            "n_kv_head": model.config.n_kv_head,
            "n_embd": model.config.n_embd,
        }
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            [opt.state_dict() for opt in optimizers],
            {
                "step": step,
                "val_bpb": val_bpb,
                "model_config": checkpoint_model_config(orig_model, model_config_kwargs),
                "synapses": use_syn,
                "synaptic_config": (
                    synaptic_config_to_meta(syn_cfg)
                    if use_syn and syn_cfg is not None
                    else None
                ),
                "provenance": (
                    config_provenance(syn_cfg)
                    if use_syn and syn_cfg is not None
                    else None
                ),
                "user_config": user_config,
                "device_batch_size": device_batch_size,
                "total_batch_size": total_batch_size,
                "max_seq_len": max_seq_len,
                "dataloader_state_dict": collated_loader_state,
                "loop_state": {
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
            train_state={
                "rng": capture_rng_state(),
                "step": step,
                "prefetched_batch": capture_prefetched_batch(x, y),
                "rank_model_state": (
                    orig_model.state_dict() if ddp_world_size > 1 else None
                ),
            },
        )

    if last_step:
        break

    if num_iterations > 0:
        progress = max(progress, step / num_iterations)
    else:
        docs_consumed = (
            int(dataloader_state_dict.get("documents_consumed", 0))
            if dataloader_state_dict
            else 0
        )
        progress = max(progress, min(1.0, docs_consumed / max(1, train_dataset_size)))

    # single training step
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            result = model(x, y, train_mode=True) if use_syn else model(x, y)
            if isinstance(result, tuple):
                logits, loss = result
            else:
                loss = result
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, dataloader_state_dict = next(train_loader)

    # step the optimizers
    lrm = get_lr_multiplier(progress)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    if muon_optimizer is not None:
        muon_momentum = get_muon_momentum(step)
        for group in muon_optimizer.param_groups:
            group["momentum"] = muon_momentum
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0

    step += 1

    # EMA loss, logging, etc.
    smooth_train_loss = (
        ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    )
    debiased_smooth_loss = (
        smooth_train_loss / (1 - ema_beta**step)
        if step > 0
        else train_loss.item()
    )
    pct_done = 100 * progress
    tok_per_sec = int(total_batch_size / dt) if dt > 0 else 0
    flops_per_sec = num_flops_per_token * total_batch_size / dt if dt > 0 else 0.0
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100
    if step > 10:
        total_training_time += dt
    print0(
        f"step {step:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | "
        f"lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | "
        f"mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m"
    )
    if step % 10 == 0:
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        })

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# Log to report
if not dry_run:
    from bio_inspired_nanochat.report import get_report

    get_report().log(
        section="Midtraining",
        data=[
            user_config,  # CLI args
            {  # stats about the training setup
                "Number of iterations": step,
                "DDP world size": ddp_world_size,
            },
            {  # stats about training outcomes
                "Minimum validation bpb": min_val_bpb,
            },
        ],
    )

# cleanup
wandb_run.finish()
compute_cleanup()

