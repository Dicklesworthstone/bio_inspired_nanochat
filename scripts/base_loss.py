"""
Loads a checkpoint, and:
- Evaluates the loss on a larger chunk of train/val splits
- Samples from the model

Example run as:
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss
"""
import os
from contextlib import nullcontext
from bio_inspired_nanochat.torch_imports import torch, F
from bio_inspired_nanochat.checkpoint_manager import load_model
from bio_inspired_nanochat.common import compute_init, print0, compute_cleanup, autodetect_device_type
from bio_inspired_nanochat.dataloader import tokenizing_distributed_data_loader
from bio_inspired_nanochat.tokenizer import get_token_bytes
from bio_inspired_nanochat.loss_eval import evaluate_bpb
from bio_inspired_nanochat.engine import Engine
from bio_inspired_nanochat.report import get_report

# Configuration
device_batch_size = 32
split_tokens = 20*524288  # number of tokens to evaluate per split
model_tag = None # optional model tag for the output directory name
model_step = None # optional model step for the output directory name
device_type = "" # cuda|cpu|mps (empty => autodetect)
with open(os.path.join('bio_inspired_nanochat', 'configurator.py')) as f:
    exec(f.read()) # nosec B102 # overrides from command line or config file

# Load the base model and the tokenizer
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=model_tag, step=model_step)
sequence_len = meta["model_config"]["sequence_len"] # could be arbitrary really
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

# Evaluate the loss on each split
tokens_per_step = device_batch_size * sequence_len * ddp_world_size
if split_tokens % tokens_per_step != 0:
    raise ValueError("split_tokens must be divisible by tokens_per_step")
steps = split_tokens // tokens_per_step
token_bytes = get_token_bytes(device=device)
bpb_results = {}
# Handle GPTSynaptic return signature for evaluate_bpb
is_synaptic = hasattr(model, 'config') and getattr(model.config, 'synapses', False)
if is_synaptic:
    orig_forward = model.forward
    def syn_forward_wrapper(idx, targets=None, kv_cache=None, loss_reduction='mean', **kwargs):
        if targets is not None:
            logits, loss = orig_forward(idx, targets, kv_cache, train_mode=False)
            if loss_reduction == 'none':
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = targets.view(-1)
                loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction='none', ignore_index=-1)
                return loss_per_token.view(targets.shape)
            return loss
        else:
            logits, _ = orig_forward(idx, None, kv_cache, train_mode=False)
            return logits
    model.forward = syn_forward_wrapper

for split_name in ["train", "val"]:
    loader = tokenizing_distributed_data_loader(device_batch_size, sequence_len, split_name, device=device)
    with autocast_ctx:
        bpb = evaluate_bpb(model, loader, steps, token_bytes)
    print0(f"{split_name} bpb: {bpb:.4f}")
    bpb_results[split_name] = bpb

# Restore original forward if we wrapped it
if is_synaptic:
    model.forward = orig_forward

# Master process also samples from the model
samples = []
if ddp_rank == 0:
    prompts = [
        "The capital of France is",
        "The chemical symbol of gold is",
        "If yesterday was Friday, then tomorrow will be",
        "The opposite of hot is",
        "The planets of the solar system are:",
        "My favorite color is",
        "If 5*x + 3 = 13, then x is",
    ]
    engine = Engine(model, tokenizer)
    for prompt in prompts:
        tokens = tokenizer(prompt, prepend="<|bos|>")
        with autocast_ctx:
            sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
        sample_str = tokenizer.decode(sample[0])
        print0(sample_str)
        samples.append(sample_str)

# Log to report
get_report().log(section="Base model loss", data=[
    {
        "train bpb": bpb_results["train"],
        "val bpb": bpb_results["val"],
    },
    {f"sample {i}": sample for i, sample in enumerate(samples)},
])

# Cleanup
compute_cleanup()
