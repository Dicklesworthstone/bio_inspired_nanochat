"""Retrofit a compatible pretrained Nanochat GPT checkpoint with synaptic dynamics.

The original script only initialized a fresh random ``GPTSynaptic`` despite being documented as
an injector.  This implementation performs the actual conversion: embeddings, output head,
attention matrices, and dense MLP matrices are copied from a real vanilla checkpoint; pretrained
MLP matrices become ``w_slow`` while all fast weights and online traces start at zero.  Optional
MoE conversion clones the dense pretrained MLP into every expert so routing starts from
functionally equivalent experts rather than random ones.

This is intentionally the architecture-compatible Nanochat converter.  Arbitrary Hugging Face
architectures require per-family adapters and are tracked by the downstream cross-architecture
adapter task rather than guessed here.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
from rich.console import Console

from bio_inspired_nanochat.checkpoint_manager import (
    checkpoint_model_config,
    config_hash,
    config_provenance,
    list_checkpoint_steps,
    load_checkpoint,
    save_checkpoint,
    synaptic_config_to_meta,
)
from bio_inspired_nanochat.gpt import GPTConfig
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.synaptic import (
    SynapticCausalSelfAttention,
    SynapticConfig,
    SynapticLinear,
    SynapticMLP,
    SynapticMoE,
)


@dataclass
class RetrofitReport:
    source_checkpoint: str
    source_step: int
    copied_tensors: int
    copied_elements: int
    expert_copies: int
    use_moe: bool
    finetune_steps: int = 0
    initial_loss: float | None = None
    final_loss: float | None = None
    dynamics_active: bool = False


def _source_config(meta_data: dict[str, Any]) -> GPTConfig:
    raw = meta_data.get("model_config")
    if not isinstance(raw, dict):
        raise ValueError("source checkpoint metadata must contain a model_config object")
    required = {"sequence_len", "vocab_size", "n_layer", "n_head", "n_embd"}
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"source model_config is missing required fields: {missing}")
    known = {field.name for field in fields(GPTConfig)}
    config = GPTConfig(**{key: value for key, value in raw.items() if key in known})
    if config.attention_type != "standard":
        raise ValueError(
            "synaptic retrofit currently requires standard Nanochat attention; "
            f"got attention_type={config.attention_type!r}"
        )
    return config


def build_synaptic(
    source_config: GPTConfig,
    *,
    syn_cfg: SynapticConfig | None = None,
    use_moe: bool = False,
    num_experts: int = 8,
    top_k: int = 2,
    device: torch.device | str = "cpu",
) -> GPTSynaptic:
    """Build and fully initialize the synaptic destination for a compatible GPT config."""
    if num_experts < 1:
        raise ValueError("num_experts must be at least 1")
    if not 1 <= top_k <= num_experts:
        raise ValueError("top_k must be in [1, num_experts]")
    target_config = GPTSynapticConfig(
        sequence_len=source_config.sequence_len,
        vocab_size=source_config.vocab_size,
        n_layer=source_config.n_layer,
        n_head=source_config.n_head,
        n_kv_head=source_config.n_kv_head,
        n_embd=source_config.n_embd,
        synapses=True,
        syn_cfg=syn_cfg or SynapticConfig(),
        dropout=0.0,
        use_moe=use_moe,
        num_experts=num_experts,
        moe_top_k=top_k,
        moe_hidden_mult=4,
        init_type=source_config.init_type,
        init_seed=source_config.init_seed,
        tie_embeddings=source_config.tie_embeddings,
    )
    with torch.device("meta"):
        model = GPTSynaptic(target_config)
    model.to_empty(device=torch.device(device))
    model.init_weights()
    return model


def _copy_tensor(
    destination: torch.Tensor,
    source: torch.Tensor,
    *,
    name: str,
    transpose: bool = False,
) -> int:
    value = source.transpose(0, 1) if transpose else source
    if destination.shape != value.shape:
        raise ValueError(
            f"shape mismatch for {name}: source {tuple(value.shape)} != "
            f"destination {tuple(destination.shape)}"
        )
    destination.copy_(value.to(device=destination.device, dtype=destination.dtype))
    return destination.numel()


def _zero_fast_state(module: SynapticLinear) -> None:
    if module.w_fast is not None:
        module.w_fast.zero_()
    if module.bias is not None:
        module.bias.zero_()
    if module.u_buf is not None:
        module.u_buf.zero_()
    if module.v_buf is not None:
        module.v_buf.zero_()
    module._plasticity_pending = False
    module._last_gate_scale = None
    if module.post is not None:
        module.post.fast.zero_()
        module.post.slow.zero_()
        module.post.camkii.zero_()
        module.post.pp1.fill_(0.5)
        module.post.bdnf.zero_()
        module.post.bdnf_hebb_accum.zero_()
        module.post._last_hebb_delta_mag.zero_()
    if module.input_ln is not None:
        module.input_ln.weight.fill_(1.0)
        module.input_ln.bias.zero_()


@torch.no_grad()
def inject_pretrained_weights(
    model: GPTSynaptic, source_state: dict[str, torch.Tensor]
) -> tuple[int, int, int]:
    """Copy compatible vanilla GPT tensors into a prepared synaptic model.

    Returns ``(copied_tensor_count, copied_element_count, expert_copy_count)``.
    """
    state = {key.removeprefix("_orig_mod."): value for key, value in source_state.items()}

    def get(name: str) -> torch.Tensor:
        value = state.get(name)
        if value is None:
            raise ValueError(f"source checkpoint is missing required tensor {name!r}")
        return value

    copied_tensors = 0
    copied_elements = 0
    expert_copies = 0

    def copy(destination: torch.Tensor, source_name: str, *, transpose: bool = False) -> None:
        nonlocal copied_tensors, copied_elements
        copied_elements += _copy_tensor(
            destination, get(source_name), name=source_name, transpose=transpose
        )
        copied_tensors += 1

    copy(model.wte.weight, "transformer.wte.weight")
    copy(model.lm_head.weight, "lm_head.weight")

    for layer_idx, block in enumerate(model.h):
        prefix = f"transformer.h.{layer_idx}"
        attention = block.attn.attn
        if not isinstance(attention, SynapticCausalSelfAttention):
            raise TypeError(f"layer {layer_idx} does not contain synaptic attention")
        copy(attention.q_proj.weight, f"{prefix}.attn.c_q.weight")
        copy(attention.k_proj.weight, f"{prefix}.attn.c_k.weight")
        copy(attention.v_proj.weight, f"{prefix}.attn.c_v.weight")
        copy(attention.o_proj.weight, f"{prefix}.attn.c_proj.weight")
        block.norm1.weight.fill_(1.0)
        block.norm1.bias.zero_()
        block.norm2.weight.fill_(1.0)
        block.norm2.bias.zero_()

        linears: list[tuple[SynapticLinear, SynapticLinear]] = []
        if isinstance(block.mlp, SynapticMoE):
            linears.extend((expert.fc1, expert.fc2) for expert in block.mlp.experts)
            expert_copies += len(block.mlp.experts)
            block.mlp.router.weight.zero_()
        else:
            dense = getattr(block.mlp, "mlp", None)
            if not isinstance(dense, SynapticMLP):
                raise TypeError(f"layer {layer_idx} does not contain a compatible synaptic MLP")
            linears.append((dense.fc, dense.proj))

        for fc, proj in linears:
            copy(fc.w_slow, f"{prefix}.mlp.c_fc.weight", transpose=True)
            copy(proj.w_slow, f"{prefix}.mlp.c_proj.weight", transpose=True)
            _zero_fast_state(fc)
            _zero_fast_state(proj)

    model.tie_weights()
    return copied_tensors, copied_elements, expert_copies


def smoke_finetune(
    model: GPTSynaptic,
    *,
    steps: int,
    learning_rate: float,
    seed: int,
) -> tuple[float, float, bool]:
    """Run a deterministic synthetic finetune smoke and prove bio state is active."""
    if steps < 1:
        raise ValueError("finetune steps must be at least 1")
    if learning_rate <= 0.0:
        raise ValueError("finetune learning rate must be positive")
    device = model.get_device()
    generator = torch.Generator(device=device).manual_seed(seed)
    sequence_len = min(16, model.config.sequence_len)
    if sequence_len < 2:
        raise ValueError("source sequence length must be at least 2 for finetuning")
    tokens = torch.randint(
        0,
        model.config.vocab_size,
        (2, sequence_len + 1),
        generator=generator,
        device=device,
    )
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    tracked = [module for module in model.modules() if isinstance(module, SynapticLinear)]
    before = [module.u_buf.detach().clone() for module in tracked if module.u_buf is not None]
    losses: list[float] = []
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets=targets, train_mode=True)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("retrofit finetune produced a non-finite loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    after = [module.u_buf.detach() for module in tracked if module.u_buf is not None]
    dynamics_active = any(not torch.equal(old, new) for old, new in zip(before, after))
    if not dynamics_active:
        raise RuntimeError("retrofit finetune did not activate any Hebbian eligibility trace")
    model.reset_sequence_state(reset_fast_weights=False, reset_consolidation=False)
    model.eval()
    with torch.no_grad():
        logits, _ = model(inputs, train_mode=False)
    if not torch.isfinite(logits).all():
        raise RuntimeError("retrofitted model produced non-finite evaluation logits")
    return losses[0], losses[-1], dynamics_active


def retrofit_checkpoint(
    source_checkpoint: str | Path,
    output_checkpoint: str | Path,
    *,
    source_step: int = -1,
    syn_cfg: SynapticConfig | None = None,
    use_moe: bool = False,
    num_experts: int = 8,
    top_k: int = 2,
    finetune_steps: int = 0,
    finetune_lr: float = 1e-4,
    finetune_seed: int = 1337,
    device: torch.device | str = "cpu",
) -> tuple[GPTSynaptic, RetrofitReport]:
    """Convert one vanilla Nanochat checkpoint and save a loadable synaptic checkpoint."""
    source_dir = Path(source_checkpoint).resolve()
    output_dir = Path(output_checkpoint).resolve()
    if output_dir == source_dir:
        raise ValueError("output checkpoint directory must differ from the source directory")
    steps = list_checkpoint_steps(str(source_dir))
    if not steps:
        raise FileNotFoundError(f"no model checkpoints found in {source_dir}")
    resolved_step = steps[-1] if source_step < 0 else source_step
    if resolved_step not in steps:
        raise FileNotFoundError(
            f"checkpoint step {resolved_step} not found in {source_dir}; available={steps}"
        )
    model_path = output_dir / "model_000000.pt"
    meta_path = output_dir / "meta_000000.json"
    if model_path.exists() or meta_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing retrofit checkpoint in {output_dir}"
        )

    source_state, _, source_meta = load_checkpoint(
        str(source_dir), resolved_step, torch.device("cpu"), load_optimizer=False
    )
    if source_meta.get("synapses", False):
        raise ValueError("source checkpoint is already synaptic; retrofit requires vanilla GPT")
    source_config = _source_config(source_meta)
    active_syn_cfg = syn_cfg or SynapticConfig()
    model = build_synaptic(
        source_config,
        syn_cfg=active_syn_cfg,
        use_moe=use_moe,
        num_experts=num_experts,
        top_k=top_k,
        device=device,
    )
    copied_tensors, copied_elements, expert_copies = inject_pretrained_weights(
        model, source_state
    )
    report = RetrofitReport(
        source_checkpoint=str(source_dir),
        source_step=resolved_step,
        copied_tensors=copied_tensors,
        copied_elements=copied_elements,
        expert_copies=expert_copies,
        use_moe=use_moe,
    )
    if finetune_steps:
        initial_loss, final_loss, dynamics_active = smoke_finetune(
            model,
            steps=finetune_steps,
            learning_rate=finetune_lr,
            seed=finetune_seed,
        )
        report.finetune_steps = finetune_steps
        report.initial_loss = initial_loss
        report.final_loss = final_loss
        report.dynamics_active = dynamics_active

    base_model_config = {
        "sequence_len": model.config.sequence_len,
        "vocab_size": model.config.vocab_size,
        "n_layer": model.config.n_layer,
        "n_head": model.config.n_head,
        "n_kv_head": model.config.n_kv_head,
        "n_embd": model.config.n_embd,
    }
    metadata = {
        "synapses": True,
        "model_config": checkpoint_model_config(model, base_model_config),
        "synaptic_config": synaptic_config_to_meta(active_syn_cfg),
        "provenance": config_provenance(active_syn_cfg),
        "retrofit": {
            **asdict(report),
            "source_model_config_hash": config_hash(source_meta["model_config"]),
            "source_provenance": source_meta.get("provenance"),
        },
    }
    save_checkpoint(
        checkpoint_dir=str(output_dir),
        step=0,
        model_data=model.state_dict(),
        optimizer_data=None,
        meta_data=metadata,
    )
    return model, report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inject synaptic dynamics into a compatible pretrained Nanochat GPT checkpoint"
    )
    parser.add_argument("--source-ckpt", required=True, help="vanilla checkpoint directory")
    parser.add_argument(
        "--source-step", type=int, default=-1, help="source step; -1 selects the latest"
    )
    parser.add_argument("--ckpt-out", required=True, help="new synaptic checkpoint directory")
    parser.add_argument("--use-moe", action="store_true", help="clone the dense MLP into MoE experts")
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument(
        "--finetune-steps",
        type=int,
        default=0,
        help="optional deterministic synthetic smoke-finetune steps",
    )
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="conversion and optional smoke-finetune device",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    console = Console()
    model, report = retrofit_checkpoint(
        args.source_ckpt,
        args.ckpt_out,
        source_step=args.source_step,
        use_moe=args.use_moe,
        num_experts=args.experts,
        top_k=args.topk,
        finetune_steps=args.finetune_steps,
        finetune_lr=args.finetune_lr,
        finetune_seed=args.seed,
        device=device,
    )
    console.print(
        "[bold green]Synaptic retrofit complete[/bold green] "
        f"source_step={report.source_step} copied_tensors={report.copied_tensors} "
        f"copied_elements={report.copied_elements:,} experts={report.expert_copies}"
    )
    if report.finetune_steps:
        console.print(
            "[cyan]Smoke finetune[/cyan] "
            f"steps={report.finetune_steps} loss={report.initial_loss:.6f}→"
            f"{report.final_loss:.6f} dynamics_active={report.dynamics_active}"
        )
    console.print(f"[dim]FLOPs estimate: {model.estimate_flops():,}[/dim]")
    console.print(f"[bold]Saved:[/bold] {Path(args.ckpt_out).resolve()}")


if __name__ == "__main__":
    main()
