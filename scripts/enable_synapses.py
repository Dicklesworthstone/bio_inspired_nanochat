# scripts/enable_synapses.py
# Helper to create and checkpoint a synaptic model (with optional MoE)

import argparse
import torch
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.synaptic import SynapticConfig
from bio_inspired_nanochat.checkpoint_manager import save_checkpoint


def build_synaptic(
    depth: int = 20,
    vocab: int = 65536,
    seq: int = 2048,
    n_head: int | None = None,
    n_kv_head: int | None = None,
    dropout: float = 0.0,
    use_moe: bool = False,
    num_experts: int = 8,
    top_k: int = 2,
    hidden_mult: int = 4,
    lb_lambda: float = 0.01,
    structural_every: int = 0,
):
    if n_head is None:
        n_head = max(1, (depth * 64 + 127) // 128)
    if n_kv_head is None:
        n_kv_head = n_head
    syn_cfg = SynapticConfig()
    cfg = GPTSynapticConfig(
        sequence_len=seq,
        vocab_size=vocab,
        n_layer=depth,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_embd=depth * 64,
        synapses=True,
        syn_cfg=syn_cfg,
        dropout=dropout,
        use_moe=use_moe,
        num_experts=num_experts,
        moe_top_k=top_k,
        moe_hidden_mult=hidden_mult,
        moe_balance_loss=lb_lambda,
        structural_every=structural_every,
    )
    return GPTSynaptic(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=20)
    ap.add_argument("--vocab", type=int, default=65536)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--use-moe", action="store_true")
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--hidden-mult", type=int, default=4)
    ap.add_argument("--lb-lambda", type=float, default=0.01)
    ap.add_argument("--structural-every", type=int, default=0)
    ap.add_argument("--ckpt_out", type=str, default="base_checkpoints/synaptic_init")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_synaptic(
        args.depth,
        args.vocab,
        args.seq,
        dropout=args.dropout,
        use_moe=args.use_moe,
        num_experts=args.experts,
        top_k=args.topk,
        hidden_mult=args.hidden_mult,
        lb_lambda=args.lb_lambda,
        structural_every=args.structural_every,
    ).to(device)
    print("FLOPs estimate:", model.estimate_flops())
    save_checkpoint(
        checkpoint_dir=args.ckpt_out,
        step=0,
        model_data=model.state_dict(),
        optimizer_data=None,
        meta_data={
            "synapses": True,
            "model_config": {
                "sequence_len": args.seq,
                "vocab_size": args.vocab,
                "n_layer": args.depth,
                "n_head": model.config.n_head,
                "n_kv_head": model.config.n_kv_head,
                "n_embd": model.config.n_embd,
                "dropout": model.config.dropout,
                "use_moe": model.config.use_moe,
                "num_experts": model.config.num_experts,
                "moe_top_k": model.config.moe_top_k,
                "moe_hidden_mult": model.config.moe_hidden_mult,
                "moe_balance_loss": model.config.moe_balance_loss,
                "structural_every": model.config.structural_every,
            },
        },
    )


if __name__ == "__main__":
    main()
