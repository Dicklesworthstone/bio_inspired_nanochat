# scripts/verify_evolution.py
# -----------------------------------------------------------------------------
# Verification script for Bio-Inspired Nanochat
# Checks:
#   1. NeuroScore integration (Efficiency, Specialization)
#   2. Genetic Evolution (Xi parameter divergence)
#   3. Synaptic MoE routing and metabolism
# -----------------------------------------------------------------------------

import torch
import os
import shutil
import argparse

from typing import Dict, cast
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig, Block
from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticMoE
from bio_inspired_nanochat.neuroviz import NeuroVizConfig, NeuroVizManager
from bio_inspired_nanochat.synaptic_splitmerge import SplitMergeConfig, SplitMergeController


def _build_synaptic_config(overrides: Dict[str, object]) -> SynapticConfig:
    cfg = SynapticConfig()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg

def main():
    parser = argparse.ArgumentParser(description="Verify Bio-Inspired Nanochat Evolution")
    parser.add_argument("--fused-presyn", action="store_true", default=None, help="Enable fused (Triton) presyn kernel")
    parser.add_argument("--no-fused-presyn", action="store_false", dest="fused_presyn", help="Disable fused presyn kernel")
    parser.add_argument("--fused-metrics", action="store_true", default=None, help="Enable fused metrics")
    parser.add_argument("--no-fused-metrics", action="store_false", dest="fused_metrics", help="Disable fused metrics")
    parser.add_argument("--fused-genetics", action="store_true", default=None, help="Enable fused genetics")
    parser.add_argument("--no-fused-genetics", action="store_false", dest="fused_genetics", help="Disable fused genetics")
    
    args = parser.parse_args()

    # 1. Setup Config
    print("[*] Setting up Bio-Synaptic Model...")
    
    torch.autograd.set_detect_anomaly(True)

    syn_kwargs = {
        "enabled": True,
        "tau_rrp": 20.0,
        "xi_dim": 4,
    }
    
    # Override defaults if flags are provided
    if args.fused_presyn is not None:
        syn_kwargs["native_presyn"] = args.fused_presyn
    if args.fused_metrics is not None:
        syn_kwargs["native_metrics"] = args.fused_metrics
    if args.fused_genetics is not None:
        syn_kwargs["native_genetics"] = args.fused_genetics

    syn_cfg = _build_synaptic_config(syn_kwargs)
    
    print(f"[*] Synaptic Config: native_presyn={syn_cfg.native_presyn}, native_metrics={syn_cfg.native_metrics}")
    
    # Tiny model for speed
    model_cfg = GPTSynapticConfig(
        sequence_len=64,
        vocab_size=1024,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=128,
        use_moe=True,
        num_experts=4,
        moe_top_k=2,
        moe_hidden_mult=2,
        syn_cfg=syn_cfg,
        moe_balance_loss=0.01
    )
    
    # 2. Initialize Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Device: {device}")
    
    model = GPTSynaptic(model_cfg).to(device).to(torch.bfloat16)
    
    # 3. Setup NeuroViz & NeuroScore
    log_dir = "runs/verify_evo"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        
    viz_cfg = NeuroVizConfig(
        log_dir=log_dir,
        tb_every=10,       # Log frequent updates
        image_every=1000,  # Skip images for speed
        interactive_every=1000,
        write_tensorboard=True
    )
    viz = NeuroVizManager(viz_cfg)
    viz.register_model(model)
    
    # 4. Setup Split/Merge (to ensure hooks work)
    sm_cfg = SplitMergeConfig(
        enabled=True,
        min_step_interval=50, # Fast cycle
        warmup_steps=10,
        verbose=True
    )
    sm_ctrl = SplitMergeController(model, sm_cfg, logger=viz)
    
    # 5. Optimizer (include Xi)
    # We want to see if Xi moves, so we need it in the optimizer
    # We can't pass Xi separately if we use model.setup_optimizers() because it returns a complex optimizer.
    # So we manually build a simple AdamW here for the test.
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if 'Xi' not in n], 'lr': 1e-3},
        {'params': [p for n, p in model.named_parameters() if 'Xi' in n], 'lr': 1e-2} # High LR for genes to see movement
    ]
    optimizer = torch.optim.AdamW(param_groups)
    
    # 6. Training Loop
    print("[*] Starting Evolution Loop (100 steps)...")
    
    # Snapshot initial genetics
    blocks = getattr(model, "h", None)
    if blocks is None:
        blocks = model.transformer.h
    block0 = cast(Block, blocks[0])
    moe_layer = cast(SynapticMoE, block0.mlp) # First layer MoE
    init_genes = moe_layer.Xi.detach().clone()
    print(f"[*] Initial Genes (Layer 0, Expert 0): {init_genes[0].float().cpu().numpy()}")
    
    model.train()
    
    for step in range(100):
        # Fake data
        idx = torch.randint(0, model_cfg.vocab_size, (4, model_cfg.sequence_len), device=device)
        targets = torch.randint(0, model_cfg.vocab_size, (4, model_cfg.sequence_len), device=device)
        
        optimizer.zero_grad()
        
        # Forward
        logits, loss = model(idx, targets)
        
        # Backward
        loss.backward()
        
        # Check gradients on Xi
        if step == 0:
            if moe_layer.Xi.grad is None:
                print("[!] WARNING: No gradient on Xi parameters! Genetics are frozen.")
            else:
                grad_norm = moe_layer.Xi.grad.norm().item()
                print(f"[*] Xi Gradient Norm: {grad_norm:.6f}")
        
        optimizer.step()
        
        # NeuroViz & NeuroScore Step
        # This calculates Efficiency, Specialization, etc.
        viz.step(model, step, loss=loss)
        
        # Split/Merge Step
        sm_ctrl.step(step, optimizer=optimizer)
        
        if step % 20 == 0:
            print(f"Step {step}: Loss {loss.item():.4f}")

    # 7. Verification
    print("\n[*] Verification Results:")
    
    # Check Genetics Divergence
    final_genes = moe_layer.Xi.detach()
    diff = (final_genes - init_genes).abs().mean().item()
    print(f"  -> Genetic Drift (Mean L1): {diff:.6f}")
    if diff > 1e-5:
        print("  [SUCCESS] Genetics are evolving!")
    else:
        print("  [FAIL] Genetics are static. Check gradient flow.")
        
    # Check NeuroScore Stats
    if viz.score:
        stats = viz.score.stats.get("moe_L0")
        if stats:
            eff = stats["efficiency"].mean().item()
            spec = stats["specialization"].mean().item()
            print(f"  -> Mean Efficiency: {eff:.4f}")
            print(f"  -> Mean Specialization: {spec:.4f}")
            
            if eff != 0 and spec != 0:
                print("  [SUCCESS] NeuroScore metrics are active.")
            else:
                print("  [FAIL] NeuroScore metrics are zero.")
        else:
            print("  [FAIL] No NeuroScore stats found for Layer 0.")
            
    # Check TensorBoard logs
    if os.path.exists(log_dir):
        print(f"  [SUCCESS] TensorBoard logs written to {log_dir}")
    else:
        print("  [FAIL] No logs found.")

    viz.close()

if __name__ == "__main__":
    main()

