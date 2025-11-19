import torch
import numpy as np
import pytest
from bio_inspired_nanochat.synaptic import SynapticConfig, build_presyn_state

# Try to import rustbpe, skip if not available
try:
    import rustbpe
except ImportError:
    rustbpe = None

@pytest.mark.skipif(rustbpe is None, reason="rustbpe not installed")
def test_presyn_step_cpu_parity():
    B, H, T, D = 2, 4, 32, 16
    cfg = SynapticConfig(native_presyn=True)
    
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    logits = torch.randn(B, H, T, T)
    # Mask logits
    mask = torch.tril(torch.ones(T, T)).bool()
    logits.masked_fill_(~mask.view(1, 1, T, T), float("-inf"))
    
    state = build_presyn_state(B, T, H, torch.device("cpu"), torch.float32, cfg)
    
    # Run Rust version
    q_np = q.numpy()
    k_np = k.numpy()
    logits_np = logits.numpy()
    state_np = {k: v.numpy() for k, v in state.items()}
    
    syn_logit_rust, state_new_rust = rustbpe.presyn_step_cpu(q_np, k_np, logits_np, state_np, cfg)
    
    # Run PyTorch reference (we need to extract the logic or use the module)
    # Since the module dispatches, we can't easily force it to use the python path unless we disable native_presyn
    # But we want to compare against the python implementation.
    
    # Let's instantiate the module with native_presyn=False
    from bio_inspired_nanochat.synaptic import SynapticPresyn
    cfg_py = SynapticConfig(native_presyn=False)
    mod = SynapticPresyn(D, cfg_py)
    
    # We need to pass causal_mask
    causal_mask = torch.tril(torch.ones(T, T)).bool()
    
    # Clone state for python run
    state_py = {k: v.clone() for k, v in state.items()}
    
    syn_logit_py, state_new_py = mod(q, k, logits, state_py, causal_mask, train_mode=False)
    
    # Compare
    # Note: Rust implementation might have slight numerical differences due to float precision order
    # But should be close.
    
    print("Comparing syn_logit...")
    # Mask out non-causal parts for comparison as they might be -inf or log(eps)
    mask_expanded = mask.view(1, 1, T, T).expand(B, H, T, T)
    
    diff = torch.abs(torch.from_numpy(syn_logit_rust) - syn_logit_py)
    diff_masked = diff[mask_expanded]
    
    print(f"Max diff: {diff_masked.max().item()}")
    assert torch.allclose(torch.from_numpy(syn_logit_rust)[mask_expanded], syn_logit_py[mask_expanded], atol=1e-4, rtol=1e-4)
    
    print("Comparing state...")
    for k in state:
        diff = torch.abs(torch.from_numpy(state_new_rust[k]) - state_new_py[k])
        print(f"State {k} max diff: {diff.max().item()}")
        assert torch.allclose(torch.from_numpy(state_new_rust[k]), state_new_py[k], atol=1e-4, rtol=1e-4)

@pytest.mark.skipif(rustbpe is None, reason="rustbpe not installed")
def test_moe_stats_cpu_parity():
    B, T, k = 2, 128, 2
    E = 8
    
    idx = torch.randint(0, E, (B, T, k))
    gates = torch.rand(B, T, k)
    
    # Python reference
    me = torch.zeros(E)
    pe = torch.zeros(E)
    for e in range(E):
        mask = idx == e
        sel = mask.any(dim=-1)
        me[e] = sel.sum()
        pe[e] = gates.masked_select(mask).sum()
        
    # Rust version
    idx_np = idx.numpy().astype("int64")
    gates_np = gates.numpy()
    
    counts_rust, probs_rust = rustbpe.accumulate_router_stats_cpu(idx_np, gates_np, E)
    
    print("Comparing MoE stats...")
    print(f"Counts max diff: {np.abs(counts_rust - me.numpy()).max()}")
    print(f"Probs max diff: {np.abs(probs_rust - pe.numpy()).max()}")
    
    assert np.allclose(counts_rust, me.numpy(), atol=1e-5)
    assert np.allclose(probs_rust, pe.numpy(), atol=1e-4)

@pytest.mark.skipif(rustbpe is None, reason="rustbpe not installed")
def test_metabolism_cpu_parity():
    E = 8
    fatigue = torch.rand(E)
    energy = torch.rand(E)
    alpha_fatigue = torch.rand(E) * 0.1
    alpha_energy = torch.rand(E) * 0.1
    util = torch.rand(E)
    
    # Python reference
    f_py = fatigue.clone()
    e_py = energy.clone()
    f_py.mul_(1.0 - alpha_fatigue).add_(alpha_fatigue * util)
    e_py.mul_(1.0 - alpha_energy).add_(alpha_energy * (1.0 - util))
    
    # Rust version
    f_rust, e_rust = rustbpe.update_metabolism_cpu(
        fatigue.numpy(), energy.numpy(), alpha_fatigue.numpy(), alpha_energy.numpy(), util.numpy()
    )
    
    print("Comparing Metabolism...")
    print(f"Fatigue max diff: {np.abs(f_rust - f_py.numpy()).max()}")
    print(f"Energy max diff: {np.abs(e_rust - e_py.numpy()).max()}")
    
    assert np.allclose(f_rust, f_py.numpy(), atol=1e-5)
    assert np.allclose(e_rust, e_py.numpy(), atol=1e-5)

if __name__ == "__main__":
    # Manual run
    if rustbpe:
        test_presyn_step_cpu_parity()
        test_moe_stats_cpu_parity()
        test_metabolism_cpu_parity()
        print("All tests passed!")
    else:
        print("rustbpe not installed, skipping tests")

