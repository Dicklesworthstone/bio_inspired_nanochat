
import torch
import torch.nn as nn
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.synaptic import SynapticConfig

def test_flex_correctness():
    print("\n=== Verifying SynapticFlexAttention Correctness ===")
    
    if not torch.cuda.is_available():
        print("Skipping: CUDA not available.")
        return

    # Use float16 because Triton atomic_add sometimes fails with bf16 on certain setups
    dtype = torch.float16
    device = torch.device("cuda:0")
    
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")

    # Config with Flex enabled
    syn_cfg = SynapticConfig(
        enable_presyn=True,
        enable_hebbian=True,
        enable_metabolism=True
    )
    
    config = GPTSynapticConfig(
        sequence_len=128, # Small seq len for correctness check
        vocab_size=1024,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=128,
        synapses=True,
        syn_cfg=syn_cfg,
        use_flex_attention=True
    )
    
    print("Initializing model with use_flex_attention=True...")
    model = GPTSynaptic(config).to(device).to(dtype)
    model = torch.compile(model) # Flex requires compile
    
    # Dummy Data
    B, T = 2, 128
    x = torch.randint(0, 1024, (B, T), device=device)
    y = torch.randint(0, 1024, (B, T), device=device)
    
    print("Running Forward Pass...")
    try:
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            logits, loss = model(x, y)
            
        print(f"Logits shape: {logits.shape}")
        print(f"Loss: {loss.item()}")
        
        # Check for NaNs
        if torch.isnan(logits).any():
            print("❌ FAILURE: Logits contain NaNs!")
        elif torch.isinf(logits).any():
            print("❌ FAILURE: Logits contain Infs!")
        else:
            print("✅ Logits are finite.")
            
        if torch.isnan(loss):
            print("❌ FAILURE: Loss is NaN!")
        else:
            print("✅ Loss is valid.")

        print("Running Backward Pass...")
        loss.backward()
        
        # Check Gradients
        has_nans = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"❌ FAILURE: Gradient NaN in {name}")
                    has_nans = True
                    break
        
        if not has_nans:
            print("✅ Gradients are finite.")
            
    except Exception as e:
        print(f"❌ CRITICAL FAILURE during execution: {e}")
        import traceback
        traceback.print_exc()

    # Multi-GPU check (Distributed mock)
    if torch.cuda.device_count() > 1:
        print("\n=== Checking Multi-GPU Availability ===")
        device2 = torch.device("cuda:1")
        print(f"Moving simple tensor to {device2}...")
        try:
            t = torch.tensor([1.0], device=device2)
            print(f"Success: {t}")
        except Exception as e:
            print(f"Failed to use {device2}: {e}")

if __name__ == "__main__":
    test_flex_correctness()
