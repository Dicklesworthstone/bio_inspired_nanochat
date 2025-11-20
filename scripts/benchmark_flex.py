
import time
import torch
import torch.nn as nn
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig
from bio_inspired_nanochat.synaptic import SynapticConfig
from bio_inspired_nanochat.common import compute_init

def benchmark(use_flex: bool, batch_size=4, seq_len=4096, n_layer=12, n_head=12, n_embd=768):
    print(f"\n--- Benchmarking with use_flex_attention={use_flex} ---")
    
    # Reset memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Config
    syn_cfg = SynapticConfig(
        enable_presyn=True,
        enable_hebbian=True,
        enable_metabolism=True
    )
    
    config = GPTSynapticConfig(
        sequence_len=seq_len,
        vocab_size=50257,
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=n_head,
        n_embd=n_embd,
        synapses=True,
        syn_cfg=syn_cfg,
        use_flex_attention=use_flex
    )
    
    device = torch.device("cuda")
    # Use float16 to avoid Triton atomic_add bf16 issues
    dtype = torch.float16
    
    print("Initializing model...")
    # Avoid meta device for benchmark to prevent buffer init issues
    model = GPTSynaptic(config)
    model.to(device)
    model.to(dtype)
    # Re-init weights just to be sure (though init usually happens in __init__)
    # model.init_weights() 
    model.train()
    
    # Compile is REQUIRED for FlexAttention
    print("Compiling model...")
    model = torch.compile(model) 
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Dummy Data
    x = torch.randint(0, 50257, (batch_size, seq_len), device=device)
    y = torch.randint(0, 50257, (batch_size, seq_len), device=device)
    
    # Warmup
    print("Warmup steps...")
    for _ in range(5):
        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        
    torch.cuda.synchronize()
    
    # Benchmark
    print("Benchmarking steps...")
    t0 = time.time()
    steps = 20
    for _ in range(steps):
        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    
    dt = t1 - t0
    tokens_per_sec = (steps * batch_size * seq_len) / dt
    max_mem = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"Time: {dt:.4f}s")
    print(f"Throughput: {tokens_per_sec:.2f} tokens/sec")
    print(f"Peak VRAM: {max_mem:.2f} GB")
    
    return tokens_per_sec, max_mem

if __name__ == "__main__":
    # Standard Baseline
    try:
        perf_std, mem_std = benchmark(use_flex=False)
        res_std = f"{perf_std:.2f} t/s | {mem_std:.2f} GB"
    except torch.OutOfMemoryError:
        print("Standard: OOM")
        res_std = "OOM"
        perf_std = 0
        mem_std = float('inf')
    except Exception as e:
        print(f"Standard Failed: {e}")
        res_std = "Failed"
        perf_std = 0
        mem_std = float('inf')

    # Flex Attention
    try:
        perf_flex, mem_flex = benchmark(use_flex=True)
        res_flex = f"{perf_flex:.2f} t/s | {mem_flex:.2f} GB"
    except Exception as e:
        print(f"Flex Failed: {e}")
        import traceback
        traceback.print_exc()
        res_flex = "Failed"
        perf_flex = 0
        mem_flex = 0

    print("\n=== Results Summary ===")
    print(f"Standard: {res_std}")
    print(f"Flex:     {res_flex}")
    if perf_std > 0:
        print(f"Speedup:  {perf_flex/perf_std:.2f}x")
        print(f"Mem Red:  {mem_flex/mem_std:.2f}x (Lower is better)")
    else:
        print("Speedup:  Infinite (Standard OOM)")
