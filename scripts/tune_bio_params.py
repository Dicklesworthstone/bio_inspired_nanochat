#!/usr/bin/env python3
"""
Bio-Hyperparameter Tuner using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).

This script optimizes the 'Biological Hyperparameters' of the SynapticConfig
(time constants, gains, enzyme kinetics) to maximize performance on a
synthetic 'Associative Recall' task that stresses working memory.

Features:
- Uses cma (CMA-ES) for derivative-free optimization in high-dimensional space.
- Rich visualization of the optimization landscape and population stats.
- Synthetic task generation for fast, reproducible iteration.
- Robust logging.

Usage:
    uv run scripts/tune_bio_params.py
"""

import math
from typing import Dict

import numpy as np
import torch
import cma

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich import box
from rich.syntax import Syntax

from bio_inspired_nanochat.synaptic import SynapticConfig
from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Use a smaller model for fast tuning loop (biophysics should generalize)
MODEL_CONFIG = GPTSynapticConfig(
    sequence_len=256,
    vocab_size=1024,  # small vocab for synthetic task
    n_layer=2,        # shallow
    n_head=4,
    n_kv_head=4,
    n_embd=128,       # thin
    synapses=True,
    use_moe=False,    # disable MoE to focus on synaptic dynamics
)

# Optimization settings
POPULATION_SIZE = 8       # CMA-ES population size
MAX_GENERATIONS = 50      # How long to run
STEPS_PER_EVAL = 100      # Training steps per candidate evaluation
BATCH_SIZE = 16

# -----------------------------------------------------------------------------
# Parameter Definitions (Search Space)
# -----------------------------------------------------------------------------

# We optimize in log-space for time constants and gains to handle scales naturally.
# Format: (name, initial_value, lower_bound, upper_bound, is_log_scale)
PARAM_SPECS = [
    # Timescales (tokens)
    ("tau_c", 4.0, 1.0, 20.0, True),
    ("tau_buf", 10.0, 2.0, 50.0, True),
    ("tau_prime", 18.0, 5.0, 100.0, True),
    ("tau_rrp", 40.0, 10.0, 200.0, True),
    ("tau_energy", 80.0, 20.0, 500.0, True),
    
    # Gains (0..1 or slightly more)
    ("alpha_ca", 0.25, 0.01, 1.0, False),
    ("alpha_buf_on", 0.6, 0.1, 2.0, False),
    ("alpha_prime", 0.10, 0.01, 0.5, False),
    ("alpha_refill", 0.04, 0.001, 0.2, False),
    
    # Energy logic
    ("energy_in", 0.03, 0.001, 0.1, False),
    ("energy_cost_rel", 0.015, 0.001, 0.1, False),
    
    # Sensors
    ("syt_fast_kd", 0.4, 0.1, 5.0, True),
    ("syt_slow_kd", 3.0, 0.5, 20.0, True),
    ("complexin_bias", 0.5, 0.0, 2.0, False),
    
    # Postsynaptic / Eligibility
    ("rho_elig", 0.95, 0.8, 0.999, False),
    ("eta_elig", 0.02, 0.001, 0.2, False),
    ("camkii_gain", 1.5, 0.1, 5.0, False),
    
    # Attention geometry
    ("barrier_strength", 0.075, 0.0, 0.5, False),
]

PARAM_NAMES = [p[0] for p in PARAM_SPECS]
LOWER_BOUNDS = np.array([p[2] for p in PARAM_SPECS])
UPPER_BOUNDS = np.array([p[3] for p in PARAM_SPECS])


def _build_synaptic_config(overrides: Dict[str, float]) -> SynapticConfig:
    cfg = SynapticConfig()
    for key, value in overrides.items():
        setattr(cfg, key, float(value))
    return cfg

def encode_params(config: SynapticConfig) -> np.ndarray:
    """Extract vector from config."""
    vals = []
    for name, init, lb, ub, is_log in PARAM_SPECS:
        v = getattr(config, name)
        if is_log:
            vals.append(math.log(v))
        else:
            vals.append(v)
    return np.array(vals)

def decode_params(vector: np.ndarray) -> Dict[str, float]:
    """Convert optimization vector back to dict."""
    res = {}
    for i, (name, init, lb, ub, is_log) in enumerate(PARAM_SPECS):
        val = vector[i]
        if is_log:
            val = math.exp(val)
        # Clip to bounds (soft-clip handled by CMA usually, but safety first)
        # Actually CMA penalizes out-of-bounds, we just clamp for the model
        val = max(lb, min(ub, float(val)))
        res[name] = val
    return res

# -----------------------------------------------------------------------------
# Synthetic Task: Associative Recall / Copy
# -----------------------------------------------------------------------------

def generate_batch(batch_size: int, seq_len: int, vocab_size: int, device: str):
    """
    Generates a 'Needle in a Haystack' / Copy task.
    Sequence: [Key1] [Val1] ... [KeyK] [ValK] ... [Query:Key1] -> [Target:Val1]
    The model must use working memory (synapses) to store bindings.
    """
    # Simple repeated ngram pattern
    # We generate a random sequence of (L/2) tokens, then repeat it.
    # The model must learn to copy the first half to the second half.
    half = seq_len // 2
    data = torch.randint(0, vocab_size, (batch_size, half), device=device)
    # Input: [Data] [Data]
    # Target: [Ignored] [Data] (predict the repetition)
    
    x = torch.cat([data, data], dim=1)  # (B, T)
    y = torch.cat([
        torch.full_like(data, -1),  # ignore first half loss
        data
    ], dim=1)
    
    return x, y

# -----------------------------------------------------------------------------
# Evaluation Loop
# -----------------------------------------------------------------------------

def evaluate_candidate(solution_vector: np.ndarray) -> float:
    """
    Instantiates a model with specific bio-parameters and runs a short training loop.
    Returns: Final Validation Loss (lower is better).
    """
    try:
        # 1. Decode parameters
        param_dict = decode_params(solution_vector)
        
        # 2. Build Config
        # Start with default config
        syn_cfg = _build_synaptic_config(param_dict)
        
        model_cfg = MODEL_CONFIG
        model_cfg.syn_cfg = syn_cfg
        
        # 3. Build Model
        # Use a different seed per eval? Or same seed for stability?
        # Same seed reduces noise in comparison, but we want robust params.
        # We'll fix the model seed, but data is random.
        torch.manual_seed(42) 
        model = GPTSynaptic(model_cfg).to(DEVICE)
        model.train()
        
        # 4. Optimizer
        # Use simple AdamW
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        
        # 5. Train Loop
        losses = []
        # Running for STEPS_PER_EVAL
        for i in range(STEPS_PER_EVAL):
            x, y = generate_batch(BATCH_SIZE, model_cfg.sequence_len, model_cfg.vocab_size, DEVICE)
            
            logits, loss = model(x, y)
            
            # Backprop
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            
            losses.append(loss.item())
            
        # Return mean of last 10 steps to capture "final" performance
        final_loss = np.mean(losses[-10:])
        
        # Penalty for NaN or explosion
        if not np.isfinite(final_loss):
            return 100.0
            
        return float(final_loss)
        
    except Exception:
        # Log error but don't crash optimization
        # print(f"Error in eval: {e}")
        return 100.0

# -----------------------------------------------------------------------------
# Main Optimization Script
# -----------------------------------------------------------------------------

def main():
    console = Console()
    
    console.print(Panel.fit(
        "[bold green]Bio-Inspired Hyperparameter Tuning[/bold green]\n"
        f"Optimizing {len(PARAM_SPECS)} biological parameters using CMA-ES\n"
        f"Device: {DEVICE}",
        border_style="green"
    ))

    # Initial guess
    defaults = SynapticConfig()
    x0 = encode_params(defaults)
    sigma0 = 0.2  # Initial exploration step size (in log space / normalized space)
    
    # Setup CMA-ES
    # Bounds must be transformed if we use internal log-space, but CMA handles raw values usually.
    # We'll handle mapping in evaluate_candidate, so CMA sees "unbounded" (or loosely bounded) space 
    # corresponding to our log/linear mix.
    # Ideally we define bounds for CMA to avoid wandering too far.
    
    # Construct lower/upper bounds vector for CMA
    # For log parameters: log(lb), log(ub)
    # For lin parameters: lb, ub
    cma_lbs = []
    cma_ubs = []
    for name, init, lb, ub, is_log in PARAM_SPECS:
        if is_log:
            cma_lbs.append(math.log(lb))
            cma_ubs.append(math.log(ub))
        else:
            cma_lbs.append(lb)
            cma_ubs.append(ub)
            
    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'popsize': POPULATION_SIZE,
        'bounds': [cma_lbs, cma_ubs],
        'verbose': -1, # we will handle logging
        'seed': 1337
    })

    # Layout for Rich
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body", ratio=1),
        Layout(name="footer", size=3)
    )
    
    # Best solution tracker
    best_loss = float("inf")
    best_params = {}
    
    table = Table(title="Optimization History", box=box.SIMPLE)
    table.add_column("Gen", justify="right", style="cyan", no_wrap=True)
    table.add_column("Min Loss", justify="right", style="green")
    table.add_column("Mean Loss", justify="right", style="yellow")
    table.add_column("Best Param Change", justify="left", style="white")

    # Start timing
    # start_time = time.time()

    with Live(table, refresh_per_second=4, console=console):
        
        for gen in range(MAX_GENERATIONS):
            if es.stop():
                break
                
            # Ask for candidates
            solutions = es.ask()
            
            # Evaluate (Parallel or Serial)
            # For simplicity and CUDA safety, we do serial here unless we engineer MP carefully.
            # Small model trains very fast (~1-2s per eval).
            fitnesses = []
            
            # Optional: Show progress bar for this generation
            # but inside Live it's tricky. We'll just execute.
            
            for sol in solutions:
                loss = evaluate_candidate(sol)
                fitnesses.append(loss)
            
            # Update strategy
            es.tell(solutions, fitnesses)
            # es.disp() # (optional standard cma log)
            
            # Statistics
            gen_min = min(fitnesses)
            gen_mean = np.mean(fitnesses)
            
            # Update best
            if gen_min < best_loss:
                best_loss = gen_min
                best_idx = np.argmin(fitnesses)
                best_params = decode_params(solutions[best_idx])
                
                # Highlight significant drift from default
                diffs = []
                for k, v in best_params.items():
                    def_val = getattr(defaults, k)
                    if abs(v - def_val) / (def_val + 1e-9) > 0.2: # >20% change
                        diffs.append(f"{k}: {def_val:.2f}->{v:.2f}")
                
                diff_str = ", ".join(diffs[:3]) # show top 3 changes
                if len(diffs) > 3:
                    diff_str += "..."
            else:
                diff_str = "-"

            # Log row
            table.add_row(
                f"{gen+1}", 
                f"{gen_min:.4f}", 
                f"{gen_mean:.4f}", 
                diff_str
            )
            
            # live object updates automatically when table changes, no explicit update needed 

    # Final Result
    console.print("\n[bold green]Optimization Complete![/bold green]")
    console.print(f"Best Loss: {best_loss:.4f}")
    
    # Print formatted best configuration
    best_syntax = Syntax(
        "best_config = SynapticConfig(\n" + 
        "\n".join([f"    {k}={v:.4f}," for k,v in best_params.items()]) + 
        "\n)",
        "python",
        theme="monokai",
        line_numbers=True
    )
    console.print(Panel(best_syntax, title="Optimized Configuration", border_style="green"))
    
    # Optionally save to file
    with open("best_synaptic_config.py", "w") as f:
        f.write("# Auto-generated by scripts/tune_bio_params.py\n")
        f.write("from bio_inspired_nanochat.synaptic import SynapticConfig\n\n")
        f.write("OPTIMIZED_SYNAPTIC_CONFIG = SynapticConfig(\n")
        for k, v in best_params.items():
            f.write(f"    {k}={v:.5f},\n")
        f.write(")\n")
    console.print("[dim]Saved to best_synaptic_config.py[/dim]")

if __name__ == "__main__":
    main()

