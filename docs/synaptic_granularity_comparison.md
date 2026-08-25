# Synaptic Granularity Comparison (bead `vap.2`)

`SynapticConfig.granularity` now changes the physical biological state, not merely a label or one eligibility-rank constant. The transformer backbone and attention score shapes remain identical across arms.

## Implemented state spectrum

| Mode | Presynaptic recurrent state | Projection eligibility | Molecular postsynaptic state |
|:---|:---|:---|:---|
| `per_connection` | `B × H × Tk`: one state per head/key connection | configured rank `R` | one gate per output neuron |
| `per_neuron` | `B × H × 1`: keys pooled within each head | `min(R, 4)` | one gate per output neuron |
| `per_expert` | `B × 1 × 1`: heads and keys pooled for the layer/expert | rank 1 | one scalar gate for the layer/expert |

Pooled state is broadcast without allocation when computing edge release. Valid edge activity is averaged back into the representative state machine after each causal query. This keeps state-update scale stable as the number of keys or heads changes. The fine-grained default retains the established Python, Rust, and Triton-compatible state shape; pooled modes use the canonical Python recurrence because the native kernels are shape-specialized for `B × H × Tk`.

## Reproducible comparison

Run the checked-in harness with the project toolchain:

```bash
uv run python -m scripts.eval_synaptic_granularity \
  --output results/granularity_comparison.json
```

The protocol uses identical architecture, optimizer settings, associative-recall batches, and seeds. Every shape-compatible model tensor is copied from the same per-connection reference initialization, preventing granularity-dependent allocation order from changing ordinary backbone weights. State footprint means persistent model buffers plus plastic parameters plus one runtime presynaptic state per attention layer.

The current checked-in CPU run uses two seeds (`42`, `1337`), 12 training steps, `L=2`, `D=64`, `H=4`, vocabulary 128, sequence length 32, and batch size 4:

| Granularity | State footprint | Reduction vs fine | Throughput (tok/s) | Validation loss | Validation BPB | Recall accuracy |
|:---|---:|---:|---:|---:|---:|---:|
| `per_connection` | 443.1 KB | baseline | 652 ± 89 | 4.8489 ± 0.0396 | 6.9955 ± 0.0571 | 0.000 ± 0.000 |
| `per_neuron` | 340.5 KB | 23.2% | 656 ± 190 | 4.8609 ± 0.0406 | 7.0128 ± 0.0586 | 0.000 ± 0.000 |
| `per_expert` | 279.5 KB | 36.9% | 759 ± 104 | 4.8614 ± 0.0430 | 7.0135 ± 0.0621 | 0.000 ± 0.000 |

The state-footprint reduction is directly established. The short run is an engineering comparison and health check, not evidence of learned recall quality: all arms remain at zero recall accuracy after 12 steps, and two CPU seeds are insufficient for a scientific throughput or quality verdict. A longer registered evaluation on the project’s scale-up hardware is still required before claiming a quality/cost Pareto frontier.

Raw configuration, per-seed loss trajectories, timing, state components, environment metadata, and aggregates are in `results/granularity_comparison.json`. Tests in `tests/test_synaptic_granularity.py` pin physical shapes, live pooled updates, invalid-config rejection, finite training/evaluation, state ordering, and report structure.
