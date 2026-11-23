# Event-driven projection and neuromorphic feasibility

`EventDrivenSparseSynapse` is a standalone PyTorch linear projection. Its input has shape
`(..., in_features)`. A feature is retained when its absolute amplitude exceeds the threshold
or an optional calcium signal requests an event. This second condition can retain a small
input. A large input remains active even when its calcium signal is zero. The scalar audit
case used input 100, weight 1, calcium 0 and threshold 0.1. Its dense output is 100.

For omitted inputs each output-coordinate error in exact arithmetic satisfies

```text
|y[o] − y_sparse[o]| ≤ max_i |omitted x[i]| × sum_i |W[o, i]|.
```

The reported bound uses the largest weight **row** sum and the actual maximum omitted amplitude
over all input vectors. Bias cancels. Floating-point reduction order can introduce additional
roundoff. Use a dense control at the same dtype to measure observed error.

## Execution and accounting

The prototype compacts retained values into a COO tensor and calls `torch.sparse.mm` for
float32/64 CPU and CUDA inputs with at most one quarter of entries active. Denser inputs and
other devices or precisions use the dense kernel. The quarter-density rule is a dispatch policy.
It makes no prediction about the timing crossover on a particular device.

The returned statistics identify the projection backend and count executed projection
multiply-adds. Sparse processing costs one multiply-add per retained input and output channel.
The dense fallback charges the full projection, including when many masked values are zero.
Mask formation, reductions, weight-norm evaluation, COO construction, transfer and kernel
dispatch also contribute to wall time. The historical `flops_saved_ratio` field was removed
because it equated input sparsity with total computational savings.

The CPU benchmark times the complete module against the dense operation on the same weights
and input using one thread, emitting median latency, observed active entries, maximum absolute
error and its truncation bound. Run under DSR with
`pytest tests/test_neuromorphic_sparse.py -o log_cli=true --log-cli-level=INFO`. This isolated
operator fixture has a manufactured activity distribution. Whole-model latency, GPU speed and
natural-text sparsity require model-path measurements. [Implementation](../bio_inspired_nanochat/neuromorphic_sparse.py) · [Tests and timing fixture](../tests/test_neuromorphic_sparse.py)

### CPU comparison on November 22, 2026

On a shared Threadripper PRO 5995WX host, the complete module was slower than dense
`F.linear` at all three tested densities. Each row used the same FP32 weights and input for
both calls, one PyTorch thread, 256 input vectors and 512 input and output features.

| Active input entries | Backend | Dense median (ms) | Module median (ms) | Dense time / module time | Max observed error | Truncation bound |
|---:|---|---:|---:|---:|---:|---:|
| 1,323 / 131,072 | Sparse COO | 2.781 | 110.036 | 0.025 | 0.006582 | 0.055604 |
| 26,351 / 131,072 | Sparse COO | 5.038 | 352.954 | 0.014 | 0.006363 | 0.055604 |
| 131,072 / 131,072 | Dense | 4.498 | 7.655 | 0.588 | 0 | 0 |

The sparse rows skipped projection arithmetic and stayed within their truncation bounds.
A second run of the timing fixture produced dense/module ratios of 0.106, 0.028 and
0.209 at the same three densities, with the same observed errors. The machine was busy
during both runs. The timings varied substantially and the module lost all six comparisons.
DSR run `7930ebbe-9b70-4be7-97ff-b6427928f1af` passed all 88 statistical and sparse tests,
plus scoped lint and type checks. The timing log's SHA-256 digest is
`b47e167dfb1d2a36962ccc84b4f4a774831f88e43b63bf7acda903f643b1164c`.
Repeat run `ed75cdd2-5753-44d0-b347-2be972b81a61` also passed all 88 tests, lint and types.
Its timing log digest is `0de576ae9769b9d04917d4368383cdeb8df00a7bd9f6168560d4bf737ca5644d`.

## Proposed device mapping

An event packet would identify a retained feature, its amplitude and its target projection.
A receiving core would read that feature's fan-out weights and accumulate contributions into
local output state. A stateful presynaptic implementation would also need calcium decay and
vesicle replenishment between events. Silent intervals therefore require either analytically
advancing state or clocked updates. The linear projection above implements neither process.

Evaluate a Loihi or SpiNNaker port by mapping the same captured workload through its event
format, fan-out limits, weight precision, local memory and state-update schedule. Compare
device outputs, dropped events, total latency and measured joules for the same input.
No on-chip execution, model-quality outcome or energy reduction has been measured here.
