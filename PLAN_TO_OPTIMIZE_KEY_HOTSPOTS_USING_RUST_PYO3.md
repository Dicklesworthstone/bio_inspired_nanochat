## Plan to Optimize Key CPU Hotspots Using Rust + PyO3 (`rustbpe` / `rust_src`)

While GPU training and prefill target Triton kernels (`PLAN_TO_OPTIMIZE_KEY_HOTSPOTS_USING_TRITON.md`), CPU execution — used across CI, deterministic unit/property tests, CPU toy screening, and local CMA-ES hyperparameter exploration — suffers from Python loop overhead and per-step PyTorch tensor allocations inside the stateful presynaptic recurrence and MoE routing logic.

This document tracks the **Rust + PyO3 CPU hotspot optimization** in `rust_src/` (`rustbpe` extension module).

---

### Phase 0 – Crate & PyO3 Groundwork (Completed)
| Goal | Implementation |
| --- | --- |
| **Crate Layout** | `rust_src/src/lib.rs` exports BPE tokenization plus submodule `rustbpe.kernels` (`rust_src/src/presyn.rs`, `rust_src/src/moe.rs`). |
| **Feature Flags** | `SynapticConfig.native_presyn = True` (or `BIO_FUSED_PRESYN=1`) dispatches CPU tensors to `rustbpe.kernels` in `bio_inspired_nanochat/synaptic.py`. |
| **Verification** | `cargo test --manifest-path rust_src/Cargo.toml` (native Rust unit tests) and `tests/test_rust_kernels.py` + `tests/test_presyn_rust_dispatch.py` (PyO3 numerical parity against the PyTorch reference). |

---

### Phase 1 – Multi-Query Presynaptic Detached Scan (`jyb.9`, Completed 2026-10-07)
**Hotspot**: `SynapticPresyn` multi-query training/prefill scan (`T_q > 1`).
**Bottleneck**: Stepping `T_q` queries sequentially through Python/Torchscript launches hundreds of small tensor ops per layer.

**Deliverables (`rust_src/src/presyn.rs::presyn_detached_scan_cpu`)**
1. Fused multi-query causal recurrence over `(B, H, T_q, T_k)` in Rust with Rayon parallelism across `(B * H)` rows.
2. Exact Philox-free deterministic RNG parity with the scripted scan (`tests/test_presyn_scan_kernel.py::test_rust_scan_matches_the_scripted_scan`).
3. **Measured Impact**: 64-query block forward+backward dropped from `60 ms` to `18 ms`; `bio_all` 2L/128d CPU training step dropped from `1262 ms` to `925 ms` (single thread). Combined with enabling `cfg.native_presyn = True` in `scripts/tune_bio_params.py`, distributed CPU CMA-ES (`--objective lm`) dropped from `~440 s/generation` to `~9.0 s/generation` (`results/cmaes_phase1_lm_2026-10-13.json`).

---

### Phase 2 – Zero-Copy L1-Tiled Canonical Decode Kernel (`ylo2`, Completed 2026-10-13)
**Hotspot**: `SynapticPresyn.release_canonical` single-query decode path (`T_q == 1`).
**Bottleneck**: Earlier `presyn_release_canonical_cpu` cloned all 5 `(B, H, T_keys)` state buffers on every decode step, allocated `O(B * H * K)` temporary vectors, evaluated `f32::ln()` per key, and spawned Rayon tasks that collided with PyTorch's OpenMP thread pool at `T_keys = 4096` (`0.75×` slower than PyTorch).

**Deliverables (`rust_src/src/presyn.rs::presyn_release_canonical_cpu`)**
1. **`StateBuffer::InPlace`**: Zero-copy in-place mutation of writable contiguous `(B, H, T_keys)` NumPy/PyTorch state tensors (`c`, `buf`, `rrp`, `pr`, `clath`, `delay_q`), eliminating 5 full-buffer heap allocations + copies per token.
2. **Stack Gather (`[TouchedKey; 64]`)**: Zero-heap-allocation gather of the `K <= 64` top-k touched keys per `(b, h)` head before background decay.
3. **L1-Tiled (`L1_TILE = 512`) Transcendental-Free SIMD Sub-Loops**: Split the full-key background pass into L1-resident tiles (`16 KB` working set across 8 f32 slices) with branchless auto-vectorized loops (`ca_step_slice`, `vesicle_recycle_slice`, `pr_from_c_slice`) and no per-key `ln()` calls.
4. **Measured Impact (`results/presyn_rust_decode_bench_2026-10-13.json`)**:
   - **`B=1, H=4, T_keys=512`**: `56.7 us` vs PyTorch `289.1 us` (**`5.10×` faster**).
   - **`B=1, H=8, T_keys=4096`**: `116.5 us` vs PyTorch `829.8 us` (**`7.12×` faster**; `10.17×` faster than old Rust `1185.1 us`).
   - **`B=4, H=8, T_keys=4096`**: `427.4 us` vs PyTorch `2119.3 us` (**`4.96×` faster**; **`12.99×` faster** than old Rust `5551.3 us`).

---

### Phase 3 – MoE Metabolic & Phenotype Kernels (`rust_src/src/moe.rs`, Implemented)
1. `moe_Loading_fused_cpu`, `update_metabolism_cpu`, and `compute_phenotype_cpu` implemented in `rust_src/src/moe.rs` and tested in `tests/test_rust_kernels.py` + `cargo test`.
2. Next step when CPU MoE routing becomes a bottleneck: wire `rustbpe.kernels` into `SynapticMoE` on CPU alongside the GPU Triton genetics path.

