# Lifelong Persistent Synaptic Memory & Privacy Threat Model (beads `re4e.4`, `re4e.4.1`)

_Persistent Memory & Multi-Tenant Isolation · Biologically-Grounded Cross-Session Consolidation. Author: GoldenRiver · 2026-08-24._

## Executive Summary & System Specification

This document specifies the architecture for **Persistent Lifelong Synaptic Memory**: extending the within-session synaptic working memory API (`r00r.9`) into durable, per-user memory stores that consolidate across sessions via offline sleep replay (`r00r.6`, `cel.2`).

---

## 1. Multi-Tenant Isolated Memory Architecture

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      Multi-User Session Ingress                        │
 └───────────────────┬────────────────────────────────┬───────────────────┘
                     │                                │
        User Alice (ID: 0x8A1F)           User Bob (ID: 0x3B9C)
                     │                                │
                     ▼                                ▼
   ┌──────────────────────────────────┐ ┌──────────────────────────────────┐
   │    Alice Scratchpad & Latch      │ │     Bob Scratchpad & Latch       │
   │  • W_fast(Alice)                 │ │  • W_fast(Bob)                   │
   │  • ReplayBuffer(Alice)           │ │  • ReplayBuffer(Bob)             │
   └─────────────────┬────────────────┘ └─────────────────┬────────────────┘
                     │                                │
                     ▼ (Session End / Sleep)          ▼ (Session End / Sleep)
   ┌──────────────────────────────────┐ ┌──────────────────────────────────┐
   │    Alice Consolidated Adapter    │ │     Bob Consolidated Adapter     │
   │  • W_slow_delta(Alice)           │ │  • W_slow_delta(Bob)             │
   │  • Strict per-user encryption    │ │  • Strict per-user encryption    │
   └──────────────────────────────────┘ └──────────────────────────────────┘
```

### 1.1 Per-User Isolated Namespacing
1. **Namespace Isolation**: Each user partition is addressed by a non-reversible cryptographic key `H(user_id || salt)`.
2. **Zero Cross-Talk Guarantee**: Under no circumstance can User A's fast weights or replayed dreams modulate the activation path of User B.
3. **Session Loading Contract**: On session start, `load_user_memory(user_id)` dynamically mounts $W_{\text{fast}}$ and user-specific slow deltas into the inference graph.

---

## 2. Cross-Session Sleep Consolidation Protocol

1. **Wake Phase (Session Live)**:
   - High-surprise user prompts and interactions accumulate in the per-user `PrioritizedReplayBuffer`.
   - Local plastic fast-weights $W_{\text{fast}}$ absorb contextual facts.
2. **Offline Sleep Phase (Session Exit)**:
   - The user-scoped `SleepConsolidationController` samples top-surprise sequences or generates synthetic dreams.
   - Fast-to-slow distillation: $\Delta W_{\text{user}} \leftarrow \Delta W_{\text{user}} + \eta_{\text{cons}} (W_{\text{fast}} \odot \text{Latch})$.
   - Synaptic Homeostatic Scaling (SHY): $\Delta W_{\text{user}} \leftarrow \min(1, \frac{C_{\text{max}}}{\|\Delta W_{\text{user}}\|}) \cdot \Delta W_{\text{user}}$.
   - $W_{\text{fast}}$ is zeroed.
3. **Capacity & Pruning Bounds**:
   - Memory growth is strictly bounded to $K \le 8$ low-rank factors or a maximum tensor footprint of 5MB per user.
   - Low-utility associations decay exponentially: $\Delta W \leftarrow \gamma \Delta W$ ($\gamma = 0.99$ per day).

---

## 3. Privacy Threat Model & Hard "Forget" Semantics

### 3.1 Right-to-be-Forgotten Protocol
When a user invokes `forget_user_memory(user_id)`:
1. **Storage Purge**: The disk artifact `user_store/{hash}.pt` is immediately unlinked and overwritten.
2. **Graph Unmount**: Active memory scratchpads are flushed: $W_{\text{fast}} \leftarrow 0$, $\Delta W_{\text{user}} \leftarrow 0$.
3. **Verification Audit**: An automated probe confirms that post-forget associative recall falls back to the unconditioned base model baseline.

### 3.2 Threat Model & Defenses
- **Threat 1: Memory Extraction / Inversion**: Adversary queries the model to extract previously stored user facts.
  - *Defense*: Clamping maximum injection norm $\|\Delta W_{\text{fast}}\| \le 2.0$ prevents single-shot memorization of verbatim high-entropy secrets without repeated consolidation.
- **Threat 2: Memory Poisoning / Backdoor Injection**: Malicious input attempts to destabilize the model via divergent weight matrices.
  - *Defense*: Strict validation rejecting NaN/Inf and norm-capping all injected deltas prior to graph attachment.
