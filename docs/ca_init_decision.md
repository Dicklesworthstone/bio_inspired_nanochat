# Decision & Analysis: Cellular Automata (CA) Weight Initialization (bead dlh)

> **Decision**: **PERMANENT DEFAULT-OFF / ARCHIVED**  
> **Mechanism**: Cellular Automata (Game of Life / 1D Rule 30 / 110) Morphogenetic Weight Initialization.  
> **Scope**: Pre-training linear & attention projection weight matrices.

> **Evidence status (updated 2026-10-14).** The raw CSV logs from the original 400-step micro-benchmark
> (`runs/ca_init_microbench/20251218_054731/` on `trj`) have been recovered and committed at
> `results/ca_init_microbench_2025-12-18/{baseline,ca_rule30,ca_rule116}.csv`. All figures below come
> directly from those three CSV files. CA init remains opt-in via `--init_type=ca_rule30|ca_rule116` on
> `scripts/base_train.py`.

---

## 1. Executive Summary & Decision

Following the 400-step micro-benchmark evaluation (`results/ca_init_microbench_2025-12-18/`), Cellular Automata weight initialization (`SynapticLinear.w_slow` and `SynapticMoE.router_embeddings`) is kept **disabled by default** (`--init_type=baseline`).

### Verdict Summary
- **Early Loss (Steps 20–80)**: CA-initialized models lagged baseline during initial descent (at step 40: `baseline = 4.2249`, `ca_rule30 = 4.2834`, `ca_rule116 = 4.3098`; at step 60: `baseline = 3.5435`, `ca_rule30 = 3.5668`, `ca_rule116 = 3.6232`).
- **Final Loss (Step 399)**: Indistinguishable from standard Gaussian initialization (`baseline = 0.024652`, `ca_rule30 = 0.024473` [$\Delta = -0.000179$], `ca_rule116 = 0.024641` [$\Delta = -0.000010$]).
- **Router Diversity (`sim_min` at Step 399)**: CA initialization slightly lowered minimum pairwise router-embedding cosine similarity (`baseline = 0.9919`, `ca_rule30 = 0.9727`, `ca_rule116 = 0.9452`), which is now achieved directly and continuously during training by the contrastive router repulsion (`router_contrastive_push` / `router_contrastive_lr`) and `SynapticGenomeDecoder` (`Xi`).

---

## 2. Experimental Measurements (`results/ca_init_microbench_2025-12-18/`)

| Metric | `baseline.csv` | `ca_rule30.csv` | `ca_rule116.csv` | Outcome |
|:---|---:|---:|---:|:---|
| **Step 0 Loss** | `5.545179` | `5.545179` | `5.545179` | Identical start |
| **Step 40 Loss** | `4.224912` | `4.283355` | `4.309817` | ⚠️ Slower initial descent |
| **Step 60 Loss** | `3.543451` | `3.566785` | `3.623249` | ⚠️ Slower initial descent |
| **Step 399 Final Loss** | `0.024652` | `0.024473` | `0.024641` | ⚠️ Neutral ($\|\Delta\| < 2 \times 10^{-4}$) |
| **Step 399 Router `sim_min`** | `0.991866` | `0.972713` | `0.945212` | Modest router spread |
| **Step 399 `wte` `sv_max` / `sv_mean`** | `26.919 / 14.968` | `26.944 / 14.972` | `26.942 / 14.972` | Neutral |

---

## 3. Governance & Guardrail Policies

1. **Default-Off Guardrail**: Any future experimental CA-init exploratory scripts must remain strictly opt-in behind explicit command-line flags.
2. **Standard Initialization**: All production models (`GPT`, `GPTSynaptic`) will continue using standard scaled normal initialization with weight tying on `wte`/`lm_head` and unit-norm router embeddings.
3. **Playbook Update**: Documented in `docs/mgr_cross_pollination_playbook.md` as an evaluated and archived morphogenetic initialization candidate.
