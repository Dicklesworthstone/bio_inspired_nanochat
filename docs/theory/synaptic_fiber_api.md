# Minimal Shared Synaptic Fiber State & API Specification (bead `0642.11.3`)

> **Scope**: The canonical dataclass and protocol contract governing all theoretical thrust implementations ($A, D, E, F$) and runtime telemetry.

---

## 1. Unified Fiber State Dataclass

```python
from dataclasses import dataclass, field
import torch
from torch import Tensor

@dataclass
class SynapticFiberState:
    """The local state vector z in fiber F_x at sequence/layer coordinate x = (t, l)."""
    
    # 1. Fast Stratum (Calcium Core & Buffer)
    calcium: Tensor          # (B, N) or (B, H, T) free intracellular Ca2+
    buffer_bound: Tensor     # (B, N) Ca2+-bound buffer fraction
    
    # 2. Medium Stratum (Vesicle Pools)
    rrp: Tensor              # (B, N) readily-releasable pool
    reserve: Tensor          # (B, N) reserve replenishment pool
    clathrin: Tensor         # (B, N) endocytic recycling intermediate
    
    # 3. Slow Stratum (Plasticity & Consolidation)
    camkii: Tensor           # (D_v,) phosphorylated CaMKII fraction in [0, 1]
    pp1: Tensor              # (D_v,) active PP1 phosphatase fraction in [0, 1]
    bdnf: Tensor             # (D_v,) accumulated metaplasticity factor
    w_fast: Tensor | None    # (D_in, D_out) online fast-weight matrix
    
    # 4. Gauge & Invariant Certificates
    gauge_u: Tensor          # (D_in, R) left rank-R factor
    gauge_v: Tensor          # (R, D_out) right rank-R factor
    retention_certified: bool = False
    spectral_radius: float = 0.0
    free_energy: float = 0.0
```

---

## 2. Protocol & Lifecycle Interface

All modular synaptic modules implement the `SynapticFiberModule` protocol:

```python
from typing import Protocol, Tuple

class SynapticFiberModule(Protocol):
    """Lifecycle protocol for fiber bundle transformations."""

    def horizontal_transport(self, z: SynapticFiberState, dx: Tuple[int, int]) -> SynapticFiberState:
        """Parallel transport fiber state along base connection A(z; dx)."""
        ...

    def vertical_drift(self, z: SynapticFiberState, dt: float) -> Tuple[Tensor, Tensor]:
        """Compute GENERIC drift [L(z) grad E + M(z) grad S]."""
        ...

    def vertical_diffusion(self, z: SynapticFiberState, dt: float) -> Tensor:
        """Sample fluctuation-dissipation preserving stochastic increment sigma(z) dW."""
        ...

    def evaluate_certificate(self, z: SynapticFiberState) -> dict[str, float]:
        """Compute live epsilon-gauge, spectral radius, and retention half-width delta*."""
        ...
```

---

## 3. Telemetry Schema Integration

The fiber state integrates directly with `bio_inspired_nanochat/metrics_schema.py` and `bio_inspired_nanochat/results_registry.py`:
- `calcium_mean` $\leftarrow \operatorname{mean}(\text{calcium})$
- `rrp_mean` $\leftarrow \operatorname{mean}(\text{rrp})$
- `camkii_mean` $\leftarrow \operatorname{mean}(\text{camkii})$
- `free_energy_delta` $\leftarrow \Delta \mathcal{F}$
- `retention_delta_star` $\leftarrow \delta^*(a)$
