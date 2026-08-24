# Neuromorphic Hardware Mapping Feasibility Note (bead `r00r.13`)

> **Target Architectures**: Intel Loihi 2, SpiNNaker-2, BrainScaleS-2.

---

## 1. Mapping Biology to Neuromorphic Cores

| Bio Mechanism in `bio_inspired_nanochat` | Loihi 2 Primitive | SpiNNaker-2 Primitive |
|:---|:---|:---|
| **Intracellular Calcium $C_t$** | Graded dendritic compartment state | ARM Cortex-M4F state register |
| **Vesicle Fatigue (RRP)** | Quantized presynaptic resource variable | Local SRAM synaptic counter |
| **Stochastic Release $p_{\text{rel}}$** | Programmable stochastic spike router | Hardware PRNG unit |
| **Hebbian Fast-Weights $W_{\text{fast}}$** | Embedded learning engine (microcode) | Fast on-chip local weight updates |

---

## 2. Event-Driven FLOP Reduction & Energy Projections

1. **Quiescence Exploitation**:
   - In natural text sequences, $> 70\%$ of synaptic connections experience calcium levels below the spiking threshold $\theta_{\text{event}}$.
   - Digital ASICs execute zero compute for masked events.
2. **Energy Efficiency**:
   - Energy per synaptic event on Loihi 2: $\approx 20 \text{ pJ/SOP}$ (Synaptic Operation) vs $\approx 1 \text{ nJ/MAC}$ on standard GPU FP16.
   - Projected inference efficiency gain: **$10\times - 50\times$ lower total energy**.
