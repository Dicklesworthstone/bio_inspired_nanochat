# Mechanism Redundancy and Saturation Hypotheses (bead 74f.5)

This note records interaction risks to test. No completed factorial experiment or
correlation artifact is currently linked from this document, so the classifications below
are hypotheses, not empirical findings or production recommendations.

## Interaction hypotheses

| Mechanism pair | Code-level interaction | Question requiring measurement |
|:---|:---|:---|
| BDNF and CaMKII/PP1 | BDNF scales update magnitude while the latch gates consolidation. | Do the signals add independent predictive value, or behave like one effective learning-rate control? |
| Doc2 and vesicle fatigue | Both influence same-step release probability during sustained activity. | Does the Doc2 gain improve useful retention under depletion, or merely offset another parameter? |
| Septin barrier and attention entropy | The barrier changes logits before softmax. | Is its effect distinguishable from temperature or other logit scaling on matched tasks? |
| Glial homeostasis and MoE metabolism | Both can alter expert routing pressure. | Do they improve load balance independently, and what quality or latency cost accompanies that change? |

The configured timescales and bounds do not by themselves prove independence, prevent
cross-talk, guarantee stability, or justify retaining every mechanism.

## Minimum credible experiment

A redundancy claim needs a factorial ablation with each mechanism off/on, interaction
terms, fixed data order, matched compute, multiple predeclared seeds, and raw per-run
artifacts. Report uncertainty for both quality and operational metrics; do not infer a
correlation or bits-per-byte improvement from unit tests or implementation structure.

Until that experiment exists, no numeric correlation, loss improvement, or saturation
curve should be attributed to this project.
