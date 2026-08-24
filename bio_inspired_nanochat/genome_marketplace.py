"""Genome Marketplace & Behavior Transplant Engine (bead re4e.11).

Curated library of pre-evolved Xi genome kinetic profiles that can be instantly
transplanted into synaptic neural networks without retraining weight matrices.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Union

import torch.nn as nn

from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticLinear, SynapticPresyn


@dataclass(frozen=True)
class XiGenomeProfile:
    name: str
    personality: str
    description: str
    config: SynapticConfig

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "personality": self.personality,
            "description": self.description,
            "config": asdict(self.config),
        }


# Curated pre-evolved Xi genome profiles
MARKETPLACE_GENOMES: Dict[str, XiGenomeProfile] = {
    "high_novelty": XiGenomeProfile(
        name="high_novelty",
        personality="explorer",
        description="High plasticity and rapid vesicle replenishment for fast OOD adaptation.",
        config=SynapticConfig(
            post_fast_lr=0.01,
            post_fast_decay=0.85,
            tau_c=3.0,
            tau_rrp=15.0,
            bdnf_scale=2.0,
        ),
    ),
    "high_retention": XiGenomeProfile(
        name="high_retention",
        personality="memorizer",
        description="Strong CaMKII bistable latch and low decay for long-term memory consolidation.",
        config=SynapticConfig(
            post_fast_lr=0.001,
            post_fast_decay=0.99,
            tau_c=12.0,
            tau_rrp=50.0,
            bistable_latch=True,
            bdnf_scale=1.5,
        ),
    ),
    "low_energy": XiGenomeProfile(
        name="low_energy",
        personality="frugal",
        description="Aggressive vesicle fatigue and low baseline release for energy-minimal inference.",
        config=SynapticConfig(
            post_fast_lr=0.0,
            tau_c=2.0,
            tau_rrp=80.0,
            bdnf_scale=0.0,
        ),
    ),
    "balanced_biomimetic": XiGenomeProfile(
        name="balanced_biomimetic",
        personality="canonical",
        description="Standard calibrated biological kinetics balancing plasticity and stability.",
        config=SynapticConfig(
            post_fast_lr=1.5e-3,
            post_fast_decay=0.95,
            tau_c=6.0,
            tau_rrp=40.0,
            bdnf_scale=1.0,
        ),
    ),
}


def list_available_genomes() -> List[XiGenomeProfile]:
    """Return all curated Xi genome profiles available in the marketplace."""
    return list(MARKETPLACE_GENOMES.values())


def get_genome(name: str) -> XiGenomeProfile:
    """Retrieve a specific genome profile by name."""
    if name not in MARKETPLACE_GENOMES:
        raise KeyError(f"Unknown genome profile '{name}'. Available: {list(MARKETPLACE_GENOMES.keys())}")
    return MARKETPLACE_GENOMES[name]


def transplant_genome(
    model: nn.Module,
    genome: Union[str, XiGenomeProfile],
) -> int:
    """Transplant a kinetic Xi genome profile into all synaptic modules of a model in place.

    Returns the count of synaptic modules updated.
    """
    if isinstance(genome, str):
        profile = get_genome(genome)
    else:
        profile = genome

    updated_count = 0
    for module in model.modules():
        if isinstance(module, SynapticLinear):
            object.__setattr__(module, "cfg", profile.config)
            if getattr(module, "post", None) is not None:
                object.__setattr__(module.post, "cfg", profile.config)
            updated_count += 1
        elif isinstance(module, SynapticPresyn):
            object.__setattr__(module, "cfg", profile.config)
            updated_count += 1

    return updated_count
