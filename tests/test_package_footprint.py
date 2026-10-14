"""Programmatic AST import-graph boundary test for package footprint (bead 3zsq / 3zsq.2).

Enforces that:
1. Every module in ``bio_inspired_nanochat`` is accounted for across the three tiers:
   - Core CLI runtime closure (reachable from the 11 user-facing CLI scripts),
   - E2E / benchmark / diagnostic script closure (reachable from ``scripts/e2e/`` and utility scripts),
   - Standalone / test-only research modules (grouped into reviewed clusters).
2. No core model module (``gpt``, ``gpt_synaptic``, ``synaptic``, ``synaptic_splitmerge``, ``engine``)
   or user-facing CLI entrypoint imports any standalone research module.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_ROOT = REPO_ROOT / "bio_inspired_nanochat"
SCRIPTS_ROOT = REPO_ROOT / "scripts"

CORE_CLI_SCRIPTS: tuple[str, ...] = (
    "base_train.py",
    "eval_matrix.py",
    "chat_cli.py",
    "chat_web.py",
    "base_eval.py",
    "mid_train.py",
    "chat_sft.py",
    "tune_bio_params.py",
    "matrix_launch.py",
    "perf_regression_gate.py",
    "enable_synapses.py",
)

STANDALONE_RESEARCH_CLUSTERS: dict[str, frozenset[str]] = {
    "agentic_research": frozenset(
        {
            "bio_inspired_nanochat.ablation_dryrun",
            "bio_inspired_nanochat.ai_neuroscientist",
            "bio_inspired_nanochat.curiosity_rl",
            "bio_inspired_nanochat.curriculum_topology",
            "bio_inspired_nanochat.dream_curriculum",
            "bio_inspired_nanochat.experiment_runner",
            "bio_inspired_nanochat.genome_marketplace",
            "bio_inspired_nanochat.multi_fidelity_evaluator",
            "bio_inspired_nanochat.neural_society",
            "bio_inspired_nanochat.result_interpreter",
            "bio_inspired_nanochat.theory_discovery",
            "bio_inspired_nanochat.xpoll",
        }
    ),
    "geometry_theory": frozenset(
        {
            "bio_inspired_nanochat.certificate_bundle",
            "bio_inspired_nanochat.composition",
            "bio_inspired_nanochat.fmri_attractor_viz",
            "bio_inspired_nanochat.gauge_consolidation",
            "bio_inspired_nanochat.homeostatic_safety",
            "bio_inspired_nanochat.master_sde",
            "bio_inspired_nanochat.neuromorphic_sparse",
            "bio_inspired_nanochat.persistent_memory",
            "bio_inspired_nanochat.synaptic_debugger",
            "bio_inspired_nanochat.ultrametric_memory",
        }
    ),
    "decoding_serving": frozenset(
        {
            "bio_inspired_nanochat.conformal_abstention",
            "bio_inspired_nanochat.constrained_energy_decoder",
            "bio_inspired_nanochat.energy_guided_search",
            "bio_inspired_nanochat.execution",
            "bio_inspired_nanochat.metacognition",
            "bio_inspired_nanochat.reasoning_trace_decoder",
            "bio_inspired_nanochat.speculative_decoder",
            "bio_inspired_nanochat.synaptic_serving_engine",
        }
    ),
    "optimization_telemetry": frozenset(
        {
            "bio_inspired_nanochat.async_population",
            "bio_inspired_nanochat.configurator",
            "bio_inspired_nanochat.determinism",
            "bio_inspired_nanochat.hybrid_optimizer",
            "bio_inspired_nanochat.metrics_fused",
            "bio_inspired_nanochat.ordinal_scheduler",
        }
    ),
}


def discover_package_modules() -> dict[str, Path]:
    modules: dict[str, Path] = {}
    for path in sorted(PKG_ROOT.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modules[".".join(parts)] = path
    return modules


def extract_package_imports(path: Path, valid_modules: set[str]) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("bio_inspired_nanochat"):
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level > 0:
                rel_parts = list(path.relative_to(REPO_ROOT).with_suffix("").parts[: -node.level])
                base = ".".join(rel_parts + ([base] if base else []))
            if base.startswith("bio_inspired_nanochat"):
                found.add(base)
                for alias in node.names:
                    candidate = f"{base}.{alias.name}"
                    if candidate in valid_modules:
                        found.add(candidate)
    return {m for m in found if m in valid_modules}


def transitive_closure(seeds: set[str], graph: dict[str, set[str]]) -> set[str]:
    visited = set(seeds)
    stack = list(seeds)
    while stack:
        current = stack.pop()
        for nxt in graph.get(current, ()):
            if nxt not in visited:
                visited.add(nxt)
                stack.append(nxt)
    return visited


def test_package_footprint_boundary_and_quarantine():
    pkg_modules = discover_package_modules()
    valid_names = set(pkg_modules)
    graph = {name: extract_package_imports(path, valid_names) for name, path in pkg_modules.items()}

    core_seeds: set[str] = set()
    for script_name in CORE_CLI_SCRIPTS:
        core_seeds |= extract_package_imports(SCRIPTS_ROOT / script_name, valid_names)
    reached_core = transitive_closure(core_seeds, graph)

    all_script_seeds: set[str] = set()
    for script_path in sorted(SCRIPTS_ROOT.rglob("*.py")):
        all_script_seeds |= extract_package_imports(script_path, valid_names)
    reached_all_scripts = transitive_closure(all_script_seeds, graph)

    standalone_expected = frozenset().union(*STANDALONE_RESEARCH_CLUSTERS.values())
    standalone_actual = valid_names - reached_all_scripts

    # 1. No core CLI entrypoint or core model file may reach any standalone research module.
    leaked_into_core = reached_core & standalone_expected
    assert not leaked_into_core, (
        f"Standalone research modules leaked into core CLI runtime graph: {sorted(leaked_into_core)}"
    )

    # 2. The standalone research set is exact (any new module must either be wired into a script
    # or explicitly classified in STANDALONE_RESEARCH_CLUSTERS).
    assert standalone_actual == standalone_expected, (
        f"Package footprint drift: unreached={sorted(standalone_actual - standalone_expected)}, "
        f"newly_reached={sorted(standalone_expected - standalone_actual)}"
    )

    # 3. Core kernel dispatcher is live-reachable from SynapticMoE.
    assert "bio_inspired_nanochat.kernels.dispatcher" in reached_core
