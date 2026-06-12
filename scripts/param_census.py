"""Parameter census for ``SynapticConfig`` (bead bio_inspired_nanochat-8j9.6).

Replaces the inaccurate "48-parameter genome" narrative with a precise,
machine-generated inventory of every tunable in ``SynapticConfig``: its
default, subsystem, where it is read, and whether it is LIVE (consumed by
runtime code) or DEAD (declared but read by nothing on any code path).

How "where is it read" is decided (the hard part)
-------------------------------------------------
A naive ``grep '.<field>'`` over-counts badly: six other config dataclasses
(``SplitMergeConfig``, ``NeuroScoreConfig``, ``DivergenceGuardConfig``,
``GPTSynapticConfig``, ``GPTConfig``, ``NeuroVizConfig``) share field names
with ``SynapticConfig`` (``enabled``, ``decay``-likes, ...). So a
``cfg.enabled`` in ``divergence_guard.py`` is NOT a ``SynapticConfig`` read.

We disambiguate with a codebase-specific rule that is correct for this repo:

* In the *synaptic-native* files (``synaptic.py``, ``flex_synaptic.py``,
  ``kernels/presyn_fused.py``) the bare ``cfg`` / ``self.cfg`` handle is
  always a ``SynapticConfig``, so ``cfg.<f>`` and ``self.cfg.<f>`` count.
* Everywhere else, only the ``syn_cfg`` handle is unambiguously a
  ``SynapticConfig``; a ``<obj>.cfg.<f>`` counts only when ``<f>`` is NOT a
  name shared with another config dataclass (the "collision set").
* The Rust kernel reads config via ``cfg.getattr("<f>")``; those count too.

Output
------
* ``docs/parameter_census.json`` — machine-readable (consumed by docs 8j9.3
  and the census-driven search-space work, hea.3).
* ``docs/parameter_census.md`` — human-readable summary grouped by subsystem,
  plus the DEAD list (feeds the prune task 8j9.5).

Run:  ``uv run python -m scripts.param_census``  (``--check`` to fail if the
committed JSON is stale, for CI).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path
from typing import Any

from bio_inspired_nanochat.synaptic import SynapticConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG = REPO_ROOT / "bio_inspired_nanochat"

# Files where a bare ``cfg`` / ``self.cfg`` is a SynapticConfig.
SYNAPTIC_NATIVE = {
    "bio_inspired_nanochat/synaptic.py",
    "bio_inspired_nanochat/flex_synaptic.py",
    "bio_inspired_nanochat/kernels/presyn_fused.py",
}

# Other config dataclasses whose field names we must not confuse with ours.
OTHER_CONFIG_FILES = [
    "bio_inspired_nanochat/synaptic_splitmerge.py",
    "bio_inspired_nanochat/neuroscore.py",
    "bio_inspired_nanochat/divergence_guard.py",
    "bio_inspired_nanochat/gpt_synaptic.py",
    "bio_inspired_nanochat/gpt.py",
    "bio_inspired_nanochat/neuroviz.py",
]

# Subsystem for every SynapticConfig field. The completeness guard below makes
# this map impossible to silently drift from the dataclass.
SUBSYSTEM: dict[str, str] = {
    "rank_eligibility": "general",
    "attn_topk": "general",
    "stochastic_train_frac": "general",
    "stochastic_mode": "general",
    "stochastic_tau": "general",
    "stochastic_count_cap": "general",
    "tau_c": "presynaptic",
    "learnable_kinetics": "presynaptic",
    "differentiable_recurrence": "presynaptic",
    "recurrence_block_size": "presynaptic",
    "recurrence_chunk_len": "presynaptic",
    "doc2_gain": "presynaptic",
    "prime_rate": "presynaptic",
    "unprime_per_release": "presynaptic",
    "nsf_recover": "presynaptic",
    "rec_rate": "presynaptic",
    "endo_delay": "presynaptic",
    "init_rrp": "initial_state",
    "init_reserve": "initial_state",
    "init_snare": "initial_state",
    "init_clamp": "initial_state",
    "init_amp": "initial_state",
    "init_energy": "initial_state",
    "energy_fill": "energy",
    "energy_use": "energy",
    "energy_max": "energy",
    "lambda_loge": "attention",
    "barrier_strength": "attention",
    "epsilon": "attention",
    "loge_bias_clamp": "attention",
    "tau_buf": "kernel_compat",
    "tau_prime": "kernel_compat",
    "tau_rrp": "kernel_compat",
    "tau_energy": "kernel_compat",
    "alpha_ca": "kernel_compat",
    "alpha_buf_on": "kernel_compat",
    "alpha_buf_off": "kernel_compat",
    "alpha_prime": "kernel_compat",
    "alpha_unprime": "kernel_compat",
    "alpha_refill": "kernel_compat",
    "energy_in": "kernel_compat",
    "energy_cost_rel": "kernel_compat",
    "energy_cost_pump": "kernel_compat",
    "syt_fast_kd": "kernel_compat",
    "syt_slow_kd": "kernel_compat",
    "complexin_bias": "kernel_compat",
    "qmax": "kernel_compat",
    "q_beta": "kernel_compat",
    "post_fast_decay": "postsynaptic",
    "post_fast_lr": "postsynaptic",
    "post_slow_lr": "postsynaptic",
    "post_trace_decay": "postsynaptic",
    "fast_weight_normalized": "postsynaptic",
    "fast_weight_eta": "postsynaptic",
    "fast_weight_max_norm": "postsynaptic",
    "camkii_up": "postsynaptic",
    "pp1_tau": "postsynaptic",
    "camkii_thr": "postsynaptic",
    "pp1_thr": "postsynaptic",
    "bdnf_tau": "postsynaptic",
    "bdnf_scale": "postsynaptic",
    "bdnf_gamma": "postsynaptic",
    "bdnf_hebb_accumulate": "postsynaptic",
    "bdnf_max": "postsynaptic",
    "bistable_latch": "latch",
    "latch_ltd_thr": "latch",
    "latch_input_gain": "latch",
    "latch_alpha_ca": "latch",
    "latch_beta_pp1": "latch",
    "latch_gamma_auto": "latch",
    "latch_hill_n": "latch",
    "latch_hill_k": "latch",
    "latch_alpha_pp1": "latch",
    "latch_beta_camkii": "latch",
    "latch_pp1_basal": "latch",
    "latch_gate_beta": "latch",
    "plasticity_during_training": "postsynaptic",
    "structural_interval": "structural",
    "structural_tau_util": "structural",
    "structural_age_bias": "structural",
    "router_embed_dim": "structural",
    "router_contrastive_lr": "structural",
    "router_contrastive_push": "structural",
    "xi_dim": "genetics",
    "enable_presyn": "toggle",
    "enable_hebbian": "toggle",
    "enable_metabolism": "toggle",
    "use_flex_attention": "toggle",
    "native_genetics": "native_toggle",
}

# Curated, human-verified nuance notes for fields whose LIVE/DEAD status needs
# more than a binary. Keyed by field name; merged into the per-field record.
NOTES: dict[str, str] = {
    "native_genetics": "LIVE: gates the fused metabolism/genetics kernel at "
    "synaptic.py (MoE forward). Its dead sibling toggles (native_presyn / "
    "native_metrics / native_plasticity) were pruned in 8j9.5.",
    "init_amp": "LIVE-but-inert: the AMP state is initialized from this and "
    "carried, but its dynamics are frozen (or4t removed amp_load/amp_leak); the "
    "canonical path uses energy->qamp instead.",
}


def synaptic_field_names() -> list[str]:
    return [f.name for f in dataclasses.fields(SynapticConfig)]


def parse_other_config_fieldnames() -> set[str]:
    """Field names of the *other* config dataclasses (the collision set)."""
    names: set[str] = set()
    field_re = re.compile(r"^\s{4}([a-z_][a-z0-9_]*)\s*[:=]")
    class_re = re.compile(r"^class\s+\w*Config\b")
    for rel in OTHER_CONFIG_FILES:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8").splitlines()
        in_class = False
        for line in text:
            if class_re.match(line):
                in_class = True
                continue
            if in_class:
                if line and not line[0].isspace():  # dedent to module level
                    in_class = False
                    continue
                m = field_re.match(line)
                if m:
                    names.add(m.group(1))
    return names


def iter_source_files() -> list[Path]:
    files: list[Path] = []
    for base in (PKG, REPO_ROOT / "scripts", REPO_ROOT / "tests"):
        files.extend(p for p in base.rglob("*.py") if p.name != "param_census.py")
    files.extend((REPO_ROOT / "rust_src" / "src").rglob("*.rs"))
    return files


def classify_read(
    rel_path: str, handle_chain: str, field: str, collision: set[str]
) -> bool:
    """Does ``<handle_chain>.<field>`` in ``rel_path`` read a SynapticConfig?"""
    last = handle_chain.split(".")[-1]
    if rel_path in SYNAPTIC_NATIVE:
        return last in {"cfg", "syn_cfg"}
    # Non-native file:
    if last == "syn_cfg":
        return True
    if last == "cfg":
        # `<obj>.cfg.<field>` is a SynapticConfig only if the name is unambiguous.
        return field not in collision
    return False


def collect_reads(
    fields: list[str], collision: set[str]
) -> dict[str, dict[str, list[str]]]:
    """For each field, evidence sites bucketed into runtime / rust / scripts / tests."""
    py_res = {
        f: re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\." + re.escape(f) + r"\b")
        for f in fields
    }
    rs_res = {f: re.compile(r'getattr\("' + re.escape(f) + r'"\)') for f in fields}
    reads: dict[str, dict[str, list[str]]] = {
        f: {"runtime": [], "rust": [], "scripts": [], "tests": []} for f in fields
    }
    for path in iter_source_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        is_rust = path.suffix == ".rs"
        is_test = rel.startswith("tests/")
        is_script = rel.startswith("scripts/")
        lines = path.read_text(encoding="utf-8").splitlines()
        for lineno, line in enumerate(lines, start=1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            for f in fields:
                accepted = False
                if is_rust:
                    accepted = bool(rs_res[f].search(line))
                else:
                    for m in py_res[f].finditer(line):
                        if classify_read(rel, m.group(1), f, collision):
                            accepted = True
                            break
                if not accepted:
                    continue
                site = f"{rel}:{lineno}"
                if is_rust:
                    reads[f]["rust"].append(site)
                elif is_test:
                    reads[f]["tests"].append(site)
                elif is_script:
                    reads[f]["scripts"].append(site)
                else:
                    reads[f]["runtime"].append(site)
    return reads


def cmaes_phase1_params() -> list[str]:
    """Names in TOP10_PARAM_SPECS, source-parsed to avoid importing `cma`."""
    src = (REPO_ROOT / "scripts" / "tune_bio_params.py").read_text(encoding="utf-8")
    block = re.search(r"TOP10_PARAM_SPECS.*?=\s*\((.*?)\n\)", src, re.DOTALL)
    if not block:
        return []
    return re.findall(r'ParamSpec\(\s*"([^"]+)"', block.group(1))


def build_census() -> dict[str, Any]:
    fields = [f for f in dataclasses.fields(SynapticConfig)]
    names = [f.name for f in fields]

    missing = [n for n in names if n not in SUBSYSTEM]
    extra = [n for n in SUBSYSTEM if n not in names]
    if missing or extra:
        raise SystemExit(
            "SUBSYSTEM map out of sync with SynapticConfig. "
            f"Missing: {missing}  Stale: {extra}"
        )

    collision = parse_other_config_fieldnames()
    reads = collect_reads(names, collision)
    tuned = set(cmaes_phase1_params())

    records: list[dict[str, Any]] = []
    for f in fields:
        r = reads[f.name]
        live = bool(r["runtime"] or r["rust"])
        default = f.default
        if not isinstance(default, (bool, int, float, str)) and default is not None:
            default = repr(default)
        records.append(
            {
                "name": f.name,
                "subsystem": SUBSYSTEM[f.name],
                "type": getattr(f.type, "__name__", str(f.type)),
                "default": default,
                "status": "LIVE" if live else "DEAD",
                "tuned_phase1": f.name in tuned,
                "runtime_read_sites": r["runtime"],
                "rust_read_sites": r["rust"],
                "script_sites": r["scripts"],
                "test_sites": r["tests"],
                "note": NOTES.get(f.name, ""),
            }
        )

    live = [r for r in records if r["status"] == "LIVE"]
    dead = [r for r in records if r["status"] == "DEAD"]
    return {
        "generated_by": "scripts/param_census.py (bead bio_inspired_nanochat-8j9.6)",
        "config_class": "bio_inspired_nanochat.synaptic.SynapticConfig",
        "field_count": len(records),
        "live_count": len(live),
        "dead_count": len(dead),
        "tuned_phase1_count": len([r for r in records if r["tuned_phase1"]]),
        "tuned_phase1_params": sorted(tuned),
        "disambiguation_rule": (
            "synaptic-native files (synaptic.py, flex_synaptic.py, "
            "kernels/presyn_fused.py) credit cfg./self.cfg./syn_cfg.; other files "
            "credit syn_cfg. always and <obj>.cfg.<f> only when <f> is not shared "
            "with another config dataclass; Rust credits getattr(\"<f>\")."
        ),
        "collision_fields": sorted(n for n in collision if n in SUBSYSTEM),
        "learned_genome_note": (
            "The biological 'genome' is the learned per-expert Xi vector "
            "(xi_dim=4: [alpha_fatigue, alpha_energy, camkii_gain, pp1_gain]), a "
            "torch parameter decoded to phenotype kinetics -- NOT the SynapticConfig "
            "hyperparameters. Every field here is a fixed hyperparameter, not a "
            "learned weight."
        ),
        "fields": records,
    }


def render_markdown(census: dict[str, Any]) -> str:
    by_sub: dict[str, list[dict[str, Any]]] = {}
    for r in census["fields"]:
        by_sub.setdefault(r["subsystem"], []).append(r)

    out: list[str] = []
    out.append("# Parameter Census — `SynapticConfig`\n")
    out.append(
        "> **Generated** by `scripts/param_census.py` (bead "
        "`bio_inspired_nanochat-8j9.6`). Do not hand-edit; re-run "
        "`uv run python -m scripts.param_census`. Machine-readable companion: "
        "[`parameter_census.json`](./parameter_census.json).\n"
    )
    out.append(
        f"`SynapticConfig` has **{census['field_count']} fields** — "
        f"**{census['live_count']} LIVE** (read by runtime code) and "
        f"**{census['dead_count']} DEAD** (declared, read by nothing). This is the "
        "ground truth behind the README's *“48-parameter genome”* framing, "
        "which conflated three different counts.\n"
    )
    out.append("## What the counts actually are\n")
    out.append(
        "- **The learned genome is 4-D, not 48.** "
        + census["learned_genome_note"]
        + "\n"
    )
    out.append(
        f"- **The wired search space is {census['tuned_phase1_count']} params**, not "
        "48. CMA-ES Phase 1 (`TOP10_PARAM_SPECS` in `scripts/tune_bio_params.py`) "
        "tunes: "
        + ", ".join(f"`{p}`" for p in census["tuned_phase1_params"])
        + ". The 48-/82-parameter figures are the *aspirational* two-phase plan, not "
        "shipping code.\n"
    )
    dead_rows = sorted(
        (r for r in census["fields"] if r["status"] == "DEAD"),
        key=lambda r: r["name"],
    )
    if dead_rows:
        out.append(
            f"- **The config surface is {census['field_count']} hyperparameters**, "
            f"of which {census['dead_count']} are dead (see prune task `8j9.5`).\n"
        )
    else:
        out.append(
            f"- **The config surface is {census['field_count']} hyperparameters**, "
            "every one of which is read by runtime code — `8j9.5` pruned the last "
            "dead fields (`enabled`, `camkii_down`, `router_sim_threshold`, "
            "`native_presyn`, `native_metrics`, `native_plasticity`).\n"
        )

    out.append("\n## Dead fields (read by nothing)\n")
    if not dead_rows:
        out.append(
            "None — every `SynapticConfig` field is read on some runtime path "
            "(invariant enforced by `tests/test_param_census.py`).\n"
        )
    else:
        out.append("| Field | Subsystem | Default | Note |")
        out.append("|---|---|---|---|")
        for r in dead_rows:
            out.append(
                f"| `{r['name']}` | {r['subsystem']} | `{r['default']}` | "
                f"{r['note'] or '—'} |"
            )

    out.append("\n## Full census by subsystem\n")
    order = [
        "meta",
        "general",
        "presynaptic",
        "initial_state",
        "energy",
        "attention",
        "kernel_compat",
        "postsynaptic",
        "latch",
        "structural",
        "genetics",
        "toggle",
        "native_toggle",
    ]
    for sub in order:
        rows = by_sub.get(sub, [])
        if not rows:
            continue
        live_n = sum(1 for r in rows if r["status"] == "LIVE")
        out.append(f"\n### `{sub}` ({live_n}/{len(rows)} live)\n")
        out.append("| Field | Default | Status | Tuned | Read at |")
        out.append("|---|---|---|---|---|")
        for r in rows:
            sites = r["runtime_read_sites"] + r["rust_read_sites"]
            where = sites[0] if sites else ("scripts only" if r["script_sites"] else "—")
            tuned = "✓" if r["tuned_phase1"] else ""
            out.append(
                f"| `{r['name']}` | `{r['default']}` | {r['status']} | {tuned} | "
                f"{where} |"
            )

    out.append(
        "\n---\n*Status = LIVE when read by a runtime module "
        "(`bio_inspired_nanochat/**` or the Rust kernel), DEAD otherwise. "
        "“Read at” shows the first runtime/Rust read site; full evidence "
        "is in the JSON.*\n"
    )
    return "\n".join(out)


def semantic_view(census: dict[str, Any]) -> dict[str, Any]:
    """The drift-relevant content of a census, ignoring evidence line numbers (which
    shift on any edit elsewhere in a file). ``--check`` compares this so CI fails only
    on real changes — a field added, removed, or rewired LIVE<->DEAD — not on incidental
    line moves. (``tests/test_param_census.py`` enforces the same invariants.)"""
    return {
        "field_count": census["field_count"],
        "live_count": census["live_count"],
        "dead_count": census["dead_count"],
        "tuned_phase1_params": census["tuned_phase1_params"],
        "fields": {
            r["name"]: {
                "subsystem": r["subsystem"],
                "status": r["status"],
                "tuned_phase1": r["tuned_phase1"],
            }
            for r in census["fields"]
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate the SynapticConfig census.")
    ap.add_argument(
        "--check",
        action="store_true",
        help="Fail if the committed census is semantically stale (CI).",
    )
    args = ap.parse_args()

    census = build_census()
    json_path = REPO_ROOT / "docs" / "parameter_census.json"
    md_path = REPO_ROOT / "docs" / "parameter_census.md"
    new_json = json.dumps(census, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not json_path.exists():
            print(
                "docs/parameter_census.json missing; run "
                "`uv run python -m scripts.param_census`.",
                file=sys.stderr,
            )
            return 1
        committed = json.loads(json_path.read_text(encoding="utf-8"))
        if semantic_view(committed) != semantic_view(census):
            print(
                "docs/parameter_census.json is stale (a field was added/removed or "
                "changed LIVE/DEAD); run `uv run python -m scripts.param_census`.",
                file=sys.stderr,
            )
            return 1
        print("parameter census up to date.")
        return 0

    json_path.write_text(new_json, encoding="utf-8")
    md_path.write_text(render_markdown(census), encoding="utf-8")
    print(
        f"Wrote {json_path.relative_to(REPO_ROOT)} and "
        f"{md_path.relative_to(REPO_ROOT)}: "
        f"{census['field_count']} fields "
        f"({census['live_count']} live / {census['dead_count']} dead), "
        f"{census['tuned_phase1_count']} tuned."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
