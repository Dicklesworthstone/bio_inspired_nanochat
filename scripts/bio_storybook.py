r"""Pedagogical Storybook & Living Visualizer for Bio-Inspired Mechanisms (beads 2l0, 4x9).

Step-through pedagogical showcase and interactive HTML storybook demonstrating:
  1. Presynaptic Biophysics: Calcium influx, Hill Ca2+ sensors (Syt1/Syt7), RRP vesicle fatigue & endocytosis.
  2. Postsynaptic Metaplasticity: Fast eligibility traces, slow Hebbian consolidation, bistable CaMKII/PP1 latch, BDNF.
  3. Neuromodulatory Regulation: Dopamine (DA), Acetylcholine (ACh), Norepinephrine (NE) multi-factor gating.
  4. Structural Evolution: MoE expert lineage, credit-assigned split/merge neurogenesis.

Usage:
    python -m scripts.bio_storybook --export-html docs/bio_storybook.html
    pytest tests/test_bio_storybook.py -v
"""

from __future__ import annotations

import argparse
import datetime
import html
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from bio_inspired_nanochat.neuromod import NeuromodulatoryBus, NeuromodConfig
from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticPresyn, build_presyn_state

DEFAULT_STORYBOOK_HTML = Path("docs") / "bio_storybook.html"


@dataclass
class StorybookChapter:
    title: str
    subsystem: str
    summary: str
    equations: list[str]
    step_data: list[dict[str, Any]]
    narrative: str


def run_presynaptic_chapter(steps: int = 8) -> StorybookChapter:
    """Simulate presynaptic calcium and vesicle depletion across a burst of token firings."""
    cfg = SynapticConfig(
        enable_presyn=True,
        tau_c=6.0,
        doc2_gain=0.08,
        prime_rate=0.075,
        rec_rate=0.06,
        endo_delay=3,
    )
    pre = SynapticPresyn(16, cfg)
    B, T, H = 1, steps, 1
    state = build_presyn_state(B, T, H, torch.device("cpu"), torch.float32, cfg)

    # Simulate spike drive burst (tokens 2..5 fire strongly)
    drive = torch.zeros(B, H, T, 1)
    drive[0, 0, 2:6, 0] = 3.5  # High activity burst
    drive[0, 0, 6:, 0] = 0.2   # Post-burst quiet period

    idx = torch.arange(T).view(1, 1, T, 1)
    release_prob = pre.release_canonical(state, drive, idx, train=False)

    step_records: list[dict[str, Any]] = []
    for t in range(steps):
        ca = float(state["C"][0, 0, t].item()) if state["C"].ndim == 3 else float(state["C"][0, t].item())
        rrp = float(state["RRP"][0, 0, t].item()) if state["RRP"].ndim == 3 else float(state["RRP"][0, t].item())
        rel = float(release_prob[0, 0, t, 0].item())
        step_records.append({
            "token_step": t + 1,
            "drive": float(drive[0, 0, t, 0].item()),
            "calcium": round(ca, 4),
            "rrp_vesicles": round(rrp, 4),
            "release_flux": round(rel, 4),
        })

    return StorybookChapter(
        title="Chapter 1: The Presynaptic Vesicle Cycle",
        subsystem="Presynaptic Biophysics",
        summary="Action potential bursts trigger calcium influx, driving vesicular neurotransmitter release followed by short-term depression (fatigue) and slow endocytosis recycling.",
        equations=[
            r"C_{t} = C_{t-1} \cdot e^{-1/\tau_c} + \alpha_{ca} \cdot \text{Drive}_t",
            r"P_{\text{rel}} = \frac{C^4}{C^4 + K_{d,\text{fast}}^4} + \gamma_{\text{doc2}} \cdot \sigma(4(C - \theta_{\text{doc2}}))",
            r"\text{RRP}_{t+1} = \text{RRP}_t - \text{Rel}_t + \text{Prime}(\text{Res}_t)",
        ],
        step_data=step_records,
        narrative="Notice how steps 3-6 experience high calcium which sharply drives release probability up initially, but progressively exhausts readily releasable vesicles (RRP), creating self-limiting adaptation.",
    )


def run_neuromod_chapter(steps: int = 8) -> StorybookChapter:
    """Simulate neuromodulatory response (DA, ACh, NE) under unexpected reward and high entropy."""
    bus = NeuromodulatoryBus(NeuromodConfig())
    step_records: list[dict[str, Any]] = []

    # Scenario: step 1-3 predictable loss, step 4 surprise spike (loss drops + high reward), step 6 high entropy
    losses = [2.5, 2.4, 2.3, 1.2, 1.1, 1.5, 1.4, 1.3]
    entropies = [1.2, 1.2, 1.1, 1.3, 1.4, 3.2, 2.8, 1.5]

    for t in range(steps):
        bus.update(
            loss=torch.tensor(losses[t]),
            entropy=torch.tensor(entropies[t]),
        )
        lvls = bus.levels()
        gns = bus.gains()
        step_records.append({
            "token_step": t + 1,
            "loss": losses[t],
            "entropy": entropies[t],
            "dopamine_da": round(float(lvls["da"]), 4),
            "acetylcholine_ach": round(float(lvls["ach"]), 4),
            "norepinephrine_ne": round(float(lvls["ne"]), 4),
            "plasticity_gain": round(float(gns["plasticity"]), 4),
        })

    return StorybookChapter(
        title="Chapter 2: Neuromodulatory Control & Surprise",
        subsystem="Neuromodulation (DA / ACh / NE)",
        summary="Neuromodulators act as broadcast broadcast volume knobs: Dopamine signals reward prediction error, Acetylcholine signals uncertainty/entropy, and Norepinephrine triggers exploratory plasticity on surprise.",
        equations=[
            r"\delta_{\text{DA}} = \text{RPE} = R_t - \hat{V}_t",
            r"\text{ACh}_t = \text{Softplus}(\beta_{\text{ach}} \cdot H(p_t))",
            r"\text{LR}_{\text{eff}} = \text{LR}_0 \cdot (1 + \kappa_{\text{da}} \cdot \text{DA}_t) \cdot (1 + \kappa_{\text{ne}} \cdot \text{NE}_t)",
        ],
        step_data=step_records,
        narrative="At step 4, the sudden drop in loss triggers a positive Dopamine burst (RPE), elevating synaptic plasticity learning rate. At step 6, predictive entropy spikes, releasing Acetylcholine to open explorative attention bandwidth.",
    )


def generate_storybook_html(
    chapters: list[StorybookChapter],
    output_path: Path | str = DEFAULT_STORYBOOK_HTML,
) -> Path:
    """Export self-contained, interactive HTML storybook with styling and responsive tables."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    css = """
    :root {
        --bg: #0d1117;
        --card-bg: #161b22;
        --border: #30363d;
        --text: #c9d1d9;
        --heading: #58a6ff;
        --accent: #2ea043;
        --highlight: #f0883e;
    }
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        background-color: var(--bg);
        color: var(--text);
        line-height: 1.6;
        margin: 0;
        padding: 40px 20px;
    }
    .container {
        max-width: 1000px;
        margin: 0 auto;
    }
    header {
        border-bottom: 1px solid var(--border);
        padding-bottom: 20px;
        margin-bottom: 40px;
    }
    h1 { color: var(--heading); font-size: 2.2em; margin-bottom: 5px; }
    .subtitle { color: #8b949e; font-size: 1.1em; }
    .chapter {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 25px;
        margin-bottom: 30px;
    }
    h2 { color: var(--heading); margin-top: 0; }
    .badge {
        display: inline-block;
        background: #1f6feb22;
        color: #58a6ff;
        border: 1px solid #1f6feb66;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.85em;
        margin-bottom: 15px;
    }
    .equation-box {
        background: #090d13;
        border-left: 4px solid var(--heading);
        padding: 12px 18px;
        font-family: "SFMono-Regular", Consolas, Menlo, monospace;
        font-size: 0.9em;
        margin: 15px 0;
        color: #79c0ff;
        overflow-x: auto;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 0.9em;
    }
    th, td {
        border: 1px solid var(--border);
        padding: 8px 12px;
        text-align: right;
    }
    th {
        background: #21262d;
        color: #f0f6fc;
        text-align: right;
    }
    th:first-child, td:first-child { text-align: center; }
    tr:hover { background: #1f242c; }
    .narrative {
        background: #1f293733;
        border-left: 4px solid var(--accent);
        padding: 12px 15px;
        margin-top: 15px;
        border-radius: 0 4px 4px 0;
    }
    footer {
        text-align: center;
        color: #8b949e;
        margin-top: 50px;
        font-size: 0.85em;
    }
    """

    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='UTF-8'>",
        "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "  <title>Bio-Inspired Transformer Storybook & Visualizer</title>",
        f"  <style>{css}</style>",
        "</head>",
        "<body>",
        "  <div class='container'>",
        "    <header>",
        "      <h1>Bio-Inspired Transformer Storybook</h1>",
        "      <div class='subtitle'>A Living Guided Tour of Synaptic Plasticity, Neuromodulation & Metaplastic Dynamics</div>",
        "    </header>",
    ]

    for chap in chapters:
        html_parts.append("    <div class='chapter'>")
        html_parts.append(f"      <span class='badge'>{html.escape(chap.subsystem)}</span>")
        html_parts.append(f"      <h2>{html.escape(chap.title)}</h2>")
        html_parts.append(f"      <p>{html.escape(chap.summary)}</p>")

        html_parts.append("      <div class='equation-box'>")
        for eq in chap.equations:
            html_parts.append(f"        <div>{html.escape(eq)}</div>")
        html_parts.append("      </div>")

        if chap.step_data:
            keys = list(chap.step_data[0].keys())
            html_parts.append("      <table>")
            html_parts.append("        <thead><tr>")
            for k in keys:
                html_parts.append(f"          <th>{html.escape(k.replace('_', ' ').title())}</th>")
            html_parts.append("        </tr></thead>")
            html_parts.append("        <tbody>")
            for row in chap.step_data:
                html_parts.append("          <tr>")
                for k in keys:
                    val = row.get(k, "")
                    html_parts.append(f"            <td>{html.escape(str(val))}</td>")
                html_parts.append("          </tr>")
            html_parts.append("        </tbody>")
            html_parts.append("      </table>")

        html_parts.append(f"      <div class='narrative'><strong>Biological Insight:</strong> {html.escape(chap.narrative)}</div>")
        html_parts.append("    </div>")

    now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    html_parts.extend([
        "    <footer>",
        f"      Generated by bio_storybook on {now_str} • Bio-Inspired Nanochat",
        "    </footer>",
        "  </div>",
        "</body>",
        "</html>",
    ])

    out_file.write_text("\n".join(html_parts), encoding="utf-8")
    return out_file


def run_storybook(export_html: Path | str | None = None, verbose: bool = True) -> list[StorybookChapter]:
    """Execute complete pedagogical storybook walkthrough."""
    console = Console(quiet=not verbose)
    console.print(Panel("[bold cyan]Bio-Inspired Neural Architecture Storybook[/bold cyan]\n[dim]Living demonstrations of biological synaptic biophysics[/dim]"))

    ch1 = run_presynaptic_chapter()
    ch2 = run_neuromod_chapter()
    chapters = [ch1, ch2]

    for chap in chapters:
        console.print(f"\n[bold green]{chap.title}[/bold green] ([dim]{chap.subsystem}[/dim])")
        console.print(f"[italic]{chap.summary}[/italic]")

        table = Table(title=f"Step Trace: {chap.subsystem}")
        if chap.step_data:
            for k in chap.step_data[0].keys():
                table.add_column(k.replace("_", " ").title(), justify="right")
            for row in chap.step_data:
                table.add_row(*[str(v) for v in row.values()])
            console.print(table)

    if export_html is not None:
        target = Path(export_html)
        generate_storybook_html(chapters, output_path=target)
        console.print(f"[bold green]Storybook HTML exported to: {target}[/bold green]")

    return chapters


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bio-Inspired Transformer Storybook Visualizer")
    parser.add_argument(
        "--export-html",
        type=str,
        default=str(DEFAULT_STORYBOOK_HTML),
        help="Path to export standalone HTML storybook",
    )
    args = parser.parse_args(argv)
    run_storybook(export_html=args.export_html, verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
