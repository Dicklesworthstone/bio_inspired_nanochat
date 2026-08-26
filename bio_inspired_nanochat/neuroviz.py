# nanochat/neuroviz.py
# ---------------------------------------------------------------------
# Visualization & logging for Synaptic-MoE:
#  - NeuroVizManager: orchestrates TensorBoard + static figures + lineage
#  - LineageBook: split/merge/reset event ledger + timeline renders
#  - Expert plotting: UMAP/PCA map, radar of top experts, histograms
#
# Dependencies:
#   - required: numpy, matplotlib, tensorboard
#   - optional: umap-learn (preferred), scikit-learn (PCA fallback), plotly
#
# Usage:
#   viz = NeuroVizManager(log_dir="runs/brain1", image_every=10000, tb_every=1000)
#   viz.register_model(model)                 # once, after model creation
#   sm_ctrl = SplitMergeController(model, cfg, logger=viz)  # to receive events
#   ...
#   for step in ...:
#       ...
#       viz.step(model, step)                 # per-step logging
#   viz.close()
# ---------------------------------------------------------------------

import importlib
import importlib.util
import csv
import os
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, cast

import numpy as np
from bio_inspired_nanochat.torch_imports import torch, nn, Tensor
from torch.utils.tensorboard import SummaryWriter
import matplotlib

from .synaptic import SynapticExpert, SynapticMoE

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _maybe_import(module: str, attr: Optional[str] = None) -> Any:
    try:
        spec = importlib.util.find_spec(module)
    except (ModuleNotFoundError, ValueError):
        # find_spec raises (rather than returning None) when a PARENT package is missing,
        # e.g. "sklearn.decomposition" without sklearn installed — treat as absent.
        return None
    if spec is None:
        return None
    mod = importlib.import_module(module)
    return getattr(mod, attr) if attr else mod


UMAP = _maybe_import("umap", "UMAP")
_HAS_UMAP = UMAP is not None
PCA = _maybe_import("sklearn.decomposition", "PCA")
_HAS_SKLEARN = PCA is not None
go = _maybe_import("plotly.graph_objects")
_HAS_PLOTLY = go is not None

if TYPE_CHECKING:
    from .neuroscore import NeuroScore, NeuroScoreConfig
else:
    try:
        from .neuroscore import NeuroScore, NeuroScoreConfig
    except Exception:
        NeuroScore = None
        NeuroScoreConfig = None

# --------------------------- utilities --------------------------------


def _to_np(x: torch.Tensor) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _cosine(u: np.ndarray, v: np.ndarray, eps: float = 1e-8) -> float:
    u = u / (np.linalg.norm(u) + eps)
    v = v / (np.linalg.norm(v) + eps)
    return float(np.dot(u, v))


def _reduce_camkii(expert: SynapticExpert) -> float:
    # average camkii over fc1/fc2 posts
    vals = []
    for fc in (expert.fc1, expert.fc2):
        if hasattr(fc, "post") and hasattr(fc.post, "camkii"):
            t = fc.post.camkii
            vals.append(float(t.mean().item()))
    return float(np.mean(vals)) if vals else 0.0


def _reduce_mgate(expert: SynapticExpert) -> float:
    vals = []
    for fc in (expert.fc1, expert.fc2):
        if hasattr(fc, "post") and hasattr(fc.post, "m_gate"):
            t = fc.post.m_gate
            vals.append(float(t.item()))
    return float(np.mean(vals)) if vals else 0.0


def _reduce_elig_norm(expert: SynapticExpert) -> float:
    vals = []
    for fc in (expert.fc1, expert.fc2):
        if hasattr(fc, "post"):
            post = fc.post
            parts = []
            if hasattr(post, "U"):
                parts.append(post.U.norm())
            if hasattr(post, "V"):
                parts.append(post.V.norm())
            if hasattr(post, "fast"):
                parts.append(post.fast.norm())
            if hasattr(post, "slow"):
                parts.append(post.slow.norm())
            if parts:
                vals.append(float(sum(p.item() for p in parts)))
    return float(np.mean(vals)) if vals else 0.0


def _weight_energy_util(energy: np.ndarray, fatigue: np.ndarray) -> np.ndarray:
    """Health = utilization × energy — the SAME formula the split/merge
    lifecycle controller uses (synaptic_splitmerge._health). ``fatigue`` is the
    usage EMA (synaptic.py), NOT an availability measure: the previous
    ``1 - fatigue`` here INVERTED health, making busy experts look dying and
    dead experts look thriving in the v2 bio panels."""
    energy = np.clip(energy, 0.0, 1.0)
    util = np.clip(fatigue, 0.0, 1.0)
    return energy * util


def _fit_2d(emb: np.ndarray) -> np.ndarray:
    if emb.shape[1] <= 2:
        return emb
    # Fallback for very small populations where UMAP/PCA might fail or be weird
    if emb.shape[0] < 4:
        # Just project to first 2 dims or random
        return emb[:, :2]
        
    if _HAS_UMAP:
        try:
            red = UMAP(
                n_neighbors=min(15, emb.shape[0] - 1),
                min_dist=0.3,
                metric="cosine",
                random_state=42,
            )
            return red.fit_transform(emb)
        except Exception:
            # UMAP failed (e.g. too few neighbors), fall back
            pass # nosec B110
            
    if _HAS_SKLEARN:
        return PCA(n_components=2).fit_transform(emb)
    # fallback: random projection
    W = np.random.normal(0, 1, (emb.shape[1], 2))
    Y = emb @ W
    Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    return Y


# --------------------------- lineage book ------------------------------


class LineageBook:
    """
    Keeps a per-layer log of events (split/merge/reset). Provides:
      - timeline PNG render
      - optional interactive HTML (plotly)
    """

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        _ensure_dir(save_dir)
        # layer_id -> [(step, operation, affected expert IDs)]
        self.events: Dict[str, List[Tuple[int, str, List[int]]]] = {}

    def log_merge(
        self, layer_name: str, step: int, parent_i: int, parent_j: int, child_idx: int
    ):
        self.events.setdefault(layer_name, []).append(
            (step, "merge", [parent_i, parent_j, child_idx])
        )
        self._persist(layer_name)

    def log_split(self, layer_name: str, step: int, parent_idx: int, child_idx: int):
        self.events.setdefault(layer_name, []).append(
            (step, "split", [parent_idx, child_idx])
        )
        self._persist(layer_name)

    def log_reset(self, layer_name: str, step: int, source_idx: int, reset_idx: int):
        """Record a healthy source reseeding a dead expert slot."""
        self.events.setdefault(layer_name, []).append(
            (step, "reset", [source_idx, reset_idx])
        )
        self._persist(layer_name)

    def _persist(self, layer_name: str):
        path = os.path.join(self.save_dir, f"{layer_name}_lineage.json")
        with open(path, "w") as f:
            json.dump(self.events.get(layer_name, []), f)

    def render_timeline_png(self, layer_name: str, step: int):
        evs = self.events.get(layer_name, [])
        if not evs:
            return
        fig, ax = plt.subplots(figsize=(12, 4))
        ys = {}
        y_next = 0
        for s, et, ids in evs:
            if et == "merge":
                i, j, c = ids
                for eid in (i, j, c):
                    if eid not in ys:
                        ys[eid] = y_next
                        y_next += 1
                ax.plot([s, s], [ys[i], ys[c]], color="purple", lw=2, alpha=0.7)
                ax.plot([s, s], [ys[j], ys[c]], color="purple", lw=2, alpha=0.7)
                ax.scatter([s], [ys[c]], color="purple", s=24, marker="o", zorder=5)
            elif et == "split":
                p, c = ids
                for eid in (p, c):
                    if eid not in ys:
                        ys[eid] = y_next
                        y_next += 1
                ax.plot([s, s], [ys[p], ys[c]], color="green", lw=2, alpha=0.7)
                ax.scatter([s], [ys[c]], color="green", s=24, marker="^", zorder=5)
            elif et == "reset":
                source, reset = ids
                for eid in (source, reset):
                    if eid not in ys:
                        ys[eid] = y_next
                        y_next += 1
                ax.plot(
                    [s, s], [ys[source], ys[reset]], color="orange", lw=2, alpha=0.7
                )
                ax.scatter(
                    [s], [ys[reset]], color="orange", s=24, marker="s", zorder=5
                )
        ax.set_title(f"Lineage — {layer_name} @ step {step:,}")
        ax.set_xlabel("step")
        ax.set_ylabel("expert ID (track)")
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        out = os.path.join(self.save_dir, f"{layer_name}_lineage_{step:09d}.png")
        plt.savefig(out, dpi=120)
        plt.close(fig)

    def render_interactive_html(self, layer_name: str, step: int):
        if not _HAS_PLOTLY:
            return
        evs = self.events.get(layer_name, [])
        if not evs:
            return
            
        fig = go.Figure()
        ymap = {}
        y_next = 0
        
        for s, et, ids in evs:
            if et == "merge":
                i, j, c = ids
                for eid in (i, j, c):
                    if eid not in ymap:
                        ymap[eid] = y_next
                        y_next += 1
                
                # Parent i -> Child c
                fig.add_trace(go.Scatter(
                    x=[s, s], y=[ymap[i], ymap[c]],
                    mode="lines", line=dict(color="purple", width=1),
                    hoverinfo="none"
                ))
                # Parent j -> Child c
                fig.add_trace(go.Scatter(
                    x=[s, s], y=[ymap[j], ymap[c]],
                    mode="lines", line=dict(color="purple", width=1),
                    hoverinfo="none"
                ))
                # Marker for child
                fig.add_trace(go.Scatter(
                    x=[s], y=[ymap[c]],
                    mode="markers", marker=dict(color="purple", size=8, symbol="circle"),
                    text=f"Merge {i}+{j}->{c}", hoverinfo="text"
                ))
                
            elif et == "split":
                p, c = ids
                for eid in (p, c):
                    if eid not in ymap:
                        ymap[eid] = y_next
                        y_next += 1
                
                # Parent p -> Child c
                fig.add_trace(go.Scatter(
                    x=[s, s], y=[ymap[p], ymap[c]],
                    mode="lines", line=dict(color="green", width=1),
                    hoverinfo="none"
                ))
                # Marker for child
                fig.add_trace(go.Scatter(
                    x=[s], y=[ymap[c]],
                    mode="markers", marker=dict(color="green", size=8, symbol="triangle-up"),
                    text=f"Split {p}->{c}", hoverinfo="text"
                ))

            elif et == "reset":
                source, reset = ids
                for eid in (source, reset):
                    if eid not in ymap:
                        ymap[eid] = y_next
                        y_next += 1

                fig.add_trace(go.Scatter(
                    x=[s, s], y=[ymap[source], ymap[reset]],
                    mode="lines", line=dict(color="orange", width=1),
                    hoverinfo="none"
                ))
                fig.add_trace(go.Scatter(
                    x=[s], y=[ymap[reset]],
                    mode="markers", marker=dict(color="orange", size=8, symbol="square"),
                    text=f"Reset {source}->{reset}", hoverinfo="text"
                ))

        fig.update_layout(
            title=f"Interactive Lineage — {layer_name} @ step {step:,}",
            xaxis_title="step",
            yaxis_title="expert track",
            template="plotly_dark",
            height=600,
            showlegend=False
        )
        out = os.path.join(self.save_dir, f"{layer_name}_lineage_{step:09d}.html")
        fig.write_html(out, include_plotlyjs="cdn")


# --------------------------- NeuroVizManager ---------------------------


@dataclass
class NeuroVizConfig:
    log_dir: str = "runs/neuroviz"
    tb_every: int = 1000
    image_every: int = 10000
    interactive_every: int = 25000
    top_n_radar: int = 6
    save_pngs: bool = True
    write_tensorboard: bool = True
    write_interactive_html: bool = True
    dead_health_threshold: float = 0.02


class NeuroVizManager:
    """
    Central orchestrator for metrics, images, TensorBoard, and lineage logging.
    Acts as a logger for the SplitMergeController (via on_merge/on_split).
    """

    def __init__(self, cfg: NeuroVizConfig):
        self.cfg = cfg
        _ensure_dir(cfg.log_dir)
        self.tb = SummaryWriter(cfg.log_dir) if cfg.write_tensorboard else None
        self.layers: List[Tuple[str, SynapticMoE]] = []  # (name, module)
        self.lineage = LineageBook(os.path.join(cfg.log_dir, "lineage"))
        self._vitals_path = os.path.join(cfg.log_dir, "vitals.csv")
        self._vitals_header_written = (
            os.path.exists(self._vitals_path) and os.path.getsize(self._vitals_path) > 0
        )
        self._last_tb = -(10**12)
        self._last_img = -(10**12)
        self._last_html = -(10**12)
        
        # NeuroScore hook
        self.score = None
        if NeuroScore is not None:
            # Default score config
            sc_cfg = NeuroScoreConfig(enabled=True, update_every=100)
            self.score = NeuroScore(sc_cfg, neuroviz=self)

    # --------- registration & events (used by controller) ----------

    def register_model(self, model: nn.Module):
        """Record every SynapticMoE with a stable name (layer index)."""
        idx = 0
        for m in model.modules():
            if isinstance(m, SynapticMoE):
                name = f"moe_L{idx}"
                self.layers.append((name, m))
                if self.score is not None:
                    self.score.register_layer(name, m.num_experts)
                idx += 1

    def on_merge(
        self,
        moe: SynapticMoE,
        parent_i: int,
        parent_j: int,
        child_idx: int,
        step: Optional[int] = None,
    ):
        name = self._name_of(moe)
        if name:
            step = (
                step if step is not None else int(time.time())
            )  # fallback when caller didn't supply step
            self.lineage.log_merge(name, step, parent_i, parent_j, child_idx)

    def on_split(
        self,
        moe: SynapticMoE,
        parent_idx: int,
        child_idx: int,
        step: Optional[int] = None,
    ):
        name = self._name_of(moe)
        if name:
            step = step if step is not None else int(time.time())
            self.lineage.log_split(name, step, parent_idx, child_idx)

    def on_reset(
        self,
        moe: SynapticMoE,
        source_idx: int,
        reset_idx: int,
        step: Optional[int] = None,
    ):
        name = self._name_of(moe)
        if name:
            step = step if step is not None else int(time.time())
            self.lineage.log_reset(name, step, source_idx, reset_idx)

    def _name_of(self, moe: SynapticMoE) -> Optional[str]:
        for nm, m in self.layers:
            if m is moe:
                return nm
        return None

    def close(self):
        if self.tb is not None:
            self.tb.close()

    def _log_vitals(
        self, model: nn.Module, step: int, loss: Optional[torch.Tensor]
    ) -> None:
        del model  # reserved for future use
        fieldnames = [
            "step",
            "loss",
            "energy_mean",
            "health_mean",
            "utilization_mean",
            "dead_expert_frac",
        ]
        row: Dict[str, object] = {
            "step": step,
            "loss": float(loss.item()) if loss is not None else "",
            "energy_mean": "",
            "health_mean": "",
            "utilization_mean": "",
            "dead_expert_frac": "",
        }

        if self.layers:
            energy_means = []
            health_means = []
            utilization_means = []
            dead_fracs = []
            for _, moe in self.layers:
                m = self._layer_metrics(moe)
                energy_means.append(float(np.mean(m["energy"])))
                health_means.append(float(np.mean(m["health"])))
                utilization_means.append(float(np.mean(m["utilization"])))
                dead_fracs.append(
                    float(np.mean(m["health"] <= self.cfg.dead_health_threshold))
                )
            row["energy_mean"] = float(np.mean(energy_means))
            row["health_mean"] = float(np.mean(health_means))
            row["utilization_mean"] = float(np.mean(utilization_means))
            row["dead_expert_frac"] = float(np.mean(dead_fracs))

        try:
            with open(self._vitals_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not self._vitals_header_written:
                    writer.writeheader()
                    self._vitals_header_written = True
                writer.writerow(row)
        except Exception:
            # Never let CSV logging crash training.
            return

    @torch.no_grad()
    def _layer_metrics(self, moe: SynapticMoE) -> Dict[str, np.ndarray]:
        emb = _to_np(moe.router_embeddings)  # (E, D)
        fatigue = _to_np(moe.fatigue)  # (E,)
        energy = _to_np(moe.energy)  # (E,)
        
        # fatigue tracks EMA of usage. So it IS the utilization metric.
        utilization = np.clip(fatigue, 0.0, 1.0)
        
        # availability is the inverse of utilization
        availability = 1.0 - utilization
        
        # health = utilization * energy
        health = energy * utilization

        # expert-specific reductions
        mgate, camkii, elig = [], [], []
        for e in moe.experts:
            e = cast(SynapticExpert, e)
            mgate.append(_reduce_mgate(e))
            camkii.append(_reduce_camkii(e))
            elig.append(_reduce_elig_norm(e))

        mgate_arr = np.asarray(mgate, dtype=np.float32)
        camkii_arr = np.asarray(camkii, dtype=np.float32)
        elig_arr = np.asarray(elig, dtype=np.float32)

        # “quantal proxy”: weight norms of slow+fast in fc2 (downstream)
        qprox = []
        for e in moe.experts:
            e = cast(SynapticExpert, e)
            slow_norm = float(e.fc2.w_slow.norm().item())
            fast_norm = (
                float(e.fc2.w_fast.norm().item()) if e.fc2.w_fast is not None else 0.0
            )
            qprox.append(slow_norm + fast_norm)
        qprox_arr = np.asarray(qprox, dtype=np.float32)

        return dict(
            embedding=emb,
            utilization=utilization,
            availability=availability,
            energy=energy,
            health=health,
            mgate=mgate_arr,
            camkii=camkii_arr,
            elig=elig_arr,
            qprox=qprox_arr,
        )

    # ------------------------- per-step logging ---------------------

    def step(self, model: nn.Module, step: int, loss: Optional[torch.Tensor] = None):
        # ensure we have layers registered
        if not self.layers:
            self.register_model(model)

        # NeuroScore update (if active)
        if self.score is not None and loss is not None:
            self.score.step(model, loss, step)

        # TensorBoard scalars/hists
        if self.tb is not None and step - self._last_tb >= self.cfg.tb_every:
            for name, moe in self.layers:
                self._log_tb_layer(name, moe, step)
            self._last_tb = step

        # Vitals CSV is the dashboard's primary telemetry — keep it on its own
        # cadence, decoupled from TensorBoard availability (write_tensorboard
        # =False used to silently disable it).
        if step - getattr(self, "_last_vitals", -(10**9)) >= self.cfg.tb_every:
            self._log_vitals(model, step, loss)
            self._last_vitals = step

        # Static images
        if self.cfg.save_pngs and step - self._last_img >= self.cfg.image_every:
            for name, moe in self.layers:
                self._write_images(name, moe, step)
                self.lineage.render_timeline_png(name, step)
            self._last_img = step

        # Optional interactive HTML
        if (
            self.cfg.write_interactive_html
            and _HAS_PLOTLY
            and step - self._last_html >= self.cfg.interactive_every
        ):
            for name, moe in self.layers:
                self.lineage.render_interactive_html(name, step)
            self._last_html = step

    # ------------------------- TensorBoard writers ------------------

    def _log_tb_layer(self, name: str, moe: SynapticMoE, step: int):
        m = self._layer_metrics(moe)
        # scalars
        if self.tb is None:
            return
        self.tb.add_scalar(f"{name}/population", moe.num_experts, step)
        self.tb.add_scalar(f"{name}/util_mean", float(np.mean(m["utilization"])), step)
        self.tb.add_scalar(f"{name}/energy_mean", float(np.mean(m["energy"])), step)
        self.tb.add_scalar(f"{name}/health_mean", float(np.mean(m["health"])), step)
        dead_frac = float(np.mean(m["health"] <= self.cfg.dead_health_threshold))
        self.tb.add_scalar(f"{name}/dead_expert_frac", dead_frac, step)
        self.tb.add_scalar(f"{name}/mgate_mean", float(np.mean(m["mgate"])), step)
        self.tb.add_scalar(f"{name}/camkii_mean", float(np.mean(m["camkii"])), step)

        # histograms (downsample if huge)
        for key in ("utilization", "energy", "health", "mgate", "camkii", "elig", "qprox"):
            arr = m[key]
            self.tb.add_histogram(f"{name}/hist/{key}", arr, step)

        # embedding projector
        emb2d = _fit_2d(m["embedding"])
        meta = [
            f"id:{i} util:{m['utilization'][i]:.3f} E:{m['energy'][i]:.2f}"
            for i in range(moe.num_experts)
        ]
        # add_embedding expects N x D; for 2D, it still works; for larger, projector clusters in higher-D
        self.tb.add_embedding(
            torch.from_numpy(m["embedding"]),
            metadata=meta,
            tag=f"{name}/router_embedding",
            global_step=step,
        )

        # save a small 2D map as a figure
        fig = plt.figure(figsize=(5, 4))
        cs = m["health"] / (np.max(m["health"]) + 1e-6)
        plt.scatter(
            emb2d[:, 0],
            emb2d[:, 1],
            c=cs,
            s=30 + 200 * m["utilization"],
            cmap="viridis",
            edgecolors="k",
            linewidths=0.3,
            alpha=0.86,
        )
        plt.title(f"{name} — 2D map @ {step:,}")
        plt.axis("off")
        self.tb.add_figure(f"{name}/map2d", fig, step)
        plt.close(fig)

    # ------------------------- image writers ------------------------

    def _write_images(self, name: str, moe: SynapticMoE, step: int):
        outdir = os.path.join(self.cfg.log_dir, "images", name)
        _ensure_dir(outdir)
        m = self._layer_metrics(moe)
        # map
        emb2d = _fit_2d(m["embedding"])
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        size = 30 + 350 * m["utilization"] * m["health"]
        color = m["energy"] * m["camkii"] if np.max(m["camkii"]) > 0 else m["energy"]
        sc = ax.scatter(
            emb2d[:, 0],
            emb2d[:, 1],
            s=size,
            c=color,
            cmap="coolwarm",
            edgecolors="k",
            linewidths=0.3,
            alpha=0.9,
        )
        ax.set_title(f"{name} map — step {step:,} | E={moe.num_experts}")
        ax.axis("off")
        fig.colorbar(sc, ax=ax, shrink=0.8, label="energy×CaMKII")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{name}_map_{step:09d}.png"), dpi=140)
        plt.close(fig)

        # radar for top-N
        self._radar(name, m, step, outdir, self.cfg.top_n_radar)

        # histograms
        self._hists(name, m, step, outdir)
        
        # contribution plot (Bio vs Static)
        self._contribution_plot(name, moe, step, outdir)
        
        # educational plots (Presyn, Hebbian, Raster)
        self._plot_presynaptic_dynamics(name, moe, step, outdir)
        self._plot_hebbian_memory(name, moe, step, outdir)
        self._plot_expert_raster(name, moe, step, outdir)
        
        # genetics and metabolism
        self._plot_genetics(name, moe, step, outdir)
        self._plot_metabolism(name, moe, step, outdir)
        

        # NOTE: no "router decision breakdown" here — the old _plot_router_decision
        # gated on last_ctx["decision_data"], a key NOTHING in the repo writes,
        # so it could never emit an artifact (removed with its dead dashboard page).

    def _plot_genetics(self, name: str, moe: SynapticMoE, step: int, outdir: str):
        if not hasattr(moe, "Xi") or not hasattr(moe, "_get_phenotype"):
            return
            
        pheno = moe._get_phenotype(moe.Xi) # (E, 4)
        pheno_np = _to_np(pheno)
        
        # [0] fatigue rate, [1] energy refill, [2] camkii, [3] pp1
        data = {
            "fatigue_rate": pheno_np[:, 0],
            "energy_refill": pheno_np[:, 1],
            "camkii_gain": pheno_np[:, 2],
            "pp1_gain": pheno_np[:, 3],
            "utilization": _to_np(moe.fatigue) # Use fatigue as proxy for util history
        }
        self._save_json(data, os.path.join(outdir, f"{name}_genetics_{step:09d}.json"))

    def _plot_metabolism(self, name: str, moe: SynapticMoE, step: int, outdir: str):
        energy = _to_np(moe.energy)
        fatigue = _to_np(moe.fatigue)
        
        # Sort by energy to show inequality
        sorted_idx = np.argsort(energy)
        
        data = {
            "energy": energy[sorted_idx],
            "fatigue": fatigue[sorted_idx],
            "ids": sorted_idx.tolist()
        }
        self._save_json(data, os.path.join(outdir, f"{name}_metabolism_{step:09d}.json"))

    def _save_json(self, data: Dict[str, Any], path: str):
        # Atomic tmp+replace: the Streamlit dashboard re-reads these artifacts
        # every ~2s in live mode; a mid-write read of the final path used to
        # raise JSONDecodeError and kill the whole app.

        def default(obj):
            if isinstance(obj, (np.ndarray, np.generic)):
                return obj.tolist()
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().tolist()
            return str(obj)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, default=default)
        os.replace(tmp, path)


    def _plot_presynaptic_dynamics(self, name: str, moe: SynapticMoE, step: int, outdir: str):
        # Simulate "Boredom": Attend to the same token repeatedly
        # We need a SynapticPresyn instance. We can't easily grab one from MoE (it's in Attention).
        # But we can instantiate a dummy one with the same config.
        from .synaptic import SynapticPresyn, build_presyn_state
        
        cfg = moe.cfg
        head_dim = 64 # Assumption, but doesn't matter for this simulation as we fake logits
        pre = SynapticPresyn(head_dim, cfg).to("cpu")
        
        T = 50
        # Scenario: Attend to token 0 for 20 steps, then token 1 for 20 steps
        logits = torch.zeros(1, 1, T, T)
        # Causal mask
        mask = torch.tril(torch.ones(T, T)).bool()
        
        # Set high logits for target
        # Steps 0-20: target 0
        logits[:, :, :25, 0] = 20.0
        # Steps 20-40: target 1
        logits[:, :, 25:, 1] = 20.0
        
        # Dummy q, k (needed for docking, but we can set them to be compatible)
        q = torch.randn(1, 1, T, head_dim)
        k = torch.randn(1, 1, T, head_dim)
        
        state = build_presyn_state(1, T, 1, "cpu", torch.float32, cfg)
        
        # Run forward
        # SynapticPresyn.forward returns (syn_logit, new_state)
        # But it updates state in-place or returns new tensors? It returns new tensors.
        # And it processes the whole sequence at once (parallel).
        
        with torch.no_grad():
            syn_logit, final_state = pre(q, k, logits, state, mask, train_mode=False)
            
        # We want to see the time-evolution of RRP, C, Release for the *target* tokens.
        # RRP is (B,H,T). It represents the pool available *at step t*.
        # Actually RRP is per-key (T_k). 
        # Wait, in SynapticPresyn:
        # RRP is (B,H,T) -> This is the RRP of the *key* token?
        # Yes: "RRP_refill = (rho_r * RRP + ...)"
        # And "used_rrp = release_frac.sum(dim=2)" -> sum over queries? No, dim=2 is T_q?
        # Let's check presyn code:
        # raw_release: (B,H,T_q, T_k)
        # used_rrp = release_frac.sum(dim=2) -> Sum over queries attending to this key.
        # So RRP[t] is the vesicle pool of token t (acting as a Key).
        
        # So we want to plot RRP of Token 0 and Token 1.
        rrp = final_state["RRP"][0, 0].numpy() # (T,)
        c_val = final_state["C"][0, 0].numpy()
        
        # Release probability? 
        # We can infer it from syn_logit or just re-calculate.
        # syn_logit is (B,H,T,T).
        # Let's look at syn_logit[:,:,t,0] (attention to token 0 at step t)
        syn_adjust = syn_logit[0, 0, :, 0].numpy() # (T,)
        
        # Save data for interactive dashboard
        data = {
            "rrp": rrp,
            "calcium": c_val,
            "logit_delta": syn_adjust,
            "steps": list(range(T))
        }
        self._save_json(data, os.path.join(outdir, f"{name}_presyn_{step:09d}.json"))
        
        fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        
        # Plot RRP of Token 0
        ax[0].plot(rrp, label="RRP (Vesicles)", color="green", lw=2)
        ax[0].set_title("Presynaptic State (Token 0)")
        ax[0].set_ylabel("Pool Size")
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)
        
        # Plot Calcium of Token 0
        ax[1].plot(c_val, label="Calcium (Excitement)", color="orange", lw=2)
        ax[1].set_ylabel("Concentration")
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)
        
        # Plot Synaptic Adjustment to Token 0
        ax[2].plot(syn_adjust, label="Logit Adjustment", color="red", lw=2)
        ax[2].set_ylabel("Logit Delta")
        ax[2].set_xlabel("Time Step")
        ax[2].legend()
        ax[2].grid(True, alpha=0.3)
        
        ax[2].text(5, -2, "Attending to Token 0...", fontsize=9, color="gray")
        ax[2].text(30, -2, "Switched to Token 1", fontsize=9, color="gray")
        
        fig.suptitle(f"The 'Boredom' Mechanism (Simulated) @ {step}")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{name}_presyn_{step:09d}.png"), dpi=120)
        plt.close(fig)

    def _plot_hebbian_memory(self, name: str, moe: SynapticMoE, step: int, outdir: str):
        # Visualize a low-rank Hebbian trace (u_buf @ v_buf) of an expert
        # Find most active expert from last_ctx
        if not hasattr(moe, "last_ctx") or not moe.last_ctx:
            return
            
        # We can also just pick expert 0 for consistency
        e_idx = 0
        expert = cast(SynapticExpert, moe.experts[e_idx])
        
        # u_buf @ v_buf gives a low-rank trace (d_in, d_out). We'll take a slice.
        if not hasattr(expert.fc1, "u_buf") or not hasattr(expert.fc1, "v_buf"):
            return
        if expert.fc1.u_buf is None or expert.fc1.v_buf is None:
            return

        u_sub = expert.fc1.u_buf[:50].detach().float().cpu()
        v_sub = expert.fc1.v_buf[:, :50].detach().float().cpu()
        H_sub = (u_sub @ v_sub).numpy()
        
        # Save data
        self._save_json({"heatmap": H_sub}, os.path.join(outdir, f"{name}_hebbian_{step:09d}.json"))
        
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(H_sub, cmap="RdBu_r", vmin=-0.01, vmax=0.01)
        ax.set_title(f"Hebbian Trace (Expert {e_idx}) - Short Term Memory")
        ax.set_xlabel("Output Dim")
        ax.set_ylabel("Input Dim")
        fig.colorbar(im, ax=ax)
        
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{name}_hebbian_{step:09d}.png"), dpi=120)
        plt.close(fig)

    def _plot_expert_raster(self, name: str, moe: SynapticMoE, step: int, outdir: str):
        if not hasattr(moe, "last_ctx") or not moe.last_ctx:
            return
            
        # gates: (B, T, k)
        gates = _to_np(moe.last_ctx["gates"])
        indices = _to_np(moe.last_ctx["indices"])
        
        # Flatten B*T
        B, T, k = gates.shape
        flat_gates = gates.reshape(-1, k)
        flat_indices = indices.reshape(-1, k)
        
        # Take first 100 tokens
        L = min(100, B*T)
        
        # Create a matrix (Experts, Time)
        E = moe.num_experts
        raster = np.zeros((E, L))
        
        for t in range(L):
            for i in range(k):
                idx = int(flat_indices[t, i])
                val = flat_gates[t, i]
                if idx < E:
                    raster[idx, t] = val
        
        # Save data
        self._save_json({"raster": raster}, os.path.join(outdir, f"{name}_raster_{step:09d}.json"))
                    
        fig, ax = plt.subplots(figsize=(10, 6))
        # Use a dark background for "brain scan" look
        im = ax.imshow(raster, aspect="auto", cmap="magma", interpolation="nearest")
        ax.set_title(f"Expert Activation Raster (First {L} tokens)")
        ax.set_xlabel("Token Time")
        ax.set_ylabel("Expert ID")
        fig.colorbar(im, ax=ax, label="Gate Probability")
        
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{name}_raster_{step:09d}.png"), dpi=120)
        plt.close(fig)

    def _contribution_plot(self, name: str, moe: SynapticMoE, step: int, outdir: str):
        # Visualize the magnitude of fast weights vs slow weights
        # We'll sample a few experts
        experts = moe.experts[:min(5, len(moe.experts))]
        
        slow_norms = []
        fast_norms = []
        ids = []
        
        for i, e in enumerate(experts):
            e = cast(SynapticExpert, e)
            # L2 norm of weights
            s = e.fc1.w_slow.norm().item() + e.fc2.w_slow.norm().item()
            f = 0.0
            if e.fc1.w_fast is not None:
                f += e.fc1.w_fast.norm().item()
            if e.fc2.w_fast is not None:
                f += e.fc2.w_fast.norm().item()
            # Add Hebbian trace norm if available
            if getattr(e.fc1, "u_buf", None) is not None and getattr(e.fc1, "v_buf", None) is not None:
                f += e.fc1.u_buf.norm().item() + e.fc1.v_buf.norm().item()
            if getattr(e.fc2, "u_buf", None) is not None and getattr(e.fc2, "v_buf", None) is not None:
                f += e.fc2.u_buf.norm().item() + e.fc2.v_buf.norm().item()
                
            slow_norms.append(s)
            fast_norms.append(f)
            ids.append(f"Exp {i}")
            
        # Save data
        self._save_json({
            "ids": ids,
            "slow_norms": slow_norms,
            "fast_norms": fast_norms
        }, os.path.join(outdir, f"{name}_contrib_{step:09d}.json"))
            
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(ids))
        width = 0.35
        
        ax.bar(x - width/2, slow_norms, width, label='Slow (Static)', color='#4472C4')
        ax.bar(x + width/2, fast_norms, width, label='Fast (Bio)', color='#ED7D31')
        
        ax.set_ylabel('Weight Norm (L2)')
        ax.set_title(f'{name} - Static vs Bio Weight Magnitude')
        ax.set_xticks(x)
        ax.set_xticklabels(ids)
        ax.legend()
        
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{name}_contrib_{step:09d}.png"), dpi=120)
        plt.close(fig)

    def _radar(
        self,
        name: str,
        m: Dict[str, np.ndarray],
        step: int,
        outdir: str,
        top_n: int = 6,
    ):
        N = min(top_n, len(m["utilization"]))
        order = np.argsort(-m["utilization"])[:N]
        labels = ["utilization", "energy", "camkii", "mgate", "elig", "qprox"]
        K = len(labels)
        th = np.linspace(0, 2 * np.pi, K, endpoint=False)
        
        # Compute global max per metric for fair comparison
        max_vals = np.array([
            np.max(m[key]) + 1e-6 for key in labels
        ], dtype=np.float32)
        
        fig = plt.figure(figsize=(7, 7))
        ax = plt.subplot(111, polar=True)
        for idx in order:
            vals = np.array(
                [
                    m["utilization"][idx],
                    m["energy"][idx],
                    m["camkii"][idx],
                    m["mgate"][idx],
                    m["elig"][idx],
                    m["qprox"][idx],
                ],
                dtype=np.float32,
            )
            # normalize by population max
            v = vals / max_vals
            ax.plot(np.r_[th, th[0]], np.r_[v, v[0]], lw=2, label=f"id {idx}")
            ax.fill(np.r_[th, th[0]], np.r_[v, v[0]], alpha=0.15)
        ax.set_xticks(th)
        ax.set_xticklabels(labels)
        ax.set_title(f"{name} — top-{N} profiles @ {step:,} (norm by pop max)")
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0), fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{name}_radar_{step:09d}.png"), dpi=140)
        plt.close(fig)

    def _hists(self, name: str, m: Dict[str, np.ndarray], step: int, outdir: str):
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        keys = ["utilization", "energy", "health", "camkii", "mgate", "qprox"]
        for ax, key in zip(axes.ravel(), keys):
            ax.hist(m[key], bins=20, color="#4472C4", alpha=0.85)
            ax.set_title(key)
            ax.grid(True, alpha=0.2)
        fig.suptitle(f"{name} distributions @ {step:,}")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(os.path.join(outdir, f"{name}_hists_{step:09d}.png"), dpi=140)
        plt.close(fig)


# --------------------------------------------------------------------------- #
# NeuroViz v2 — synaptic bio-panel collection + dependency-light HTML
# dashboard (bead dow). Pure addition: consumes the live model's module
# state (postsynaptic fast/slow weights, CaMKII/PP1/BDNF, MoE metabolism)
# and optionally a per-position presynaptic trace from a non-mutating
# KV-cache forward. Renders a static self-contained HTML dashboard plus a
# JSON sidecar; no matplotlib required for the HTML path.
# --------------------------------------------------------------------------- #


def _panel_sparkline(values: List[float], width: int = 240, height: int = 48) -> str:
    """One inline SVG sparkline for a 1-D series (dependency-free HTML path)."""
    vals = [float(v) for v in values if np.isfinite(v)] or [0.0]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    n = max(len(vals), 2)
    pts = " ".join(
        f"{width * i / (n - 1):.1f},{height - height * (v - lo) / span:.1f}"
        for i, v in enumerate(vals)
    )
    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f'<polyline points="{pts}" fill="none" stroke="#4472C4" stroke-width="2"/></svg>'
    )


def collect_bio_panels(
    model: nn.Module,
    *,
    probe_batch: Optional[Tensor] = None,
    n_head: Optional[int] = None,
    n_layer: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Collect the dow v2 panels from a live synaptic model.

    Panels (all values JSON-serializable floats/ints):
      ``postsyn``  — fast/slow weight norms, CaMKII / PP1 / BDNF means, latch fraction.
      ``moe``      — per-layer energy / fatigue / health means + utilization gini
                     (empty dict for dense models).
      ``presyn_trace`` — only with ``probe_batch``: per-position mean calcium (C),
                     vesicle pool (RRP), and energy (E) from the deepest layer's
                     KV-cache presyn state via a NON-MUTATING eval forward
                     (grad enabled + train_mode=False so no plasticity runs — same
                     contract as scripts/e2e/train_bio.presyn_position_trace).
    """
    from bio_inspired_nanochat.synaptic import PostsynapticHebb, SynapticLinear

    posts = [m for m in model.modules() if isinstance(m, PostsynapticHebb)]
    lins = [
        m for m in model.modules()
        if isinstance(m, SynapticLinear) and m.post is not None
    ]

    postsyn: Dict[str, float] = {}
    if lins:
        fast_norms = [float(m.post.fast.detach().norm()) for m in lins]
        slow_norms = [float(m.w_slow.detach().norm()) for m in lins]
        camkii = torch.cat([m.camkii.detach().flatten() for m in posts]) if posts else None
        pp1 = torch.cat([m.pp1.detach().flatten() for m in posts]) if posts else None
        bdnf_vals = (
            torch.cat([m.bdnf.detach().flatten() for m in posts])
            if posts and all(hasattr(m, "bdnf") for m in posts) else None
        )
        thr = float(posts[0].cfg.camkii_thr) if posts else 1.0
        postsyn.update({
            "fast_weight_norm_mean": float(np.mean(fast_norms)),
            "fast_weight_norm_max": float(np.max(fast_norms)),
            "slow_weight_norm_mean": float(np.mean(slow_norms)),
            "latched_frac": float((camkii > thr).float().mean()) if camkii is not None else 0.0,
            "camkii_mean": float(camkii.mean()) if camkii is not None else 0.0,
            "pp1_mean": float(pp1.mean()) if pp1 is not None else 0.0,
            "bdnf_mean": float(bdnf_vals.mean()) if bdnf_vals is not None else 0.0,
            "n_postsyn_modules": len(lins),
        })

    moe: Dict[str, Any] = {}
    for idx, m in enumerate(m for m in model.modules() if isinstance(m, SynapticMoE)):
        energy = _to_np(m.energy.detach())
        fatigue = _to_np(m.fatigue.detach())
        health = _weight_energy_util(energy, fatigue)
        util = getattr(m, "utilization", None)
        if util is not None:
            u = np.sort(_to_np(util.detach()).astype(np.float64).ravel())
            k = max(len(u), 1)
            gini = float((2.0 * np.sum((np.arange(1, k + 1)) * u) / (k * u.sum() + 1e-12)) - (k + 1) / k)
        else:
            gini = float("nan")
        moe[f"layer_{idx}"] = {
            "energy_mean": float(energy.mean()),
            "fatigue_mean": float(fatigue.mean()),
            "health_mean": float(np.nanmean(health)) if health.size else 0.0,
            "utilization_gini": gini,
            "num_experts": int(getattr(m, "num_experts", 0)),
        }

    panels: Dict[str, Dict[str, Any]] = {"postsyn": postsyn, "moe": moe}

    if probe_batch is not None:
        from bio_inspired_nanochat.engine import KVCache

        cfg = model.config
        heads = int(n_head if n_head is not None else getattr(cfg, "n_head", 1))
        layers_n = int(n_layer if n_layer is not None else getattr(cfg, "n_layer", 1))
        B, T = int(probe_batch.shape[0]), int(probe_batch.shape[1])
        kv = KVCache(
            batch_size=B, num_heads=heads, seq_len=T,
            head_dim=int(cfg.n_embd) // heads, num_layers=layers_n,
        )
        was_training = model.training
        model.eval()
        try:  # grad ON + train_mode=False: plasticity stays gated OFF (see train_bio probe)
            model(probe_batch, None, kv_cache=kv, train_mode=False)
        finally:
            if was_training:
                model.train()
        states = kv.presyn_state if isinstance(kv.presyn_state, list) else []
        trace: Dict[str, List[float]] = {}
        if states:
            last = states[-1]
            for key in ("C", "RRP", "E"):
                v = last.get(key)
                if torch.is_tensor(v) and v.dim() >= 1:
                    trace[key] = [
                        float(x) for x in
                        v.detach().float().reshape(-1, v.shape[-1]).mean(dim=0)
                    ]
        panels["presyn_trace"] = cast(Dict[str, Any], trace)
    return panels


def render_html_dashboard(
    panels: Dict[str, Dict[str, Any]],
    out_path: str,
    *,
    title: str = "NeuroViz v2 — bio panels",
) -> str:
    """Render panels as one self-contained static HTML file. Returns the path written."""
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{title}</title>",
        "<style>body{font-family:sans-serif;margin:24px;background:#fafafa}",
        "h2{border-bottom:2px solid #4472C4;padding-bottom:4px}",
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 10px}",
        ".metric{color:#333}</style></head><body>",
        f"<h1>{title}</h1>",
    ]
    postsyn = panels.get("postsyn") or {}
    if postsyn:
        parts.append("<h2>Postsynaptic plasticity</h2><table>")
        for k, v in postsyn.items():
            parts.append(f"<tr><th class='metric'>{k}</th><td>{v:.5g}</td></tr>")
        parts.append("</table>")
    moe = panels.get("moe") or {}
    if moe:
        parts.append("<h2>MoE metabolism &amp; specialization</h2><table>")
        for layer, stats in moe.items():
            cells = " ".join(f"<td>{k}={v:.4g}</td>" for k, v in stats.items())
            parts.append(f"<tr><th>{layer}</th>{cells}</tr>")
        parts.append("</table>")
    trace = panels.get("presyn_trace") or {}
    if trace:
        parts.append("<h2>Presynaptic per-position traces</h2>")
        for key, series in trace.items():
            parts.append(f"<div><b>{key}</b> {_panel_sparkline(series)}</div>")
    parts.append("</body></html>")
    _ensure_dir(os.path.dirname(out_path) or ".")
    with open(out_path, "w") as f:
        f.write("\n".join(parts))
    return out_path


def write_bio_dashboard(
    manager_or_model: Any,
    step: int,
    *,
    probe_batch: Optional[Tensor] = None,
    log_dir: Optional[str] = None,
) -> Dict[str, str]:
    """Collect panels and write ``bio_panels_<step>.json`` + ``bio_dashboard_<step>.html``.

    Accepts either a :class:`NeuroVizManager` (uses its ``log_dir``/TensorBoard writer)
    or a bare model plus an explicit ``log_dir``. Returns the artifact paths.
    """
    if hasattr(manager_or_model, "cfg") and hasattr(manager_or_model, "layers"):
        manager = manager_or_model
        model = next(m for _, m in manager.layers) if manager.layers else None
        if model is None:
            raise ValueError("manager has no registered SynapticMoE; pass a model instead")
        out_dir = log_dir or manager.cfg.log_dir
    else:
        manager, model = None, manager_or_model
        out_dir = log_dir or "runs/neuroviz"
    panels = collect_bio_panels(model, probe_batch=probe_batch)
    _ensure_dir(out_dir)
    json_path = os.path.join(out_dir, f"bio_panels_{step:09d}.json")
    with open(json_path, "w") as f:
        json.dump(panels, f, indent=2, default=float)
    html_path = render_html_dashboard(
        panels, os.path.join(out_dir, f"bio_dashboard_{step:09d}.html")
    )
    if manager is not None and manager.tb is not None:
        postsyn = panels.get("postsyn") or {}
        for k, v in postsyn.items():
            manager.tb.add_scalar(f"bio_v2/{k}", float(v), step)
    return {"json": json_path, "html": html_path}
