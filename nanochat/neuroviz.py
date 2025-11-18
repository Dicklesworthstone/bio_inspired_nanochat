# nanochat/neuroviz.py
# ---------------------------------------------------------------------
# Visualization & logging for Synaptic-MoE:
#  - NeuroVizManager: orchestrates TensorBoard + static figures + lineage
#  - LineageBook: split/merge event ledger + timeline renders
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

import os
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from umap import UMAP

    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False

try:
    from sklearn.decomposition import PCA

    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

try:
    import plotly.graph_objects as go

    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

from torch.utils.tensorboard import SummaryWriter

# lazy import to avoid circulars
try:
    from .synaptic import SynapticMoE, SynapticExpert
except Exception:
    from synaptic import SynapticMoE, SynapticExpert


# --------------------------- utilities --------------------------------


def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


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
            vals.append(float(fc.post.camkii.item()))
    return float(np.mean(vals)) if vals else 0.0


def _reduce_mgate(expert: SynapticExpert) -> float:
    vals = []
    for fc in (expert.fc1, expert.fc2):
        if hasattr(fc, "post") and hasattr(fc.post, "m_gate"):
            vals.append(float(fc.post.m_gate.item()))
    return float(np.mean(vals)) if vals else 0.0


def _reduce_elig_norm(expert: SynapticExpert) -> float:
    vals = []
    for fc in (expert.fc1, expert.fc2):
        if hasattr(fc, "post"):
            vals.append(
                float(
                    fc.post.U.norm().item()
                    + fc.post.V.norm().item()
                    + fc.post.H_fast.norm().item()
                )
            )
    return float(np.mean(vals)) if vals else 0.0


def _weight_energy_util(energy: np.ndarray, fatigue: np.ndarray) -> np.ndarray:
    energy = np.clip(energy, 0.0, 1.0)
    util = 1.0 - np.clip(_to_np(fatigue), 0.0, 1.0)
    return energy * util


def _fit_2d(emb: np.ndarray) -> np.ndarray:
    if emb.shape[1] <= 2:
        return emb
    if _HAS_UMAP:
        red = UMAP(
            n_neighbors=min(15, emb.shape[0] - 1),
            min_dist=0.3,
            metric="cosine",
            random_state=42,
        )
        return red.fit_transform(emb)
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
    Keeps a per-layer log of events (split/merge). Provides:
      - timeline PNG render
      - optional interactive HTML (plotly)
    """

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        _ensure_dir(save_dir)
        # layer_id -> list of events [(step, "merge", w,l,child), (step, "split", parent, child)]
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
        # very lightweight: just scatter the events along time with text labels
        xs, ys, texts, colors = [], [], [], []
        ymap = {}
        ycur = 0
        for s, et, ids in evs:
            if et == "merge":
                i, j, c = ids
                for eid in (i, j, c):
                    if eid not in ymap:
                        ymap[eid] = ycur
                        ycur += 1
                xs += [s, s, s]
                ys += [ymap[i], ymap[j], ymap[c]]
                texts += [f"merge parent {i}", f"merge parent {j}", f"child {c}"]
                colors += ["purple", "purple", "purple"]
            elif et == "split":
                p, c = ids
                for eid in (p, c):
                    if eid not in ymap:
                        ymap[eid] = ycur
                        ycur += 1
                xs += [s, s]
                ys += [ymap[p], ymap[c]]
                texts += [f"split parent {p}", f"child {c}"]
                colors += ["green", "green"]
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers+text",
                    text=texts,
                    marker=dict(color=colors, size=10),
                )
            ]
        )
        fig.update_layout(
            title=f"Interactive Lineage — {layer_name} @ step {step:,}",
            xaxis_title="step",
            yaxis_title="expert track",
            template="plotly_dark",
            height=500,
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
        self._last_tb = -(10**12)
        self._last_img = -(10**12)
        self._last_html = -(10**12)

    # --------- registration & events (used by controller) ----------

    def register_model(self, model: nn.Module):
        """Record every SynapticMoE with a stable name (layer index)."""
        idx = 0
        for m in model.modules():
            if isinstance(m, SynapticMoE):
                name = f"moe_L{idx}"
                self.layers.append((name, m))
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

    def _name_of(self, moe: SynapticMoE) -> Optional[str]:
        for nm, m in self.layers:
            if m is moe:
                return nm
        return None

    # ------------------------- per-step logging ---------------------

    def step(self, model: nn.Module, step: int):
        # ensure we have layers registered
        if not self.layers:
            self.register_model(model)

        # TensorBoard scalars/hists
        if self.tb is not None and step - self._last_tb >= self.cfg.tb_every:
            for name, moe in self.layers:
                self._log_tb_layer(name, moe, step)
            self._last_tb = step

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

    def close(self):
        if self.tb is not None:
            self.tb.close()

    # ------------------------- metrics extraction -------------------

    @torch.no_grad()
    def _layer_metrics(self, moe: SynapticMoE) -> Dict[str, np.ndarray]:
        emb = _to_np(moe.router_embeddings)  # (E, D)
        fatigue = _to_np(moe.fatigue)  # (E,)
        energy = _to_np(moe.energy)  # (E,)
        util = 1.0 - np.clip(fatigue, 0.0, 1.0)  # util proxy
        health = _weight_energy_util(energy, fatigue)

        # expert-specific reductions
        mgate, camkii, elig = [], [], []
        for e in moe.experts:
            mgate.append(_reduce_mgate(e))
            camkii.append(_reduce_camkii(e))
            elig.append(_reduce_elig_norm(e))

        mgate = np.asarray(mgate, dtype=np.float32)
        camkii = np.asarray(camkii, dtype=np.float32)
        elig = np.asarray(elig, dtype=np.float32)

        # “quantal proxy”: weight norms of slow+fast in fc2 (downstream)
        qprox = []
        for e in moe.experts:
            s = float(e.fc2.w_slow.norm().item() + e.fc2.w_fast.norm().item())
            qprox.append(s)
        qprox = np.asarray(qprox, dtype=np.float32)

        return dict(
            embedding=emb,
            util=util,
            energy=energy,
            health=health,
            mgate=mgate,
            camkii=camkii,
            elig=elig,
            qprox=qprox,
        )

    # ------------------------- TensorBoard writers ------------------

    def _log_tb_layer(self, name: str, moe: SynapticMoE, step: int):
        m = self._layer_metrics(moe)
        # scalars
        if self.tb is None:
            return
        self.tb.add_scalar(f"{name}/population", moe.num_experts, step)
        self.tb.add_scalar(f"{name}/util_mean", float(np.mean(m["util"])), step)
        self.tb.add_scalar(f"{name}/energy_mean", float(np.mean(m["energy"])), step)
        self.tb.add_scalar(f"{name}/health_mean", float(np.mean(m["health"])), step)
        self.tb.add_scalar(f"{name}/mgate_mean", float(np.mean(m["mgate"])), step)
        self.tb.add_scalar(f"{name}/camkii_mean", float(np.mean(m["camkii"])), step)

        # histograms (downsample if huge)
        for key in ("util", "energy", "health", "mgate", "camkii", "elig", "qprox"):
            arr = m[key]
            self.tb.add_histogram(f"{name}/hist/{key}", arr, step)

        # embedding projector
        emb2d = _fit_2d(m["embedding"])
        meta = [
            f"id:{i} util:{m['util'][i]:.3f} E:{m['energy'][i]:.2f}"
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
            s=30 + 200 * m["util"],
            cmap="viridis",
            edgecolors="k",
            linewidths=0.3,
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
        size = 30 + 350 * m["util"] * m["health"]
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

    def _radar(
        self,
        name: str,
        m: Dict[str, np.ndarray],
        step: int,
        outdir: str,
        top_n: int = 6,
    ):
        N = min(top_n, len(m["util"]))
        order = np.argsort(-m["util"])[:N]
        labels = ["util", "energy", "camkii", "mgate", "elig", "qprox"]
        K = len(labels)
        th = np.linspace(0, 2 * np.pi, K, endpoint=False)
        fig = plt.figure(figsize=(7, 7))
        ax = plt.subplot(111, polar=True)
        for idx in order:
            vals = np.array(
                [
                    m["util"][idx],
                    m["energy"][idx],
                    m["camkii"][idx],
                    m["mgate"][idx],
                    m["elig"][idx],
                    m["qprox"][idx],
                ],
                dtype=np.float32,
            )
            # normalize each axis for readability
            vmax = np.maximum(vals.max(), 1e-6)
            v = vals / (vmax + 1e-6)
            ax.plot(np.r_[th, th[0]], np.r_[v, v[0]], lw=2, label=f"id {idx}")
            ax.fill(np.r_[th, th[0]], np.r_[v, v[0]], alpha=0.15)
        ax.set_xticks(th)
        ax.set_xticklabels(labels)
        ax.set_title(f"{name} — top-{N} profiles @ {step:,}")
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0), fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{name}_radar_{step:09d}.png"), dpi=140)
        plt.close(fig)

    def _hists(self, name: str, m: Dict[str, np.ndarray], step: int, outdir: str):
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        keys = ["util", "energy", "health", "camkii", "mgate", "qprox"]
        for ax, key in zip(axes.ravel(), keys):
            ax.hist(m[key], bins=20, color="#4472C4", alpha=0.85)
            ax.set_title(key)
            ax.grid(True, alpha=0.2)
        fig.suptitle(f"{name} distributions @ {step:,}")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(outdir, f"{name}_hists_{step:09d}.png"), dpi=140)
        plt.close(fig)
