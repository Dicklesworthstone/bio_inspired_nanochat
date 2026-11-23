"""Summarize benchmark NVTX ranges from an Nsight Systems SQLite export."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rich.console import Console

from bio_inspired_nanochat.perf_regression import STANDARD_BENCHMARK_CONFIGS


@dataclass
class Window:
    name: str
    steps: int
    wall_ms: float
    kernel_count: int
    kernel_busy_ms: float
    kernel_busy_percent: float
    kernel_time_by_name_ms: dict[str, float]
    runtime_api_time_by_name_ms: dict[str, float]
    kernel_busy_percent_by_bin: list[float]


def _top_ms(totals: Counter[str]) -> dict[str, float]:
    return {name: duration / 1e6 for name, duration in totals.most_common(8)}


def _window(connection: sqlite3.Connection, name: str, steps: int) -> Window:
    ranges = connection.execute(
        "SELECT start, end, globalTid FROM NVTX_EVENTS WHERE text = ? AND end > start", (name,)
    ).fetchall()
    if len(ranges) != 1:
        raise ValueError(f"expected one measured window for {name}, found {len(ranges)}")
    start, end, tid = ranges[0]
    # Nsight serializes the process identity in the upper bits of GlobalId.
    pid = (tid >> 24) << 24
    kernels = connection.execute(
        "SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON s.id = k.shortName WHERE k.globalPid = ? "
        "AND k.start < ? AND k.end > ? ORDER BY k.start, k.end", (pid, end, start)
    ).fetchall()
    if not kernels:
        raise ValueError(f"no GPU kernels overlap {name}")
    api = connection.execute(
        "SELECT s.value, SUM(MIN(r.end, ?) - MAX(r.start, ?)) FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN StringIds s ON s.id = r.nameId WHERE (r.globalTid >> 24) = ? "
        "AND r.start < ? AND r.end > ? GROUP BY s.value", (end, start, tid >> 24, end, start)
    )
    totals: Counter[str] = Counter()
    bins = [0] * 100
    span = end - start
    frontier = start
    for first, last, kernel_name in kernels:
        first, last = max(first, start), min(last, end)
        totals[kernel_name] += last - first
        # Union across streams counts overlapping execution once. Split each
        # newly covered interval on integer boundaries to conserve nanoseconds.
        pos = max(frontier, first)
        while pos < last:
            idx = min((pos - start) * len(bins) // span, len(bins) - 1)
            boundary = start + ((idx + 1) * span + len(bins) - 1) // len(bins)
            stop = min(boundary, last)
            bins[idx] += stop - pos
            pos = stop
        frontier = max(frontier, last)
    busy = sum(bins)
    return Window(name, steps, span / 1e6, len(kernels), busy / 1e6, 100 * busy / span,
                  _top_ms(totals), _top_ms(Counter(dict(api))),
                  [100 * ticks * len(bins) / span for ticks in bins])


def _draw(windows: list[Window], destination: Path) -> None:
    fig, axes = plt.subplots(len(windows), 1, figsize=(10, 7.5), sharex=True, sharey=True)
    for ax, item in zip(axes, windows, strict=True):
        color = "#176978" if item.name.startswith("synaptic") else "#385485"
        ax.bar(range(100), item.kernel_busy_percent_by_bin, width=1, align="edge", color=color)
        ax.set_title(item.name.replace("_transformer", "").replace("_cached_tq1", " cached").replace("_", " "),
                     loc="left", fontsize=11, fontweight="bold")
        ax.set_title(f"{item.kernel_count / item.steps:.0f} kernels/step  ·  "
                     f"{item.kernel_busy_percent:.2f}% kernel-active  ·  {item.wall_ms:.0f} ms window",
                     loc="right", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xlim(0, 100)
    axes[-1].set_xlabel("Elapsed fraction of each measured window (%)")
    fig.supylabel("Wall time with a GPU kernel active per bin (%)", fontsize=11)
    fig.suptitle("CUDA work during training and cached decode", fontsize=17, x=0.09, ha="left")
    fig.text(0.09, 0.935, "Measured NVTX windows only. Training uses five optimizer updates. Decode uses 200 one-token forwards.", fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.87, bottom=0.09, hspace=0.61)
    fig.savefig(destination, format="svg", metadata={"Date": None})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path, help="SQLite export from nsys export --type=sqlite")
    parser.add_argument("--out", type=Path, required=True, help="prefix for .json and .svg outputs")
    args = parser.parse_args()
    database: Path = args.database.resolve()
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        windows = [_window(connection, cfg.name, cfg.measure_steps) for cfg in STANDARD_BENCHMARK_CONFIGS]
    with database.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {"sqlite_sha256": digest, "scope": "Four profiled, synchronized CUDA benchmark windows. "
              "Kernel-name times sum individual launches. Busy time is their union. CUDA API durations "
              "may overlap across threads. Time outside kernels includes host work, scheduling and profiler overhead.",
              "windows": [asdict(item) for item in windows]}
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _draw(windows, args.out.with_suffix(".svg"))
    Console().print(f"Wrote four GPU timing windows to {args.out}.json and {args.out}.svg")


if __name__ == "__main__":
    main()
