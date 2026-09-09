# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# /// script
# dependencies = ["matplotlib==3.10.8"]
# ///
"""Render the recorded routing benchmarks: uv run plot.py."""

import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
with (ROOT / "results.csv").open() as stream:
    rows = list(csv.DictReader(stream))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), layout="constrained")
fig.set_facecolor("#f6f8fc")
colors = {"before": "#75859c", "after": "#007f83"}
for ax, workload, title, subtitle in zip(
    axes,
    ("conditional", "routing", "graph"),
    ("Unchanged branch layouts", "Circuits that need routing", "Graph traversal"),
    ("2 qubits · 32 dynamic branches", "8 qubits · 64 CX gates", "Acyclic star · 100 traversals"),
    strict=False,
):
    sizes = sorted({int(row["size"]) for row in rows if row["workload"] == workload})
    for variant, offset in (("before", -0.19), ("after", 0.19)):
        groups = [
            [
                float(r["milliseconds"])
                for r in rows
                if r["workload"] == workload and int(r["size"]) == size and r["variant"] == variant
            ]
            for size in sizes
        ]
        values = [statistics.median(group) for group in groups]
        quartiles = [statistics.quantiles(group, n=4) for group in groups]
        bars = ax.bar(
            [i + offset for i in range(len(sizes))],
            values,
            0.34,
            color=colors[variant],
            label=variant.title(),
            yerr=[
                [v - q[0] for v, q in zip(values, quartiles, strict=False)],
                [q[2] - v for v, q in zip(values, quartiles, strict=False)],
            ],
            capsize=3,
            error_kw={"linewidth": 1, "ecolor": "#263449"},
        )
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=5, fontsize=9)
    ax.set_title(f"{title}\n{subtitle}", loc="left", fontsize=12, pad=16)
    ax.set_xticks(range(len(sizes)), [str(size) for size in sizes])
    ax.set_xlabel("Graph vertices" if workload == "graph" else "Target sites")
    ax.set_ylabel("Time (ms) · lower is better")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.17)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
axes[0].legend(frameon=False, loc="upper left")
fig.suptitle("Routing cleanup · before / after", fontsize=22, fontweight="bold", color="#18334a")
fig.supxlabel(
    "Median and interquartile range · 9 alternating process pairs x 5 samples · fixed CPU, seed 42\n"
    "Mapped IR hashes and SWAP counts match. Graph case is synthetic; it is not a mapper speedup.",
    fontsize=10,
    color="#405268",
)
fig.savefig(ROOT / "before-after.png", dpi=180, facecolor=fig.get_facecolor())
