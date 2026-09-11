"""Summarize interleaved device-pipeline measurements and draw their spread."""

import argparse
import json
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

root = Path(__file__).parent
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--prefix", default="matched")
prefix = parser.parse_args().prefix
data = {}
for variant in ("before", "after"):
    runs = [json.loads((root / f"{prefix}-{variant}-{run}.json").read_text()) for run in range(1, 4)]
    assert all(run["native_modules"] == runs[0]["native_modules"] for run in runs)
    rows = []
    for measurements in zip(*(run["rows"] for run in runs), strict=True):
        first = measurements[0]
        assert all(row["source_sha256"] == first["source_sha256"] for row in measurements)
        row = {key: first[key] for key in ("family", "width", "source_sha256")}
        if "error" in first:
            assert all("error" in run for run in measurements)
            rows.append(row | {"error": first["error"]})
            continue
        assert all(run["bitcode_bytes"] == first["bitcode_bytes"] for run in measurements)
        row["bitcode_bytes"] = first["bitcode_bytes"]
        for metric in ("compile", "execution"):
            samples = [value * 1000 for run in measurements for value in run.get(f"{metric}_seconds", [])]
            if samples:
                row[metric] = {"median_ms": median(samples), "min_ms": min(samples), "max_ms": max(samples)}
        rows.append(row)
    data[variant] = rows
assert [r["source_sha256"] for r in data["before"]] == [r["source_sha256"] for r in data["after"]]
(root / f"{prefix}-summary.json").write_text(json.dumps(data, indent=2) + "\n")

fig, axes = plt.subplots(1, 3, figsize=(14, 6), layout="constrained")
colors = {"before": "#65788a", "after": "#0073aa"}
for ax, metric, title in zip(
    axes,
    ("compile", "bitcode_bytes", "execution"),
    ("Device compilation (ms)", "Bitcode (KiB)", "1,024-shot execution (ms)"),
    strict=True,
):
    pairs = [pair for pair in zip(data["before"], data["after"], strict=True) if any(metric in row for row in pair)]
    for index, (variant, offset) in enumerate((("before", -0.18), ("after", 0.18))):
        rows = [pair[index] for pair in pairs if metric in pair[index]]
        positions = np.array([i for i, pair in enumerate(pairs) if metric in pair[index]]) + offset
        values = [row[metric] / 1024 if metric == "bitcode_bytes" else row[metric]["median_ms"] for row in rows]
        errors = (
            None
            if metric == "bitcode_bytes"
            else [
                [row[metric]["median_ms"] - row[metric]["min_ms"] for row in rows],
                [row[metric]["max_ms"] - row[metric]["median_ms"] for row in rows],
            ]
        )
        ax.barh(positions, values, height=0.32, xerr=errors, color=colors[variant], label=variant, capsize=2)
        for i, pair in enumerate(pairs):
            if "error" in pair[index]:
                ax.text(0, i + offset, "unsupported", va="center", fontsize=7, color=colors[variant])
    ax.set_yticks(np.arange(len(pairs)), [f"{pair[0]['family']} {pair[0]['width']}" for pair in pairs])
    ax.set_ylim(len(pairs) - 0.5, -0.5)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
axes[0].legend(handles=[Patch(color=color, label=variant) for variant, color in colors.items()])
fig.suptitle(
    "Structured quantum loops: complete target pipeline\nMedians and min–max, 9 samples; shared DGX Spark host",
    fontsize=14,
)
fig.savefig(root / f"{prefix}-performance.png", dpi=160)
