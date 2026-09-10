import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
python_keys = [
    f"{kind}_10000"
    for kind in ["explicit_source", "explicit_typed", "device_source", "device_typed"]
]
native_keys = ["ddsim_index_100k", "sc_index_100k", "ddsim_snapshot", "sc_snapshot"]
python_labels = ["Explicit\nsource", "Explicit\nQCO", "DDSIM\nsource", "DDSIM\nQCO"]
native_labels = [
    "100k DDSIM\nindex reads",
    "100k static\nindex reads",
    "Full DDSIM\nsnapshot",
    "100-site static\nsnapshot",
]

for offset, phase, color in [(-0.18, "before", "#667085"), (0.18, "after", "#147d92")]:
    python = {}
    for line in (root / f"python-{phase}.txt").read_text().splitlines():
        prefix, samples = line.split(" durations_s ")
        durations, gaps = samples.split(" gaps_s ")
        python[prefix.split()[0]] = (
            ast.literal_eval(gaps),
            ast.literal_eval(durations),
        )
    native = {}
    for line in (root / f"native-{phase}.txt").read_text().splitlines():
        if not line.startswith("checksum="):
            name, *fields = line.split()
            native[name] = dict(field.split("=") for field in fields)
    for ax, index, title in [
        (axes[0, 0], 0, "Maximum Python heartbeat gap"),
        (axes[1, 0], 1, "Total compilation time"),
    ]:
        samples = np.array([python[key][index] for key in python_keys]) * 1000
        median = np.median(samples, axis=1)
        ax.bar(
            np.arange(4) + offset,
            median,
            width=0.34,
            label=phase.title(),
            color=color,
            yerr=[median - samples.min(axis=1), samples.max(axis=1) - median],
            capsize=3,
        )
        ax.set(
            title=title,
            ylabel="Milliseconds (log scale)" if index == 0 else "Milliseconds",
            yscale="log" if index == 0 else "linear",
            xticks=np.arange(4),
            xticklabels=python_labels,
        )
    allocations = [int(native[key]["allocations"]) / 1000 for key in native_keys]
    axes[0, 1].bar(
        np.arange(4) + offset, allocations, width=0.34, label=phase.title(), color=color
    )
    samples = [
        np.array(native[key]["samples_ms"].strip(",").split(","), dtype=float)
        for key in native_keys
    ]
    median = np.array([np.median(values) for values in samples])
    axes[1, 1].bar(
        np.arange(4) + offset,
        median,
        width=0.34,
        label=phase.title(),
        color=color,
        yerr=[
            median - np.array([min(values) for values in samples]),
            np.array([max(values) for values in samples]) - median,
        ],
        capsize=3,
    )

axes[0, 1].set(
    title="Native allocation count",
    ylabel="Thousands of allocations",
    xticks=np.arange(4),
    xticklabels=native_labels,
)
axes[1, 1].set(
    title="Native query and snapshot time",
    ylabel="Milliseconds (log scale)",
    yscale="log",
    xticks=np.arange(4),
    xticklabels=native_labels,
)
for ax in axes.flat:
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
fig.suptitle(
    "QDMI pre-release fixes · DGX Spark / CPython 3.14 / LLVM 23\nMedians; timing whiskers show min–max (Python n=3, native n=5)",
    fontsize=14,
)
fig.savefig(root / "before-after.png", dpi=160)
