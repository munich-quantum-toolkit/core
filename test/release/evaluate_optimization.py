#!/usr/bin/env python3
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compare installed variants: evaluate_optimization.py ARTIFACT_ROOT VARIANT... ."""

# This experiment executes trusted local Python environments and emits result paths.
# ruff: file-ignore[subprocess-without-shell-equals-true, print, missing-type-function-argument]
import argparse
import csv
import hashlib
import json
import math
import operator
import os
import random
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("artifact_root", type=Path)
parser.add_argument("variants", nargs="*")
parser.add_argument("--analyze", type=Path, help="Recompute summaries from an existing measurement file")
parser.add_argument("--cpu", type=int, help="Linux CPU affinity; defaults to the first available CPU")
parser.add_argument("--reference", help="Variant used for latency ratios and regression checks")
args = parser.parse_args()
root = args.artifact_root.resolve()
project = Path(__file__).resolve().parents[2]
samples: list[dict[str, Any]] = []
if args.analyze:
    output = args.analyze.resolve()
    data = json.loads(output.read_text())
    names = data["variants"]
    samples = data["samples"]
else:
    names = args.variants
    assert names
    assert len(names) == len(set(names))
    cpu = None
    if hasattr(os, "sched_setaffinity"):
        available = os.sched_getaffinity(0)
        cpu = min(available) if args.cpu is None else args.cpu
        if cpu not in available:
            parser.error(f"CPU {cpu} is not available to this process")
        os.sched_setaffinity(0, {cpu})
    elif args.cpu is not None:
        parser.error("this platform does not support CPU affinity")
    env = os.environ | {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MQT_CORE_QDMI_CONFIG_JSON": '{"schema-version":1,"qdmi":{"devices":[]}}',
    }
    env.pop("PYTHONPATH", None)
    output = root / "evaluation" / (time.strftime("comparison-%Y%m%d-%H%M%S") + ".json")
    output.parent.mkdir(exist_ok=True)
    print(output, flush=True)
    rotation_step = max(1, math.ceil(len(names) / 12))
    while math.gcd(rotation_step, len(names)) != 1:
        rotation_step += 1
    metadata = {
        "cpu": cpu,
        "affinity_supported": hasattr(os, "sched_setaffinity"),
        "variants": names,
        "rotation_step": rotation_step,
        "benchmark_sha256": hashlib.sha256(
            (project / "test/release/benchmark_optimization.py").read_bytes()
        ).hexdigest(),
        "thread_environment": {key: env[key] for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"]},
    }
    artifacts = {}
    for name in names:
        size_file = root / "measurements" / f"{name}-size.json"
        input_file = root / "measurements" / f"{name}-inputs.json"
        size = json.loads(size_file.read_text())
        inputs = json.loads(input_file.read_text())
        assert inputs["benchmark_sha256"] == size["benchmark_sha256"] == metadata["benchmark_sha256"]
        assert size_file.stat().st_mtime >= input_file.stat().st_mtime
        wheel = Path(size["wheel"])
        with wheel.open("rb") as stream:
            wheel_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        artifacts[name] = size | inputs | {"wheel_sha256": wheel_hash}
    metadata["artifacts"] = artifacts
    for round_ in range(12):
        for offset in range(len(names)):
            name = names[(round_ * rotation_step + offset) % len(names)]
            venv = root / "venvs" / name
            python = venv / "bin/python"
            started = time.perf_counter()
            subprocess.run(
                [str(python), "-c", "import mqt.core.dd,mqt.core.mlir"], env=env, check=True, stdout=subprocess.DEVNULL
            )
            startup = time.perf_counter() - started
            process = subprocess.run(
                [str(python), str(project / "test/release/benchmark_optimization.py"), "--expected-root", str(venv)],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            data = json.loads(process.stdout.strip().splitlines()[-1])
            samples.append({"variant": name, "round": round_, "startup_seconds": startup, **data})
            output.write_text(json.dumps(metadata | {"samples": samples}, indent=2))
            print(name, round_, flush=True)
keys = ["startup", *list(samples[0]["results"])]
lookup = {(s["variant"], s["round"]): s for s in samples}
reference_name = args.reference or names[0]
if reference_name not in names:
    parser.error("the reference must be one of the measured variants")


def value(sample, key) -> float:
    """Return the latency recorded for one workload.

    Returns:
        Latency in seconds.
    """
    return sample["startup_seconds"] if key == "startup" else sample["results"][key]["seconds"]


rows: list[dict[str, Any]] = []
rng = random.Random(237)  # ruff: ignore[suspicious-non-cryptographic-random-usage]
families = sorted({key.split("/")[0] for key in keys})


def family_ratio(name: str, family: str, indices, reference: str = reference_name) -> float:
    """Return the geometric mean of latency ratios within a family.

    Returns:
        Relative latency for the selected resampled rounds.
    """
    workloads = [key for key in keys if key.split("/")[0] == family]
    return statistics.geometric_mean(
        statistics.median(value(lookup[name, r], key) for r in indices)
        / statistics.median(value(lookup[reference, r], key) for r in indices)
        for key in workloads
    )


for name in names:
    for key in keys:
        values = [value(lookup[name, r], key) for r in range(12)]
        base = [value(lookup[reference_name, r], key) for r in range(12)]
        ratios = []
        medians = []
        for _ in range(2000):
            indices = [rng.randrange(12) for _ in range(12)]
            median = statistics.median(values[i] for i in indices)
            medians.append(median)
            ratios.append(median / statistics.median(base[i] for i in indices))
        ratios.sort()
        medians.sort()
        q = statistics.quantiles(values, n=4)
        rows.append({
            "variant": name,
            "workload": key,
            "median_seconds": statistics.median(values),
            "median_ci_low_seconds": medians[49],
            "median_ci_high_seconds": medians[1949],
            "workloads_per_second": 1 / statistics.median(values),
            "throughput_ci_low_per_second": 1 / medians[1949],
            "throughput_ci_high_per_second": 1 / medians[49],
            "iqr_seconds": q[2] - q[0],
            "ratio": statistics.median(values) / statistics.median(base),
            "ratio_ci_low": ratios[49],
            "ratio_ci_high": ratios[1949],
        })
with output.with_suffix(".csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(output, flush=True)

ranking: list[dict[str, Any]] = []
for name in names:
    draws = []
    for _ in range(2000):
        indices = [rng.randrange(12) for _ in range(12)]
        draws.append(statistics.geometric_mean(family_ratio(name, family, indices) for family in families))
    draws.sort()
    ranking.append({
        "variant": name,
        "reference": reference_name,
        "equal_family_ratio": statistics.geometric_mean(family_ratio(name, family, range(12)) for family in families),
        "ci_low": draws[49],
        "ci_high": draws[1949],
        "regressions_above_3_percent": [r["workload"] for r in rows if r["variant"] == name and r["ratio"] > 1.03],
        "confirmed_regressions": [r["workload"] for r in rows if r["variant"] == name and r["ratio_ci_low"] > 1.03],
    })
best = min(ranking, key=operator.itemgetter("equal_family_ratio"))["variant"]
for entry in ranking:
    name = entry["variant"]
    draws = []
    for _ in range(2000):
        indices = [rng.randrange(12) for _ in range(12)]
        draws.append(statistics.geometric_mean(family_ratio(name, family, indices, best) for family in families))
    draws.sort()
    entry.update({
        "best_reference": best,
        "ratio_to_best": statistics.geometric_mean(family_ratio(name, family, range(12), best) for family in families),
        "ratio_to_best_ci_low": draws[49],
        "ratio_to_best_ci_high": draws[1949],
        "distinguishable_from_best": draws[49] > 1,
    })
output.with_suffix(".ranking.json").write_text(
    json.dumps(sorted(ranking, key=operator.itemgetter("equal_family_ratio")), indent=2)
)
