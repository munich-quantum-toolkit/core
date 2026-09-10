#!/usr/bin/env python3
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Apply the matched-SDK runtime gate to two independent paired measurement cohorts."""

from __future__ import annotations

# Analysis executes the adjacent, trusted evaluation script.
# ruff: file-ignore[subprocess-without-shell-equals-true]
import argparse
import json
import operator
import subprocess
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    """Require a 10% confidence-supported gain without confirmed 3% regressions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohorts", type=Path, nargs=2)
    parser.add_argument("--native", action="append", required=True)
    parser.add_argument("--matched", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cohorts[0].resolve() == args.cohorts[1].resolve():
        parser.error("two distinct cohort files are required")
    if set(args.native) & set(args.matched):
        parser.error("native and matched variants must be disjoint")
    selected = set(args.native + args.matched)
    records = [json.loads(path.read_text()) for path in args.cohorts]
    if not records[0].get("benchmark_sha256") or records[0]["benchmark_sha256"] != records[1].get("benchmark_sha256"):
        parser.error("cohorts must record the same held-out benchmark hash")
    for data in records:
        if not selected <= set(data["variants"]):
            parser.error("every requested variant must appear in both cohorts")
        for name in selected:
            samples = [sample for sample in data["samples"] if sample["variant"] == name]
            if len(samples) != 12 or {sample["round"] for sample in samples} != set(range(12)):
                parser.error("each variant needs twelve unique paired rounds")
            if data["artifacts"][name]["wheel_sha256"] != records[0]["artifacts"][name]["wheel_sha256"]:
                parser.error("a variant's wheel hash changed between cohorts")
    cohorts: list[dict[str, Any]] = []
    for path in args.cohorts:
        command = [
            sys.executable,
            str(Path(__file__).with_name("evaluate_optimization.py")),
            str(path.parent),
            "--analyze",
            str(path),
        ]
        subprocess.run(command, check=True)
        ranking_path = path.with_suffix(".ranking.json")
        ranking = json.loads(ranking_path.read_text())
        native = min(
            (row for row in ranking if row["variant"] in args.native), key=operator.itemgetter("equal_family_ratio")
        )
        subprocess.run([*command, "--reference", native["variant"]], check=True)
        ranking = json.loads(ranking_path.read_text())
        cohorts.append({
            "source": str(path.resolve()),
            "native_reference": native["variant"],
            "matched": {row["variant"]: row for row in ranking if row["variant"] in args.matched},
        })
    decisions = {}
    for name in args.matched:
        passed = all(
            cohort["matched"][name]["ci_high"] <= 0.90 and not cohort["matched"][name]["confirmed_regressions"]
            for cohort in cohorts
        )
        decisions[name] = {"runtime_gate_passed": passed}
    result = {
        "minimum_gain": 0.10,
        "maximum_confirmed_workload_regression": 0.03,
        "cohorts": cohorts,
        "decisions": decisions,
        "remaining_gates": [
            "installed package correctness",
            "C++ consumer compatibility",
            "complete cold hosted pipeline",
            "runner memory and disk limits",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
