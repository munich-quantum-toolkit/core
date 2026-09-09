# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# /// script
# dependencies = []
# ///
"""Collect alternating runs: python collect.py BEFORE_BINARY AFTER_BINARY."""

import csv
import io
import subprocess
import sys
from pathlib import Path

before, after = sys.argv[1:]
rows = []
for pair in range(9):
    variants = (("before", before), ("after", after))
    if pair % 2:
        variants = variants[::-1]
    for variant, binary in variants:
        # Execute the benchmark binaries explicitly selected by the caller.
        result = subprocess.run([binary], check=True, capture_output=True, text=True)  # ruff: ignore[subprocess-without-shell-equals-true]
        rows.extend({"variant": variant, "pair": pair, **row} for row in csv.DictReader(io.StringIO(result.stdout)))

for workload, size in sorted({(row["workload"], row["size"]) for row in rows}):
    group = [row for row in rows if (row["workload"], row["size"]) == (workload, size)]
    if len({(row["swaps"], row["hash"]) for row in group}) != 1:
        msg = f"Mapped output changed for {workload}, size {size}"
        raise RuntimeError(msg)


with Path(__file__).with_name("results.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
