# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Run paired A/B measurements of the MLIR mapping benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

SCENARIOS = ("frontier", "routing")


def run(command: list[str], *, cwd: Path | None = None) -> str:
    """Run a command.

    Returns:
        The command's standard output without surrounding whitespace.
    """
    return subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def worktree_metadata(executable: Path) -> dict[str, str | bool]:
    """Return source-worktree metadata associated with an executable path."""
    directory = str(executable.parent)
    return {
        "revision": run(["git", "-C", directory, "rev-parse", "HEAD"]),
        "dirty": bool(run(["git", "-C", directory, "status", "--porcelain"])),
    }


def executable_digest(executable: Path) -> str:
    """Return the SHA-256 digest of an executable."""
    digest = hashlib.sha256()
    with executable.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def cmake_metadata(executable: Path) -> dict[str, str]:
    """Read compiler and MLIR metadata from the executable's CMake cache.

    Returns:
        Available build type, compiler, and MLIR configuration values.
    """
    cache = executable.parents[1] / "CMakeCache.txt"
    if not cache.exists():
        return {}

    values: dict[str, str] = {}
    for line in cache.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.split(":", maxsplit=1)[0] in {
            "CMAKE_BUILD_TYPE",
            "CMAKE_CXX_COMPILER",
            "MLIR_DIR",
        }:
            values[key.split(":", maxsplit=1)[0]] = value

    compiler = values.get("CMAKE_CXX_COMPILER")
    if compiler:
        values["CMAKE_CXX_COMPILER_VERSION"] = run([compiler, "--version"]).splitlines()[0]
    return values


def measure(
    executable: Path,
    *,
    scenario: str,
    qubits: int,
    layers: int,
    seed: int,
    lookahead: int,
    lambda_: float,
    iterations: int,
    trials: int,
) -> int:
    """Measure one mapping pass execution.

    Returns:
        The elapsed time in nanoseconds.
    """
    output = run([
        str(executable),
        f"--scenario={scenario}",
        f"--qubits={qubits}",
        f"--layers={layers}",
        f"--seed={seed}",
        f"--lookahead={lookahead}",
        f"--lambda={lambda_}",
        f"--iterations={iterations}",
        f"--trials={trials}",
    ])
    return int(output)


def trimmed_mean(samples: list[float], proportion: float = 0.1) -> float:
    """Return a symmetrically trimmed mean.

    Returns:
        The mean after removing the requested proportion from both tails.
    """
    ordered = sorted(samples)
    count = int(len(ordered) * proportion)
    selected = ordered[count:-count] if count else ordered
    return statistics.fmean(selected)


def summarize(samples: list[int]) -> dict[str, float]:
    """Summarize samples.

    Returns:
        Robust summary statistics in milliseconds.
    """
    milliseconds = [sample / 1_000_000 for sample in samples]
    median = statistics.median(milliseconds)
    return {
        "median_ms": median,
        "mad_ms": statistics.median(abs(sample - median) for sample in milliseconds),
        "trimmed_mean_ms": trimmed_mean(milliseconds),
    }


def paired_speedup(baseline: list[int], candidate: list[int], *, seed: int) -> dict[str, float]:
    """Summarize paired speedups.

    Returns:
        The median ratio and its bootstrap 95% confidence interval.
    """
    ratios = [before / after for before, after in zip(baseline, candidate, strict=True)]
    point = statistics.median(ratios)
    rng = random.Random(seed)  # ruff:ignore[suspicious-non-cryptographic-random-usage]
    bootstraps = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(10_000))
    return {
        "median_ratio": point,
        "percent": (point - 1) * 100,
        "bootstrap_95_percent_low": (bootstraps[249] - 1) * 100,
        "bootstrap_95_percent_high": (bootstraps[9749] - 1) * 100,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--qubits", type=int, default=36)
    parser.add_argument("--layers", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1930)
    parser.add_argument("--lookahead", type=int, default=20)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--trials", type=int, default=18)
    parser.add_argument("--scenario", choices=SCENARIOS, action="append")
    return parser.parse_args()


def main() -> None:
    """Run the benchmark comparison and write raw and summarized results."""
    args = parse_args()
    scenarios = args.scenario or list(SCENARIOS)
    executables = {
        "baseline": args.baseline.resolve(),
        "candidate": args.candidate.resolve(),
    }
    results: dict[str, Any] = {
        "configuration": {
            "samples": args.samples,
            "warmups": args.warmups,
            "qubits": args.qubits,
            "layers": args.layers,
            "seed": args.seed,
            "lookahead": args.lookahead,
            "lambda": args.lambda_,
            "iterations": args.iterations,
            "trials": args.trials,
        },
        "executables": {
            name: {
                "path": str(executable),
                "worktree": worktree_metadata(executable),
                "sha256": executable_digest(executable),
                "cmake": cmake_metadata(executable),
            }
            for name, executable in executables.items()
        },
        "scenarios": {},
    }

    order_rng = random.Random(args.seed)  # ruff:ignore[suspicious-non-cryptographic-random-usage]
    for scenario in scenarios:
        for _ in range(args.warmups):
            for executable in executables.values():
                measure(
                    executable,
                    scenario=scenario,
                    qubits=args.qubits,
                    layers=args.layers,
                    seed=args.seed,
                    lookahead=args.lookahead,
                    lambda_=args.lambda_,
                    iterations=args.iterations,
                    trials=args.trials,
                )

        raw = {"baseline": [], "candidate": []}
        orders: list[list[str]] = []
        for _ in range(args.samples):
            order = list(executables)
            order_rng.shuffle(order)
            orders.append(order)
            for name in order:
                raw[name].append(
                    measure(
                        executables[name],
                        scenario=scenario,
                        qubits=args.qubits,
                        layers=args.layers,
                        seed=args.seed,
                        lookahead=args.lookahead,
                        lambda_=args.lambda_,
                        iterations=args.iterations,
                        trials=args.trials,
                    )
                )

        results["scenarios"][scenario] = {
            "execution_order": orders,
            "raw_nanoseconds": raw,
            "baseline": summarize(raw["baseline"]),
            "candidate": summarize(raw["candidate"]),
            "speedup": paired_speedup(raw["baseline"], raw["candidate"], seed=args.seed),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for scenario, result in results["scenarios"].items():
        speedup = result["speedup"]
        sys.stdout.write(
            f"{scenario}: {speedup['percent']:+.2f}% "
            f"[{speedup['bootstrap_95_percent_low']:+.2f}%, "
            f"{speedup['bootstrap_95_percent_high']:+.2f}%]\n"
        )


if __name__ == "__main__":
    main()
