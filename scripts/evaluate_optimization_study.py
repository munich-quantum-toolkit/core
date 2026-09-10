#!/usr/bin/env python3
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Install trial artifacts and run one isolated, paired evaluation cohort."""

from __future__ import annotations

# The study installs hash-checked artifacts from the selected workflow runs.
# ruff: file-ignore[subprocess-without-shell-equals-true, start-process-with-partial-path]
import argparse
import hashlib
import json
import operator
import os
import platform
import subprocess
import sys
from pathlib import Path

from linux_optimization import run


def digest(path: Path) -> str:
    """Return the SHA-256 digest of one artifact."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    """Evaluate only comparable, verified wheels and preserve their individual identities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cohort", choices=[1, 2], type=int, required=True)
    parser.add_argument("--variants", nargs="*", help="Exact variant names; omitted means every downloaded variant")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = args.root.resolve()
    if root.exists():
        parser.error("use a fresh evaluation root for every cohort")
    benchmark = digest(project / "test/release/benchmark_optimization.py")
    variants = {}
    for manifest_path in sorted(args.downloads.rglob("artifacts.json")):
        manifest = json.loads(manifest_path.read_text())
        requirements = manifest_path.with_name("requirements.txt")
        if manifest["benchmark_sha256"] != benchmark or digest(requirements) != manifest["requirements_sha256"]:
            parser.error("the benchmark or locked dependencies changed since the wheels were built")
        for artifact in manifest["artifacts"]:
            name = f"sdk-{manifest['sdk_lto']}_core-{manifest['core_lto']}_pgo-{manifest['pgo']}_{artifact['name']}"
            if args.variants and name not in args.variants:
                continue
            wheels = list(manifest_path.parent.rglob(Path(artifact["wheel"]).name))
            if name in variants or len(wheels) != 1 or digest(wheels[0]) != artifact["wheel_sha256"]:
                parser.error("every selected variant needs one unambiguous wheel with the recorded hash")
            variants[name] = (manifest, artifact, wheels[0].resolve(), requirements.resolve())
    if not variants or (args.variants and set(args.variants) != set(variants)):
        parser.error("one or more requested variants are missing")
    reference = next(iter(variants.values()))[0]
    for manifest, _, _, _ in variants.values():
        for key in ["core_source_trees", "llvm_source_id", "compiler_version", "requirements_sha256", "machine"]:
            if not manifest.get(key) or manifest[key] != reference.get(key):
                parser.error(f"variant inputs differ: {key}")
    native = [name for name, (manifest, _, _, _) in variants.items() if manifest["sdk_lto"] == "OFF"]
    if not native:
        parser.error("include at least one native-SDK reference")
    measurements = root / "measurements"
    measurements.mkdir(parents=True)
    env = {"PYTHONPATH": "", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    for name, (manifest, artifact, wheel, requirements) in variants.items():
        venv = root / "venvs" / name
        commands = [
            ("venv", ["uv", "venv", "--python", sys.executable, str(venv)]),
            ("dependencies", ["uv", "pip", "sync", "--python", str(venv / "bin/python"), str(requirements)]),
            ("install", ["uv", "pip", "install", "--no-deps", "--python", str(venv / "bin/python"), str(wheel)]),
        ]
        for label, command in commands:
            if run(measurements / f"{name}-{label}.json", command, project, env):
                parser.exit(1, f"{name}: {label} failed\n")
        (measurements / f"{name}-inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (measurements / f"{name}-size.json").write_text(
            json.dumps(artifact | {"wheel": str(wheel), "benchmark_sha256": benchmark}, indent=2) + "\n"
        )
    host = {
        "cohort": args.cohort,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": sys.version,
        "runner": {key: os.environ.get(key) for key in ["RUNNER_OS", "RUNNER_ARCH", "ImageVersion"]},
    }
    if platform.system() == "Linux":
        host["cpuinfo"] = Path("/proc/cpuinfo").read_text(encoding="utf-8").split("\n\n")[0]
    else:
        host["hardware"] = subprocess.check_output(["sysctl", "hw.model", "hw.memsize", "hw.ncpu"], text=True)
    (root / "host.json").write_text(json.dumps(host, indent=2) + "\n")
    command = [sys.executable, str(project / "test/release/evaluate_optimization.py"), str(root), *variants]
    subprocess.run(command, check=True, env=os.environ | env)
    ranking_file = next((root / "evaluation").glob("*.ranking.json"))
    ranking = json.loads(ranking_file.read_text())
    finalists = {}
    for group, names in [("native", native), ("matched", [name for name in variants if name not in native])]:
        if not names:
            continue
        winner = min((row for row in ranking if row["variant"] in names), key=operator.itemgetter("equal_family_ratio"))
        manifest = variants[winner["variant"]][0]
        finalists[group] = {
            "variant": winner["variant"],
            "sdk": manifest["sdk_lto"],
            "core": manifest["core_lto"],
            "pgo": manifest["pgo"],
            "ranking": winner,
        }
    (root / "finalists.json").write_text(json.dumps(finalists, indent=2) + "\n")


if __name__ == "__main__":
    main()
