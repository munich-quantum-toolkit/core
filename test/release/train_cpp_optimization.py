#!/usr/bin/env python3
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Run Core's C++ suites per executable to bound the number of PGO profile files."""

from __future__ import annotations

# Commands come from the trusted local CMake build.
# ruff: file-ignore[subprocess-without-shell-equals-true, start-process-with-partial-path]
import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main() -> None:
    """Run the registered suites and keep CTest's handling of non-GoogleTest cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    args = parser.parse_args()
    build = args.build.resolve()
    listing_env = os.environ | {
        "LLVM_PROFILE_FILE": str(build / "profile-discovery/%m-%p.profraw"),
        "MQT_PGO_PROFILE_DIR": str(build / "profile-discovery"),
    }
    listing = json.loads(
        subprocess.check_output(["ctest", "--test-dir", str(build), "--show-only=json-v1"], text=True, env=listing_env)
    )
    groups = {}
    other = []
    for test in listing["tests"]:
        command = test["command"]
        options = dict(arg.split("=", 1) for arg in command if arg.startswith("TEST_") and "=" in arg)
        properties = {prop["name"]: prop["value"] for prop in test["properties"]}
        if "TEST_EXECUTABLE" in options:
            assert not options.get("TEST_EXECUTOR")
            assert not options.get("TEST_EXTRA_ARGS")
            executable = options["TEST_EXECUTABLE"]
        elif any(arg.startswith("--gtest_") for arg in command[1:]):
            assert all(arg.startswith("--gtest_") for arg in command[1:])
            executable = command[0]
        else:
            other.append(test["name"])
            continue
        assert not properties.get("ENVIRONMENT")
        assert not properties.get("ENVIRONMENT_MODIFICATION")
        groups[executable, properties["WORKING_DIRECTORY"]] = None
    output = build / "profile-test-results"
    output.mkdir(exist_ok=True)
    selected = output / "ctest.txt"
    selected.write_text("\n".join(other) + "\n", encoding="utf-8")
    subprocess.run(
        ["ctest", "--test-dir", str(build), "--tests-from-file", str(selected), "--output-on-failure", "-j", "4"],
        check=True,
    )

    def run_suite(item: tuple[int, tuple[str, str]]) -> dict:
        index, (executable, cwd) = item
        report = output / f"gtest-{index}.json"
        subprocess.run(
            [
                executable,
                "--gtest_filter=*",
                "--gtest_repeat=1",
                "--gtest_shuffle=0",
                "--gtest_also_run_disabled_tests=0",
                f"--gtest_output=json:{report}",
            ],
            cwd=cwd,
            check=True,
        )
        return json.loads(report.read_text(encoding="utf-8"))

    with ThreadPoolExecutor(max_workers=4) as pool:
        reports = list(pool.map(run_suite, enumerate(groups)))
    summary = {key: sum(report[key] for report in reports) for key in ["tests", "failures", "disabled"]}
    summary |= {
        "executables": len(groups),
        "other_ctest_entries": len(other),
        "skipped": sum(
            test.get("result") == "SKIPPED"
            for report in reports
            for suite in report["testsuites"]
            for test in suite["testsuite"]
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
