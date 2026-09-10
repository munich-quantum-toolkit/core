# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""The measurement runner must preserve failures and replay explicit inputs."""

# These checks execute explicit local Python commands.
# ruff: file-ignore[implicit-namespace-package, subprocess-without-shell-equals-true]
from __future__ import annotations

import base64
import csv
import hashlib
import json
import math
import os
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def test_failed_command_and_replay(tmp_path: Path) -> None:
    """A failed workload stays failed, including after replay."""
    runner = Path(__file__).resolve().parents[2] / "scripts/linux_optimization.py"
    record = tmp_path / "failed.json"
    command = [
        sys.executable,
        str(runner),
        "--output",
        str(record),
        "--env",
        "MQT_EXPERIMENT_CHECK=kept",
        "--",
        sys.executable,
        "-c",
        'import os,sys; print(os.environ["MQT_EXPERIMENT_CHECK"]); sys.exit(7)',
    ]
    assert subprocess.run(command, check=False).returncode == 7
    result = json.loads(record.read_text())
    assert result["returncode"] == 7
    assert result["elapsed_seconds"] > 0
    assert record.with_suffix(".log").read_text().strip() == "kept"
    replay = tmp_path / "replayed.json"
    assert (
        subprocess.run(
            [sys.executable, str(runner), "--output", str(replay), "--replay", str(record)], check=False
        ).returncode
        == 7
    )
    assert replay.with_suffix(".log").read_text().strip() == "kept"


def test_memory_in_nested_process_group(tmp_path: Path) -> None:
    """Ninja-like child process groups still contribute to aggregate RSS."""
    runner = Path(__file__).resolve().parents[2] / "scripts/linux_optimization.py"
    record = tmp_path / "memory.json"
    child = "import time; data=bytearray(32*1024*1024); time.sleep(1.5)"
    command = (
        f'import subprocess,sys; subprocess.run([sys.executable,"-c",{child!r}], start_new_session=True, check=True)'
    )
    subprocess.run(
        [sys.executable, str(runner), "--output", str(record), "--", sys.executable, "-c", command], check=True
    )
    assert json.loads(record.read_text())["sampled_tree_rss_bytes"] >= 32 * 1024 * 1024


def test_windows_records_failures_without_linux_resource_claims(tmp_path: Path) -> None:
    """Compatibility checks retain exit status without inventing Windows RSS data."""
    runner = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/linux_optimization.py"))
    record = tmp_path / "windows.json"
    with patch("platform.system", return_value="Windows"):
        result = runner["run"](record, [sys.executable, "-c", "raise SystemExit(7)"], tmp_path, {})
    assert result == 7
    saved = json.loads(record.read_text())
    assert saved["returncode"] == 7
    assert saved["sampled_tree_rss_bytes"] is None
    assert saved["cgroup_peak_bytes"] is None


@pytest.mark.skipif(sys.version_info < (3, 14), reason="SDK extraction uses Python 3.14 zstd support")
def test_sdk_archive_with_two_gib_window(tmp_path: Path) -> None:
    """The portable SDK's long-window frames must extract through the shared helper."""
    archive = tmp_path / "sdk.tar.zst"
    archive.write_bytes(
        base64.b64decode(
            "KLUv/QSoHAIAdAJwYXlsb2FkADAwMDA2NDQAMDAwMDAwMDQwNzMxMwAgMHVzdGFyADAJAJD+kAPxhtQun8AI20C5F4NdOcpu"
            "oisBcQIAEAACABAAAgAQAAIAEAACABAAAgAQAAIAEAADwAAAFcw1Hg=="
        )
    )
    destination = tmp_path / "sdk"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[2] / "scripts/extract_study_sdk.py"),
            str(archive),
            str(destination),
        ],
        check=True,
    )
    assert (destination / "payload").read_bytes() == bytes(1024 * 1024)


def test_downloaded_wheel_hash_is_checked_before_installation(tmp_path: Path) -> None:
    """A damaged trial artifact must fail before creating a benchmark environment."""
    project = Path(__file__).resolve().parents[2]
    (tmp_path / "requirements.txt").write_text("")
    (tmp_path / "candidate.whl").write_bytes(b"wrong artifact")
    manifest = {
        "benchmark_sha256": hashlib.sha256(
            (project / "test/release/benchmark_optimization.py").read_bytes()
        ).hexdigest(),
        "requirements_sha256": hashlib.sha256(b"").hexdigest(),
        "sdk_lto": "OFF",
        "core_lto": "OFF",
        "pgo": "none",
        "artifacts": [{"name": "plain", "wheel": "/original/candidate.whl", "wheel_sha256": "wrong"}],
    }
    (tmp_path / "artifacts.json").write_text(json.dumps(manifest))
    root = tmp_path / "evaluation"
    result = subprocess.run(
        [
            sys.executable,
            str(project / "scripts/evaluate_optimization_study.py"),
            "--downloads",
            str(tmp_path),
            "--root",
            str(root),
            "--cohort",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "recorded hash" in result.stderr
    assert not root.exists()


def test_equal_family_weight_and_paired_intervals(tmp_path: Path) -> None:
    """Adding scales within a family must not increase that family's weight."""
    samples = []
    for variant in ["base", "slower"]:
        samples.extend(
            {
                "variant": variant,
                "round": round_,
                "startup_seconds": 1,
                "results": {
                    key: {"seconds": 4 if variant == "slower" and key.startswith("parse/") else 1}
                    for key in ["parse/32", "parse/128", "parse/512", "dd/32"]
                },
            }
            for round_ in range(12)
        )
    record = tmp_path / "synthetic.json"
    record.write_text(json.dumps({"variants": ["base", "slower"], "samples": samples}))
    evaluator = Path(__file__).with_name("evaluate_optimization.py")
    subprocess.run([sys.executable, str(evaluator), str(tmp_path), "--analyze", str(record)], check=True)
    ranking = json.loads(record.with_suffix(".ranking.json").read_text())
    slower = next(row for row in ranking if row["variant"] == "slower")
    expected = 4 ** (1 / 3)
    assert math.isclose(slower["equal_family_ratio"], expected)
    assert math.isclose(slower["ratio_to_best_ci_low"], expected)
    assert math.isclose(slower["ratio_to_best_ci_high"], expected)
    assert slower["distinguishable_from_best"]


def test_discovery_profiles_stay_outside_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CTest discovery must not add initialization-only profiles to training."""
    raw = tmp_path / "training"
    monkeypatch.setenv("LLVM_PROFILE_FILE", str(raw / "%m-%p.profraw"))
    monkeypatch.setenv("MQT_PGO_PROFILE_DIR", str(raw))
    monkeypatch.setattr(sys, "argv", ["train_cpp_optimization.py", str(tmp_path)])
    runner = runpy.run_path(str(Path(__file__).with_name("train_cpp_optimization.py")))
    with patch("subprocess.check_output", return_value='{"tests": []}') as discovery, patch("subprocess.run"):
        runner["main"]()
    env = discovery.call_args.kwargs["env"]
    assert not Path(env["LLVM_PROFILE_FILE"]).is_relative_to(raw)
    assert not Path(env["MQT_PGO_PROFILE_DIR"]).is_relative_to(raw)
    assert os.environ["LLVM_PROFILE_FILE"] == str(raw / "%m-%p.profraw")
    assert os.environ["MQT_PGO_PROFILE_DIR"] == str(raw)


def test_absolute_latency_and_throughput_intervals(tmp_path: Path) -> None:
    """Throughput intervals invert the upper and lower latency bounds."""
    samples = [
        {
            "variant": "base",
            "round": round_,
            "startup_seconds": 1 if round_ < 6 else 2,
            "results": {"probe/1": {"seconds": 1 if round_ < 6 else 2}},
        }
        for round_ in range(12)
    ]
    record = tmp_path / "absolute.json"
    record.write_text(json.dumps({"variants": ["base"], "samples": samples}))
    evaluator = Path(__file__).with_name("evaluate_optimization.py")
    subprocess.run([sys.executable, str(evaluator), str(tmp_path), "--analyze", str(record)], check=True)
    with record.with_suffix(".csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    for row in rows:
        assert math.isclose(float(row["median_seconds"]), 1.5)
        assert math.isclose(float(row["median_ci_low_seconds"]), 1)
        assert math.isclose(float(row["median_ci_high_seconds"]), 2)
        assert math.isclose(float(row["workloads_per_second"]), 2 / 3)
        assert math.isclose(float(row["throughput_ci_low_per_second"]), 0.5)
        assert math.isclose(float(row["throughput_ci_high_per_second"]), 1)


def test_matched_sdk_gate_uses_best_native_and_every_cohort(tmp_path: Path) -> None:
    """A fast matched SDK must beat the best native recipe in both cohorts."""
    names = ["native", "native-pgo", "matched", "regressed"]
    paths = []
    for cohort in range(2):
        samples = []
        for name in names:
            seconds = {"native": 1, "native-pgo": 0.8, "matched": 0.7, "regressed": 0.7 if cohort == 0 else 0.75}[name]
            samples.extend(
                {
                    "variant": name,
                    "round": round_,
                    "startup_seconds": seconds,
                    "results": {"compile/32": {"seconds": seconds}},
                }
                for round_ in range(12)
            )
        path = tmp_path / f"cohort-{cohort}.json"
        path.write_text(
            json.dumps({
                "variants": names,
                "samples": samples,
                "benchmark_sha256": "same",
                "artifacts": {name: {"wheel_sha256": name} for name in names},
            })
        )
        paths.append(path)
    output = tmp_path / "decision.json"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("decide_optimization.py")),
            *map(str, paths),
            "--native",
            "native",
            "--native",
            "native-pgo",
            "--matched",
            "matched",
            "--matched",
            "regressed",
            "--output",
            str(output),
        ],
        check=True,
    )
    result = json.loads(output.read_text())
    assert all(cohort["native_reference"] == "native-pgo" for cohort in result["cohorts"])
    assert result["decisions"]["matched"]["runtime_gate_passed"]
    assert not result["decisions"]["regressed"]["runtime_gate_passed"]
