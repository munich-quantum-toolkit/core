# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the evaluation of JSON results."""

from __future__ import annotations

from math import inf
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mqt.core.dd_evaluation import __aggregate, __flatten_dict, __post_processing

if TYPE_CHECKING:
    from pytest_console_scripts import ScriptRunner

path_base = Path(__file__).resolve().parent / "results_baseline.json"
path_feature = Path(__file__).resolve().parent / "results_feature.json"


def test_flatten_dict() -> None:
    """Test flatten_dict."""
    d1 = {"a": {"b": {"main": 1}}}
    assert __flatten_dict(d1) == {"a.b.main": 1}
    d2 = {"a": {"b": {"main": 1, "feature": 2}}, "d": {"main": 2}}
    assert __flatten_dict(d2) == {"a.b.main": 1, "a.b.feature": 2, "d.main": 2}


def test_post_processing() -> None:
    """Test postprocessing."""
    with pytest.raises(ValueError, match=r"Benchmark a.b is missing algorithm, task, number of qubits or metric!"):
        __post_processing("a.b")
    assert __post_processing("BV.Functionality.1024.runtime") == {
        "algo": "BV",
        "task": "Functionality",
        "n": 1024,
        "component": "",
        "metric": "runtime",
    }
    assert __post_processing("GHZ.Simulation.128.dd.active_memory_mib") == {
        "algo": "GHZ",
        "task": "Simulation",
        "n": 128,
        "component": "",
        "metric": "active_memory_mib",
    }
    assert __post_processing("RandomClifford.Simulation.14.dd.real_numbers.cache_manager.memory_used_MiB_peak") == {
        "algo": "RandomClifford",
        "task": "Simulation",
        "n": 14,
        "component": "real_numbers_cache_manager",
        "metric": "memory_used_MiB_peak",
    }
    assert __post_processing("QPE.Functionality.15.dd.matrix.unique_table.total.lookups") == {
        "algo": "QPE",
        "task": "Functionality",
        "n": 15,
        "component": "matrix_unique_table",
        "metric": "total_lookups",
    }


def test_aggregate() -> None:
    """Test the data aggregation method.

    Raises:
        AssertionError: If the test fails.
    """
    try:
        df_all = __aggregate(path_base, path_feature)
        lookups = df_all[df_all["metric"] == "lookups"]
        assert len(lookups.index) == 1
        assert lookups["before"].to_numpy()[0] == 8172
        assert lookups["after"].to_numpy()[0] == 0
        assert lookups["ratio"].to_numpy()[0] == 0.0
        hit_ratio = df_all[df_all["metric"] == "hit_ratio*"]
        assert len(hit_ratio.index) == 1
        assert hit_ratio["before"].to_numpy()[0] == 0.5
        assert hit_ratio["after"].to_numpy()[0] == 0.8
        assert hit_ratio["ratio"].to_numpy()[0] == 0.625
        memory_mib = df_all[df_all["metric"] == "memory_MiB"]
        assert len(memory_mib.index) == 1
        assert memory_mib["after"].to_numpy()[0] != memory_mib["after"].to_numpy()[0]
        assert memory_mib["ratio"].to_numpy()[0] != memory_mib["ratio"].to_numpy()[0]
        peak_num_entries = df_all[df_all["metric"] == "peak_num_entries"]
        assert len(peak_num_entries.index) == 1
        assert peak_num_entries["before"].to_numpy()[0] != peak_num_entries["before"].to_numpy()[0]
        assert peak_num_entries["ratio"].to_numpy()[0] != peak_num_entries["ratio"].to_numpy()[0]
        num_entries = df_all[df_all["metric"] == "num_entries"]
        assert len(num_entries.index) == 1
        assert num_entries["before"].to_numpy()[0] == 0
        assert num_entries["after"].to_numpy()[0] == 0
        assert num_entries["ratio"].to_numpy()[0] == 1.0
        num_buckets = df_all[(df_all["metric"] == "num_buckets") & (df_all["before"] == df_all["before"])]
        assert len(num_buckets.index) == 1
        assert num_buckets["before"].to_numpy()[0] == 0
        assert num_buckets["after"].to_numpy()[0] == 16384
        assert num_buckets["ratio"].to_numpy()[0] == inf

    except Exception as e:
        msg = "__aggregate() should not raise exception!"
        raise AssertionError(msg) from e


def test_compare_with_negative_factor(script_runner: ScriptRunner) -> None:
    """Testing factor -0.1."""
    ret = script_runner.run([
        "mqt-core-dd-compare",
        path_base,
        path_feature,
        "--factor=-0.1",
    ])
    assert not ret.success
    assert "Factor must be positive!" in ret.stderr


@pytest.mark.script_launch_mode("subprocess")
def test_compare_with_invalid_sort_option(script_runner: ScriptRunner) -> None:
    """Testing an invalid sort option."""
    ret = script_runner.run([
        "mqt-core-dd-compare",
        path_base,
        path_feature,
        "--sort=after",
    ])
    assert not ret.success
    assert "Invalid sort option!" in ret.stderr


@pytest.mark.script_launch_mode("subprocess")
def test_cli_with_num_qubits_specified_without_algorithm(script_runner: ScriptRunner) -> None:
    """Testing the error case when num_qubits is specified without algorithm."""
    ret = script_runner.run([
        "mqt-core-dd-compare",
        path_base,
        path_feature,
        "--factor=0.2",
        "--num_qubits=1024",
    ])
    assert not ret.success
    assert "num_qubits can only be specified if algorithm is also specified!" in ret.stderr


@pytest.mark.script_launch_mode("subprocess")
def test_cli_with_default_parameters(script_runner: ScriptRunner) -> None:
    """Testing the CLI functionality with default parameters."""
    ret = script_runner.run([
        "mqt-core-dd-compare",
        path_base,
        path_feature,
    ])
    assert "Runtime:" in ret.stdout
    assert "Benchmarks that have improved:" in ret.stdout
    assert "Benchmarks that have stayed the same:" in ret.stdout
    assert "Benchmarks that have worsened:" in ret.stdout
    assert "DD Package details:" not in ret.stdout
    assert ret.success


@pytest.mark.script_launch_mode("subprocess")
def test_cli_with_factor_point_three(script_runner: ScriptRunner) -> None:
    """Testing the CLI functionality with default parameters, except that factor is set to 0.3.

    DD details should be shown.
    """
    ret = script_runner.run([
        "mqt-core-dd-compare",
        path_base,
        path_feature,
        "--factor=0.3",
        "--dd",
    ])
    assert "Runtime:" in ret.stdout
    assert "Benchmarks that have improved:" in ret.stdout
    assert "Benchmarks that have stayed the same:" in ret.stdout
    assert "Benchmarks that have worsened:" in ret.stdout
    assert "DD Package details:" in ret.stdout
    assert ret.success


@pytest.mark.script_launch_mode("subprocess")
def test_cli_with_only_changed(script_runner: ScriptRunner) -> None:
    """Testing the CLI functionality with factor set to 0.2 and only_changed set."""
    ret = script_runner.run([
        "mqt-core-dd-compare",
        path_base,
        path_feature,
        "--factor=0.2",
        "--only_changed",
    ])
    assert "Benchmarks that have improved:" in ret.stdout
    assert "Benchmarks that have stayed the same:" not in ret.stdout
    assert "Benchmarks that have worsened:" in ret.stdout
    assert ret.success


@pytest.mark.script_launch_mode("subprocess")
def test_cli_with_only_changed_and_no_split(script_runner: ScriptRunner) -> None:
    """Testing the CLI functionality with factor=0.1 per default, but both only_changed and no_split are set to true.

    DD details should be shown.
    """
    ret = script_runner.run([
        "mqt-core-dd-compare",
        path_base,
        path_feature,
        "--dd",
        "--only_changed",
        "--no_split",
    ])
    assert "Runtime:" in ret.stdout
    assert "Benchmarks that have improved:" not in ret.stdout
    assert "Benchmarks that have stayed the same:" not in ret.stdout
    assert "Benchmarks that have worsened:" not in ret.stdout
    assert "DD Package details:" in ret.stdout
    assert ret.success


@pytest.mark.script_launch_mode("subprocess")
def test_cli_with_no_split(script_runner: ScriptRunner) -> None:
    """Testing the CLI functionality with default parameters, except for no_split set to true.

    DD details should be shown and the output tables should only show benchmarks from the Functionality task.
    """
    ret = script_runner.run([
        "mqt-core-dd-compare",
        path_base,
        path_feature,
        "--no_split",
        "--dd",
        "--task=functionality",
    ])
    assert "Benchmarks that have improved:" not in ret.stdout
    assert "Benchmarks that have stayed the same:" not in ret.stdout
    assert "Benchmarks that have worsened:" not in ret.stdout
    assert "Simulation" not in ret.stdout
    assert "Functionality" in ret.stdout
    assert ret.success


@pytest.mark.script_launch_mode("subprocess")
def test_cli_with_sort_by_algorithm(script_runner: ScriptRunner) -> None:
    """Testing the CLI functionality with sort set with "algorithm" and no_split set to true.

    DD details should be shown and the output tables should only show benchmarks from the BV algorithm with 1024 qubits.
    """
    ret = script_runner.run([
        "mqt-core-dd-compare",
        path_base,
        path_feature,
        "--sort=algorithm",
        "--no_split",
        "--dd",
        "--algorithm=bv",
        "--num_qubits=1024",
    ])
    assert "Benchmarks that have improved" not in ret.stdout
    assert "Benchmarks that have stayed the same" not in ret.stdout
    assert "Benchmarks that have worsened" not in ret.stdout
    assert "GHZ" not in ret.stdout
    assert ret.success
