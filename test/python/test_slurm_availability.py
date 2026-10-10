# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Check that unavailable probes leave scheduler admission closed."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


@pytest.fixture
def monitor() -> ModuleType:
    """Load the standalone administrative example.

    Returns:
        The monitor module.
    """
    path = Path(__file__).parents[2] / "examples" / "slurm" / "update_availability.py"
    spec = importlib.util.spec_from_file_location("slurm_availability", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("failure", [1, subprocess.TimeoutExpired("checker", 1), FileNotFoundError("checker")])
def test_probe_failure_keeps_block_until_recovery(
    monitor: ModuleType, monkeypatch: pytest.MonkeyPatch, failure: int | Exception
) -> None:
    """Never probe with admission open, and preserve other devices' reservations."""
    reservations = {"another-device": "other:1"}
    name = "qdmi-unavailable-example.device"

    def scontrol(*arguments: str) -> str:
        if arguments[0] == "--json":
            return json.dumps({"reservations": [{"name": key} for key in reservations]})
        fields = dict(argument.split("=", maxsplit=1) for argument in arguments[1:])
        if arguments[0] == "delete":
            del reservations[fields["ReservationName"]]
        else:
            reservations[fields["ReservationName"]] = fields["Licenses"]
        return ""

    def probe(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert reservations[name] == "example.device:2"
        if isinstance(failure, Exception):
            raise failure
        return subprocess.CompletedProcess("checker", failure)

    monkeypatch.setattr(monitor, "scontrol", scontrol)
    monkeypatch.setattr(monitor.subprocess, "run", probe)
    arguments = ("--license", "example.device:2")
    assert monitor.main(arguments) == 1
    assert reservations[name] == "example.device:2"
    failure = 0
    assert monitor.main(arguments) == 0
    assert reservations == {"another-device": "other:1"}


def test_controller_failure_never_runs_probe(monitor: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed admission update cannot lead to a healthy, open result."""

    def unavailable(*_arguments: str) -> str:
        msg = "scontrol"
        raise subprocess.TimeoutExpired(msg, 10)

    monkeypatch.setattr(monitor, "scontrol", unavailable)
    monkeypatch.setattr(monitor.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("probe ran before blocking"))
    assert monitor.main(("--license", "example.device:2")) == 2
