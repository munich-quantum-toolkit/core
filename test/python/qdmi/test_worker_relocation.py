# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Check that installed DDSIM workers move with their provider."""

from __future__ import annotations

import shutil
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path


def test_worker_relocation_and_startup_failure(tmp_path: Path) -> None:
    """A missing worker fails one job; restoring it permits later jobs."""
    package = Path(str(distribution("mqt-core").locate_file("mqt/core")))
    runtime = package / ("bin" if sys.platform == "win32" else "lib")
    worker_name = "mqt-core-ddsim-worker" + (".exe" if sys.platform == "win32" else "")
    worker = runtime / worker_name
    assert worker.is_file()
    relocated = tmp_path / "relocated"
    shutil.copytree(runtime, relocated, ignore=shutil.ignore_patterns(worker_name))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    script = r"""
import json
import shutil
import sys
from pathlib import Path

import pytest
from mqt.core.qdmi import Job, ProgramFormat
from mqt.core.qdmi import builtin_driver

relocated = Path(sys.argv[1])
manifest = json.loads((relocated / "mqt-core-qdmi-ddsim-device.qdmi.json").read_text())
definition = manifest["qdmi"]["devices"][0]
definition["id"] = "test.relocated"
manifest_path = relocated / "relocated.qdmi.json"
manifest_path.write_text(json.dumps(manifest))
builtin_driver.add_manifest(manifest_path)
device = builtin_driver.open_device("test.relocated")
program = 'OPENQASM 3.0; include "stdgates.inc"; qubit q; x q; bit c = measure q;'
failed = device.submit_job(program, ProgramFormat.QASM3, num_shots=4)
assert failed.wait(30)
assert failed.check() == Job.Status.FAILED
with pytest.raises(RuntimeError):
    failed.get_counts()
with pytest.raises(RuntimeError):
    failed.get_dense_statevector()

worker = Path(sys.argv[2])
shutil.copy2(worker, relocated / worker.name)
for _ in range(2):
    valid = device.submit_job(program, ProgramFormat.QASM3, num_shots=4, custom1=7)
    assert valid.wait(30)
    assert valid.check() == Job.Status.DONE
    assert valid.get_counts() == {"1": 4}
    assert valid.get_dense_statevector() == pytest.approx([0, 1])
"""
    # The command and script are fixed; paths are separate arguments, with no shell.
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", script, str(relocated), str(worker)],
        cwd=elsewhere,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
