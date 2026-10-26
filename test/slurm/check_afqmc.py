# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Check real provider readout or completed AFQMC artifacts inside the playground."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


def readout(device: str) -> None:
    """Exercise all basis states with reversed classical destinations through QDMI."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/slurm/afqmc"))
    from workflow import quantum  # ruff: ignore[import-outside-top-level] - standalone example

    directory = Path(tempfile.mkdtemp(prefix="readout-", dir="/jobs"))
    (directory / "source").mkdir()
    hashes = []
    for value in range(16):
        gates = "\n".join(f"x q[{q}];" for q in range(4) if value & (1 << q))
        measurements = "\n".join(f"measure q[{q}] -> c[{3 - q}];" for q in range(4))
        source = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\ncreg c[4];\n' + gates + "\n" + measurements
        (directory / "source" / f"{value}.qasm").write_text(source)
        hashes.append(hashlib.sha256(source.encode()).hexdigest())
    (directory / "prepared.json").write_text(
        json.dumps({"snapshots": 16, "shots": 32, "seed": 17, "source_sha256": hashes})
    )
    np.savez(directory / "bases.npz", perm=np.tile(range(8), (16, 1)), sign=np.ones((16, 8)))
    quantum(directory, device)
    samples = np.load(directory / "shots.npz")["logical"]
    for value, shots in enumerate(samples):
        assert set(shots) == {f"{value:04b}"[::-1]}, (device, value, set(shots))
    sys.stdout.write(f"All 16 reversed-output basis states passed on {device}\n")


def artifacts(directory: Path) -> None:
    """Check full inputs, scientific invariants, and disjoint real allocations."""
    prepared, quantum, result = [
        json.loads((directory / name).read_text()) for name in ("prepared.json", "quantum.json", "result.json")
    ]
    assert prepared["snapshots"] == quantum["snapshots"]
    assert prepared["shots"] == quantum["shots_per_snapshot"]
    assert prepared["majorana_max_error"] < 1e-10
    for index, expected in enumerate(prepared["source_sha256"]):
        assert hashlib.sha256((directory / "source" / f"{index}.qasm").read_bytes()).hexdigest() == expected
    for index, expected in enumerate(quantum["payload_sha256"]):
        assert hashlib.sha256((directory / "compiled" / f"{index}.txt").read_bytes()).hexdigest() == expected
    samples = np.load(directory / "shots.npz")["logical"]
    assert samples.shape == (prepared["snapshots"], prepared["shots"])
    assert prepared["finished"]["time_ns"] < quantum["started"]["time_ns"]
    assert quantum["finished"]["time_ns"] < result["started"]["time_ns"]
    assert prepared["started"]["allocation"]["SLURM_JOB_PARTITION"] == "compute"
    assert quantum["started"]["allocation"]["SLURM_JOB_PARTITION"] == "quantum"
    assert quantum["started"]["allocation"]["SLURM_JOB_LICENSES"].removesuffix(":1") == quantum["device"]
    assert len({rank["node"] for rank in result["rank_allocations"]}) == result["ranks"] == 4
    assert all(not rank["allocation"].get("SLURM_JOB_LICENSES") for rank in result["rank_allocations"])
    assert all(rank["allocation"]["SLURM_JOB_PARTITION"] == "compute" for rank in result["rank_allocations"])
    assert np.isclose(result["fci_energy"], -0.9834727290331733, rtol=0, atol=1e-9)
    for key, tolerance in (("overlap", 1e-9), ("force_bias", 1e-7), ("eigenstate_energy", 1e-9)):
        assert result["invariant_errors"][key] < tolerance
    curves = result["curves_mean_stderr_effective_walkers"]
    assert all(np.all(np.isfinite(curve)) for curve in curves.values())
    assert all(np.all(np.asarray(curve)[:, 1:] >= 0) for curve in curves.values())
    sys.stdout.write(f"Validated {quantum['snapshots']} circuits on {quantum['device']} and four classical ranks\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", help="Run in a matching quantum HRES allocation")
    parser.add_argument("--directory", type=Path, help="Validate a completed scientific run")
    args = parser.parse_args()
    if args.device:
        readout(args.device)
    elif args.directory:
        artifacts(args.directory)
    else:
        parser.error("Choose --device or --directory")
