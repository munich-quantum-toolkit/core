# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Collect a small H2 matchgate-shadow batch through PennyLane and Core DDSIM.

Reuse the pinned public AFQMC helpers from --source. This captures the quantum
data-collection step, not classical AFQMC propagation or an energy estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter_ns
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import pennylane as qml

from mqt.core.qdmi import Job

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

T = TypeVar("T")

HERE = Path(__file__).resolve().parent
REVISION = "16cd791da7c3ec7e104851eb3bc502d00be1c1c3"
SOURCE_URL = f"https://github.com/amazon-braket/amazon-braket-examples/tree/{REVISION}/examples/hybrid_quantum_algorithms/Quantum_Monte_Carlo_Chemistry"
SOURCE_HASHES = {
    "afqmc/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "afqmc/utils/matchgate.py": "bdff3d85aa6f3a4ef1d31a5ad123343b49a23f5da5e1e5dface9c9834330aa45",
    "afqmc/utils/shadow.py": "ece374005d6934aab1c84c642fc801f123effe06cf73dd4b015ed66a72c8f5fc",
    "afqmc/utils/linalg.py": "f6b0dc16bae9d3ff8bf83c924e7c71c32ee2dcbbeedb8310ef479d685223d2af",
}


def main() -> None:
    """Verify the scientific convention and retain actual samples and payloads.

    Raises:
        ValueError: If pinned source, circuit identities, job status, or independent result views disagree.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="Pinned upstream Quantum_Monte_Carlo_Chemistry directory"
    )
    parser.add_argument("--output", type=Path, default=HERE / "captures/afqmc.json")
    parser.add_argument("--snapshots", type=int, default=16)
    parser.add_argument("--shots", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.snapshots <= 0 or args.shots <= 0:
        msg = "Snapshots and shots must be positive"
        raise ValueError(msg)
    for name, expected in SOURCE_HASHES.items():
        if hashlib.sha256((args.source / name).read_bytes()).hexdigest() != expected:
            msg = f"Upstream source does not match pinned revision: {name}"
            raise ValueError(msg)
    sys.path.insert(0, str(args.source.resolve()))

    matchgate = importlib.import_module("afqmc.utils.matchgate")
    shadow = importlib.import_module("afqmc.utils.shadow")
    apply_gaussian_givens = matchgate.apply_gaussian_givens
    apply_pauli_layer = matchgate.apply_pauli_layer

    np.random.seed(args.seed)  # ruff: ignore[numpy-legacy-random] - upstream helpers consume the legacy global RNG
    rotations = [shadow.random_signed_permutation(8) for _ in range(max(32, args.snapshots))]
    compiled = [matchgate.compile_gaussian_givens(rotation) for rotation in rotations]
    schedule = compiled[0][0]
    if any(entry[0] != schedule for entry in compiled):
        msg = "Matchgate schedule must be identical across snapshots"
        raise ValueError(msg)
    angles = -np.stack([entry[1] for entry in compiled])
    paulis = np.stack([entry[2] for entry in compiled])
    gammas = [
        np.asarray(qml.matrix(qml.prod(*([qml.Z(q) for q in range(p)] + [operator(p)])), wire_order=range(4)))
        for p in range(4)
        for operator in (qml.X, qml.Y)
    ]
    errors = []
    uncorrected_errors = []

    for index, rotation in enumerate(rotations):
        for thetas, errors_to_append in ((angles[index], errors), (-angles[index], uncorrected_errors)):
            with qml.queuing.AnnotatedQueue() as queue:
                apply_gaussian_givens(thetas, schedule)
                apply_pauli_layer(paulis[index])
            circuit = qml.tape.QuantumScript.from_queue(queue)
            unitary = np.asarray(qml.matrix(qml.prod(*reversed(circuit.operations)), wire_order=range(4)))
            error = max(
                np.max(np.abs(unitary.conj().T @ gamma @ unitary - sum(rotation[j, k] * gammas[k] for k in range(8))))
                for j, gamma in enumerate(gammas)
            )
            errors_to_append.append(float(error))
    if max(errors) > 1e-10:
        msg = "Matchgate circuit violates the declared Majorana transformation"
        raise ValueError(msg)
    rotations = rotations[: args.snapshots]
    angles, paulis = angles[: args.snapshots], paulis[: args.snapshots]

    device = qml.device("mqt.ddsim.default", wires=4)

    @qml.set_shots(shots=args.shots)
    @qml.qnode(device)
    def hydrogen_shadow_circuit(
        thetas: NDArray[np.float64], pauli_vectors: NDArray[np.float64]
    ) -> tuple[qml.measurements.SampleMP, qml.measurements.CountsMP]:
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.DoubleExcitation(0.12, wires=[0, 1, 2, 3])
        apply_gaussian_givens(thetas, schedule)
        apply_pauli_layer(pauli_vectors)
        return qml.sample(wires=range(4)), qml.counts(wires=range(4))

    started = perf_counter_ns()
    samples_batch, counts_batch = hydrogen_shadow_circuit(angles, paulis)
    finished = perf_counter_ns()
    batch_ms = (finished - started) / 1e6
    if device.last_job is None or len(device.last_job.entries) != args.snapshots:
        msg = "PennyLane did not retain one QDMI job per snapshot"
        raise ValueError(msg)
    compact = {name: [] for name in ("perm", "sign", "bits", "count", "snap_id")}
    snapshots = []
    variants = []

    def observe(name: str, call: Callable[[], T], events: list[dict[str, Any]]) -> T:
        begin = perf_counter_ns()
        value = call()
        end = perf_counter_ns()
        events.append({
            "time_ms": (begin - started) / 1e6,
            "duration_ms": (end - begin) / 1e6,
            "actor": "AFQMC capture",
            "target": "Core QDMI Job",
            "operation": name,
            "status": "returned",
        })
        return value

    for index, (rotation, samples, counts_result, entry) in enumerate(
        zip(rotations, samples_batch, counts_batch, device.last_job.entries, strict=True)
    ):
        job = entry.attempts[-1].handle
        events = [
            {
                "time_ms": 0.0,
                "duration_ms": batch_ms,
                "actor": "AFQMC application",
                "target": "PennyLane / MQT Core",
                "operation": "hydrogen_shadow_circuit(angles, paulis)",
                "status": "returned",
                "detail": f"Actual broadcast call: {args.snapshots} circuits, {args.shots} shots each",
            }
        ]

        if job is None or observe("Job.check()", job.check, events) != Job.Status.DONE:
            msg = "A shadow job did not succeed"
            raise ValueError(msg)
        events[-1]["status"] = "DONE"
        shots = observe("Job.get_shots()", job.get_shots, events)
        raw_counts = observe("Job.get_counts()", job.get_counts, events)
        program = observe("Job.program", lambda job=job: job.program, events)
        prepared = device.last_job._prepared[index][0]  # ruff: ignore[private-member-access] - compare the adapter's exact pre-submission payload
        if program != prepared.payload or job.program_format != prepared.program_format:
            msg = "PennyLane's submitted QDMI payload changed"
            raise ValueError(msg)
        if Counter(shots) != raw_counts or len(shots) != args.shots:
            msg = "QDMI shots and histogram disagree"
            raise ValueError(msg)
        wire_shots = ["".join(str(int(bit)) for bit in row) for row in samples]
        counts = {str(key): int(value) for key, value in counts_result.items()}
        if wire_shots != [shot[::-1] for shot in shots] or Counter(wire_shots) != counts:
            msg = "PennyLane did not preserve the actual QDMI shot sequence and wire order"
            raise ValueError(msg)
        permutation, sign = shadow.signed_permutation_to_compact(rotation)
        compact["perm"].append(permutation.tolist())
        compact["sign"].append(sign.tolist())
        for bitstring, count in counts.items():
            compact["bits"].append([int(bit) for bit in bitstring])
            compact["count"].append(count)
            compact["snap_id"].append(index)
        snapshots.append({
            "id": index,
            "angles": angles[index].tolist(),
            "pauli_vector": paulis[index].tolist(),
            "shots": wire_shots,
            "counts": counts,
            "qdmi_shots": shots,
            "qdmi_counts": raw_counts,
            "program_format": job.program_format.name,
            "payload": program,
            "payload_sha256": hashlib.sha256(program.encode()).hexdigest(),
            "payload_identity_verified": True,
            "terminal_status": "DONE",
            "majorana_max_error": errors[index],
        })
        variants.append({
            "id": f"snapshot-{index}",
            "label": f"Matchgate snapshot {index + 1}",
            "stages": [
                {
                    "id": "pennylane",
                    "label": "PennyLane application",
                    "language": "python",
                    "code": inspect.getsource(hydrogen_shadow_circuit.func),
                }
            ],
            "exports": [
                {"id": "openqasm3", "label": "Actual Core PennyLane payload", "language": "qasm", "code": program}
            ],
            "layout": {"initial": [], "final": [], "swaps": []},
            "execution": {
                "format": "QASM3",
                "backend": "MQT DDSIM",
                "compilation_target": "DDSIM native gate set via the PennyLane adapter",
                "payload_sha256": hashlib.sha256(program.encode()).hexdigest(),
                "payload_identity_verified": True,
                "num_shots": args.shots,
                "terminal_status": "DONE",
                "shots": shots,
                "counts": raw_counts,
                "duration_ms": batch_ms,
                "duration_scope": f"Whole {args.snapshots}-circuit PennyLane batch, shared by all snapshots",
                "trace_kind": "python-observed-api",
                "events": events,
            },
        })
    data = {
        "schema_version": 1,
        "title": "Hydrogen matchgate-shadow collection",
        "label": "Actual PennyLane → MQT Core → QDMI → DDSIM batch",
        "scope": "Quantum shadow collection only; no AFQMC propagation or molecular energy calculation",
        "workload": {
            "molecule": "H2",
            "spin_orbitals": 4,
            "ansatz": "Vacuum reference superposition with DoubleExcitation(0.12), following the upstream H2 notebook",
            "snapshots": args.snapshots,
            "shots_per_snapshot": args.shots,
            "total_shots": args.snapshots * args.shots,
            "submitted_qdmi_jobs": device.submitted_jobs,
            "batch_duration_ms": batch_ms,
            "measurement_order": "PennyLane wire order 0,1,2,3; raw QDMI strings are stored separately",
            "permutation_seed": args.seed,
            "simulator_seed": "independent system entropy per QDMI job",
            "schedule": schedule,
        },
        "source": inspect.getsource(hydrogen_shadow_circuit.func),
        "snapshots": snapshots,
        "shadow": compact,
        "validation": {
            "majorana_checks": len(errors),
            "majorana_max_error": max(errors),
            "uncorrected_upstream_majorana_max_error": max(uncorrected_errors),
            "checked_identity": "U(Q)† gamma_j U(Q) = sum_k Q[j,k] gamma_k",
            "ordered_shots_match_qdmi": True,
            "counts_match_actual_shots": True,
        },
        "scenario": {
            "id": "afqmc",
            "label": "H2 matchgate shadows",
            "application": True,
            "summary": (
                "Real four-wire PennyLane → Core → QDMI shadow collection; no molecular-energy claim. "
                "Uses ideal DDSIM directly, not the Emerald compilation target."
            ),
            "parameters": {"qubits": 4, "snapshots": args.snapshots, "shots_per_snapshot": args.shots},
            "device_label": "Local ideal four-wire DDSIM; application capture",
            "variants": variants,
        },
        "provenance": {
            "source_url": SOURCE_URL,
            "source_revision": REVISION,
            "source_license": "Apache-2.0",
            "source_sha256": SOURCE_HASHES,
            "core_revision": subprocess.check_output(  # ruff: ignore[subprocess-without-shell-equals-true] - fixed read-only Git query
                [shutil.which("git") or "/usr/bin/git", "rev-parse", "HEAD"], cwd=HERE, text=True
            ).strip(),
            "capture_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "generated_at": datetime.now(UTC).isoformat(),
            "patches": [
                (
                    "Negate upstream compile_gaussian_givens angles to satisfy its documented Majorana convention "
                    "in PennyLane; verify each circuit by exact 16x16 matrices"
                ),
                "Collect sample rows alongside counts and execute a small broadcast batch on MQT DDSIM",
            ],
            "omitted": "PySCF chemistry setup, overlap reconstruction, walker propagation, and energy estimation",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
