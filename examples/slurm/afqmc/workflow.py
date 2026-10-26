# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Prepare, sample, and propagate the SC26 H2 QC-AFQMC workflow in separate jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import time_ns
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

    from mqt.core.mlir import QIRProgram

import numpy as np
from science import hydrogen_trial, import_cached_linalg, propagate


def record() -> dict:
    """Record actual Slurm allocation and process identity."""
    return {
        "time_ns": time_ns(),
        "node": socket.gethostname(),
        "allocation": {
            name: value
            for name, value in os.environ.items()
            if name
            in {"SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_JOB_PARTITION", "SLURM_JOB_LICENSES", "SLURM_NTASKS"}
        },
    }


def write_json(path: Path, value: object) -> None:
    """Atomically publish a complete artifact for dependent jobs."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare(directory: Path, snapshots: int, shots: int, seed: int) -> None:
    """Generate every shadow circuit and check its Majorana transformation."""
    import pennylane as qml
    from afqmc.utils import matchgate, shadow

    started = record()
    np.random.seed(seed)  # ruff: ignore[numpy-legacy-random] - upstream uses the global RNG
    rotations = [shadow.random_signed_permutation(8) for _ in range(snapshots)]
    programs = directory / "source"
    programs.mkdir()
    gammas = np.asarray([
        qml.matrix(qml.prod(*([qml.Z(q) for q in range(p)] + [operator(p)])), wire_order=range(4))
        for p in range(4)
        for operator in (qml.X, qml.Y)
    ])
    permutations, signs, errors, hashes = [], [], [], []
    for index, rotation in enumerate(rotations):
        schedule, angles, paulis = matchgate.compile_gaussian_givens(rotation)
        # The pinned AWS decomposition uses the opposite Majorana angle convention.
        with qml.queuing.AnnotatedQueue() as queue:
            matchgate.apply_gaussian_givens(-angles, schedule)
            matchgate.apply_pauli_layer(paulis)
        basis = qml.tape.QuantumScript.from_queue(queue)
        unitary = np.asarray(qml.matrix(basis, wire_order=range(4)))
        error = float(np.max(abs(unitary.conj().T @ gammas @ unitary - np.einsum("jk,kab->jab", rotation, gammas))))
        if error > 1e-10:
            msg = "Matchgate circuit violates its Majorana transformation"
            raise ValueError(msg)
        errors.append(error)
        with qml.queuing.AnnotatedQueue() as queue:
            qml.Hadamard(0)
            qml.CNOT([0, 1])
            hydrogen_trial()
            matchgate.apply_gaussian_givens(-angles, schedule)
            matchgate.apply_pauli_layer(paulis)
            qml.sample(wires=range(4))
        tape = qml.tape.QuantumScript.from_queue(queue, shots=shots)
        source = qml.to_openqasm(tape, wires=range(4))
        (programs / f"{index}.qasm").write_text(source, encoding="utf-8")
        hashes.append(hashlib.sha256(source.encode()).hexdigest())
        permutation, sign = shadow.signed_permutation_to_compact(rotation)
        permutations.append(permutation)
        signs.append(sign)
    np.savez_compressed(directory / "bases.npz", perm=permutations, sign=signs)
    write_json(
        directory / "prepared.json",
        {
            "snapshots": snapshots,
            "shots": shots,
            "seed": seed,
            "source_sha256": hashes,
            "majorana_max_error": max(errors),
            "started": started,
            "finished": record(),
        },
    )


def quantum(directory: Path, device_id: str) -> None:
    """Compile for the allocated device, submit real QDMI jobs, and retain readout."""
    from mqt.core.mlir import CompilationOptions, CompilerTarget, OutputFormat, compile_program
    from mqt.core.qdmi import Job, ProgramFormat, builtin_driver

    allocated = os.environ.get("SLURM_JOB_LICENSES", "").removesuffix(":1")
    if allocated != device_id:
        msg = "The selected device must match the single HRES allocation"
        raise ValueError(msg)
    started = record()
    metadata = json.loads((directory / "prepared.json").read_text())
    sources = [(directory / "source" / f"{i}.qasm").read_text() for i in range(metadata["snapshots"])]
    if [hashlib.sha256(source.encode()).hexdigest() for source in sources] != metadata["source_sha256"]:
        msg = "Prepared circuit identity changed"
        raise ValueError(msg)
    device = builtin_driver.open_device(device_id)
    target = CompilerTarget.from_device(device)
    is_braket = device_id.startswith("amazon.braket.")
    output = (
        OutputFormat.QISKIT
        if is_braket
        else (
            OutputFormat.IQM_JSON
            if ProgramFormat.IQM_JSON in device.supported_program_formats()
            else OutputFormat.QIR_BASE
        )
    )
    options = CompilationOptions(seed=metadata["seed"])

    def compile_source(source: str) -> str | QuantumCircuit:
        program = compile_program(source, target=target, output=output, options=options)
        if is_braket:
            return cast("QuantumCircuit", program)
        # Release each native compiler context before retaining the next payload.
        return str(program) if output == OutputFormat.IQM_JSON else cast("QIRProgram", program).to_llvm_ir()

    with ThreadPoolExecutor(max_workers=4) as workers:
        compiled = list(workers.map(compile_source, sources))
    compiled_at = record()
    if is_braket:
        from amazon.braket.qdmi.qiskit import AmazonBraketBackend

        circuits = cast("list[QuantumCircuit]", compiled)
        # This example uses a four-site logical simulator; the provider owns its wire mapping.
        if any(
            any(circuit.find_bit(wire).index >= 4 for instruction in circuit.data for wire in instruction.qubits)
            for circuit in circuits
        ):
            msg = "The Braket example requires dense placement on the first four simulator sites"
            raise ValueError(msg)
        adapter_job = AmazonBraketBackend(device=device).run(circuits, shots=metadata["shots"], memory=True)
        attempts = [entry.attempts[-1] for entry in adapter_job.entries]
        handles = [(attempt.handle, attempt.program_index) for attempt in attempts]
        write_json(directory / "submission.json", {"job_ids": sorted({job.id for job, _ in handles})})
        adapter_job.wait_for_final_state(timeout=1800, wait=0.25)
        result = adapter_job.result()
        logical_shots = [result.get_memory(i) for i in range(metadata["snapshots"])]
    else:
        program_format = ProgramFormat.IQM_JSON if output == OutputFormat.IQM_JSON else ProgramFormat.QIR_BASE_STRING
        payloads = cast("list[str]", compiled)
        job = device.submit_job(payloads, program_format, metadata["shots"])
        write_json(directory / "submission.json", {"job_ids": [job.id]})
        if not job.wait(1800):
            job.cancel()
            msg = "QDMI sampling did not complete in 30 minutes"
            raise TimeoutError(msg)
        handles = [(job, index) for index in range(len(payloads))]
        logical_shots = [job.get_shots(index) for index in range(len(payloads))]
    if len(handles) != metadata["snapshots"]:
        msg = "The provider returned an incomplete batch"
        raise ValueError(msg)
    for job in {job for job, _ in handles}:
        if job is None or job.check() != Job.Status.DONE:
            msg = "A quantum program did not complete successfully"
            raise ValueError(msg)
    payload_directory = directory / "compiled"
    payload_directory.mkdir()
    raw, bits, counts, snapshot_ids, payload_hashes = [], [], [], [], []
    for index, ((job, program_index), samples) in enumerate(zip(handles, logical_shots, strict=True)):
        native = job.get_shots(program_index)
        if Counter(native) != job.get_counts(program_index) or len(samples) != metadata["shots"]:
            msg = "Quantum shots, counts, or shot count disagree"
            raise ValueError(msg)
        if any(len(sample) != 4 or set(sample) - {"0", "1"} for sample in samples):
            msg = "Expected four logical measurement bits"
            raise ValueError(msg)
        payload = job.get_program(program_index)
        if not isinstance(payload, str):
            msg = "This example retains text payloads"
            raise TypeError(msg)
        (payload_directory / f"{index}.txt").write_text(payload, encoding="utf-8")
        payload_hashes.append(hashlib.sha256(payload.encode()).hexdigest())
        raw.append(samples)
        for outcome, count in Counter(sample[::-1] for sample in samples).items():
            bits.append([int(bit) for bit in outcome])
            counts.append(count)
            snapshot_ids.append(index)
    bases = np.load(directory / "bases.npz")
    np.savez_compressed(
        directory / "shadow.npz", perm=bases["perm"], sign=bases["sign"], bits=bits, count=counts, snap_id=snapshot_ids
    )
    np.savez_compressed(directory / "shots.npz", logical=raw)
    write_json(
        directory / "quantum.json",
        {
            "device": device_id,
            "snapshots": metadata["snapshots"],
            "shots_per_snapshot": metadata["shots"],
            "payload_sha256": payload_hashes,
            "started": started,
            "compiled": compiled_at,
            "finished": record(),
        },
    )


def classical(directory: Path, walkers: int, steps: int) -> None:
    """Run AFQMC on classical nodes after the quantum allocation has ended."""
    from mpi4py import MPI

    if os.environ.get("SLURM_JOB_LICENSES"):
        msg = "Classical propagation must not hold the quantum HRES"
        raise ValueError(msg)
    started = record()
    metadata = json.loads((directory / "prepared.json").read_text())
    shadow = dict(np.load(directory / "shadow.npz"))
    result = propagate(shadow, walkers, steps, metadata["seed"], MPI.COMM_WORLD)
    allocations = MPI.COMM_WORLD.gather(record(), root=0)
    if result is not None:
        write_json(
            directory / "result.json",
            {**result, "started": started, "finished": record(), "rank_allocations": allocations},
        )


def main() -> None:
    """Run one stage with the retained upstream scientific implementation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "quantum", "classical"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("/workload"))
    parser.add_argument("--device", default="mqt.ddsim.default")
    parser.add_argument("--snapshots", type=int, default=4096)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--walkers", type=int, default=128)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if min(args.snapshots, args.shots, args.steps) < 1 or args.walkers < 2:
        parser.error("Positive counts and at least two walkers are required")
    hashes = json.loads(Path(__file__).with_name("source-hashes.json").read_text())
    for name, expected in hashes.items():
        if hashlib.sha256((args.source / name).read_bytes()).hexdigest() != expected:
            msg = f"Upstream AFQMC source differs from the pinned revision: {name}"
            raise ValueError(msg)
    sys.path.insert(0, str(args.source.resolve()))
    if args.stage != "quantum":
        import_cached_linalg()
    if args.stage == "prepare":
        prepare(args.directory, args.snapshots, args.shots, args.seed)
    elif args.stage == "quantum":
        quantum(args.directory, args.device)
    else:
        classical(args.directory, args.walkers, args.steps)


if __name__ == "__main__":
    main()
