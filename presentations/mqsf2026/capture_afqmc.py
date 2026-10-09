# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Run the pinned H2 quantum-classical AFQMC example locally through MQT DDSIM.

Record molecular references, quantum shadows, reconstructed overlaps, phaseless
walker propagation, and weighted molecular energies for offline presentation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from functools import partial
from itertools import combinations
from pathlib import Path
from time import perf_counter_ns
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import pennylane as qml

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
    "afqmc/utils/chemical_preparation.py": "6959626928280c83908405f948382ea4d9b6eb39bf310448d5ea7b36c04dab30",
    "afqmc/trial_wavefunction/quantum_ovlp.py": "ec44db91f324582396e98940bd5e296b742fa7a0d01e75859b25af233a47f482",
    "afqmc/trial_wavefunction/single_slater.py": "da13a89f1391d9fc18e927edd71aa0b3110d9a3d8eb1e75dfe9998afa25c82b0",
    "afqmc/estimators/ci.py": "fa90f3d017fa2fae20c866ae78babb14f29d9e43f5b8d03dc6765ca9a140b4a2",
    "afqmc/estimators/greens_function.py": "f5023b733dc721733c680ea01cc2a1f1c30d42093dac903c4f2a6701c0191aba",
    "afqmc/estimators/local_energy.py": "0c1d456b8d17e930362d995d8305cd010c8c82f9e9914b3d2a57cdfd8db85ab8",
    "afqmc/qmc/quantum_shadow.py": "6e262e2c0696a801e3f6019205c8cfc3f1a3f181a54134be9b84bb2e19cdf44e",
    "afqmc/qmc/classical.py": "132ca6cb1f43376d05bad994eebacd1b86cfef266418da65321adf4e9cffc5df",
}


DETERMINANTS = np.asarray(list(combinations(range(4), 2)))


def validate_native_batch(submitted_jobs: int, snapshot_count: int, snapshots: list[dict]) -> None:
    """Require the captured circuits to occupy one native job in input order.

    Raises:
        ValueError: If fallback, missing results, or inconsistent program indices weaken the batch claim.
    """
    if (
        submitted_jobs != 1
        or snapshot_count <= 0
        or len(snapshots) != snapshot_count
        or any(
            snapshot["native_program_index"] != index or snapshot["native_job_num_programs"] != snapshot_count
            for index, snapshot in enumerate(snapshots)
        )
    ):
        msg = "All shadow circuits must occupy one native QDMI job with sequential program indices"
        raise ValueError(msg)


def determinant_amplitudes(walker: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Expand a four-orbital, two-electron Slater determinant without approximation.

    Returns:
        Amplitudes in lexicographic occupied-orbital order.
    """
    return np.linalg.det(walker[DETERMINANTS])


def weighted_energy_statistics(energies: NDArray, weights: NDArray) -> tuple[float, float, float]:
    """Return the ratio estimate, independent-walker SE, and effective sample size.

    The standard error conditions on the single collected shadow data set. It
    excludes shadow sampling error, time-step error, and phaseless bias.

    Raises:
        ValueError: If energies or weights do not define a finite weighted estimate.
    """
    if len(energies) < 2 or not np.all(np.isfinite(energies)) or not np.all(np.isfinite(weights)):
        msg = "Energy statistics require at least two finite walkers"
        raise ValueError(msg)
    if np.any(weights < 0) or np.sum(weights) <= 0:
        msg = "AFQMC weights must be nonnegative with positive total weight"
        raise ValueError(msg)
    normalized = weights / np.sum(weights)
    mean = float(normalized @ energies.real)
    variance = np.sum(normalized**2 * (energies.real - mean) ** 2) * len(weights) / (len(weights) - 1)
    return mean, float(np.sqrt(variance)), float(1 / np.sum(normalized**2))


def capture_h2_afqmc(compact: dict, walkers: int, steps: int, dtau: float, seed: int) -> dict:
    """Complete the pinned tutorial's H2 chemistry and phaseless AFQMC workflow.

    Reconstruct six determinant overlaps once, then reuse their exact linear
    expansion in the unchanged upstream energy, force-bias, and propagation
    routines. This shortcut is specific to the tiny demonstration space.

    Returns:
        Verified chemistry, reconstructed overlaps, and measured propagation data.

    Raises:
        ValueError: If a scientific identity fails or walker weights become invalid.
    """
    openfermion = importlib.import_module("openfermion")
    molecular_data = importlib.import_module("openfermion.chem.molecular_data")
    pyscf = importlib.import_module("pyscf")

    chemistry = importlib.import_module("afqmc.utils.chemical_preparation")
    quantum_trial = importlib.import_module("afqmc.trial_wavefunction.quantum_ovlp")
    classical_trial = importlib.import_module("afqmc.trial_wavefunction.single_slater")
    quantum_qmc = importlib.import_module("afqmc.qmc.quantum_shadow")
    classical_qmc = importlib.import_module("afqmc.qmc.classical")
    pyscf.lib.num_threads(1)
    begin = perf_counter_ns()
    molecule = pyscf.gto.M(atom="H 0 0 0; H 0 0 0.75", basis="sto-3g", verbose=0)
    hf = molecule.RHF().run()
    reference_energy = float(pyscf.fci.FCI(hf).kernel()[0])
    prop = chemistry.chemistry_preparation(molecule, hf)
    one_body, two_body = molecular_data.spinorb_from_spatial(prop.h1e, prop.eri)
    full_hamiltonian = openfermion.get_sparse_operator(
        openfermion.InteractionOperator(prop.nuclear_repulsion, one_body, 0.5 * two_body)
    ).toarray()
    basis_indices = [sum(1 << (3 - int(orbital)) for orbital in occupied) for occupied in DETERMINANTS]
    hamiltonian = full_hamiltonian[np.ix_(basis_indices, basis_indices)]
    if not np.allclose(hamiltonian, hamiltonian.conj().T, atol=1e-12) or not np.isclose(
        np.linalg.eigvalsh(hamiltonian)[0], reference_energy, atol=1e-10
    ):
        msg = "The independent two-electron Hamiltonian disagrees with PySCF FCI"
        raise ValueError(msg)
    chemistry_ms = (perf_counter_ns() - begin) / 1e6
    begin = perf_counter_ns()
    shadow_trial = quantum_trial.QTrial(prop, {key: np.asarray(value) for key, value in compact.items()})
    bra = np.asarray([shadow_trial.compute_ovlp(np.eye(4)[:, occupied]) for occupied in DETERMINANTS])
    exact_bra = np.asarray([np.cos(0.06), 0, 0, 0, 0, -np.sin(0.06)], dtype=np.complex128)

    class CachedTrial(quantum_trial.QTrial):
        # shortcut: enumerate six determinants only for H2, use scalable overlap routines for larger spaces.
        def compute_ovlp(self, walker: NDArray) -> complex:
            return complex(self.coefficients @ determinant_amplitudes(walker))

    cached_trial = CachedTrial(prop, shadow_trial.shadow)
    cached_trial.coefficients = bra
    rng = np.random.default_rng(seed + 1)
    overlap_errors, local_energy_errors = [], []
    for _ in range(8):
        spatial, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
        walker = np.kron(spatial[:, :1], np.eye(2))
        overlap = cached_trial.compute_ovlp(walker)
        overlap_errors.append(abs(overlap - shadow_trial.compute_ovlp(walker)))
        energy = cached_trial.compute_local_energy(walker, overlap)[0] / overlap + prop.nuclear_repulsion
        amplitudes = determinant_amplitudes(walker)
        local_energy_errors.append(abs(energy - bra @ hamiltonian @ amplitudes / overlap))
    if max(overlap_errors + local_energy_errors) > 1e-9:
        msg = "Cached shadow overlaps or upstream local energies violate the independent determinant-space check"
        raise ValueError(msg)
    reconstruction_ms = (perf_counter_ns() - begin) / 1e6
    initial = np.eye(4, 2, dtype=np.complex128)
    hf_trial = classical_trial.SingleSlater(prop, initial)
    curves = []
    begin = perf_counter_ns()
    for label, trial in (("Quantum shadow trial", cached_trial), ("Hartree-Fock trial", hf_trial)):
        # Both curves receive the same auxiliary-field sequence for a controlled comparison.
        np.random.seed(seed)  # ruff: ignore[numpy-legacy-random] - upstream propagators consume the global RNG
        states = [initial.copy() for _ in range(walkers)]
        weights = np.ones(walkers)
        curve = {
            "label": label,
            "energy": [],
            "stderr": [],
            "effective_walkers": [],
            "frames": [],
            "walker_energies": [],
            "walker_weights": [],
        }
        curve_begin = perf_counter_ns()
        for step in range(steps):
            energies = np.empty(walkers, dtype=np.complex128)
            new_states, new_weights = [], []
            for index, (walker, weight) in enumerate(zip(states, weights, strict=True)):
                if label == "Quantum shadow trial":
                    energy, state, new_weight = quantum_qmc.cqa_imag_time_propagator(
                        dtau, trial, walker, weight, float(hf.e_tot)
                    )
                else:
                    energy, state, new_weight = classical_qmc.imag_time_propagator(
                        dtau, trial, walker, weight, prop, float(hf.e_tot)
                    )
                energies[index] = energy
                new_states.append(state)
                new_weights.append(new_weight)
            mean, stderr, effective = weighted_energy_statistics(energies, weights)
            curve["energy"].append(mean)
            curve["stderr"].append(stderr)
            curve["effective_walkers"].append(effective)
            curve["walker_energies"].append(energies.real.tolist())
            curve["walker_weights"].append(weights.tolist())
            if step % max(1, steps // 24) == 0 or step == steps - 1:
                curve["frames"].append({
                    "step": step,
                    "tau": step * dtau,
                    "walkers": [
                        {
                            "id": index,
                            "weight": float(weight),
                            "local_energy": float(energy.real),
                            "occupations": np.sum(np.abs(state) ** 2, axis=1).tolist(),
                            "overlap_magnitude": float(abs(trial.compute_ovlp(state))),
                        }
                        for index, (state, weight, energy) in enumerate(zip(states, weights, energies, strict=True))
                    ],
                })
            states = new_states
            weights = np.asarray(new_weights)
            if not np.all(np.isfinite(weights)) or np.sum(weights) <= 0:
                msg = "AFQMC walker weights became invalid or all walkers died"
                raise ValueError(msg)
            weights /= np.mean(weights)
        curve["duration_ms"] = (perf_counter_ns() - curve_begin) / 1e6
        curves.append(curve)
    propagation_ms = (perf_counter_ns() - begin) / 1e6
    return {
        "chemistry": {
            "molecule": "H2",
            "geometry_angstrom": [[0, 0, 0], [0, 0, 0.75]],
            "basis": "STO-3G",
            "electrons": 2,
            "spin_orbitals": 4,
            "hf_energy": float(hf.e_tot),
            "fci_energy": reference_energy,
            "trial_energy": float((exact_bra @ hamiltonian @ exact_bra).real),
            "shadow_trial_energy": float((bra @ hamiltonian @ bra.conj() / (bra @ bra.conj())).real),
            "nuclear_repulsion": float(prop.nuclear_repulsion),
            "duration_ms": chemistry_ms,
            "hamiltonian_real": hamiltonian.real.tolist(),
            "energy_unit": "Ha",
        },
        "overlap": {
            "determinants": [
                "".join("1" if wire in occupied else "0" for wire in range(4)) for occupied in DETERMINANTS
            ],
            "shadow_bra_real": bra.real.tolist(),
            "shadow_bra_imag": bra.imag.tolist(),
            "exact_bra_real": exact_bra.real.tolist(),
            "max_absolute_coefficient_error": float(np.max(abs(bra - exact_bra))),
            "cached_overlap_max_error": float(max(overlap_errors)),
            "local_energy_max_error": float(max(local_energy_errors)),
            "validation_walkers": len(overlap_errors),
            "duration_ms": reconstruction_ms,
            "method": "Six Pfaffian shadow overlaps, then exact Slater determinant expansion in the H2 space",
        },
        "propagation": {
            "algorithm": "Phaseless importance-sampled AFQMC; upstream propagators unchanged",
            "walkers": walkers,
            "steps": steps,
            "dtau": dtau,
            "seed": seed,
            "tau": (dtau * np.arange(steps)).tolist(),
            "curves": curves,
            "duration_ms": propagation_ms,
            "weight_control": "Common weight normalization after each step; no population resampling",
            "stderr_scope": (
                "One standard error across independent walkers, conditional on the measured shadows; "
                "excludes shadow error and systematic bias"
            ),
            "frame_scope": (
                "Actual orbital occupations and weights; "
                "point positions in the presentation are schematic, not electron trajectories"
            ),
            "limits": [
                "Four spin orbitals and two electrons; no quantum advantage claim",
                "Finite shadow sample shared by every walker; its sampling error is not in the bands",
                "Finite projection time, finite walker population, finite time step, and phaseless approximation",
                "Exact state and FCI are validation references only; shadow walkers use measured coefficients",
            ],
        },
    }


def main() -> None:
    """Verify the scientific convention and retain actual samples and payloads.

    Raises:
        ValueError: If pinned source, circuit identities, job status, or independent result views disagree.
    """
    qdmi = importlib.import_module("mqt.core.qdmi")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="Pinned upstream Quantum_Monte_Carlo_Chemistry directory"
    )
    parser.add_argument("--output", type=Path, default=HERE / "captures/afqmc.json")
    parser.add_argument("--snapshots", type=int, default=512)
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--walkers", type=int, default=64)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--dtau", type=float, default=0.02)
    args = parser.parse_args()
    if min(args.snapshots, args.shots, args.walkers, args.steps) <= 0 or not 0 < args.dtau <= 0.05:
        msg = "Counts must be positive and the imaginary-time step must be in (0, 0.05]"
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
        msg = "PennyLane did not retain one batch entry per snapshot"
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
        attempt = entry.attempts[-1]
        job = attempt.handle
        program_index = attempt.program_index
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

        if job is None or observe("Job.check()", job.check, events) != qdmi.Job.Status.DONE:
            msg = "A shadow job did not succeed"
            raise ValueError(msg)
        events[-1]["status"] = "DONE"
        shots = observe("Job.get_shots(program_index)", partial(job.get_shots, program_index), events)
        raw_counts = observe("Job.get_counts(program_index)", partial(job.get_counts, program_index), events)
        program = observe("Job.get_program(program_index)", partial(job.get_program, program_index), events)
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
            "native_program_index": program_index,
            "native_job_num_programs": job.num_programs,
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
                "native_program_index": program_index,
                "native_job_num_programs": job.num_programs,
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
    validate_native_batch(device.submitted_jobs, args.snapshots, snapshots)
    science = capture_h2_afqmc(compact, args.walkers, args.steps, args.dtau, args.seed)
    data = {
        "schema_version": 1,
        "title": "Hydrogen quantum-classical AFQMC",
        "label": "Actual PennyLane → MQT Core → QDMI → DDSIM batch",
        "scope": "H2/STO-3G quantum trial shadows and classical phaseless AFQMC, with exact small-system references",
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
            "simulator_seed": "DDSIM default randomness; actual ordered shots are retained",
            "schedule": schedule,
        },
        **science,
        "source": inspect.getsource(hydrogen_shadow_circuit.func),
        "classical_source": inspect.getsource(capture_h2_afqmc),
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
            "label": "H2 quantum-classical AFQMC",
            "application": True,
            "summary": (
                "Real four-wire PennyLane → Core → QDMI shadows drive classical phaseless AFQMC walkers. "
                "Uses ideal DDSIM directly, not the Emerald compilation target."
            ),
            "parameters": {"qubits": 4, "snapshots": args.snapshots, "shots_per_snapshot": args.shots},
            "device_label": "Local ideal four-wire DDSIM; application capture",
            "variants": variants[:1],
        },
        "provenance": {
            "source_url": SOURCE_URL,
            "source_revision": REVISION,
            "source_license": "Apache-2.0",
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "pennylane", "pyscf", "openfermion", "numba", "scipy")
            },
            "source_sha256": SOURCE_HASHES,
            "core_revision": subprocess.check_output(  # ruff: ignore[subprocess-without-shell-equals-true] - fixed read-only Git query
                [shutil.which("git") or "/usr/bin/git", "rev-parse", "HEAD"], cwd=HERE, text=True
            ).strip(),
            "command": "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python " + " ".join(sys.argv),
            "capture_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "generated_at": datetime.now(UTC).isoformat(),
            "patches": [
                (
                    "Negate upstream compile_gaussian_givens angles to satisfy its documented Majorana convention "
                    "in PennyLane; verify each circuit by exact 16x16 matrices"
                ),
                "Collect sample rows alongside counts and execute a broadcast batch on MQT DDSIM",
                "Cache six determinant overlaps; exact linearity replaces repeated Pfaffian reconstruction",
                "Use upstream phaseless propagators with seeded walkers and capture all energies and weights",
            ],
            "omitted": "Hardware noise, large active spaces, population resampling, and full uncertainty analysis",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
