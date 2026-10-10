# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Run a LiH active-space quantum-classical AFQMC example locally through MQT DDSIM.

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
import multiprocessing
import operator as op
import os
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from functools import partial
from itertools import combinations, starmap
from pathlib import Path
from time import perf_counter_ns
from typing import TYPE_CHECKING, Any, TypeVar
from unittest.mock import patch

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


DETERMINANTS = np.asarray(list(combinations(range(6), 2)))


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
    """Expand a two-electron Slater determinant without approximation.

    Returns:
        Amplitudes in lexicographic occupied-orbital order.
    """
    first, second = np.triu_indices(len(walker), 1)
    return walker[first, 0] * walker[second, 1] - walker[second, 0] * walker[first, 1]


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


TRIAL_PARAMETERS = np.asarray([0.01138048, 0.29796188, 0.06563527, -0.02079659, 1.00227801])
_WORKER: dict[str, Any] = {}


def lithium_hydride_trial() -> None:
    """Prepare the fixed number-conserving LiH active-space trial on an occupied reference."""
    for angle, wires in zip(TRIAL_PARAMETERS[:2], ([0, 1, 2, 3], [0, 1, 4, 5]), strict=True):
        qml.DoubleExcitation(angle, wires=wires)
    for angle, (first, last) in zip(TRIAL_PARAMETERS[2:], ((0, 2), (0, 4), (2, 4)), strict=True):
        for spin in (0, 1):
            qml.FermionicSingleExcitation(float(angle), wires=range(first + spin, last + spin + 1))


def make_cached_trial(prop: Any, shadow: dict, bra: NDArray, hamiltonian: NDArray) -> Any:  # ruff: ignore[any-type] - pinned external helper has no type stubs
    """Use exact active-space contractions for the reconstructed quantum trial.

    Returns:
        An adapter accepted by the pinned upstream phaseless propagator.
    """
    quantum_trial = importlib.import_module("afqmc.trial_wavefunction.quantum_ovlp")
    openfermion = importlib.import_module("openfermion")
    indices = [sum(1 << (5 - int(orbital)) for orbital in occupied) for occupied in DETERMINANTS]
    one_body_matrices = []
    for operator in prop.L_gamma:
        full = openfermion.get_sparse_operator(
            openfermion.InteractionOperator(0, operator, np.zeros((6, 6, 6, 6)))
        ).toarray()
        one_body_matrices.append(full[np.ix_(indices, indices)])

    class CachedTrial(quantum_trial.QTrial):
        # shortcut: enumerate this 15-state active space; larger spaces need scalable overlap estimators.
        def compute_ovlp(self, walker: NDArray) -> complex:
            return complex(self.coefficients @ determinant_amplitudes(walker))

        def compute_local_energy(self, walker: NDArray, overlap: complex) -> tuple[complex, dict]:
            amplitudes = determinant_amplitudes(walker)
            electronic = self.coefficients @ self.hamiltonian @ amplitudes - self.nuclear_repulsion * overlap
            return complex(electronic), {}

        def compute_one_body_local(self, walker: NDArray, operators: list, overlap: complex, _cache: dict) -> NDArray:
            assert operators is self.L_gamma
            return self.coefficients @ self.one_body_matrices @ determinant_amplitudes(walker) / overlap

    trial = CachedTrial(prop, shadow)
    trial.coefficients = bra
    trial.hamiltonian = hamiltonian
    trial.one_body_matrices = np.asarray(one_body_matrices)
    return trial


def initialize_worker(prop: Any, shadow: dict, bra: NDArray, hamiltonian: NDArray, reference: float) -> None:  # ruff: ignore[any-type] - pinned external helper has no type stubs
    """Create one immutable trial per worker process, shared by that process's independent walkers."""
    single_slater = importlib.import_module("afqmc.trial_wavefunction.single_slater")
    _WORKER.update({
        "prop": prop,
        "reference": reference,
        "quantum": make_cached_trial(prop, shadow, bra, hamiltonian),
        "classical": single_slater.SingleSlater(prop, np.eye(6, 2, dtype=np.complex128)),
    })


def propagate_one_walker(label: str, walker_id: int, seed: int, steps: int, dtau: float) -> dict:
    """Propagate one independent walker and record actual worker/step completion times.

    Returns:
        Raw local energies, old weights, sampled states, and monotonic-clock observations.

    Raises:
        ValueError: If a walker produces a nonfinite weight or energy.
    """
    qmc = importlib.import_module(f"afqmc.qmc.{'quantum_shadow' if label == 'quantum' else 'classical'}")
    np.random.seed(seed)  # ruff: ignore[numpy-legacy-random] - the pinned propagators consume the global RNG
    trial = _WORKER[label]
    walker, weight = np.eye(6, 2, dtype=np.complex128), 1.0
    started = perf_counter_ns()
    energies, weights, frames = [], [], []
    for step in range(steps):
        if label == "quantum":
            energy, new_walker, new_weight = qmc.cqa_imag_time_propogator(  # spellchecker:disable-line
                dtau, trial, walker, weight, _WORKER["reference"]
            )
        else:
            energy, new_walker, new_weight = qmc.imag_time_propogator(  # spellchecker:disable-line
                dtau, trial, walker, weight, _WORKER["prop"], _WORKER["reference"]
            )
        if not np.isfinite(energy) or not np.isfinite(new_weight) or new_weight < 0:
            msg = "AFQMC produced a nonfinite local energy or invalid importance weight"
            raise ValueError(msg)
        energies.append(float(energy.real))
        weights.append(float(weight))
        if step % max(1, steps // 80) == 0 or step == steps - 1:
            frames.append({
                "step": step,
                "tau": step * dtau,
                "completed_ns": perf_counter_ns(),
                "id": walker_id,
                "weight": float(weight),
                "local_energy": float(energy.real),
                "occupations": np.sum(np.abs(walker) ** 2, axis=1).tolist(),
                "overlap_magnitude": float(abs(trial.compute_ovlp(walker))),
            })
        walker, weight = new_walker, new_weight
    return {
        "label": label,
        "walker_id": walker_id,
        "seed": seed,
        "pid": os.getpid(),
        "started_ns": started,
        "finished_ns": perf_counter_ns(),
        "energies": energies,
        "weights": weights,
        "frames": frames,
    }


def capture_lih_afqmc(compact: dict, walkers: int, steps: int, dtau: float, seed: int, processes: int) -> dict:
    """Run a genuine LiH CAS(2,3) calculation with four local classical worker processes.

    The trial amplitudes come only from the measured matchgate shadows. Dense
    contractions replace the upstream H2 estimator, which fails an independent
    local-energy check for this larger active space. Its propagator is unchanged.

    Returns:
        Molecular references, shadow overlaps, raw parallel observations, and energy curves.

    Raises:
        ValueError: If reference Hamiltonians, overlaps, or importance weights fail validation.
    """
    openfermion = importlib.import_module("openfermion")
    molecular_data = importlib.import_module("openfermion.chem.molecular_data")
    pyscf = importlib.import_module("pyscf")
    mcscf = importlib.import_module("pyscf.mcscf")
    chemistry = importlib.import_module("afqmc.utils.chemical_preparation")
    quantum_trial = importlib.import_module("afqmc.trial_wavefunction.quantum_ovlp")
    pyscf.lib.num_threads(1)
    begin = perf_counter_ns()
    molecule = pyscf.gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
    hf = molecule.RHF().run()
    active = [2, 3, 6]
    cas = mcscf.CASCI(hf, 3, (1, 1))
    cas.kernel(cas.sort_mo(active))
    reference_energy = float(cas.e_tot)
    prop = chemistry.chemistry_preparation(molecule, hf, active_orbitals=active, nel=(1, 1))
    one_body, two_body = molecular_data.spinorb_from_spatial(prop.h1e, prop.eri)
    full_hamiltonian = openfermion.get_sparse_operator(
        openfermion.InteractionOperator(prop.nuclear_repulsion, one_body, 0.5 * two_body)
    ).toarray()
    basis_indices = [sum(1 << (5 - int(orbital)) for orbital in occupied) for occupied in DETERMINANTS]
    hamiltonian = full_hamiltonian[np.ix_(basis_indices, basis_indices)]
    if (
        not np.allclose(hamiltonian, hamiltonian.conj().T, atol=1e-12)
        or not np.isclose(np.linalg.eigvalsh(hamiltonian)[0], reference_energy, atol=1e-10)
        or not np.isclose(hamiltonian[0, 0], hf.e_tot, atol=1e-10)
    ):
        msg = "The independent active-space Hamiltonian disagrees with PySCF CASCI or Hartree-Fock"
        raise ValueError(msg)
    with qml.queuing.AnnotatedQueue() as queue:
        lithium_hydride_trial()
    unitary = np.asarray(qml.matrix(qml.tape.QuantumScript.from_queue(queue), wire_order=range(6)))
    exact_bra = unitary[basis_indices, 48].conj()
    if not np.isclose(np.vdot(exact_bra, exact_bra), 1, atol=1e-12) or not np.allclose(unitary[:, 0], np.eye(64)[:, 0]):
        msg = "The trial circuit must conserve the two-electron sector and leave the vacuum reference unchanged"
        raise ValueError(msg)
    chemistry_ms = (perf_counter_ns() - begin) / 1e6
    begin = perf_counter_ns()
    shadow = {key: np.asarray(value) for key, value in compact.items()}
    shadow_trial = quantum_trial.QTrial(prop, shadow)
    bra = np.asarray([shadow_trial.compute_ovlp(np.eye(6)[:, occupied]) for occupied in DETERMINANTS])
    cached = make_cached_trial(prop, shadow, bra, hamiltonian)
    rng = np.random.default_rng(seed + 1)
    overlap_errors, upstream_energy_errors, force_bias_errors = [], [], []
    for _ in range(12):
        spatial, _ = np.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
        walker = np.kron(spatial[:, :1], np.eye(2))
        overlap = cached.compute_ovlp(walker)
        overlap_errors.append(abs(overlap - shadow_trial.compute_ovlp(walker)))
        forces = cached.compute_one_body_local(walker, cached.L_gamma, overlap, {})
        for one_body_operator, force in zip(cached.L_gamma, forces, strict=True):
            delta = 1e-5 * one_body_operator
            derivative = (
                cached.compute_ovlp((np.eye(6) + delta) @ walker) - cached.compute_ovlp((np.eye(6) - delta) @ walker)
            ) / (2e-5 * overlap)
            force_bias_errors.append(abs(derivative - force))
        numerator = cached.compute_local_energy(walker, overlap)[0]
        upstream_energy_errors.append(abs(numerator - shadow_trial.compute_local_energy(walker, overlap)[0]))
    if max(force_bias_errors) > 1e-7:
        msg = "The active-space force bias disagrees with the independent overlap derivative"
        raise ValueError(msg)
    if max(overlap_errors) > 1e-9:
        msg = "The cached determinant overlaps disagree with the measured-shadow Pfaffian estimator"
        raise ValueError(msg)
    # An exact eigenstate trial must give its eigenvalue for every nonorthogonal walker.
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    oracle_trial = make_cached_trial(prop, shadow, eigenvectors[:, 0].conj(), hamiltonian)
    oracle_errors = []
    for _ in range(12):
        walker, _ = np.linalg.qr(rng.normal(size=(6, 2)) + 1j * rng.normal(size=(6, 2)))
        overlap = oracle_trial.compute_ovlp(walker)
        oracle_errors.append(
            abs(
                oracle_trial.compute_local_energy(walker, overlap)[0] / overlap
                + prop.nuclear_repulsion
                - eigenvalues[0]
            )
        )
    if max(oracle_errors) > 1e-9:
        msg = "The active-space local-energy adapter failed the exact-eigenstate identity"
        raise ValueError(msg)
    reconstruction_ms = (perf_counter_ns() - begin) / 1e6
    seeds = np.random.SeedSequence(seed).generate_state(walkers).tolist()
    started = perf_counter_ns()
    tasks, results, completion_events = [], [], []
    with ProcessPoolExecutor(
        max_workers=processes,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_worker,
        initargs=(prop, shadow, bra, hamiltonian, float(hf.e_tot)),
    ) as pool:
        for label in ("quantum", "classical"):
            for walker_id, walker_seed in enumerate(seeds):
                tasks.append(pool.submit(propagate_one_walker, label, walker_id, walker_seed, steps, dtau))
        for completed in as_completed(tasks):
            result = completed.result()
            completion_events.append({
                "time_ms": (perf_counter_ns() - started) / 1e6,
                "operation": "Future.result()",
                "walker_id": result["walker_id"],
                "label": result["label"],
                "worker_pid": result["pid"],
                "status": "returned",
            })
            results.append(result)
    propagation_ms = (perf_counter_ns() - started) / 1e6
    curves = []
    for label, title in (("quantum", "Quantum shadow trial"), ("classical", "Hartree-Fock trial")):
        selected = sorted((r for r in results if r["label"] == label), key=op.itemgetter("walker_id"))
        energies = np.asarray([r["energies"] for r in selected]).T
        weights = np.asarray([r["weights"] for r in selected]).T
        statistics = np.asarray(list(starmap(weighted_energy_statistics, zip(energies, weights, strict=True))))
        frames = []
        for index, sample in enumerate(selected[0]["frames"]):
            frame = {"step": sample["step"], "tau": sample["tau"], "walkers": []}
            for result in selected:
                record = result["frames"][index].copy()
                record["completed_ms"] = (record.pop("completed_ns") - started) / 1e6
                record["worker_pid"] = result["pid"]
                frame["walkers"].append(record)
            frames.append(frame)
        curves.append({
            "label": title,
            "energy": statistics[:, 0].tolist(),
            "stderr": statistics[:, 1].tolist(),
            "effective_walkers": statistics[:, 2].tolist(),
            "frames": frames,
            "walker_energies": energies.tolist(),
            "walker_weights": weights.tolist(),
            "duration_ms": (max(r["finished_ns"] for r in selected) - min(r["started_ns"] for r in selected)) / 1e6,
        })
    intervals = [
        {
            "label": r["label"],
            "walker_id": r["walker_id"],
            "seed": r["seed"],
            "worker_pid": r["pid"],
            "started_ms": (r["started_ns"] - started) / 1e6,
            "finished_ms": (r["finished_ns"] - started) / 1e6,
        }
        for r in results
    ]
    return {
        "chemistry": {
            "molecule": "LiH",
            "elements": ["Li", "H"],
            "geometry_angstrom": [[0, 0, 0], [0, 0, 1.6]],
            "basis": "STO-3G",
            "electrons": 2,
            "total_electrons": 4,
            "spin_orbitals": 6,
            "active_space": "CAS(2 electrons, 3 spatial orbitals)",
            "active_orbitals_one_based": active,
            "frozen_core": "Doubly occupied Li 1s orbital; omit the two π virtual orbitals",
            "reference_scope": "FCI within the stated active space; includes frozen-core and nuclear energies",
            "hf_energy": float(hf.e_tot),
            "fci_energy": reference_energy,
            "trial_energy": float((exact_bra @ hamiltonian @ exact_bra.conj()).real),
            "shadow_trial_energy": float((bra @ hamiltonian @ bra.conj() / (bra @ bra.conj())).real),
            "nuclear_and_frozen_core_energy": float(prop.nuclear_repulsion),
            "duration_ms": chemistry_ms,
            "hamiltonian_real": hamiltonian.real.tolist(),
            "energy_unit": "Ha",
            "trial_parameters": TRIAL_PARAMETERS.tolist(),
            "trial_method": (
                "Fixed parameters tuned classically in the 15-state model; not a quantum VQE or advantage claim"
            ),
        },
        "overlap": {
            "determinants": [
                "".join("1" if wire in occupied else "0" for wire in range(6)) for occupied in DETERMINANTS
            ],
            "shadow_bra_real": bra.real.tolist(),
            "shadow_bra_imag": bra.imag.tolist(),
            "exact_bra_real": exact_bra.real.tolist(),
            "exact_bra_imag": exact_bra.imag.tolist(),
            "max_absolute_coefficient_error": float(np.max(abs(bra - exact_bra))),
            "cached_overlap_max_error": float(max(overlap_errors)),
            "local_energy_max_error": float(max(oracle_errors)),
            "force_bias_max_error": float(max(force_bias_errors)),
            "upstream_local_energy_numerator_max_error": float(max(upstream_energy_errors)),
            "validation_walkers": len(overlap_errors),
            "duration_ms": reconstruction_ms,
            "method": "15 measured Pfaffian overlaps; exact Slater expansion and Hamiltonian/force-bias contractions",
        },
        "propagation": {
            "algorithm": "Phaseless AFQMC; unchanged upstream propagator with verified active-space contractions",
            "walkers": walkers,
            "steps": steps,
            "dtau": dtau,
            "seed": seed,
            "walker_seeds": seeds,
            "tau": (dtau * np.arange(steps)).tolist(),
            "curves": curves,
            "duration_ms": propagation_ms,
            "processes": processes,
            "worker_pids": sorted({r["pid"] for r in results}),
            "tasks": intervals,
            "events": completion_events,
            "clock": "perf_counter_ns shared monotonic clock; measured local worker intervals and completion calls",
            "parallel_source": inspect.getsource(propagate_one_walker),
            "weight_control": "Raw positive importance weights; no population resampling",
            "stderr_scope": (
                "One walker-only standard error conditional on measured shadows; excludes shadow and systematic errors"
            ),
            "frame_scope": (
                "Recorded orbital occupations/weights at imaginary-time steps; worker timestamps are wall time"
            ),
            "limits": [
                "Six-qubit active space and local ideal simulation; no hardware or quantum-advantage claim",
                "Dense 15-state post-processing is specific to this demonstration",
                "Finite shadows, projection time, walker population, time step, and phaseless approximation",
                "Two curves use the same per-walker auxiliary-field seeds; their errors are correlated",
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
    parser.add_argument("--snapshots", type=int, default=2048)
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--walkers", type=int, default=128)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--dtau", type=float, default=0.02)
    parser.add_argument("--processes", type=int, default=4)
    args = parser.parse_args()
    if (
        min(args.snapshots, args.shots, args.steps, args.processes) <= 0
        or args.walkers < 2
        or not 0 < args.dtau <= 0.05
    ):
        msg = "Counts must be positive, walkers at least two, and the imaginary-time step in (0, 0.05]"
        raise ValueError(msg)
    for name, expected in SOURCE_HASHES.items():
        if hashlib.sha256((args.source / name).read_bytes()).hexdigest() != expected:
            msg = f"Upstream source does not match pinned revision: {name}"
            raise ValueError(msg)
    sys.path.insert(0, str(args.source.resolve()))

    matchgate = importlib.import_module("afqmc.utils.matchgate")
    shadow = importlib.import_module("afqmc.utils.shadow")
    # Check the pinned external API before collecting any quantum data.
    quantum_qmc = importlib.import_module("afqmc.qmc.quantum_shadow")
    classical_qmc = importlib.import_module("afqmc.qmc.classical")
    assert callable(quantum_qmc.cqa_imag_time_propogator)  # spellchecker:disable-line
    assert callable(classical_qmc.imag_time_propogator)  # spellchecker:disable-line
    apply_gaussian_givens = matchgate.apply_gaussian_givens
    apply_pauli_layer = matchgate.apply_pauli_layer

    np.random.seed(args.seed)  # ruff: ignore[numpy-legacy-random] - upstream helpers consume the legacy global RNG
    rotations = [shadow.random_signed_permutation(12) for _ in range(max(32, args.snapshots))]
    compiled = [matchgate.compile_gaussian_givens(rotation) for rotation in rotations]
    schedule = compiled[0][0]
    if any(entry[0] != schedule for entry in compiled):
        msg = "Matchgate schedule must be identical across snapshots"
        raise ValueError(msg)
    angles = -np.stack([entry[1] for entry in compiled])
    paulis = np.stack([entry[2] for entry in compiled])
    gammas = [
        np.asarray(qml.matrix(qml.prod(*([qml.Z(q) for q in range(p)] + [operator(p)])), wire_order=range(6)))
        for p in range(6)
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
            unitary = np.asarray(qml.matrix(qml.prod(*reversed(circuit.operations)), wire_order=range(6)))
            error = max(
                np.max(np.abs(unitary.conj().T @ gamma @ unitary - sum(rotation[j, k] * gammas[k] for k in range(12))))
                for j, gamma in enumerate(gammas)
            )
            errors_to_append.append(float(error))
    if max(errors) > 1e-10:
        msg = "Matchgate circuit violates the declared Majorana transformation"
        raise ValueError(msg)
    rotations = rotations[: args.snapshots]
    angles, paulis = angles[: args.snapshots], paulis[: args.snapshots]

    device = qml.device("mqt.ddsim.default", wires=6)

    @qml.set_shots(shots=args.shots)
    @qml.qnode(device)
    def lithium_hydride_shadow_circuit(
        thetas: NDArray[np.float64], pauli_vectors: NDArray[np.float64]
    ) -> tuple[qml.measurements.SampleMP, qml.measurements.CountsMP]:
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        lithium_hydride_trial()
        apply_gaussian_givens(thetas, schedule)
        apply_pauli_layer(pauli_vectors)
        return qml.sample(wires=range(6)), qml.counts(wires=range(6))

    job_class = importlib.import_module("mqt.core.plugins.pennylane.job").PennyLaneJob
    original_submit, original_result, original_samples = (
        job_class._submit_programs,  # ruff: ignore[private-member-access] - observe the real adapter boundary
        job_class.result,
        job_class._samples,  # ruff: ignore[private-member-access] - observe indexed result decoding
    )
    methods = (original_submit, original_result, original_samples)
    client_source = "\n".join(inspect.getsource(method) for method in methods)
    source_lines = client_source.splitlines()
    batch_events = []
    started = perf_counter_ns()

    def traced_call(operation: str, needle: str, method: Callable[..., T], *arguments: object, **details: object) -> T:
        begin = perf_counter_ns()
        value = method(*arguments)
        end = perf_counter_ns()
        batch_events.append({
            "time_ms": (begin - started) / 1e6,
            "duration_ms": (end - begin) / 1e6,
            "actor": "PennyLane application",
            "target": "Core QDMI adapter",
            "operation": operation,
            "status": "returned",
            "source_line": next(i + 1 for i, line in enumerate(source_lines) if needle in line),
            **details,
        })
        return value

    def submit(batch: object, indices: list) -> object:
        return traced_call(
            "Device.try_submit_job(programs)",
            "self._device.qdmi_device.try_submit_job(",
            original_submit,
            batch,
            indices,
            programs=len(indices),
        )

    def collect(batch: object) -> object:
        return traced_call("PennyLaneJob.result(): wait and collect", "self._batch.complete()", original_result, batch)

    def samples(batch: object, index: int, job: object, program_index: int) -> object:
        return traced_call(
            "QDMI indexed shots and PennyLane decoding",
            "self._shots_or_counts(job, program_index)",
            original_samples,
            batch,
            index,
            job,
            program_index,
            program_index=program_index,
        )

    with (
        patch.object(job_class, "_submit_programs", submit),
        patch.object(job_class, "result", collect),
        patch.object(job_class, "_samples", samples),
    ):
        samples_batch, counts_batch = lithium_hydride_shadow_circuit(angles, paulis)
    finished = perf_counter_ns()
    batch_ms = (finished - started) / 1e6
    if device.last_job is None or len(device.last_job.entries) != args.snapshots:
        msg = "PennyLane did not retain one batch entry per snapshot"
        raise ValueError(msg)
    representative_tape = qml.workflow.construct_tape(lithium_hydride_shadow_circuit)(angles[0], paulis[0])
    representative_source = qml.to_openqasm(representative_tape, wires=range(6))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_name("afqmc-source.qasm").write_text(representative_source)
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
                "operation": "lithium_hydride_shadow_circuit(angles, paulis)",
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
                    "code": inspect.getsource(lithium_hydride_shadow_circuit.func),
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
    science = capture_lih_afqmc(compact, args.walkers, args.steps, args.dtau, args.seed, args.processes)
    data = {
        "schema_version": 1,
        "title": "Lithium hydride quantum-classical AFQMC",
        "label": "Actual PennyLane → MQT Core → QDMI → DDSIM batch",
        "scope": (
            "LiH CAS(2,3)/STO-3G quantum trial shadows and classical phaseless AFQMC, "
            "with exact small-system references"
        ),
        "workload": {
            "molecule": "LiH",
            "spin_orbitals": 6,
            "ansatz": "Vacuum reference plus a six-qubit number-conserving LiH CAS(2,3) trial",
            "snapshots": args.snapshots,
            "shots_per_snapshot": args.shots,
            "total_shots": args.snapshots * args.shots,
            "submitted_qdmi_jobs": device.submitted_jobs,
            "batch_duration_ms": batch_ms,
            "measurement_order": "PennyLane wire order 0,1,2,3,4,5; raw QDMI strings are stored separately",
            "permutation_seed": args.seed,
            "simulator_seed": "DDSIM default randomness; actual ordered shots are retained",
            "schedule": schedule,
        },
        **science,
        "execution": {
            "events": sorted(batch_events, key=op.itemgetter("time_ms")),
            "client_source": client_source,
            "client_source_note": (
                "Actual Core PennyLane adapter methods; "
                "observed Python call boundaries include native wait and decoding"
            ),
            "duration_ms": batch_ms,
            "completed_ms": batch_ms,
            "num_programs": args.snapshots,
            "shots_per_program": args.shots,
            "terminal_status": "DONE",
            "submitted_qdmi_jobs": device.submitted_jobs,
            "trace_kind": "python-observed-adapter-calls",
            "backend": "Local ideal MQT DDSIM",
        },
        "batch_source": "\n".join(
            line.strip()
            for line in inspect.getsource(main).splitlines()
            if line.strip().startswith((
                "angles = -np.stack",
                "paulis = np.stack",
                "samples_batch, counts_batch = lithium_hydride_shadow_circuit",
            ))
        ),
        "representative_source": {
            "code": representative_source,
            "sha256": hashlib.sha256(representative_source.encode()).hexdigest(),
            "language": "qasm",
            "format": "OpenQASM 2.0",
            "label": "Actual first LiH shadow circuit before device compilation",
        },
        "source": inspect.getsource(lithium_hydride_shadow_circuit.func),
        "trial_source": inspect.getsource(lithium_hydride_trial),
        "classical_source": inspect.getsource(capture_lih_afqmc),
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
            "label": "LiH quantum-classical AFQMC",
            "application": True,
            "summary": (
                "Real six-wire PennyLane → Core → QDMI shadows drive classical phaseless AFQMC walkers. "
                "Uses ideal DDSIM directly, not the Emerald compilation target."
            ),
            "parameters": {"qubits": 6, "snapshots": args.snapshots, "shots_per_snapshot": args.shots},
            "device_label": "Local ideal six-wire DDSIM; application capture",
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
                    "in PennyLane; verify each circuit by exact 64x64 matrices"
                ),
                "Collect sample rows alongside counts and execute a broadcast batch on MQT DDSIM",
                "Cache 15 determinant overlaps; exact linearity replaces repeated Pfaffian reconstruction",
                "Use local processes with independent seeded walkers and actual worker timings",
                (
                    "Replace the upstream larger-space local-energy and force-bias contractions with independently "
                    "verified 15-state Hamiltonian contractions; retain its phaseless propagator unchanged"
                ),
            ],
            "omitted": "Hardware noise, large active spaces, population resampling, and full uncertainty analysis",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
