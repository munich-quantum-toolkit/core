# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Four-qubit H2 AFQMC contractions checked against the pinned AWS example.

The six-determinant cache is specific to this active space. Statistical errors
condition on one shadow data set and exclude shadow error and systematic biases.
"""

from __future__ import annotations

import importlib
import operator
from functools import partial
from itertools import combinations, starmap
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import numpy as np

if TYPE_CHECKING:
    from mpi4py import MPI
    from numpy.typing import NDArray

DETERMINANTS = np.asarray(list(combinations(range(4), 2)))
TRIAL_PARAMETERS = np.asarray([0.6])


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


def hydrogen_trial() -> None:
    """Prepare the paired H2 trial from |1100>, preserving the vacuum reference.

    This state preparation equals DoubleExcitation on the two occupied input
    columns only; it is not a replacement for that gate on arbitrary inputs.
    """
    import pennylane as qml

    qml.CNOT(wires=[0, 1])
    qml.SingleExcitation(float(TRIAL_PARAMETERS[0]), wires=[0, 2])
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[2, 3])


def make_cached_trial(prop: Any, shadow: dict, bra: NDArray, hamiltonian: NDArray) -> Any:  # ruff: ignore[any-type] - pinned external helper has no type stubs
    """Use exact active-space contractions for the reconstructed quantum trial.

    Returns:
        An adapter accepted by the pinned upstream phaseless propagator.
    """
    quantum_trial = importlib.import_module("afqmc.trial_wavefunction.quantum_ovlp")
    openfermion = importlib.import_module("openfermion")
    indices = [sum(1 << (3 - int(orbital)) for orbital in occupied) for occupied in DETERMINANTS]
    one_body_matrices = []
    for matrix in prop.L_gamma:
        full = openfermion.get_sparse_operator(
            openfermion.InteractionOperator(0, matrix, np.zeros((4, 4, 4, 4)))
        ).toarray()
        one_body_matrices.append(full[np.ix_(indices, indices)])

    class CachedTrial(quantum_trial.QTrial):
        # shortcut: enumerate this 6-state active space; larger spaces need scalable overlap estimators.
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


def import_cached_linalg() -> None:
    """Reuse image-built Numba kernels without changing the pinned upstream source."""
    numba = importlib.import_module("numba")
    # Explicit upstream signatures otherwise compile at import in every fresh Slurm rank.
    with (
        patch.object(numba, "njit", partial(numba.njit, cache=True)),
        patch.object(numba, "guvectorize", partial(numba.guvectorize, cache=True)),
    ):
        importlib.import_module("afqmc.utils.linalg")


def prepare_trial(shadow: dict, seed: int) -> tuple:
    """Reconstruct measured overlaps and check Hamiltonian and force-bias identities."""
    import openfermion
    import pennylane as qml
    import pyscf
    from afqmc.trial_wavefunction.quantum_ovlp import QTrial
    from afqmc.utils.chemical_preparation import chemistry_preparation
    from openfermion.chem.molecular_data import spinorb_from_spatial
    from pyscf import mcscf

    pyscf.lib.num_threads(1)
    molecule = pyscf.gto.M(atom="H 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
    hf = molecule.RHF().run()
    cas = mcscf.CASCI(hf, 2, (1, 1))
    cas.kernel(cas.sort_mo([1, 2]))
    prop = chemistry_preparation(molecule, hf, active_orbitals=[1, 2], nel=(1, 1))
    one, two = spinorb_from_spatial(prop.h1e, prop.eri)
    full = openfermion.get_sparse_operator(
        openfermion.InteractionOperator(prop.nuclear_repulsion, one, 0.5 * two)
    ).toarray()
    indices = [sum(1 << (3 - int(orbital)) for orbital in occupied) for occupied in DETERMINANTS]
    hamiltonian = full[np.ix_(indices, indices)]
    assert np.allclose(hamiltonian, hamiltonian.conj().T, atol=1e-12)
    assert np.isclose(np.linalg.eigvalsh(hamiltonian)[0], cas.e_tot, atol=1e-10)
    assert np.isclose(hamiltonian[0, 0], hf.e_tot, atol=1e-10)
    with qml.queuing.AnnotatedQueue() as queue:
        hydrogen_trial()
    unitary = np.asarray(qml.matrix(qml.tape.QuantumScript.from_queue(queue), wire_order=range(4)))
    original = np.asarray(qml.matrix(qml.DoubleExcitation(0.6, wires=range(4))))
    assert np.allclose(unitary[:, [0, 12]], original[:, [0, 12]], atol=1e-12)
    assert np.allclose(unitary[:, 0], np.eye(16)[:, 0], atol=1e-12)
    exact_bra = unitary[indices, 12].conj()
    trial = QTrial(prop, shadow)
    bra = np.asarray([trial.compute_ovlp(np.eye(4)[:, occupied]) for occupied in DETERMINANTS])
    cached = make_cached_trial(prop, shadow, bra, hamiltonian)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    oracle = make_cached_trial(prop, shadow, eigenvectors[:, 0].conj(), hamiltonian)
    rng = np.random.default_rng(seed + 1)
    errors = {"overlap": [], "force_bias": [], "eigenstate_energy": []}
    for _ in range(8):
        spatial, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
        walker = np.kron(spatial[:, :1], np.eye(2))
        overlap = cached.compute_ovlp(walker)
        errors["overlap"].append(abs(overlap - trial.compute_ovlp(walker)))
        forces = cached.compute_one_body_local(walker, cached.L_gamma, overlap, {})
        for matrix, force in zip(cached.L_gamma, forces, strict=True):
            delta = 1e-5 * matrix
            derivative = (
                cached.compute_ovlp((np.eye(4) + delta) @ walker) - cached.compute_ovlp((np.eye(4) - delta) @ walker)
            ) / (2e-5 * overlap)
            errors["force_bias"].append(abs(derivative - force))
        overlap = oracle.compute_ovlp(walker)
        energy = oracle.compute_local_energy(walker, overlap)[0] / overlap + prop.nuclear_repulsion
        errors["eigenstate_energy"].append(abs(energy - eigenvalues[0]))
    maxima = {key: float(max(values)) for key, values in errors.items()}
    assert maxima["overlap"] < 1e-9
    assert maxima["force_bias"] < 1e-7
    assert maxima["eigenstate_energy"] < 1e-9
    summary = {
        "hf_energy": float(hf.e_tot),
        "fci_energy": float(cas.e_tot),
        "shadow_bra_real": bra.real.tolist(),
        "shadow_bra_imag": bra.imag.tolist(),
        "shadow_coefficient_max_error": float(np.max(abs(bra - exact_bra))),
        "invariant_errors": maxima,
    }
    return prop, bra, hamiltonian, summary


def propagate(shadow: dict, walkers: int, steps: int, seed: int, comm: MPI.Comm) -> dict | None:
    """Distribute independent measured-trial and Hartree-Fock walkers over MPI ranks."""
    from afqmc.qmc import classical, quantum_shadow
    from afqmc.trial_wavefunction.single_slater import SingleSlater

    prepared = prepare_trial(shadow, seed) if comm.rank == 0 else None
    prop, bra, hamiltonian, summary = comm.bcast(prepared, root=0)
    trials = {
        "quantum": make_cached_trial(prop, shadow, bra, hamiltonian),
        "classical": SingleSlater(prop, np.eye(4, 2, dtype=np.complex128)),
    }
    seeds = np.random.SeedSequence(seed).generate_state(walkers)
    tasks = [(label, index, int(value)) for label in ("quantum", "classical") for index, value in enumerate(seeds)]
    local = []
    for label, index, value in tasks[comm.rank :: comm.size]:
        np.random.seed(value)  # ruff: ignore[numpy-legacy-random] - pinned propagators use the global RNG
        walker, weight = np.eye(4, 2, dtype=np.complex128), 1.0
        energies, weights = [], []
        for _ in range(steps):
            if label == "quantum":
                energy, updated, new_weight = quantum_shadow.cqa_imag_time_propogator(
                    0.02, trials[label], walker, weight, summary["hf_energy"]
                )
            else:
                energy, updated, new_weight = classical.imag_time_propogator(
                    0.02, trials[label], walker, weight, prop, summary["hf_energy"]
                )
            if not np.isfinite(energy) or not np.isfinite(new_weight) or new_weight < 0:
                msg = "AFQMC produced invalid energy or importance weight"
                raise ValueError(msg)
            energies.append(float(energy.real))
            weights.append(float(weight))
            walker, weight = updated, new_weight
        local.append({
            "label": label,
            "index": index,
            "seed": value,
            "energy": energies,
            "weight": weights,
            "rank": comm.rank,
        })
    gathered = comm.gather(local, root=0)
    if comm.rank != 0:
        return None
    results = [item for batch in gathered for item in batch]
    curves = {}
    for label in trials:
        selected = sorted((item for item in results if item["label"] == label), key=operator.itemgetter("index"))
        assert len(selected) == walkers
        energies = np.asarray([item["energy"] for item in selected]).T
        weights = np.asarray([item["weight"] for item in selected]).T
        curves[label] = list(starmap(weighted_energy_statistics, zip(energies, weights, strict=True)))
    return {
        **summary,
        "curves_mean_stderr_effective_walkers": curves,
        "walker_results": results,
        "ranks": comm.size,
        "dtau": 0.02,
        "energy_unit": "Ha",
        "limits": (
            "H2/STO-3G, ideal simulation; walker-only uncertainty conditional on measured shadows, "
            "no quantum advantage claim"
        ),
    }
