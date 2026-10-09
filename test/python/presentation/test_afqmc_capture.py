# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Copyright (c) 2026 Munich Quantum Software Company GmbH
# SPDX-License-Identifier: MIT

"""Scientific contracts of the small AFQMC presentation capture."""

from __future__ import annotations

import gzip
import importlib.util
import json
from itertools import starmap
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pennylane")
SPEC = importlib.util.spec_from_file_location(
    "capture_afqmc", Path(__file__).resolve().parents[3] / "presentations/mqsf2026/capture_afqmc.py"
)
assert SPEC is not None
assert SPEC.loader is not None
CAPTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAPTURE)


def test_determinant_expansion_preserves_complex_phase_and_antisymmetry() -> None:
    """Cached overlaps retain the phase and orbital ordering of a complex walker."""
    walker = np.array([[1, 0], [0, 1j], [1j, 0], [0, 1]], dtype=complex) / np.sqrt(2)
    amplitudes = CAPTURE.determinant_amplitudes(walker)
    np.testing.assert_allclose(amplitudes, [0.5j, 0, 0.5, 0.5, 0, 0.5j], atol=1e-14)
    np.testing.assert_allclose(CAPTURE.determinant_amplitudes(walker[:, ::-1]), -amplitudes, atol=1e-14)
    np.testing.assert_allclose(np.vdot(amplitudes, amplitudes), 1)


def test_walker_uncertainty_is_invariant_to_common_weight_normalization() -> None:
    """Weight normalization must preserve the energy, SE, and effective population."""
    energies = np.array([-1.1, -1.2, -1.3])
    weights = np.array([1.0, 2.0, 1.0])
    statistics = CAPTURE.weighted_energy_statistics(energies, weights)
    np.testing.assert_allclose(statistics, [-1.2, np.sqrt(0.001875), 8 / 3])
    np.testing.assert_allclose(statistics, CAPTURE.weighted_energy_statistics(energies, weights * 1e30))
    with pytest.raises(ValueError, match="positive total"):
        CAPTURE.weighted_energy_statistics(energies, np.zeros(3))
    with pytest.raises(ValueError, match="finite walkers"):
        CAPTURE.weighted_energy_statistics(np.array([float("nan"), 0]), np.ones(2))


def test_committed_afqmc_energies_recompute_from_actual_walkers() -> None:
    """Bind the presentation's scientific curves to its saved walker observations."""
    capture_path = Path(__file__).resolve().parents[3] / "presentations/mqsf2026/captures/demo.json.gz"
    application = json.loads(gzip.decompress(capture_path.read_bytes()))["application"]
    CAPTURE.validate_native_batch(
        application["workload"]["submitted_qdmi_jobs"], application["workload"]["snapshots"], application["snapshots"]
    )
    chemistry = application["chemistry"]
    hamiltonian = np.asarray(chemistry["hamiltonian_real"])
    np.testing.assert_allclose(hamiltonian, hamiltonian.T, atol=1e-12)
    np.testing.assert_allclose(np.linalg.eigvalsh(hamiltonian)[0], chemistry["fci_energy"], atol=1e-12)
    propagation = application["propagation"]
    for curve in propagation["curves"]:
        energies = np.asarray(curve["walker_energies"])
        weights = np.asarray(curve["walker_weights"])
        assert energies.shape == weights.shape == (propagation["steps"], propagation["walkers"])
        statistics = np.asarray(list(starmap(CAPTURE.weighted_energy_statistics, zip(energies, weights, strict=True))))
        np.testing.assert_allclose(
            statistics.T, [curve["energy"], curve["stderr"], curve["effective_walkers"]], atol=1e-12
        )
        for frame in curve["frames"]:
            for walker in frame["walkers"]:
                np.testing.assert_allclose(sum(walker["occupations"]), chemistry["electrons"], atol=1e-12)
                assert walker["local_energy"] == energies[frame["step"], walker["id"]]
                assert walker["weight"] == weights[frame["step"], walker["id"]]


@pytest.mark.parametrize("failure", ["fallback", "reordered", "incomplete", "single-program"])
def test_rejects_a_batch_that_is_not_one_native_job(failure: str) -> None:
    """Do not silently retain the batching claim after fallback or result misindexing."""
    snapshots = [{"native_program_index": index, "native_job_num_programs": 3} for index in range(3)]
    CAPTURE.validate_native_batch(1, 3, snapshots)
    submitted_jobs = 1
    if failure == "fallback":
        submitted_jobs = 3
    elif failure == "reordered":
        snapshots.reverse()
    elif failure == "incomplete":
        snapshots.pop()
    else:
        snapshots[1]["native_job_num_programs"] = 1
    with pytest.raises(ValueError, match="one native QDMI job"):
        CAPTURE.validate_native_batch(submitted_jobs, 3, snapshots)
