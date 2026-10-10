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
import hashlib
import importlib.util
import json
import operator
from itertools import combinations, pairwise, starmap
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


def test_six_orbital_expansion_matches_independent_minor_determinants() -> None:
    """The active-space shortcut preserves every complex Slater amplitude."""
    rng = np.random.default_rng(31)
    walker, _ = np.linalg.qr(rng.normal(size=(6, 2)) + 1j * rng.normal(size=(6, 2)))
    expected = [np.linalg.det(walker[list(occupied)]) for occupied in combinations(range(6), 2)]
    np.testing.assert_allclose(CAPTURE.determinant_amplitudes(walker), expected, atol=1e-14)
    np.testing.assert_allclose(np.vdot(expected, expected), 1, atol=1e-14)


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
    assert CAPTURE.__file__ is not None
    # Match the captured LF source even when Git checks it out as CRLF on Windows.
    assert (
        application["provenance"]["capture_script_sha256"]
        == hashlib.sha256(Path(CAPTURE.__file__).read_text(encoding="utf-8").encode()).hexdigest()
    )
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


def test_committed_afqmc_parallel_records_share_one_observed_clock() -> None:
    """Recorded tasks and frames must support the pool and time-evolution animation."""
    capture_path = Path(__file__).resolve().parents[3] / "presentations/mqsf2026/captures/demo.json.gz"
    application = json.loads(gzip.decompress(capture_path.read_bytes()))["application"]
    propagation = application["propagation"]
    assert len(propagation["worker_pids"]) == propagation["processes"] == 4
    tasks = {(task["label"], task["walker_id"]): task for task in propagation["tasks"]}
    assert len(tasks) == 2 * propagation["walkers"]
    assert len(propagation["events"]) == len(tasks)
    assert {(event["label"], event["walker_id"]) for event in propagation["events"]} == tasks.keys()
    assert len(set(propagation["walker_seeds"])) == propagation["walkers"]
    for event in propagation["events"]:
        task = tasks[event["label"], event["walker_id"]]
        assert event["worker_pid"] == task["worker_pid"]
        assert task["seed"] == propagation["walker_seeds"][task["walker_id"]]
        assert 0 <= task["started_ms"] <= task["finished_ms"] <= event["time_ms"] <= propagation["duration_ms"]
    for pid in propagation["worker_pids"]:
        assigned = sorted(
            (task for task in tasks.values() if task["worker_pid"] == pid), key=operator.itemgetter("started_ms")
        )
        assert all(first["finished_ms"] <= second["started_ms"] for first, second in pairwise(assigned))
    for label, curve in zip(("quantum", "classical"), propagation["curves"], strict=True):
        for frame in curve["frames"]:
            assert frame["tau"] == frame["step"] * propagation["dtau"]
            for walker in frame["walkers"]:
                task = tasks[label, walker["id"]]
                assert task["worker_pid"] == walker["worker_pid"]
                assert task["started_ms"] <= walker["completed_ms"] <= task["finished_ms"]


def test_committed_afqmc_batch_trace_points_into_captured_client_source() -> None:
    """Source highlights and timestamps refer to actual adapter calls in the saved batch."""
    capture_path = Path(__file__).resolve().parents[3] / "presentations/mqsf2026/captures/demo.json.gz"
    application = json.loads(gzip.decompress(capture_path.read_bytes()))["application"]
    execution = application["execution"]
    source_lines = execution["client_source"].splitlines()
    events = execution["events"]
    assert len(events) == application["workload"]["snapshots"] + 2
    assert [event["time_ms"] for event in events] == sorted(event["time_ms"] for event in events)
    for event in events:
        assert 0 <= event["time_ms"] <= event["time_ms"] + event["duration_ms"] <= execution["duration_ms"]
        line = source_lines[event["source_line"] - 1]
        if "program_index" in event:
            assert "self._shots_or_counts(job, program_index)" in line
        elif "programs" in event:
            assert "self._device.qdmi_device.try_submit_job(" in line
            assert event["programs"] == application["workload"]["snapshots"]
        else:
            assert "self._batch.complete()" in line
    assert [event["program_index"] for event in events if "program_index" in event] == list(
        range(application["workload"]["snapshots"])
    )
    assert len(application["batch_source"].splitlines()) == 3


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
