# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""QIR sampling and coherent state extraction through public program APIs."""

from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from mqt.core.mlir import OutputFormat, QIRProgram, compile_program, sample, simulate

ROOT = Path(__file__).parent / "qir"


@pytest.mark.parametrize("output", [OutputFormat.QIR_BASE, OutputFormat.QIR_ADAPTIVE])
def test_compiled_qir_preserves_phase_and_program(output: OutputFormat) -> None:
    """Compiled profiles retain phase and support repeated, non-consuming calls."""
    circuit = QuantumCircuit(2, 2, global_phase=0.37)
    circuit.h(0)
    circuit.sdg(0)
    circuit.x(1)
    expected = Statevector(circuit).data
    circuit.measure(0, 0)
    circuit.measure(1, 1)
    program = compile_program(circuit, output=output)
    assert isinstance(program, QIRProgram)
    before = str(program)
    np.testing.assert_allclose(program.simulate(), expected, atol=1e-12)
    np.testing.assert_allclose(simulate(program), expected, atol=1e-12)
    counts = program.sample(shots=1024, seed=71)
    assert set(counts) == {"10", "11"}
    assert abs(counts["11"] - 512) < 6 * np.sqrt(1024 / 4)
    assert sample(program, shots=1024, seed=71) == counts
    assert str(program) == before


@pytest.mark.parametrize("bitcode", [False, True])
def test_base_payload_order(*, bitcode: bool) -> None:
    """Text and bitcode preserve asymmetric output and qubit ordering."""
    payload = (ROOT / "asymmetric.bc").read_bytes() if bitcode else (ROOT / "asymmetric.ll").read_text()
    assert sample(payload, shots=17) == {"011": 17}
    expected = np.zeros(8, dtype=complex)
    expected[3] = 1
    np.testing.assert_array_equal(simulate(payload), expected)


@pytest.mark.parametrize("bitcode", [False, True])
def test_adaptive_feedback_and_reset(*, bitcode: bool) -> None:
    """Feedback samples agree analytically and cannot yield a coherent state."""
    payload = (ROOT / "feedback.bc").read_bytes() if bitcode else (ROOT / "feedback.ll").read_text()
    counts = sample(payload, shots=1024, seed=71)
    assert set(counts) == {"000", "011"}
    assert abs(counts["011"] - 512) < 6 * np.sqrt(1024 / 4)
    with pytest.raises(RuntimeError, match="QIR execution failed"):
        simulate(payload)


@pytest.mark.parametrize("abort", [False, True])
def test_raw_phase_and_failure_recovery(*, abort: bool) -> None:
    """A failed worker does not prevent a later phase-correct simulation."""
    payload = (ROOT / "phase.ll").read_text().replace("base_profile", r"base_\70rofile")
    expected = np.exp(0.37j) * np.array([0, 0, 1, -1j]) / np.sqrt(2)
    if abort:
        failed = payload.replace(
            "  br label %1",
            "  %bad = call ptr @__quantum__rt__array_create_1d(i32 0, i64 1)\n  br label %1",
            1,
        )
        failed += "\ndeclare ptr @__quantum__rt__array_create_1d(i32, i64)\n"
    else:
        failed = payload.replace("ret i64 0", "ret i64 42")
    with pytest.raises(RuntimeError, match="QIR execution failed"):
        simulate(failed)
    np.testing.assert_allclose(simulate(payload), expected, atol=1e-12)


@pytest.mark.parametrize("profile", [None, "custom_profile"])
def test_profile_metadata(profile: str | None) -> None:
    """Raw LLVM requires a supported profile on its entry point."""
    attribute = "" if profile is None else f'"qir_profiles"="{profile}"'
    payload = (ROOT / "phase.ll").read_text().replace('"qir_profiles"="base_profile"', attribute)
    with pytest.raises(ValueError, match="profile"):
        sample(payload)


def test_invalid_bitcode() -> None:
    """Invalid binary input fails before job submission."""
    with pytest.raises(ValueError, match="bitcode"):
        sample(b"invalid bitcode")


def test_program_method_receiver() -> None:
    """Unbound methods reject non-program receivers at the Python boundary."""
    for name in ("sample", "simulate"):
        with pytest.raises(TypeError):
            getattr(QIRProgram, name)(object())
