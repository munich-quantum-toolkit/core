# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Native verification owns its evidence and borrows explicit QCO programs."""

from __future__ import annotations

import gc
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from mqt.core import mlir
from mqt.core.mlir import (
    QCOProgram,
    VerificationContract,
    VerificationDomain,
    VerificationMethod,
    VerificationOptions,
    VerificationPhase,
    VerificationRelation,
    VerificationVerdict,
    verify,
)


def _program(gate: str = "x", phase: float = 0.0) -> QCOProgram:
    """Construct a one-qubit circuit with an explicit global phase.

    Returns:
        An owned QCO program.
    """
    return QCOProgram.from_mlir_str(f"""
module {{
  func.func @main() attributes {{mqt.entry_point}} {{
    %phase = arith.constant {phase:.17e} : f64
    qco.gphase(%phase)
    %q = qco.static 0 : !qco.qubit
    %r = qco.{gate} %q : !qco.qubit
    qco.sink %r : !qco.qubit
    return
  }}
}}
""")


def _bound_program() -> QCOProgram:
    """Construct scalar-bound branching and loop-carried quantum execution.

    Returns:
        An owned QCO program.
    """
    return QCOProgram.from_mlir_str("""
module {
  func.func @main(%flag: i1, %count: index, %angle: f64) attributes {mqt.entry_point} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %q = qco.static 0 : !qco.qubit
    %r = qco.if %flag args(%left = %q) -> (!qco.qubit) {
      %rotated = qco.rz(%angle) %left : !qco.qubit
      qco.yield %rotated : !qco.qubit
    } else args(%right = %q) {
      qco.yield %right : !qco.qubit
    }
    %s = scf.for %i = %zero to %count step %one iter_args(%wire = %r) -> (!qco.qubit) {
      %flipped = qco.x %wire : !qco.qubit
      scf.yield %flipped : !qco.qubit
    }
    qco.sink %s : !qco.qubit
    return
  }
}
""")


def test_default_preserves_phase_and_programs() -> None:
    """Default checking preserves phase and leaves both owners usable."""
    left = _program()
    right = _program(phase=0.4)
    before = (str(left), str(right))
    result = verify(left, right)
    assert result.verdict is VerificationVerdict.NOT_EQUIVALENT
    assert result.phase is VerificationPhase.PRESERVE
    assert result.relation is VerificationRelation.EXACT
    assert (str(left), str(right)) == before

    options = VerificationOptions()
    options.relation = VerificationRelation.PROCESS_DISTANCE
    options.phase = VerificationPhase.IGNORE
    options.tolerance = 0.01
    options.method = VerificationMethod.CONSTRUCTION
    result = verify(left, right, options=options)
    assert result.verdict is VerificationVerdict.EQUIVALENT
    assert result.distance == pytest.approx(0, abs=1e-7)
    assert result.tolerance == pytest.approx(0.01)


def test_bindings_limit_the_claim_and_remain_recorded() -> None:
    """SCF uses concrete per-program bindings without mutating the source."""
    program = _bound_program()
    source = str(program)
    bindings = {0: True, 1: 1, 2: 0.3}
    result = verify(program, program, left_bindings=bindings, right_bindings=bindings)
    assert result.verdict is VerificationVerdict.EQUIVALENT
    assert result.left_bindings == result.right_bindings
    assert len(result.left_bindings) == 3
    assert any("arg2" in binding for binding in result.left_bindings)
    assert str(program) == source
    changed = verify(program, program, left_bindings=bindings, right_bindings={**bindings, 1: 2})
    assert changed.verdict is VerificationVerdict.NOT_EQUIVALENT
    changed_branch = verify(program, program, left_bindings=bindings, right_bindings={**bindings, 0: False})
    assert changed_branch.verdict is VerificationVerdict.NOT_EQUIVALENT
    assert verify(program, program).verdict is VerificationVerdict.UNSUPPORTED


@pytest.mark.parametrize("bindings", [{3: 0}, {-1: 0}, {True: 1}, {0: 2}, {1: 1 << 80}, {2: float("nan")}])
def test_rejects_invalid_scalar_values_and_indices(bindings: dict[int, bool | int | float]) -> None:
    """Invalid scalar data never gets truncated into a different binding."""
    program = _bound_program()
    with pytest.raises(ValueError, match="verification binding"):
        verify(program, program, left_bindings=bindings)


@pytest.mark.parametrize("bindings", [{0: 1.2}, {2: True}, {2: "0.3"}])
def test_rejects_scalar_type_mismatches(bindings: dict[int, object]) -> None:
    """Argument types determine scalar conversion, including Boolean values."""
    program = _bound_program()
    with pytest.raises(TypeError):
        vars(mlir)["verify"](program, program, left_bindings=bindings)


def test_results_outlive_owners_and_have_no_truth_value() -> None:
    """Result records own all evidence and force explicit verdict handling."""
    result = verify(_program(), _program())
    gc.collect()
    assert result.verdict is VerificationVerdict.EQUIVALENT
    assert result.checkers
    assert result.num_inputs == 1
    assert result.peak_dd_bytes > 0
    assert result.contract.domain is VerificationDomain.CIRCUIT_INPUTS
    with pytest.raises(TypeError, match="verdict"):
        bool(result)
    attribute = "verdict"
    with pytest.raises(AttributeError):
        setattr(result, attribute, VerificationVerdict.NOT_EQUIVALENT)


def test_simulation_and_resource_limits_are_inconclusive() -> None:
    """Passing samples and exhausted DD storage never become proofs."""
    program = _program()
    options = VerificationOptions()
    options.method = VerificationMethod.SIMULATION
    options.max_simulation_samples = 2
    options.seed = 17
    result = verify(program, program, options=options)
    assert result.verdict is VerificationVerdict.INCONCLUSIVE
    assert result.checkers[0].samples == 2
    options.method = VerificationMethod.CONSTRUCTION
    options.limits.max_dd_bytes = 1
    assert verify(program, program, options=options).verdict is VerificationVerdict.INCONCLUSIVE
    with pytest.raises(ValueError, match="timeout"):
        options.limits.timeout_ms = -1


def test_explicit_contract_is_copied_into_evidence() -> None:
    """The returned quantum domain is independent of subsequent option edits."""
    contract = VerificationContract()
    contract.domain = VerificationDomain.INITIALIZED_PROGRAM
    result = verify(_program(), _program(), contract=contract)
    assert result.verdict is VerificationVerdict.EQUIVALENT
    contract.domain = VerificationDomain.CIRCUIT_INPUTS
    assert result.contract.domain is VerificationDomain.INITIALIZED_PROGRAM


def test_requires_unconsumed_qco_programs() -> None:
    """Verification never implicitly compiles or consumes its inputs."""
    program = _program()
    with pytest.raises(TypeError):
        vars(mlir)["verify"]("OPENQASM 3.0; qubit q;", program)
    program.to_qc()
    with pytest.raises(RuntimeError, match="consumed"):
        verify(program, _program())


def test_concurrent_borrowed_checks_share_no_mutable_options() -> None:
    """Two calls can borrow the same immutable program while the GIL is released."""
    program = _program()
    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(verify, program, program) for _ in range(2)]
        results = [future.result(timeout=20) for future in futures]
    assert all(result.verdict is VerificationVerdict.EQUIVALENT for result in results)


def test_concurrent_unsupported_checks_keep_their_diagnostics() -> None:
    """Unsupported traces must report the failure from their own caller."""
    program = QCOProgram.from_mlir_str("""
module {
  func.func @main(%n: index, %reset: i1, %theta: f64) attributes {mqt.entry_point} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %q = qco.static 0 : !qco.qubit
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%a = %q) -> !qco.qubit {
      %x = qco.x %a : !qco.qubit
      scf.yield %x : !qco.qubit
    }
    %out = qco.if %reset args(%a = %r) -> !qco.qubit {
      %x = qco.reset %a : !qco.qubit
      qco.yield %x : !qco.qubit
    } else args(%a = %r) {
      %x = qco.rx(%theta) %a : !qco.qubit
      qco.yield %x : !qco.qubit
    }
    qco.sink %out : !qco.qubit
    return
  }
}
""")
    with ThreadPoolExecutor(max_workers=2) as workers:
        for _ in range(3):
            reset = workers.submit(verify, program, program, left_bindings={0: 20_000, 1: True})
            # Stagger bounded traces to exercise overlapping context handlers.
            time.sleep(0.002)
            unbound = workers.submit(verify, program, program, left_bindings={0: 100_000, 1: False})
            reset_result = reset.result(timeout=20)
            unbound_result = unbound.result(timeout=20)
            assert reset_result.verdict is VerificationVerdict.UNSUPPORTED
            assert unbound_result.verdict is VerificationVerdict.UNSUPPORTED
            assert "resets are not supported" in reset_result.reason
            assert "no concrete QCO DD binding" in unbound_result.reason
