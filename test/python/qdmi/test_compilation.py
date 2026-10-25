# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Device-directed compilation and submission contracts."""

from __future__ import annotations

import gc
import subprocess
import sys
from fractions import Fraction
from typing import TYPE_CHECKING, assert_type, cast

import pytest

from mqt.core.bench import qpe, repeat_until_success, shor
from mqt.core.mlir import (
    CompiledProgram,
    CompilerTarget,
    OpenQASMProgram,
    OutputFormat,
    QCProgram,
    QIRProfile,
    compile_program,
    sample,
    submit_program,
)
from mqt.core.qdmi import CustomProperty, Job, ProgramFormat, open_device

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

BELL = 'OPENQASM 3.0; include "stdgates.inc"; qubit[2] q; bit[2] c; h q[0]; cx q[0],q[1]; c = measure q;'


@pytest.mark.parametrize("program_format", [ProgramFormat.QASM3, ProgramFormat.QIR_ADAPTIVE_MODULE])
def test_runtime_controlled_fractional_power(program_format: ProgramFormat) -> None:
    """Runtime fixed-gate powers preserve relative phase under control."""
    source = """OPENQASM 3.1;
      include "stdgates.inc";
      qubit[2] q;
      x q[1];
      bit choose = measure q[1];
      float turns = 0.25 + float(choose) / 4.0;
      h q[0];
      ctrl @ pow(turns) @ z q[0], q[1];
      ctrl @ pow(turns) @ z q[0], q[1];
      h q[0];
      output bit[2] result;
      result = measure q;
    """
    job = submit_program(source, target="mqt.ddsim.default", program_format=program_format, num_shots=16, custom1=17)
    assert job.wait()
    assert job.get_counts() == {"11": 16}


@pytest.mark.parametrize("program_format", [ProgramFormat.QASM3, ProgramFormat.QIR_ADAPTIVE_MODULE])
def test_computed_classical_outputs(program_format: ProgramFormat) -> None:
    """Record mixed measured and computed bits in declared order."""
    source = """OPENQASM 3.1;
      include "stdgates.inc";
      qubit q;
      x q;
      output bit measured;
      measured = measure q;
      output bit[4] computed;
      computed = "0000";
      for int i in [0:2] { computed[i] = bool(measured) != bool(i % 2); }
      computed[0] = false;
    """
    capture = program_format == ProgramFormat.QIR_ADAPTIVE_MODULE
    job = submit_program(
        source, target="mqt.ddsim.default", program_format=program_format, num_shots=4, custom1=17, custom2=capture
    )
    assert job.wait()
    assert job.get_counts() == {"01001": 4}
    assert job.get_shots() == ["01001"] * 4
    if capture:
        output = job.get_custom_result(CustomProperty.CUSTOM1, str)
        assert isinstance(output, str)
        for index, value in enumerate(("false", "false", "true", "false")):
            assert output.count(f"OUTPUT\tBOOL\t{value}\tcomputed_{index}\n") == 4


@pytest.mark.parametrize(
    ("element_type", "low", "high", "sentinel", "comparison"),
    [("i1", "0", "1", "1", "arith.cmpi eq"), ("f64", "0.25", "0.75", "0.5", "arith.cmpf oeq")],
)
def test_classical_array_copy_snapshot(element_type: str, low: str, high: str, sentinel: str, comparison: str) -> None:
    """Copies preserve subview offsets and values across later source writes."""
    program = QCProgram.from_mlir_str(f"""module {{
      func.func @main() -> i1 attributes {{mqt.entry_point}} {{
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %two = arith.constant 2 : index
        %low = arith.constant {low} : {element_type}
        %high = arith.constant {high} : {element_type}
        %sentinel = arith.constant {sentinel} : {element_type}
        %source = memref.alloca() : memref<3x{element_type}>
        %target = memref.alloca() : memref<3x{element_type}>
        memref.store %sentinel, %source[%zero] : memref<3x{element_type}>
        memref.store %low, %source[%one] : memref<3x{element_type}>
        memref.store %high, %source[%two] : memref<3x{element_type}>
        memref.store %sentinel, %target[%zero] : memref<3x{element_type}>
        %q = qc.alloc : !qc.qubit
        qc.x %q : !qc.qubit
        %bit = qc.measure %q : !qc.qubit -> i1
        %index = arith.index_castui %bit : i1 to index
        %from = memref.subview %source[%index][2][1]
          : memref<3x{element_type}> to memref<2x{element_type}, strided<[1], offset: ?>>
        %to = memref.subview %target[1][2][1]
          : memref<3x{element_type}> to memref<2x{element_type}, strided<[1], offset: 1>>
        memref.copy %from, %to
          : memref<2x{element_type}, strided<[1], offset: ?>>
          to memref<2x{element_type}, strided<[1], offset: 1>>
        memref.store %low, %source[%two] : memref<3x{element_type}>
        %prefix = memref.load %target[%zero] : memref<3x{element_type}>
        %first = memref.load %target[%one] : memref<3x{element_type}>
        %second = memref.load %target[%two] : memref<3x{element_type}>
        %prefix_ok = {comparison}, %prefix, %sentinel : {element_type}
        %first_ok = {comparison}, %first, %low : {element_type}
        %second_ok = {comparison}, %second, %high : {element_type}
        %copied_ok = arith.andi %first_ok, %second_ok : i1
        %ok = arith.andi %prefix_ok, %copied_ok : i1
        qc.dealloc %q : !qc.qubit
        return %ok : i1
      }}
    }}""")
    assert sample(program, shots=1, seed=1) == {"1": 1}
    qir = program.to_qir(QIRProfile.ADAPTIVE)
    job = open_device("mqt.ddsim.default").submit_job(qir.to_bitcode(), ProgramFormat.QIR_ADAPTIVE_MODULE, num_shots=1)
    assert job.wait()
    assert job.get_counts() == {"1": 1}


def test_boolean_scalar_outputs() -> None:
    """DD and QIR preserve Boolean snapshots and repeated mixed outputs."""
    program = QCProgram.from_mlir_str("""module {
      func.func @main() -> (i1, !cbit.reg<1>, i1, i1, !cbit.reg<1>) attributes {mqt.entry_point} {
        %zero = arith.constant 0 : index
        %true = arith.constant true
        %q = qc.alloc : !qc.qubit
        %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
        %first = qc.measure %q : !qc.qubit -> i1
        cbit.store %first, %reg[%zero] : !cbit.reg<1>
        %snapshot = arith.xori %first, %true : i1
        qc.x %q : !qc.qubit
        %later = qc.measure %q : !qc.qubit -> i1
        cbit.store %later, %reg[%zero] : !cbit.reg<1>
        qc.dealloc %q : !qc.qubit
        return %snapshot, %reg, %first, %snapshot, %reg : i1, !cbit.reg<1>, i1, i1, !cbit.reg<1>
      }
    }""")
    assert sample(program, shots=1, seed=1) == {"11011": 1}
    qir = program.to_qir(QIRProfile.ADAPTIVE)
    job = open_device("mqt.ddsim.default").submit_job(qir.to_bitcode(), ProgramFormat.QIR_ADAPTIVE_MODULE, num_shots=1)
    assert job.wait()
    assert job.get_counts() == {"11011": 1}


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("scalar_output", [False, True])
def test_adaptive_measurement_snapshots(*, overwrite: bool, scalar_output: bool) -> None:
    """Preserve a measurement value across runtime-indexed stores and repeated outputs."""
    intervening = (
        """
      qc.x %q : !qc.qubit
      %later = qc.measure %q : !qc.qubit -> i1
      cbit.store %later, %out[%zero] : !cbit.reg<2>
      %position = arith.index_castui %later : i1 to index
    """
        if overwrite
        else ""
    )
    result_type = "(!cbit.reg<2>, i1)" if scalar_output else "!cbit.reg<2>"
    outputs = "%out, %first : !cbit.reg<2>, i1" if scalar_output else "%out : !cbit.reg<2>"
    source = f"""module {{
      func.func @main() -> {result_type} attributes {{mqt.entry_point}} {{
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %q = qc.alloc : !qc.qubit
        %out = cbit.alloc(#cbit.init<zero>) : !cbit.reg<2>
        qc.x %q : !qc.qubit
        %first = qc.measure %q : !qc.qubit -> i1
        {intervening}
        cbit.store %first, %out[{"%position" if overwrite else "%zero"}] : !cbit.reg<2>
        cbit.store %first, %out[%one] : !cbit.reg<2>
        qc.dealloc %q : !qc.qubit
        return {outputs}
      }}
    }}"""
    program = QCProgram.from_mlir_str(source).to_qir(QIRProfile.ADAPTIVE)
    job = open_device("mqt.ddsim.default").submit_job(
        program.to_bitcode(), ProgramFormat.QIR_ADAPTIVE_MODULE, num_shots=1
    )
    assert job.wait()
    assert job.get_counts() == {"111" if scalar_output else "11": 1}


@pytest.mark.parametrize("width", [50, 100, 150])
@pytest.mark.parametrize("program_format", [ProgramFormat.QIR_BASE_MODULE, ProgramFormat.QIR_ADAPTIVE_MODULE])
def test_entry_point_local_bits(width: int, program_format: ProgramFormat) -> None:
    """Promote local bits and preserve the measured output's width and order."""
    bits = [int(index % 3 == 1) for index in range(width)]
    assignments = "\n".join(f"local[{index}] = {bit};" for index, bit in enumerate(bits))
    controls = "\n".join(f"if (local[{index}]) {{ x q[{index}]; }}" for index in range(width))
    source = f"""OPENQASM 3.0;
      include "stdgates.inc";
      qubit[{width}] q;
      bit[{width}] local;
      output bit[{width}] result;
      {assignments}
      {controls}
      result = measure q;
    """
    job = submit_program(source, target="mqt.ddsim.default", program_format=program_format, num_shots=1)
    assert job.wait()
    assert job.get_counts() == {"".join(map(str, reversed(bits))): 1}


@pytest.mark.parametrize("program_format", [ProgramFormat.QASM3, ProgramFormat.QIR_ADAPTIVE_MODULE])
def test_runtime_indices_and_integer_powers(program_format: ProgramFormat) -> None:
    """Execute valid runtime arithmetic and indexing without assertion lowering."""
    source = """OPENQASM 3.1;
      include "stdgates.inc";
      qubit q;
      x q;
      bit choose = measure q;
      reset q;
      int exponent = 2;
      if (choose) { exponent = 3; }
      int value = 3 ** exponent;
      bit[3] bits = "000";
      int index = -1;
      bits[index] = true;
      int turns = 3;
      if (bits[index] && value == 27) { pow(turns) @ x q; }
      output bit result;
      result = measure q;
    """
    job = submit_program(source, target="mqt.ddsim.default", program_format=program_format, num_shots=16, custom1=17)
    job.wait()
    assert job.get_counts() == {"1": 16}


def test_indexed_openqasm_device_execution() -> None:
    """Specialize mapped OpenQASM indices and preserve the sampled result."""
    source = QCProgram.from_mlir_str("""module {
      func.func @main() -> !cbit.reg<2> attributes {mqt.entry_point} {
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %two = arith.constant 2 : index
        %four = arith.constant 4 : index
        %angles = arith.constant dense<[3.141592653589793, 0.0, 0.0, 3.141592653589793]> : tensor<4xf64>
        %qubits = memref.alloc() : memref<2x!qc.qubit>
        %bits = cbit.alloc(#cbit.init<zero>) : !cbit.reg<2>
        scf.for %i = %zero to %four step %one {
          %slot = arith.remui %i, %two : index
          %q = memref.load %qubits[%slot] : memref<2x!qc.qubit>
          %angle = tensor.extract %angles[%i] : tensor<4xf64>
          qc.rx(%angle) %q : !qc.qubit
        }
        scf.for %i = %zero to %two step %one {
          %q = memref.load %qubits[%i] : memref<2x!qc.qubit>
          %bit = qc.measure %q : !qc.qubit -> i1
          cbit.store %bit, %bits[%i] : !cbit.reg<2>
        }
        memref.dealloc %qubits : memref<2x!qc.qubit>
        return %bits : !cbit.reg<2>
      }
    }""")
    device = open_device("mqt.ddsim.default")
    compiled = compile_program(source, target=device, output=OutputFormat.OPENQASM3)
    assert "for int " not in compiled.source
    assert "switch (" not in compiled.source
    job = device.submit_job(compiled.source, ProgramFormat.QASM3, num_shots=32, custom1=17)
    job.wait()
    assert job.get_counts() == {"11": 32}


@pytest.mark.parametrize("program_format", [ProgramFormat.QASM3, ProgramFormat.QIR_ADAPTIVE_MODULE])
@pytest.mark.parametrize(("number", "factors"), [(15, (3, 5)), (21, (3, 7))])
def test_shor_device_execution(number: int, factors: tuple[int, int], program_format: ProgramFormat) -> None:
    """Factor small numbers through modular arithmetic and the bundled DDSIM device."""
    device = open_device("mqt.ddsim.default")

    def run(benchmark: shor.Shor) -> dict[str, int]:
        job = submit_program(
            benchmark.generate(), target=device, program_format=program_format, num_shots=64, custom1=17
        )
        job.wait()
        counts = job.get_counts()
        assert sum(counts.values()) == 64
        assert all(len(bits) == benchmark.output.width for bits in counts)
        assert benchmark.evaluate(counts).factors == factors
        return counts

    result = shor.factor(number, run)
    assert result.status == shor.FactorStatus.SUCCESS
    assert result.factors == factors
    assert result.attempts == 1


@pytest.mark.parametrize("program_format", [ProgramFormat.QASM3, ProgramFormat.QIR_ADAPTIVE_MODULE])
@pytest.mark.parametrize("method", [qpe.Method.STANDARD, qpe.Method.ITERATIVE])
def test_qpe_device_execution(method: qpe.Method, program_format: ProgramFormat) -> None:
    """Recover QPE's exact phase through mapped OpenQASM and QIR execution."""
    benchmark = qpe.QPE(qpe.Options(precision=8, phase=Fraction(3, 8), method=method))
    device = open_device("mqt.ddsim.default")
    job = submit_program(benchmark.generate(), target=device, program_format=program_format, num_shots=32, custom1=17)
    job.wait()
    counts = job.get_counts()
    assert counts == {"01100000": 32}
    assert benchmark.evaluate(counts).total_variation_distance == pytest.approx(0)


@pytest.mark.parametrize(
    "program_format",
    [
        ProgramFormat.QASM3,
        ProgramFormat.QIR_BASE_STRING,
        ProgramFormat.QIR_BASE_MODULE,
        ProgramFormat.QIR_ADAPTIVE_STRING,
        ProgramFormat.QIR_ADAPTIVE_MODULE,
    ],
)
@pytest.mark.parametrize("swapped", [False, True])
def test_result_bit_order(program_format: ProgramFormat, *, swapped: bool) -> None:
    """Shots and counts preserve classical-bit order across payload formats."""
    source = (
        'OPENQASM 3.0; include "stdgates.inc"; qubit[3] q; bit[2] a; bit b; '
        f"x q[0]; a[{int(swapped)}] = measure q[0]; "
        f"a[{int(not swapped)}] = measure q[1]; b = measure q[2];"
    )
    job = submit_program(source, target="mqt.ddsim.default", num_shots=4, program_format=program_format)
    job.wait()
    expected = "010" if swapped else "001"
    assert job.get_shots() == [expected] * 4
    assert job.get_counts() == {expected: 4}


@pytest.mark.parametrize("data_qubits", [1, 4])
def test_repeat_until_success_device_execution(data_qubits: int) -> None:
    """Preserve phase-sensitive RUS results through target placement and QIR."""
    benchmark = repeat_until_success.RepeatUntilSuccess(repeat_until_success.Options(data_qubits=data_qubits))
    device = open_device("mqt.ddsim.default")
    job = submit_program(benchmark.generate(), target=device, num_shots=4096, custom1=17)
    job.wait()
    assert benchmark.evaluate(job.get_counts()).total_variation_distance < 0.02


@pytest.mark.parametrize("form", ["artifact", "source", "device_id"])
def test_submission_forms(form: str) -> None:
    """All forms default to 1024 shots and preserve seeded samples."""
    device = open_device("mqt.ddsim.default")
    if form == "artifact":
        program = compile_program(BELL, target=device)
        assert_type(program, CompiledProgram)
        assert isinstance(program, CompiledProgram)
        assert program.program_format == ProgramFormat.QIR_ADAPTIVE_MODULE
        job = submit_program(program, target=device, custom1=7)
    elif form == "source":
        job = submit_program(BELL, target=device, custom1=7)
    else:
        job = submit_program(BELL, target="mqt.ddsim.default", custom1=7)
    assert isinstance(job, Job)
    del device
    gc.collect()
    job.wait()
    shots = job.get_shots()
    assert len(shots) == 1024
    assert set(shots) <= {"00", "11"}
    again = submit_program(BELL, target="mqt.ddsim.default", custom1=7)
    again.wait()
    assert again.get_shots() == shots


def test_compiled_program_reopens_device() -> None:
    """Compiled programs retain their native payload after the device is reopened."""
    device = open_device("mqt.ddsim.default")
    by_device = compile_program(BELL, target=device)
    by_id = compile_program(BELL, target="mqt.ddsim.default")
    assert by_id.payload == by_device.payload
    assert by_id.program_format == ProgramFormat.QIR_ADAPTIVE_MODULE
    assert isinstance(by_id.payload, bytes)
    with pytest.raises(AttributeError):
        by_id.payload = b"changed"  # ty: ignore[invalid-assignment]
    del device
    gc.collect()
    job = submit_program(by_id, target=open_device("mqt.ddsim.default"), num_shots=0)
    job.wait()
    assert job.get_dense_statevector() == pytest.approx([2**-0.5, 0, 0, 2**-0.5])


def test_rejects_invalid_submission_options() -> None:
    """Reject invalid counts and formats before consuming a typed program."""
    device = open_device("mqt.ddsim.default")
    source = compile_program(BELL, output=OutputFormat.QCO)
    with pytest.raises(ValueError, match="nonnegative"):
        submit_program(source, target=device, num_shots=-1)
    with pytest.raises(ValueError, match="cannot emit"):
        submit_program(source, target=device, program_format=ProgramFormat.QASM2)
    assert source.is_valid
    compiled = compile_program(source, target=device, copy=True)
    assert source.is_valid
    with pytest.raises(ValueError, match="conflicts"):
        submit_program(compiled, target=device, program_format=ProgramFormat.QASM3)


def test_explicit_target_preserves_input_format() -> None:
    """Hardware snapshots produce directly executable OpenQASM by default."""
    device = open_device("mqt.ddsim.default")
    target = CompilerTarget.from_device(device)
    output = compile_program(BELL, target=target)
    assert isinstance(output, OpenQASMProgram)
    raw = device.submit_job(output.source, ProgramFormat.QASM3, num_shots=0)
    raw.wait()
    assert raw.get_dense_statevector() == pytest.approx([2**-0.5, 0, 0, 2**-0.5])


def test_qdmi_does_not_import_compiler() -> None:
    """Raw QDMI submission does not load the compiler module."""
    script = """
import sys
from mqt.core.qdmi import ProgramFormat
from mqt.core.qdmi import open_device

assert "mqt.core.mlir" not in sys.modules
device = open_device("mqt.ddsim.default")
source = "OPENQASM 3.0; qubit q; bit c = measure q;"
raw = device.submit_job(source, ProgramFormat.QASM3, num_shots=1)
assert raw.wait()
assert "mqt.core.mlir" not in sys.modules
from mqt.core.mlir import submit_program
job = submit_program(source, target=device, num_shots=2)
assert job.wait()
assert job.get_counts() == {"0": 2}
"""
    subprocess.run([sys.executable, "-c", script], check=True)  # ruff: ignore[subprocess-without-shell-equals-true]


@pytest.mark.parametrize("targeted", [False, True])
def test_source_path_is_read_once(tmp_path: Path, *, targeted: bool) -> None:
    """Source submission imports once; artifact resubmission needs no source."""
    path = tmp_path / "bell.qasm"
    path.write_text(BELL)

    class Source:
        calls = 0

        def __fspath__(self) -> str:
            self.calls += 1
            assert self.calls == 1
            return str(path)

    source = Source()
    device = open_device("mqt.ddsim.default")
    job = submit_program(source, target=device, num_shots=2)
    job.wait()
    assert source.calls == 1
    compiled_source = Source()
    compiled = compile_program(compiled_source, target=device if targeted else None)
    assert compiled_source.calls == 1
    path.unlink()
    job = submit_program(compiled, target=device, num_shots=2)
    job.wait()
    assert len(job.get_shots()) == 2


@pytest.mark.parametrize("program_format", [ProgramFormat.QASM3, ProgramFormat.QIR_ADAPTIVE_MODULE])
def test_payload_forward_branching(program_format: ProgramFormat) -> None:
    """Guaranteed forward branching permits feedback and Base rejects it."""
    source = BELL.replace("c = measure q;", "c[0] = measure q[0]; if (c[0]) { x q[1]; } c[1] = measure q[1];")
    device = open_device("mqt.ddsim.default")
    job = submit_program(source, target=device, program_format=program_format, num_shots=32)
    job.wait()
    assert set(job.get_counts()) <= {"00", "01"}
    with pytest.raises(RuntimeError, match="Not supported"):
        job.get_dense_statevector()
    assert len(job.get_shots()) == 32
    with pytest.raises(ValueError, match="Compilation failed"):
        submit_program(source, target=device, program_format=ProgramFormat.QIR_BASE_MODULE)


@pytest.mark.parametrize("program_format", [ProgramFormat.QASM3, ProgramFormat.QIR_BASE_MODULE])
def test_compiled_terminal_sampling_retains_state(program_format: ProgramFormat) -> None:
    """Eligible compiled payloads expose all lazy results without changing shots."""
    job = submit_program(BELL, target="mqt.ddsim.default", num_shots=64, program_format=program_format, custom1=7)
    job.wait()
    shots = job.get_shots()
    for _ in range(2):
        assert job.get_dense_statevector() == pytest.approx([2**-0.5, 0, 0, 2**-0.5])
        assert job.get_sparse_statevector() == pytest.approx({"00": 2**-0.5, "11": 2**-0.5})
        assert job.get_dense_probabilities() == pytest.approx([0.5, 0, 0, 0.5])
        assert job.get_sparse_probabilities() == pytest.approx({"00": 0.5, "11": 0.5})
    assert job.get_shots() == shots


def test_adaptive_payload_supports_optional_computations() -> None:
    """DDSIM accepts optional computations in the preferred Adaptive payload."""
    source = (
        'OPENQASM 3.0; include "stdgates.inc"; qubit q; bit c; h q; c = measure q; '
        "if (true) { int[32] k = int[32](c); if (k + 1 == 2) { x q; } } c = measure q;"
    )
    device = open_device("mqt.ddsim.default")
    compiled = compile_program(source, target=device)
    assert compiled.program_format == ProgramFormat.QIR_ADAPTIVE_MODULE
    assert "qir.int-computations" in {c.capability_id for c in compiled.payload_specification.capabilities}
    job = submit_program(compiled, target=device, num_shots=16)
    job.wait()
    assert job.get_counts() == {"0": 16}


@pytest.mark.parametrize(
    ("body", "counts"),
    [
        ("for int i in [0:2] { x q; } c = measure q;", {"1": 8}),
        ("x q; c = measure q; while (c) { x q; c = measure q; }", {"0": 8}),
    ],
)
def test_default_payload_supports_loops(body: str, counts: dict[str, int]) -> None:
    """The preferred Adaptive payload retains counted and measurement-controlled loops."""
    source = 'OPENQASM 3.0; include "stdgates.inc"; qubit q; bit c; ' + body
    job = submit_program(source, target="mqt.ddsim.default", num_shots=8)
    job.wait()
    assert job.get_counts() == counts


@pytest.mark.parametrize(("width", "staged"), [(32, False), (64, True)])
def test_qir_artifact_rejects_unsupported_entry_result(
    width: int, capfd: pytest.CaptureFixture[str], *, staged: bool
) -> None:
    """Keep scalar program outputs distinct from the QDMI QIR status return."""
    source = f"OPENQASM 3.0; qubit q; bit c = measure q; int[{width}] k = int[{width}](c);"
    program = compile_program(source, output=OutputFormat.QCO) if staged else source
    with pytest.raises(ValueError, match="Compilation failed"):
        compile_program(program, target="mqt.ddsim.default")
    assert "scalar program outputs" in capfd.readouterr().err


@pytest.mark.parametrize("num_shots", [0, 8])
def test_empty_source_has_a_successful_entry_point(num_shots: int) -> None:
    """An output-free source receives a success status without inventing outputs."""
    job = submit_program("OPENQASM 3.0;", target="mqt.ddsim.default", num_shots=num_shots)
    job.wait()
    assert job.check() == Job.Status.DONE
    assert job.get_dense_statevector() == [1 + 0j]


def test_openqasm31_switch() -> None:
    """The OpenQASM payload supports measurement-controlled switch statements."""
    source = """OPENQASM 3.1;
include "stdgates.inc";
qubit q;
bit c;
h q;
c = measure q;
switch (int(c)) { case 1 { x q; } default {} }
c = measure q;
"""
    job = submit_program(source, target="mqt.ddsim.default", program_format=ProgramFormat.QASM3, num_shots=16)
    job.wait()
    assert job.get_counts() == {"0": 16}


@pytest.mark.parametrize(
    "program_format",
    [
        ProgramFormat.QIR_BASE_STRING,
        ProgramFormat.QIR_BASE_MODULE,
        ProgramFormat.QIR_ADAPTIVE_STRING,
        ProgramFormat.QIR_ADAPTIVE_MODULE,
    ],
)
@pytest.mark.parametrize("capture", [False, True])
def test_binary_qir_results(program_format: ProgramFormat, *, capture: bool) -> None:
    """Binary QIR sampling exposes counts, shots, and eligible retained states."""
    source = 'OPENQASM 3.0; include "stdgates.inc"; qubit[2] q; h q[0]; bit[2] c = measure q;'
    job = submit_program(
        source, target="mqt.ddsim.default", program_format=program_format, num_shots=8, custom1=7, custom2=capture
    )
    assert job.wait()
    shots = job.get_shots()
    assert len(shots) == 8
    assert job.get_counts() == {bits: shots.count(bits) for bits in set(shots)}
    output = job.get_custom_result(CustomProperty.CUSTOM1, str)
    if capture:
        assert isinstance(output, str)
        recorded_shots = []
        for frame in output.split("START\n")[1:]:
            bits = [
                line.split("\t")[2]
                for line in frame.splitlines()
                if line.startswith(("OUTPUT\tRESULT\t", "OUTPUT\tRESULT_ARRAY\t"))
            ]
            recorded_shots.append("".join(bits)[::-1])
        assert recorded_shots == shots
    else:
        assert output is None
    if program_format in {ProgramFormat.QIR_BASE_STRING, ProgramFormat.QIR_BASE_MODULE}:
        assert job.get_dense_statevector() == pytest.approx([2**-0.5, 2**-0.5, 0, 0])
    else:
        with pytest.raises(RuntimeError, match="Not supported"):
            job.get_dense_statevector()


@pytest.mark.parametrize(("program_format", "shots"), [(ProgramFormat.QASM3, 1), (ProgramFormat.QIR_BASE_MODULE, 0)])
def test_qir_output_capture_requires_qir_sampling(program_format: ProgramFormat, shots: int) -> None:
    """Reject output requests that cannot produce a sampled QIR stream."""
    device = open_device("mqt.ddsim.default")
    payload = (
        BELL
        if program_format == ProgramFormat.QASM3
        else compile_program(BELL, output=OutputFormat.QIR_BASE).to_bitcode()
    )
    with pytest.raises(RuntimeError, match="Not supported"):
        device.submit_job(
            cast("Sequence[str] | Sequence[bytes]", [payload]),
            program_format,
            num_shots=shots,
            custom2=True,
        )
    # The RTTI-free adapter can lose the nested exception text on macOS.
    with pytest.raises(ValueError, match="Failed to submit compiled program"):
        submit_program(BELL, target=device, program_format=program_format, num_shots=shots, custom2=True)
