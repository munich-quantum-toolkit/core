---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# Using the MQT Compiler Collection from Python

The {py:mod}`mqt.core.mlir` module provides Python access to the MQT Compiler
Collection. It accepts OpenQASM, MQT {py:class}`~mqt.core.ir.QuantumComputation`
objects, Qiskit circuits, and typed compiler programs. The requested output
format determines where compilation stops and which program type is returned.

Install {doc}`MQT Core <../installation>` and import the compiler interface:

```{code-cell} ipython3
from mqt.core.mlir import OutputFormat, QCProgram, QIRProfile, compile_program
```

## Compile an OpenQASM program

The following OpenQASM program prepares a Bell state and records the outcome of
measuring both qubits.

```{code-cell} ipython3
bell_qasm = """OPENQASM 3.0;
include "stdgates.inc";

qubit[2] q;
bit[2] result;

h q[0];
cx q[0], q[1];
result = measure q;
"""

compiled = compile_program(bell_qasm)
print(compiled.ir)
```

By default, `compile_program()` runs the standard optimization pipeline and
returns a {py:class}`~mqt.core.mlir.QCProgram`. Its `ir` property exposes the
textual MLIR representation for inspection and debugging. Programs do not need
to be written in MLIR to use the compiler.

:::{important}
The compiler removes dead code. A circuit that only prepares a state has no
observable effect and can be removed by optimization. Programs intended for
execution should measure the relevant qubits and return the measurement results.

In OpenQASM 3, assigning measurements to a classical register, as in the example
above, makes those results return values of the imported program. When
constructing MLIR directly, return the values produced by the measurement
operations.
:::

## Select an output format

The compiler accepts source strings, `.qasm`, `.mlir`, and `.jeff` files, MQT
and Qiskit circuit objects, and typed compiler programs. Select an output format
to stop the pipeline at a particular representation:

| Purpose | Output format | Result type |
| --- | --- | --- |
| Inspect frontend translation | `OutputFormat.QC_IMPORT` | `QCProgram` |
| Inspect QCO immediately after conversion | `OutputFormat.QCO` | `QCOProgram` |
| Inspect QCO after optimization | `OutputFormat.QCO_OPTIMIZED` | `QCOProgram` |
| Obtain the optimized circuit | `OutputFormat.QC` (default) | `QCProgram` |
| Serialize a compiler program | `OutputFormat.JEFF` | `JeffProgram` |
| Generate QIR | `OutputFormat.QIR_BASE` or `OutputFormat.QIR_ADAPTIVE` | `QIRProgram` |

For example, select optimized QCO to inspect the representation after the
default QCO pass pipeline:

```{code-cell} ipython3
optimized = compile_program(bell_qasm, output=OutputFormat.QCO_OPTIMIZED)
print(optimized.ir)
```

## Run passes explicitly

`QCProgram`, `QCOProgram`, `JeffProgram`, and `QIRProgram` own their MLIR
modules. A conversion consumes its source program by default, avoiding an
implicit copy of a potentially large module. Pass `copy=True` when the source
must remain available.

The following example keeps the imported QC program, applies transformations to
QCO, and converts the result back to QC:

```{code-cell} ipython3
qc = QCProgram.from_qasm_str(bell_qasm)
qco = qc.to_qco(copy=True)
qco.cleanup()
qco.merge_single_qubit_rotation_gates()
qco.lift_hadamards()
final_qc = qco.to_qc()

assert qc.is_valid
assert not qco.is_valid
print(final_qc.ir)
```

Architecture-independent QCO transformations can also be composed with MLIR's
textual pass-pipeline syntax. The same pass names and options are accepted by
`mqt-cc`:

```{code-cell} ipython3
custom = compile_program(
    bell_qasm,
    output=OutputFormat.QCO_OPTIMIZED,
    qco_pipeline="hadamard-lifting,merge-single-qubit-rotation-gates",
)
```

The `qco_pipeline` argument replaces the default QCO optimization pipeline. It
is applied when compilation proceeds beyond the raw `OutputFormat.QCO`
checkpoint.

## Serialize programs and generate QIR

Jeff is a serializable representation that can be stored and compiled again in a
later process.

```{code-cell} ipython3
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as directory:
    path = Path(directory) / "bell.jeff"
    jeff = compile_program(bell_qasm, output=OutputFormat.JEFF)
    jeff.write(path)
    restored = compile_program(path, output=OutputFormat.QC)

assert restored.is_valid
```

To generate QIR, select a target profile. `QIRProgram` provides the QIR MLIR
through `ir` and the translated LLVM IR through `llvm_ir`.

```{code-cell} ipython3
qir = compile_program(bell_qasm, output=OutputFormat.QIR_BASE)
assert qir.profile is QIRProfile.BASE
print(qir.llvm_ir)
```

Use `qir.to_bitcode()` to obtain LLVM bitcode as `bytes`, or
`qir.write_bitcode(path)` to write a `.bc` file directly.

The {doc}`QC <QC>`, {doc}`QCO <QCO>`, and {doc}`QTensor <QTensor>` references
describe the underlying operations. See {doc}`Conversions` for the lowering
steps between dialects.
