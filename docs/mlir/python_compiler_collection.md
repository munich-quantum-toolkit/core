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
objects, Qiskit circuits, and compiler programs. Depending on the selected
output format, it returns an optimized circuit representation, an
optimization-oriented representation, a serializable program, or QIR.

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
returns a {py:class}`~mqt.core.mlir.QCProgram`: the QC representation after an
optimized QCO round trip. Its `ir` property exposes the textual MLIR
representation for inspection and debugging. Writing MLIR is not required to use
the compiler.

:::{important}

### Preserve observable results

The compiler removes dead code. A circuit that only prepares a state, including
a Bell state without measurements and returned results, has no observable effect
and can be removed by optimization. Programs intended for execution should
measure the relevant qubits and return the measurement results.

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
| Obtain an optimized circuit representation | `OutputFormat.QC` (default) | `QCProgram` |
| Inspect or transform the optimization IR | `OutputFormat.QCO` | `QCOProgram` |
| Serialize a compiler program | `OutputFormat.JEFF` | `JeffProgram` |
| Generate QIR | `OutputFormat.QIR_BASE` or `OutputFormat.QIR_ADAPTIVE` | `QIRProgram` |

For example, select QCO to inspect the representation used for optimization:

```{code-cell} ipython3
optimized = compile_program(bell_qasm, output=OutputFormat.QCO)
print(optimized.ir)
```

## Construct a pipeline

`QCProgram`, `QCOProgram`, `JeffProgram`, and `QIRProgram` own their MLIR
modules. A conversion consumes its source program by default, avoiding an
implicit copy of a potentially large module. Pass `copy=True` when the source
must remain available.

The following pipeline retains the imported QC program, explicitly runs cleanup
and optimization on QCO, and converts the result back to QC:

```{code-cell} ipython3
qc = QCProgram.from_qasm_str(bell_qasm)
qco = qc.to_qco(copy=True)
qco.cleanup()
qco.optimize(enable_hadamard_lifting=True)
final_qc = qco.to_qc()

assert qc.is_valid
assert not qco.is_valid
print(final_qc.ir)
```

The dialect reference documents the representations used in this pipeline:

- {doc}`QC <QC>` uses reference semantics and provides interoperability with
  OpenQASM, Qiskit, QIR, and related tools.
- {doc}`QCO <QCO>` uses value semantics and linear typing for optimization.

The {doc}`QTensor <QTensor>` dialect extends QCO with linear, one-dimensional
qubit registers. See the {doc}`conversion reference <Conversions>` for the
transformations between these representations.

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

To generate QIR, select a QIR target profile. `QIRProgram` provides its lowered
MLIR through `ir` and the generated LLVM IR through `llvm_ir`.

```{code-cell} ipython3
qir = compile_program(bell_qasm, output=OutputFormat.QIR_BASE)
assert qir.profile is QIRProfile.BASE
print(qir.llvm_ir)
```

The dialect and conversion references provide the operation-level details for
applications that need to control individual compiler stages.
