---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# Using the MQT Compiler Collection from Python

The {py:mod}`mqt.core.mlir` module provides Python access to the MQT Compiler
Collection. It accepts source strings, {code}`.qasm`, {code}`.mlir`, and
{code}`.jeff` files, Qiskit {py:class}`~qiskit.circuit.QuantumCircuit` objects,
and typed compiler programs. The requested output format determines where
compilation stops and which program type is returned.

The compiler does not accept legacy {py:class}`~mqt.core.ir.QuantumComputation`
objects. Convert such an object to OpenQASM 3 with
{py:meth}`~mqt.core.ir.QuantumComputation.qasm3_str` before passing it to the
compiler.

Install {doc}`MQT Core <../installation>` and import the compiler interface:

```{code-cell} ipython3
from mqt.core.mlir import OutputFormat, QCProgram, QIRProfile, compile_program
```

To compile for a configured QDMI device, see
{doc}`target compilation <target_compilation>`.

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
returns a {py:class}`~mqt.core.mlir.QCProgram`. Its
{py:attr}`~mqt.core.mlir.Program.ir` property exposes the textual MLIR
representation for inspection and debugging. Programs do not need to be written
in MLIR to use the compiler.

:::{important}
The compiler removes dead code. A circuit that only prepares a state has no
observable effect and will be removed by optimizations. Programs intended for
execution should measure the relevant qubits and return the measurement results.

In OpenQASM 3, assigning measurements to a classical register, as in the example
above, makes those results return values of the imported program. When
constructing MLIR directly, return the values produced by the measurement
operations.
:::

## Select an output format

Select an output format to stop the pipeline at a particular representation:

| Purpose                                  | Output format                                          | Result type       |
| ---------------------------------------- | ------------------------------------------------------ | ----------------- |
| Inspect frontend translation             | `OutputFormat.QC_IMPORT`                               | `QCProgram`       |
| Inspect QCO immediately after conversion | `OutputFormat.QCO`                                     | `QCOProgram`      |
| Inspect QCO after optimization           | `OutputFormat.QCO_OPTIMIZED`                           | `QCOProgram`      |
| Obtain the optimized circuit             | `OutputFormat.QC` (default)                            | `QCProgram`       |
| Emit an optimized OpenQASM program       | `OutputFormat.OPENQASM3`                               | `OpenQASMProgram` |
| Serialize a compiler program             | `OutputFormat.JEFF`                                    | `JeffProgram`     |
| Generate QIR                             | `OutputFormat.QIR_BASE` or `OutputFormat.QIR_ADAPTIVE` | `QIRProgram`      |

For example, select optimized QCO to inspect the representation after the
default QCO pass pipeline:

```{code-cell} ipython3
optimized = compile_program(bell_qasm, output=OutputFormat.QCO_OPTIMIZED)
print(optimized.ir)
```

## Emit OpenQASM

Request {py:attr}`~mqt.core.mlir.OutputFormat.OPENQASM3` to emit the program
after the normal QCO optimization and conversion back to QC:

```{code-cell} ipython3
openqasm = compile_program(bell_qasm, output=OutputFormat.OPENQASM3)
print(openqasm.source)
```

The returned {py:class}`~mqt.core.mlir.OpenQASMProgram` owns its source and can
write it directly:

```{code-cell} ipython3
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as directory:
    path = Path(directory) / "bell.qasm"
    openqasm.write(path)
    reparsed = QCProgram.from_qasm_file(path)

assert reparsed.is_valid
```

Use {py:meth}`~mqt.core.mlir.QCProgram.to_openqasm3` to clean up and export the
current QC program without QCO optimization. The resulting
{py:class}`~mqt.core.mlir.OpenQASMProgram` can be passed directly to
{py:func}`~mqt.core.mlir.compile_program`:

```{code-cell} ipython3
recompiled = compile_program(openqasm, output=OutputFormat.QC_IMPORT)
assert isinstance(recompiled, QCProgram)
```

The exporter targets practical structured programs with static qubit and bit
indices. Dynamic indexing, dynamic ranges, surviving runtime assertions,
checked-index machinery, and live poison values fail with an MLIR diagnostic.
See {doc}`OpenQASM` for the complete support table.

## Use Qiskit circuits directly

Install the optional Qiskit integration with {code}`mqt-core[qiskit]`. Qiskit
2.5.x circuits can be translated directly between
{py:class}`~qiskit.circuit.QuantumCircuit` and
{py:class}`~mqt.core.mlir.QCProgram`:

```{code-cell} ipython3
from qiskit import QuantumCircuit

qiskit_bell = QuantumCircuit(2, 2)
qiskit_bell.h(0)
qiskit_bell.cx(0, 1)
qiskit_bell.measure(range(2), range(2))

direct = QCProgram.from_qiskit(qiskit_bell)
restored = direct.to_qiskit()
compiled_qiskit = compile_program(qiskit_bell)

assert direct.is_valid  # Export does not consume the QC program.
assert restored.count_ops() == qiskit_bell.count_ops()
assert compiled_qiskit.is_valid
```

This compiler route does not construct an intermediate
{py:class}`~mqt.core.ir.QuantumComputation`. The existing
{py:func}`~mqt.core.plugins.qiskit.qiskit_to_mqt`,
{py:func}`~mqt.core.plugins.qiskit.mqt_to_qiskit`, and {py:func}`mqt.core.load`
interfaces remain independent and retain their existing version range and
behavior.

Import and export have different contracts because Qiskit 2.5 can inspect more
program structures than its C API can construct.

| Circuit feature                                                   | Import               | Export         |
| ----------------------------------------------------------------- | -------------------- | -------------- |
| Standard gates, constructible numeric modifiers, and global phase | Supported            | Supported      |
| Other finite numeric modifiers                                    | Supported            | Rejected       |
| Measurement, reset, and barrier                                   | Supported            | Supported      |
| Canonical named registers and leading loose bits                  | Supported            | Supported      |
| Custom instructions with finite, acyclic definitions              | Recursively expanded | Not applicable |
| Nested `if`/`else`, `for`, `while`, and `switch`                  | Supported            | Rejected       |
| Classical-bit and register conditions                             | Supported            | Rejected       |
| Constant Boolean, `Uint` up to 64 bits, and `Float` expressions   | Supported            | Rejected       |
| Clbit and ClassicalRegister expression variables                  | Supported            | Rejected       |
| Standalone classical runtime variables                            | Rejected             | Rejected       |
| Free symbols and supported real parameter expressions             | Supported            | Supported      |
| Parameter-vector elements                                         | Rejected             | Not emitted    |
| Dense numeric unitaries up to eight qubits                        | Supported            | Supported      |
| Register aliases or interleaved membership                        | Rejected             | Rejected       |
| Transpiler layout metadata                                        | Accepted and ignored | Not emitted    |

Classical-expression variables may refer to Clbits or ClassicalRegisters in the
containing circuit. This includes values used only by the condition or switch
target and not by a control-flow block. Standalone runtime variables remain
unsupported.

Free standalone symbols become named {code}`f64` program inputs.
Parameter-vector elements are rejected because converting them to standalone
parameters would change positional binding order. Standalone parameter names
that contain brackets remain ordinary scalar names. Parameter-expression trees
support at most 64 levels and 4,096 nodes. Import and export support real
addition, subtraction, multiplication, division, power, negation, trigonometric
and inverse trigonometric functions, exponential, logarithm, absolute value, and
real conjugation. Other parameter-expression functions are rejected. Lexically
bound {code}`for`-loop induction parameters are supported and remain distinct
from free symbols. Parameterized custom-instruction definitions are expanded
after their symbols and expressions are resolved. Definition expansion rejects
missing definitions, cycles, operand arity mismatches, nesting beyond 64 levels,
and more than 10 million expanded operations.

Dense numeric unitaries remain explicit matrix operations during import and
export. Target compilation synthesizes supported one- and two-qubit matrices to
the target gate set. Dense unitary operations support at most eight qubits.
Qiskit import preserves inverse, numeric power, and closed-control modifiers on
dense-unitary operations. Export preserves inverse and closed-control modifiers.
Other powers require canonicalization or synthesis.

A circuit remains valid when {code}`circ.layout` is present. The importer
translates the circuit operations and deliberately does not preserve physical or
virtual layout metadata.

Input validation finishes before an MLIR module is created. Output validation
finishes before a Qiskit circuit is allocated. Unsupported programs therefore
fail without modifying the source object or exposing a partial result.

The binding imports Qiskit only when circuit translation is requested. It
accepts versions in the registered {code}`>=2.5.0,<2.6.0` range and verifies the
native API version before reading a circuit.

## Run passes explicitly

{code}`QCProgram`, {code}`QCOProgram`, {code}`JeffProgram`, and
{code}`QIRProgram` own their MLIR modules. Conversions between these MLIR-backed
program objects consume their source by default, avoiding an implicit copy of a
potentially large module. Pass {code}`copy=True` when the source must remain
available. {code}`OpenQASMProgram` instead owns immutable source text and
remains reusable when passed to {code}`compile_program`.

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
{code}`mqt-cc`:

```{code-cell} ipython3
custom = compile_program(
    bell_qasm,
    output=OutputFormat.QCO_OPTIMIZED,
    qco_pipeline="hadamard-lifting,merge-single-qubit-rotation-gates",
)
```

The raw qubit-reuse pass and its composite preparation pipeline are both
available through the compiler collection:

```{code-cell} ipython3
raw_reuse = compile_program(bell_qasm, output=OutputFormat.QCO)
raw_reuse.reuse_qubits()

composite_reuse = compile_program(bell_qasm, output=OutputFormat.QCO)
composite_reuse.run_qubit_reuse_pipeline()
```

The same flows can be composed with the default optimization pipeline in the
compiler driver:

```console
mqt-cc input.qasm --emit=qco-optimized \
  --pass-pipeline='builtin.module(reuse-qubits,mqt-qco-default)'
mqt-cc input.qasm --emit=qco-optimized \
  --pass-pipeline='builtin.module(mqt-qubit-reuse,mqt-qco-default)'
```

The {code}`mqt-qubit-reuse` pipeline lifts measurements and replaces classical
controls before applying the raw {code}`reuse-qubits` pass.

The {code}`qco_pipeline` argument replaces the default QCO optimization
pipeline. It is applied when compilation proceeds beyond the raw
{code}`OutputFormat.QCO` checkpoint.

## Serialize programs and generate QIR

{code}`jeff` is a serializable representation that can be stored and compiled
again in a later process.

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

To generate QIR, select a target profile. {py:class}`~mqt.core.mlir.QIRProgram`
provides the QIR MLIR through {py:attr}`~mqt.core.mlir.Program.ir` and the
translated LLVM IR through {py:attr}`~mqt.core.mlir.QIRProgram.llvm_ir`.

```{code-cell} ipython3
qir = compile_program(bell_qasm, output=OutputFormat.QIR_BASE)
assert qir.profile is QIRProfile.BASE
print(qir.llvm_ir)
```

Use {py:meth}`~mqt.core.mlir.QIRProgram.to_bitcode` to obtain LLVM bitcode as
{code}`bytes`, or {py:meth}`~mqt.core.mlir.QIRProgram.write_bitcode` to write a
{code}`.bc` file directly. The
[QIR guide](../qir/index.md#executing-generated-qir-from-python) shows how to
execute the generated bytes directly with QIR-Runner's `qirrunner` Python
package.

The {code}`mqt-cc` compiler driver selects the QIR serialization from the output
filename. Use {code}`.ll` for textual LLVM IR and {code}`.bc` for LLVM bitcode:

```console
mqt-cc input.qasm --emit=qir-base -o output.ll
mqt-cc input.qasm --emit=qir-adaptive -o output.bc
```

Writing QIR to standard output also produces textual LLVM IR. All other output
filenames, including filenames without an extension, retain the bitcode output
used by earlier versions.

The {doc}`QC <QC>`, {doc}`QCO <QCO>`, and {doc}`QTensor <QTensor>` references
describe the underlying operations. See {doc}`Conversions` for the lowering
steps between dialects.
