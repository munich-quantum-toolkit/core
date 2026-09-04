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

The compiler collection is the circuit and program interface in MQT Core v4.

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

## Inspect a QC program

Use the inspection methods of a {py:class}`~mqt.core.mlir.QCProgram` to count
gates without parsing the textual IR:

```{code-cell} ipython3
print("Gates:", compiled.num_gates())
print("Single-qubit gates:", compiled.num_single_qubit_gates())
print("Two-qubit gates:", compiled.num_two_qubit_gates())
```

These counts describe the entry-point IR. A gate in each structured control-flow
region counts once, regardless of the runtime path or loop iteration count.
Barriers do not count, and operations inside gate modifiers do not count again.

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

This compiler route is the Qiskit circuit interface in MQT Core v4.

Qiskit 2.5's C API cannot construct classical expressions or structured control
flow, so export uses Qiskit's public Python classes for these operations.

| Circuit feature                                                      | Import               | Export                        |
| -------------------------------------------------------------------- | -------------------- | ----------------------------- |
| Standard gates, constructible numeric modifiers, and global phase    | Supported            | Supported                     |
| Other finite numeric modifiers                                       | Supported            | Rejected                      |
| Measurement, reset, and barrier                                      | Supported            | Supported                     |
| Canonical named registers and leading loose bits                     | Supported            | Explicit registers            |
| Custom instructions with finite, acyclic definitions                 | Recursively expanded | Not applicable                |
| Nested `if`/`else`, `for`, `while`, and `switch`                     | Supported            | Supported                     |
| Classical-bit and register conditions                                | Supported            | Supported                     |
| Constant Boolean, `Uint` up to 64 bits, and `Float` expressions      | Supported            | Supported                     |
| Clbit and ClassicalRegister expression variables                     | Supported            | Supported                     |
| Fixed-width bitwise operations, comparisons, and bounded shifts      | Supported            | Supported                     |
| Direct complete-register comparisons wider than 64 bits              | Supported            | Supported                     |
| Clbit, indexed-register, and whole-register `Store` assignments      | Supported            | Supported                     |
| Initialized local classical variables and enclosing captures         | Supported            | Native `expr.Var` and `Store` |
| External runtime input variables                                     | Rejected             | Rejected                      |
| `break` in `for` and `while`, including nested conditionals/switches | Supported            | Native `BreakLoopOp`          |
| Free symbols and supported real parameter expressions                | Supported            | Supported                     |
| Parameter-vector elements                                            | Supported            | Supported                     |
| Dense numeric unitaries up to eight qubits                           | Supported            | Supported                     |
| Register aliases or interleaved membership                           | Rejected             | Rejected                      |
| Transpiler layout metadata                                           | Accepted and ignored | Not emitted                   |

Classical-expression variables may refer to Clbits or ClassicalRegisters in the
containing circuit. This includes values used only by the condition or switch
target and not by a control-flow block. External runtime inputs remain
unsupported.

Free symbols become named {code}`f64` program inputs. Parameter-vector elements
retain their grouping and index, preserving vector order and positional binding
across a round trip; similarly named standalone parameters remain standalone.
Elements used in different structured-control blocks are restored into one
shared vector for the complete circuit tree. Free parameter vectors and their
combined declared size in one translated circuit are each limited to 65,536
elements. Parameter-expression trees support at most 64 levels and 4,096 nodes.
Import and export support real addition, subtraction, multiplication, division,
power, negation, trigonometric and inverse trigonometric functions, exponential,
logarithm, absolute value, and real conjugation. Other parameter-expression
functions are rejected. Lexically bound {code}`for`-loop induction parameters
are supported and remain distinct from free symbols. Parameterized
custom-instruction definitions are expanded after their symbols and expressions
are resolved. Definition expansion rejects missing definitions, cycles, operand
arity mismatches, nesting beyond 64 levels, and more than 10 million expanded
operations.

Structured-control export supports scalar results from {code}`scf.if` and
{code}`scf.index_switch`, carried scalar state in constant-range
{code}`scf.for`, and general two-region {code}`scf.while`. Ordinary conditions
retain native while-loop conditions. More general loops use a constant-true
`WhileLoopOp`, before-region instructions, a conditional `BreakLoopOp`, and
after-region instructions. The condition's values become the loop results on
exit and the after-region arguments on continuation. Edge updates preserve
parallel-assignment semantics, including swaps and unequal before/after tuples.

Local scalar state covers Boolean values, integers of widths 1–64, and `f64`
using supported backend operations. It uses native `expr.Var` declarations,
`Store` assignments, and captures in nested circuits. Compiler variables do not
increase the classical-bit or register counts. Import accepts initialized local
variables and captures from enclosing supported circuits. External runtime
inputs remain unsupported. Runtime classical values cannot be used as symbolic
gate parameters.

Canonical short-circuit Boolean expressions retain their native expression form.
A live {code}`scf.for` induction value must reduce to an affine {code}`f64` gate
parameter. The exporter preserves one Qiskit parameter identity for that value
throughout its lexical body. An {code}`scf.index_switch` selector must be a
constant index or a supported Boolean/Uint expression converted with
{code}`arith.index_castui`. Switch labels must be nonnegative constants that fit
the target width.

Nested blocks may capture existing qubits, classical bits, and local variables
but may not allocate or release circuit resources. Control flow and classical
expressions may nest up to 64 levels, and classical expression trees may contain
at most 16,384 nodes (parameter-expression limits are unchanged). Integer values
use exact widths from 1 through 64. The only wider form is a direct unsigned
comparison between one complete `ClassicalRegister` and one same-width literal;
computed, packed, and signed wide values remain rejected. Both expression
conditions and tuple conditions such as `if_test((register, value))` support
this form. Tuple equalities with a value outside the register range become
false. Standard `arith.cmpi` handles every comparison: signed ordering is
encoded by XOR-biasing both operands' sign bits, including computed operands.
Casts preserve truncation and sign/zero extension. Bitwise operations, modular
arithmetic, integer selection, and shifts share these typed rules. Import guards
runtime shifts so overshifts produce zero; export preserves the guards.
Rotations and population count are expanded through the same bounded integer
lowering used by jeff. Unsupported operations, invalid widths, non-finite
constants, unsupported index uses, and dynamic for-loop bounds fail during
validation. Core's constant-zero `i64` status return is not a classical output.
Whole-register reads map to Qiskit `ClassicalRegister` expressions, and writes
map to atomic Qiskit `Store` operations. Indexed stores assume that their
runtime index is in bounds. The Qiskit C API does not expose `Store`, so the
adapter inspects and constructs that instruction through Qiskit's public Python
classes, as it already does for structured control flow. Internal entry-block
CBit storage becomes additional Qiskit registers, ordered before returned
registers; Qiskit exposes all circuit storage. OpenQASM remains the source
interchange path for arbitrary register widths.

Every public CBit output is exported as a Qiskit `ClassicalRegister`; an unnamed
allocation receives a collision-free `_mqt_cN` name. This preserves the CBit
register boundary and gives whole-register writes a valid Qiskit lvalue. Loose
input Clbits therefore round trip semantically, but not as loose output bits.

Conditions and switch targets may read a zero-initialized CBit register. An
undefined CBit may be read only after a definite write to that bit, and every
bit of an undefined returned register must be definitely initialized. Branches
intersect their initialization facts. A while loop's before region executes at
least once; its after region may execute zero times. A captured classical
snapshot must be materialized before a later write to the same register or
satisfy the existing snapshot checks.

Each exported measurement must write to one static public CBit in the same
block. Destinations may be reused; later measurements overwrite earlier values
in program order. A measurement's destination store must follow it directly,
apart from constant operations. A conditional or otherwise delayed destination
store is rejected because Qiskit cannot preserve it as one measurement
instruction. The measurement result may feed supported classical expressions
after that store. General scalar control flow saves live measurement results in
native variables before later writes. Deferred measurement expressions require
an unchanged destination CBit.

Dense numeric unitaries remain explicit matrix operations during import and
export. Target compilation synthesizes supported one- and two-qubit matrices to
the target gate set. Dense unitary operations support at most eight qubits.
Qiskit import preserves inverse, numeric power, and closed-control modifiers on
dense-unitary operations. Export preserves inverse and closed-control modifiers.
Other powers require canonicalization or synthesis.

A circuit remains valid when {code}`circ.layout` is present. The importer
translates the circuit operations and deliberately does not preserve physical or
virtual layout metadata.

Input validation finishes before an MLIR module is created. Generic output
validation finishes before Qiskit construction starts; the version-specific
adapter validates its constructed blocks before returning the top-level circuit.
Unsupported programs therefore fail without modifying the source object or
exposing a partial result.

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

Pauli twirling is available as an opt-in textual pass. It supports CX, CZ, ECR,
and iSWAP gates, keeps every inserted Pauli operation (including identities)
explicit, and preserves the exact global phase. The seed defaults to {code}`42`:

```{code-cell} ipython3
twirled = compile_program(bell_qasm, output=OutputFormat.QCO)
twirled.run_pass_pipeline("pauli-twirl-2q-gates{seed=42}")
```

Each invocation produces one deterministic realization; omitting the seed
reproduces the realization selected by {code}`seed=42`. To construct an ensemble
for noise tailoring, transform copies with different seeds, execute them, and
aggregate their measurement results. Different seeds are not guaranteed to
produce distinct realizations.

This is a raw-QCO transformation. It does not place twirling relative to target
mapping or synthesis and does not guarantee that the result uses a target's
native gate set. Target-aware twirling is not currently available through the
target compilation pipeline.

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

Integer expressions support widths through 64 bits. Integer absolute value and
power require jeff's native widths: 1, 8, 16, 32, or 64. Import preserves
straight-line array snapshots, but rejects live old array values across mutating
control flow and shared array updates inside switch or while regions. Scalar
branch results and loop state are preserved, including different loop input and
result tuples. Quantum allocations and deallocations inside conditional regions
remain unsupported.

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
