---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# MQT Compiler Collection

The MQT Compiler Collection compiles, optimizes, and exchanges structured
quantum programs. Its three interfaces share the same MLIR representations and
passes:

| Interface    | Entry point                                                         |
| ------------ | ------------------------------------------------------------------- |
| Command line | `mqt-cc`, with examples below                                       |
| Python       | {py:mod}`mqt.core.mlir`                                             |
| C++          | [Source-tree compiler API](target_compilation.md#c-source-tree-api) |

For a guided explanation of the representations and transformations, start with
the {doc}`getting-started tutorial <../tutorials/index>`. This guide documents
the interfaces and options for applying those concepts.

The Python examples below accept source strings, {code}`.qasm`, {code}`.mlir`,
and {code}`.jeff` files, Qiskit {py:class}`~qiskit.circuit.QuantumCircuit`
objects, and typed compiler programs. The requested output format determines
where compilation stops and which program type is returned.

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

For versionless OpenQASM text, use
`compile_program(QCProgram.from_openqasm_str(source))`. Automatic source-string
detection uses the `OPENQASM` header; see {doc}`OpenQASM` for the import
contract.

## Inspect a QC program

Use the inspection methods of a {py:class}`~mqt.core.mlir.QCProgram` to count
gates without parsing the textual IR:

```{code-cell} ipython3
print("Gates:", compiled.num_gates())
print("Single-qubit gates:", compiled.num_single_qubit_gates())
print("Two-qubit gates:", compiled.num_two_qubit_gates())
```

These are static gate counts of the entry-point IR. A gate in each structured
control-flow region counts once, regardless of the runtime path or loop
iteration count. Barriers do not count, and operations inside gate modifiers do
not count again. The counts do not expand function calls or estimate the gates
executed at runtime.

## Select an output format

Select an output format to stop the pipeline at a particular representation:

| Purpose                                  | Output format                                          | Result type       |
| ---------------------------------------- | ------------------------------------------------------ | ----------------- |
| Inspect frontend translation             | `OutputFormat.QC_IMPORT`                               | `QCProgram`       |
| Inspect QCO immediately after conversion | `OutputFormat.QCO`                                     | `QCOProgram`      |
| Inspect QCO after optimization           | `OutputFormat.QCO_OPTIMIZED`                           | `QCOProgram`      |
| Obtain the optimized circuit             | `OutputFormat.QC` (default)                            | `QCProgram`       |
| Emit an optimized OpenQASM program       | `OutputFormat.OPENQASM3`                               | `OpenQASMProgram` |
| Convert to the jeff dialect              | `OutputFormat.JEFF`                                    | `JeffProgram`     |
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
    reparsed = QCProgram.from_openqasm_file(path)

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

The exporter supports structured control flow and runtime classical-bit indices.
Quantum register indices and ranges must be static. Surviving runtime
assertions, checked-index machinery, and live poison values fail with an MLIR
diagnostic. See {doc}`OpenQASM` for the complete support table.

## Use Qiskit circuits directly

Install the optional Qiskit integration with {code}`mqt-core[qiskit]`. The extra
also supports SDK uses with older Qiskit releases; direct compiler translation
requires a registered version, currently Qiskit 2.5.x. These circuits can be
translated between {py:class}`~qiskit.circuit.QuantumCircuit` and
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

QCO programs also provide {py:meth}`~mqt.core.mlir.QCOProgram.to_qiskit`. It
converts a copy through QC and leaves the original program unchanged, even if
export fails. Both exporters accept `target=target` to map static target site
IDs to dense physical-qubit indices in target site order.

```{code-cell} ipython3
qco = direct.to_qco(copy=True)
restored = qco.to_qiskit()
assert qco.is_valid
```

See {doc}`qiskit` for supported circuit features and translation limitations.

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
qc = QCProgram.from_openqasm_str(bell_qasm)
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
assert custom.is_valid
print(custom.ir)
```

Pauli twirling is available as an opt-in textual pass. It supports CX, CZ, ECR,
and iSWAP gates, keeps every inserted Pauli operation (including identities)
explicit, and preserves the exact global phase. The seed defaults to {code}`42`:

```{code-cell} ipython3
twirled = compile_program(bell_qasm, output=OutputFormat.QCO)
twirled.run_pass_pipeline("pauli-twirl-2q-gates{seed=42}")
print(twirled.ir)
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
available through the compiler collection. Two independent measured qubits can
share one physical qubit, with a reset between uses. This example uses scalar
QCO values directly; register operations and intervening classical stores can
prevent the raw pass from proving that reuse is safe:

```{code-cell} ipython3
from mqt.core.mlir import QCOProgram

independent_qubits = """module {
  func.func @main() -> (i1, i1) attributes {mqt.entry_point} {
    %q0 = qco.alloc : !qco.qubit
    %q1 = qco.alloc : !qco.qubit
    %h0 = qco.h %q0 : !qco.qubit -> !qco.qubit
    %h1 = qco.h %q1 : !qco.qubit -> !qco.qubit
    %m0, %c0 = qco.measure %h0 : !qco.qubit
    %m1, %c1 = qco.measure %h1 : !qco.qubit
    qco.sink %m0 : !qco.qubit
    qco.sink %m1 : !qco.qubit
    return %c0, %c1 : i1, i1
  }
}
"""
raw_reuse = QCOProgram.from_mlir_str(independent_qubits)
before = raw_reuse.ir.count("qco.alloc")
raw_reuse.reuse_qubits()

composite_reuse = QCOProgram.from_mlir_str(independent_qubits)
composite_reuse.run_qubit_reuse_pipeline()
assert raw_reuse.is_valid and composite_reuse.is_valid
after = raw_reuse.ir.count("qco.alloc")
assert before == 2 and after == 1
print(f"Qubit allocations: {before} → {after}")
print(raw_reuse.ir)
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
controls before applying the raw {code}`reuse-qubits` pass. It also runs
{code}`remove-dead-gates`, which can remove gates whose results are unused,
including operations on unmeasured qubits. Use these passes only when those
results may be discarded.

The {code}`qco_pipeline` argument replaces the default QCO optimization
pipeline. It is applied when compilation proceeds beyond the raw
{code}`OutputFormat.QCO` checkpoint.

## Serialize programs and generate QIR

{py:class}`~mqt.core.mlir.JeffProgram` holds MLIR in the {code}`jeff` dialect.
Use {py:meth}`~mqt.core.mlir.JeffProgram.to_bytes` or
{py:meth}`~mqt.core.mlir.JeffProgram.write` to serialize it. The bytes or file
can be loaded and compiled again in a later process.

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
filenames, including filenames without an extension, produce bitcode.

The {doc}`QC <QC>`, {doc}`QCO <QCO>`, and {doc}`QTensor <QTensor>` references
describe the underlying operations. See {doc}`Conversions` for conversions
between dialects.
