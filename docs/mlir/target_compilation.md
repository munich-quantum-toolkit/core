# Compile for a QDMI device

An MLIR {code}`mlir::CompilerTarget` is an immutable snapshot of a circuit-model
device. It contains the device sites, topology, native operations, and available
calibration data. Compilation decomposes supported multi-qubit operations,
optimizes and maps the program, synthesizes native gates, and verifies that the
result conforms to the target.

The snapshot is independent of its originating QDMI session. It can therefore be
stored, copied cheaply, and reused for multiple compilations.

## Python

Open a configured QDMI device and snapshot it as a compiler target:

```python
from mqt.core.mlir import CompilerTarget, OutputFormat, compile_program

target = CompilerTarget.from_device_id("mqt.sc.iqm.garnet")
compiled = compile_program(
    "bell.qasm",
    target=target,
    output=OutputFormat.QCO_OPTIMIZED,
)
```

Target compilation accepts optimized QCO, QC, or QIR output and uses the
canonical QCO pipeline; it cannot be combined with a custom `qco_pipeline`.

The target can also be constructed directly. Omitting `couplings` selects
all-to-all connectivity; omitting `operations` means that every operation is
native:

```python
target = CompilerTarget(3, couplings=[(0, 1), (1, 2)])
```

Runtime classical control is opt-in. By default, a target accepts only
straight-line quantum programs. Declare each supported control form explicitly;
for example, a target with measurement-conditioned forward branching uses:

```python
target = CompilerTarget(
    3,
    classical_control=[CompilerTarget.ClassicalControl.CONDITIONAL],
)
```

Target compilation checks these capabilities before cleanup, mapping, or
synthesis. An unsupported program therefore fails without partially mapping the
circuit, and the diagnostic names both the missing capability and the operation
that requires it. The four independent capabilities are:

- {code}`CONDITIONAL` for runtime {code}`qco.if` and {code}`scf.if` operations.
- {code}`ITERATION` for counted {code}`scf.for` loops.
- {code}`CONDITIONAL_LOOP` for runtime {code}`scf.while` loops.
- {code}`MULTIWAY_BRANCH` for {code}`qco.index_switch` and
  {code}`scf.index_switch` operations.

Conditional support does not imply loop or multiway-branch support. The
preflight follows only the selected region of an {code}`if` or
{code}`index_switch` whose selector is constant and still checks every reachable
nested operation. The following cleanup pipeline removes a static
{code}`qco.index_switch` before mapping. Dynamic qubit indexing, unstructured
control flow, and qubit tensors carried through generic structured control
remain unsupported. A {code}`qco.if` may carry statically shaped, rank-one qubit
tensors only when cleanup can scalarize every constant-index branch access;
dynamic, partial, or repeated tensor access remains unsupported. The preflight
rejects unsupported forms before mapping even when every listed capability is
enabled.

Use {py:meth}`~mqt.core.mlir.QCOProgram.compile_for_target` to apply target
compilation to an existing QCO program. Compilation runs in place. If a pass
fails, earlier passes may already have changed the program. Copy the program
before compilation if the caller must preserve the input. For pass-level
benchmarking, the C++ API exposes separate factories for pre-routing
optimization, mapping, native synthesis, and conformance verification.

Target compilation preserves quantum operations even when their final qubit
values are not measured or returned. This supports measurement-free programs,
such as state preparation or larger building blocks compiled to a target-native
instruction set. Dead gates are removed only by the explicit `remove-dead-gates`
pass and by pipelines that include it, such as `mqt-qubit-reuse`.

## Command line from a source build

List the stable IDs of configured QDMI devices:

```console
mqt-cc --qdmi-list-devices
```

Select a device when compiling:

```console
mqt-cc --qdmi-device=mqt.sc.iqm.garnet \
  --emit=qco-optimized input.qasm
```

An explicit registry file can be selected before device discovery:

```console
mqt-cc --qdmi-config=/path/to/qdmi.json \
  --qdmi-device=example.device input.qasm
```

Target compilation produces optimized QCO, QC, or QIR. It cannot be combined
with a custom `--passes` pipeline because the canonical target pipeline owns the
required pass ordering.

## C++ source-tree API

The source build provides a narrow, non-throwing QDMI bridge between a stable
device ID and the compiler-owned target:

```cpp
#include "mlir/Compiler/QDMIAdapter.h"
#include "mlir/Compiler/Programs.h"
#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>

auto target = mlir::compilerTargetFromDeviceId("mqt.sc.iqm.garnet");
if (!target) {
  llvm::errs() << "Failed to create compiler target: "
               << llvm::toString(target.takeError()) << '\n';
  return 1;
}

auto qc = mlir::QCProgram::fromQASMFile("input.qasm");
if (!qc) {
  return 1;
}
auto qco = std::move(*qc).intoQCO();
if (!qco || !qco->compileForTarget(*target)) {
  return 1;
}
```

The adapter accepts circuit-model devices whose operations are available
throughout the topology in both operand orientations. Operand-symmetric gates,
such as CZ, may report each edge once. Neutral-atom zone models require a
different compilation model and are rejected with a diagnostic.

The bundled Garnet and Emerald snapshots contain available T1, T2, and fidelity
data. Operation durations are absent because they were unavailable. See
{doc}`../qdmi/sc_device` for their stable IDs and {doc}`../qdmi/configuration`
for registry configuration.

If the program should use fewer physical qubits, run the {code}`mqt-qubit-reuse`
pipeline before target compilation.

## Qiskit export

When exporting a program that has already been mapped to a
{py:class}`~mqt.core.mlir.CompilerTarget`, pass the same target to
{py:meth}`~mqt.core.mlir.QCProgram.to_qiskit`. The exporter maps each static
target site ID to its index in {py:attr}`~mqt.core.mlir.CompilerTarget.sites`
and creates a canonical physical Qiskit circuit. The circuit has one register
named {code}`q` with {py:attr}`~mqt.core.mlir.CompilerTarget.num_qubits` qubits.
This option does not run target compilation or emit Qiskit layout metadata.
Target-aware export requires static qubits whose site IDs belong to that target.
