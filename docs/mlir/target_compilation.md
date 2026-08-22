# Compile for a QDMI device

An MLIR {code}`mlir::CompilerTarget` is an immutable snapshot of a circuit-model
device. It contains the device sites, topology, native operations, available
calibration data, and, when reported, payload-specific execution profiles.
Compilation decomposes supported multi-qubit operations, optimizes and maps the
program, synthesizes native gates, and verifies that the result conforms to the
target.

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

Target compilation accepts optimized QCO, QC, OpenQASM 3, or QIR output and uses
the canonical QCO pipeline; it cannot be combined with a custom `qco_pipeline`.

The target can also be constructed directly. Omitting {code}`couplings` selects
all-to-all connectivity; omitting {code}`operations` means that every operation
is native:

```python
target = CompilerTarget(3, couplings=[(0, 1), (1, 2)])
```

Execution features belong to a program format, not to the device globally. For
example, a directly constructed target that accepts measurement-feedback QIR
Adaptive programs can describe that payload as follows:

```python
from mqt.core.mlir import CompilerTarget

Feature = CompilerTarget.ProgramFeature
descriptor = CompilerTarget.PayloadDescriptor("qir", "2.1.0", "adaptive", CompilerTarget.PayloadEncoding.TEXT)
adaptive = CompilerTarget.ExecutionProfile(
    descriptor,
    capabilities=[
        CompilerTarget.ProgramCapability(Feature.MID_CIRCUIT_MEASUREMENT),
        CompilerTarget.ProgramCapability(Feature.MEASURED_QUBIT_REUSE),
        CompilerTarget.ProgramCapability(Feature.MEASUREMENT_RESULT_USE),
        CompilerTarget.ProgramCapability(Feature.BOOLEAN_COMPUTATION),
        CompilerTarget.ProgramCapability(Feature.FORWARD_BRANCHING),
    ],
)
target = CompilerTarget(
    3,
    execution_profiles=[adaptive],
)
```

{code}`ProgramFeature` values describe atomic execution semantics, including
mid-circuit measurement, measurement-result use, Boolean computation, forward
branching, counted or condition-terminated iteration, and multiway branching.
They are independent: forward branching does not imply loops, switches, or
general integer and floating-point computation. Runtime branch conditions must
use computation capabilities supported by the selected payload. Returning a
measurement result or storing it in a terminal output register is reporting, not
adaptive result use; another runtime consumer requires
{code}`MEASUREMENT_RESULT_USE`.

Target compilation keeps three questions separate:

1. The residual program has semantic requirements, such as using a measurement
   result in a forward branch.
2. The selected {code}`ExecutionProfile` lists the device features for one exact
   payload descriptor: ID, version, profile, and encoding.
3. Compiler legality determines whether the current IR can be lowered, for
   example whether quantum state is represented in a supported structural form.

The compiler serializes the selected descriptor, features, metadata
completeness, sites, topology, native operations, and timing data on the module
as the typed {code}`mqt.target_env` attribute. Mapping, synthesis, and
conformance passes read this snapshot from the IR, so textual pass pipelines do
not depend on hidden C++ state. Pass {code}`payload_descriptor` to
{py:meth}`~mqt.core.mlir.QCOProgram.compile_for_target` when the encoding or
another descriptor field differs from the compiler output's default.

The target pipeline normalizes before testing legality. SCCP and QCO cleanup
remove constant branches, switches, and structural regions such as
{code}`scf.execute_region`. If the selected profile lacks
{code}`COUNTED_ITERATION`, finite {code}`scf.for` loops are fully unrolled and
the result is cleaned again. Full unrolling fails before it would create more
than 65536 operations. Only a residual loop requires runtime counted iteration.
A residual {code}`qco.index_switch` lowers to nested {code}`qco.if` operations
when the payload supports forward branching but not multiway branching.

Legality starts at the operation marked {code}`mqt.entry_point` and follows
MLIR's call graph. Unsupported control in an unused helper does not reject the
program; control and computation in reachable helpers do.

Failed in-place target compilation is transactional: the pipeline runs on a copy
and replaces the original {code}`QCOProgram` only after every pass succeeds.

For QIR 2.1, the MLIR LLVM-dialect pipeline derives integer and floating-point
widths, helper functions, branch modes, return points, arrays, and dynamic
allocation. It validates these requirements against the selected payload before
LLVM translation. The translation boundary only serializes the derived QIR
string tuples and repairs the scalar module-flag widths required by QIR.

Targets created from QDMI preserve the distinction between unavailable feature
metadata and a reported empty list. QIR Adaptive adds its normative baseline to
the optional features reported for that exact descriptor. No feature leaks to
another version, profile, or encoding. QDMI target factories do not accept
caller-supplied feature augmentation.

Use {py:meth}`~mqt.core.mlir.QCOProgram.compile_for_target` to apply target
compilation to an existing QCO program. Pass its {code}`format` argument when
the intended payload is not optimized QCO. For pass-level benchmarking, the C++
API exposes separate factories for pre-routing optimization, mapping, native
synthesis, and conformance verification.

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

Target compilation produces optimized QCO, QC, OpenQASM 3, or QIR. It cannot be
combined with a custom `--passes` pipeline because the canonical target pipeline
owns the required pass ordering.

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
