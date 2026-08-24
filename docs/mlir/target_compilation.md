# Compile for a QDMI device

An MLIR {code}`mlir::CompilerTarget` is an immutable snapshot of a circuit-model
device. It contains the device sites, topology, native operations, and available
calibration and ordered-applicability data. Compilation decomposes supported
multi-qubit operations, optimizes and maps the program, synthesizes native
gates, and verifies that the result conforms to the target.

The snapshot is independent of its originating QDMI session. It can therefore be
stored, copied cheaply, and reused for multiple compilations.

## Python

Open a configured QDMI device and snapshot it as a compiler target:

```python
from mqt.core.mlir import (
    CompilerTarget,
    PayloadFormat,
    PayloadEncoding,
    PayloadSpecification,
    TargetEnvironment,
    compile_program,
)

target = CompilerTarget.from_device_id("mqt.sc.iqm.garnet")
payload = PayloadSpecification(PayloadFormat("qir", "2.1.0", "base", PayloadEncoding.BINARY))
environment = TargetEnvironment(target, payload)
compiled = compile_program(
    "bell.qasm",
    target_environment=environment,
)
```

The payload specification identifies the exact representation selected for the
device. MQT Core derives the compiler output from that specification and uses
the canonical QCO pipeline. The targeted overload therefore accepts one
`TargetEnvironment` and no independent output or custom pipeline. MQT Core's
QDMI adapter does not yet translate QDMI program-format and feature metadata, so
callers must construct the payload specification from the device documentation.

The example has no reported execution capabilities. A producer must add every
effective capability, including the selected format's baseline. Set
`optional_capabilities_known=True` only when the producer also knows that the
list contains every optional device capability.

### Payload control flow

Target compilation removes unused symbols, lifts reducible ControlFlow dialect
graphs to SCF, and propagates constants. It checks constant loop ranges with
widened arithmetic before generic canonicalization, unrolls unsupported static
loops, and then runs the standard QCO cleanup pipeline. It applies these
structural capabilities to the remaining control flow:

| Capability           | Residual operations                                 |
| -------------------- | --------------------------------------------------- |
| `forward-branching`  | `qco.if` and classical `scf.if`                     |
| `counted-iteration`  | `scf.for`                                           |
| `conditional-loop`   | `scf.while`                                         |
| `multiway-branching` | `qco.index_switch` and classical `scf.index_switch` |

A finite `scf.for` that exceeds the selected counted-iteration contract is fully
unrolled when this clones at most 65,536 body operations. Cleanup runs again
because unrolling can make nested bounds and conditions constant. An unsupported
index switch is lowered to nested forward branches when that form fits the
selected contract. Generic SCF branches cannot capture or return QCO qubits or
quantum tensors; use the corresponding QCO branch operation for linear quantum
state. SCF loops must carry linear quantum state through their iteration
arguments instead of capturing it.

The supported constraints are `max-control-flow-nesting-depth` on all four
capabilities, `max-iteration-count` on both iteration capabilities, and
`max-case-count` on multiway branching. Limits are inclusive. The compiler must
prove a constrained loop's trip count. It currently proves constant `scf.for`
bounds and rejects a constrained `scf.while` because no general termination
bound is available. The compiler rejects a constant range when MLIR's native
trip-count result disagrees with widened arithmetic. A zero, unknown, or
misapplied constraint makes that capability group unusable. Missing or
incomplete optional metadata never implies support.

This stage checks structural control flow only. Later lowering stages remain
responsible for scalar types and operations, measurement provenance, function
features, allocation, and final payload-profile conformance.

The target can also be constructed directly. Connectivity and native-operation
support are required:

```python
target = CompilerTarget(
    3,
    connectivity=CompilerTarget.Connectivity([(0, 1), (1, 2)]),
    native_operations=CompilerTarget.NativeOperations([
        CompilerTarget.Operation(
            "gphase",
            arity=CompilerTarget.OperationArity.fixed(0),
            num_parameters=1,
        ),
        CompilerTarget.Operation("u", arity=1, num_parameters=3),
        CompilerTarget.Operation(
            "cx",
            arity=2,
            num_parameters=0,
            site_tuples=[(1, 0), (1, 2)],
        ),
        CompilerTarget.Operation("measure", arity=1, num_parameters=0),
        CompilerTarget.Operation("reset", arity=1, num_parameters=0),
    ]),
)
```

Use `CompilerTarget.Connectivity.all_to_all()` for an all-to-all target. An
empty `CompilerTarget.NativeOperations([])` reports that no quantum operation is
native. It can be used with passes that need only topology, but target
compilation cannot lower quantum operations without a synthesis basis. Use
`CompilerTarget.NativeOperations.unrestricted()` only when the target accepts
every operation. Creating a target from a QDMI device fails if the device does
not provide a complete connectivity model and a representable native-operation
set. An explicit operation arity is either fixed or variadic with a positive,
inclusive minimum. Fixed zero represents a global-phase operation. A variadic
capability accepts every total width from its minimum through the target's site
count; site tuples are therefore available only for fixed, positive arities. An
empty `site_tuples` list makes an operation available on every valid placement.
A nonempty list contains all supported ordered placements. Each tuple may carry
calibration values; omitted values inherit the operation-wide defaults. Retain
placements without calibration in this list, and omit operations that are not
available anywhere. Structural and program-format constructs are not
compiler-target operations.

Use plain tuples for placements without calibration. Use
`CompilerTarget.SiteTuple([1, 0], duration=40, fidelity=0.99)` to attach
calibration to a placement; both forms can appear in the same list.

Routing uses undirected adjacency; native synthesis repairs unsupported operand
directions. Target compilation requires a known static physical site for each
qubit. Structured branch exits must agree on sites, and loop backedges must
preserve the entry sites. Unsupported or inconsistent site transfers are
diagnosed, including after all-to-all placement. A synthesis basis must provide
the same one-qubit gate family on every site and an entangler on every routing
edge in at least one direction.

Target synthesis preserves a native `gphase`. If the target does not support
`gphase`, target synthesis preserves relative phase effects and removes only the
unobservable global phase of the entry point.

Use {py:meth}`~mqt.core.mlir.QCOProgram.compile_for_target` with the target
environment to apply target compilation to an existing QCO program. Compilation
runs in place. If a pass fails, the environment and earlier pass changes remain
on the program. Copy the program before compilation if the caller must preserve
the input. The target passes read the typed `mqt.target_env` module attribute.
The mapping, native-synthesis, and conformance factories also work in textual
MLIR pass pipelines. Target compilation keeps deterministic placement on
all-to-all targets and uses mapping only for explicit topology.

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
  --payload-spec='#mqt.payload_spec<format = <id = "qir", version = "2.1.0", profile = "base", encoding = binary>, capabilities = [], optional_capabilities_known = false>' \
  -o output.bc input.qasm
```

An explicit registry file can be selected before device discovery:

```console
mqt-cc --qdmi-config=/path/to/qdmi.json \
  --qdmi-device=example.device \
  --payload-spec='#mqt.payload_spec<format = <id = "qir", version = "2.1.0", profile = "base", encoding = binary>, capabilities = [], optional_capabilities_known = false>' \
  input.qasm
```

The payload specification selects the emitted format and encoding. For targeted
QIR, the selected encoding takes precedence over the output filename extension.
Target compilation rejects `--emit` and custom `--passes` pipelines because the
target contract owns the output and required pass ordering.

## C++ source-tree API

The source build provides a narrow, non-throwing QDMI bridge between a stable
device ID and the compiler-owned target:

```cpp
#include "mlir/Compiler/QDMIAdapter.h"
#include "mlir/Compiler/Programs.h"
#include "mlir/Compiler/TargetEnvironment.h"
#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>

auto target = mlir::compilerTargetFromDeviceId("mqt.sc.iqm.garnet");
if (!target) {
  llvm::errs() << "Failed to create compiler target: "
               << llvm::toString(target.takeError()) << '\n';
  return 1;
}

auto payload = mlir::PayloadSpecification::create({
    .id = "qir",
    .version = "2.1.0",
    .profile = "base",
    .encoding = mlir::PayloadEncoding::Binary,
});
if (!payload) {
  llvm::errs() << llvm::toString(payload.takeError()) << '\n';
  return 1;
}
mlir::TargetEnvironment environment(*target, *payload);

auto qc = mlir::QCProgram::fromQASMFile("input.qasm");
if (!qc) {
  return 1;
}
auto qco = std::move(*qc).intoQCO();
if (!qco || !qco->compileForTarget(environment)) {
  return 1;
}
```

The adapter accepts circuit-model devices whose two-qubit operations cover every
topology edge in at least one operand orientation and preserves the exact
ordered tuples reported by the device. Operations with arity above two must
report every ordered tuple of distinct sites. Neutral-atom zone models require a
different compilation model and are rejected with a diagnostic.

QDMI 1.3 cannot report an operation-arity range. The bundled DDSIM device uses
an exact, versioned custom-operation marker to state that each canonical
standard gate with one or more targets accepts arbitrary positive controls. The
adapter turns such a base gate into a variadic capability whose minimum is the
base gate's target count. For example, DDSIM reports `h` with minimum one, `rxx`
with minimum two, and `rccx` with minimum three; each also accepts any
additional number of controls up to the simulator's site count. Controlled
aliases such as `mcx` and `mcp` are not enumerated as compiler capabilities.
This private bridge can be removed when QDMI standardizes equivalent metadata.

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
named {code}`q` with {py:attr}`~mqt.core.mlir.CompilerTarget.num_sites` qubits.
This option does not run target compilation or emit Qiskit layout metadata.
Target-aware export requires static qubits whose site IDs belong to that target.
