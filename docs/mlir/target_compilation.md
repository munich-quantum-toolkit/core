---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# Compile for a QDMI device

Compile before submission: `compile_program` maps a program to native gates and
device sites; `submit_program` can then reuse the resulting
{py:class}`~mqt.core.mlir.CompiledProgram`. For a worked comparison of hardware
targets, see the
{doc}`hardware-compilation tutorial <../tutorials/hardware_compilation>`.

## Python

```{code-cell} ipython3
from mqt.core.mlir import compile_program, submit_program
from mqt.core.qdmi import open_device

bell_qasm = """OPENQASM 3;
include "stdgates.inc";
qubit[2] q;
bit[2] result;
h q[0];
cx q[0], q[1];
result = measure q;
"""
device = open_device("mqt.ddsim.default")
compiled = compile_program(bell_qasm, target=device)
job = submit_program(compiled, target=device, num_shots=64)
assert job.wait()
assert sum(job.get_counts().values()) == 64
```

`target` accepts a device, registered device ID, `CompilerTarget`, or Qiskit
backend/target. Device targets return a `CompiledProgram`; Qiskit targets return
a `QuantumCircuit`. An explicit `output` selects another representation. For a
`CompilerTarget`, the default preserves the input's program family.

A list or tuple of one input kind returns results in input order. Typed MLIR
inputs are consumed unless `copy=True`; source text and Qiskit circuits remain
reusable. A failed batch compilation returns no results, but processed typed
inputs may already be consumed.

Submission checks the compiled device contract, including sites, operations,
topology, and payload capabilities. IQM JSON also checks site names. Changes to
calibration values alone do not invalidate a compiled program. Raw payloads can
be submitted with `device.submit_job`.

### Choose a format

```{code-cell} ipython3
from mqt.core.qdmi import ProgramFormat

compiled = compile_program(bell_qasm, target=device, program_format=ProgramFormat.QASM3)
assert submit_program(compiled, target=device, num_shots=16).wait()
```

Device defaults prefer Adaptive QIR (binary, then text), OpenQASM 3, IQM JSON,
then Base QIR (binary, then text). `program_format` selects an executable device
format; `output` selects an inspection or interchange representation.
`QC_IMPORT` and raw `QCO` stop before target compilation and cannot take a
target. Jeff output does not retain hardware placement.

OpenQASM includes its standard gate library. Targeted output retains native
calls; nonstandard names require receiver definitions, and standard names must
keep their standard semantics. Provider packages adapt service syntax. The
exporter uses the lowest required language version and omits explicit `output`
designations when every eligible value is an output. See {doc}`OpenQASM` and
{doc}`../qir/index` for format contracts.

### Terminal readout and results

Use {py:func}`~mqt.core.mlir.serialize_program` for an already-native Qiskit
circuit or `QCProgram`. It serializes without synthesis, routing, or quantum
optimization. The caller is responsible for native gate and placement validity;
the Qiskit backend checks these before its built-in serializers run.

Some receivers return measurements in physical-site order. For those receivers,
`terminal_readout=TerminalReadout.MEASURED` emits one measurement for each
selected site, in ascending site order. `TerminalReadout.ALL` measures every
addressed site, including gaps. `physical_qubits=False` emits virtual OpenQASM
indices instead of physical `$N` references. Provider packages choose these
generic controls and handle their service syntax.

Terminal normalization accepts flat circuits with direct measurement stores and
untouched zero-filled slots. It supports measured subsets and destination
permutations, coalesces repeated terminal measurements, and retains the final
write to each classical slot. Feedback, gates on already measured qubits, and
classical computation are rejected. General function calls must precede
measurements. With `terminal_readout=None` (the default), the chosen format
retains its ordinary classical and control-flow semantics.

The compiled program retains a {py:class}`~mqt.core.mlir.ResultLayout` when raw
measurement order differs from logical output order. QDMI jobs expose raw
results; direct callers apply this mapping:

```python
from collections import Counter

counts = Counter()
layout = compiled.result_layout
for bits, count in job.get_counts().items():
    counts[layout.decode_bits(bits) if layout is not None else bits] += count
```

`decode_bits` checks binary digits and raw width. Map individual samples to
preserve shot correlations. Qiskit's job adapter does this automatically,
including register formatting and aggregation of equal projected outcomes.
Retain the compiled program when persisting a job: its ID alone cannot recover
the frontend mapping.

### Compilation options

```python
from mqt.core.mlir import CompilationOptions, MappingOptions

options = CompilationOptions(seed=7, mapping=MappingOptions(trials=4, iterations=2))
compiled = compile_program(bell_qasm, target=device, options=options)
```

{py:class}`~mqt.core.mlir.CompilationOptions` owns randomness, timing,
statistics, and mapping controls. Repeatable mapping requires the same build,
input, target, seed, and explicit trial count. Sampling uses a separate seed.
For routing costs, search limits, and synthesis behavior, see the
`place-and-route` pass in {doc}`QCO`.

### Define a target

A {py:class}`~mqt.core.mlir.CompilerTarget` describes hardware without
connecting to it. This example has three physical sites and two ordered CNOT
placements:

```{code-cell} ipython3
from mqt.core.mlir import CompilerTarget, OutputFormat

target = CompilerTarget(
    3,
    connectivity=CompilerTarget.Connectivity([(0, 1), (1, 2)]),
    native_operations=CompilerTarget.NativeOperations([
        CompilerTarget.OperationCapability("u", 1, 3),
        CompilerTarget.OperationCapability("cx", 2, 0, site_tuples=[(1, 0), (1, 2)]),
        CompilerTarget.OperationCapability("measure", 1, 0),
    ]),
)
mapped = compile_program(bell_qasm, target=target, output=OutputFormat.QC)
```

Capabilities describe arity, parameter count, ordered placements, optional fixed
parameters and inclusive bounds, duration, and fidelity. Alternatives are
separate capabilities. Site IDs name physical hardware; they need not be dense.
Use {py:class}`~mqt.core.mlir.TargetEnvironment` with a
{py:class}`~mqt.core.mlir.PayloadSpecification` to restrict the payload's format
and control flow explicitly. A QDMI format list alone does not describe these
restrictions; the adapter uses the compiler-supported format baseline.

QDMI target snapshots preserve the exact ordered operation placements and
calibration reported by circuit-model devices. Additional operations may be
available on only some sites. Native synthesis checks its required common basis
separately, including entangler coverage across topology edges. Target
conformance checks each emitted operation at its assigned sites. Neutral-atom
zone models are not supported. The bundled SC snapshots are compilation models,
not executing devices; see {doc}`../qdmi/sc_device`.

### Synthesis without routing

{py:meth}`~mqt.core.mlir.QCOProgram.synthesize_for_target` translates to native
gates while preserving static site IDs. Dynamic qubits require all-to-all
connectivity. The method checks target support and topology; it mutates the
program, including on failure, so copy first when needed.

### Payload control flow

Target compilation accepts structured QCO/SCF input. It folds constants, expands
finite loops when placement or the selected payload requires it, and checks
remaining branches and loops against the payload contract. Unsupported dynamic
behavior fails compilation. See `legalize-control-flow` and
`unroll-loops-for-payload` in {doc}`QCO`, and
{py:class}`~mqt.core.mlir.ProgramCapability` /
{py:class}`~mqt.core.mlir.ProgramConstraint` for explicit restrictions.

## IQM JSON circuits

Select `OutputFormat.IQM_JSON` for an inspectable circuit or
`ProgramFormat.IQM_JSON` for a compiled submission. The target supplies physical
site names. The exporter supports constant native PRX/CZ calls and terminal
measurement records; it discards global phase. Unsupported reset, feedback,
MOVE, or classical result computation fails before submission.

Raw IQM JSON export requires each output slot to be measured once. Optional
terminal normalization also handles logical zero holes and permutations, using
the retained result layout. QDMI-on-IQM owns service transport, authentication,
calibration selection, and conversion of measurement columns to QDMI bitstrings.

## Command line from a source build

```console
mqt-cc --qdmi-list-devices
mqt-cc --qdmi-device=mqt.sc.iqm.garnet \
  --payload-spec='#mqt.payload_spec<format = <id = "iqm-json", version = "", profile = "", encoding = text>, capabilities = [], optional_capabilities_known = false>' \
  -o circuit.json input.qasm
```

`--qdmi-config` selects a registry file. The payload specification selects
format, encoding, and execution capabilities; its encoding takes precedence over
the filename extension. Target compilation owns the pass ordering and rejects
custom `--pass-pipeline` pipelines.

## C++ source-tree API

Program factories and the coordinated compiler pipeline return
`llvm::Expected<T>`; text-program writes return `llvm::Error`. These results own
their error details. MLIR transformations on an existing program retain
`mlir::FailureOr<T>` or `mlir::LogicalResult` and report diagnostics through
that program's context.

Compile a file and submit it to DDSIM:

```cpp
#include "mqt/Compiler/Programs.h"
#include "mqt/Compiler/QDMIAdapter.h"
#include "qdmi/QDMI.hpp"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <utility>

int main() {
  auto device = qdmi::Session::openDevice("mqt.ddsim.default");
  if (!device) {
    llvm::errs() << llvm::toString(device.takeError()) << '\n';
    return 1;
  }
  auto input = mlir::QCProgram::fromOpenQASMFile("input.qasm");
  if (!input) {
    llvm::errs() << llvm::toString(input.takeError()) << '\n';
    return 1;
  }
  auto compiled = mlir::compileProgram(std::move(*input), *device);
  if (!compiled) {
    llvm::errs() << llvm::toString(compiled.takeError()) << '\n';
    return 1;
  }
  auto job = mlir::submitProgram(*device, *compiled);
  if (!job) {
    llvm::errs() << llvm::toString(job.takeError()) << '\n';
    return 1;
  }
  auto completed = job->wait();
  if (!completed) {
    llvm::errs() << llvm::toString(completed.takeError()) << '\n';
    return 1;
  }
  return *completed ? 0 : 1;
}
```

`compilerTargetFromDevice` and `compilerTargetFromDeviceId` provide hardware
snapshots for staged compilation with a `TargetEnvironment`.

### Temporary QDMI parameter contract

Providers can report parameter constraints as a NUL-terminated UTF-8 JSON string
in `QDMI_OPERATION_PROPERTY_CUSTOM2`, queried without sites or parameter values.
This versioned contract is temporary and is **not part of the QDMI standard**.
Core reads it for native compiler targets and the generic Qiskit backend.

For example, an RX operation restricted to two angles reports:

```json
{
  "schema": "mqt.operation-parameters.v1",
  "alternatives": [
    { "fixed": [1.5707963267948966] },
    { "fixed": [-1.5707963267948966] }
  ]
}
```

A bounded RZZ operation instead uses
`"alternatives": [{"bounds": [[0, 1.5707963267948966]]}]` with the same schema.
Alternatives form a union; `fixed` and `bounds` within one alternative both
apply. Each supplied array has exactly `PARAMETERSNUM` entries in the
operation's parameter order. `null` leaves that parameter unconstrained, and an
omitted array imposes no restriction. Numeric values must be finite and use the
native operation's units (radians for RX and RZZ). Bounds are inclusive, ordered
pairs; a fixed value must also satisfy its bounds. The constraints apply to
every reported placement and do not express relations between parameters.

Absent properties and unknown schemas are ignored. Malformed metadata with the
recognized schema is rejected before compilation or submission. Replace this
custom property when QDMI provides a standard parameter-domain contract.

## Qiskit export

`compile_program(circuit, target=...)` returns physical device indices and
preserves the full Qiskit `TranspileLayout`, including routing and unused sites.
Apply that layout when remapping observables.

For lower-level export with `to_qiskit(target=compiler_target)`, static site IDs
map to positions in `compiler_target.sites`. The result has one `q` register
with `compiler_target.num_sites` qubits. Passing the recorded target is required
to reconstruct its layout. Target-free export keeps static IDs but omits layout
metadata.

## Layout metadata

Dynamic target compilation attaches placement and routing metadata when the
input has fixed-size entry-block allocations. Static programs already name
physical sites; compilation preserves those IDs and does not attach a layout.
Static and dynamic qubits cannot be mixed. Compilation rejects an already
attached layout; later transformations clear it. See
[transpiler layouts](qiskit.md#transpiler-layouts) for the full contract.
