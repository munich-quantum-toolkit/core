---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# Compile for a QDMI device

A device can execute only certain gates, often on specific pairs of qubits.
`compile_program` translates your program to those gates and assigns its qubits
to suitable device sites. It returns a
{py:class}`~mqt.core.mlir.CompiledProgram` containing the serialized program and
the information needed to interpret its results. You can submit that program
repeatedly with `submit_program`.

The example below uses the local DDSIM simulator. For a comparison of hardware
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
if not job.wait():
    raise RuntimeError("The job did not complete.")
print(f"Counts: {job.get_counts()}")
```

`target` accepts a device, registered device ID,
{py:class}`~mqt.core.mlir.CompilerTarget`,
{py:class}`~mqt.core.mlir.TargetEnvironment`, or Qiskit backend/target. Device,
`QDMIBackend`, and non-Qiskit device environment targets return a
`CompiledProgram`. Other Qiskit backends, targets, and Qiskit output
environments return a `QuantumCircuit`. An explicit `output` selects another
representation. For a `CompilerTarget` or a non-Qiskit gate-set environment, the
default preserves the input's program family.

Use `TargetEnvironment.from_qiskit(target, strict=True)` to reject applicable
gates or constraints MQT Core cannot represent. It imports control flow as
execution capabilities. For native synthesis, pass the circuit's width as
`native_num_qubits` to import a gate-set target and skip wider gates.

Discrete targets, including Clifford+T, accept circuits whose gates are already
native. Routing can lower SWAPs to CX gates and reverse CX direction when H is
supported on both sites. Gates that remain non-native after decomposition
require a supported synthesis basis. MQT Core does not approximate arbitrary
rotations with Clifford+T gates.

A list or tuple of one input kind returns results in input order. Typed MLIR
inputs are consumed unless `copy=True`; source text and Qiskit circuits remain
reusable. A failed batch compilation returns no results, but processed typed
inputs may already be consumed.

Before submission, MQT Core checks that the destination supports the sites,
operations, connectivity, and program features recorded during compilation. IQM
JSON also requires matching site names. A change in calibration values alone
does not invalidate the compiled program. To submit program text or bytes
directly, use `device.submit_job`.

See [placement and routing](#placement-and-routing) for site eligibility and
[resonator-mediated compilation](#resonator-mediated-compilation) for MOVE
routing on supported devices.

### Unused qubits

Device compilation removes unused qubits before checking whether the program
fits on the device. For custom pipelines, `remove-unused-qubits` removes unused
allocation/release pairs, and the QC and QTensor register-shrinking passes
remove unused slots. Canonicalization alone retains allocations. A gate-set
target retains the program's qubits because no device placement is requested.

### Choose a format

```{code-cell} ipython3
from mqt.core.mlir import TargetEnvironment
from mqt.core.qdmi import ProgramFormat

environment = TargetEnvironment.from_device(device, program_format=ProgramFormat.QASM3)
compiled = compile_program(bell_qasm, target=environment)
print(f"Selected format: {compiled.program_format}")
```

Device defaults prefer QPY when a supported Qiskit version is installed,
OpenQASM 3, Adaptive Profile QIR (binary, then text), IQM JSON, Base Profile QIR
(binary, then text), then OpenQASM 2. An environment selects an executable
device format and can be reused for compilation and serialization. `output`
selects an inspection or interchange representation. `QC_IMPORT` and raw `QCO`
stop before target compilation and cannot take a target. jeff output does not
retain hardware placement.

OpenQASM includes its standard gate library. Targeted output retains native
calls; nonstandard names require receiver definitions, and standard names must
keep their standard semantics. Provider packages adapt service syntax. The
exporter uses the lowest required language version and omits explicit `output`
designations when every eligible value is an output. See {doc}`OpenQASM` and
{doc}`../qir/index` for supported formats and their requirements.

### Terminal readout and results

Use {py:func}`~mqt.core.mlir.serialize_program` for an already-native Qiskit
circuit or `QCProgram`. It serializes without synthesis, routing, or quantum
optimization. The caller is responsible for native gate and placement validity;
the Qiskit backend checks these before native serialization.

For programs whose qubits already name physical sites, serialization arranges
final measurements in ascending site order in one register. The program can
measure a subset of sites, send measurements to different output positions,
repeat a measurement, or leave output positions at zero. MQT Core records how to
reconstruct those requested outputs from the device's measurement order.

Operations on unmeasured qubits are moved before the final measurements. A
native gate call cannot use a qubit after it has been measured. Calls to
functions with defined bodies are rejected by this readout conversion because
they may access physical sites that are not visible in the call's arguments.
Programs with dynamically allocated qubits retain their classical outputs,
subject to the selected exporter's restrictions. Provider libraries handle
service requirements such as simulator qubit declarations and measurement of
every qubit.

OpenQASM 3 uses plain classical declarations as implicit outputs unless a
non-output classical variable is emitted at global scope. In that case, output
qualifiers distinguish the returned values. Programs without outputs may expose
unused temporaries as implicit outputs.

When device measurement order differs from the program's output order, the
compiled program stores a {py:class}`~mqt.core.mlir.ResultLayout` describing the
mapping. QDMI jobs return raw device results, so apply this mapping before using
them. Keep the compiled program alongside the job; passing source directly to
`submit_program` is rejected when this mapping would be needed.

Decode counts or individual shots with the compiled program:

```python
from mqt.core.mlir import decode_counts, decode_shots

counts = decode_counts(job.get_counts(), compiled.result_width, 64, compiled.result_layout)
samples = decode_shots(job.get_shots(), compiled.result_width, 64, compiled.result_layout)
```

These helpers ignore ASCII register-separator spaces, validate binary digits,
width, and the requested shot total, and reconstruct logical output order.
`decode_counts` merges equal outcomes; `decode_shots` preserves shot
correlations. Qiskit's job adapter applies this decoding and register formatting
automatically. Retain the compiled program when persisting a job: its ID alone
cannot recover the frontend mapping.

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

Controlled-gate decomposition can borrow other allocated qubits as ancilla
qubits when this reduces estimated entangling cost. It restores their states and
correlations and allocates no additional qubits. See
`decompose-multi-controlled` in {doc}`QCO` for eligibility and target
restrictions.

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

Each operation capability describes a gate's qubit and parameter counts, the
ordered sites on which it can act, and any parameter restrictions or calibration
values. Represent alternative implementations as separate capabilities. Site IDs
identify hardware locations and need not be consecutive. Use
{py:class}`~mqt.core.mlir.TargetEnvironment` with a
{py:class}`~mqt.core.mlir.PayloadSpecification` to restrict the output format
and supported control flow explicitly. A QDMI format list alone does not
describe these restrictions; the adapter uses the compiler-supported format
baseline.

When a target is created from a QDMI device, it records the reported gate
placements and calibration. Some gates may be available only on particular
sites. Synthesis requires a common basis of single-qubit gates and a supported
entangling gate across the routing edges; the compiler also checks each emitted
operation at its assigned sites.

This interface supports circuit-model devices, but not neutral-atom zone models.
The bundled SC targets describe hardware for compilation and do not execute
programs; see {doc}`../qdmi/sc_device`.

QDMI target snapshots omit the unsupported IQM dynamic primitives `cc_prx` and
`measure_ff`.

Use `CompilerTarget.SiteTuple` to attach duration or fidelity to a placement;
plain tuples represent placements without calibration. Routing uses undirected
adjacency, and synthesis checks the supported operand directions on each site.
Branch exits and loop backedges must retain compatible physical sites.

### Native single-qubit gates

The
[GPi/GPi2 gate set](https://docs.ionq.com/features/getting-started-with-native-gates)
uses half and quarter turns about an equatorial axis. Core's `gpi` synthesis
option represents these using `R(π, φ)` and `R(π/2, φ)`, where `φ` is the axis
angle in radians. These rotation matrices differ from the native gate matrices
by

```{math}
\operatorname{GPi}(\phi) = i R(\pi, \phi), \qquad
\operatorname{GPi2}(\phi) = R(\pi/2, \phi).
```

Native OpenQASM export therefore emits `gphase(-π/2)` when replacing `R(π, φ)`
with `gpi(φ)`; `gpi2` needs no phase correction. Core's OpenQASM angles are in
radians. IonQ's API expresses the phase in turns, so the same axis angle there
is `φ / (2π)`. Check the destination interface's units when exchanging these
gates.

### Placement and routing

Placement assigns each qubit in the input program (a {term}`program qubit`) to a
hardware location. Routing then inserts operations that let interacting qubits
reach connected sites. The
{py:attr}`~mqt.core.mlir.CompilerTarget.placeable_sites` property lists
candidate sites: each must support a compiler-recognized single-qubit gate with
the expected parameter count. If the target advertises measurement, each
candidate must also support readout. Reset and barriers alone do not make a site
placeable. Targets with unrestricted native-operation support allow every site.

A candidate site need not support every operation in the circuit or be connected
to every other candidate. Target preparation can narrow this set to a connected
subset; synthesis and target-conformance checks enforce the required gate
support. When excluding sites from an explicit topology, preparation keeps the
largest connected component, including any
[virtual resonator connections](#resonator-mediated-compilation). Only the
retained sites count toward the available qubit capacity.

Mapping runs multiple initial-layout trials. Set `MappingOptions.trials` and
`CompilationOptions.seed` for reproducible choices. Each trial retains the
layout with the fewest native two-qubit gates among the routes it evaluates,
using depth to break ties. Each trial has a bounded search-memory budget,
configured with `MappingOptions.search_memory_limit`; this does not cap total
process memory. See {doc}`QCO` for routing and synthesis details.

### Custom pass sequences

Pass a `target_environment` to
{py:meth}`~mqt.core.mlir.QCOProgram.run_pass_pipeline` to use target-aware
passes in a custom sequence. The environment replaces any stored environment
after the pipeline parses successfully and remains available to later calls.
Omitting it, or passing `None`, keeps the stored environment.

To compare pass orders, prepare the source once and compile independent copies.
`mqt-target-prepare` records the input qubits before inlining and cleanup, so
the [layout](#layout-metadata) can include idle inputs when the allocation shape
is supported. For a device target, it limits placement to usable sites and adds
connections available through resonators, just as `compile_for_target` does.
Gate-set targets retain the allocated program qubits and need no placement.

End each sequence with `mqt-target-finalize`. It cleans up the program,
synthesizes native gates, checks the result, and restores the execution
environment saved during preparation. Copies and saved IR retain this
information. Supplying another environment replaces it, so start again from the
source when changing targets.

The following example compares two pass orders with the default compiler using a
fixed seed and routing trial count:

```{code-cell} ipython3
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from mqt.core.mlir import (
    CompilationOptions, CompilerTarget, MappingOptions, PayloadFormat,
    PayloadSpecification, QCOProgram, QCProgram, TargetEnvironment,
)

circuit = QuantumCircuit(4)  # Input qubit 0 stays idle.
circuit.h(1)
circuit.rx(0.3, 1)
circuit.cx(1, 3)
circuit.swap(1, 3)
search_target = CompilerTarget(
    4,
    connectivity=CompilerTarget.Connectivity([(0, 1), (1, 2), (2, 3)]),
    native_operations=CompilerTarget.NativeOperations([
        CompilerTarget.OperationCapability(name, arity, parameters)
        for name, arity, parameters in [
            ("u", 1, 3), ("cz", 2, 0), ("gphase", 0, 1), ("measure", 1, 0)
        ]
    ]),
)
search_environment = TargetEnvironment(
    search_target, PayloadSpecification(PayloadFormat("openqasm", "3.1"))
)
options = CompilationOptions(seed=7, mapping=MappingOptions(trials=1, iterations=1))
prepared = QCProgram.from_qiskit(circuit).to_qco()
baseline = prepared.copy()
baseline.compile_for_target(search_environment, options=options)
prepared.run_pass_pipeline("mqt-target-prepare", target_environment=search_environment)
checkpoint = prepared.ir
actions = (
    "combine-controlled-gates{use-target=true}",
    "decompose-multi-controlled{use-target=true}",
    "fuse-single-qubit-unitary-runs{use-target=true}",
    "fuse-two-qubit-gates{use-target=true}",
)
results = {"default": baseline}
for number, order in enumerate((actions, tuple(reversed(actions)))):
    candidate = QCOProgram.from_mlir_str(checkpoint)
    for action in (*order, "unroll-modifiers", "elide-terminal-swaps",
                   "place-and-route{ntrials=1 niterations=1}", "mqt-target-finalize"):
        candidate.run_pass_pipeline(action, options=options, preserve_layout=True)
    results[f"order {number}"] = candidate

for name, result in results.items():
    exported = result.to_qiskit(target=search_target, copy=True)
    assert exported.layout is not None
    assert np.allclose(Operator.from_circuit(exported).data, Operator(circuit).data)
    count = result.inspect().num_two_qubit_gates
    depth = exported.depth(filter_function=lambda instruction: len(instruction.qubits) == 2)
    print(f"{name}: {count} two-qubit gates, two-qubit depth {depth}")
```

The `use-target=true` option selects the target-aware form of each optimization;
omitting it keeps the target-independent behavior. For all-to-all device
targets, use `place-qubits` instead of `place-and-route`. For gate-set targets,
omit placement and terminal-SWAP elision to retain program wire order. Set
custom routing controls in the textual pass options;
`CompilationOptions.mapping` configures the default target compiler.
`unroll-modifiers` distributes composite gates before placement and native
synthesis. The runner executes only the requested passes, so each sequence must
meet its passes' input requirements.

The default `preserve_layout=False` clears existing layout metadata. Set
`preserve_layout=True` throughout the sequence above: terminal-SWAP elision and
mapping maintain the layout, while synthesis and verification preserve it. Use
this flag only when passes keep wire and site identities or update the layout
themselves. The flag does not create or validate a mapping; qubit reuse, wire
reordering, or retargeting must not leave stale metadata.

Use `copy()` or an IR checkpoint before trying another sequence: a failed pass
can leave the supplied environment and partial changes on the program. A
pipeline parsing error leaves the program unchanged. Native inspection counts
are static; Qiskit depth and the unitary check above apply to flat circuits.

### Gate-set synthesis

Create a target with {py:meth}`~mqt.core.mlir.CompilerTarget.for_gate_set` to
synthesize a program into native gates without device placement:

```python
gates = CompilerTarget.NativeOperations.from_qiskit(qiskit_target)
target = CompilerTarget.for_gate_set(gates)
native = compile_program(circuit, target=target)
```

The result preserves program-qubit order, terminal permutations, and idle
qubits. Qiskit output has no layout metadata; OpenQASM output declares program
qubits. The default output follows the input format family; an environment with
the Qiskit format selects Qiskit output. Each circuit in a sequence retains its
own width. Inputs must use fixed-width qubit allocations in the entry block,
without static device-site references or existing layout metadata. Manually
constructed capabilities with site-specific placements are rejected.

### Synthesis without routing

{py:meth}`~mqt.core.mlir.QCOProgram.synthesize_for_target` translates to native
gates. Gate-set targets preserve the program qubit interface as described above.
For device targets, allocated program qubits require all-to-all connectivity and
are assigned target sites. Qubits referenced by static site IDs retain those
IDs. This method does not require measurement support. It checks target support
and topology; it mutates the program, including on failure, so copy first when
needed.

### Control flow supported by the output format

Target compilation accepts structured QCO/SCF input. It folds constants, expands
finite loops when placement or the selected execution format requires it, and
checks remaining branches and loops against the selected execution capabilities.
Unsupported runtime-dependent behavior fails compilation. See
`legalize-control-flow` and `unroll-loops-for-payload` in {doc}`QCO`, and
{py:class}`~mqt.core.mlir.ProgramCapability` /
{py:class}`~mqt.core.mlir.ProgramConstraint` for explicit restrictions.

## Resonator-mediated compilation

A {term}`computational resonator` can store a device qubit's state temporarily
and mediate gates between device qubits. Supported IQM devices expose an ordered
`move(qubit, resonator)` operation to transfer the state into the resonator and
back. The resonator is a target site, but it is excluded from program-qubit
placement because it lacks the required single-qubit gates and readout.

Target preparation combines native MOVE and CZ placements to expose virtual CZ
connections between placeable sites. For example, `move(a, r)` and `cz(r, b)`
permit a virtual `cz(a, b)` through an idle resonator `r`. The native circuit
contains the following sequence, which returns the state to the original qubit:

```text
move(a, r); cz(r, b); move(a, r)
```

The native CZ operand order follows the target's advertised placement. Each
sequence returns the transferred state to its device qubit, and terminal readout
follows the MOVE sequences. MQT Core inserts these sequences automatically when
producing a native circuit, including after custom target stages. QCO and jeff
compiler checkpoints retain virtual CZs.

Use `compile_program(circuit, target=device)` with `submit_program`,
`backend.run(backend.compile(qc))` through Qiskit, or the PennyLane device.
Qiskit `transpile` does not support these targets. Native circuits that already
contain MOVE use `serialize_program`; native serialization does not route. See
[IQM JSON circuits](#iqm-json-circuits) for the serialization requirements. IQM
execution requires its default
[full MOVE frame tracking](https://docs.iqm.tech/iqm-client/api/iqm.iqm_client.models.CircuitCompilationOptions.html)
to correct MOVE frames.

## IQM JSON circuits

Select `OutputFormat.IQM_JSON` for an inspectable circuit or
`ProgramFormat.IQM_JSON` for a compiled submission. The target supplies physical
site names. The exporter supports constant native PRX, CZ and MOVE calls and
terminal measurement records; it discards global phase. MOVE operands preserve
the advertised qubit/resonator order. Native serialization does not route or
synthesize MOVE; the IQM server checks MOVE pairing rules. Unsupported reset,
feedback or classical result computation fails before submission.

Direct IQM JSON export requires one measurement per output position. Compiling
through `CompiledProgram` also supports output positions that stay zero and
measurements returned in a different order, using the result mapping described
[above](#terminal-readout-and-results). QDMI-on-IQM handles authentication,
communication with the service, calibration selection, and conversion of its
measurement columns to QDMI bitstrings.

## Command line from a source build

```console
mqt-core compile --qdmi-list-devices
mqt-core compile --qdmi-device=mqt.sc.iqm.garnet \
  --payload-spec='#mqt.payload_spec<format = <id = "iqm-json", version = "", profile = "", encoding = text>, capabilities = [], optional_capabilities_known = false>' \
  -o circuit.json input.qasm
```

`--qdmi-config` selects a registry file. The format specification selects
format, encoding, and execution capabilities; its encoding takes precedence over
the filename extension. Target compilation owns the pass ordering and rejects
custom `--pass-pipeline` pipelines.

## C++ source-tree API

Program factories and the default compiler pipeline return `llvm::Expected<T>`;
text-program writes return `llvm::Error`. These results own their error details.
MLIR transformations on an existing program retain `mlir::FailureOr<T>` or
`mlir::LogicalResult` and report diagnostics through that program's context.

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

## Qiskit export

Qiskit output for a device uses physical qubit indices. By default, Core
attaches available placement and routing information, including unused sites, as
a `TranspileLayout`. Apply the layout when remapping observables. See
[transpiler layouts](qiskit.md#transpiler-layouts) for layout options.

For lower-level export with `to_qiskit(target=compiler_target)`, static site IDs
map to positions in `compiler_target.sites`. The result has one `q` register
with `compiler_target.num_sites` qubits. Passing the recorded target is required
to reconstruct its layout. Target-free export keeps static IDs but omits layout
metadata.

## Layout metadata

When compiling allocated program qubits for a device, Core attaches placement
and routing metadata if the input has fixed-size entry-block allocations.
Programs with static device-site references already name physical sites;
compilation preserves those IDs and does not attach a layout. Static device-site
references and allocated program qubits cannot be mixed. Native target
compilation rejects an already attached layout. See
[custom pass sequences](#custom-pass-sequences) for layout preservation between
passes and [transpiler layouts](qiskit.md#transpiler-layouts) for the import and
export requirements.

## Metadata for device implementers

The following extensions let a device library report restrictions that the
compiler needs when preparing programs. They are MQT-specific conventions using
QDMI custom properties.

### Temporary QDMI parameter metadata

Providers can report parameter constraints as a NUL-terminated UTF-8 JSON string
in `QDMI_OPERATION_PROPERTY_CUSTOM2`, queried without sites or parameter values.
This versioned metadata format is temporary and is
**not part of the QDMI standard**. MQT Core reads it for native compiler targets
and the generic Qiskit backend.

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
recognized schema is rejected before compilation or submission. Standardization
of parameter constraints is tracked in
[QDMI issue #601](https://github.com/Munich-Quantum-Software-Stack/QDMI/issues/601).

### Temporary QDMI execution capabilities

`QDMI_DEVICE_PROPERTY_CUSTOM3` can supply NUL-terminated UTF-8 JSON with schema
`mqt.payload-capabilities.v1` and a `capabilities` array. Each entry contains an
`id`, optional unsigned `value` (default zero), and an optional `constraints`
object mapping constraint IDs to unsigned limits. This temporary metadata format
is not part of the QDMI standard. The reported list restricts the selected
format's capabilities; absent capabilities are unavailable. Without recognized
metadata, control flow and classical assignment are disabled. Unknown schemas
are ignored, and malformed recognized metadata is rejected. See the
[temporary property slots](../qdmi/driver.md#temporary-compiler-metadata).

For example, a device supporting only unnested register comparisons can report:

```json
{
  "schema": "mqt.payload-capabilities.v1",
  "capabilities": [{
    "id": "forward-branching",
    "constraints": {
      "max-control-flow-nesting-depth": 1,
      "max-condition-width": 32,
      "requires-register-equality": 1,
      "unitary-only-branches": 1
    }
  }]
}
```

`max-condition-width` bounds integer widths used by the condition.
`requires-register-equality` and `unitary-only-branches` accept the value `1`;
they require comparisons of a classical register or bit with a constant and
exclude measurements, resets, and classical writes inside branches. These
constraints apply to compilation and native serialization, including QPY.
