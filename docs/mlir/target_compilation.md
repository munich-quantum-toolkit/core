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
if not job.wait():
    raise RuntimeError("The job did not complete.")
print(f"Counts: {job.get_counts()}")
```

`target` accepts a device, registered device ID,
{py:class}`~mqt.core.mlir.CompilerTarget`,
{py:class}`~mqt.core.mlir.TargetEnvironment`, or Qiskit backend/target. Device,
`QDMIBackend`, and non-Qiskit device environment targets return a
`CompiledProgram`. Other Qiskit backends, targets, and Qiskit payload
environments return a `QuantumCircuit`. An explicit `output` selects another
representation. For a `CompilerTarget` or a non-Qiskit gate-set environment, the
default preserves the input's program family.

Use `TargetEnvironment.from_qiskit(target, strict=True)` to reject applicable
gates or constraints MQT Core cannot represent. It imports control flow as
payload capabilities. For native synthesis, pass the circuit's width as
`native_num_qubits` to import a gate-set target and skip wider gates.

A list or tuple of one input kind returns results in input order. Typed MLIR
inputs are consumed unless `copy=True`; source text and Qiskit circuits remain
reusable. A failed batch compilation returns no results, but processed typed
inputs may already be consumed.

Submission checks the compiled device contract, including sites, operations,
topology, and payload capabilities. IQM JSON also checks site names. Changes to
calibration values alone do not invalidate a compiled program. Raw payloads can
be submitted with `device.submit_job`.

QDMI target snapshots omit the unsupported IQM dynamic primitives `cc_prx` and
`measure_ff`. Native GPI/GPI2 capabilities use fixed-angle R semantics
internally and export as one-parameter OpenQASM pulses with their phase
correction.

Target compilation places qubits only on sites with a supported single-qubit
gate. If the target supports measurement, these sites must also support readout.
With explicit couplings, it uses their largest connected component, including
connections through resonators. Targets with ordered `move(qubit, resonator)`
placements can route CZ gates through idle resonators. MQT Core emits closed
`move; cz; move` sandwiches before readout. Use
`compile_program(circuit, target=device)` with `submit_program`,
`backend.run(backend.compile(qc))` through Qiskit, or the PennyLane device.
Qiskit `transpile` does not support these targets. Native circuits that already
contain MOVE use `serialize_program` instead. QCO and Jeff compiler checkpoints
retain virtual CZs; MOVE insertion happens when producing QC and executable
payloads. The command-line target pass pipeline does not provide this placement
reduction or MOVE insertion.

IQM execution requires its default
[full MOVE frame tracking](https://docs.iqm.tech/iqm-client/api/iqm.iqm_client.models.CircuitCompilationOptions.html)
to correct MOVE frames.

### Unused qubits

Canonicalization retains owned qubit allocations. The `remove-unused-qubits`
pass removes unused allocation/release pairs; QC and QTensor register-shrinking
passes remove unused register slots. These transformations retain program qubits
when a gate-set target is attached. Device compilation performs this cleanup
before placement checks device capacity.

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
{doc}`../qir/index` for format contracts.

### Terminal readout and results

Use {py:func}`~mqt.core.mlir.serialize_program` for an already-native Qiskit
circuit or `QCProgram`. It serializes without synthesis, routing, or quantum
optimization. The caller is responsible for native gate and placement validity;
the Qiskit backend checks these before native serialization.

Static programs use canonical terminal readout: one register with measurements
in ascending physical-site order. This supports measured subsets, destination
permutations, repeated terminal measurements, and zero-filled output slots.
Gates and bodyless native calls on unmeasured qubits are scheduled before
terminal measurements. A native call cannot reuse a measured qubit. Calls to
defined functions decline terminal normalization because they can access hidden
static sites. Dynamic programs retain their classical outputs; the selected
exporter checks whether the receiver supports them. Provider transports handle
service-specific requirements such as simulator qubit declarations and all-qubit
measurement.

OpenQASM 3 uses plain classical declarations as implicit outputs unless a
non-output classical variable is emitted at global scope. In that case, output
qualifiers distinguish the returned values. Programs without outputs may expose
unused temporaries as implicit outputs; no synthetic conditional scope is added.

The compiled program retains a {py:class}`~mqt.core.mlir.ResultLayout` when raw
measurement order differs from logical output order. QDMI jobs expose raw
results; direct callers apply this mapping. Source-only `submit_program` rejects
programs requiring a layout: compile explicitly and retain the artifact instead.

Direct decoding:

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

Use `CompilerTarget.SiteTuple` to attach duration or fidelity to a placement;
plain tuples represent placements without calibration. Routing uses undirected
adjacency, and synthesis checks the supported operand directions on each site.
Branch exits and loop backedges must retain compatible physical sites.

Mapping runs multiple initial-layout trials. Set `MappingOptions.trials` and
`CompilationOptions.seed` for reproducible choices. Each trial retains the
layout with the fewest native two-qubit gates across its refinement and final
routes, using depth to break ties, without adding routing passes. Each trial has
a bounded search-memory budget, configured with
`MappingOptions.search_memory_limit`; this does not cap total process memory.
See {doc}`QCO` for routing and synthesis details.

### Custom pass sequences

Pass a `target_environment` to
{py:meth}`~mqt.core.mlir.QCOProgram.run_pass_pipeline` to use target-aware
passes in a custom sequence. The environment replaces any stored environment
after the pipeline parses successfully and remains available to later calls.
Omitting it, or passing `None`, keeps the stored environment.

Using the target defined above:

```{code-cell} ipython3
from mqt.core.mlir import CompilationOptions, PayloadFormat, PayloadSpecification, QCProgram, TargetEnvironment

environment = TargetEnvironment(target, PayloadSpecification(PayloadFormat("openqasm", "3.1")))
source = QCProgram.from_openqasm_str(bell_qasm).to_qco()
candidate = source.copy()
candidate.cleanup()
candidate.run_pass_pipeline(
    "place-and-route{ntrials=1},target-native-synthesis,drop-global-phases",
    target_environment=environment,
    options=CompilationOptions(seed=7),
)
candidate.run_pass_pipeline("verify-target-conformance")
```

The runner executes only the requested passes; it does not add the preparation
or verification stages of the full target compiler. Choose an order that meets
each pass's input requirements. The example drops unobservable global phases
because this target does not support them. Set mapping controls in the textual
`place-and-route` options; `CompilationOptions.mapping` configures the full
target pipeline.

The default `preserve_layout=False` clears existing layout metadata. Use
`preserve_layout=True` only for passes that preserve wire identity and site
order. For example, continue from a program compiled with layout metadata:

```{code-cell} ipython3
placed = source.copy()
placed.compile_for_target(environment)
placed.run_pass_pipeline("target-native-synthesis,drop-global-phases", preserve_layout=True)
placed.run_pass_pipeline("verify-target-conformance", preserve_layout=True)
```

This option retains existing metadata; it does not generate a layout or remap
wires. Do not enable it for qubit reuse, wire reordering, or retargeting. A
failed pass can leave the supplied environment and earlier changes on the
program; copy the input before trying another sequence. A pipeline parsing error
leaves the program unchanged.

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
a Qiskit payload selects Qiskit output. Each circuit in a sequence retains its
own width. Inputs must use fixed-width qubit allocations in the entry block,
without static qubits or existing layout metadata. Manually constructed
capabilities with site-specific placements are rejected.

### Synthesis without routing

{py:meth}`~mqt.core.mlir.QCOProgram.synthesize_for_target` translates to native
gates. Gate-set targets preserve the program qubit interface as described above.
For device targets, dynamic qubits require all-to-all connectivity. Dynamic
qubits are assigned target sites; static qubits retain their site IDs. This
method does not require measurement support. It checks target support and
topology; it mutates the program, including on failure, so copy first when
needed.

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
site names. The exporter supports constant native PRX, CZ and MOVE calls and
terminal measurement records; it discards global phase. MOVE operands preserve
the advertised qubit/resonator order. Native serialization does not route or
synthesize MOVE; the IQM server checks MOVE pairing rules. Unsupported reset,
feedback or classical result computation fails before submission.

Raw IQM JSON export requires each output slot to be measured once.
Compiled-program terminal normalization also handles logical zero holes and
permutations, using the retained result layout. QDMI-on-IQM owns service
transport, authentication, calibration selection, and conversion of measurement
columns to QDMI bitstrings.

## Command line from a source build

```console
mqt-core compile --qdmi-list-devices
mqt-core compile --qdmi-device=mqt.sc.iqm.garnet \
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
MQT Core reads it for native compiler targets and the generic Qiskit backend.

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
custom property when QDMI provides a standard parameter-domain contract; see
[QDMI issue #601](https://github.com/Munich-Quantum-Software-Stack/QDMI/issues/601).

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
attached layout. Custom pipelines clear it unless `preserve_layout=True` is set
for passes that preserve wire identity and site order. See
[transpiler layouts](qiskit.md#transpiler-layouts) for the full contract.

### Temporary QDMI payload contract

`QDMI_DEVICE_PROPERTY_CUSTOM3` can supply NUL-terminated UTF-8 JSON with schema
`mqt.payload-capabilities.v1` and a `capabilities` array. Each entry contains an
`id`, optional unsigned `value` (default zero), and an optional `constraints`
object mapping constraint IDs to unsigned limits. This temporary contract is not
part of the QDMI standard. The reported list restricts the selected format's
capabilities; absent capabilities are unavailable. Without recognized metadata,
control flow and classical assignment are disabled. Unknown schemas are ignored,
and malformed recognized metadata is rejected. See the
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
