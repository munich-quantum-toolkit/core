---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# Compile for a QDMI device

`compile_program` maps a program to a device's topology and native operations.
For a QDMI device, it returns a {py:class}`~mqt.core.mlir.CompiledProgram` by
default. Submit it with `submit_program` in the MLIR submodule.

For a hands-on comparison of connectivity, native gates, and logical outputs,
see the
{doc}`hardware-compilation tutorial <../tutorials/hardware_compilation>`.

## Python

Compile and submit a Bell circuit to the bundled DDSIM device:

```{code-cell} ipython3
from mqt.core.mlir import compile_program, submit_program
from mqt.core.qdmi import open_device

bell_qasm = """OPENQASM 3.1;
include "stdgates.inc";
qubit[2] q;
bit[2] result;
h q[0];
cx q[0], q[1];
result = measure q;
"""

device = open_device("mqt.ddsim.default")
compiled = compile_program(bell_qasm, target=device)
job = submit_program(compiled, target=device)
job.wait()
print(job.get_counts())
```

For several programs, pass a homogeneous Python sequence, such as a list or
tuple. All elements share the target and compilation options. The returned list
follows input order; an empty sequence returns `[]`.

```{code-cell} ipython3
sources = (bell_qasm, bell_qasm.replace("h q[0];", "x q[0];"))
for compiled in compile_program(sources, target=device):
    job = submit_program(compiled, target=device, num_shots=16)
    job.wait()
    print(job.get_counts())
```

Elements must share one input kind (subclasses are accepted). Strings and paths
are single inputs; generators and binary buffers are unsupported. An explicit
`output` selects every result's representation. Output defaults follow the same
rules as for a single program.

Typed MLIR inputs are consumed; use `copy=True` to preserve them. Input kinds
are checked before compilation. A compilation failure stops the sequence and
returns no results; processed inputs may be consumed, while later inputs remain
usable.

`target` accepts an open device or a registered device ID. To compile and submit
in one call:

```{code-cell} ipython3
job = submit_program(bell_qasm, target=device)
job.wait()
print(job.get_counts())
```

Or use a device ID directly:

```{code-cell} ipython3
job = submit_program(bell_qasm, target="mqt.ddsim.default")
job.wait()
print(job.get_counts())
```

Pass `num_shots` to `submit_program` to choose the number of samples. For
simulator statevectors and probabilities, see {doc}`../qdmi/ddsim_device`. A
compiled program can be submitted again without recompilation. Submission checks
that the device still has matching sites, topology, operations, timing units,
and program capabilities. IQM JSON also requires unchanged site names because
the payload embeds them. Calibration-only changes do not require recompilation.
Use `device.submit_job` to submit raw payloads.

### Compilation options

```python
from mqt.core.mlir import CompilationOptions, MappingOptions

options = CompilationOptions(
    seed=7,
    mapping=MappingOptions(trials=4, iterations=2, lookahead=10, search_memory_limit=8 * 1024 * 1024),
)
compiled = compile_program(bell_qasm, target=device, options=options)
```

The same `options` argument is available on typed compilation methods and source
submission. Set `enable_timing` and `enable_statistics` on this object; compiler
entry points accept these controls only through `options`. An explicit seed
overrides compiler randomness, including custom pass seeds; `None` preserves
existing pass settings. Execution sampling has a separate seed. For the CLI:

```console
mqt-cc input.qasm --qdmi-device mqt.sc.iqm.garnet \
  '--payload-spec=#mqt.payload_spec<format = <id = "qir", version = "2.1.0", profile = "base", encoding = text>, capabilities = [], optional_capabilities_known = false>' \
  --seed 7 --mapping-trials 4 --mapping-iterations 2 --mapping-lookahead 10 \
  --mapping-search-memory-limit 8388608
```

Add `--emit=qco-optimized -o mapped.mlir` to write the mapped QCO program with
layout metadata. The payload specification still defines the target's execution
capabilities.

Trials must be positive. Omitted trials use the logical CPU count; iterations
default to one forward/backward refinement round. Zero iterations score each
initial layout directly. Lookahead is the number of additional two-qubit gates
considered during routing. It defaults to 20; zero considers only the current
gate. All-to-all placement ignores valid mapping controls. Repeatable mapping
requires the same build, input, target, seed, and mapping controls, including an
explicit trial count. Layouts may change between releases.

For explicit native gate sets with a usable entangler basis, routing ranks its
candidates by estimated native two-qubit gate count, using qubit-dependency
depth to break ties. A lower gate count can therefore win even when depth
increases. Read-only analysis shares synthesis decisions without cloning IR or
running passes for each candidate. Final synthesis determines the emitted count.
Counts are static for structured programs; depth is the maximum within a block
and does not model classical scheduling or runtime control flow. The
`place-and-route` pass in the {doc}`QCO reference <QCO>` describes the search
and fallback behavior.

### Choose a format

For a submission payload, the compiler selects the first supported format in
this order: Adaptive QIR (binary, then text), OpenQASM 3.1, IQM JSON, then Base
QIR (binary, then text). To select an exact execution format, compile and submit
in one call:

```{code-cell} ipython3
from mqt.core.qdmi import ProgramFormat

job = submit_program(bell_qasm, target=device, program_format=ProgramFormat.QASM3, num_shots=64)
assert job.wait()
```

`compile_program` uses `output` to select a typed representation, such as
`OutputFormat.QC`, `OutputFormat.QCO_OPTIMIZED`, or `OutputFormat.QISKIT`. The
device still supplies the compilation constraints. The raw `QC_IMPORT` and `QCO`
checkpoints stop before target compilation and cannot be combined with a target.

Typed outputs support inspection and interchange. In particular, jeff output
does not retain hardware placement. The default `CompiledProgram` retains the
automatically selected native payload for repeated QDMI submissions; exact
wire-format selection belongs to `submit_program`.

### Define a target

Use `CompilerTarget` to describe a device for compilation without opening a
connection. Bundled SC devices such as `mqt.sc.iqm.garnet` provide ready-made
hardware models; they do not execute programs.

With an explicit `CompilerTarget`, the default result preserves the input's
program family. Select `output` for another representation. To define a target
with three sites and nearest-neighbor connectivity:

```{code-cell} ipython3
from mqt.core.mlir import CompilerTarget, OutputFormat

target = CompilerTarget(
    3,
    connectivity=CompilerTarget.Connectivity([(0, 1), (1, 2)]),
    native_operations=CompilerTarget.NativeOperations([
        CompilerTarget.OperationCapability(
            "gphase",
            arity=CompilerTarget.OperationArity.fixed(0),
            num_parameters=1,
        ),
        CompilerTarget.OperationCapability("u", arity=1, num_parameters=3),
        CompilerTarget.OperationCapability(
            "cx",
            arity=2,
            num_parameters=0,
            site_tuples=[(1, 0), (1, 2)],
        ),
        CompilerTarget.OperationCapability("measure", arity=1, num_parameters=0),
        CompilerTarget.OperationCapability("reset", arity=1, num_parameters=0),
    ]),
)
mapped = compile_program(
    bell_qasm, target=target, output=OutputFormat.QIR_BASE
)
print(mapped.ir)
```

The CLI also accepts a saved target as a typed MLIR attribute. Serialize a
Python target with `str(target)`, or pass the attribute directly:

```console
mqt-cc input.qasm \
  '--target=#mqt.compilation_target<sites = [<id = 0>, <id = 1>], connectivity = all_to_all, couplings = [], native_operations = unrestricted, operations = []>' \
  '--payload-spec=#mqt.payload_spec<format = <id = "openqasm", version = "3.1.0", profile = "", encoding = text>, capabilities = [], optional_capabilities_known = false>' \
  -o compiled.qasm
```

`--target` requires `--payload-spec` and is mutually exclusive with
`--qdmi-device`. It uses the same mapping, synthesis, and output selection as
QDMI target compilation, without loading a device. Add `--emit=qco-optimized` to
inspect the mapped program. A saved target records capabilities at capture time;
it does not query current device availability or submit a job.

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

`fixed_parameters` constrains individual parameters to finite constants. A
nonempty list has one entry per parameter; `None` leaves a parameter
unrestricted. An omitted or empty list leaves every parameter unrestricted.
Multiple capabilities for the same operation form a union of supported values
and placements. For example, these capabilities accept four RX angles:

```python
from math import pi

rx_gates = [
    CompilerTarget.OperationCapability("rx", 1, 1, fixed_parameters=[angle]) for angle in (pi / 2, -pi / 2, pi, -pi)
]
```

Constants match with absolute tolerance `1e-15`, without angle wrapping. Unbound
symbolic values cannot satisfy fixed parameters or `parameter_bounds`. Bounds
are inclusive `(lower, upper)` pairs, with `None` for unbounded parameters;
relations between parameters are not represented. A device-specific instruction
name can specify its compiler operation with `canonical_name`, for example
`OperationCapability("rx_90", 1, 1, fixed_parameters=[pi / 2], canonical_name="rx")`.
The reported name remains available to exporters.
`CompilerTarget.NativeOperations.from_qiskit(source)` imports gate capabilities
and parameter constraints independently of device width and placement. Combine
them with connectivity and a circuit width using the `CompilerTarget`
constructor.

Target compilation requires a single-qubit synthesis basis available on every
site. `ZSXX` accepts unrestricted RZ and SX, RX(±π/2), RY(±π/2), or R(±π/2, 0).
Matching native half turns can shorten the decomposition. Euler synthesis emits
these gates directly, including global-phase corrections.

The {term}`GPI basis` accepts R(π, φ) and R(π/2, φ) with unrestricted φ on every
site. It emits at most three pulses for a single-qubit unitary, including
runtime parameters, and preserves global phase. θ and φ use radians and Core's
convention R(θ, φ) = exp[-i θ (cos(φ) X + sin(φ) Y) / 2].

Qiskit target import recognizes `gpi(phi)` and `gpi2(phi)` defined as
`i R(pi, phi)` and `R(pi/2, phi)`, respectively, with **radian** parameters.
These are fixed-parameter R capabilities inside Core; Qiskit export with the
target restores their native names and phase. These gates provide the GPI basis
when their azimuths are unrestricted on every site. A provider using turns or
another phase convention needs an explicit adapter.

Synthesis compares supported entangler families and chooses the recipe with the
fewest two-qubit gates. Ties retain the target's preference order, which places
unrestricted RXX, RYY, RZX, and RZZ before fixed alternatives. These rotations
use up to three native gates for numeric two-qubit synthesis. Runtime two-qubit
Pauli rotations need one arbitrary-angle native entangler or two fixed native
entanglers, including square-root iSWAP. Fixed Clifford entanglers use Pauli
conjugation; commuting symbolic rotations can share their entanglers.
Square-root iSWAP uses two gates for two commuting runtime Pauli rotations and
four for three independent rotations. Binding all parameters first permits
numerical synthesis with at most three gates. The fixed capability
`XXPlusYY(-pi/2, 0)` also enables square-root iSWAP synthesis. Single-controlled
Pauli rotations and phase gates use the same decomposition. Runtime
`XXPlusYY(theta, 0)` and `XXMinusYY(theta, 0)` share the two-rotation circuit;
other relative phases require a native gate or constant parameters. Bounded
Pauli entanglers that include `[0, pi/2]` use the same numeric synthesis with
local Pauli corrections. Unknown runtime angles use two native `pi/2` rotations.
Bind parameters before compilation to use fractional entanglers directly. Qiskit
import conservatively recognizes this interval through its public bound
predicates.

Resynthesis of constant two-qubit gates uses an average gate fidelity floor of
`1 - 1e-12` per decomposition, including direct Pauli lowering. It may remove
small entangling angles or shorten near-Clifford rotations. This is a local
bound, not a whole-circuit error budget; unbound angles are not approximated.

Synthesis merges Z rotations through diagonal entanglers on native RZ targets.
Equatorial R targets use at most two R gates per single-qubit unitary and absorb
Z rotations into R axes. Measurement and reset discard their incoming Z frames;
other boundaries retain the frame and global phase. RX/RZ targets prefer ZXZ to
minimize physical RX rotations.

### Placements and calibration

Use plain tuples for placements without calibration. Use
`CompilerTarget.SiteTuple([1, 0], duration=40, fidelity=0.99)` to attach
calibration to a placement; both forms can appear in the same list.

Routing uses undirected adjacency; native synthesis repairs unsupported operand
directions. Target compilation requires a known static physical site for each
qubit. Structured branch exits must agree on sites, and loop backedges must
preserve the entry sites. Unsupported or inconsistent site transfers are
diagnosed, including after all-to-all placement. A synthesis basis must provide
the same one-qubit gate family on every site. Its entangler is optional:
one-qubit synthesis does not need one. Two-qubit synthesis requires an entangler
on every routing edge in at least one direction.

Mapping explores one initial-layout trial per available logical CPU by default,
using LLVM's affinity-aware CPU count with a minimum of one. An explicit
`ntrials` value overrides this default. Set both `ntrials` and `seed` on the
`place-and-route` pass for reproducible results across machines. Disabling
multithreading runs the same trials sequentially. The trial budget includes a
greedy layout when available, followed by identity if a slot remains and random
layouts for the remaining slots. Every trial uses the same refinement count.

Each routing search limits its estimated node and layout storage to 256 MiB by
default. When the budget is exhausted, it checks queued states before falling
back to SWAPs that reduce the leading interaction's distance. Set
`MappingOptions.search_memory_limit` in bytes, or use the CLI's
`--mapping-search-memory-limit`, to trade memory for routing quality; zero
disables node expansion. The equivalent `place-and-route` pass option is
`search-memory-limit`. Each concurrent trial reuses its bounded node and layout
storage across searches and releases it when the trial finishes. For example, 20
active trials with 512 MiB each allow about 10 GiB of estimated search storage.
Container overhead, target distance caches, and IR storage are additional; this
setting does not cap total process memory. Changing the budget can change
layouts and gate counts; more memory does not guarantee fewer gates.

When routing is needed, mapping prepares a read-only table of at most 1024 sets
of Weyl coordinates from original gates and constant two-qubit runs, in both
operand orders and with adjacent SWAPs. Preparation stops when the table is
full. Already-adjacent greedy layouts skip preparation and cost tracking. Trials
share this table. Each live region's cost tracker retains up to 64 additional
coordinate sets for routing-dependent matrices. The coordinates serve all
candidate entangler families. The numerical payload is about 288 KiB for the
shared table and 18 KiB per local cache, plus indexing, allocator, and per-site
tracking overhead. Backward refinement also retains one pending single-qubit
matrix per site to preserve circuit order. Each traversal resets accounting
while retaining numerical caches. These allocations are separate from the search
budget. Native synthesis caches up to 64 full decompositions per analysis (about
55 KiB including single-qubit factors). Caches are local to one pass invocation
or traversal and retain no IR handles. Exact matrix and entangler matches
preserve numerical decisions for the same compilation seed; failed numerical
analyses remain unavailable. Target support and operand direction are checked
before lookup. Cache misses use normal synthesis analysis, so precomputation
need not predict every routed run.

Native synthesis collects constant runs on the same two qubits, including
interleaved single-qubit gates, and resynthesizes them in the target's selected
basis. It replaces a run only when the result uses fewer native two-qubit gates
than preserving supported operations and lowering the others individually. For
example, a non-native RZZ followed by RXX can require two CZ gates together,
compared with four when lowered separately. Already-native operations are
preserved unless block synthesis reduces their native gate count.

Before placement, both target pipelines also fuse runs when this reduces the
number of two-qubit operations in the IR and their native gate count. This
removes cancelled interactions before routing. Native support at this stage is
checked without physical sites; the later synthesis pass checks assigned sites.

Native support includes physical sites and operand direction. Barriers,
non-unitary operations, and unavailable matrices stop a run. If block
decomposition fails numerically, synthesis falls back to individual lowering.
Each operation or run uses its cheapest supported entangler family. If circuit
construction fails, synthesis tries the next candidate. Routing counts are
estimates and do not include such numerical fallbacks. Selection does not
optimize mixtures of entangler families within one recipe or use calibration
costs.

Target synthesis preserves a native `gphase`. If the target does not support
`gphase`, target synthesis preserves relative phase effects and removes only the
unobservable global phase of the entry point, including its classical branches
and loops. Global phases in helper functions and those that remain inside QCO
modifiers are retained.

Non-native two-qubit operations outside these rotation decompositions require a
compile-time unitary matrix.

Use {py:meth}`~mqt.core.mlir.QCOProgram.compile_for_target` with the target
environment to apply target compilation to an existing QCO program. Compilation
runs in place. If a pass fails, the environment and earlier pass changes remain
on the program. In Python, `compile_for_target` raises `RuntimeError` with the
emitted MLIR diagnostics, including operation and source-location details when
available. Copy the program before compilation if the caller must preserve the
input. The pipeline takes one `TargetEnvironment`, replaces any existing
`mqt.target_env` module attribute, and shares the prepared target with all
target passes without rebuilding its connectivity tables. The selected
environment must remain unchanged during pipeline execution. Standalone passes
decode the typed module attribute once through a cached analysis. The mapping,
native-synthesis, and conformance factories also work in textual MLIR pass
pipelines. Target compilation keeps deterministic placement on all-to-all
targets and uses mapping only for explicit topology. The high-level program API
registers the required inliner extensions; callers that populate the low-level
target pipeline directly must register inliner extensions for every callable
dialect in their context.

### Synthesis without routing

Use {py:meth}`~mqt.core.mlir.QCOProgram.synthesize_for_target` to translate a
QCO program to a target's native gate set without routing. Dynamic qubits
require all-to-all connectivity and receive an initial layout. Static qubits
keep their device site IDs and may use an explicit topology. The pipeline
inlines calls, decomposes non-native controlled gates, places dynamic qubits,
performs native synthesis, and checks target support and topology. It accepts
structured QCO/SCF input and uses the same target environment and global-phase
policy as target compilation.

Both target pipelines decompose controlled composite gates, including inverse
bodies and constant integer powers of operations on disjoint wires. Other
composite powers require native target support or a synthesis rule for that
operation. Gates acting on three or more qubits need a target-independent
decomposition before native synthesis and routing, unless the target supports
them natively. Explicit-topology routing handles only one- and two-qubit gates;
static circuits may keep native wider gates at supported device sites.

Before decomposition, the pipelines group adjacent non-native controlled gates
with the same ordered controls and targets so existing canonicalization can
simplify their bodies. Already-native groups remain unchanged, including gates
with fixed supported angles. Runtime body fusion produces controlled U only when
the target supports it. After this fusion opportunity, remaining composites are
split before routing or placement and native synthesis. OpenQASM payloads keep
runtime rotations separate when fusion would require unsupported quaternion
extraction expressions.

Synthesis runs in place and raises `RuntimeError` with MLIR diagnostics on
failure. Earlier pass changes may remain on the program, so copy it first when
the input must be preserved. The C++ counterpart is
`QCOProgram::synthesizeForTarget`; low-level clients can populate a pass manager
with `populateTargetSynthesisPipeline`.

### Payload control flow

For explicit restrictions, use the constants on
{py:class}`~mqt.core.mlir.ProgramCapability` and
{py:class}`~mqt.core.mlir.ProgramConstraint`. Custom identifiers are also
accepted.

Target compilation requires structured QCO/SCF input. Producers of raw CFG
branches must normalize them before target compilation. The pipeline removes
unused symbols, propagates constants, and runs QCO cleanup before deciding which
loops need expansion. It then specializes loops required by the selected payload
or by placement, cleans up the resulting IR, and checks the remaining control
flow with `legalize-control-flow`:

| Capability           | Residual operations                                 |
| -------------------- | --------------------------------------------------- |
| `forward-branching`  | `qco.if` and classical `scf.if`                     |
| `counted-iteration`  | `scf.for`                                           |
| `conditional-loop`   | `scf.while`                                         |
| `multiway-branching` | `qco.index_switch` and classical `scf.index_switch` |

A finite `scf.for` that exceeds the selected counted-iteration contract is fully
unrolled when this clones at most one billion body operations by default. The
`unroll-loops-for-payload` pass exposes this limit as `max-operations`. The same
bound applies to loops unrolled for qubit placement. Between expansion rounds,
local classical memory is promoted and expressions fold to expose constant array
accesses, nested bounds, and conditions. Standalone Base QIR export uses this
same unroller with an explicit `payload-spec`, without requiring hardware
metadata. An unsupported index switch is lowered to a linear chain of nested
forward branches when that form fits the selected contract. Before expansion,
the compiler checks the selected forward-branching nesting limit and a compiler
safety limit of 256 total control-flow levels, including enclosing control flow.
This compiler limit is not a QDMI requirement and does not apply to switches
retained under multiway branching.

Generic SCF branches cannot capture or return QCO qubits or quantum tensors; use
the corresponding QCO branch operation for linear quantum state. SCF loops must
carry linear quantum state through their iteration arguments instead of
capturing it. Both control-flow passes validate this loop input restriction
before transforming loops or lowering switches. It is separate from QCO's
exactly-one-SSA-use check.

Cleanup shares constant-slot scalarization across `qco.if`, `scf.for`, and
`scf.while`. Each region must extract distinct constant indices, reinsert every
extracted qubit, and pass the tensor to its terminator. The loop body must
return each tensor to its original iteration argument; a while condition may
reorder the before-region results. Untouched slots remain outside the control
flow. Runtime indices and incomplete or nested tensor updates do not match this
scalarization.

For Adaptive QIR on an all-to-all target whose operations have empty
`site_tuples`, placement assigns physical sites to the allocation's slots and
retains indexed registers. Loop bodies do not grow with their iteration counts.
The site list requires space proportional to the register width. Capacity,
physical site IDs, qubit origins, native operations, and payload limits are
still checked. This path uses target metadata and does not depend on a device
name.

Other payloads, explicit topology, and site-specific operations require exact
quantum addresses. Bounded specialization exposes those addresses before
placement or routing. Residual unsupported tensor control flow produces a
diagnostic before allocation changes. Mapped OpenQASM uses static physical
qubits; indexed tensor loops must fit the default one-billion-operation
unrolling budget. Runtime-dependent indices that cannot be specialized are
unsupported. Logical qubit indices can remain dynamic in targetless OpenQASM
export. Constant rank-one `f64` table reads use switches that group equal
entries and require unrestricted multiway branching from the selected payload.
Their size grows with the table data and number of reads.

The supported constraints are `max-control-flow-nesting-depth` on all four
capabilities, `max-iteration-count` on both iteration capabilities, and
`max-case-count` on multiway branching, counting explicit cases without the
default region. One explicit case plus a default is a supported index switch and
does not require forward branching. Limits are inclusive. The compiler must
prove a constrained loop's trip count. It currently proves constant `scf.for`
bounds and rejects a constrained `scf.while` because no general termination
bound is available. The proof requires literal loop bounds and a literal step;
it does not infer a trip count from symbolic bounds. MLIR computes static trip
counts; full unrolling additionally requires bounds and scaled steps that fit
its signed arithmetic. The scaled step must also fit the loop induction-variable
type. A zero, unknown, or misapplied constraint makes that capability group
unusable. A capability absent from the selected specification is unsupported.

This stage checks structural control flow. Final QIR output also checks its
emitted feature requirements: dynamic qubit and result management, arrays, IR
functions, multiple return points, integer and floating-point computations,
multiway branching, and counted or conditional loops. The check applies to both
typed QIR and submission payloads. QIR loop bounds are not re-proved after
lowering; a loop capability required by emitted QIR must be unconstrained.

An explicit output combines the output format's capabilities with the target's
contract. Matching language versions and profiles retain only capabilities
supported by both, regardless of text or binary encoding. Different formats
share structural control-flow restrictions; QIR-specific features are not
inferred from their absence in a Qiskit contract. Later lowering stages remain
responsible for scalar types and operations, measurement provenance, allocation,
and payload-profile conformance.

## IQM JSON circuits

Select `OutputFormat.IQM_JSON` to compile a Qiskit or OpenQASM circuit for IQM.
A target is required and supplies IQM native operations, connectivity, and
physical site names:

```{code-cell} ipython3
import json
from mqt.core.mlir import CompilerTarget, OutputFormat, compile_program

iqm_target = CompilerTarget(
    [CompilerTarget.Site(0, "QB1"), CompilerTarget.Site(1, "QB2")],
    connectivity=CompilerTarget.Connectivity.all_to_all(),
    native_operations=CompilerTarget.NativeOperations([
        CompilerTarget.OperationCapability("r", 1, 2),
        CompilerTarget.OperationCapability("cz", 2, 0),
        CompilerTarget.OperationCapability("measure", 1, 0),
    ]),
)
iqm_circuit = compile_program(
    bell_qasm, target=iqm_target, output=OutputFormat.IQM_JSON
)
print(json.loads(iqm_circuit.source)["instructions"][-1])
```

The returned `IQMJSONProgram` contains one circuit object. Use `.source`,
`str(program)`, or `.write(path)` to read or save it. For a QDMI-on-IQM device,
pass `program_format=ProgramFormat.IQM_JSON` to `submit_program` to select the
execution format explicitly.

Compilation lowers the placed circuit to Base Profile QIR, then converts its PRX
and CZ calls to IQM JSON. PRX angle and phase use radians. Physical QIR qubit
handles map to target site IDs, not positions in the target's site list.

The converter emits one IQM measurement per QIR result record, in record order.
Array labels, sizes, and positions provide IQM Qiskit SDK measurement keys:
`registerName_registerSize_registerIndex_bitIndex`. Original register names,
including underscores, are preserved. Scalar outputs form one-bit registers;
unlabeled QIR outputs receive names `c0`, `c1`, and so on.

For supported Qiskit and OpenQASM register readout, the compiler emits these
records in source classical-slot order. The keys match the SDK even when its
source instruction order differs. Complete terminal readout gives each output
slot exactly one measurement of a distinct qubit. IQM measurement columns
therefore already follow the source slots; QDMI-on-IQM reverses them into QDMI
bit strings, and Qiskit formats the original registers without a decoding hook.
This does not require measuring unused physical target qubits.

Qiskit initializes untouched classical bits to zero. The IQM JSON result path
does not reconstruct those slots, so compilation requires complete readout even
for Qiskit inputs. Use `QuantumCircuit(2)` with `measure_all()` to let Qiskit
allocate the measured register without an extra untouched register.

Gate parameters must be finite constants, and qubit and result handles must be
static. The converter accepts one Base Profile entry point with acyclic,
unconditional control flow. Missing or overwritten output slots, repeated qubit
measurements, reset, feedback, MOVE, and classical result computation are
unsupported. QDMI-on-IQM transports the circuit, authentication, calibration
selection, and execution options. IQM owns pulse compilation.

## Command line from a source build

List the stable IDs of configured QDMI devices:

```console
mqt-cc --qdmi-list-devices
```

To synthesize and place for a named IQM target:

```console
mqt-cc --qdmi-device=mqt.sc.iqm.garnet \
  --payload-spec='#mqt.payload_spec<format = <id = "iqm-json", version = "", profile = "", encoding = text>, capabilities = [], optional_capabilities_known = false>' \
  -o circuit.json input.qasm
```

Select a device when compiling:

```console
mqt-cc --qdmi-device=mqt.ddsim.default \
  --payload-spec='#mqt.payload_spec<format = <id = "qir", version = "2.1", profile = "base", encoding = binary>, capabilities = [], optional_capabilities_known = false>' \
  -o output.bc input.qasm
```

An explicit registry file can be selected before device discovery:

```console
mqt-cc --qdmi-config=/path/to/qdmi.json \
  --qdmi-device=example.device \
  --payload-spec='#mqt.payload_spec<format = <id = "qir", version = "2.1", profile = "base", encoding = binary>, capabilities = [], optional_capabilities_known = false>' \
  input.qasm
```

The payload specification selects the emitted format and encoding. For targeted
QIR, the selected encoding takes precedence over the output filename extension.
Target compilation rejects `--emit` and custom `--pass-pipeline` pipelines
because the target contract owns the output and required pass ordering.

## C++ source-tree API

Program factories and conversions return `mlir::FailureOr<T>`. Fallible in-place
and file operations return `mlir::LogicalResult`. Check results with
`mlir::failed` or `mlir::succeeded`; compiler diagnostics describe the errors.

Compile a file and submit it to DDSIM:

```cpp
#include "mqt/Compiler/QDMIAdapter.h"
#include "qdmi/QDMI.hpp"
#include "llvm/Support/Error.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

auto device = qdmi::Session::openDevice("mqt.ddsim.default");
auto input = mlir::QCProgram::fromOpenQASMFile("input.qasm");
if (mlir::failed(input)) {
  return 1;
}
auto compiled = mlir::compileProgram(std::move(*input), device);
if (!compiled) {
  llvm::errs() << llvm::toString(compiled.takeError()) << '\n';
  return 1;
}
auto job = mlir::submitProgram(device, *compiled);
if (!job) {
  llvm::errs() << llvm::toString(job.takeError()) << '\n';
  return 1;
}
if (!job->wait()) {
  return 1;
}
```

`compilerTargetFromDevice` and `compilerTargetFromDeviceId` also remain
available for hardware snapshots and staged compilation with a
`TargetEnvironment`.

### Payload support

Targeted payloads use the device's advertised operation names and ordered
placements. OpenQASM emits native calls without portable gate definitions;
untargeted export retains its portable helpers. Native measurement and reset
must use the OpenQASM keywords `measure` and `reset`. QIR preserves the native
name in each QIS call and uses the controlled specialization for variadic
operations. OpenQASM skips `U`, `u2`, `u3`, and `cu3`, whose standardized phase
conventions differ from QC. The compatibility spelling `u` preserves QC's exact
U-gate matrix. A QC U2 gate can use an advertised `u` with its first parameter
fixed to `pi / 2`, including under control. Compilation fails if no compatible
advertised operation can represent a gate in the selected format. QIR can use
these native names directly.

The `sqrt_iswap` synthesis shorthand requires a matching `xx_plus_yy` capability
for native payload export.

QDMI's supported-format list does not provide per-format capability
restrictions. The adapter therefore uses the compiler-supported baseline for the
selected format (OpenQASM 3.1, IQM JSON, or QIR 2.1), with optional capability
metadata marked as unknown. Use `TargetEnvironment` and `PayloadSpecification`
for staged compilation with explicit restrictions.

QIR submission requires a parameterless entry point. QC-to-QIR conversion
records direct measurements and CBit-register results, then emits runtime status
0 for successful execution. Adaptive Profile also records computed Boolean
(`i1`) results. All explicit QC and QCO returns are program data; other scalar
entry results, including a constant-zero `i64`, are unsupported with or without
a target. Keep other classical temporaries local, or select OpenQASM 3 for wider
scalar outputs. In OpenQASM 3, declare `output bit` registers explicitly when
global scalar variables are only temporaries. See the
{doc}`QIR execution contracts <../qir/index>`.

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

If the program should use fewer device qubits, run the {code}`mqt-qubit-reuse`
pipeline before target compilation.

## Qiskit export

When exporting a program that has already been mapped to a
{py:class}`~mqt.core.mlir.CompilerTarget`, pass the same target to
{py:meth}`~mqt.core.mlir.QCOProgram.to_qiskit` or
{py:meth}`~mqt.core.mlir.QCProgram.to_qiskit`. The exporter maps each static
target site ID to its index in {py:attr}`~mqt.core.mlir.CompilerTarget.sites`
and creates a device circuit using applicable standard gate names from the
target. The circuit has one register named {code}`q` with
{py:attr}`~mqt.core.mlir.CompilerTarget.num_sites` qubits. Target-aware export
requires static qubits whose site IDs belong to that target. Target compilation
attaches layout metadata when possible.

## Layout metadata

For dynamic qubits, `compile_for_target` assigns program qubits to device qubits
and attaches the resulting layout to the QCO program.

```python
program = QCProgram.from_openqasm_str(bell_qasm).to_qco()
program.compile_for_target(environment)
circuit = program.to_qiskit(target=environment.target)
print(circuit.layout.final_index_layout())
```

The attached layout records placement and routing through unused device qubits.
The Qiskit exporter attaches it only when given a target with the recorded site
order. Otherwise, it exports the circuit without a layout. Target compilation
rejects a program with an attached layout. See
[transpiler layouts](qiskit.md#transpiler-layouts).

Static qubits name device sites directly. A program must use either static or
dynamic qubits. Static circuits must fit the target topology; compilation and
synthesis preserve their site IDs and do not attach a layout. For dynamic
qubits, layout metadata requires fixed-size allocations in the entry block. If a
program declares more qubits than the device but shrinks to fit during
compilation, it compiles without an attached layout. Later transformations clear
layout metadata.
