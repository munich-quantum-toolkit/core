# Native target synthesis

Status: complete.

## Scope and ownership

Native ion operations use existing R and RZZ gates. GPI/GPI2 remain Qiskit
target aliases with exact radian definitions and explicit global-phase
correction. Virtual RZ is an advertised capability. Providers own units, angle
ranges, frame tracking, and hardware serialization. MS is unnecessary for the
supported Bench catalogue.

The decomposition layer owns numeric and runtime synthesis. Target synthesis
owns placement and replacement; native cost analysis reads the same synthesis
choices without creating IR. Single-qubit fusion is an optional optimization,
not a prerequisite for lowering a parameterized gate.

## Decisions

Represent an elementary rotation by its Pauli axes and original scalar angle.
Constant Clifford frames change axes without runtime trigonometry. An arbitrary
native Pauli entangler realizes each two-qubit rotation in one instruction; a
fixed Clifford entangler conjugates a single-qubit rotation into a two-qubit
Pauli rotation. Constant local corrections implement its inverse. SQRTISWAP uses
two native gates and symbolic single-qubit corrections. P and CP are exact
compositions of these rotations and a global phase. Numerical KAK remains the
specialization for constant gates and fused runs. Constant single-qubit fusion
reuses numerical Euler synthesis; quaternion arithmetic is confined to runtime
fusion.

A supplied target determines singleton preservation, controlled-body ownership,
and native symbolic runs. Native fusion uses direct Euler identities; general
quaternion composition is confined to standalone fusion and controlled U bodies.
Native synthesis fuses one wire at a time while traversing the program for
synthesis. The standalone pass uses the same driver. U targets can shrink runs
before routing without invoking a separate optimizer. A rewrite listener carries
phase-generated wires into site analysis. Fusion declarations live in
`NativeSynthesis/SingleQubitFusion.h`; the Euler emitter remains independent of
traversal. Equatorial frame propagation lives in native synthesis, shares Euler
parameter extraction, and absorbs final Z frames directly into R gates. It fuses
constant local factors without converting symbolic runs through U. The synthesis
listener invalidates MLIR folder entries when constants are erased by other
rewrites. Bounded Pauli entanglers use their supported pi/2 endpoint for unknown
angles; numeric angles retain local corrections. NativeOperations imports Qiskit
capabilities without placement, so Bench uses the ordinary target constructor
for native compilation.

Normalize constant full turns with their phase correction, omit phase-only U
gates, and choose an equivalent Euler representative only when it removes gates.

Each recognized two-qubit Pauli sequence has exactly one entangling term.
CompilerTarget owns fixed and unrestricted entangler queries; fusion receives
the complete synthesis basis. Native cost analysis selects its numerical cache
in one place.

Reuse the existing scalar-or-SSA gate builder parameters and synthesis-basis
capabilities. Do not add symbolic dense matrices or a general algebra system.
Emit selected native RX/RY/R quarter turns directly from the Euler emitter, with
RY implemented by constant Euler-frame offsets. Keep arbitrary RZ as the free
axis; arbitrary fixed-angle synthesis is outside scope. Prefer unrestricted
Pauli entanglers over fixed alternatives when both are globally available.

Bench's private standard-gate target and local equivalences continue to lower
native aliases without changing public targets or Qiskit's session library.

## Validation

Current local checks: 95 native-synthesis, 204 optimization, and 317
decomposition C++ tests; 773 Python MLIR/Qiskit tests. These cover native gate
counts, parameter identity and resource limits, global phase, large finite
angles, barriers, and control flow. Repository lint, regenerated stubs, and
whole-file C++ lint pass. After rebasing onto the MQT attribute implementation
split, 217 focused C++ checks and all 773 Python tests pass again; whole-file
lint also covers the relocated bound verifier. Bench validates the exact
compiler pin in its companion PR. Executable docs and documentation links passed
before these internal expression changes; product documentation and public
signatures are unchanged.

Against `5004c418c`, a two-qubit R/CZ workload repeats `RZ(a_i); R(0.3,0); CZ`
256 times. It retains 258 R and 256 CZ gates while reducing scalar operations
from 5,125 to 2,048. On DGX Spark arm64 with LLVM/MLIR 23.1, release builds, and
Qiskit 2.5.2, median-of-three synthesis times fell from 34.9 to 24.6 ms and
export times from 758 to 142 ms. A 20-qubit symbolic SU2 circuit retains 220 R
and 60 CZ gates, with export falling from 10.7 to 4.6 ms. Repeating the same
symbolic RY angle 1,024 times on a ZSXX/CZ target reduces export from 25.8 to
22.1 ms. Long RZ runs and other Euler bases show no comparable speedup; these
are workload-specific results, not a universal performance claim.

Qiskit still expands shared expressions at its boundary. For long accumulated
frames, bind in Core before export when executable numeric output is needed. No
algebra dependency, new binding API, or floating-point reassociation flag is
required.

Expression simplification uses MLIR folding and CSE, plus quantum identities
before scalar emission. Integer affine simplification does not apply to these
floating-point rotations. Commuting RZ runs cancel inverse SSA terms before
normalization, retaining survivor order. Qiskit conversion caches shared
expression children by owning identity within each circuit writer. Export
preflight validates expanded node/depth budgets before memoized conversion;
numbers and symbols retain their existing conversion paths.
