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
a cached CZ sandwich with four native gates. P and CP are exact compositions of
these rotations and a global phase. Numerical KAK remains the specialization for
constant gates and fused runs. Constant single-qubit fusion reuses numerical
Euler synthesis; quaternion arithmetic is confined to runtime fusion.

The fusion contracts from Core #2649 are integrated here. An explicit policy
owns singleton preservation, controlled-body ownership, native symbolic runs,
and permitted runtime expressions. Native synthesis fuses one wire at a time
while traversing the program for synthesis. The standalone pass uses the same
driver. U targets can shrink runs before routing without invoking a separate
optimizer. A rewrite listener carries phase-generated wires into site analysis.
Fusion declarations live in `NativeSynthesis/SingleQubitFusion.h`; the Euler
emitter remains independent of traversal policy.

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

The compiler, decomposition, optimization, native-synthesis, mapping, and
global-phase suites pass all 1040 C++ tests; Python MLIR/Qiskit suites pass all
967 tests. Coverage includes runtime binding, full global phase, reversed
placements, native gate counts, fixed RX/RY/R constraints, aliases, and
large-angle normalization.

A 20-case comparison against the preceding PR head found no gate-count
regressions. With native U available, CZ-based runtime RZZ needs three local
gates instead of five; iSWAP needs two entanglers instead of four. Regression
checks constrain these counts and exclude runtime trigonometry from Pauli
lowering. Constant fusion shares numerical Euler synthesis while full-unitary
checks permit equivalent Euler coordinates.

Against published head `7790b88c2`, 58 compilation/synthesis cases cover six
automatically selected bases, numeric and symbolic SU2 circuits (2/20 qubits),
the 100-qubit symbolic ZSXX workload from #2614, and 1,000/10,000 rotations and
cancellations. All jointly exportable cases have identical gate counts and
depths. Qiskit and jeff export retain symbolic parameters; small cases agree in
full matrix and phase with maximum error `8.2e-15`. The two previously failing
20-qubit symbolic U exports now succeed: global-phase normalization balances
sums so expression depth grows logarithmically. The exporters retain their input
depth and size limits. Python coverage is retained; compiler semantics and
fusion regressions are checked in their C++ owners.

A focused timing comparison against `7790b88c2` used DGX Spark arm64, LLVM/MLIR
23.1, release builds without IPO, seed 42, two warmups, and separate persistent
processes. In 21 alternating pairs across eight U-target cases, median-time
geometric ratios were 1.002 for compilation and 1.005 for synthesis. The earlier
short-case timing outlier did not persist. Import, copying, export, and
validation were outside the timed interval; these measurements do not establish
a general speedup.

Repository and whole-file C++ lint, executable docs with warnings as errors, and
generated documentation links pass. No binding signatures changed in this
iteration. The synthesis-wide Ponytail audit has no outstanding findings. Bench
records its exact-pin integration and minimum-version results in its companion
PR.

Fixed-entangler synthesis can require four square-root-iSWAP gates; a
specialized symbolic optimizer for that basis remains outside scope.
