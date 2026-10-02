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
fixed entangler uses two cached native CX decompositions around a single-qubit
rotation. P and CP are exact compositions of these rotations and a global phase.
Numerical KAK remains the specialization for constant gates and fused runs.

Each recognized two-qubit Pauli sequence has exactly one entangling term.
CompilerTarget owns fixed and unrestricted entangler queries; fusion receives
the complete synthesis basis. Native cost analysis selects its numerical cache
in one place.

Reuse the existing scalar-or-SSA gate builder parameters and synthesis-basis
capabilities. Do not add symbolic dense matrices or a general algebra system.
Emit selected native RX/R quarter turns directly from the Euler emitter,
removing its deferred target-specific lowering walk. Prefer unrestricted Pauli
entanglers over fixed alternatives when both are globally available.

Bench's private standard-gate target and local equivalences continue to lower
native aliases without changing public targets or Qiskit's session library.

## Validation

The compiler, decomposition, optimization, native-synthesis, and mapping suites
pass all 1002 C++ tests; Python MLIR/Qiskit suites pass all 926 tests. Coverage
includes runtime binding, full global phase, reversed placements, native gate
counts, fixed-angle constraints, aliases, and large-angle normalization.
Generated stubs, repository and whole-file C++ lint, executable docs with
warnings as errors, and generated documentation links pass. Independent
quantum-synthesis, symbolic-computation, and correctness/complexity reviews have
no outstanding code findings. Bench records its exact-pin integration and
minimum-version results in its companion PR.

Fixed-entangler synthesis through CX can require four square-root-iSWAP gates; a
specialized symbolic optimizer for that basis remains outside scope.
