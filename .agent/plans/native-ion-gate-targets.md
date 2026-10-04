# Native target synthesis

Status: implemented and validated. Performance measurements, frozen inputs, and
plots are maintained separately from the source PR.

## Scope and ownership

Native ion operations use existing R and RZZ gates. GPI/GPI2 are Qiskit target
aliases with radian definitions and explicit phase correction. Virtual RZ is
advertised. Providers own units and hardware serialization. MS is unnecessary
for the supported Bench catalogue.

The decomposition layer owns numeric and runtime synthesis. Target synthesis
owns placement and replacement; native cost analysis reads the same synthesis
choices without creating IR. Single-qubit fusion is an optimization, not a
prerequisite for lowering parameterized gates. Native synthesis propagates Z
frames on equatorial and native RZ targets and discards incoming frames at
measurement or reset.

## Decisions

Pauli axes and scalar angles describe elementary rotations. Constant Clifford
frames change axes without runtime trigonometry. An unrestricted native Pauli
entangler realizes each two-qubit rotation in one instruction. A fixed Clifford
entangler sandwiches local rotations to realize one or two commuting Pauli
products with two native gates. SQRTISWAP retains its separate, numerically
stable two-gate identity. P and CP carry explicit global phase. Numerical KAK
owns constant-led runs; symbolic composition requires a reduction in native
entanglers. Single- and two-qubit composition share a traversal, with generated
wire sites tracked by the synthesis listener.

Constant nonlocal synthesis shares the Weyl fidelity floor of 1 - 1e-12 across
fixed, fractional, and direct Pauli decompositions. Near-Clifford constants use
the matrix planner for approximation; unknown runtime angles remain exact. This
bounds individual decompositions, not the complete circuit.

Same-generator runs share exact inverse cancellation and balanced angle sums.
Scale angles before normalization, and normalize each surviving term before
addition. This preserves small corrections beside very large finite angles. Only
exactly opposite coefficients cancel: floating-point reassociation of unequal
coefficients can change the phase. Generated symbolic global phases are
normalized before accumulation for the same reason.

Reuse MLIR folding and CSE; neither a general algebra engine nor symbolic dense
matrices are needed. Runtime quaternion composition remains confined to
standalone fusion and controlled U bodies. Native synthesis uses Euler
identities and absorbs Z frames into equatorial gates when scalar dominance
allows it. Fixed native angles remain exact capabilities.

Propagate native RZ frames after local fusion. Resynthesizing shifted runs can
reduce total gate count while adding physical rotations. Isolated RZ parameters
stay untouched. Every rewrite while the constant folder is live shares its
listener so erased constants cannot remain cached.

The module cleanup owns canonicalization and dead symbols after inlining; the
inliner needs no separate callable-optimization pipeline. Keep dead-value
cleanup after placement, which can change structured-control-flow results.

Prefer unrestricted Pauli entanglers when available. Bounded entanglers use
local corrections for numeric angles and their supported pi/2 endpoint for
unknown angles. Arbitrary fixed-angle synthesis is outside scope.
NativeOperations imports Qiskit capabilities without placement; Bench uses the
ordinary target constructor for native compilation. Its private standard-gate
target lowers aliases without changing public targets or Qiskit's session
library.

## Validation

The complete C++ suite passes 3922 tests, with one existing job-ID test skipped.
Python MLIR/Qiskit tests pass all 791 cases. The frozen smaller benchmark suite
produces 400 validated exports. Oracles cover full phase, native constraints,
ordered wires, scalar dominance, barriers, large finite angles, and exporter
parameter identity. Repository lint and whole-file C++ lint pass. Regression
checks preserve physical rotation counts when Z frames move across diagonal
entanglers.

Large-circuit evaluation uses 68 frozen inputs at 24, 36, 54, 104, and 156
qubits across seven native contracts. Release builds use LLVM/MLIR 23.1 and
Qiskit 2.5.2, one CPU, five timed samples, and all-to-all connectivity. Timing
excludes import/export. Large-width checks establish native gate and angle
conformance, not full-unitary equivalence. Routing, calibration fidelity, and
device execution are outside the study. Ad hoc scripts, raw data, and plots
remain outside the repository.

Qiskit still expands shared expressions at its boundary. Its resource limits
remain enforced. Unused top-level inputs may disappear from exported circuits;
custom gate definitions retain their formal-parameter contract.
