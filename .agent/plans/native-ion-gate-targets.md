# Native target synthesis

Status: complete. Native synthesis and export are validated against the shared
gate contracts, with a measured comparison to upstream main.

## Scope and ownership

Native ion operations use existing R and RZZ gates. GPI/GPI2 are Qiskit target
aliases with radian definitions and explicit phase correction. Virtual RZ is
advertised. Providers own units and hardware serialization. MS is unnecessary
for the supported Bench catalogue.

The decomposition layer owns numeric and runtime synthesis. Target synthesis
owns placement and replacement; native cost analysis reads the same synthesis
choices without creating IR. Single-qubit fusion is an optimization, not a
prerequisite for lowering parameterized gates. Equatorial frame propagation
belongs to native synthesis and discards frames at measurement or reset.

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

Prefer unrestricted Pauli entanglers when available. Bounded entanglers use
local corrections for numeric angles and their supported pi/2 endpoint for
unknown angles. Arbitrary fixed-angle synthesis is outside scope.
NativeOperations imports Qiskit capabilities without placement; Bench uses the
ordinary target constructor for native compilation. Its private standard-gate
target lowers aliases without changing public targets or Qiskit's session
library.

## Validation

The native-synthesis, decomposition, optimization, and compiler C++ test
binaries pass, as do the Python MLIR/Qiskit suites. Oracles cover full phase,
native constraints, ordered wires, scalar dominance, barriers, large finite
angles, and exporter parameter identity. Generated stubs, repository lint, and
whole-file C++ lint pass. Independent matrix checks cover native Clifford
families, reversed placement, and signed quarter/half turns.

Against main at `32f1b3314`, 50 frozen circuits across eight native contracts
produce 400 validated exports here versus 271 on main. The 234 jointly supported
Core/scalable cases exclude identity microcases from aggregate ratios. IQM's
median native one-qubit count is one third of main's; median synthesis time is
5% lower. Numeric W-state and Pauli-layer cases take up to 20% longer while
using about two thirds fewer R gates. QFT-16 uses 120 bounded RZZ gates instead
of the 240 CZ gates of the fixed-entangler target; main cannot express those
parameter bounds.

Release builds use LLVM/MLIR 23.1 and Qiskit 2.5.2, one CPU, five timed samples,
and all-to-all connectivity. Timing excludes import/export. Measured-output
checks and sampled statevectors are weaker than full-unitary equivalence.
Routing, calibration fidelity, and device execution are outside this study. Ad
hoc scripts, raw data, and plots remain outside the repository.

Qiskit still expands shared expressions at its boundary. Its resource limits
remain enforced. Unused top-level inputs may disappear from exported circuits;
custom gate definitions retain their formal-parameter contract.
