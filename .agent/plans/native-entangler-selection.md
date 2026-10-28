# Native entangler selection and symbolic square-root iSWAP

Status: complete. Base: `3f1ccfd69b12f101587be35c9f5eb05c42adf211`; PR #2775.

## Scope and decisions

Recognize a square-root iSWAP declared as `xx_plus_yy` with parameters
`(-pi/2, 0)`, fuse commuting runtime Pauli rotations with this gate, and choose
native entanglers by two-qubit gate count. Preserve numerical and phase checks,
ordered placements, deterministic ties, and existing native operations.

- `CompilerTarget` owns capability recognition and the ordered set of entanglers
  usable on every coupling. Keep its existing single-qubit basis and default
  entangler API. Use the widest supported angle domain for each gate.
- `NativeCostAnalysis` owns selection for routing, individual synthesis, and
  constant or symbolic fusion. Choose the cheapest supported homogeneous recipe
  and keep the established preference order on ties. This does not search
  circuits that mix entangler families or optimize hardware duration or error.
- Cache validated Weyl coordinates by matrix and seed, independent of the
  entangler. Keep failure entries and operand orientations distinct. Full
  synthesis still validates construction of native factors.
- Runtime Pauli fusion compares complete run costs. The two-generator
  square-root iSWAP circuit uses two gates. Three independent runtime generators
  use four; a full symbolic Cartan solver is outside this change. Binding first
  enables the numerical solver's three-gate bound.
- Use exact principal-branch identities for local frames and scaled square-root
  expressions for small angles. The formulas use the arithmetic and
  trigonometric operations supported by symbolic exporters. OpenQASM exports
  floating-point absolute value through its existing scalar expression path.
- Keep a cold Weyl result only through selection and construction. Reject fusion
  that cannot reduce the native count before constructing factors; preserve warm
  caches and numerical fallback.
- Classify symbolic angles once across entangler candidates. Represent
  `XXPlusYY(theta, 0)` and `XXMinusYY(theta, 0)` as two commuting scaled Pauli
  terms and share their pair emission with fused runs.
- Share the counts-based selector with grouped fusion. Keep coordinate counting
  behind validated Weyl decomposition; do not add matrix-count or forwarding
  wrappers without production callers. Preserve numerical failure coverage in
  the native-cost analysis tests.
- Bound Qiskit parameter expressions at 32,768 expanded nodes and 64 levels.
  Measure ordinary short fusion runs; do not introduce a symbolic algebra
  subsystem or a gate-specific export threshold. Qiskit binding still grows
  costly with fused expressions; bind numerical values in Core before export.
- Keep the scaled norms and angle normalization. Rewriting the norms using
  `cos(2*x) + cos(2*y)` produces incorrect zero-bound expressions in Qiskit
  2.5.1 and 2.6.0. Removing inner normalization or using deferred substitution
  increases expression construction time on these versions.

## Validation

All 1,568 tests in the C++ suites below pass. The 848 tests in
`test/python/test_mlir.py`, `test/python/test_mlir_qiskit_target.py`, and
`test/python/test_mlir_qiskit_translation.py` pass on Qiskit 2.6.0. The 39
runtime and expression-budget cases also pass on Qiskit 2.5.1. Repository lint
and whole-file C++ lint pass; stub regeneration changes no API.

Existing regressions cover phase-sensitive matrices, count/emission agreement,
cache eviction and ownership, rejected fusion, ordered sites, exchange signs,
cancellation, nonzero relative-phase limits, and expression-budget failures.
Build and run the corresponding MQT utility/transform, decomposition,
optimization, native synthesis, compiler, OpenQASM, and jeff unit-test targets.
Run the root guide's lint, C++ lint, and stub-generation sessions.

The pair formula passes 24,461 numerical samples and 1,121 export-then-bind
samples, including zero, tiny, boundary, and large angles. Matrix errors stay
below `1e-15`. Independent numerical and design reviews found no remaining
correctness issue. Hosted CI is separate from local validation.

## Performance evidence and limits

Against upstream `8ed992b64`, full `compileForTarget` measurements on 224-gate,
eight-qubit circuits kept single-family CPU time within about 1%. Mixed-family
selection reduced Haar-circuit entanglers from 672 to 497 and local gates from
1,352 to 1,002, with CPU time falling from 37.89 to 36.49 ms. A repeated-matrix
workload fell from 30.02 to 27.82 ms. Routed square-root iSWAP compilation fell
from 13.74 to 12.47 ms with unchanged gate counts. Peak RSS changed by less than
0.3 MiB. All 32 full-output matrix checks passed, with maximum entry error below
`1.9e-13`.

Comparing both operand orientations has a cost when it cannot reduce gates. Ten
alternating rounds on directed two-CZ blocks measured 32.19 to 33.28 ms (3.45%
median paired overhead), with identical output counts. The undirected control
changed by 0.11%. Keep orientation validation and the final emitted-count
benefit check; neither warrants a separate fast path.

These local measurements use ARM64, Clang 23, release ThinLTO, and CPU affinity,
not reserved cores. The pipeline benchmark excludes parsing, source copying,
target creation, and export. Exact upstream sources rebuilt the affected
libraries against the same unchanged dependencies and generated headers. The
measured branch was `e27841d79`, before removing unused APIs and sharing the
counts selector with symbolic fusion.

Five alternating rounds pinned to one CPU checked 12,000 reconstructed native
circuits. Reusing cold Weyl results reduced full-selection CPU time by 22–46%.
Count-first fusion rejection reduced it by 50–74% and avoided factor
allocations; only the initial 18,944-byte coordinate cache remains. Warm
full-selection cost stayed within 3%, and warm paths allocated nothing. These
routine comparisons use the implementation before cold-result reuse as their
baseline, not upstream main.

For 1,000 symbolic selections over nine families, classifying once reduced CPU
time from 14.08 to 2.06 ms at expression depth 8 and from 97.84 to 11.36 ms at
depth 64. Allocations fell ninefold. Single-family costs stayed unchanged.
Selected families and counts matched in both implementations.

On Qiskit 2.6.0, exporting two, four, and eight independent rotations per axis
fell from 17.7/33.8/85.2 ms to 13.6/28.3/57.4 ms. Bound matrices agreed below
`1e-14`. The enlarged budget also exports the 32-per-axis probe, whose parameter
binding was not benchmarked. These are local routine CPU measurements, not
whole-compiler speedups. They do not remove Qiskit's symbolic binding cost.
Copying the compiled Core program, binding values, and exporting took
1.3/1.5/1.6 ms for two/four/eight rotations per axis and 3.1 ms for 32. The
exported matrices agreed below `1e-14`. This uses the existing binding API.
