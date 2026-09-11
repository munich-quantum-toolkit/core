# Fuse single-qubit runs with dynamic angles

Status: historical implementation record.

## Goal and scope

The `fuse-single-qubit-unitary-runs` MLIR pass currently stops when a gate angle
is an SSA value instead of a compile-time constant. After this change, the pass
will compose supported mixed constant and dynamic single-qubit runs and emit an
equivalent sequence in the requested basis. A user can verify the behavior by
passing a circuit such as `h; rz(%theta)` through the pass and observing that
the original run is replaced while the output still depends on `%theta` and has
the same exact unitary after `%theta` is assigned a value.

The implementation reuses the symbolic quaternion and Euler code that already
powers `merge-single-qubit-rotation-gates`. It must not add a second symbolic
matrix engine. The existing constant-matrix path remains the fast path. Dynamic
`pow`, arbitrary dynamic dense unitaries, and other operations that do not
expose named SSA angle parameters remain run boundaries.

## Constraints

- The hard symbolic composition algorithm already exists in
  `mlir/lib/Dialect/QCO/Transforms/Optimizations/MergeSingleQubitRotationGates.cpp`.
  Its `Val<Value>` path emits `arith` and `math` operations, composes named
  gates as quaternions, extracts ZYZ Euler angles, handles gimbal cases, and
  preserves exact global phase. Evidence: focused tests already cover dynamic
  `rz; rz`, `h; rz`, and `p; p` chains.

- `fuse-single-qubit-unitary-runs` still scans only gates that return a concrete
  `Matrix2x2`. A dynamic gate therefore ends the scan. Evidence:
  `getRunMemberMatrix` returns `std::nullopt` when `getUnitaryMatrix` cannot
  fold every parameter.

- RX, RY, RZ, P, R, U2, and U have closed-form exact decompositions for the U,
  ZYZ, ZXZ, and ZSXX bases. Direct singleton lowering can therefore preserve
  native gates and use structural shortcuts such as one-SX U2-to-ZSXX without
  emitting trigonometric operations.

- The normal target compilation pipeline already runs
  `merge-single-qubit-rotation-gates`. Parameterized layers therefore reach
  target-native synthesis as a dynamic `qco.u`. Planning supported named
  parameterized gates directly in target-native synthesis preserves its
  no-partial-rewrite guarantee and makes a separate post-mapping fuser
  unnecessary.

- Target-native synthesis already treats modifier bodies as opaque. The later
  `skip-controlled-bodies` pass option had no in-tree caller, and disabling the
  dynamic pattern also disabled useful top-level canonicalization. Removing that
  option restores one consistent standalone fuser behavior without changing the
  pre-existing constant-pattern API.

- `CompilerTarget::resolveSynthesisBasis` selects XZX for RX/RZ targets and
  never selects ZXZ. The dynamic path must therefore emit XZX directly even
  though both bases use RX and RZ gates. Direct emission also keeps the pass
  result aligned with the requested basis.

- Conjugating a unit quaternion by Hadamard maps X to Z, Y to negative Y, and Z
  to X. The existing ZYZ extractor can therefore synthesize XZX, XYX, and R
  after one axis transform and fixed angle and phase shifts. Independent
  numerical checks covered random and singular matrices before the MLIR tests
  were added.

- QCO has seven primitive parameterized one-qubit gates: RX, RY, RZ, P, R, U2,
  and U. The table-driven regression now crosses these seven gates with all
  seven synthesis bases and proves exact matrices after binding representative
  values.

- An MLIR rewrite pattern may not emit helper operations and then return
  `failure()`. The first implementation of the dynamic U emitter materialized
  constants before rejecting the no-op U basis, which caused the greedy rewrite
  driver to fail. Checking basis support before building any IR fixed the
  rewrite-contract violation.

- The fuser-specific quaternion composer already had every value needed to emit
  the canonical bases. Emitting those bases there removes the intermediate U
  operation, its one-caller public synthesis API, and its second dynamic rewrite
  without changing the phase equations. Evidence: the direct production refactor
  removes 98 lines while all 236 decomposition tests and 122 optimization tests
  pass.

- The early generic merger and the cleanup pipelines recurse into `qco.ctrl`, so
  an `h; rz(%theta)` body is merged to U before mapping. The target-native
  planner intentionally treats modifier bodies as opaque, leaving that U in
  place while lowering any dynamic phase lifted out of the body. This makes the
  controlled-body regression an ordering test rather than a promise that every
  compilation stage preserves the original body.

- Running a separate target-basis fuser before target-native synthesis allows
  the fuser to mutate the module before native preflight rejects a later
  unsupported operation. Planning supported runtime one-qubit actions alongside
  constant-matrix actions restores the pass's existing no-partial-rewrite
  guarantee.

## Decisions

- Preserve the matrix-based fuser as the first path and add a symbolic fallback
  only for supported named gates. Rationale: Constant matrices provide shorter,
  folded output without runtime arithmetic, and the issue explicitly asks to
  retain that path.

- Reuse the quaternion and phase logic from `MergeSingleQubitRotationGates.cpp`.
  Rationale: That implementation already handles mixed SSA values, angle
  wrapping, gimbal cases, `atan2(0,0)` avoidance, and exact phase correction.
  Duplicating it would create two correctness surfaces.

- Treat unsupported dynamic unitary shells, including dynamic `qco.pow`, as run
  boundaries. Rationale: `UnitaryOpInterface` exposes a numeric matrix but no
  symbolic matrix expression from which the pass could recover arbitrary SSA
  dependence.

- Start with exact dynamic `u`, `zyz`, `zxz`, and `zsxx` emission, then evaluate
  the remaining bases. Rationale: These bases use closed-form transformations of
  canonical U parameters and include the `zsxx` basis used by IBM-style targets.
  The `xzx`, `xyx`, and `r` bases need either the existing inverse-trigonometric
  extraction after an axis transform or a longer noncanonical expansion.

- Extend dynamic extraction to XZX, XYX, and R with a Hadamard quaternion
  transform. Rationale: This reuses the tested gimbal, wrapping, and phase
  logic. It avoids a second inverse-trigonometric implementation and gives every
  accepted basis the same primitive-gate support.

- Use the unconditional runtime sequence length for profitability and reuse the
  existing RX, RY, RZ, and P canonicalizers for short same-axis runs. Rationale:
  General dynamic Euler synthesis cannot assume an angle is zero, while
  same-axis addition is exact and reduces two gates to one without runtime
  control flow.

- Enable symbolic fusion for every resolvable single-qubit target basis.
  Explicit targets need dynamic non-native gates lowered before matrix-only
  synthesis. Integrate symbolic actions into native preflight and its rewrite
  plan so a later failure cannot leave a partially transformed module.

- Emit every requested dynamic basis in the fuser-specific quaternion composer.
  Rationale: the composer already owns the runtime Euler angles and accumulated
  phase. Direct emission removes an intermediate U operation and a public API
  with one caller while keeping the normal merge pass's U output unchanged.

- Keep generic one-qubit merging and generic two-qubit fusion unconditional
  before mapping. Rationale: U and U/CZ are the target-independent intermediate
  forms consumed by mapping; target capabilities should affect only the
  post-mapping native-synthesis stage.

- Teach target-native synthesis to plan and lower the seven supported named
  parameterized one-qubit operations directly. Rationale: each planned symbolic
  operation can reuse the existing quaternion emitter without a greedy rewrite
  over the whole module. The compiler pipeline stays explicit, hidden modifier
  bodies remain untouched, operation pointers stay stable, and no IR is changed
  until preflight succeeds.

- Lower symbolic singletons directly for U, ZYZ, ZXZ, and ZSXX, and retain
  quaternion/Euler extraction for composed runs and transformed XZX, XYX, and R
  bases. Rationale: named-gate identities avoid unnecessary trigonometric IR
  while the general algorithm remains the single fallback for cases that need
  it.

- Remove the newly introduced `skip-controlled-bodies` option and stream
  quaternion accumulation during gate conversion. Rationale: neither abstraction
  has a required caller or rollback benefit; removing them makes the
  implementation smaller without changing supported behavior.

## Outcome and validation

The scoped implementation is complete. `fuse-single-qubit-unitary-runs` now
reuses the existing `Val<Value>` quaternion path to compose profitable named
dynamic runs and emits exact U, ZYZ, ZXZ, XZX, XYX, ZSXX, or R sequences. A
table-driven regression checks all seven primitive parameterized one-qubit gates
in all seven bases. It proves SSA dependence before binding and exact matrices
after binding. A second 28-case regression proves that supported singleton
lowering to U, ZYZ, ZXZ, and ZSXX uses direct parameter identities rather than
runtime quaternion/Euler extraction. A standalone U regression covers both Euler
gimbal branches in the transformed bases. The complete target flow now first
performs target-independent one-qubit merging to U and two-qubit fusion to U/CZ,
then maps, and finally runs one atomic target-native synthesis pass. That pass
plans both constant matrices and supported runtime one-qubit operations before
rewriting, and the shared composer emits the final runtime basis directly.
Constant-only behavior, controlled gate recognition, and the existing dynamic
merge pass remain green. Dynamic `pow`, arbitrary dynamic unitary shells, and
Qiskit parameter-vector import remain separate work because they do not expose
the named angle operands required by the symbolic composer.

## Code and ownership

MQT Core represents optimized quantum programs in the QCO MLIR dialect. A gate
parameter is a compile-time constant when MLIR can fold its SSA value to a
number. A dynamic angle is an SSA value, such as an `f64` function argument,
whose value is known only when the compiled program runs.

`mlir/lib/Dialect/QCO/Transforms/NativeSynthesis/FuseSingleQubitUnitaryRuns.cpp`
implements the pass from issue #1764. It walks one qubit wire, multiplies the
constant two-by-two matrices of adjacent gates, and calls
`synthesizeUnitary1QEuler` to emit the selected basis. The scan stops at the
first operation without a compile-time matrix.

`mlir/lib/Dialect/QCO/Transforms/Optimizations/MergeSingleQubitRotationGates.cpp`
already composes supported dynamic gates. It converts each gate to a unit
quaternion. A quaternion is four real values that represent a single-qubit
unitary up to global phase. Its `Val<double>` backend uses host arithmetic for
constant parameters; its `Val<Value>` backend emits MLIR `arith` and `math`
operations for dynamic parameters. The pass emits one `qco.u` plus a
`qco.gphase` correction.

`mlir/lib/Dialect/QCO/Transforms/Decomposition/Euler.cpp` contains the
constant-matrix Euler emitter.
`mlir/include/mlir/Dialect/QCO/Transforms/Decomposition/Euler.h` declares its
shared API. The dynamic path uses only the small pattern-population interface
and singleton synthesis entry point declared there. Basis emission stays inside
the shared implementation and does not expose intermediate synthesis results as
a public contract.

`mlir/unittests/Dialect/QCO/Transforms/Decomposition/test_euler_decomposition.cpp`
owns tests for the fuser. Dynamic dependency and binding helpers can follow the
patterns in
`mlir/unittests/Dialect/QCO/Transforms/Optimizations/test_qco_merge_single_qubit_rotation.cpp`.

The pass accepts seven basis names: `zyz`, `zxz`, `xzx`, `xyx`, `u`, `zsxx`, and
`r`. For a dynamic canonical U gate, the direct closed-form emitters use these
exact identities, in circuit application order:

- `u(theta, phi, lambda)` is already the `u` basis.
- `rz(lambda); ry(theta); rz(phi)` plus `gphase((phi + lambda) / 2)` is the
  `zyz` basis.
- `rz(lambda - pi / 2); rx(theta); rz(phi + pi / 2)` plus the same phase is the
  `zxz` basis.
- `rz(lambda); sx; rz(theta + pi); sx; rz(phi + pi)` plus
  `gphase((phi + lambda) / 2 + pi / 2)` is the general `zsxx` basis.

XZX, XYX, and R use the quaternion produced for the full input run. Conjugating
that quaternion by Hadamard changes its components from `(w, x, y, z)` to
`(w, z, -y, x)`. The existing ZYZ extractor then supplies transformed `theta`,
`phi`, `lambda`, and phase values. XZX shifts `phi` by positive pi over two and
`lambda` by negative pi over two, then emits RX-RZ-RX. XYX and R shift both
outer angles and the phase by pi. XYX emits RX-RY-RX. R emits the same axes as
`R(lambda, 0)`, `R(theta, pi / 2)`, and `R(phi, 0)`.

Symbolic composition emits the final basis sequence and one combined phase
correction. ZYZ, ZXZ, XZX, XYX, and R add the input and Euler-wrap phases. ZSXX
also adds pi over two. U subtracts `(phi + lambda) / 2` for the intrinsic U
phase.

## Acceptance

Acceptance requires a focused test that fails on the baseline and passes after
the change. Before binding, each original gate parameter must reach an emitted
gate angle or phase. After binding, every emitted parameter must fold to a
finite value and the product of the emitted gates and global phase must match
the input run within the existing matrix tolerance. The test must cover RX, RY,
RZ, P, R, U2, and U in all seven bases. A separate standalone dynamic U case
must bind the beta-zero and beta-pi singularities for XZX, XYX, and R.

Constant-only fuser tests must remain unchanged and pass for all seven bases.
Dynamic gates that the symbolic composer does not support must remain intact and
must not cause a pass failure. The MLIR verifier must accept every rewritten
module. The pass description must name the supported dynamic scope and explain
that runtime values use a conservative fixed gate sequence.

## Interfaces

Use `mlir::Value` for dynamic angles and `mlir::RewritePatternSet` for reusable
pattern population. The existing merge pattern converts every supported gate
before mutating the run, so a rejected gate cannot cause a partial rewrite. Use
the existing `mlir::qco::decomposition::SingleQubitBasis` enum and QCO gate
builders. Do not expose a separate value-backed synthesis result. Do not add an
external dependency.

The dynamic composer must keep support aligned with the existing merge pass:
`rx`, `ry`, `rz`, `p`, `r`, `u2`, `u`, `x`, `y`, `z`, `h`, `s`, `sdg`, `t`,
`tdg`, `sx`, `sxdg`, and `id`. Unsupported modifiers or arbitrary unitary shells
must terminate a dynamic run safely.
