# Fuse single-qubit runs with dynamic angles

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

The `fuse-single-qubit-unitary-runs` MLIR pass currently stops when a gate angle
is an SSA value instead of a compile-time constant. After this change, the pass
will compose supported mixed constant and dynamic single-qubit runs and emit an
equivalent sequence in the requested basis. A user can verify the behavior by
passing a circuit such as `h; rz(%theta)` through the pass and observing that
the original run is replaced while the output still depends on `%theta` and has
the same exact unitary after `%theta` is assigned a value.

The first implementation milestone must reuse the symbolic quaternion and Euler
code that already powers `merge-single-qubit-rotation-gates`. It must not add a
second symbolic matrix engine. The existing constant-matrix path remains the
fast path. Dynamic `pow`, arbitrary dynamic dense unitaries, and other
operations that do not expose named SSA angle parameters remain run boundaries.

## Progress

- [x] (2026-08-24 11:42Z) Created a clean work branch from current `origin/main`
      and confirmed that issue #1764 is not labeled `good first issue`.
- [x] (2026-08-24 11:42Z) Traced the constant fuser, the symbolic rotation
  merger, the Euler emitter, the target pipeline, and their focused tests.
- [x] (2026-08-24 11:49Z) Added a mixed `h; rz(%theta)` regression and observed
      it fail on the matrix-only implementation because the original `rz`
      remained.
- [x] (2026-08-24 11:58Z) Exposed the existing dynamic composition pattern in a
  fuser-specific mode without changing the normal merge pass.
- [x] (2026-08-24 12:00Z) Added exact dynamic emission and phase correction for
  the `u`, `zyz`, `zxz`, and `zsxx` bases.
- [x] (2026-08-24 12:04Z) Passed 234 decomposition tests, 122 optimization
  tests, 131 compiler tests, `git diff --check`, and `uvx nox -s lint`.
- [x] (2026-08-24 12:12Z) Completed an independent read-only review. No
  correctness, phase, rewrite-contract, or dialect blockers were found; the
  conservative same-axis and target-pipeline gaps are explicitly recorded.
- [x] (2026-08-24 13:24Z) Reproduced `compileForTarget` rejecting the dynamic
  `qco.u` produced from `h; rz(%theta)` for an explicit ZSXX target.
- [x] (2026-08-24 13:31Z) Inserted target-configured symbolic fusion after
      mapping cleanup and before target-native synthesis, while leaving
      operations nested in `qco.ctrl` unchanged for native CX/CZ recognition.
- [x] (2026-08-24 13:35Z) Passed the end-to-end ZSXX regression, all 132
  compiler tests, 234 decomposition tests, 122 optimization tests,
  `git diff --check`, and `uvx nox -s lint`.
- [x] (2026-08-24 13:43Z) Addressed the final independent review by routing
      RX/RZ targets, which resolve to XZX, through the equivalent closed-form
      ZXZ sequence. The extended end-to-end test, all compiler tests, and lint
      pass.
- [x] (2026-08-24 14:28Z) Added direct symbolic XZX, XYX, and R extraction by
      applying a Hadamard axis transform to the existing quaternion Euler
      algorithm. All seven named bases now accept every primitive parameterized
      one-qubit gate.
- [x] (2026-08-24 14:31Z) Added a 7-by-7 gate-and-basis regression, short
      same-axis fusion coverage, and end-to-end target tests for every basis
      that `CompilerTarget` can select.
- [x] (2026-08-24 14:36Z) Added a standalone dynamic U regression at the
      beta-zero and beta-pi Euler singularities for XZX, XYX, and R. The focused
      tests pass.
- [x] (2026-08-24 14:49Z) Completed final PR validation after adding the beta-pi
      regression: fresh builds, 236 decomposition tests, 122 optimization tests,
      132 compiler tests, `git diff --check`, and `uvx nox -s lint` pass. Three
      independent read-only reviews found no blocking correctness or scope
      issue.
- [x] (2026-08-24 14:52Z) Committed and pushed the reviewed implementation,
      opened pull request #2228, and folded its reference into the existing
      unreleased single-qubit optimization changelog entry.
- [x] (2026-08-24 15:08Z) Simplified the reviewed implementation so the dynamic
      composer emits all seven bases directly. Removed the intermediate dynamic
      U synthesis API and collapsed the compiler target fixtures and gate-count
      checks. The production refactor removes 98 lines and the test refactor
      removes 32 lines.
- [x] (2026-08-24 15:11Z) Rebuilt the affected targets and passed all 236
      decomposition tests, 122 optimization tests, 132 compiler tests,
      `git diff --check`, and `uvx nox -s lint` after the simplification.
- [x] (2026-08-24 15:19Z) Inspected the failed online C++ lint annotations.
      Direct emission removed the three obsolete Euler warnings. Designated
      initializers, a direct `<cstddef>` include, and file-scope placement fix
      the remaining warnings. Local Clang-Tidy reports no warning in the changed
      production sources.
- [x] (2026-08-24 15:21Z) Committed and pushed the reviewed simplification and
      online C++ lint fixes to pull request #2228.
- [x] (2026-08-25 15:14Z) Reworked target compilation around the canonical stage
      order: generic one-qubit merging to U and generic two-qubit fusion,
      mapping, then one atomic target-native synthesis pass.
- [x] (2026-08-25 15:14Z) Extended target-native planning and lowering with the
      existing parameterized quaternion emitter. Removed the separate late fuser
      and its basis-string adapter from the compiler pipeline.
- [x] (2026-08-25 15:14Z) Strengthened the controlled-body regression so it
      uniquely proves the early generic merge occurred, and added direct native
      synthesis and no-partial-rewrite coverage for runtime one-qubit gates.
- [x] (2026-08-25 15:32Z) Passed 236 decomposition tests, 122 optimization
      tests, 24 target-synthesis tests, 133 compiler tests, `git diff --check`,
      `uvx nox -s lint`, and local Clang-Tidy on every changed translation unit.
- [x] (2026-08-26 22:02Z) Addressed final review feedback: supported symbolic
      singleton gates now use closed-form U/ZYZ/ZXZ/ZSXX lowering, quaternion
      accumulation is streamed in the conversion loop, and the unused
      `skip-controlled-bodies` pass option is gone.
- [x] (2026-08-26 22:02Z) Added a 28-case direct-lowering regression that checks
      exact gate counts, exact matrices including global phase, and the absence
      of runtime trigonometric operations. Passed 237 decomposition tests, 122
      optimization tests, 24 target-synthesis tests, and 133 compiler tests.
- [x] (2026-08-26 22:17Z) Rebased onto current `origin/main` at `f4d8cdb21` and
      passed 237 decomposition tests, 193 optimization tests, 24
      target-synthesis tests, 135 compiler tests, and the repository lint suite.
- [x] (2026-08-27 12:31Z) Removed 34 lines of duplicate assertions while
      preserving singularity, atomicity, all-bases, and fast-path coverage.
- [x] (2026-08-27 13:54Z) Merged current `main`, applied the new MLIR handle,
      parameter, and documentation rules to the changed code, and passed 237
      decomposition tests, 193 optimization tests, 24 target-synthesis tests,
      135 compiler tests, `uvx nox -s lint`, and `uvx nox -s cpp-lint`.
- [x] (2026-08-27 14:32Z) Completed a branch-wide audit against the merged
      development policies. Renamed the remaining `module` variable, used the
      required C typedef spelling, removed `const` from range views, fixed
      implementation comments, and made this plan describe the final pipeline. A
      fresh build and all 589 focused tests pass. `uvx nox -s lint` passes. The
      local `cpp-lint` rerun stops because the host has Clang-Tidy 21.1.1 and
      the new session requires Clang-Tidy 22.

## Surprises & Discoveries

- Observation: The hard symbolic composition algorithm already exists in
  `mlir/lib/Dialect/QCO/Transforms/Optimizations/MergeSingleQubitRotationGates.cpp`.
  Its `Val<Value>` path emits `arith` and `math` operations, composes named
  gates as quaternions, extracts ZYZ Euler angles, handles gimbal cases, and
  preserves exact global phase. Evidence: focused tests already cover dynamic
  `rz; rz`, `h; rz`, and `p; p` chains.
- Observation: `fuse-single-qubit-unitary-runs` still scans only gates that
  return a concrete `Matrix2x2`. A dynamic gate therefore ends the scan.
  Evidence: `getRunMemberMatrix` returns `std::nullopt` when `getUnitaryMatrix`
  cannot fold every parameter.
- Observation: RX, RY, RZ, P, R, U2, and U have closed-form exact decompositions
  for the U, ZYZ, ZXZ, and ZSXX bases. Direct singleton lowering can therefore
  preserve native gates and use structural shortcuts such as one-SX U2-to-ZSXX
  without emitting trigonometric operations.
- Observation: The normal target compilation pipeline already runs
  `merge-single-qubit-rotation-gates`. Parameterized layers therefore reach
  target-native synthesis as a dynamic `qco.u`. Planning supported named
  parameterized gates directly in target-native synthesis preserves its
  no-partial-rewrite guarantee and makes a separate post-mapping fuser
  unnecessary.
- Observation: Target-native synthesis already treats modifier bodies as opaque.
  The later `skip-controlled-bodies` pass option had no in-tree caller, and
  disabling the dynamic pattern also disabled useful top-level canonicalization.
  Removing that option restores one consistent standalone fuser behavior without
  changing the pre-existing constant-pattern API.
- Observation: `CompilerTarget::resolveSynthesisBasis` selects XZX for RX/RZ
  targets and never selects ZXZ. The dynamic path must therefore emit XZX
  directly even though both bases use RX and RZ gates. Direct emission also
  keeps the pass result aligned with the requested basis.
- Observation: Conjugating a unit quaternion by Hadamard maps X to Z, Y to
  negative Y, and Z to X. The existing ZYZ extractor can therefore synthesize
  XZX, XYX, and R after one axis transform and fixed angle and phase shifts.
  Independent numerical checks covered random and singular matrices before the
  MLIR tests were added.
- Observation: QCO has seven primitive parameterized one-qubit gates: RX, RY,
  RZ, P, R, U2, and U. The table-driven regression now crosses these seven gates
  with all seven synthesis bases and proves exact matrices after binding
  representative values.
- Observation: An MLIR rewrite pattern may not emit helper operations and then
  return `failure()`. The first implementation of the dynamic U emitter
  materialized constants before rejecting the no-op U basis, which caused the
  greedy rewrite driver to fail. Checking basis support before building any IR
  fixed the rewrite-contract violation.
- Observation: The local Ninja log repeatedly reports
  `premature end of file; recovering`, so incremental builds rebuild more
  objects than expected. All requested targets still compile and their test
  binaries pass.
- Observation: The fuser-specific quaternion composer already had every value
  needed to emit the canonical bases. Emitting those bases there removes the
  intermediate U operation, its one-caller public synthesis API, and its second
  dynamic rewrite without changing the phase equations. Evidence: the direct
  production refactor removes 98 lines while all 236 decomposition tests and 122
  optimization tests pass.
- Observation: The early generic merger and the cleanup pipelines recurse into
  `qco.ctrl`, so an `h; rz(%theta)` body is merged to U before mapping. The
  target-native planner intentionally treats modifier bodies as opaque, leaving
  that U in place while lowering any dynamic phase lifted out of the body. This
  makes the controlled-body regression an ordering test rather than a promise
  that every compilation stage preserves the original body.
- Observation: Running a separate target-basis fuser before target-native
  synthesis allows the fuser to mutate the module before native preflight
  rejects a later unsupported operation. Planning supported runtime one-qubit
  actions alongside constant-matrix actions restores the pass's existing
  no-partial-rewrite guarantee.

## Decision Log

- Decision: Preserve the matrix-based fuser as the first path and add a symbolic
  fallback only for supported named gates. Rationale: Constant matrices provide
  shorter, folded output without runtime arithmetic, and the issue explicitly
  asks to retain that path. Date/Author: 2026-08-24, Codex.
- Decision: Reuse the quaternion and phase logic from
  `MergeSingleQubitRotationGates.cpp`. Rationale: That implementation already
  handles mixed SSA values, angle wrapping, gimbal cases, `atan2(0,0)`
  avoidance, and exact phase correction. Duplicating it would create two
  correctness surfaces. Date/Author: 2026-08-24, Codex.
- Decision: Treat unsupported dynamic unitary shells, including dynamic
  `qco.pow`, as run boundaries. Rationale: `UnitaryOpInterface` exposes a
  numeric matrix but no symbolic matrix expression from which the pass could
  recover arbitrary SSA dependence. Date/Author: 2026-08-24, Codex.
- Decision: Start with exact dynamic `u`, `zyz`, `zxz`, and `zsxx` emission,
  then evaluate the remaining bases. Rationale: These bases use closed-form
  transformations of canonical U parameters and include the `zsxx` basis used by
  IBM-style targets. The `xzx`, `xyx`, and `r` bases need either the existing
  inverse-trigonometric extraction after an axis transform or a longer
  noncanonical expansion. Date/Author: 2026-08-24, Codex.
- Decision: Extend dynamic extraction to XZX, XYX, and R with a Hadamard
  quaternion transform. Rationale: This reuses the tested gimbal, wrapping, and
  phase logic. It avoids a second inverse-trigonometric implementation and gives
  every accepted basis the same primitive-gate support. Date/Author: 2026-08-24,
  Codex.
- Decision: Use the unconditional runtime sequence length for profitability and
  reuse the existing RX, RY, RZ, and P canonicalizers for short same-axis runs.
  Rationale: General dynamic Euler synthesis cannot assume an angle is zero,
  while same-axis addition is exact and reduces two gates to one without runtime
  control flow. Date/Author: 2026-08-24, Codex.
- Decision: Enable symbolic fusion in target compilation for every single-qubit
  basis that `CompilerTarget` can resolve. Rationale: permissive targets
  intentionally leave native programs unchanged; explicit targets need dynamic
  non-native gates lowered before target-native synthesis rejects their
  unavailable compile-time matrix. Date/Author: 2026-08-24, Codex.
- Decision (superseded on 2026-08-25): Keep symbolic fusion as a separate pass
  immediately before target-native synthesis. Rationale: this reused the public
  pass and preserved atomicity of the native pass in isolation. It was
  superseded because the two-pass stage could still partially rewrite the module
  before native preflight failed; direct symbolic actions in the native plan
  preserve atomicity for the complete stage. Date/Author: 2026-08-24, Codex.
- Decision: Emit every requested dynamic basis in the fuser-specific quaternion
  composer. Rationale: the composer already owns the runtime Euler angles and
  accumulated phase. Direct emission removes an intermediate U operation and a
  public API with one caller while keeping the normal merge pass's U output
  unchanged. Date/Author: 2026-08-24, Codex.
- Decision: Keep generic one-qubit merging and generic two-qubit fusion
  unconditional before mapping. Rationale: U and U/CZ are the target-independent
  intermediate forms consumed by mapping; target capabilities should affect only
  the post-mapping native-synthesis stage. Date/Author: 2026-08-25, Codex.
- Decision: Teach target-native synthesis to plan and lower the seven supported
  named parameterized one-qubit operations directly. Rationale: each planned
  symbolic operation can reuse the existing quaternion emitter without a greedy
  rewrite over the whole module. The compiler pipeline stays explicit, hidden
  modifier bodies remain untouched, operation pointers stay stable, and no IR is
  changed until preflight succeeds. Date/Author: 2026-08-25, Codex.
- Decision: Lower symbolic singletons directly for U, ZYZ, ZXZ, and ZSXX, and
  retain quaternion/Euler extraction for composed runs and transformed XZX, XYX,
  and R bases. Rationale: named-gate identities avoid unnecessary trigonometric
  IR while the general algorithm remains the single fallback for cases that need
  it. Date/Author: 2026-08-26, Codex.
- Decision: Remove the newly introduced `skip-controlled-bodies` option and
  stream quaternion accumulation during gate conversion. Rationale: neither
  abstraction has a required caller or rollback benefit; removing them makes the
  implementation smaller without changing supported behavior. Date/Author:
  2026-08-26, Codex.

## Outcomes & Retrospective

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

## Context and Orientation

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

## Plan of Work

First, add a focused test with a mixed constant and dynamic run such as
`h; rz(%theta)`. The test must show that the baseline pass does not fuse the
run. Change the expected behavior so that the final test requires one equivalent
basis sequence, SSA dependence on `%theta`, valid IR, and exact matrix equality
after assigning a representative value.

Next, expose the smallest reusable entry point around the dynamic composition
pattern in `MergeSingleQubitRotationGates.cpp`. The normal merge pass must keep
its current behavior. The fuser-specific entry point must rewrite only a run
that contains a dynamic parameter and that either contains a gate outside the
requested basis or is longer than the conservative runtime basis sequence. A
single dynamic U in the U basis must not rewrite repeatedly.

Then make the fuser-specific quaternion composer emit the requested basis
directly. Emit the unconditional general form for runtime values; do not
introduce runtime control flow to imitate constant-angle gate-count shortcuts.
Keep the normal merge pass's U output unchanged. Add `MathDialect` to the
fuser's dependent dialects because the reused composition path emits
trigonometric operations.

For `xzx`, `xyx`, and `r`, reuse the existing quaternion Euler extraction after
Hadamard conjugation and apply fixed angle and phase shifts. Emit the requested
basis directly so the greedy composer cannot loop through an intermediate U. Do
not add a second inverse-trigonometric implementation.

Then integrate symbolic lowering into the existing target-native synthesis
stage. Keep target compilation in this order: the default QCO optimization
pipeline merges target-independent one-qubit runs to U, two-qubit fusion creates
the U/CZ intermediate form, mapping assigns target sites, and
`TargetNativeSynthesisPass` lowers the mapped program to the target basis. In
`mlir/lib/Dialect/QCO/Transforms/NativeSynthesis/TargetSynthesis.cpp`, plan
every required rewrite before changing the IR. Accept a single-qubit operation
when it has either a constant matrix or named runtime parameters supported by
`canSynthesizeParameterizedUnitary1Q`. After the complete plan succeeds, lower
runtime operations with `synthesizeParameterizedUnitary1Q` and keep the existing
matrix Euler path for constant operations. This keeps target synthesis atomic
and avoids a separate post-mapping fuser.

Add native-synthesis tests for direct runtime lowering and failed-preflight
atomicity. Add compiler tests that exercise the complete stage order, every
target-selected one-qubit basis, and modifier bodies. Finally, run all four
affected MLIR test binaries, the whole-file C++ lint session, and the repository
lint suite. Inspect the final diff and update this plan with the results.

## Concrete Steps

Run all commands from the repository root.

Configure the release preset, build the four affected test targets, and
establish the focused baseline:

    cmake --preset release
    cmake --build --preset release --target \
      mqt-core-mlir-unittest-decomposition \
      mqt-core-mlir-unittest-optimizations \
      mqt-core-mlir-unittest-target-synthesis \
      mqt-core-mlir-unittests-compiler
    build/release/mlir/unittests/Dialect/QCO/Transforms/Decomposition/\
      mqt-core-mlir-unittest-decomposition \
      --gtest_filter='FuseSingleQubitUnitaryRunsTest.*'

During implementation, rebuild the same targets and run the new dynamic test
alone. A successful test must report one passing test, no verifier error, and no
NaN parameter after binding the dynamic input.

After the focused tests pass, run:

    build/release/mlir/unittests/Dialect/QCO/Transforms/Decomposition/\
      mqt-core-mlir-unittest-decomposition
    build/release/mlir/unittests/Dialect/QCO/Transforms/Optimizations/\
      mqt-core-mlir-unittest-optimizations
    build/release/mlir/unittests/Dialect/QCO/Transforms/NativeSynthesis/\
      mqt-core-mlir-unittest-target-synthesis
    build/release/mlir/unittests/Compiler/\
      mqt-core-mlir-unittests-compiler
    uvx nox -s cpp-lint
    uvx nox -s lint

Record exact pass counts and any unavailable check in this plan.

## Validation and Acceptance

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

## Idempotence and Recovery

All build and test commands are repeatable. CMake places generated files under
`build/`, which is not committed. Source edits remain limited to the QCO
decomposition, fusion, and target-synthesis transforms; the compiler target
pipeline; their focused tests; `CHANGELOG.md`; and this plan. If a rewrite fails
during development, use the failing focused test to restore a valid milestone;
do not reset or discard unrelated work. The worktree was clean when the task
started, so any later unrelated change must be preserved and reported.

## Artifacts and Notes

Baseline source evidence:

    getRunMemberMatrix(dynamic_gate) -> std::nullopt
    FuseSingleQubitUnitaryRunsTest.IgnoresDynamicPowerExponent -> pass leaves pow

The existing dynamic merger computes:

    correction = totalInputPhase - (phi + lambda) / 2 + eulerWrapPhase

Final test evidence:

    decomposition: 237 passed
    optimizations: 193 passed
    target synthesis: 24 passed
    compiler: 135 passed
    lint, formatting, and diff checks: passed
    whole-file C++ lint rerun: unavailable; Clang-Tidy 22 is not installed

Retain the clamp around `acos`, the pure-Z gimbal path, the sanitized
`atan2(0,0)` input, and the Euler wrap phase when reusing this code.

## Interfaces and Dependencies

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

Revision note (2026-08-24): Created the initial plan after tracing the current
matrix fuser, symbolic merger, Euler emitter, target pipeline, and related test
coverage.

Revision note (2026-08-24): Recorded the implemented four-basis milestone, the
rewrite-contract fix, and successful focused and broad validation.

Revision note (2026-08-24): Closed the first-milestone review with no blockers
and retained same-axis shortcuts and target-pipeline integration as explicit
follow-up work.

Revision note (2026-08-24): Extended the completed design to all seven bases,
recorded same-axis and target integration, added full cross-product and singular
coverage, and removed superseded limits from the outcome.

Revision note (2026-08-24): Recorded the post-review simplification that emits
all dynamic bases directly and removes the intermediate U synthesis API.

Revision note (2026-08-27): Recorded the final merge, current MLIR policy
alignment, and validation evidence.

Revision note (2026-08-27): Aligned every branch-added file with the merged
development policies and updated the work plan, commands, scope, and evidence to
describe the final target-native design.
