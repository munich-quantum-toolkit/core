# Prove affine OpenQASM quantum indices

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

OpenQASM 3 permits an implementation to reject quantum-register indices that are
not compile-time constants. MQT Core currently accepts arbitrary runtime indices
and emits bounds, negative-index, overflow, and operand-distinctness guards.
Those guards prevent simple structured programs from lowering to `jeff`. After
this change, MQT Core accepts a larger-than-required but precisely defined
subset: constants and integer affine expressions whose safety follows from
enclosing positive-step `for` ranges. The frontend rejects every other quantum
index before MLIR emission. A user can compile the loop from issue #2188 to
`jeff`, and benchmark-shaped triangular loops such as QFT retain direct
`scf.for` structure without runtime guard scaffolding.

## Progress

- [x] (2026-08-20 15:44Z) Refreshed PR #2135 and confirmed its head remains
  `5d8eb8f26ba8c99ef94d1dace9d47429d27b8a16`.
- [x] (2026-08-20 15:44Z) Inspected the semantic analyzer, emitter, tests, and
  installed MLIR Presburger API.
- [x] (2026-08-20 16:27Z) Added the affine proof context and typed-frontend
  invariants.
- [x] (2026-08-20 16:41Z) Removed quantum runtime guards and directly lowered
  proven positive ranges.
- [x] (2026-08-20 17:06Z) Added acceptance, rejection, structural, and
  end-to-end regressions.
- [x] (2026-08-20 17:13Z) Updated user-facing documentation and migration notes.
- [x] (2026-08-20 17:48Z) Passed focused tests, all 2,789 MLIR tests, the full
  release build, lint, and final diff checks.
- [x] (2026-08-20 22:59Z) Audited merged PRs #2150, #2189, and #2202, then
  rebased the work onto `0c50dd30815638517aa159d20e78290cd449323e`.
- [x] (2026-08-20 23:05Z) Completed a read-only review-agent pass and fixed its
  mixed-range correctness and affine-proof resource findings.
- [x] (2026-08-20 23:12Z) Passed the full release build, all 2,804 MLIR tests,
  169 OpenQASM tests, the issue-to-`jeff` regression, lint, and diff checks.
- [x] (2026-08-20 23:13Z) Passed all 4,301 configured release CTests; one QDMI
  test skipped under its own condition.
- [x] (2026-08-20 23:15Z) Published draft pull request #2203 from the signed
  rebased branch and linked the changelog entry to the pull request.
- [x] (2026-08-21 22:13Z) Restored compile-time normalization of constant
  negative indices, restricted aliases to `const` values, and charged the
  distinctness budget only for Presburger queries.
- [x] (2026-08-21 22:27Z) Passed the 169 OpenQASM and 126 compiler tests in
      debug and release builds, the reported Clang-Tidy check, repository lint,
      and a second Ponytail review with no further findings.

## Surprises & Discoveries

- Observation: The existing constant-range `scf.for` path still reconstructs its
  source induction value with `i128` multiply and add operations. Evidence:
  `OpenQASMToQCEmitter.cpp` creates `counterWide`, `offset`, and `inductionWide`
  in `emitFor`. The direct positive-range path must replace this code as well as
  the quantum-index guards.
- Observation: PR #2135 constructs QC modules directly, but QFT-shaped source
  requires relational constraints such as `j >= i + 1`. Independent intervals
  cannot prove that `q[j]` and `q[i]` differ.
- Observation: An access before an assignment in a repeating loop is not safe on
  later iterations. The semantic analyzer must find loop mutations before it
  analyzes the first iteration.
- Observation: Emitting proven loop arithmetic as `i64` still left index
  materialization that the `jeff` pipeline could not accept. Emitting the affine
  expression in MLIR's `index` type removes that conversion while an `i64` view
  remains available to ordinary scalar consumers.
- Observation: Existing compiler fixtures classified arbitrary runtime quantum
  indices as accepted input. The new frontend contract makes those fixtures
  semantic-rejection cases and moves the induction-index fixture into the
  `jeff`-compatible set.
- Observation: Mixed signed and unsigned range endpoints use unsigned comparison
  semantics. A direct signed-index loop is valid only when signed endpoints are
  proven nonnegative.
- Observation: Pairwise Presburger checks run before emitter resource checks.
  Wide affine barriers and broadcast gate calls therefore need their own
  semantic comparison limit.
- Observation: Mutable scalar aliases can retain runtime `i64` arithmetic after
  the semantic analyzer accepts the captured affine initializer. Evidence: the
  source `int x = i + 1; h q[x];` failed `jeff` emission because
  `jeff.int_binary_op` requires the same operand and result types.
- Observation: The distinctness counter charged constant differences before the
  no-solver fast path. A broadcast of one proven operand pair could exhaust the
  budget even though each comparison was constant.
- Observation: OpenQASM defines constant negative indices relative to the end of
  a register. Compile-time normalization preserves that source behavior without
  runtime guards.

## Decision Log

- Decision: Place all proof state in `OpenQASMSemantics.cpp` and expose only
  proof results through the typed OpenQASM frontend data. Rationale: OpenQASM
  owns the optional quantum-index extension; QC and QCO must remain independent
  of source-language policy. Date/Author: 2026-08-20, Codex.
- Decision: Use `MLIRPresburger` integer emptiness checks over affine loop
  domains. Rationale: bounds and same-register distinctness in triangular loops
  require relational reasoning, while this existing dependency provides exact
  integer checks. Date/Author: 2026-08-20, Codex.
- Decision: Admit only compile-time positive loop steps into the proven range
  path and ignore step congruence in the proof domain. Rationale: this is a safe
  over-approximation that covers the benchmark patterns and keeps lowering to a
  direct positive-step `scf.for`. Date/Author: 2026-08-20, Codex.
- Decision: Preserve generic loop and dynamic classical-index behavior.
  Rationale: only quantum-register indexing is optional in the base OpenQASM
  contract, and unrelated runtime behavior is outside issue #2188. Date/Author:
  2026-08-20, Codex.
- Decision: Scan each repeating loop body for assignments before semantic
  analysis and exclude those scalar symbols from affine facts throughout that
  loop. Rationale: generation checks alone see only the first source-order
  iteration and could accept an index that is unsafe after the back edge.
  Date/Author: 2026-08-20, Codex.
- Decision: Admit direct affine expressions and `const` aliases, but reject
  mutable scalar aliases in quantum indices. Rationale: the emitter cannot
  recover a mutable alias's affine initializer without more typed state, while a
  direct expression or `const` declaration has the required compile-time form.
  Date/Author: 2026-08-21, Codex.
- Decision: Use direct positive-range lowering for mixed signed and unsigned
  endpoints only when every signed endpoint is proven nonnegative. Rationale:
  this condition makes unsigned promotion value-preserving; otherwise generic
  lowering retains the source semantics. Date/Author: 2026-08-20, Codex.
- Decision: Limit each barrier or gate call to 1,024 affine distinctness solver
  queries and resolve constant differences without charging the limit.
  Rationale: source limits permit wide operand lists, but repeated constant
  proofs do not consume Presburger resources. Date/Author: 2026-08-21, Codex.

## Outcomes & Retrospective

The frontend now proves affine quantum indices against nested Presburger loop
domains and rejects unproved bounds or operand distinctness. The emitter is four
lines smaller than before this work: it directly emits proven `index` arithmetic
and no longer emits quantum bounds, wrapping, or distinctness guards. The
semantic analyzer is larger because it owns the source-level proof, including
exact relational domains, overflow checks, `const` aliases, and loop-back-edge
mutation safety. Interval bounds or source-order generation checks would not
cover the required triangular and `j - step` benchmark shapes.

The exact issue #2188 program reaches `jeff` without `i128`, `arith.select`, or
`cf.assert`. PR #2135 remained at the audited head and no benchmark source was
changed. After the rebase, the full release build, 169 OpenQASM tests, all 2,804
MLIR tests, all 4,301 configured release CTests, and `uvx nox -s lint` passed.
One QDMI test skipped under its own condition. The first lint run formatted
changed files; the second run passed without edits. Draft pull request #2203
contains the signed result.

## Context and Orientation

`mlir/lib/Target/OpenQASM/OpenQASMSemantics.cpp` converts parsed syntax into the
typed structures declared in `mlir/include/mlir/Target/OpenQASM/Frontend.h`.
Before this change, quantum-register references carried an optional
`dynamicIndex` with no static-safety invariant.
`mlir/lib/Dialect/QC/Translation/OpenQASMToQCEmitter.cpp` then emits runtime
checks for such references and lowers only fully constant source ranges through
`scf.for`; other ranges become `scf.while`.

An affine expression is a constant plus integer coefficients multiplied by
active loop induction variables. A Presburger domain is a set of integer
solutions to linear equalities and inequalities. For an inclusive OpenQASM range
`[lower:step:upper]`, this work records `lower <= induction <= upper` when
`step` is a positive compile-time integer. Omitting the step's congruence class
adds possible values, so a property proved over this larger set is still safe.

The semantic analyzer will prove a quantum index by showing that adding either
`index < 0` or `index >= width` makes the active domain empty. It will prove
same-register operands distinct by showing that adding `left == right` makes the
domain empty. It will also prove every supported arithmetic node fits the signed
64-bit representation emitted by this frontend. The analyzer folds `const`
aliases before it builds an affine form. A pre-scan excludes scalars assigned
anywhere in a repeating loop so that the proof remains valid after the loop back
edge.

## Plan of Work

Add a small private affine-expression representation and active-domain state to
`OpenQASMSemantics.cpp`. Build forms for integer constants, active induction
variables, folded `const` aliases, negation, addition, subtraction, constant
multiplication, and value-preserving integer casts. Use arbitrary-precision
coefficients while constructing constraints. Reject unsupported forms only when
they reach a quantum index or are needed to mark a loop as a proven range.

During `analyzeFor`, recognize positive constant steps and affine start and stop
expressions whose direct lowering arithmetic fits signed 64-bit. Append the
induction variable and its inclusive bounds to the active domain while analyzing
the body, then restore the outer domain. Mark the resulting typed `ForStatement`
as a proven positive range. Find loop-body assignments before body analysis so
an outer scalar cannot supply a proof that fails on a later iteration.

Rename the optional expression in `QubitReference` from `dynamicIndex` to
`provenIndex`. Resolve every nonconstant quantum index through the affine proof
before constructing this reference. Check pairwise same-register gate and
explicit barrier operands during semantic analysis. Unprovable accesses report
clear source diagnostics and never enter the typed program.

In `OpenQASMToQCEmitter.cpp`, emit a proven affine integer expression without
the generic signed-overflow checks. Use this path for proven quantum indices and
proven positive loop endpoints. Lower such source loops directly to `scf.for`
with an exclusive upper bound of `stop + 1`. Remove quantum bounds and
distinctness assertions and revise preflight emission costs. Leave the checked
index path for classical bits and the existing generic loop path intact.

Add semantic tests for accepted and rejected expressions, emitter tests for the
shape of issue #2188 and nested benchmark patterns, and an in-process compiler
pipeline regression that produces a `JeffProgram`. Update the documented
OpenQASM subset, `UPGRADING.md`, and `CHANGELOG.md` without editing generated or
template-managed files.

## Concrete Steps

From the repository root, edit the frontend header, semantic analyzer, emitter,
their CMake dependency lists, and the matching OpenQASM and compiler tests. Run
the narrow test binaries after building them:

    cmake --build --preset release --target mqt-core-mlir-unittest-openqasm-target
    ./build/release/mlir/unittests/Target/OpenQASM/mqt-core-mlir-unittest-openqasm-target
    cmake --build --preset release --target mqt-core-mlir-unittests-compiler
    ./build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler

After focused tests pass, run:

    cmake --build --preset release
    ctest --test-dir build/release/mlir --output-on-failure
    uvx nox -s lint
    git diff --check

## Validation and Acceptance

The issue #2188 source must analyze, verify as QC, and pass the default compiler
pipeline to `jeff`. Its emitted QC must contain direct qubit loads inside
`scf.for` and no `i128`, quantum-index `cf.assert`, or negative-index
`arith.select`. A nested QFT-shaped loop with `j` ranging from `i + 1` through
the final register index must prove both bounds and operand distinctness. A
QFT-adder-shaped `j - step` index and reverse `N - 1 - i` index must also pass.
Constant negative indices must normalize relative to the register width.

Programs with possibly out-of-range, possibly aliased, measurement-derived,
mutated, nonlinear, or unsupported-step quantum indices must fail semantic
analysis with a precise diagnostic. Dynamic classical indexing and generic
runtime loops that do not supply quantum indices must retain their current
behavior. All existing test suites and lint checks must pass.

Revision note (2026-08-21): The review follow-up removed mutable affine aliases,
restored constant negative-index normalization, and limited the resource counter
to actual solver queries. Focused debug and release tests, Clang-Tidy, lint, and
the follow-up simplicity review passed.

## Idempotence and Recovery

All edits and tests are local and repeatable. Build output stays under `build/`.
No remote state is changed. If a proof rule proves too little, add a focused
failing test before extending that rule; do not restore runtime quantum guards.
If Presburger integration proves unsuitable, keep the typed-interface and test
changes isolated until the proof implementation is replaced, and record the
reason in this plan.
