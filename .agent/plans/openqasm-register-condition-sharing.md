# Keep wide OpenQASM register conditions linear

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

OpenQASM 2 represents feedback as `if (creg == integer) gate`. The typed
frontend lowers that comparison into short-circuit Boolean QC operations. When
many gates reuse a wide register condition, materializing the tree for every
gate makes the QC program quadratic in the register width and number of
conditions. The generic QC-to-OpenQASM exporter also cannot express the
result-bearing `scf.if` tree directly.

After this change, identical pure condition roots share one SSA value while the
classical register snapshot is unchanged. The OpenQASM exporter recognizes only
the complete register-equality form produced by the frontend and emits it as
`if (c == N)`. It also fuses a measurement with its proven direct CBit
destination. Conditions of 151 and 301 bits therefore remain linear and can be
consumed by Qiskit's OpenQASM importer without recursive Boolean expansion.

The feature is deliberately narrower than general classical-expression export.
Partial, duplicate, mixed-register, dynamically indexed, modified, or stale
snapshots continue to fail closed or use the ordinary scalar form.

## Progress

- [x] (2026-08-19) Isolated the OpenQASM portion of the former aggregate
  classical-control branch on the current head of #2158.
- [x] (2026-08-19) Ported classical storage matching from memrefs to first-class
  CBit allocations, loads, and stores.
- [x] (2026-08-19) Added snapshot-scoped sharing of pure literal, static-bit,
  not, and/or condition roots in the OpenQASM-to-QC emitter.
- [x] (2026-08-19) Preserved arbitrary-width register equality and compact
  measurement destinations after the QC cleanup forwards CBit loads.
- [x] (2026-08-19) Ported positive and fail-closed tests to CBit and passed all
  183 QC translation tests and all 167 OpenQASM frontend/target tests.
- [x] (2026-08-19) Keyed reconstructed equalities by the consuming `scf.if` and
      rejected store-backed observations that execute after their consumer.
- [x] (2026-08-19 19:54Z) Rebase the focused branch after #2158 landed, rebuild
  both affected targets, and pass all 183 QC translation tests and all 167
  OpenQASM frontend and target tests.
- [x] (2026-08-19 19:54Z) Pass the post-rebase Markdown and full repository lint
  checks and `git diff --check`.
- [ ] If #2169 lands before publication, rebase semantically onto its finalized
  angle semantics and rerun the full validation matrix. #2169 remains open.
- [ ] Add the final changelog link after the new pull request number exists.

## Surprises & Discoveries

- Observation: CBit cleanup correctly forwards a load from the most recent
  store. For an OpenQASM 2 register initialized to zero with one measured high
  bit, a 301-bit equality can consequently reduce to the measurement SSA value.
  The exporter must recover the destination register from the CBit store rather
  than assume that every bit still has a load operation.

- Observation: Recovering omitted bits from zero initialization is safe only if
  every preceding write to the register is represented by the matched snapshot.
  Otherwise emitting a whole-register equality could constrain an overwritten
  bit to zero and change behavior. The matcher rejects that case.

- Observation: Caching recursively emitted right-hand operands is invalid.
  Short-circuit operands can be defined inside an `scf.if` region and do not
  dominate a later statement. Only complete statement-condition roots are
  cached.

- Observation: A mutation must create a new cache generation. Clearing the map
  alone is insufficient because saved region state could otherwise resurrect a
  value from an older classical snapshot.

- Observation: Snapshot validity belongs to the consuming branch, not only to
  its condition SSA value. The same value may feed another branch after an
  intervening CBit write, so exporter matches are keyed by consumer operation.

- Observation: Cleanup can expose a measurement result before its destination
  store. A future store does not establish the register snapshot seen by an
  earlier branch and must be rejected during reconstruction.

## Decision Log

- Decision: Keep this work separate from target capability declarations,
  mapping, and direct Qiskit structured-control export. Rationale: OpenQASM
  register compaction is independently testable and the other concerns have
  distinct reviewers and dependencies.

- Decision: Use `!cbit.reg<N>` as the only classical storage model and stack
  this branch after the focused mapping follow-up to #2158. Rationale: #2158
  makes CBit the canonical, non-aliasing representation, while the mapping
  parent lets the complete Benchpress path be tested without mixing either
  implementation into this review.

- Decision: Cache only structurally canonical, pure condition roots and clear
  the cache on every classical bit or scalar mutation and at structured-region
  boundaries. Rationale: This provides linear frontend output without turning
  CSE into a correctness mechanism or sharing values across dominance/memory
  boundaries.

- Decision: Reconstruct only complete register equality, with arbitrary-width
  `APInt` values, and retain exact snapshot checks. Rationale: The 41 Benchpress
  gaps all use this OpenQASM 2 form. General SCF result export and arbitrary
  classical expressions belong to the Qiskit structured-control task.

- Decision: When cleanup forwards CBit loads, accept a stored SSA bit as
  register provenance only when there is exactly one static destination and all
  omitted bits come from a zero-initialized, otherwise unmodified snapshot.
  Rationale: This retains the compact representation after normal QC cleanup
  while rejecting ambiguous or stale state.

- Decision: Fuse only a measurement destination whose other uses are entirely
  within the matched equality and whose store occurs in the same block with only
  constants in between. Rationale: Qiskit's OpenQASM importer accepts
  `c[i] = measure q;` but does not accept the generic temporary-plus-assignment
  form. The narrow ordering rule prevents an early register write from changing
  a prior classical snapshot.

## Outcomes & Retrospective

The focused implementation now uses CBit throughout. Repeated OpenQASM
conditions share at the frontend until a classical mutation, and the exporter
emits exact register equality for widths 1, 64, 151, and 301. The implementation
retains negative coverage for changed snapshots and adds a zero-initialization
guard for writes that cleanup did not represent in the condition. Consumer-
specific matches and store-before-consumer validation prevent reuse across a
later classical state.

The local branch now follows merged #2158 and the rebased focused mapping
follow-up. The rebase applied without a source conflict, and both complete
affected test binaries pass with their prior counts. The branch remains
unpublished. #2169 is still open and changes OpenQASM semantics and tests, so
integration is required only if it lands before this branch is published.

## Context and Orientation

`mlir/lib/Target/OpenQASM/OpenQASMSemantics.cpp` recognizes the source-level
whole-register comparison and produces a typed Boolean condition. The QC emitter
in `mlir/lib/Dialect/QC/Translation/OpenQASMToQCEmitter.cpp` turns typed
conditions into SSA and owns the snapshot-scoped cache. The reverse translation
in `mlir/lib/Dialect/QC/Translation/TranslateQCToOpenQASM3.cpp` recognizes the
narrow complete-equality pattern and writes the direct OpenQASM expression.

The focused tests live in
`mlir/unittests/Target/OpenQASM/test_openqasm_emitter.cpp`,
`mlir/unittests/Target/OpenQASM/test_openqasm_semantics.cpp`, and
`mlir/unittests/Dialect/QC/Translation/test_openqasm3_emission.cpp`.

## Plan of Work

First, canonicalize only pure typed condition trees and reuse complete roots in
one region and classical-state generation. Invalidate the cache at every store,
mutable scalar assignment, measurement destination, and structured-control
boundary.

Second, match the frontend's complete register conjunction in the reverse
translator. Require one CBit register, static unique indices, every relevant
bit, side-effect-free result expressions, and no write between evaluation and
use. Use zero initialization only for omitted bits whose register history
contains no unrepresented write. Emit the expected value with arbitrary-width
`APInt`.

Third, pair a matched measurement result with its CBit store only when every
additional use belongs to the folded equality. Emit the measurement directly to
that destination and omit the folded condition-expression operations.

Finally, exercise widths 1, 64, 151, and 301, repeated conditions, mutation
boundaries, shared expression trees, dead trees, measurement multi-use, partial
and stale snapshots, frontend initialization rules, and strict reparse.

## Concrete Steps

From the repository root, configure/build the two focused targets and run them:

    cmake --build build/release --target \
      mqt-core-mlir-unittest-qc-translation \
      mqt-core-mlir-unittest-openqasm-target --parallel 8
    build/release/mlir/unittests/Dialect/QC/Translation/\
      mqt-core-mlir-unittest-qc-translation
    build/release/mlir/unittests/Target/OpenQASM/\
      mqt-core-mlir-unittest-openqasm-target

Then run repository formatting/document checks and whitespace validation:

    uvx nox -s lint
    git diff --check

The current base includes #2158. If #2169 lands before publication, rebase this
branch, resolve its OpenQASM semantic and test overlap, rebuild, and rerun the
same matrix.

## Validation and Acceptance

Acceptance requires:

- full focused suites pass;
- the 1/64/151/301 cases emit a direct measurement destination and exact decimal
  register equality;
- emitted source reparses in strict OpenQASM 3.1 mode and translates back to QC;
- identical conditions share before a mutation and do not share after it;
- partial, duplicate, mixed, dynamic, side-effecting, or stale forms never get
  compacted as a whole-register equality;
- no source module is mutated by export; and
- formatting, documentation lint, and `git diff --check` pass.

## Idempotence and Recovery

All build and test commands are repeatable. The translator works on existing
modules without mutating them. If the future rebase conflicts with #2169, retain
its finalized semantic analyzer and tests first, then reapply only the CBit
condition-sharing and reverse-emission behavior described here. Do not rewrite
or push #2158.

## Artifacts and Notes

The important scale proof is that a 301-bit comparison no longer produces one
fresh condition tree per guarded gate. The direct emitted condition remains the
decimal OpenQASM literal, including values wider than 64 bits.

## Interfaces and Dependencies

No public Python or compiler-target API changes. The implementation depends on
the CBit dialect from #2158, MLIR SCF/Arith/MemoryEffect interfaces, and LLVM
`APInt`. It does not depend on the Qiskit structured-control writer.

Plan revision note: Recorded the completed post-#2158 rebase and validation, and
kept the semantic rebase for #2169 as an open publication task because #2169 has
not landed.
