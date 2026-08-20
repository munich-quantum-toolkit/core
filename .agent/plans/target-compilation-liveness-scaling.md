# Keep target-compilation liveness bounded

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

Target compilation must finish for large mapped programs without repeating a
whole-module liveness analysis after target-native synthesis. The second QCO
cleanup already removes globally dead values. Target-native synthesis preserves
value signatures and use chains, and the following CSE pass removes trivially
dead local operations. This change removes only the final redundant
`RemoveDeadValues` pass and retains target-conformance verification.

A focused compiler regression compiles real structured control for an explicit
`u`/`cz` target. The compilation lowers non-native one- and two-qubit gates,
keeps a private helper signature and call, and serializes the complete module.
Running `RemoveDeadValues` explicitly on that result must leave the serialized
module byte-for-byte unchanged.

## Progress

- [x] (2026-08-20 04:22Z) Isolated the final post-synthesis `RemoveDeadValues`
      pass as the remaining large-program compilation limit after the mapper
      received its low-memory topological sorter.
- [x] (2026-08-20 04:22Z) Ran an explicit post-compilation differential on 14
      curated target programs; all 14 modules were unchanged by
      `RemoveDeadValues`.
- [x] (2026-08-20 04:38Z) Removed only the final target-pipeline
      `RemoveDeadValues` pass and documented why CSE remains sufficient after
      target-native synthesis.
- [x] (2026-08-20 04:39Z) Added a compiler fixed-point regression with a private
      classical helper, `qco.if`, non-native one-qubit gates, and a non-native
      controlled-X lowered for a `u`/`cz` target.
- [x] (2026-08-20 04:46Z) Ran the complete affected native binaries (1,224/1,224
      passed), full CTest suite (4,321/4,321 passed with one expected skip),
      repository lint session, and `git diff --check`.
- [ ] Publication: after a pull-request number exists, add the concise
      `CHANGELOG.md` entry with its PR and author links. Do not commit a
      placeholder link before publication.

## Surprises & Discoveries

- Observation: The first low-memory mapping run reached the final
  `RemoveDeadValues` pass but the operating system killed the process there.
  Evidence: the same workload completed when the experimental pipeline omitted
  only that pass.
- Observation: The successful `bwt_n37-linear` compile took 232.991 seconds and
  used at most 9,185,083,392 bytes of resident memory. The earlier stock MLIR
  topological scan had remained unfinished after more than 32 minutes 27 seconds
  at 4.018 GiB resident memory; the explicit-adjacency replacement was also
  killed.
- Observation: Reapplying `RemoveDeadValues` to 14 representative target
  compilation results changed none of them.

## Decision Log

- Decision: Keep this optimization in one child commit above the mapping commit.
  Rationale: the mapper and the target pipeline have independent performance
  limits and different reviewers. Date/Author: 2026-08-20, Codex.
- Decision: Retain the post-synthesis CSE pass. Rationale: synthesis can create
  duplicate or trivially dead local operations even though it preserves global
  value flow. Date/Author: 2026-08-20, Codex.
- Decision: Protect the removal with an end-to-end fixed-point test rather than
  a mock synthesis pass. Rationale: the test must fail if a future real
  target-native synthesis changes global or region liveness. Date/Author:
  2026-08-20, Codex.
- Decision: Compare the full textual module before and after an explicit
  `RemoveDeadValues` pass. Rationale: exact text catches operation, signature,
  attribute, and region changes without weakening the assertion to selected
  operation counts. Date/Author: 2026-08-20, Codex.

## Outcomes & Retrospective

The implementation removes one whole-module pass from the end of target
compilation. The earlier cleanup remains unchanged, target-native synthesis and
CSE remain in the same order, and target-conformance verification still closes
the pipeline.

The direct regression covers real `H`, `X`, and controlled-X synthesis to the
target's `u`/`cz` basis inside `qco.if`. It also keeps a private classical
helper signature and call live across the pipeline. Identity and global-phase
edge semantics remain outside this regression; the synthesis and decomposition
suites own those semantics. Forcing a specific identity or phase decomposition
into this liveness test would couple the fixed-point contract to numerical
synthesis details. The 14-case differential supplies broader fixed-point
evidence.

The seven affected native binaries pass 1,224/1,224 tests: 148 compiler, 493 QCO
IR, 233 decomposition, 91 mapping, 23 native synthesis, 121 optimization, and
115 QCO utility tests. The complete CTest suite passes all 4,321 registered
tests with one expected skip. The repository lint session and `git diff --check`
pass.

## Context and Orientation

`mlir/lib/Compiler/TargetCompilation.cpp` builds the target pipeline. It runs a
QCO cleanup before decomposition and another cleanup after mapping. Each QCO
cleanup ends in `createRemoveDeadValuesPass()`, so the second cleanup
establishes global liveness before target-native synthesis.

`qco::createTargetNativeSynthesis` replaces unsupported unitary shells with
native one- and two-qubit sequences. The rewrite preserves the input/output
qubit signatures and reconnects all existing uses. Global-phase normalization
also runs inside synthesis. CSE follows synthesis and handles duplicate and
trivially dead local operations.

The regression belongs in `mlir/unittests/Compiler/test_compiler_pipeline.cpp`
because it protects the composition of cleanup, mapping, synthesis, CSE, and
target conformance. The dedicated native-synthesis suite continues to test
synthesis semantics in isolation.

## Plan of Work

Remove the final `pm.addPass(createRemoveDeadValuesPass())` after
`createTargetNativeSynthesis` and CSE in `TargetCompilation.cpp`. Add a concise
comment that identifies the second cleanup as the global-liveness fixed point
and limits the CSE claim to trivially dead local operations.

Add one end-to-end compiler regression. Build a module with a live private
classical helper, a call that supplies a runtime `qco.if` condition, and two
qubits threaded through both branches. Put a controlled-X and a Hadamard in one
branch and ordinary X gates in the other. Compile for a target whose explicit
basis contains only `u` and `cz` and whose capabilities admit conditionals.
Verify that native U and controlled-Z forms replaced the non-native gates.

Serialize the complete compiled module. Run `createRemoveDeadValuesPass()`
explicitly with a new pass manager, serialize again, and require exact string
equality.

## Concrete Steps

Run these commands from the repository root:

    cmake --build build/release --parallel 4 --target \
      mqt-core-mlir-unittests-compiler \
      mqt-core-mlir-unittest-qco-ir \
      mqt-core-mlir-unittest-decomposition \
      mqt-core-mlir-unittest-mapping \
      mqt-core-mlir-unittest-target-synthesis \
      mqt-core-mlir-unittest-optimizations \
      mqt-core-mlir-unittest-qco-utils
    build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler \
      --gtest_filter='CompilerPipelineTest.TargetCompilationLeavesDeadValueCleanupAtFixedPoint'
    build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler \
      --gtest_brief=1
    cmake --build build/release --parallel 4
    ctest --test-dir build/release --output-on-failure
    git diff --check
    uvx nox -s lint

## Validation and Acceptance

The focused regression must show that the compiled module retains its private
helper call and `qco.if`, contains native U and controlled-Z operations, and no
longer contains H or X operations. An explicit `RemoveDeadValues` run must
produce identical full-module text.

All affected native binaries, the complete CTest suite, `git diff --check`, and
the repository lint session must pass. No temporary diagnostic instrumentation
may appear in the diff.

## Idempotence and Recovery

The build and test commands are repeatable. The production edit removes one
pipeline line and adds its rationale. The explicit fixed-point pass exists only
inside the regression. If the regression fails after a future synthesis change,
restore the final pipeline cleanup or update synthesis so it preserves global
liveness; do not weaken the exact module comparison without new evidence.

No command in this plan pushes a branch or changes a remote repository.

## Artifacts and Notes

The measured diagnosis is:

    stock MLIR topological scan: >32:27 elapsed, 4.018 GiB resident, unfinished
    explicit outgoing-adjacency sorter: operating-system kill
    low-memory sorter plus final RemoveDeadValues: reached the pass, then killed
    low-memory sorter without final RemoveDeadValues: 232.991 s elapsed,
      9,185,083,392 bytes maximum resident, success
    explicit post-target RemoveDeadValues differential: 14/14 unchanged

Final validation is:

    affected native binaries: 1,224/1,224 passed
    ctest --test-dir build/release --output-on-failure:
      4,321/4,321 passed, one expected skip
    uvx nox -s lint: passed
    git diff --check: passed

The mapping fix is its own parent commit. This plan owns only the final target
pipeline cleanup, its rationale, its compiler regression, and this evidence.

## Interfaces and Dependencies

This change does not add or modify a public interface. It uses the existing
`populateTargetCompilationPipeline`, `createTargetNativeSynthesis`,
`createCSEPass`, and `createRemoveDeadValuesPass` interfaces. The production
pipeline no longer calls the last interface after synthesis; the regression
calls it explicitly to enforce the fixed-point contract.

Plan revision note: Created after the BWT diagnosis separated the mapping and
post-synthesis scalability limits. The changelog link is deliberately deferred
until publication assigns the pull-request number.
