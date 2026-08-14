# Preserve unobserved quantum operations during target compilation

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

Target compilation currently removes every quantum operation whose final qubit
value is only discarded. That is a useful default for executable programs, but
it prevents callers such as compiler benchmarks from compiling state-preparation
circuits without adding synthetic measurements. After this change, callers of
`QCOProgram::compileForTarget` and Python's `QCOProgram.compile_for_target` can
opt into preserving such unobserved quantum operations while the existing
default remains unchanged.

The behavior is visible by compiling an unmeasured two-qubit circuit that
applies H and reset to the first qubit and H to the second qubit. The default
result contains no unitary operations. With
`preserve_unobserved_quantum_operations=True`, the mapped, target-native result
still contains unitary operations and passes target-conformance verification.

## Progress

- [x] (2026-08-14 19:22Z) Audited target-pipeline cleanup and identified the two
      labeled QCO canonicalization patterns that perform quantum dead-gate
      elimination.
- [x] (2026-08-14 19:22Z) Confirmed that no open issue or pull request already
      implements a target-compilation preservation option.
- [x] (2026-08-14 19:24Z) Added a stable shared pattern label and selectively
      filtered unobserved quantum-operation elimination in QCO cleanup.
- [x] (2026-08-14 19:30Z) Added the default-compatible C++ and Python
      target-compilation options API.
- [x] (2026-08-14 19:34Z) Added compiler and Python regressions plus
      target-compilation documentation.
- [x] (2026-08-14 19:38Z) Built the affected targets, regenerated Python stubs,
      and passed the complete compiler unit-test binary.
- [x] (2026-08-14 19:40Z) Passed all 49 tests in `test/python/test_mlir.py`
      after rebuilding the Python environment with its standard QDMI devices.
- [x] (2026-08-14 19:42Z) Passed the complete repository lint after applying its
      formatting changes.
- [x] (2026-08-14 19:47Z) Built the complete HTML documentation with Sphinx
      warnings treated as errors.
- [x] (2026-08-14 19:52Z) Rebuilt the final API state and repeated the complete
      compiler and MLIR Python test sets.
- [x] (2026-08-14 21:16Z) Corrected the regression description, completed the
      preservation example, and documented the cleanup parameter's selective
      scope after review.
- [x] (2026-08-14 21:16Z) Repeated all affected validation. The compiler output
      ended with `PASS: [  PASSED  ] 235 tests.`, Python ended with
      `PASS: 49 passed in 4.58s`, and the documentation build ended with
      `PASS: nox > Session docs was successful in 59 seconds.` Lint runs
      immediately after prose edits had `FAIL` status after their format hooks
      changed files. Each no-change rerun passed; one exact output was
      `PASS: nox > Session lint was successful in 6 seconds.` The `UNAVAILABLE`
      count was zero.
- [x] (2026-08-17 06:55Z) Rebasing onto `main` at `53325922b` resolved the
      binding include and changelog conflicts introduced by the removal of the
      legacy compiler translation and the new LLVM build support.
- [x] (2026-08-17 07:05Z) Reconfigured and rebuilt against the current release
      preset. The complete compiler binary passed 134 tests, all 49 MLIR Python
      tests passed, and stub generation reproduced the checked-in stubs.
- [x] (2026-08-17 07:09Z) Generated the MLIR documentation fragments and passed
      the complete warnings-as-errors documentation build.
- [x] (2026-08-17 07:10Z) The first lint run formatted this updated plan and had
      `FAIL` status. The immediate no-change rerun ended with
      `PASS: nox > Session lint was successful in 6 seconds.` No required check
      was `UNAVAILABLE`.

## Surprises & Discoveries

- Observation: The final generic `createRemoveDeadValuesPass()` is not the pass
  that erases unobserved quantum circuits. Evidence: pass tracing retains the
  mapped gates until the post-mapping canonicalizer, whose `SinkOp` dead-gate
  pattern walks backward through the quantum SSA chain.
- Observation: Reset operations provide a second dead-gate-elimination path.
  Evidence: both `SinkOp.cpp` and `ResetOp.cpp` register a pattern named
  `DeadGateElimination`, so filtering only the sink pattern is incomplete.
- Observation: LLVM 22's canonicalizer accepts stable debug labels as a filter.
  Evidence: `createCanonicalizerPass(config, disabledPatterns)` filters patterns
  registered through `RewritePatternSet::addWithLabel` while retaining all
  unrelated canonicalization.
- Observation: Repository `main` removed the worktree-local `.agent/run.sh`
  wrapper. Evidence: the current `AGENTS.md` specifies direct CMake, UV, and Nox
  commands, so this plan now uses those commands as well.
- Observation: The complete documentation session expects generated MLIR
  fragments. Evidence: the first current-main run failed with 19 missing-include
  warnings; `cmake --build --preset release --target mlir-doc` generated the
  fragments, and the immediate rerun passed.

## Decision Log

- Decision: Expose a positive `preserveUnobservedQuantumOperations` option with
  a default of `false`. Rationale: this states the observable contract and keeps
  all existing calls behaviorally compatible. Date/Author: 2026-08-14 / Codex.
- Decision: Filter only unobserved-quantum cleanup patterns instead of skipping
  the full canonicalizer or generic dead-value removal. Rationale: callers still
  receive normalization, common-subexpression elimination, register shrinking,
  and target verification. Date/Author: 2026-08-14 / Codex.
- Decision: Apply the option to both QCO cleanup stages. Rationale: direct
  scalar QCO input can be erased by the pre-mapping cleanup, while mapped
  QC/QTensor input can be erased by the post-mapping cleanup. Date/Author:
  2026-08-14 / Codex.
- Decision: Keep `runDefaultPipeline`, `compile_program`, and `mqt-cc` unchanged
  in this focused change. Rationale: the requested typed QCO API is sufficient,
  and widening every entry point can be a mechanical follow-up without bloating
  the semantic change. Date/Author: 2026-08-14 / Codex.
- Decision: Keep empty allocation/static-sink removal enabled in preservation
  mode. Rationale: the option controls dead-gate elimination only; lifetimes
  with no quantum operations remain ordinary cleanup. Date/Author: 2026-08-14 /
  Codex.

## Outcomes & Retrospective

The implementation preserves sink- and reset-terminated quantum chains only when
requested, while the existing default continues to eliminate both. On current
`main`, the complete compiler binary passed 134 tests, and the MLIR Python
module passed all 49 tests. The first full Python attempt failed because it
reused the stub-generation build, which intentionally disables bundled QDMI
devices. A clean rebuild with the standard device configuration made the three
unrelated device tests pass.

The current compiler and Python evidence is:

    PASS: [==========] 134 tests from 10 test suites ran. (2813 ms total)
          [  PASSED  ] 134 tests.
    PASS: 49 passed in 12.98s
    PASS: nox > Session stubs was successful in 4 seconds.
    PASS: build succeeded.
          nox > Session docs was successful in 20 seconds.

Lint runs immediately after prose edits reported
`Files were modified by following hooks` and ended with
`nox > Session lint failed.` after applying format changes. Each immediate
no-change rerun passed. One exact output was:

    PASS: nox > Session lint was successful in 6 seconds.

The resolved formatting runs had `FAIL` status. No required check was
`UNAVAILABLE`.

The first documentation run after the rebase had `FAIL` status because the new
worktree did not contain generated MLIR documentation fragments. The run ended
with
`build finished with problems, 19 warnings (with warnings treated as errors).`
Building the `mlir-doc` target generated those files, and the next documentation
run passed as shown above.

The option is deliberately limited to the typed QCO target-compilation API. The
generic compiler pipeline, `compile_program`, and `mqt-cc` retain their current
default cleanup behavior and can expose the same option in a focused follow-up
if a concrete caller requires it.

## Context and Orientation

MQT Core represents target-independent quantum programs in the QCO MLIR dialect.
A quantum operation consumes one qubit SSA value and produces the next; an
unobserved chain ends in `qco.sink`. During canonicalization,
`mlir/lib/Dialect/QCO/IR/QubitManagement/SinkOp.cpp` recursively erases gates
feeding a sink, and `mlir/lib/Dialect/QCO/IR/Operations/ResetOp.cpp` can erase
dead gates feeding a reset.

`mlir/lib/Support/Passes.cpp` defines the standard QCO cleanup pipeline.
`mlir/lib/Compiler/TargetCompilation.cpp` runs that cleanup before decomposition
and mapping and again after mapping. `mlir/include/mlir/Compiler/Programs.h` and
`mlir/lib/Compiler/Programs.cpp` expose target compilation as a typed QCO API;
`bindings/mlir/register_mlir.cpp` exposes the same operation to Python. The
checked-in `python/mqt/core/mlir.pyi` file is generated and must be refreshed by
the repository's stub-generation session rather than edited manually.

A stable pattern label is a string attached to an MLIR rewrite pattern. The
canonicalizer can disable every pattern carrying that label without relying on
anonymous C++ type names. The label will live in
`mlir/include/mlir/Dialect/QCO/QCOUtils.h` so both pattern registration and the
cleanup pipeline share one spelling.

## Plan of Work

First, define a shared unobserved-quantum-operation elimination pattern label in
`QCOUtils.h`. Register the sink and reset dead-gate patterns with that label
while leaving unrelated canonicalization unfiltered.

Second, add an optional `eliminateUnobservedQuantumOperations` parameter to
`populateQCOCleanupPipeline`, defaulting to `true`. When false, construct the
canonicalizer with the shared label in its disabled-pattern list. Retain phase
normalization, CSE, QTensor shrinking, and `createRemoveDeadValuesPass()`.

Third, add `TargetCompilationOptions` in
`mlir/include/mlir/Compiler/TargetCompilation.h` with
`preserveUnobservedQuantumOperations = false`. Pass the inverse of that field to
both cleanup stages. Keep the current `QCOProgram::compileForTarget` overload
and make it delegate to a new options-based overload, preserving source
compatibility for positional timing/statistics arguments.

Fourth, replace the direct nanobind member adapter for `compile_for_target` with
a small lambda. The Python keyword
`preserve_unobserved_quantum_operations=False` constructs the C++ options and
then calls the new overload. Regenerate the `.pyi` file with the repository's
stub session.

Finally, add regression tests. A C++ target-pipeline test must prove that the
default removes an unobserved H/reset/H program and that preservation mode
retains the mapped unitary and reset chains. A focused cleanup test must prove
that the filter preserves gates feeding sinks and resets. A Python test must
exercise the new keyword and inspect the resulting QCO IR. Update
target-compilation documentation with the intended use and the consequence that
preserved operations have no classical observation.

## Concrete Steps

Run all commands from the repository root.

After editing, configure the release build and build the compiler tests. Set
`MLIR_DIR` to the local MLIR package directory when it is not already set:

    MLIR_DIR=/path/to/lib/cmake/mlir cmake --preset release
    cmake --build build/release --target mqt-core-mlir-unittests-compiler

Run the focused compiler test binary with a filter for the new tests, then run
the complete compiler test binary:

    build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler --gtest_filter='*Preserve*:*Unobserved*'
    build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler

Build the Python extension, regenerate stubs through the repository Nox session,
and run the MLIR Python tests. Do not edit the generated stub directly:

    uv sync --locked --only-group dev
    MLIR_DIR=/path/to/lib/cmake/mlir uv sync --inexact --no-dev --no-build-isolation-package mqt-core
    uvx nox -s stubs
    uv run --no-sync pytest -q test/python/test_mlir.py

End validation with:

    cmake --build --preset release --target mlir-doc
    uvx nox --non-interactive -s docs
    uvx nox -s lint

Record exact command output and any documented environment limitation in this
plan before opening the pull request.

## Validation and Acceptance

Acceptance requires all of the following observable behavior. The existing
default target compilation must still erase a completely unobserved quantum
chain. Passing preservation mode must retain at least one target-native unitary
operation. The resulting module must pass MLIR verification and the
target-conformance pass. A reset-terminated chain must also survive, proving
that both labeled patterns are filtered. The default path must remain compact
and unchanged. Existing compiler and focused Python tests must remain green.

## Idempotence and Recovery

All source edits are additive or default-preserving and can be reapplied safely.
CMake configuration and builds are idempotent. If stub generation changes more
than `python/mqt/core/mlir.pyi`, inspect the generator output and retain only
expected generated changes. If a test reveals that a required target pass
depends on dead-gate elimination, keep the default path unchanged and narrow the
preservation-mode test rather than weakening target verification.

## Interfaces and Dependencies

The new C++ interface is:

    struct TargetCompilationOptions {
      bool preserveUnobservedQuantumOperations = false;
    };

    void populateTargetCompilationPipeline(
        OpPassManager&, const CompilerTarget&,
        const TargetCompilationOptions& = {});

`QCOProgram` retains its existing overload and gains an overload accepting the
options object before timing and statistics booleans. Python gains one keyword
on `QCOProgram.compile_for_target`:

    preserve_unobserved_quantum_operations: bool = False

No new third-party dependency is required. The implementation relies on the
already required LLVM/MLIR canonicalizer pattern-filter API.
