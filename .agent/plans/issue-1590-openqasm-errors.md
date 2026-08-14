# Propagate OpenQASM semantic errors without exceptions

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

The OpenQASM frontend should diagnose invalid programs even when its translation
unit is compiled with C++ exceptions disabled. After this change, semantic
analysis reports the same first diagnostic and produces the same typed program
for valid input, but uses MLIR result values rather than throwing and catching a
private exception. This is observable through the existing frontend tests, a
late recursive-gate failure test, and a direct `-fno-exceptions` compilation of
the semantic analyzer.

### Progress

- [x] (2026-08-13 21:48Z) Reconstructed the semantic analyzer change directly on
      current `main` without replaying the obsolete stacked commits.
- [x] (2026-08-13 21:48Z) Replaced exception unwinding with
      `mlir::LogicalResult` and `mlir::FailureOr<T>` while retaining one stored
      diagnostic.
- [x] (2026-08-13 21:48Z) Added the late recursive-gate regression with exact
      message and source-position assertions.
- [x] (2026-08-13 21:52Z) Built and passed all 88 frontend tests and all 164
      tests in the complete OpenQASM target binary.
- [x] (2026-08-13 21:52Z) Compiled the complete `obj.MLIROpenQASMFrontend`
      target with `-fno-exceptions` as the final compiler option.
- [x] (2026-08-13 21:52Z) Verified the semantic source has no exception syntax
      and completed repository lint, `git diff --check`, and final diff
      inspection.

### Surprises & Discoveries

- Observation: The obsolete branch depended on unrelated fixed-width angle work,
  so replaying its commits would reintroduce changes outside this analyzer.
  Evidence: applying only the semantic commit as a behavioral reference required
  adapting helper signatures to the current, simpler scalar representation.
- Observation: Current `main` has 87 `OpenQASMFrontendTest.*` cases, and all
  passed after the result propagation compiled. Evidence: the focused test run
  reported `87 tests from 1 test suite` and `PASSED` before adding the new
  regression.

### Decision Log

- Decision: Keep one translation-unit-local assignment-and-propagation macro and
  undefine it after `SemanticAnalyzer`. Rationale: MLIR's result types remain
  explicit at function boundaries while repetitive `FailureOr<T>` extraction
  stays compact; no shared repository abstraction is introduced. Date/Author:
  2026-08-13 / Codex.
- Decision: Store only the first `Diagnostic` in `SemanticAnalyzer` and stop
  every phase on failure. Rationale: this exactly preserves the frontend's
  existing single-semantic-diagnostic contract and avoids partial typed
  programs. Date/Author: 2026-08-13 / Codex.
- Decision: Leave public types and `analyzeOpenQASM` unchanged. Rationale:
  callers should not participate in this internal control-flow refactor.
  Date/Author: 2026-08-13 / Codex.

### Outcomes & Retrospective

The analyzer now propagates semantic failures explicitly while preserving the
public frontend result and first-diagnostic behavior. The focused and complete
OpenQASM test suites pass, and a separate build proved that every source in the
frontend object library compiles with exceptions disabled. The product change is
limited to the analyzer and one focused regression; no public interface, shared
helper, or compiler policy changed.

### Context and Orientation

`mlir/lib/Target/OpenQASM/OpenQASMSemantics.cpp` converts the parser's syntax
program into a `TypedProgram`. Before this work it used a private exception to
leave deeply nested semantic helpers after recording a `Diagnostic`. A
`Diagnostic` contains the message, source location, and include stack returned
through the unchanged `AnalysisResult` API.

MLIR provides two small result types in `mlir/Support/LogicalResult.h`.
`mlir::LogicalResult` represents success or failure for procedures, and
`mlir::FailureOr<T>` additionally carries a value on success. The analyzer uses
the former for mutating helpers and the latter for helpers that compute IDs,
constants, types, resolved operands, or statements. `std::optional` remains
appropriate only where absence is a successful outcome, such as a nonconstant
index.

The focused regression belongs in
`mlir/unittests/Target/OpenQASM/test_openqasm_semantics.cpp`. It defines a gate
that calls itself. The analyzer discovers this only after analyzing the body,
making it a useful proof that late failures propagate without constructing a
typed program.

This task changes only this plan, the semantic analyzer, and that focused test.
It does not change compiler flags, public headers, repository-wide exception
policy, or later OpenQASM semantic work. Follow `AGENTS.md` and
`docs/ai_usage.md`; this plan does not authorize pushing a branch or editing a
pull request.

### Plan of Work

In `OpenQASMSemantics.cpp`, remove the private semantic exception and exception
headers. Give `SemanticAnalyzer` one optional diagnostic. Make `fail` record
that diagnostic and return `mlir::failure()`. Convert procedural helpers to
`LogicalResult` and value-producing helpers to `FailureOr<T>`, then immediately
propagate every failure so the first diagnostic cannot be overwritten.

Use one local macro to evaluate a `FailureOr<T>`, return failure if necessary,
and move its value into a named local. Keep the macro next to the analyzer and
undefine it immediately after the class. In `run`, execute each phase in order
and construct `TypedProgram` only if version validation, depth validation, body
analysis, gate-cycle validation, and output finalization all succeed.

Add the recursive-gate test after the existing uninitialized-scalar-output test.
Assert failure, exactly one diagnostic, the exact recursive-gate message, and
line 4, column 3.

### Concrete Steps

From the repository root, configure and build the focused test target with the
repository wrapper:

    MLIR_DIR=/path/to/llvm/lib/cmake/mlir ./.agent/run.sh cmake --preset release
    ./.agent/run.sh cmake --build --preset release --target mqt-core-mlir-unittest-openqasm-target -j2

Run the focused suite and then the whole binary:

    build/release/mlir/unittests/Target/OpenQASM/mqt-core-mlir-unittest-openqasm-target --gtest_filter='OpenQASMFrontendTest.*'
    build/release/mlir/unittests/Target/OpenQASM/mqt-core-mlir-unittest-openqasm-target

Compile the semantic source in a separate build context with `-fno-exceptions`.
The compile command must place `-fno-exceptions` after any configured
`-fexceptions` flag so the final option is authoritative. Then verify the source
contains no exception syntax and run repository checks:

    rg -n '\bthrow\b|\bcatch\b|<exception>|<stdexcept>' mlir/lib/Target/OpenQASM/OpenQASMSemantics.cpp
    ./.agent/run.sh uvx nox -s lint
    git diff --check

The `rg` command succeeds by printing nothing. Both test commands report no
failures, and lint plus `git diff --check` exit with status zero.

### Validation and Acceptance

Acceptance requires all frontend semantic tests and the complete OpenQASM target
test binary to pass. The recursive-gate test must observe exactly
`recursive custom gate definition involving 'recursive'` at line 4, column 3,
with no second diagnostic.

The analyzer object must compile while exceptions are disabled, and the source
must contain no `throw`, `catch`, `<exception>`, or `<stdexcept>`. A valid input
must still yield a populated `AnalysisResult::program`; an invalid semantic
input must yield one diagnostic with its existing message, source location, and
include stack. No public header or caller may change.

### Idempotence and Recovery

Configuration, builds, tests, searches, lint, and diff checks are repeatable.
Use a separate build directory or a copied compiler invocation for the
`-fno-exceptions` proof so normal release configuration is not altered. If a
build stops after a source edit, rerun the same target; Ninja rebuilds only
out-of-date objects. Never discard unrelated changes or operate in another
task's worktree.

### Artifacts and Notes

The final focused and complete test evidence was:

    [==========] Running 88 tests from 1 test suite.
    [  PASSED  ] 88 tests.
    [==========] Running 164 tests from 2 test suites.
    [  PASSED  ] 164 tests.

The separate build's semantic compile command ended with:

    ... -fexceptions ... -fno-exceptions -o OpenQASMSemantics.cpp.o -c OpenQASMSemantics.cpp

Repository lint ended with `nox > Session lint was successful`, and the source
exception search plus `git diff --check` produced no output.

### Interfaces and Dependencies

The public `analyzeOpenQASM`, `AnalysisResult`, `Diagnostic`, and `TypedProgram`
interfaces remain unchanged. The implementation depends only on MLIR's
`LogicalResult` and `FailureOr<T>` from `mlir/Support/LogicalResult.h`. There is
no new shared result type, utility header, external library, or repository-wide
compiler option.

Revision note: Created this standalone plan after reconstructing the analyzer on
current `main`, then completed it with the focused, full-suite,
exception-disabled, lint, and diff-validation evidence.
