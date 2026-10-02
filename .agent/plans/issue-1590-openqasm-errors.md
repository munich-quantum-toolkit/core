# Propagate OpenQASM semantic errors without exceptions

Status: historical implementation record.

## Goal and scope

The OpenQASM frontend should diagnose invalid programs even when its translation
unit is compiled with C++ exceptions disabled. After this change, semantic
analysis reports the same first diagnostic and produces the same typed program
for valid input, but uses MLIR result values rather than throwing and catching a
private exception. This is observable through the existing frontend tests, a
late recursive-gate failure test, and a direct `-fno-exceptions` compilation of
the semantic analyzer.

## Constraints

- The obsolete branch depended on unrelated fixed-width angle work, so replaying
  its commits would reintroduce changes outside this analyzer. Evidence:
  applying only the semantic commit as a behavioral reference required adapting
  helper signatures to the current, simpler scalar representation.

- Current `main` has 87 `OpenQASMFrontendTest.*` cases, and all passed after the
  result propagation compiled. Evidence: the focused test run reported
  `87 tests from 1 test suite` and `PASSED` before adding the new regression.

## Decisions

- Keep one translation-unit-local assignment-and-propagation macro and undefine
  it after `SemanticAnalyzer`. Rationale: MLIR's result types remain explicit at
  function boundaries while repetitive `FailureOr<T>` extraction stays compact;
  no shared repository abstraction is introduced.

- Store only the first `Diagnostic` in `SemanticAnalyzer` and stop every phase
  on failure. Rationale: this exactly preserves the frontend's existing
  single-semantic-diagnostic contract and avoids partial typed programs.

- Leave public types and `analyzeOpenQASM` unchanged. Rationale: callers should
  not participate in this internal control-flow refactor.

## Outcome and validation

The analyzer now propagates semantic failures explicitly while preserving the
public frontend result and first-diagnostic behavior. The focused and complete
OpenQASM test suites pass, and a separate build proved that every source in the
frontend object library compiles with exceptions disabled. The product change is
limited to the analyzer and one focused regression; no public interface, shared
helper, or compiler policy changed.

## Code and ownership

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

## Acceptance

Acceptance requires all frontend semantic tests and the complete OpenQASM target
test binary to pass. The recursive-gate test must observe exactly
`recursive custom gate definition involving 'recursive'` at line 4, column 3,
with no second diagnostic.

The analyzer object must compile while exceptions are disabled, and the source
must contain no `throw`, `catch`, `<exception>`, or `<stdexcept>`. A valid input
must still yield a populated `AnalysisResult::program`; an invalid semantic
input must yield one diagnostic with its existing message, source location, and
include stack. No public header or caller may change.

## Interfaces

The public `analyzeOpenQASM`, `AnalysisResult`, `Diagnostic`, and `TypedProgram`
interfaces remain unchanged. The implementation depends only on MLIR's
`LogicalResult` and `FailureOr<T>` from `mlir/Support/LogicalResult.h`. There is
no new shared result type, utility header, external library, or repository-wide
compiler option.
