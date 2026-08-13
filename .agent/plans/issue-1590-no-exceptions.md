# Return explicit compiler-target errors

This ExecPlan is a living document maintained according to `.agent/PLANS.md`.

## Purpose / Big Picture

Compiler-target construction and FoMaC target discovery must report invalid data
without throwing through the MLIR compiler libraries. C++ callers receive
`llvm::Expected`; Python callers continue to receive ordinary `ValueError`
exceptions at the nanobind boundary. This pull request is independently based on
`main` and does not change the repository-wide exception build policy.

## Progress

- [x] (2026-08-12 17:00Z) Replace fallible target constructors with named
  `create` factories and propagate failures through LLVM error types.
- [x] (2026-08-12 18:00Z) Preserve Python exception behavior and update C++,
  command-line, mapping, synthesis, and compiler callers and tests.
- [x] (2026-08-12 19:00Z) Update the C++ target-compilation guide and generic
  compiler-target changelog entry.
- [x] (2026-08-12 22:03Z) Use in-place nanobind constructors, remove redundant
  private constructor tags, and remove unrelated QIR subprocess coverage.
- [x] (2026-08-12 22:12Z) Validate the simplified implementation with focused
      C++ and Python suites, generated-signature inspection, repository lint,
      and whitespace checks.

## Surprises & Discoveries

- `Target.cpp` owned the target model's exception syntax, while the nanobind
  registration file already provided the correct boundary for translating
  explicit C++ errors into Python exceptions.
- Official MLIR bindings use `None` for ordinary absence and exceptions for
  failed checked construction. nanobind recommends in-place `__init__` bindings
  over value-returning `nb::new_` when the value can be constructed in
  caller-provided storage.
- The isolated stub-generation session could not finish because GitHub returned
  an empty response while downloading an unchanged build dependency. The
  worktree package built successfully, and nanobind's runtime signatures for all
  eight constructors match the checked-in stubs exactly.

## Decision Log

- Use `llvm::Expected<T>` for fallible construction and `llvm::Error` for
  validation helpers. These types are native to the existing LLVM dependency and
  retain explanatory diagnostics without an exception-capable ABI.
- Keep routing accessors as preconditioned fast queries. Construction validates
  the target graph once, and callers that already hold validated vertices do not
  need a second fallible API.
- Keep Python `ValueError` behavior by consuming LLVM errors inside nanobind.
  LLVM result wrappers are not exposed in the Python API.
- Keep Python constructor spelling and placement-construct each value only after
  its `llvm::Expected` succeeds. Returning `None` would discard the validation
  diagnostic and weaken every constructor's return type.
- Keep validated value constructors private, but do not add empty passkey tag
  types. Private access already prevents callers from bypassing the factories.
- Keep QIR subprocess tests focused on QIR output. Compiler-target and Python
  tests cover target discovery and validation without adding unrelated `mqt-cc`
  option cases to that script.

## Outcomes & Retrospective

Target values are constructed only after validation succeeds. The compiler,
FoMaC adapter, command-line tool, Python bindings, mapping, and synthesis paths
now propagate explicit errors. Valid target behavior and Python-facing failure
semantics remain unchanged. Repository-wide exception disabling is deliberately
left to a later, separately reviewable change. All 232 compiler, 81 mapping, and
21 target-synthesis tests passed. Six focused Python target tests passed against
the rebuilt extension. Repository lint and `git diff --check` passed. No tracked
stub changed, and runtime signature inspection confirmed the four target
overloads and four nested metadata constructors retain their checked-in Python
signatures.

## Context and Orientation

`mlir/include/mlir/Compiler/Target.h` and `mlir/lib/Compiler/Target.cpp` define
the immutable target model. `mlir/lib/Compiler/FoMaCAdapter.cpp` constructs
targets from device snapshots. `bindings/mlir/register_mlir.cpp` is the Python
language boundary. Mapping and native synthesis consume validated targets and
therefore unwrap only fixtures or locally known-valid constructions.

## Plan of Work

Add static factories for `DurationUnit`, `Site`, `SiteTuple`, `Operation`, and
`CompilerTarget`. Move validation into helpers returning LLVM errors and keep
constructors private. Make FoMaC target discovery return an expected target,
propagate failures through compiler and command-line callers, and translate
errors to `nanobind::value_error` in bindings. Bind the unchanged Python
constructors with in-place `__init__` functions that construct only after an
expected result succeeds. Update focused tests for all constructor overloads,
valid nested metadata, and exact invalid-input diagnostics. Document the C++
error-handling pattern in the target-compilation guide and the existing generic
compiler-target changelog entry.

## Milestones

The first milestone replaces exception-throwing target construction with LLVM
result types. It is complete when invalid metadata produces a descriptive
`llvm::Error` and valid fixtures retain the same target values.

The second milestone propagates those results through FoMaC, compiler,
command-line, mapping, synthesis, and Python call sites. It is complete when no
caller discards an error and Python still reports invalid input as `ValueError`.

The final milestone updates the target-compilation guide and validates the
standalone change. It is complete when the focused compiler, mapping, synthesis,
and Python suites, generated-stub checks, repository lint, and whitespace checks
pass.

## Concrete Steps

Run cache-producing commands through `.agent/run.sh` from the repository root:

    ./.agent/run.sh cmake --preset release
    ./.agent/run.sh cmake --build build/release --target \
      mqt-core-mlir-unittests-compiler \
      mqt-core-mlir-unittest-mapping \
      mqt-core-mlir-unittest-target-synthesis mqt-core-mlir-bindings
    ./build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler
    ./.agent/run.sh uv run --no-sync pytest test/python/test_mlir.py -k target
    ./.agent/run.sh uvx nox -s stubs
    ./.agent/run.sh uvx nox -s lint
    git diff --check

## Validation and Acceptance

Invalid C++ target metadata must return an `llvm::Error` with a useful message;
valid targets must preserve their mapping and synthesis behavior; invalid Python
construction must still raise `ValueError`; and the changed compiler-target core
must not throw for validation failures. Focused C++ and Python tests, repository
lint, `git diff --check`, and exact-head CI must pass before merge readiness.

## Idempotence and Recovery

Configuration, builds, tests, and lint are repeatable in this worktree. A failed
factory result leaves no partially initialized target. The implementation does
not modify global state, external services, or another task's worktree.

## Artifacts and Notes

The focused test summaries are:

    [  PASSED  ] 232 tests.
    [  PASSED  ] 81 tests.
    [  PASSED  ] 21 tests.
    6 passed in 3.82s
    nox > Session lint was successful

The runtime nanobind metadata reports four `CompilerTarget.__init__` overloads
and one unchanged `__init__` signature for each nested metadata class. The
generated Python stub files have no diff.

## Interfaces and Dependencies

The public factories return `llvm::Expected<DurationUnit>`,
`llvm::Expected<Site>`, `llvm::Expected<SiteTuple>`,
`llvm::Expected<Operation>`, and `llvm::Expected<CompilerTarget>`.
`compilerTargetFromDevice` returns `llvm::Expected<CompilerTarget>`. LLVM/MLIR
22 and C++20 remain the only relevant language and compiler dependencies.

Revision note (2026-08-12): aligned the Python bindings with MLIR and nanobind
checked-construction conventions, removed redundant constructor passkeys, and
removed unrelated QIR command-line coverage.
