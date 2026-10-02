# Return explicit compiler-target errors

Status: historical implementation record.

## Goal and scope

Compiler-target construction and FoMaC target discovery must report invalid data
without throwing through the MLIR compiler libraries. C++ callers receive
`llvm::Expected`; Python callers continue to receive ordinary `ValueError`
exceptions at the nanobind boundary. This pull request is independently based on
`main` and does not change the repository-wide exception build policy.

## Constraints

- Target validation returns explicit C++ errors; the nanobind boundary
  translates them to Python exceptions.
- Historical stub generation did not complete. Runtime signatures matched the
  checked-in constructors, but that check did not replace stub generation.

## Decisions

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

## Outcome and validation

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

## Code and ownership

`mlir/include/mlir/Compiler/Target.h` and `mlir/lib/Compiler/Target.cpp` define
the immutable target model. `mlir/lib/Compiler/FoMaCAdapter.cpp` constructs
targets from device snapshots. `bindings/mlir/register_mlir.cpp` is the Python
language boundary. Mapping and native synthesis consume validated targets and
therefore unwrap only fixtures or locally known-valid constructions.

## Acceptance

Invalid C++ target metadata must return an `llvm::Error` with a useful message;
valid targets must preserve their mapping and synthesis behavior; invalid Python
construction must still raise `ValueError`; and the changed compiler-target core
must not throw for validation failures. Focused C++ and Python tests, repository
lint, `git diff --check`, and exact-head CI must pass before merge readiness.

## Interfaces

The public factories return `llvm::Expected<DurationUnit>`,
`llvm::Expected<Site>`, `llvm::Expected<SiteTuple>`,
`llvm::Expected<Operation>`, and `llvm::Expected<CompilerTarget>`.
`compilerTargetFromDevice` returns `llvm::Expected<CompilerTarget>`. LLVM/MLIR
22 and C++20 remain the only relevant language and compiler dependencies.
