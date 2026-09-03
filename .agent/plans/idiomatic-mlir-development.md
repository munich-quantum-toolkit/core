# Institutionalize idiomatic MLIR development

Status: historical implementation record.

## Goal and scope

MQT Core consumes MLIR but currently applies the repository's broad C++ lint
policy to MLIR code. That policy can recommend code that the MLIR project
explicitly rejects, most visibly `const` on values and operations in the mutable
intermediate-representation graph. After this work, contributors and coding
agents can find a short MQT-owned MLIR policy, repository tools reject the most
common objective violation, and the code base has a clean baseline.

The result is visible in three ways. The documentation explains the chosen rules
and vocabulary, a scoped `AGENTS.md` presents the high-impact subset to coding
agents, and `uvx prek run disallow-const-mlir-handles --all-files` rejects
prohibited `const` declarations without a new dependency.

## Constraints

- `mlir/.clang-tidy` currently inherits every root style family and only
  subtracts selected checks. Evidence: its `Checks` list has no leading `-*`, so
  `readability-non-const-parameter` remains active even though
  `*-const-correctness` is disabled.

- a suffix-only `*Op` gate is unsound. Evidence: `PlanOp`,
  `ForbiddenModifierBodyOp`, and `CBitModifierBodyOp` are ordinary project
  types, not MLIR operation handles. A text check cannot distinguish them from
  generated MLIR wrappers.

- the existing pre-commit configuration already uses the native `pygrep`
  language. Evidence: the `disallow-caps` hook provides the exact
  dependency-free mechanism needed for the source gate.

- the first gate expression matched prefixes of longer types. Evidence: it
  reported `ValueRange` and `OwningOpRef`; adding word boundaries made the
  repository baseline pass without excluding those useful types.

- the system Python selected by a direct lint preset lacks the repository's
  nanobind package. Evidence: the first lint configure stopped at
  `find_package(nanobind)`, while `uv run --no-sync cmake --fresh --preset lint`
  selected the managed environment and configured successfully.

- changed-file clang-tidy needs the complete non-unity lint build. Evidence: its
  first pass could not find generated dialect headers; after
  `uv run --no-sync cmake --build --preset lint`, the same analysis completed.

- `llvm-prefer-static-over-anonymous-namespace` requires a static helper to live
  outside the anonymous namespace, not merely to gain the `static` specifier.
  Evidence: clang-tidy diagnosed both forms until the two touched helpers were
  placed in the named namespace with internal linkage.

- `mqt-cc` registered only MLIR's pass-manager options. Evidence: its help
  omitted the documented assembly-printer and context debug options until their
  standard MLIR option groups were registered.

## Decisions

- Treat `docs/development.md` and `docs/mlir/development.md` as the canonical
  policy, while agent files and configuration are condensed views or
  enforcement. Rationale: one normative source prevents duplicated guidance from
  drifting.

- Keep the root formatting policy and use a group-wise clang-tidy allowlist for
  defects plus selected style checks. Rationale: MQT deliberately differs from
  upstream LLVM formatting, while broad inherited style checks are the source of
  conflicting advice.

- Use one `pygrep` hook rather than a custom AST checker. Rationale: explicitly
  named core handles and views are lexical, and the repository already has the
  tool. `const auto` and typed operation wrappers need semantic type information
  and remain policy-only until repeated violations justify a custom checker.

- Keep this ExecPlan implementation-specific and use GitHub issues as the
  campaign source of truth. Rationale: the agreed plan explicitly selected an
  issue tree for follow-up audits.

## Outcome and validation

General and MLIR policies, terminology, scoped guidance, a source gate for
mutable handles, and an aligned clang-tidy profile were added. The source
cleanup removed avoidable const qualifiers without changing runtime APIs.
`mqt- cc` exposes standard MLIR debug options.

Release and non-unity builds, CI-equivalent C++ lint, MLIR and binding suites,
stubs, documentation, link checks, and source-gate positive/negative checks
passed. Further contract, performance, and terminology work is tracked by issue
`#2250`; documentation-comment migration is tracked by `#2267`.

## Code and ownership

`AGENTS.md` is the repository-owned instruction file for coding agents.
`docs/contributing.md`, `docs/ai_usage.md`, and `docs/tooling.md` are generated
from the MQT templates repository and must not be edited here. New policy pages
therefore live in repository-owned files and are linked from `docs/index.md` and
`docs/mlir/index.md`.

`.clang-tidy` is the broad repository policy. `mlir/.clang-tidy` is the nearest
configuration for the compiler collection, while
`mlir/lib/Dialect/QIR/Execution/.clang-tidy` contains external-ABI exceptions.
`bindings/mlir/.clang-tidy` is a sibling configuration and cannot inherit the
MLIR file through directory ancestry. It is therefore a regular, byte-identical
mirror of `mlir/.clang-tidy`; a shared comment marks both copies.

An MLIR value, operation, block, or region is a small C++ handle into a mutable
intermediate-representation graph. Adding C++ `const` to one handle does not
make the referenced graph immutable, so MLIR rejects that model. MLIR types and
attributes are immutable value objects and are not part of the prohibition.

## Acceptance

The policy is accepted when a reader can navigate from the main documentation to
the general development policy, MLIR policy, and glossary. The scoped agent
guide must stay below 100 lines and contain the no-const rule, pass and verifier
contracts, test policy, debugging sequence, and glossary rule.

The source gate is accepted when it rejects leading and trailing `const` on MLIR
`Value`, `TypedValue`, `BlockArgument`, `OpResult`, `ValueRange`,
`OperandRange`, `ResultRange`, `Operation`, `Block`, `Region`, and `ModuleOp`,
but accepts `Type`, `Attribute`, and ordinary C++ types. The policy must also
prohibit type-deduced handles and typed operation wrappers that the text gate
cannot identify reliably.

The implementation is accepted when lint, release build, MLIR unit tests, MLIR
documentation, full documentation, and link checking pass. If an external
service or hosted check cannot run locally, report it separately rather than
claiming it passed.

## Interfaces

This work adds no runtime library dependency, custom executable, or new test
framework. A dedicated tooling group pins cpp-linter 1.13.0 and nanobind 3.x for
local CI reproduction. The remaining tools are MyST/Sphinx, clang-tidy 22.1.8,
pre-commit's existing `pygrep` language, GoogleTest, CMake, and CTest. Its only
command-line change exposes standard MLIR debug options through `mqt-cc`.
