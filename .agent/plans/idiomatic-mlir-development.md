# Institutionalize idiomatic MLIR development

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

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

## Progress

- [x] (2026-08-26 12:10Z) Inspect the current agent guide, generated-document
      boundaries, clang-tidy layering, pre-commit setup, and MLIR const
      baseline.
- [x] (2026-08-26 12:14Z) Add the canonical development policy, MLIR policy,
  glossary, and scoped agent guidance.
- [x] (2026-08-26 12:15Z) Replace MLIR's inherited broad clang-tidy policy with
  the approved allowlist while preserving QIR ABI exceptions.
- [x] (2026-08-27 11:10Z) Audit MLIR handle and view declarations semantically,
  remove every avoidable `const`, and expand the dependency-free named gate.
- [x] (2026-08-27 10:58Z) Run documentation, lint, source-gate, release build,
      CI-equivalent and all-files cpp-linter, and the complete MLIR test label.
- [x] (2026-08-27 11:00Z) Create the authorized GitHub campaign issue tree and
      v4 documentation-comment migration issue, record the successful one-off
      agent smoke test, and submit the focused pull request with required
      disclosure, assignee, and labels.

## Surprises & Discoveries

- Observation: `mlir/.clang-tidy` currently inherits every root style family and
  only subtracts selected checks. Evidence: its `Checks` list has no leading
  `-*`, so `readability-non-const-parameter` remains active even though
  `*-const-correctness` is disabled.
- Observation: a suffix-only `*Op` gate is unsound. Evidence: `PlanOp`,
  `ForbiddenModifierBodyOp`, and `CBitModifierBodyOp` are ordinary project
  types, not MLIR operation handles. A text check cannot distinguish them from
  generated MLIR wrappers.
- Observation: the existing pre-commit configuration already uses the native
  `pygrep` language. Evidence: the `disallow-caps` hook provides the exact
  dependency-free mechanism needed for the source gate.
- Observation: the first gate expression matched prefixes of longer types.
  Evidence: it reported `ValueRange` and `OwningOpRef`; adding word boundaries
  made the repository baseline pass without excluding those useful types.
- Observation: the system Python selected by a direct lint preset lacks the
  repository's nanobind package. Evidence: the first lint configure stopped at
  `find_package(nanobind)`, while `uv run --no-sync cmake --fresh --preset lint`
  selected the managed environment and configured successfully.
- Observation: changed-file clang-tidy needs the complete non-unity lint build.
  Evidence: its first pass could not find generated dialect headers; after
  `uv run --no-sync cmake --build --preset lint`, the same analysis completed.
- Observation: `llvm-prefer-static-over-anonymous-namespace` requires a static
  helper to live outside the anonymous namespace, not merely to gain the
  `static` specifier. Evidence: clang-tidy diagnosed both forms until the two
  touched helpers were placed in the named namespace with internal linkage.
- Observation: `mqt-cc` registered only MLIR's pass-manager options. Evidence:
  its help omitted the documented assembly-printer and context debug options
  until their standard MLIR option groups were registered.

## Decision Log

- Decision: Treat `docs/development.md` and `docs/mlir/development.md` as the
  canonical policy, while agent files and configuration are condensed views or
  enforcement. Rationale: one normative source prevents duplicated guidance from
  drifting. Date/Author: 2026-08-26 / Codex.
- Decision: Keep the root formatting policy and use a group-wise clang-tidy
  allowlist for defects plus selected style checks. Rationale: MQT deliberately
  differs from upstream LLVM formatting, while broad inherited style checks are
  the source of conflicting advice. Date/Author: 2026-08-26 / Codex.
- Decision: Use one `pygrep` hook rather than a custom AST checker. Rationale:
  explicitly named core handles and views are lexical, and the repository
  already has the tool. `const auto` and typed operation wrappers need semantic
  type information and remain policy-only until repeated violations justify a
  custom checker. Date/Author: 2026-08-26 / Codex.
- Decision: Keep this ExecPlan implementation-specific and use GitHub issues as
  the campaign source of truth. Rationale: the agreed plan explicitly selected
  an issue tree for follow-up audits. Date/Author: 2026-08-26 / Codex.

## Outcomes & Retrospective

The repository now owns concise general and MLIR development policies, a seeded
terminology contract, and scoped agent guidance. The existing pre-commit stack
enforces the mutable-handle rule without a new dependency, and the explicit MLIR
clang-tidy profile no longer imports conflicting const inference.

The baseline cleanup is NFC: a one-off AST inventory found 666 candidate
declarations across 80 files and the source review removed all 642 avoidable
uses, including type-deduced values, references, and structured bindings. The 24
remaining matches are const-container pointers, genuinely generic code, or
compiler-synthesized proxy types with no `const` in source. No runtime API
changed. `mqt-cc` now exposes MLIR's standard assembly-printer and context debug
options. Local validation passed: all pre-commit/nox lint hooks; release
configure and build; the non-unity lint configure and build; the CI-equivalent
cpp-linter session and its all-files mode; 2,895 tests under the
`mqt-mlir-unittests` label; 279 focused binding tests; stub validation; MLIR
documentation; the complete warning-as-error documentation build; and link
checking. The source gate also failed on temporary prohibited `OpResult`, range,
and typed-value declarations, accepted an ordinary project type ending in `Op`,
and passed after the violations were removed.

The follow-up contract, test/diagnostic, performance/determinism, and
terminology audits intentionally remain separate campaign issues. No `lit`
infrastructure, custom clang-tidy plugin, container migration, speculative API
rename, or broad formatting patch was added.

The public campaign is tracked by
[issue #2250](https://github.com/munich-quantum-toolkit/core/issues/2250), with
five focused child issues. The foundation is submitted as
[PR #2256](https://github.com/munich-quantum-toolkit/core/pull/2256), assigned
to `@burgholzer` and labeled for MLIR, code quality, documentation, tooling,
pre-commit, and changelog exclusion. All agent-authored public bodies carry the
repository's visible disclosure. The v4 documentation-comment migration is
tracked separately in
[issue #2267](https://github.com/munich-quantum-toolkit/core/issues/2267).

## Context and Orientation

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

## Plan of Work

First, add `docs/development.md`, `docs/mlir/development.md`, and
`docs/glossary.md`. Link them from the existing repository-owned toctrees. Add
short scoped agent files under `mlir/` and `bindings/mlir/`, then teach the root
guide that policy outranks historical precedent and that the nearest scoped
guide applies.

Second, replace the MLIR clang-tidy additions with a leading `-*` allowlist.
Keep compiler diagnostics, static analysis, bug-prone, performance, and
portability families with current exclusions. Keep only the approved LLVM, C++
Core Guidelines, Google, modernization, and readability checks. Disable checks
that infer constness. Retain the deeper QIR file for external ABI naming and C
interface exceptions.

Third, use a one-off semantic inventory to remove `const` from MLIR graph
handles and views in `mlir/` and `bindings/mlir/`, including type-deduced values
and typed operation wrappers. Do not alter unrelated project types such as
`CompilerTarget::Operation`, `qdmi::Operation`, or decomposition planning
records. Add a local `pygrep` hook that rejects explicitly named core handles
and views; keep deductions and wrappers policy-only because a text check cannot
identify them without false positives.

Finally, validate the documentation, hook, clang-tidy configuration, build, and
MLIR tests. Create one public parent issue for the campaign and focused child
issues for policy/tooling, contracts, tests and diagnostics, performance and
determinism, and terminology and APIs. Public bodies include the required AI
disclosure. Submit the implementation as one focused pull request because the
policy, clean baseline, and gate must agree at every commit shipped to users.

## Concrete Steps

Run all commands from the repository root.

1. Edit files only with patches and inspect `git diff --check` after each
   milestone.
2. Run `uvx prek run disallow-const-mlir-handles --all-files`. It must report
   `Passed` after the baseline cleanup.
3. Create a temporary untracked C++ file below `mlir/` containing a prohibited
   declaration, run the hook against that file, observe failure, and remove the
   file.
4. Run `uvx nox -s lint`, `cmake --preset release`,
   `cmake --build --preset release`, and
   `ctest --preset release -L mqt-mlir-unittests`.
5. Run `uvx nox -s cpp-lint`,
   `cmake --build --preset release --target mlir-doc`,
   `uvx nox --non-interactive -s docs`, and `uvx nox -s docs -- -b linkcheck`.
6. Sign commits, verify each with `git verify-commit HEAD`, push the branch,
   create the authorized issues, and submit the pull request assigned to
   `@burgholzer` with appropriate existing labels.

## Validation and Acceptance

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

## Idempotence and Recovery

Documentation edits, searches, lint, builds, and tests are repeatable. The
negative gate test uses one explicit temporary file and removes it immediately.
No generated or template-managed source is edited. GitHub issue creation is not
repeatable, so record returned issue numbers before creating children and check
for an existing matching issue before retrying.

## Artifacts and Notes

The initial baseline search found inappropriate `const` on core MLIR handles in
production code, bindings, and unit tests. It also found legitimate const
references to project records whose names end in `Op` or `Operation`; these must
remain unchanged.

The disposable fresh-agent smoke patch added a nested-operation counter and a
direct GoogleTest. It used `ModuleOp` by value, passed focused build/test and
the complete lint session (including the source gate), and proposed the
conforming subject `✨ Add nested MLIR operation counter`. The temporary
worktree and patch were removed after recording the result in issue #2250.

## Interfaces and Dependencies

This work adds no runtime library dependency, custom executable, or new test
framework. A dedicated tooling group pins cpp-linter 1.13.0 and nanobind 3.x for
local CI reproduction. The remaining tools are MyST/Sphinx, clang-tidy 22.1.8,
pre-commit's existing `pygrep` language, GoogleTest, CMake, and CTest. Its only
command-line change exposes standard MLIR debug options through `mqt-cc`.

Revision note: created the implementation plan after repository inspection,
recorded the selected minimal enforcement design, finalized it with local
validation and public campaign artifacts, and narrowed the source gate after
review showed that suffix-based operation matching caused false positives.
