# MQT Core Agent Guide

Use relevant sections of the [development policy](docs/development.md) and the
[AI usage policy](docs/ai_usage.md). Load supporting documents only as needed.

## Repository layout

- Public C++ headers live in `include/mqt-core/`; implementations live in
  `src/`.
- `bindings/` contains nanobind bindings; `python/mqt/core/` contains Python
  code and generated stubs.
- `mlir/` contains dialects, transforms, tools, and unit tests. It requires
  LLVM/MLIR 23.1 or newer.
- `test/` contains C++ and Python tests, generally mirroring production code.
- `docs/` contains Sphinx and MyST documentation; `json/` contains schemas and
  data. `cmake/` and `CMakePresets.json` define supported builds.
- Keep generated build output in `build/` and out of commits.

## Working principles

- Inspect the working tree first and preserve user changes. Keep the diff
  focused on the assigned task; avoid unrelated cleanup, formatting, and
  upgrades.
- Complete authorized work, including relevant validation and fixes, without
  routine approval checkpoints. Ask only about consequential ambiguity or work
  outside scope; cite any instruction that blocks progress.
- For behavioral changes, trace producers, shared helpers, and consumers. Fix
  the owning layer and reuse existing code or dependency facilities. Do not
  reconstruct a shared contract in each frontend, exporter, or caller.
- State supported inputs, failure behavior, and ownership before expanding an
  API. Prefer the smallest complete solution while preserving correctness and
  runtime efficiency. Fewer lines alone do not prove a simpler design.
- Existing code is evidence, not authority over current policy. Follow
  repository policy, enforcing configuration, and explicitly adopted upstream
  guidance in that order.
- Apply Orwell's six writing rules and the relevant ASD-STE100 principles in
  every category of prose: choose short familiar words, remove needless words,
  use active voice, keep each sentence direct, and prefer clarity over rigid
  application of a style rule. Do not claim formal ASD-STE100 compliance.
- Use established technical terms, one term per concept, and the spelling in
  `docs/glossary.md`. Preserve project names such as `jeff` and `jeff-mlir`.
  Update the glossary when introducing or changing public or ambiguous terms.
- Document contracts, reasons, ownership, numerical limits, and useful examples.
  Remove repetition of code, boilerplate parameters, change narration, and
  unsupported assurances. Use symbol references instead of brittle line
  pointers. Keep prompts and review history out of code and API docs.
- Test changed behavior and concrete regressions, not implementation details.
  Low-impact edits need no new tests. Before weakening a test, check history,
  callers, invariants, and resource limits; equal line coverage or a shared
  failure does not establish redundancy.
- Put tests in the owning subsystem's test tree. Use direct unit tests for
  semantic contracts and subprocesses only for irreducible CLI behavior. Do not
  put MLIR tests under production tools or enable an optional production tool
  solely to satisfy a subprocess test.
- Diagnose failed checks before changing code or build policy. Distinguish
  defects from stale output, dependency mismatches, and service failures. Use
  supported presets and keep machine setup in local configuration.
- Remove obsolete scaffolding and suppressions. Retain only necessary, narrowly
  scoped workarounds with a technical reason, reproducer, and removal condition.
- Follow the
  [release documentation policy](docs/development.md#release-documentation).
  Record notable changes under Unreleased. Fold refinements to never-released
  functionality into its feature entry; document migrations from released APIs,
  not intermediate unreleased designs.
- Changelog entries name the PR and every contributing author, for example
  `([#123]) ([**@username**])`, with link definitions at the bottom.
- Never commit or print secrets or personal data. Use documented environment
  variables and repository secrets.
- Do not edit files marked as generated from an external template. Contribute
  those changes to the MQT templates repository or its update workflow.

## C++ and MLIR

Use C++20 and CMake 3.28 or newer. The development policy owns detailed include,
comment, data-structure, diagnostic, and debugging guidance.

- Use `#pragma once`, direct includes, and standard-library facilities before
  adding abstractions. Use C typedefs such as `size_t` and `uint64_t` without
  `std::`. Do not use C-style casts, including casts to `void`.
- Use `///` for Doxygen documentation and `//` for ordinary implementation and
  namespace closing comments. Preserve trailing `//!<` or `///<`, inline block
  comments, and block documentation inside continued macros.
- Use `moduleOp` instead of the C++20 keyword `module` for an MLIR module
  handle. Generally give non-public data members a trailing underscore.
- Never add `const` to MLIR `Value` forms, range views, `Operation`, `Block`,
  `Region`, `ModuleOp`, or typed operation wrappers, including via `const auto`.
  Copy cheap handles and views. Do not add top-level `const` to by-value
  parameters. Ordinary C++ objects retain normal const-correctness.
- In the `mlir` namespace, prefer suitable LLVM facilities and unqualified names
  imported by `mlir/Support/LLVM.h`; include their defining headers directly.
- Trace IR contracts through builders/frontends, verifiers, interfaces,
  transformations, and consumers. TableGen is not the complete contract.
- Passes must not crash on valid IR and successful output must verify. Verifiers
  own operation invariants; conversions and exporters diagnose their supported
  subsets. Failed rewrite matches leave IR unchanged. Do not silently broaden
  support or emit partial success.
- QCO qubits and QTensors have exactly one use in valid IR. Validate with
  `qco::verifyLinearity` at boundaries; avoid redundant rewrite guards.
  Linearity does not imply positional wire correspondence.
- Search upstream MLIR before adding operations, interfaces, traits,
  conversions, or utilities. Reuse folding, canonicalization, and analysis
  facilities. Keep custom state only for a concrete correctness or complexity
  requirement.
- Preserve deterministic output; never expose pointer or unordered traversal
  order. Require evidence for performance rewrites.
- Use GoogleTest/CTest, not `lit` or FileCheck. Assert semantics and required
  normal forms. Exact trees, text, target choices, and operation counts need a
  contract reason. Preserve phase, wire identity, numerical limits, and negative
  cases when replacing an oracle.
- Review the MLIR policy and `mlir/.clang-tidy` on major LLVM/MLIR upgrades.

## Python and bindings

- Use Google-style docstrings. Fix `ruff` and `ty` diagnostics instead of
  suppressing them unless a documented exception is necessary.
- Preserve supported Python APIs unless a breaking change is authorized. Keep
  optional integration imports lazy where the binding already does so.
- Choose finite-shot tolerances with low false-failure probability; keep
  expected values away from tolerance boundaries.
- Regenerate stubs after every binding change with `uvx nox -s stubs`. Never
  edit generated `.pyi` files manually. MLIR handle rules apply in bindings too.

## Build and validation

Run checks relevant to the change and required by repository policy. Local
machine overrides may select another supported preset.

| Task                                            | Command                                                            |
| ----------------------------------------------- | ------------------------------------------------------------------ |
| Configure / build native release                | `cmake --preset release` / `cmake --build --preset release`        |
| Native tests                                    | `ctest --preset release`                                           |
| C++ lint before pushing C++ changes             | `uvx nox -s cpp-lint`                                              |
| Install development dependencies                | `uv sync --locked --only-group dev`                                |
| Install editable package                        | `uv sync --inexact --no-dev --no-build-isolation-package mqt-core` |
| Focused Python tests                            | `uv run --no-sync pytest <file-or-filter>`                         |
| Supported Python sessions                       | `uvx nox -s tests` / `uvx nox -s minimums`                         |
| MLIR reference generation                       | `cmake --build --preset release --target mlir-doc`                 |
| Complete executable documentation               | `uvx nox --non-interactive -s docs`                                |
| External documentation links                    | `uvx nox --non-interactive -s docs -- -b linkcheck`                |
| Repository lint before handoff                  | `uvx nox -s lint`                                                  |

Use `debug` for debug builds. Run component binaries directly with GoogleTest
filters when useful, such as
`build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler`. Python
3.14 test sessions are `tests-3.14` and `minimums-3.14`.

C++ lint checks every line of each changed C++ file against `origin/main` by
default; changed-line clang-tidy alone is insufficient. Inspect which files ran.
Keep pass and option documentation aligned with actual scope, defaults,
supported shapes, limitations, and failure modes.

Inspect the final diff and status. Exclude generated, template-managed, secret,
and unrelated files. Tie validation to the final code: rerun affected checks
following edits and distinguish passes from skipped, blocked, or pending checks.
Report checks run and their outcomes; stop after required gates pass unless a
concrete remaining risk justifies more validation.

## Plans and audits

Keep an [ExecPlan](.agent/PLANS.md) in `.agent/plans/` when design or
coordination needs a durable record, or on request. Small tasks need no plan
file.

Use [audit guidance](.agent/AUDITS.md) for contract reviews. Save findings in
`.agent/audits/` only when a durable record is useful or requested.

## Performance evidence

Benchmark only when requested or needed for a concrete performance question;
prefer existing tools. Keep ad hoc harnesses, data, and plots outside the
repository unless explicitly requested. Support performance claims with the
workload, baseline, environment, measurements, and limits.

## Git and public contributions

- Begin issue, PR, and commit titles with the established gitmoji. Commit
  subjects are imperative, target 50 characters, never exceed 72, and have no
  final period. Separate the body with a blank line and explain reasons and
  constraints.
- Preserve human authorship trailers. Record AI assistance with `Assisted-by`,
  never an AI `Co-authored-by` trailer.
- Act only within the human's delegated scope. Do not push, open or merge PRs,
  post GitHub text, or otherwise change remote state without authorization.
  Request fresh authorization for external actions outside that scope.
- Scoped authorization to create or update public text permits those actions
  without per-message approval. Humans must review agent-assisted work before
  acceptance or merging and remain accountable for the result.
- Every agent-authored or agent-edited public body starts exactly with
  `🤖 *AI text below* 🤖`. Titles are exempt.
- Do not use agents on issues labeled `good first issue` or generate spam,
  repetitive reviews, or unreviewed contributions.
- Pushing or opening a PR does not request CI monitoring. Unless asked to
  monitor, report the available status at handoff and stop.
- Reviews focus on correctness, contracts, maintainability, tests,
  documentation, licensing, and validation rather than optional process
  metadata.
