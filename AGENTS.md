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
- Write for readers who know neither the implementation nor its design history.
  Apply Orwell's six writing rules throughout. Prioritize technical accuracy,
  reader understanding, and natural flow over brevity or rigid style rules.
- Use ASD-STE100 principles for implementation comments, API summaries, and
  parameter, return, and error descriptions. Do not require STE in guides or
  detailed API explanations, or claim formal compliance. Follow the
  [prose policy](docs/development.md#prose-and-terminology).
- Explain the task and concepts before implementation details. Integrate
  additions into the existing explanation. Preserve useful examples,
  limitations, ownership, and failure behavior; link to the owning reference for
  specialist detail.
- Use established quantum-computing and software terms, explaining unfamiliar
  terms where readers first need them. Identify MQT-specific names as such.
  Consult `docs/glossary.md`, but check definitions against primary sources
  before adding or changing terms. Keep API-object and schema definitions in
  their owning references. Use “program qubits” for unmapped circuit qubits;
  distinguish placement, allocation, and fault-tolerant encoding.
- Keep prompts, conversations, review instructions, and iteration history out of
  documentation, comments, and PR descriptions. Describe the current system and
  the reasons readers need to understand it. Put release history and migration
  advice in their designated documents.
- Follow the [testing priorities](docs/development.md#testing-priorities):
  extend a small end-to-end workflow before adding a unit test. Assert
  observable semantics and resource bounds, not private structure or a
  particular rewrite. A bug fix does not need a separate test when an existing
  workflow can expose it.
- Cover native semantics in C++ once. Python tests own SDK integration, binding
  conversions, ownership, and Python failure behavior; do not repeat native
  parameter sweeps across the Python/dependency matrix. Keep workloads small.
- Before removing a test, check its callers, history, and surviving assertions.
  Keep distinct phase, wire-order, ownership, failure, and complexity contracts;
  equal line coverage alone does not establish redundancy. Put tests in the
  owning subsystem and use subprocesses only for irreducible CLI behavior. Do
  not put MLIR tests under production tools or enable an optional production
  tool solely to satisfy a subprocess test.
- Diagnose failed checks before changing code or build policy. Distinguish
  defects from stale output, dependency mismatches, and service failures. Use
  supported presets and keep machine setup in local configuration.
- Remove obsolete scaffolding and suppressions. Retain only necessary, narrowly
  scoped workarounds with a technical reason, reproducer, and removal condition.
- Follow the
  [release documentation policy](docs/development.md#release-documentation).
  Describe user-facing changes and required migrations in PR descriptions. Do
  not change `CHANGELOG.md` or `UPGRADING.md` except in release-preparation PRs
  or when explicitly asked. Document migrations from released APIs, not
  intermediate unreleased designs.
- Changelog entries name the PR and every contributing author, for example
  `([#123]) ([**@username**])`, with link definitions at the bottom.
- Never commit or print secrets or personal data. Use documented environment
  variables and repository secrets.

## C++ and MLIR

Use C++20 and CMake 3.28 or newer. The development policy owns detailed include,
comment, data-structure, diagnostic, and debugging guidance.

- Use `#pragma once`, direct includes, and standard-library facilities before
  adding abstractions. Use C typedefs such as `size_t` and `uint64_t` without
  `std::`. Do not use C-style casts, including casts to `void`.
- Use LLVM streams and buffers for I/O and `StringRef` for borrowed text. Use
  existing synchronized diagnostics for concurrent logging. Do not include STL
  stream headers, including in tests. Keep owned strings when ownership is
  required.
- Use `//` for ordinary code comments. Never use `///` for them.
- Use `///` for Doxygen documentation, including structural commands. Start
  descriptions with a short summary paragraph, separated from further content by
  a blank `///` line. Marker-only blocks need no summary. Follow the
  [C++ documentation policy](docs/development.md#c-documentation-comments) for
  exceptions.
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
  `qco::verifyLinearity` at boundaries; avoid redundant rewrite guards. The same
  verifier checks positional quantum function and region results;
  exactly-one-use alone does not establish wire correspondence.
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

- Use Google-style docstrings, including in nanobind binding strings. Start each
  with a short summary on one line, then a blank line before any further
  content. Follow the
  [Python docstring policy](docs/development.md#python-docstrings).
- Fix `ruff` and `ty` diagnostics instead of suppressing them unless a
  documented exception is necessary.
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

Choose build configurations that exercise the change, including affected
libraries and bindings. Before removing a build setting, check its history and
reproduce the configuration that required it.

C++ lint checks every line of each changed C++ file against `origin/main` by
default; changed-line clang-tidy alone is insufficient. Inspect which files ran.
Stage new files before all-files lint, or pass them explicitly. Rerun hooks
after their fixes. Keep pass and option documentation aligned with actual scope,
defaults, supported shapes, limitations, and failure modes.

Inspect the final diff and status. Check changed comments and docstrings against
the documentation policy. Exclude generated, secret, and unrelated files. Tie
validation to the final code: rerun affected checks following edits and
distinguish passes from skipped, blocked, or pending checks. Report checks run
and their outcomes; stop after required gates pass unless a concrete remaining
risk justifies more validation.

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
- Do not monitor CI unless explicitly requested.
- Reviews focus on correctness, contracts, maintainability, tests,
  documentation, licensing, and validation rather than optional process
  metadata.
