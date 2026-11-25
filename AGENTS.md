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

- Inspect the working tree and preserve user changes. Complete authorized work,
  including validation, without routine approval checkpoints. Keep reviews
  read-only when requested. Ask only about consequential ambiguity or work
  outside scope; cite any instruction that blocks progress.
- Challenge feature scope and design before polishing code. Preserve explicit
  user commitments and supported contracts; tests, docs, and earlier agent
  decisions do not make an incidental implementation choice a requirement.
- Trace the complete public workflow through producers, transformations,
  serialization/export, and consumers, including failures and resource lifetime.
  Fix the owning boundary; do not document an accidental gap as a supported
  limit.
- Prefer deletion, existing dependency facilities, and direct data flow over
  custom controls, copied state, and conversions. A shared helper can remove
  complexity across callers. Judge the complete design, not the shortest diff.
  Preserve deliberate support limits and avoid unrelated cleanup or expansion.
- Existing code is evidence, not authority over current policy. Follow
  repository policy, enforcing configuration, and explicitly adopted upstream
  guidance in that order. Retain workarounds only with a technical reason and
  removal condition.
- Passing tests establish neither feature value nor lean design. Resolve design
  questions separately from validation; leaving code unchanged is valid when
  credible alternatives offer no concrete benefit.
- Never commit or print secrets or personal data. Keep machine setup and build
  output local; use documented configuration and supported presets.

## Documentation

Always use the full project name “MQT Core” in prose, including documentation,
comments, commit messages, and pull requests. Never shorten it to “Core”.

Follow the [prose policy](docs/development.md#prose-and-terminology) and
Orwell's rules. Write for readers unfamiliar with the implementation and its
history. Explain tasks and concepts first, integrate additions into existing
explanations, and preserve useful examples, ownership, failures, and deliberate
limits.

Use natural prose in guides and detailed API explanations. Apply STE principles
only to concise comments and reference fields; do not claim formal compliance.
Check unfamiliar terms against primary sources and `docs/glossary.md`; flag
disputed terminology before renaming APIs. Use “program qubits” for unmapped
circuit qubits; distinguish placement, allocation, and fault-tolerant encoding.

Keep conversation and iteration history out of docs, comments, and PR bodies.
Follow the
[release documentation policy](docs/development.md#release-documentation):
describe user-facing changes and released-API migrations in PR descriptions.
Edit `CHANGELOG.md` or `UPGRADING.md` only for release preparation or on
request; changelog entries credit the PR and all contributing authors.

## Testing

Follow the [testing priorities](docs/development.md#testing-priorities). Extend
small end-to-end workflows before adding tests. Assert observable semantics with
independent oracles and required resource bounds, not private structure or a
particular rewrite. A regression does not automatically need a new test.

Cover native semantics in C++ once. Python owns SDK integration, binding
conversions, ownership, and Python failures; avoid repeating native sweeps
across the interpreter/dependency matrix. Keep inputs and parameter sets only as
large as the contract requires.

Before deleting tests, check history and surviving assertions. Preserve distinct
phase, wire-order, numerical, ownership, failure, and complexity contracts;
equal line coverage alone does not prove redundancy. Put tests in the owning
subsystem. Use subprocesses only for irreducible CLI behavior or necessary
process isolation, such as fatal native failures. Do not put MLIR tests under
production tools or enable an optional tool solely for a subprocess test.

## C++ and MLIR

Use C++20 and CMake 3.29 or newer. The development policy owns detailed include,
comment, data-structure, diagnostic, and debugging guidance.

- Use `#pragma once`, direct includes, and standard-library facilities before
  adding abstractions. Use C typedefs such as `size_t` and `uint64_t` without
  `std::`. Do not use C-style casts, including casts to `void`.
- Use LLVM streams and buffers for I/O and `StringRef` for borrowed text. Use
  existing synchronized diagnostics for concurrent logging. Do not include STL
  stream headers, including in tests. Keep owned strings when ownership is
  required.
- Use `//` for implementation comments and `///` for Doxygen. Follow the
  [C++ documentation policy](docs/development.md#c-documentation-comments).
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
- Use GoogleTest/CTest, not `lit` or FileCheck. Exact trees, text, target
  choices, and operation counts need a contract reason.
- Review the MLIR policy and `mlir/.clang-tidy` on major LLVM/MLIR upgrades.

## Python and bindings

- Use Google-style docstrings, including in nanobind binding strings; follow the
  [Python docstring policy](docs/development.md#python-docstrings).
- Fix `ruff` and `ty` diagnostics instead of suppressing them unless a
  documented exception is necessary.
- Preserve released Python APIs and explicit compatibility commitments unless a
  breaking change is authorized. Unreleased PR designs can be simplified within
  scope. Keep optional integration imports lazy where the binding already does
  so.
- Choose finite-shot tolerances with low false-failure probability; keep
  expected values away from tolerance boundaries.
- Regenerate stubs after every binding change with `uvx nox -s stubs`. Never
  edit generated `.pyi` files manually. MLIR handle rules apply in bindings too.

## Build and validation

Run checks relevant to the change and required by repository policy. Diagnose
failures before changing code or build policy; separate defects from stale
output, dependency mismatches, and infrastructure failures. Local machine
overrides may select another supported preset.

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

Inspect the final diff, comments, and status; exclude unrelated files. Verify
checks actually exercised changed files and paths. Rerun affected checks after
edits, and distinguish final-code passes from skipped, blocked, pending, or
older-revision results. Stop repeating validation once required gates pass
unless a concrete risk remains; this does not replace the design review.

## Plans and audits

Keep an [ExecPlan](.agent/PLANS.md) in `.agent/plans/` when design or
coordination needs a durable record, or on request. Small tasks need no plan
file.

Use [audit guidance](.agent/AUDITS.md) for contract reviews. Save findings in
`.agent/audits/` only when a durable record is useful or requested.

## Performance evidence

Measure when requested or needed for a concrete performance question. Trace
costs through representative public workflows before tuning a helper; check
whether work is removed or moved. Prefer existing tools and keep ad hoc
harnesses and data outside the repository unless requested. Report revisions,
workloads, conditions, variance, and limits. Microbenchmarks do not establish
whole-workflow gains; inconclusive measurements justify neither a speedup claim
nor rejection of an independently useful design improvement.

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
