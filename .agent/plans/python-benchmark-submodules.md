# Organize Python benchmarks into family submodules

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
stay current as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

The Python benchmark API currently puts every family type into `mqt.core.bench`.
That flat module is already crowded with five families and will become hard to
browse as the catalog grows. After this change, users can import one family
submodule, such as `mqt.core.bench.qft`, while shared result types remain in
`mqt.core.bench`.

The public example is `qft.QFT(qft.Options(...))`. The primary benchmark class
keeps its algorithm name, while module-local helper types use unprefixed names
such as `Options`, `Method`, `Topology`, and `Basis`. The extra hierarchy
`mqt.core.bench.benchmarks.qft` is deliberately not added because `bench`
already identifies the domain.

## Progress

- [x] (2026-08-31 14:45Z) Inspect the current binding, generated stubs, tests,
  documentation, and existing native submodule precedents.
- [x] (2026-08-31 15:08Z) Create direct native submodules and split family
  registrations.
- [x] (2026-08-31 15:08Z) Regenerate package-shaped type stubs and update
  editable installation.
- [x] (2026-08-31 15:08Z) Update Python tests and documentation to use family
  submodules and concise helper names.
- [x] (2026-08-31 15:21Z) Build and validate the result, then inspect the final
      diff. The extension rebuild, official stub session, runtime smoke tests,
      eight focused Python tests, repository lint, and diff checks passed. The
      C++ lint build was stopped on explicit user instruction before analysis
      ran.

## Surprises & Discoveries

- Observation: MQT Core already exposes native submodules from extension
  modules. Evidence: `mqt.core.ir.operations` and `mqt.core.qdmi.driver` use
  nanobind `def_submodule`, and recursive stub generation creates one `.pyi`
  file per submodule.
- Observation: `/opt/llvm` changed while validation was running. The incomplete
  tree first lacked `mlir-tblgen`; its LLVM 23 contents then failed while CMake
  generated the `jeff-mlir` precompiled header. Pinning both package directories
  to `/opt/llvm-22` let the complete official `stubs` Nox session pass.

## Decision Log

- Decision: Use `mqt.core.bench.bv`, `.ghz`, `.grover`, `.qft`, and `.qpe`. Keep
  `Output` and `Evaluation` in the root module and put `Phase` in `qpe`.
  Rationale: direct submodules scale without repeating the word “benchmark.”
  Date/Author: 2026-08-31, Daniel Haag and Codex.
- Decision: Keep primary classes such as `qft.QFT`, but expose helper types as
  `qft.Options`, `qft.Method`, `ghz.Topology`, and similar unprefixed names. Do
  not provide flat or prefixed compatibility aliases. Rationale: the submodule
  supplies the family context, while the primary class still names the algorithm
  represented by an instance. Date/Author: 2026-08-31, Daniel Haag.

## Outcomes & Retrospective

The Python API now groups five benchmark families into direct native submodules.
The primary types retain their algorithm names, helper types use concise local
names, and shared results stay at the root. The generated stubs mirror that
runtime hierarchy. The extension build, official stub generation, focused
behavior tests, repository lint, and diff checks passed. The user stopped the
full C++ lint session during its build, before clang-tidy analyzed the changed
files; no C++ lint result is claimed.

## Context and Orientation

`bindings/bench/register_bench.cpp` defines the native `mqt.core.bench`
extension and currently registers every shared and family-specific Python type
in one module. `python/mqt/core/bench.pyi` is generated from that extension by
the `stubs` Nox session. `bindings/bench/CMakeLists.txt` builds the extension
and installs generated stubs for editable builds. `test/python/test_bench.py`
and `docs/benchmarks.md` exercise and document the public Python API.

A nanobind submodule is a Python module created by the native extension with
`def_submodule`. Recursive stub generation represents this structure as
`python/mqt/core/bench/__init__.pyi` and one sibling `.pyi` file for each
family. Generated stub files must never be edited by hand.

## Plan of Work

Keep the root extension entry point small. Register `Output` and `Evaluation`
there, create the five family submodules, and delegate each family to a separate
C++ registration source. Put each benchmark class, its options, and its enums in
its family submodule. Keep the benchmark class name and give helper types short
names that do not repeat the module name. Put the exact QPE phase type in `qpe`
because no other family uses it. Preserve all behavior, signatures, JSON
formats, and MLIR generation calls.

Update the binding CMake source list and change editable stub installation from
the flat `bench.pyi` file to the generated files under `bench/`. Rebuild the
package, run recursive stub generation, and confirm that the obsolete flat stub
is removed rather than retained as a second API description.

Update Python tests to import family modules and assert that shared output and
evaluation objects still use root-module types. Update the QFT documentation
example and the extension instructions to show the new module structure.

## Concrete Steps

From the repository root, build and install the updated extension:

    uv sync --inexact --no-dev --no-build-isolation-package mqt-core

Regenerate stubs through the repository-owned session with LLVM 22:

    env MLIR_DIR=/opt/llvm-22/lib/cmake/mlir \
      LLVM_DIR=/opt/llvm-22/lib/cmake/llvm uvx nox -s stubs

Run focused behavior and import tests:

    uv run --no-sync pytest test/python/test_bench.py

Run the relevant final checks:

    uvx nox -s lint

Do not run the C++ lint session for this change. The user explicitly stopped
that validation after the build had reached 1,054 of 1,067 targets and before
clang-tidy ran.

Inspect `git diff --check`, the generated stub tree, and `git status --short`
before handoff.

## Validation and Acceptance

Direct imports of all five family modules must succeed. Each family must create
its existing typed benchmark, serialize and parse its existing JSON, evaluate
counts, and generate a QC program. `Output` and `Evaluation` must remain
available only from `mqt.core.bench`; `Phase` must be available from
`mqt.core.bench.qpe`. The old flat family names must be absent. The focused
Python suite and relevant build targets must pass. Generated stubs must describe
the same hierarchy and contain no stale flat `bench.pyi` file.

## Idempotence and Recovery

Build, installation, stub generation, and tests are repeatable. Stub generation
is the only authorized writer for `.pyi` files. If it fails after creating a
partial tree, fix the binding or installation issue and rerun the full session;
do not repair generated files manually. Preserve unrelated worktree changes and
do not push or post GitHub text without separate authorization.
