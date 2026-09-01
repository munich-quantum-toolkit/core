# Centralize typed benchmark family registration

This ExecPlan is a living document. Keep `Progress`, `Surprises & Discoveries`,
`Decision Log`, and `Outcomes & Retrospective` current while the work proceeds.
Maintain this file according to `.agent/PLANS.md` from the repository root.

## Purpose / Big Picture

Adding a typed benchmark currently repeats its C++ type, function-name stem,
stable JSON ID, and definition version across the semantic JSON registry and the
MLIR generation registry. A missed MLIR entry can let semantic validation accept
a family before generation silently returns no program. After this refactor, one
X-macro catalog owns those four values and generates the repeated registry and
adapter code. Family-specific validation, schemas, reference models, emitters,
bindings, and tests remain ordinary C++.

Users observe no API or serialization change. The existing benchmark list,
schemas, case IDs, manifests, generated programs, Python interface, and output
ordering must remain identical.

## Progress

- [x] (2026-09-01) Reviewed the semantic, MLIR, and binding extension points and
      selected the repeated family metadata that belongs in the catalog.
- [x] (2026-09-01) Added and installed the catalog, then generated semantic JSON
      metadata, registry rows, and trivial public adapters from it.
- [x] (2026-09-01) Generated the repeated MLIR implementation wrappers and
      runtime dispatch rows from the same catalog.
- [x] (2026-09-01) Updated the extension documentation and completed focused
      validation and final diff review.

## Surprises & Discoveries

- Observation: `docs/benchmarks.md` explicitly rejects a second family catalog.
  The new catalog must replace the synchronized private registry lists instead
  of coexisting with them.
- Observation: The repository already uses X-macro catalogs such as
  `mlir/include/mlir/Conversion/GateTable.def` and installs `.inc` catalogs for
  the IR library.

## Decision Log

- Decision: Store only `(TYPE, STEM, ID, DEFINITION_VERSION)` in the catalog.
  Rationale: these values are repeated mechanical metadata. Schema fields,
  documentation prose, option bindings, and algorithms differ by family and do
  not belong in a macro language. Date/Author: 2026-09-01 / Daniel Haag and
  Codex.
- Decision: Generate implementation glue and the public JSON declarations, but
  keep the typed MLIR and emitter declarations explicit. Rationale: the MLIR
  declarations carry family-specific Doxygen text, while the implementation
  registries contain the costly synchronized duplication. Date/Author:
  2026-09-01 / Codex.
- Decision: Keep the Python family binding bodies explicit. Rationale: QPE,
  Grover, and the option classes differ materially. A large nanobind macro would
  hide the public Python surface and produce poor diagnostics. Date/Author:
  2026-09-01 / Codex.

## Outcomes & Retrospective

`BenchmarkFamilies.inc` now owns the five families' type, function stem, JSON
ID, and definition version. The semantic and MLIR registries and their simple
adapters expand from that catalog. Adding a family still requires explicit
parameter handling, schemas, emitters, bindings, and tests.

The focused build passed with LLVM/MLIR 23. All 37 semantic benchmark tests, 10
MLIR generation tests, the benchmark CLI test, and 20 Python benchmark and CLI
tests passed. The resulting benchmark list, schemas, case IDs, manifests, and
generated programs remain covered by the existing tests.

## Context and Orientation

`include/mqt-core/bench/JSON.hpp` declares the named per-family JSON API.
`src/bench/JSON.cpp` owns the semantic registry, family metadata, and the
trivial adapters around family-specific parameter parsing and schemas.
`mlir/bench/Generate.cpp` owns typed program-building adapters and a second
runtime registry that dispatches a JSON instance specification to an emitter.

The new `include/mqt-core/bench/BenchmarkFamilies.inc` file is included several
times. Each inclusion defines `MQT_BENCHMARK_FAMILY` for one narrow expansion,
includes the catalog, and relies on the catalog to undefine the macro. This is
the same X-macro structure used by existing operation and conversion catalogs.

## Plan of Work

Add `BenchmarkFamilies.inc` with one row per family. Update the bench library
header file set so installed users of `JSON.hpp` receive the included catalog.

In `JSON.hpp`, expand the catalog into the five named per-family APIs. In
`JSON.cpp`, expand it into metadata traits, private schema and evaluation
declarations, the semantic registry, evaluation adapters, and the trivial public
wrapper definitions. Use normal templates for shared logic. Keep each parameter
parser, parameter JSON function, reference JSON function, and schema body
explicit.

In `mlir/bench/Generate.cpp`, expand the same catalog into the typed generation
definitions and runtime registry rows. Infer registry sizes rather than storing
the family count. Keep each structured emitter body and its explicit declaration
unchanged.

Update `docs/benchmarks.md` so the catalog is the sole family metadata list and
the documented extension steps match the resulting implementation.

## Concrete Steps

Run commands from the dedicated worktree. Build and test the semantic layer:

    cmake --build build/release --target mqt-core-bench-test
    ./build/release/test/bench/mqt-core-bench-test

Build and test generation:

    cmake --build build/release --target \
        mqt-core-mlir-unittests-benchmark mqt-core-bench
    ./build/release/mlir/unittests/bench/mqt-core-mlir-unittests-benchmark
    ctest --test-dir build/release \
        -R '^mqt-core-mlir-benchmark-cli$' --output-on-failure

Run the focused Python tests against the editable build:

    uv run --no-sync pytest test/python/test_bench.py test/python/test_cli.py

Finish with formatting checks, `git diff --check`, and a final source review. Do
not run a separate C++ lint session; the user has asked to avoid those sessions.

## Validation and Acceptance

The benchmark registry must still list the same five IDs in lexical order with
definition version 1. Every instance specification, manifest, case ID, and
evaluation test must pass without expected-output changes.

Every typed MLIR benchmark must still generate valid QC and lower to `jeff`.
Generic JSON generation must dispatch all five catalog rows, and the existing
command-line test must still list and generate registered families.

The installed bench target must include `BenchmarkFamilies.inc`. No binding,
stub, schema, emitter, or test behavior may change. The final diff must contain
no generated build output or unrelated changes.

## Idempotence and Recovery

All builds and tests are repeatable and write only below `build/`. If a catalog
expansion fails, inspect the preprocessed declaration or registry row. Do not
duplicate a second family list to bypass the failure.

## Interfaces and Dependencies

The catalog contract is:

    MQT_BENCHMARK_FAMILY(TYPE, STEM, ID, DEFINITION_VERSION)

`TYPE` names a class in `mqt::bench`. `STEM` is a token used to form functions
such as `qftFromManifestJSON`. `ID` is an independent string literal so future
IDs need not be valid C++ identifiers. `DEFINITION_VERSION` is an integer
literal used by semantic manifests and schemas.

The refactor adds no dependency. It uses the C++ preprocessor, C++20 templates,
and the existing CMake header file-set mechanism.
