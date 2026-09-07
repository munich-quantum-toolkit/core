# Centralize typed benchmark family registration

Status: historical implementation record.

## Goal and scope

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

## Constraints

- `docs/benchmarks.md` explicitly rejects a second family catalog. The new
  catalog must replace the synchronized private registry lists instead of
  coexisting with them.

- The repository already uses X-macro catalogs such as
  `mlir/include/mlir/Conversion/GateTable.def` and installs `.inc` catalogs for
  the IR library.

## Decisions

- Store only `(TYPE, STEM, ID, DEFINITION_VERSION)` in the catalog. Rationale:
  these values are repeated mechanical metadata. Schema fields, documentation
  prose, option bindings, and algorithms differ by family and do not belong in a
  macro language.

- Generate implementation glue and the public JSON declarations, but keep the
  typed MLIR and emitter declarations explicit. Rationale: the MLIR declarations
  carry family-specific Doxygen text, while the implementation registries
  contain the costly synchronized duplication.

- Keep the Python family binding bodies explicit. Rationale: QPE, Grover, and
  the option classes differ materially. A large nanobind macro would hide the
  public Python surface and produce poor diagnostics.

## Outcome and validation

`BenchmarkFamilies.inc` now owns the five families' type, function stem, JSON
ID, and definition version. The semantic and MLIR registries and their simple
adapters expand from that catalog. Adding a family still requires explicit
parameter handling, schemas, emitters, bindings, and tests.

The focused build passed with LLVM/MLIR 23. All 37 semantic benchmark tests, 10
MLIR generation tests, the benchmark CLI test, and 20 Python benchmark and CLI
tests passed. The resulting benchmark list, schemas, case IDs, manifests, and
generated programs remain covered by the existing tests.

## Code and ownership

`include/mqt-core/bench/JSON.hpp` declares the named per-family JSON API.
`src/bench/JSON.cpp` owns the semantic registry, family metadata, and the
trivial adapters around family-specific parameter parsing and schemas.
`mlir/bench/Generate.cpp` owns typed program-building adapters and a second
runtime registry that dispatches a JSON instance specification to an emitter.

The new `include/mqt-core/bench/BenchmarkFamilies.inc` file is included several
times. Each inclusion defines `MQT_BENCHMARK_FAMILY` for one narrow expansion,
includes the catalog, and relies on the catalog to undefine the macro. This is
the same X-macro structure used by existing operation and conversion catalogs.

## Acceptance

The benchmark registry must still list the same five IDs in lexical order with
definition version 1. Every instance specification, manifest, case ID, and
evaluation test must pass without expected-output changes.

Every typed MLIR benchmark must still generate valid QC and lower to `jeff`.
Generic JSON generation must dispatch all five catalog rows, and the existing
command-line test must still list and generate registered families.

The installed bench target must include `BenchmarkFamilies.inc`. No binding,
stub, schema, emitter, or test behavior may change. The final diff must contain
no generated build output or unrelated changes.

## Interfaces

The catalog contract is:

    MQT_BENCHMARK_FAMILY(TYPE, STEM, ID, DEFINITION_VERSION)

`TYPE` names a class in `mqt::bench`. `STEM` is a token used to form functions
such as `qftFromManifestJSON`. `ID` is an independent string literal so future
IDs need not be valid C++ identifiers. `DEFINITION_VERSION` is an integer
literal used by semantic manifests and schemas.

The refactor adds no dependency. It uses the C++ preprocessor, C++20 templates,
and the existing CMake header file-set mechanism.
