# Contract audit: benchmark generation tests

Status: Applied. Baseline: `11729cba241d942c0bd74a935d5caa57381572cf` with no
in-scope edits. Date: 2026-09-05. Source and tests: `mlir/bench/programs/`,
`mlir/unittests/bench/`.

## Result

The generation tests can be split by benchmark family without removing or
weakening a contract. A shared `jeff` round-trip check also strengthens the
cross-family smoke test without fixing any serialized bytes.

## Findings

### Split the generation tests by family

The single source file mixed cross-family checks with family-specific IR
contracts. The split keeps the cross-family generation and reset checks in
`test_benchmark_generate.cpp`. Each other source now owns the tests for the
matching emitter in `mlir/bench/programs/`. All sources remain in one GoogleTest
executable, and all existing test names and assertions remain unchanged.

This change reduces the scope of later benchmark additions and reviews. It does
not enable a production-code change. The main cost is more translation units,
which can increase a clean build's total compiler work.

### Check every generated `jeff` program after serialization

The former cross-family smoke test stopped after lowering produced a
`JeffProgram`. The largest multiplexer test also serialized, parsed, validated,
and reserialized its program. The shared helper applies that round trip to every
benchmark method and retains the maximum-size multiplexer check.

The check protects successful binary serialization, parsing, validation, and
stable reserialization. It does not constrain exact bytes across revisions.

## Unresolved questions

None in this scope. Cross-family execution against analytic references remains a
separate test concern and needs its own runtime and support assessment.

## Validation

The focused build passed:

```console
CCACHE_DISABLE=1 cmake --build --preset release --target mqt-core-mlir-unittests-benchmark -j4
```

The resulting test binary ran all 12 tests successfully.
