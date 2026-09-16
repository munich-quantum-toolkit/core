# Native layouts and SDK layout interchange

Status: complete. Draft PR #2553 resolves #2070.

## Scope and ownership

Native compilation accepts initial placement and returns a detached
`MappingResult`. Imported provenance uses the frontend-neutral `mqt.layout`
schema in `mlir/include/mqt/Dialect/MQT/IR/MQTDialect.td`; only the versioned
adapter handles SDK objects. Import and export preserve partial assignments,
physical gaps, ancillary inputs, register groups, and output order.

Copies, serialization, and plain QC/QCO conversions retain provenance.
Resource-changing and custom pipelines invalidate it. Export rejects stale
layouts; formats without layout support require explicit discard. Native results
do not compose with imported layouts or survive as serialized metadata. Explicit
placement requires complete assignments to fixed-size local entry-block
allocations; see `docs/mlir/target_compilation.md` for the API contract.

## Tracking decisions

`mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp` records source indices
during target preparation. Tensor shrinking retains surviving entries; removed
slots follow workspace permutations without extra operations. Keeping ordinary
discovery order and the indexed-placement path avoids routing regressions.
Sharing target preparation avoids another pass and its IR verification. Results
are published only after successful compilation.

## Validation

At `a4077e2ed`, 642 Python tests, 397 native tests, three CLI CTests, both PR
examples, repository lint, and full changed-file C++ lint passed. Compiler tests
compare ordinary and tracked IR and check unitary semantics with an idle input
used as routing workspace. Metadata tests cover schema and lifetime rules.

Reproduce with the compiler, mapping, MQT IR, and tensor-transform unit
binaries, the adapter translation tests, and
`pytest test/python/test_mlir.py test/python/qdmi/test_compilation.py`. Follow
`AGENTS.md` for build and lint setup.
