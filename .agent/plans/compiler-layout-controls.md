# Native layouts and SDK layout interchange

Status: complete. Implemented in PR #2553.

## Scope and ownership

Native compilation accepts initial placement and returns a detached
`MappingResult`. Imported provenance uses the frontend-neutral `mqt.layout`
schema in `mlir/include/mqt/Dialect/MQT/IR/MQTDialect.td`; only the versioned
adapter handles SDK objects. Import and export preserve partial assignments,
physical gaps, ancillary inputs, register groups, and output order.

Copies, serialization, and plain QC/QCO conversions retain provenance. The
shared pipeline runner invalidates it before transformations. Export rejects
stale layouts; formats without layout support require explicit discard. Native
results do not compose with imported layouts or survive as serialized metadata.
Explicit placement requires complete assignments to fixed-size local entry-block
allocations; see `docs/mlir/target_compilation.md` for the API contract.

## Tracking decisions

`mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp` records source indices
during target preparation. Tensor shrinking retains surviving entries; removed
slots follow workspace permutations without extra operations. Keeping ordinary
discovery order and the indexed-placement path avoids routing regressions.
Sharing target preparation avoids another pass and its IR verification. Results
are published only after successful compilation.

Layout invalidation belongs to `runWithCompilationOptions`, once per pipeline.
Individual transforms stay independent of imported provenance; raw pass managers
use the shared runner. Lossy format boundaries reject metadata before export.
Discard and output checks visit the full module tree. Invalidation also marks
the pipeline root so cleanup cannot remove the last marker with a private child.

## Validation

Validation passes 642 Python tests, 399 native tests, three CLI CTests, and both
PR examples. Repository lint and full changed-file C++ lint pass. Regressions
cover ordinary/tracked IR equivalence, idle-input unitary semantics, recursive
discard, private nested-module removal, failed pipeline invalidation, and CLI
preservation during plain conversion.

Reproduce with the compiler, mapping, MQT IR, and tensor-transform unit
binaries, the adapter translation tests, and
`pytest test/python/test_mlir.py test/python/qdmi/test_compilation.py`. Follow
`AGENTS.md` for build and lint setup.
