# OpenQASM import and export corrections

Status: complete.

## Outcome and scope

The frontend builds persistent syntax IDs directly, bounds affine analysis per
proof, and uses set-based static distinctness. Affine quantum indexing remains
supported. Typed register comparisons use the existing bit-vector form. QC
emission counts inserted operations instead of predicting lowering costs.

The exporter preserves result order and zero-valued data, rejects repeated
register results, and keeps absent outputs void. It materializes snapshots at
their definition, shares frontend name validation, avoids unused measurement
storage, and initializes registers with exact-width bit strings. Float casts and
unordered floating inequality round-trip. A shared native-U helper preserves
global phase, including under controls. Loop-local bit storage does not carry
global register-name metadata.

The public importer accepts frontend policy and emission limits through
`QASM3ImportOptions`; tests use this API without including private emitter
headers. Direct Qiskit export assigns names to unnamed private gate parameters
and folds constant integer-to-float casts. OpenQASM callers need no QCO
conversion or manual pass pipeline for these cases.

The implementation lives in `mlir/lib/Target/OpenQASM` and
`mlir/lib/Dialect/QC/Translation`, with headers under `mlir/include` and
regression tests in the corresponding `mlir/unittests` directories. The source
contract is documented in `docs/mlir/OpenQASM.md`. The audit evidence and
accepted dispositions are in `../audits/openqasm-import-export.md`.

## Decisions and limits

- Keep affine quantum-index proofs, intermediate-overflow checks, and scope
  invalidation. Cache only within one proof context.
- Require full-register initialization before dynamic reads. Static bit facts
  and scalar generations remain; dynamic partial-initialization facts do not.
- Store entry-function snapshots explicitly. Gate bodies retain bounded inline
  expressions because OpenQASM gates cannot declare local classical state.
- Preserve ordered results; diagnose repeated register aliases rather than
  changing their meaning. A returned integer zero is data.
- Keep the existing include search order and document it. General arrays,
  runtime angles, and static-only quantum indexing are outside this change.

## Validation

The release build passes with `ENABLE_IPO=OFF`; GCC LTO encountered duplicate
MLIR symbols in the local linker. The configured CTest suite has 3,201 passes
and one optional QDMI job-ID test skipped, including 191 passing OpenQASM target
tests. All 1,152 Python tests pass. Strict-policy helper matrices check
controlled global phase; Python round trips use the native OpenQASM frontend and
direct Qiskit export. Snapshot tests distinguish one loop iteration from two. An
instrumented run covers 18 previously missed changed lines through the existing
emission-budget boundary test. Repository lint and complete C++ lint pass. The
documentation build passes with the PennyLane capability-link fix from upstream
main. Generated Python stubs are unchanged. Hosted CI is separate from these
local results.

The audit records baseline/revised parser memory and timing measurements and
scaling checks for shared affine expressions and large controlled gates.
