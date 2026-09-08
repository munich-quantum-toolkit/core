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
global phase, including under controls.

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

The release OpenQASM target suite passes 190 tests; QC translation passes 209,
including strict-policy helper matrices with controlled global phase. Repository
lint passes. The release build and all 3,215 MLIR CTests pass. Complete C++ lint
on the PR diff passes with zero findings. Hosted CI is separate from these local
results.

The audit records baseline/revised parser memory and timing measurements and
scaling checks for shared affine expressions and large controlled gates.
