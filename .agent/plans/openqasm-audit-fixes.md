# OpenQASM import and export corrections

Status: complete.

## Outcome and scope

The frontend builds syntax IDs directly, shares leaf types, bounds affine
analysis per proof, and deduplicates static operands with sets. Typed register
comparisons use bit vectors. QC emission counts actual inserted operations.
Analysis accepts `GatePolicy` directly; `OpenQASMImportOptions` owns the policy
and emission limit.

OpenQASM export preserves ordered results and zero-valued data, rejects repeated
register outputs, and keeps absent outputs void. It materializes scalar
snapshots, shares source-name validation, avoids unused measurement storage, and
emits exact-width bit strings. Float casts and unordered floating inequality
round-trip. A shared native-U helper preserves phase, including under controls.

Private Qiskit gate formals are positional and receive generated local names.
Export enables native folding on a clone before reconstructing Qiskit
expressions. Public parameter names, vector metadata, and shared identities
remain validated.

## Decisions and limits

Keep affine quantum indexing, intermediate-overflow checks, scope invalidation,
and actual emission bounds. Dynamic reads require full-register initialization.
Gate bodies retain bounded inline expressions; entry functions use snapshots.
Include lookup retains its documented policy. General arrays, runtime angles,
and register subroutines remain separate features.

The audit at `../audits/openqasm-import-export.md` retains reproducers, measured
scaling improvements, and the reasons for keeping semantic tests. Source
contracts live in `docs/mlir/OpenQASM.md` and
`docs/mlir/python_compiler_collection.md`.

## Validation

The release build and CTest suite pass: 3,201 passed and one optional QDMI
job-ID test skipped. All 1,156 Python tests pass, including 328 Qiskit
translation tests. Repository lint passes; regenerated stubs are unchanged.
Full-file C++ lint of the exact PR diff passes with zero findings. Hosted CI is
separate from these local results.
