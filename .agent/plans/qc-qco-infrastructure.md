# QC/QCO infrastructure fixes

Status: complete.

## Outcome and decisions

The [audit findings](../audits/qc-qco-infrastructure.md) are implemented or
covered by upstream fixes. The conversion owns the proof that quantum region
results can become in-place QC references. It rejects unproved permutations
before rewriting; the QCO dialect continues to permit those permutations.
Function-return validation uses the same proven origins rather than assuming
positional loop results. QTensor slot updates remain stores where required.

Allocation mode is an immutable result of module preflight, so factory and
caller order does not affect deallocation. An ordered scalar map makes sink
emission stable. The QTensor cache retains equivalent-index handling and
invalidation, with a direct lookup before scanning.

The shared modifier matrix composer owns width and wire order. Queries include
idle targets, support ordered full-width operations, and return no matrix for
unsupported embeddings or reordered yields. The ten-qubit dense-matrix bound
includes controls. Barrier arity belongs to the operation verifier.

Native MLIR function patterns replace forwarding classes. No new abstraction,
cache, dependency, or register-function ABI is introduced. Register-function
support remains a separate contract under #2428.

## Validation

All 1,978 tests passed in the corresponding release binaries:

| Suite               | Tests |
| ------------------- | ----: |
| QCO-to-QC           |   153 |
| QC-to-QCO           |   175 |
| QC/QCO round trip   |     6 |
| QC IR               |   348 |
| QCO IR and matrices |   506 |
| QCO utilities       |   185 |
| Decomposition       |   238 |
| Optimizations       |   196 |
| Compiler            |   171 |

The release build also includes `mqt-cc`. `uvx nox -s lint`,
`cmake --build --preset release --target mlir-doc`, and `git diff --check`
passed. The final full-file C++ lint comparison is pinned to the PR merge base,
`3be5ee96f3907659bd99fdd6d54cd74c4eea7da9`, to exclude unrelated changes to
main.
