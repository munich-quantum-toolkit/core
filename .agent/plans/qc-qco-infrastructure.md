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

Before rebasing, all 1,978 tests passed in the corresponding release binaries:

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
`ec799daa09f855bd0edcbc5592a5fedd90836516`, to exclude unrelated changes to
main.

After rebasing on that main commit, all 505 conversion, round-trip, and compiler
tests passed again, as did lint and full-file C++ lint. The complete strict
`uvx nox --non-interactive -s docs` build, including notebook execution, passed
after qualifying the inherited PennyLane capability reference with a local
attribute docstring. This fixes the Read the Docs failure in PR #2464.

## Linearity and modifier follow-up

The development policy and MLIR agent guide state the exactly-one-use contract.
Five redundant rewrite guards are removed; the boundary verifier, debug
assertions, and classical/reference use checks remain. Native MLIR use-list
iterators provide constant-time access without a new type interface or wrapper.
Extract/insert cancellation uses a rewrite that deletes both operations,
avoiding the temporary duplicate tensor use left by a root-only fold.

DCX cancellation requires reversed target order. A lone controlled SWAP remains
a conditional unitary; it cannot become an unconditional output permutation.
Exact-matrix tests cover both DCX orientations and controlled SWAP behavior. The
rebase on `11d983fa59831007a8ce7e8bf17b2f06f85b3a20` retains the upstream
PennyLane documentation fix. The follow-up passed 880 tests: QCO IR (509),
QTensor IR (37), QC-to-QCO (175), QCO-to-QC (153), and round trip (6). Strict
Sphinx documentation, lint, and full-file C++ lint passed. Local release
validation uses `ENABLE_IPO=OFF` to avoid a GCC/LTO duplicate symbol in the
installed MLIR library; no source workaround is included.
