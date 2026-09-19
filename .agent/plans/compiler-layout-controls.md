# One entry-point layout and successful compilation results

Status: complete. Implements the revised plan for PR #2553 at
`0ba1b2c71acd987a197b82ed158d890c271dbd6b`. The
[preceding review](../audits/pr2553-layout-review.md) records the original
evidence.

## Outcome and decisions

- `mqt.layout` and its mutually exclusive invalidation marker belong to the
  `mqt.entry_point` function. Copies, serialization, and plain QC/QCO conversion
  preserve them; transformations invalidate them. Discard and export checks use
  this single owner. Recursive invalidation and root-marker propagation are
  gone.
- Layout-bearing programs and tracked compilation require one direct entry point
  and reject additional nested entry points. Attribute verification checks the
  owning function; constructors and the CLI check the supplied program module,
  independently of MLIR's temporary parsing container. Unrelated library-only IR
  retains its existing contract.
- `QCOProgram::compileForTargetWithLayout` owns stack-local tracking state and
  returns its detached result after the full pipeline, target conformance,
  pass-manager verification, and final linearity verification succeed. The
  public result-output pipeline API, publication pass, shared ownership, and
  destination pointer are removed. Ordinary target pipeline population remains
  available.
- Qiskit import reads public source-register membership, with the existing bit
  register fallback for manually constructed layouts. Export restores membership
  while removing assignments filled by `add_register`, preserving partial maps
  and absent register slots. It creates loose qubits only for unregistered
  inputs.
- Imported provenance, dense routing indices, and target-site snapshots remain
  separate value types. Fixed local allocations, complete initial placement, and
  idle-input tracking remain supported. Imported layouts are not composed with
  native results; program contents after failure remain unspecified.

## Validation

- Release build with Clang 23, ThinLTO, and mold passed.
- `ctest --preset release-clang-ipo -j 8`: 3,574 passed, one existing skip
  (`ScQDMIJobSpecificationTest.QueryJobId`). This includes compiler/CLI, MQT IR,
  mapping, tensor transformation, and conversion tests.
- `pytest test/python/test_mlir.py test/python/test_mlir_qiskit_translation.py test/python/qdmi/test_compilation.py`:
  644 passed with Qiskit 2.5.2 and the built local DDSIM/SC devices.
- Repository lint, stub generation, and MLIR reference generation passed. Stub
  generation produced no signature changes.
- `uvx nox -s cpp-lint` passed after the final review fixes, checking every line
  of each changed C++ file. The final diff was also checked against the PR merge
  base.

## Final ponytail review

One review of the implementation found the following cuts. Locations refer to
the reviewed working tree; all three findings were applied.

- `Mapping.cpp:L522`: delete: reset of newly constructed tracking state. Nothing
  replaces it.
- `Mapping.cpp:L610`: shrink: assignment-only `finishLayout` wrapper. Assign the
  result of `sourceLayout` at its two callers.
- `TargetCompilation.cpp:L148`: shrink: conditional mapping-pass construction.
  Use `createMappingPass(mappingOptions, tracking)` for both pointer values.

net: -7 lines possible.

## Size checks

DGX Spark ARM64, LLVM/MLIR 23.1.0, Clang 23.1.2 Release, CPython 3.14.7.
Measurements pin each process to one 3.9 GHz-class CPU. All `n` qubits are
active: apply H and RX(0.17) to each; for `i = 0 .. n-6`, apply CX(i, i+5),
RZ(0.31) to qubit i+5, and CX(i, i+1). This gives `5n-15` input gates on a line
target with unrestricted native operations and a base-QIR payload. Controls are
seed 7, one trial, one refinement iteration, lookahead 20, and the default 64
MiB per-search storage estimate.

Each width uses three alternating ordinary/tracked pairs, except the exploratory
1,000-qubit case, which uses one pair with a 60-second limit per process. Times
cover compilation only; peak RSS includes the Python process and input setup. No
statevectors or unitary matrices are allocated. Every pair produced identical
verified output IR and complete initial/final input mappings.

| Qubits | Input gates | Ordinary / tracked time | Ordinary / tracked peak RSS |
| -----: | ----------: | ----------------------: | --------------------------: |
|     50 |         235 |          21.1 / 21.6 ms |             46.5 / 46.5 MiB |
|    100 |         485 |          68.0 / 79.0 ms |             50.9 / 51.0 MiB |
|    150 |         735 |        157.0 / 154.5 ms |             55.4 / 55.6 MiB |
|    300 |       1,485 |        702.9 / 706.6 ms |             72.0 / 72.0 MiB |
|  1,000 |       4,985 |    21804.6 / 23236.3 ms |           184.7 / 184.8 MiB |

These are measurements of one routed workload, not general speedup claims.
100–150 qubits remains the primary range, 300 secondary, and 1,000 exploratory;
these priorities impose no input limit. The separate ancilla-scan optimization
remains deferred because the earlier import/export profile did not justify it.
Ad hoc measurement scripts and raw samples remain outside the repository.
