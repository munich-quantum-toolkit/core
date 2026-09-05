# Lower reusable functions with standard MLIR passes

Status: complete.

## Goal

Keep reusable QC and QCO functions visible for structured format export, but
inline calls before target-specific decomposition, mapping, synthesis, and
conformance checks. Explicit QC and QCO cleanup should remove unreachable
private functions without changing live unitary signatures.

## Design

- Require `mqt.entry_point` on one public, defined, module-level `func.func` so
  MLIR symbol DCE has a durable program root.
- Use MLIR's standard inliner at the target-compilation boundary. The typed API
  and `mqt-cc` skip their QIR-preparation inliner when a target pipeline owns
  inlining.
- Run symbol DCE after local canonicalization in explicit QC and QCO cleanup.
  Keep it out of the QC export pipeline because OpenQASM can preserve unused
  gate declarations.
- Omit `RemoveDeadValues` from function-preserving QC and QCO cleanup. The pass
  can rewrite live unitary signatures; QIR and jeff cleanup retain it after
  calls have been lowered.
- Keep `WireIterator` unchanged. `qco.call` already implements
  `UnitaryOpInterface` with positional input/output correspondence, while
  generic `func.call` remains an intentional wire boundary.
- Do not add custom specialization, tensor promotion, auxiliary-qubit hoisting,
  call-graph infrastructure, or IPO passes without a measured workload.

## Implementation

`populateTargetCompilationPipeline` starts with the standard MLIR inliner.
`populateQCCleanupPipeline` and `populateQCOCleanupPipeline` finish with symbol
DCE; `populateQCExportPipeline` preserves all representable gate definitions.
The typed compiler pipeline and CLI avoid duplicate inlining for targeted QIR
output.

Tests cover the entry-point visibility contract, pruning before and after
canonicalization, unused-gate OpenQASM export, native jeff calls through target
compilation, and target compilation from a caller-owned context.

## Validation

Run from the repository root:

    cmake --preset release
    cmake --build --preset release
    ctest --preset release
    uvx nox -s cpp-lint -- 6328d48c77370cc99e089ce38e57bcd9053e48c6
    uvx nox -s lint

The release build and all 3,916 registered tests pass, with one expected skip.
Repository lint and whole-file C++ lint pass with zero findings. Hosted CI is
separate evidence and must run on the published commit.

## Outcome

The implementation uses standard MLIR infrastructure and adds no custom pass or
framework. The post-rebase specialist review found the inliner ownership and
WireIterator behavior idiomatic, identified the export/cleanup distinction
above, and found no further production-code simplification.
