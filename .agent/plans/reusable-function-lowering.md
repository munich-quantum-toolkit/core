# Lower reusable functions with standard MLIR passes

Status: complete.

## Goal

Keep reusable QC and QCO functions visible for structured format export, but
inline calls before target-specific decomposition, mapping, synthesis, and
conformance checks. Cleanup and export should remove unreachable private
functions, and cleanup may shrink live unitary signatures.

## Design

- Require `mqt.entry_point` on one public, defined, module-level `func.func` so
  MLIR symbol DCE has a durable program root.
- Use MLIR's standard inliner at the target-compilation boundary. The typed API
  and `mqt-cc` skip their QIR-preparation inliner when a target pipeline owns
  inlining.
- Run symbol DCE after local canonicalization in QC export and QCO cleanup so
  every export can omit unused gate declarations.
- Use `RemoveDeadValues` in QC and QCO cleanup. Unused unitary parameters and
  qubits need not remain in live function and call signatures.
- Keep `WireIterator` unchanged. `qco.call` already implements
  `UnitaryOpInterface` with positional input/output correspondence, while
  generic `func.call` remains an intentional wire boundary.
- Do not add custom specialization, tensor promotion, auxiliary-qubit hoisting,
  call-graph infrastructure, or IPO passes without a measured workload.

## Implementation

`populateTargetCompilationPipeline` starts with the standard MLIR inliner. QC
export and both cleanup pipelines remove unreachable symbols; explicit QC and
QCO cleanup additionally remove dead values. The typed compiler pipeline and CLI
avoid duplicate inlining for targeted QIR output.

Tests cover the entry-point visibility contract, unreachable gate removal,
unitary-signature shrinking, native jeff calls through target compilation, and
target compilation from a caller-owned context.

## Validation

Run from the repository root:

    cmake --preset release
    cmake --build --preset release
    ctest --preset release
    uvx nox -s cpp-lint -- 6328d48c77370cc99e089ce38e57bcd9053e48c6
    uvx nox -s lint

The release build and all 3,918 registered tests pass, with one expected skip.
All 294 Qiskit translation tests, repository lint, and all-file C++ lint pass.
Hosted CI is separate evidence and must run on the published commit.

## Outcome

The implementation uses standard MLIR infrastructure and adds no custom pass or
framework. Exporters omit unreachable functions, and standard dead-value removal
owns cleanup-time unitary-signature shrinking.
