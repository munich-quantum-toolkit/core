# QIR conversion contracts and resource lifetime

Status: complete; local validation passes, hosted CI is reported separately.

## Outcome and scope

QC-to-QIR conversion preserves return order and repeated scalar/register output
without duplicating ownership. Base conversion rejects later use of measured
qubits, including packed operands and retained helper calls. Unsupported helper
storage and entry results fail with diagnostics. The QIR builder owns unfinished
IR, transfers ownership on move/finalization, and returns an empty module when
final verification fails.

Resource metadata follows static IDs through helper calls and returns, selects,
and CFG arguments. Loop capabilities cover every emitted function, computed
measurement feedback, early continues, nested loops, and irreducible cycles.
Native pointer buffers conservatively require array support. Output labels are
unique and only reuse matching immutable globals in the default address space.

The owning code is shared QC-to-QIR preparation, profile-specific measurement
lowering, and QIR builder, utilities, and attribute passes. Their descriptions
and `docs/qir/index.md` state the supported boundaries.

## Decisions and limits

- Keep output occurrences separate from owned allocations. Ordered output and
  release lists preserve repetition and deterministic emission. Register
  descriptors are borrowed only while recording output.
- Check Base qubit reuse after reconciling conversion casts. Existing marked
  pointer stores retain packed qubit identities; passing measured qubits to a
  retained helper is rejected conservatively.
- Require measurements and CBit allocations in the entry function. Accept direct
  measurements, direct CBit allocations, and at most one `i64` exit code as
  returns. Other expressions must be lowered first. A scalar sharing fused
  register storage requires distinct constant store indices; summarize only
  registers that need this check.
- Reuse one Adaptive scratch Result for measurements not retained as outputs.
  Each live Boolean is copied into SSA before the next measurement overwrites
  that Result. Returned scalar handles and fused register slots remain distinct.
- Own the builder module with `OwningOpRef` instead of a raw pointer and
  separate finalized flag. Require dynamic qubit allocation before control flow;
  hoisting allocations would change fresh-qubit semantics. Repeated register
  loads are borrowed references, not ownership transfers.
- Visit CFG edges directly, index defined-helper callers once, cache symbol
  lookup, and deduplicate resource values when queued. Visit helper returns once
  per resource role. Do not index external QIS callsites.
- Collect Adaptive capabilities in one function-body traversal and both static
  resource roles in one module traversal. Retain helper/CFG tracing and loop
  dependence analysis because supported inputs require their precision.
- Reuse MLIR `CFGLoopInfo` for natural loops and LLVM irreducibility detection.
  Unknown dependencies and irreducible cycles require both backward-branch
  capabilities. This is not a finite-trip-count proof or external-pointer alias
  analysis.
- Hoist release declarations, reuse array-size constants, cache Boolean output
  declarations, and remove stale allocator/string storage, duplicate
  loaded-qubit tracking, unused Base ownership entries, and the builder's MemRef
  dependency.
- Create output declarations lazily through one emitter. Preserve output-label
  uniqueness across batches and leave Adaptive cleanup ordering unchanged.
- Consolidate test setup and diagnostic capture; use existing tables for new
  forwarding and loop cases. Preserve parameterized gate cases and direct
  assertions about output identity, ordering, ownership, and rejected inputs.

## Validation

Validated with GCC 13.3, LLVM/MLIR 23.1.0, and the release preset on upstream
`3bf8849e5d5b671fe03b168aa38e3357a4bded0c`. Build `mqt-cc`, the Base/Adaptive
conversion, QIR IR, and compiler test targets, then run their executables. All
728 tests pass (138 Base, 164 Adaptive, 135 QIR, 291 compiler). Consolidation
removes two executable tests relative to the original baseline while preserving
semantic cases and adding regressions. New cases offset the deleted boilerplate;
this is not a net reduction in test-source lines or a coverage-percentage claim.

All 61 semantic export probes retain their expected outcomes. Forty native JIT
shots verify ordered/repeated/mixed outputs and helper resource capacity. Four
more retain eight alternating measurement snapshots; a wrong expected value
fails. Both 1,024-measurement probes now emit one result allocation/release pair
while retaining all measurements and reads. LeakSanitizer reports no leaks for
1,000 abandoned, finalized, moved, or moved-and-finalized builders per mode.
Generated MLIR references, repository lint, and whole-file C++ lint pass. C++
lint checks all nine changed implementation/test files.

After simplifying metadata collection and output emission, all 728 native, three
CLI, and 56 Python QDMI compilation tests pass again. All 61 export probes
retain their outcomes; the 18 successful LLVM exports verify and have identical
metadata and no LLVM function differences from the preceding commit. Repository
formatting runs before whole-file C++ lint, which reports zero findings.
Existing helper, CFG, and 100,000-gate scaling probes show no material timing
regression; these simplifications do not establish a general speedup or memory
improvement.

Targeted three-run median wall times compare the contract-fix baseline with the
earlier performance-fix executable, using CPU affinity on a shared DGX Spark.
Parsing and verification are included; these are diagnostic scaling checks, not
a general compiler speedup claim. Ad hoc harnesses and data stay outside the
repository.

| Workload                             | Metadata/conversion passes |  Before |   After |
| ------------------------------------ | -------------------------: | ------: | ------: |
| 4,096 helper arguments               |                          3 | 1.530 s | 0.077 s |
| 2,048 incoming LLVM switch edges     |                          1 | 1.455 s | 0.007 s |
| 2,048 incoming Adaptive switch edges |                          1 | 1.527 s | 0.033 s |
| 2,048 backedges                      |                          3 | 0.187 s | 0.022 s |

At 2,048 switch edges, peak RSS falls from 54,436 to 22,136 KiB for metadata and
73,000 to 30,428 KiB for standalone Adaptive conversion. A 100,000-gate control
remains near 0.40 s before and after the changes. The full compiler does not
support the unstructured Adaptive switch fixture before QC-to-QIR lowering.

Not covered: the full repository/Python test suites, rendered documentation,
release wheels, or hardware execution. Hosted CI is reported separately.
