# Tensor placement and ownership

Status: all findings implemented and validated locally.

The audit baseline is PR #2506 at `7fca20764e0f4b4fbb995fd6e2f759c74275c6c1`.
The integration base is `a66a9b206` on 2026-09-10, using LLVM/MLIR 23.1.0.

## Resolved findings

1. **Preserve indexed loops where target and payload permit them.**
   `TargetEnvironment::supportsIndexedQubits` selects Adaptive QIR with
   all-to-all connectivity and no operation-specific site tuples.
   `Mapping.cpp::placeIndexedAllocations` assigns each allocation slot once,
   using `qco.static` and `qtensor.from_elements`. Synthesis tracks placed
   origins and still checks native operations. Sparse physical IDs and capacity
   remain checked. QIR metadata uses compiler-marked qubit stores so the
   declared capacity includes every possible static qubit operand.

   The previous mapper required initial extraction and final reinsertion and
   forced tensor loops to expand. The 100,001-iteration regression now retains
   one loop and one gate body. Other payloads, explicit topology, and site-bound
   operations keep bounded specialization and exact-site conformance. Storage
   grows with register width, not the number of loop iterations.

2. **Share constant-slot scalarization across structured control flow.**
   `QTensorCanonicalization.cpp` keeps one lifetime/access proof and two local
   extraction/reinsertion helpers. Its typed if, while, and for adapters
   preserve classical state, induction arguments, region order, attributes, and
   wire identity. Cleanup runs before loop specialization. Runtime indices and
   incomplete or nested tensor updates do not match this scalarization; the
   indexed placement path handles supported dynamic accesses instead.

   The previous test that required tensor-loop expansion now asserts the payload
   boundary. Semantic and structural tests cover successful scalarization,
   runtime-index non-matches, and retained large loop bodies. The requested
   ponytail review found no need for a generic control-flow framework or a
   second ownership of the slot-access proof.

3. **Release only tensor slots that still own qubits.** `QCOToQC.cpp` previously
   kept extracted references in a buffer and later released every slot. Sinking
   an extracted qubit or transferring it to another tensor could therefore
   release it twice. A complete-lifetime proof avoids extra state in the common
   case. Other dynamic lifetimes use a boolean mask; extraction clears ownership
   and insertion restores it.

   `qc.dealloc_register` expresses the conditional quantum release that generic
   `memref.dealloc` cannot express. Adaptive QIR lowers it to standard qubit
   release calls. Local reference buffers and masks need no runtime API. Base
   Profile removes dynamic ownership bookkeeping. Direct DDSIM regressions cover
   fixed and dynamic shapes, a runtime index through a loop, both release
   orders, transferred ownership, and a returned classical register.

   Tensor ownership is local to a function. QCO-to-QC diagnoses residual tensor
   signatures and requires inlining, which the target pipeline already performs.
   The complete-lifetime proof falls back to runtime state beyond 64 nested
   regions. This bounds analysis recursion without dropping supported behavior.

## Measurements and limits

Historical measurements below predate the current upstream integration. The
[benchmark record](../benchmarks/compiler-placement/README.md) also contains the
fresh current-main comparison, including unsupported inputs.

[Matched measurements](../benchmarks/compiler-placement/README.md) retain exact
inputs, native module hashes, raw samples, reproduction commands, and a plot.
Both arms run the complete device pipeline. Three alternating batches provide
nine samples per workload. QPE and RUS execution checks the analytic reference
at 1,024 shots; widths 32 and 64 are compile-only.

Standard QPE at width 64 changes from 386.36 ms to 51.09 ms median compilation
and from 22,876 to 4,480 bytes. Small timing differences are inconclusive on
this shared host. Retained loops can cost execution time: RUS at width 16
changes from 42.53 to 72.84 ms median, including job/JIT setup. RUS at width 4
grows from 3,640 to 3,748 bytes. No universal runtime speedup is claimed.

The earlier device-versus-direct-QIR probe remains diagnostic evidence only; one
arm skipped target passes. Its reduced partial-release input exposed the
independent ownership defect. Those findings are now protected by native tests.

## Integration and validation

Retain the merged control-flow work reduction, measurement/store analysis,
QTensor canonicalization, builder tracking, and routing improvements. The
[upstream reassessment](compiler-upstream-reassessment.md) records why the
stricter mapper and modifier contracts remain and which QIR analysis was
replaced. The local-buffer alias rule stays in the shared measurement/store
analysis.

The updated implementation passes 992 focused native tests, all 27 Python
device-compilation tests, and all executed benchmark reference checks. Full-file
C++ lint against `a66a9b206` reports zero formatting or tidy findings. Generated
MLIR documentation and general lint pass. The stacked README and RtD checks
belong to PR #2509.

The requested final ponytail review found no further justified C++ cuts. It
removed 16 superseded diagnostic scripts, raw runs, and generated LLVM snapshots
(1,478 lines), retaining native regressions, minimized inputs, and reproducible
matched benchmarks. No dependency or abstraction was added. Hosted CI monitoring
is outside this task.
