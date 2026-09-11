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

## Integration and validation

Retain the merged control-flow work reduction, measurement/store analysis,
QTensor canonicalization, builder tracking, and routing improvements. The
[upstream reassessment](compiler-upstream-reassessment.md) records why the
stricter mapper and modifier contracts remain and which QIR analysis was
replaced. The local-buffer alias rule stays in the shared measurement/store
analysis.

After the rebase onto `fce58f02d`, the implementation passes 1,026 selected
native tests and 305 Python compiler and device tests. Full-file C++ lint
against that base reports zero formatting or tidy findings. Generated MLIR
documentation and general lint pass. The stacked README and RtD checks belong to
PR #2509.

The final ponytail review found no further justified C++ cuts. Durable
regressions cover the compiler behavior; temporary benchmark artifacts are not
part of the PR. No dependency or abstraction was added. Hosted CI monitoring is
outside this task.
