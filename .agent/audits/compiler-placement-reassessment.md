# Tensor placement and ownership

Status: implementation, validation, and review complete.

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

3. **Keep placed tensor references separate from dynamic ownership.** Placement
   assigns static qubits. QTensor buffers then need only local `memref.alloca`,
   loads, and stores; their release does not release physical qubits. No QC
   dialect addition or runtime occupancy mask is needed.

   Dynamic tensor conversion requires every extracted slot to be restored before
   region exit or deallocation. The converter rejects unsupported lifetimes and
   tensors assembled from dynamic qubits before rewriting. This is a conversion
   limit: QTensor still permits extracted qubits to outlive their original
   tensor. Tensor function arguments and results require inlining. The bounded
   lifetime proof rejects nesting beyond 64 regions.

4. **Normalize QIR results at the DDSIM QDMI boundary.** QIR records retain
   their program order in the runtime. The device reverses each flattened
   recorded shot before constructing the histogram, so shots and counts use the
   same most-significant-bit first order as OpenQASM. Runtime record IDs and
   physical qubit IDs do not determine output-bit positions.

   Regressions cover both QIR profiles and encodings, multiple classical
   registers, swapped measurement destinations, repeated output records,
   variable-length Adaptive outcomes, and direct QPE evaluation without a
   caller-side reversal. The generic QIR runtime is unchanged.

## Integration and validation

Retain the merged control-flow work reduction, measurement/store analysis,
QTensor canonicalization, builder tracking, and routing improvements. The
[upstream reassessment](compiler-upstream-reassessment.md) records why the
stricter mapper and modifier contracts remain and which QIR analysis was
replaced. The measurement/store analysis keeps its existing effect checks.

Validation against `fce58f02d` passes 922 native tests and 315 Python tests,
including direct QPE evaluation and matching shots/histograms across QIR
profiles and encodings. The final test simplification also passes its 22
conversion regressions and 37 device-compilation tests. Generated MLIR
documentation builds. Full-file C++ lint against the fixed integration base and
general lint pass. The stacked README and RtD execution checks belong to PR

## 2509

The final ponytail review removed unused state in tensor-allocation patterns,
mask-only memref legality and measurement/store exceptions, and unnecessary
filtering in the register-store regression. The remaining diff needs no new QC
operation, dependency, or generic control-flow abstraction. Durable regressions
cover the compiler behavior; temporary benchmark artifacts are not part of the
PR. Hosted CI monitoring is outside this task.
