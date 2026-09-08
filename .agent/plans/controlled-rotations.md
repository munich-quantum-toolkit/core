# Multi-controlled Pauli rotations

Status: complete.

## Goal and scope

`decompose-multi-controlled` supports RX, RY, and RZ with numeric and symbolic
angles. It preserves phase, wire order, native target operations, and the
existing `min-qubits` policy. Synthesis uses no additional qubits and scales
linearly in entangling gates before routing.

The owning implementation is
`mlir/lib/Dialect/QCO/Transforms/Decomposition/DecomposeMultiControlled.cpp`.
Pass tests reside in the existing decomposition unit-test file. No new public
API or dependency was added.

## Decisions

- Lower controlled Y as `S†`, MCX, then `S` on the target, reusing the existing
  MCX decomposition and its width and native-target policies.
- For RY and RZ, split controls into two balanced groups and alternate their MCX
  operations with quarter-angle rotations. Borrow controls from the other group
  through Core's exact dirty-helper MCX decomposition. Helpers must be restored
  coherently, including relative phases.
- Obtain RX by Hadamard conjugation of RZ. Controlled rotations through `2*pi`
  retain their conditional phase; the phase-gate normalization rules do not
  apply.
- Reuse the existing fixed-angle MCX plans. Keep symbolic rotation angles as SSA
  values, including computations defined within a control region.

## Validation

The complete `mqt-core-mlir-unittest-decomposition` binary passes 297 tests.
These check phase-exact operators, runtime and region-local angles, native
target and threshold policy, and numeric and symbolic rotation CX budgets
through 64 controls. Shared Pauli tests cover X, Y, and Z with the same CX
counts, including coherent states at synthesis boundaries.

The Python `test_qco_program_decomposes_multi_controlled` API test covers X, Y,
RX, RY, and RZ, including the minimum-width argument and its error handling. One
symbolic RY round trip checks export and binding of generated angle expressions.
These six cases pass; synthesis matrices and resource bounds remain in the
native tests.

The test pass manager verifies each output once; the helper retains separate
input/output linearity checks. The runtime comparison and rejected cache
experiment are recorded in
[`controlled-synthesis-test-runtime.md`](../audits/controlled-synthesis-test-runtime.md).

MCY import, decomposition, and export preserve the exact operator at 2, 3, and 5
controls. General lint, stub generation, and whole-changed-file C++ lint pass
with no remaining findings.

## Performance and limits

Compare Qiskit 2.5 public `mcrx`/`mcry`/`mcrz` synthesis with Core's
decomposition pass, using no extra qubits and the `u,cx` basis. At 2 and 3
controls, RX/RY use 4 and 14 CX gates versus Qiskit's 8 and 20. All other
sampled widths through 64 controls match Qiskit's CX count, for numeric and
symbolic angles.

With identical level-3 post-optimization, larger Core outputs are 15 layers
deeper: RY has depth 186 versus 171 at 8 controls and 1978 versus 1963 at 64.
Local nine-sample median synthesis times were lower for all sampled cases in a
MinSizeRel build. These timings exclude frontend import, basis normalization,
routing, and full target compilation; they are not an end-to-end speed claim.
