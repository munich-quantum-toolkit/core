# Multi-controlled Pauli rotations

Status: implemented. Rebased onto main
`4c5e45855e5bb50c42b68ddbf4f9a4dababb8737`.

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

The complete `mqt-core-mlir-unittest-decomposition` binary passes 303 tests.
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
with no remaining findings. Stub generation leaves no diff. The complete
`uvx nox --non-interactive -s docs` HTML build passes, as does the native
`mlir-doc` target. The compiler help lists the supported gate families.

## Performance and limits

The reproducible benchmark and applied audit findings are recorded in
[`pr2467-controlled-synthesis-review.md`](../audits/pr2467-controlled-synthesis-review.md).
Core uses fewer CX gates than Qiskit 2.5.2 for two- and three-control RX/RY and
matches other sampled counts through 64 controls. Larger outputs are up to 15
layers deeper. Fair synthesis-only timings are slower for Core in this run;
previous timing claims are superseded.

A reversed-helper prototype beats Qiskit's depth at sampled widths from nine
through 64 controls, with unchanged CX counts. An unbalanced eight-control split
saves eight CX gates but increases depth. These are measured follow-up
candidates, not production changes or an optimality claim.
