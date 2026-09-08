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

- Derive the implementation from Pauli rotation identities and Core's existing
  decomposition helpers. Do not consult or adapt Qiskit source. Use Qiskit's
  public APIs only as an external performance comparator.
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

The complete `mqt-core-mlir-unittest-decomposition` binary passes 263 tests,
including 26 rotation tests. These check phase-exact full operators for 2–8
controls, runtime and region-local angles, native target and threshold policy,
and linear resources through 64 controls.

`pytest test/python/test_mlir_qiskit_translation.py -k multi_controlled_rotations`
passes all 24 cases. Numeric target compilation requests native `gphase` to
retain overall phase, as required by the existing target contract. Symbolic
synthesis is exported and bound before exact matrix comparison.

All 384 Python translation and typed-program checks pass, including the two QDMI
device cases with the built native device configured. General lint, stub
generation, and whole-changed-file C++ lint pass with no remaining findings.

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

Symbolic decomposition and export work. Full symbolic target compilation can
still produce `math.atan2`, which the existing Qiskit exporter does not support.
Bind parameters before target compilation when using that export path. A general
symbolic exporter change is outside this synthesis implementation.
