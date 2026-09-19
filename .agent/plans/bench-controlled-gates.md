# Controlled composite gates

Status: complete.

## Scope and ownership

The versioned circuit adapter must retain the complete definition of controlled
gates whose targets span more wires than their base operation. Generic control
unwrapping applies only when operand widths agree and all controls are closed.
Open-controlled gates retain Qiskit's X-conjugated definition; annotated open
controls remain unsupported.

Multi-control decomposition expands composite controls only when they meet its
width and target-support conditions. Reuse the existing control-unrolling
implementation and control canonicalization to preserve nested controls, operand
order, and controlled global phases. Add no pipeline passes or early
single-qubit merging: symbolic merging expands classical IR and slows later
passes. Two-qubit controls retain their existing handling. Nested inverse and
power bodies remain a limitation tracked in
[#2588](https://github.com/munich-quantum-toolkit/core/issues/2588).

## Validation

Compiler (230), decomposition (313), MQT transforms (32), QC IR (368), QCO IR
(571), and Qiskit translation (458) tests pass. Eleven added open-control cases
fail on e3295d31 and pass with the fix. Stub generation, repository lint, and
full changed-file C++ lint pass. Native validation uses Clang 23 with ThinLTO.

Earlier measurements at e3295d31 against b897f04 used 21 warmed samples per
case: medians differ by −2.5% to +2.8% for all-to-all compilation/synthesis and
−3.6% to +3.8% for linear routing. Workloads cover GHZ, QFT, numeric and
symbolic rotations, and native gates on 12–32 qubits; quantum operation counts
match the baseline. Environment: Python 3.13, MLIR 23.1.1, MinSizeRel on macOS
ARM64, seed 10 and four mapping trials. These measurements were not repeated for
the open-control follow-up and do not establish zero cost for every circuit.
