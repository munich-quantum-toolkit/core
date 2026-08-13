# Phase-estimation examples

MQT Core includes runnable MLIR examples of standard quantum phase estimation
(QPE) and iterative quantum phase estimation (IPE):

- [`qc_quantum_phase_estimation.mlir`](https://github.com/munich-quantum-toolkit/core/blob/main/mlir/examples/qc_quantum_phase_estimation.mlir)
- [`qc_iterative_phase_estimation.mlir`](https://github.com/munich-quantum-toolkit/core/blob/main/mlir/examples/qc_iterative_phase_estimation.mlir)

Equivalent OpenQASM 3 programs exercise the native mqt-cc frontend:

- [`qasm3_quantum_phase_estimation.qasm`](https://github.com/munich-quantum-toolkit/core/blob/main/mlir/examples/qasm3_quantum_phase_estimation.qasm)
- [`qasm3_iterative_phase_estimation.qasm`](https://github.com/munich-quantum-toolkit/core/blob/main/mlir/examples/qasm3_iterative_phase_estimation.qasm)

Both estimate the exactly representable phase $3/8=0.011_2$ of the phase gate
$P(3\pi/4)$ acting on its eigenstate $|1\rangle$. Choosing an exact three-bit
phase makes the expected measurement result deterministic and keeps the examples
suitable for regression tests.

The circuit structure follows MQT Bench's `qpeexact` and `iqpe` benchmarks,
specialized to three-bit precision and the fixed phase $3/8$. The fixtures are
checked in rather than importing MQT Bench at test time, so MQT Core retains no
additional runtime or test dependency.

MQT Bench constructs IQPE as a straight-line Qiskit circuit whose classically
conditioned rotations consume earlier measurements. The OpenQASM fixture
preserves that structure. The hand-written QC counterpart uses an equivalent
`scf.for` formulation to demonstrate how the same algorithm can retain
structured classical control in MLIR.

## Why the examples use QC and QCO

The source programs use the {doc}`QC dialect <QC>`. Its reference semantics
match source languages such as OpenQASM: a qubit can be used repeatedly inside
structured classical control without explicitly threading a new SSA value
through every operation.

The tests convert each program to the {doc}`QCO dialect <QCO>`. This conversion
introduces value semantics and threads the evolving qubit states through loops
and modifier regions. The converted programs then pass through the QCO cleanup
pipeline and the decision-diagram simulator verifies the expected classical
result. Consequently, the examples exercise both user-facing representation and
optimization-oriented representation rather than duplicating each circuit in two
dialects.

## Standard QPE

Standard QPE uses three counting qubits. Controlled applications of $U$, $U^2$,
and $U^4$ encode the eigenphase, and an inverse quantum Fourier transform
produces the bits `011`. This representation emphasizes coherent quantum control
and modifier regions.

## Iterative QPE

IPE reuses one counting qubit for all three bits. Its `scf.for` loop combines:

- `arith` operations that compute the controlled-unitary power and feedback;
- a `memref<3xi1>` classical result register;
- mid-circuit `qc.measure` and `qc.reset` operations; and
- a loop-carried feedback angle for the semiclassical inverse QFT.

For a current feedback angle $f$ and newly measured bit $m$, the next round uses

$$
f_{\mathrm{next}} = \frac{f}{2} - m\frac{\pi}{2}.
$$

This compact recurrence produces the same conditional phase corrections as an
explicit branch for every previously measured bit. The measurements occur from
least to most significant (`110` in encounter order), while the memref stores
the final estimate in the conventional most-significant-bit-first order (`011`).
