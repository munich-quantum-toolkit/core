# VQE ansatz sources

Status: complete. The native and Python APIs, structured emitter, JSON registry,
and executable documentation implement the circuit below.

## Definition

[Peruzzo et al.](https://arxiv.org/html/1304.3061) introduce VQE and allow a
device-specific ansatz with a fixed sequence of parameterized operations. Their
photonic experiment prepares arbitrary two-qubit states; it does not define a
unique scalable circuit. Cite this paper for VQE, not as the source of the exact
gate sequence below. The benchmark covers state preparation and computational-
basis sampling, not Hamiltonian estimation or classical optimization.

Use the established real-amplitudes ansatz with linear entanglement. The precise
reference is
[`real_amplitudes(n, entanglement="linear", reps=depth)`](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.real_amplitudes),
with the final rotation layer included and unentangled qubits retained. These
are the documented defaults. Explicitly select `linear`: Qiskit's default is
`reverse_linear`, which has a different gate order.

## Circuit contract

Start in the all-zero state. For each repetition `layer` from zero through
`depth - 1`:

1. Apply `RY(thetas[layer * n + q])` to every qubit `q`.
2. Apply `CX(q, q + 1)` in ascending order, from `q = 0` through `n - 2`.

Then apply `RY(thetas[depth * n + q])` to every qubit and measure all qubits.
[`n_local`](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.n_local)
defines the ascending linear chain and the final rotation layer. Thus `depth`
counts entangling repetitions, not total circuit depth. The flat angle array has
`n * (depth + 1)` entries in layer-major, then qubit-major order. For one qubit,
the entangling layers are empty; the state is `RY(sum(thetas)) |0>`.

These are implementation choices for this benchmark, not claims about the
Peruzzo experiment. Keep only the fixed topology and one angle array; no ansatz
selector, connectivity option, optimizer, or Hamiltonian is needed.

## Independent checks

- Compare bound small circuits with Qiskit's reference for unequal positive and
  negative angles, multiple repetitions, and at least three qubits. Three qubits
  distinguish forward from reverse chain order.
- For one qubit, check `P(1) = sin(sum(angles) / 2)^2`.
- For three qubits and one repetition, setting only the first angle to `pi/2`
  yields equal probabilities for `000` and `111`. This checks the direction and
  completeness of the CNOT chain.
- With all preceding angles zero, setting one final-layer angle to `pi` yields
  the corresponding single-bit outcome. This checks the final layer and
  indexing.
- Check compact generation at large width and repetition count without binding
  an angle array or allocating a statevector. Check the product of width and
  rotation-layer count before constructing the tensor dimension.

No claim is made that this ansatz can represent the ground state of every
Hamiltonian.

## Export boundary

Keep reusable named templates as QC. Binary `jeff` does not preserve input
names, so bind before serialization when the result will be executed. The
current QCO-to-jeff conversion maps indices to signed 32-bit integers; large
flattened angle indices can exceed that range even though QC generation remains
compact. This is an exporter limit, not an ansatz width limit.

## Validation

Native and MLIR benchmark suites, Python benchmark tests, CLI integration, stub
generation, repository and C++ lint, and executable documentation with local
link checks pass. The tests compare small states against an independent matrix
calculation and Qiskit, check compact QC generation at 1,024 qubits and one
billion repetitions, and round-trip a one-million-repetition template through
`jeff` within its index range.
