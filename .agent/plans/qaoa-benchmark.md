# QAOA benchmark

Status: complete. The benchmark generates reusable symbolic circuits for MaxCut
on an unweighted cycle, with native and Python scoring and JSON manifests.

## Scope and definition

Configure `qubits >= 3` and `depth >= 1`, both within signed 64-bit dimensions.
The generated function takes `gammas` and `betas`, each a one-dimensional `f64`
tensor of length `depth`, and measures all qubits. Angle optimization and
arbitrary graphs are outside this benchmark.

[Farhi, Goldstone, and Gutmann](https://arxiv.org/html/1411.4028), equations
(2)–(6), define a uniform initial state followed by cost and mixer layers:

```text
|ψ⟩ = U_B(β_d) U_C(γ_d) … U_B(β_1) U_C(γ_1) |+⟩^n
U_C(γ) = exp(-i γ C), U_B(β) = exp(-i β Σ_j X_j)
```

Equation (12) defines each MaxCut term as `(1 - Z_j Z_k)/2`. Section IV uses a
cycle, whose maximum cut is `n` for even `n` and `n - 1` for odd `n`.

## Decisions and ownership

- `mlir/bench/programs/QAOA.cpp` uses `RZZ(-gamma)` for each edge, which agrees
  with the cost unitary up to global phase, and `Rx(2 beta)` for each mixer.
  Apply even-start edges, odd-start edges, then the closing edge once. This
  bounds cost-gate dependency depth at two for even cycles and three for odd
  cycles. Register and layer loops keep generation compact in both dimensions.
- Reuse shared array binding, including length and finite-value validation and
  classical folding. Bind a copy to reuse the template. Base QIR compilation
  owns loop legalization; QAOA needs no export-specific lowering.
- `src/bench/QAOA.cpp` scores cuts using shared count and outcome validation.
  Compute weighted means in extended precision without integer products, then
  bound the mean by the exact optimum before calculating the approximation
  ratio. Scores measure solution quality, not agreement with an ideal quantum
  distribution.
- The manifest and case ID identify the template. Record bound angles
  separately. Binary jeff preserves argument positions and types but not input
  names, so bind angles before binary serialization.
- Qiskit export requires explicit `QCOProgram.unrollQuantumLoops()` to resolve
  loop-indexed qubits. Compact generation does not imply compact execution:
  angle storage scales with depth, and compiler and executor resources still
  limit concrete runs.

## Validation

- Native and MLIR benchmark suites pass: 160 scoring/JSON tests and 52
  generation tests, plus the benchmark CLI test. The QAOA state oracle applies
  cost phases directly to amplitudes, then the mixer matrix, independently of
  the emitted gates. It covers three, four, and five qubits, unequal layers, and
  negative angles. One-layer cases also check cost-gate dependency depth.
- Python benchmark, parameter-binding, and compiler tests pass: 428 tests.
  Sampling covers zero-angle layers and copies of a reusable template without
  explicit cleanup after binding. Cleanup after unrolling remains necessary for
  the native dependency-depth assertion.
- A 1,024-qubit template with two or one billion layers passes jeff round-trip
  and operation-count bounds without allocating a statevector.
- A bound two-layer, four-qubit program passes Qiskit statevector comparison
  after explicit unrolling. Direct Base QIR compilation and DDSIM execution
  match the expected cut score over 8,192 shots. The documentation example runs
  without an explicit cleanup call.
- Stub generation, whole-file C++ lint, repository lint, executable
  documentation, and local documentation links pass. Rendered appearance, hosted
  CI, and large-instance execution remain unverified.
