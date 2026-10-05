# Litinski 15-to-1 distillation

Status: complete; implementation, generated stubs, and focused execution checks
are validated.

## Scope and decisions

Replace the decoder-based benchmark with the logical five-qubit circuit in
[Litinski, Figure 3](https://arxiv.org/html/1905.06903v3#S0.F3).
The fifteen ordered Pauli rotations prepare T†|+⟩ on the first qubit; X
measurements of the other four qubits detect faults. The benchmark still reports
rejection and a root-state check, with ideal outcome `00`.

Preserve levels 1–4 with real quantum-state consumption: higher levels inject
each rotation from a preceding-level output. Reuse five workspace qubits per
level. A private function per level keeps generation compact. Each block runs
once, even if a preceding block rejects; there is no retry or noise model and no
surface-code layout. Keep definition version 1 because the benchmark has not
been released.

## Validation

With LLVM/MLIR 23.1.0, the Release build and these checks passed:

- `build/release/mlir/unittests/bench/mqt-core-mlir-unittests-benchmark --gtest_filter='*Distillation*'`:
  four tests cover ideal levels 1–2, all 15 single and 105 double faults, the 35
  undetected logical-error triples, and lower-level rejection and quantum-state
  propagation. Levels 1–4 retain the documented workspace bound and compact QCO,
  and round-trip through `jeff`.
- `build/release/test/bench/mqt-core-bench-test` covers the benchmark
  references, JSON contracts, and registry.
- The focused Python distillation suite passes all 11 tests against the rebuilt
  bindings, including direct sampling and both DDSIM payload paths.
- `uvx nox -s stubs` regenerates the binding documentation.
- `uvx nox -s lint` passes all repository hooks.

Full documentation builds, C++ lint, and simulation of levels 3–4 were not run.
