# Litinski 15-to-1 distillation

Status: complete; local tests, documentation, and repository lint pass.

## Scope and decisions

Replace the decoder-based benchmark with the logical five-qubit circuit in
[Litinski, Figure 3](https://arxiv.org/html/1905.06903v3#S0.F3).
The fifteen ordered Pauli rotations prepare T†|+⟩ on the first qubit; X
measurements of the other four qubits detect faults. The benchmark still reports
rejection and a root-state check, with ideal outcome `00`.

Higher levels inject each rotation from a preceding-level output. Reuse five
workspace qubits per level and retain a private function per level. Accept
positive level counts whose workspace dimensions fit signed 64-bit indices; the
previous four-level cap is not a circuit requirement. Each shot executes
`15**levels` leaf rotations, so the linear workspace bound does not imply linear
runtime. Backend and resource limits still apply.

Each block runs once, even if a preceding block rejects; there is no retry or
noise model and no surface-code layout. Keep definition version 1 because the
benchmark has not been released.

## Validation

Validation uses LLVM/MLIR 23.1.0 and the local Clang 23 ThinLTO preset.

- `mqt-core-mlir-unittests-benchmark`: all 43 tests pass. Four distillation
  tests cover ideal levels 1–2, all 15 single and 105 double faults, the 35
  undetected logical-error triples, and lower-level rejection and quantum-state
  propagation. The rejection probe forces only the first child to reject;
  replacing accumulated rejection with the latest child's result fails it.
  Levels 1–4 and 8 retain the workspace bound and compact QCO, and round-trip
  through `jeff`.
- `mqt-core-bench-test`: all 137 tests pass, covering benchmark references, JSON
  contracts, and the registry.
- All 12 focused Python distillation tests pass against the rebuilt bindings,
  including direct sampling and both DDSIM payload paths.
- `uvx nox -s stubs` regenerates the binding documentation.
- `uvx nox -s lint` passes all repository hooks.
- `uvx nox --non-interactive -s docs` passes, including executed examples and
  local link checks.

Simulation above level 2 is outside the validation scope.
