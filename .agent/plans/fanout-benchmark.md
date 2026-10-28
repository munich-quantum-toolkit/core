# Quantum fan-out benchmark

Status: complete.

## Goal and scope

The `fan-out` family is available through the typed benchmark, JSON, Python, and
QC generation interfaces. It implements the Hamming-weight construction in
[Høyer and Špalek, Figure 5](https://arxiv.org/html/quant-ph/0208043)
at the fixed angle π/2. This is a concrete controlled-S instance, not a general
commuting-unitary parallelizer or the paper's rotation-by-value variant.

## Decisions

- Prepare `n` controls uniformly and a target in the plus state. Fan out the
  target to `n - 1` clean ancillas, apply one controlled phase per workspace
  qubit, then undo fan-out. Measure the target in the X basis and the controls
  in the Z basis. The target is the least-significant logical output bit.
- The paper's `Rz` is `diag(1, exp(i angle))`; use controlled phase, not Core's
  controlled Rz. The coherent-state test must distinguish these conventions.
- Use register storage and structured loops; do not expand a gate list in the
  generator. CX implements fan-out, so no constant-depth hardware claim applies.
- `controls` ranges from 1 to 1073, for up to 2146 physical qubits. At this
  boundary the smallest nonzero joint probability, `2^-(controls+1)`, is the
  smallest positive double. Compute probabilities exactly from Hamming weight
  modulo four, without machine-word bitstring conversion or trigonometry.
- Reuse registry macros, strict JSON parsing, evaluation helpers, builder loops,
  and test utilities. Keep family JSON checks in `test_fan_out.cpp`.

## Validation

Local checks passed:

- Release native benchmark suite: 146 tests.
- Release MLIR benchmark suite: 49 tests, including coherent equivalence to a
  sequential circuit on 514 physical qubits, sampled output mapping, and compact
  generation and jeff round trip at 2146 physical qubits. The three fan-out
  tests also passed after the final test-only lint correction.
- Benchmark CLI CTest and all 86 Python benchmark tests.
- `uvx nox -s stubs`, `uvx nox -s lint`, and the executable documentation
  session `uvx nox --non-interactive -s docs`; generated API math is parsed as
  math.
- `uvx nox -s cpp-lint` and explicit whole-file cpp-linter checks for modified
  existing C++ files omitted by uncommitted branch-diff discovery; no findings.

Hosted CI has not run. The construction supports only the fixed π/2 rotation and
uniform controls described above; it does not claim constant hardware depth.
