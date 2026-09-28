# W-state preparation

Status: complete; based on upstream main and validated locally.

## Outcome and scope

The `w-state` family provides C++ `WState`, Python `mqt.core.bench.w_state`, and
the existing benchmark CLI interfaces. It prepares the equal, positive-amplitude
superposition of single-excitation states, returning every Z measurement in
`result`. Qubit zero is the least significant displayed bit. The required
positive `qubits` value must fit signed 64-bit circuit indices. The parameter
name matches GHZ, QFT, and Multiplexer.

Generation in `mlir/bench/programs/WState.cpp` initializes qubit zero, then uses
an `scf.for` sweep of controlled RY and reverse CX. Each iteration computes
`2 * acos(1 / sqrt(qubits - index))` with scalar arithmetic and math operations.
This removes the angle table and its storage limit. jeff supports these
operations; the DD interpreter evaluates `math.acos` with MLIR's existing
constant folder.

## Decisions and ownership

`src/bench/WState.cpp` provides the analytic probability: `1/n` for a
single-excitation bitstring, zero otherwise. It reuses the existing counts
metrics, JSON registry, and binding patterns. No state container, sparse input
API, or new simulator entry point is needed for this distribution evaluation.
Existing family interfaces, manifests, and case IDs remain unchanged.

Generation tests use existing QCO DD simulation and `dd::makeWState` to check
amplitudes and coherence. These implementation checks do not expand the public
benchmark API. The 1,024-qubit test checks structured generation, jeff
serialization and reload, and simulation without dense extraction. Reference
size alone does not establish a simulation-time bound for arbitrary circuits.

The benchmark uses main's existing compiler and DD interpreter. Counted loops
have no simulation step limit, so W-state needs no budget changes or dependency
on the RTTI and exception-handling work in PR #2545. The larger conditional-loop
and compilation budgets proposed in PR #2605 are not needed for this circuit.

## Validation

Run the release preset's `mqt-core-bench-test`,
`mqt-core-mlir-unittests-benchmark`, and `mqt-core-mlir-unittest-qco-utils`
binaries, plus the `mqt-core-mlir-benchmark-cli` CTest. Run the Python
benchmark, MLIR, QCO DD, and loop suites.

Regenerate stubs, build the executable documentation, and run full-file C++ lint
and repository lint as required by [AGENTS.md](../../AGENTS.md). Compare C++
lint against `origin/main`, the PR's base branch. The docs include checked 3-
and 256-qubit sampling examples; the native regression retains the 1,024-qubit
check.

Local validation passed: 65 benchmark tests, 33 generation tests, 200 QCO
utility tests, the CLI CTest, and 228 focused Python tests. Stub generation,
full-file C++ lint, repository lint, and executable documentation with local
link checks passed. Hosted checks are reported separately in the PR.
