# W-state preparation

Status: complete; local validation passed. Hosted CI is reported in the PR.

## Outcome and scope

The `w-state` family provides C++ `WState`, Python `mqt.core.bench.w_state`, and
the existing benchmark CLI interfaces. It prepares the equal, positive-amplitude
superposition of single-excitation states, returning every Z measurement in
`result`. Qubit zero is the least significant displayed bit. The required
positive `qubits` value must fit circuit indices and angle storage.

Generation in `mlir/bench/programs/WState.cpp` initializes qubit zero, then uses
an `scf.for` sweep of controlled RY and reverse CX. A rank-one tensor holds
precomputed angles, so jeff needs no runtime transcendental operations.

## Decisions and ownership

`src/bench/WState.cpp` provides the analytic probability: `1/n` for a
single-excitation bitstring, zero otherwise. It reuses the existing counts
metrics, JSON registry, and binding patterns. No state container, sparse input
API, or new simulator entry point is needed for this distribution evaluation.
Existing family interfaces, manifests, and case IDs remain unchanged.

Generation tests use existing QCO DD simulation and `dd::makeWState` to check
amplitudes and coherence. These implementation checks do not expand the public
benchmark API. The 4,096-qubit test checks structured generation, jeff
serialization and reload, and simulation without dense extraction. Reference
size alone does not establish a simulation-time bound for arbitrary circuits.

The QCO DD interpreter shares a 100-million-step budget across loops, branches,
and calls. It rejects resolved `scf.for` trip counts above the remaining budget
before execution; per-step accounting remains for nested flow and while loops.
Budget tests use oversized inner loops to fail promptly and retain a successful
case above the previous 10,000-step boundary. No public configuration was added.

The benchmark CLI returns `llvm::Error` / `llvm::Expected` for command and file
failures, preserving temporary-file cleanup and refusing to overwrite outputs.
Its single exception handler translates the existing CoreBench validation API at
the executable boundary. The JSON generation adapter enables exception support
so validation failures reach that handler; program emitters retain LLVM's
default policy. A regression check rejects the previous abort on invalid JSON.
JSON integer extraction no longer catches an exception after type and range
validation have already made conversion safe.

The algorithm has its own Unreleased changelog entry. The published v4 entry
remains unchanged.

## Validation

Run the release preset's `mqt-core-bench-test`,
`mqt-core-mlir-unittests-benchmark`, and `mqt-core-mlir-unittest-qco-utils`
binaries, plus the `mqt-core-mlir-benchmark-cli` CTest. Run the Python
benchmark, MLIR, QCO DD, and loop suites. CLI failure cases must return a
handled error, including missing files, invalid destinations, and existing
outputs.

Regenerate stubs, build the complete executable documentation, and run full-file
C++ lint and repository lint as required by [AGENTS.md](../../AGENTS.md).
`docs/benchmarks.md` includes checked small and 4,096-qubit sampling examples.
Local validation passed: 65 benchmark tests, 33 generation tests, 196 QCO
utility tests, the CLI CTest, and 199 focused Python tests. The Python JSON
adapter also reported invalid input without aborting. Generated stubs, full-file
C++ lint, repository lint, and complete executable documentation passed. Hosted
CI is tracked separately in the PR.
