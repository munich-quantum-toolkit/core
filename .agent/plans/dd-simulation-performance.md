# DD simulation performance investigations

Status: proposed. No item is measured yet. Each item below needs a baseline
measurement before any code change.

## Goal and scope

Find and remove avoidable end-to-end work in DD-based simulation. The work
covers the DD package (`include/mqt-core/dd/`, `src/dd/`) and its simulation
consumers:

- QCO interpreter: `mlir/lib/Dialect/QCO/Utils/DDFunctionality.cpp`
- QIR JIT execution: `mlir/lib/Dialect/QIR/Execution/JIT/Session.cpp` and
  `mlir/lib/Dialect/QIR/Execution/Runtime/Runtime.cpp`
- QDMI DDSIM worker: `src/qdmi/devices/dd/WorkerMain.cpp`
- Qiskit estimator: `python/mqt/core/plugins/qiskit/estimator.py`

Each item is an investigation with a keep-or-discard criterion. An item that
does not show a material gain on its named workload ends with a short recorded
result and no code change. Do not change public Python or C++ APIs without
maintainer approval; item 2 names the one place where this may come up.

Out of scope: redesigning the unique tables, compute tables, or garbage
collector; gate fusion; new simulation backends; thread-safe packages.

## Current design to keep

These parts already sit in the right layer. Do not move or duplicate them:

- `Package::makeGateDD` is the only gate-embedding path. QCO and QIR reach it
  through `mlir::qco::makeGateDD` (`DDAdapter.cpp`).
- `Package::sampleCounts` owns terminal sampling for both frontends. It splits
  shot groups with binomial draws and samples small groups shot by shot.
- One `Package` per worker persists across shots, and `GateCache` keeps gate DDs
  as roots, so prefix replays hit the matrix-vector compute table.
- The `Package::garbageCollect` cache invalidation is required for correctness.
  Collecting a real number forces node collection, because unrooted nodes could
  otherwise keep dangling weights. Any change must keep this invariant.

## Baseline harness

Build the `release` preset. Keep ad hoc harnesses, data, and plots outside the
repository ([performance evidence](../../AGENTS.md#performance-evidence)). Reuse
existing tools first:

- benchmark program generators in `mlir/bench/programs/` (QFT, QAOA, Grover,
  GHZ, QFT adder, weak-measurement Grover);
- `dd/statistics/PackageStatistics.hpp` (`getStatisticsString`,
  `computePeakMemoryMiB`) for compute-table hits, unique-table sizes, and
  garbage-collection runs;
- a small C++ or Python driver that calls the public entry points named per
  item.

Record per workload: the commit, the machine, the build type, wall time (median
of at least five runs), peak memory, and the relevant table statistics. Report
the variance with the numbers.

## Investigations

### 1. Pauli expectation values without an intermediate state

Owner: `Package::expectationValues` in `src/dd/Package.cpp`.

Current cost: for each distinct Pauli string, the code builds a matrix DD,
computes `multiply(P, state)`, takes `innerProduct(state, P·state)`, and calls
`garbageCollect()`. Each term creates up to one new vector node per state node,
with normalization, real-number lookups, and unique-table inserts. A collection
that frees nodes also clears the inner-product and matrix-vector tables. This
path serves `DDEstimator` and `mqt.core.mlir.expectation_values`.

Proposal: a private recursion that computes ⟨ψ|P|ψ⟩ directly over node pairs
at the same level. X swaps the successors, Z negates the one-successor, Y does
both with a phase, and I pairs equal indices. Below the lowest non-identity
Pauli, reuse `innerProduct` and its compute table. The memo is valid for one
Pauli string only; clear it or key it per term. Keep the public signature,
validation, error behavior, and the borrowed-state contract unchanged.

Expected gain (hypothesis): a constant factor, not an asymptotic one. The
current multiply already skips identity levels above and below the operator.

Workload: a molecular-style Hamiltonian with at least 1,000 terms on a state
with at least 20 qubits, plus a sparse case of fewer than 10 terms to check that
small inputs do not regress.

Keep if the large workload improves by at least 1.5× wall time or reduces peak
nodes materially, with no regression on the small case. Otherwise discard.

Tests: `test/dd/test_pauli_expectations.cpp` already compares all three-qubit
strings against a dense oracle and covers invalid inputs. Add cases for
operators whose lowest non-identity qubit is above zero, and for repeated
strings across observables.

### 2. Estimator overhead per parameter binding

Owner: `DDEstimator._run_pub` in `python/mqt/core/plugins/qiskit/estimator.py`.

Current cost: each binding calls `bind_parameters(..., copy=True)`, which
deep-copies the MLIR module. `mlir::qco::expectationValues` then creates a new
`Package`. The DD layer already accepts runtime `DDArgumentBindings`, but the
Python `expectation_values` binding does not expose them.

Constraint: runtime bindings support fewer programs than IR binding. Composite
modifiers other than a sole standard gate in `qco.ctrl`, and custom matrices,
need compile-time-known matrices (see `buildFunctionality` in
`mlir/include/mqt/Dialect/QCO/Utils/DDFunctionality.h`). Any change must keep
the current path as a fallback and must not narrow supported circuits.

First step: profile a sweep of at least 100 bindings on small (≤10 qubits) and
medium (≈20 qubits) parameterized circuits. Measure the split between module
copy, package setup, and simulation.

Keep only if module copy and setup take at least 30% of the small-circuit sweep.
Exposing bindings in Python is a public API change and needs maintainer approval
first. Otherwise record the profile and close the item.

### 3. One owner for the automatic worker count

Owners: `sampleProgram` in `DDFunctionality.cpp` and `qasmProgram` in
`src/qdmi/devices/dd/WorkerMain.cpp`.

Both compute `clamp(shots / 256, 1, 8)`. The QDMI copy exists to apply
`maxWorkers_`, because `Worker::execute` rejects worker-slot requests above the
job maximum. Retuning one copy without the other changes QDMI behavior silently.

Proposal: when `automaticWorkers_` is set, pass `workers = 0` and cap the
request inside the worker's `requestWorkers` callback with
`std::min(desired, maxWorkers_)`. Explicit worker counts keep their current
path.

This is a maintainability change; no benchmark is needed. Check that QDMI DDSIM
tests in `test/qdmi/devices/dd/` still pass, and that a job with
`maxWorkers = 1` stays serial.

### 4. Cache loss from garbage collection during simulation

Owner: `Package::garbageCollect` and the garbage-collection limits in
`UniqueTable` and `RealNumberUniqueTable`.

Question: how often does a collection clear the matrix-vector and add compute
tables during long simulations, and what does that cost? The real-number table
starts with a lower limit (65,536) than the node tables (131,072) and forces
node collection when it frees anything.

Measure on QFT, QAOA, and Grover at 20–30 qubits: garbage-collection runs,
compute-table hit rates, and wall time. Then rerun with larger initial limits
set through `DDPackageConfig` or the table constructors only.

Keep a default change only if it improves wall time by at least 10% on two or
more workloads without raising peak memory by more than 25%. Do not change the
invalidation rules. Finer-grained invalidation, such as validating entries by
generation, is a redesign and needs a separate plan.

### 5. Prefix replay in sampling with mid-circuit measurements

Owners: `sampleBranches` and `sampleImpl` in `DDFunctionality.cpp`, and
`JitSession::sampleCounts` in `Session.cpp`.

Known limitations:

- Structured branch sampling runs on one thread. A `shortcut:` comment in
  `sampleProgram` records that shared IR evaluation state corrupts concurrent
  shot groups.
- A program that measures inside a called function falls back to one full
  simulation per shot (the `forceDynamic` loop in `sampleImpl`).
- QIR branch sampling replays the native program from the start for each
  distinct branch path. Compiled code cannot fork mid-run, so this owner is
  correct.

All three depend on compute-table hits during replay. Item 4 can defeat those
hits.

Measure with the weak-measurement Grover and magic-state distillation
generators, plus a QFT-based circuit with k mid-circuit measurements for k = 4,
8, 12, at 10⁴ shots. Record the distinct groups or paths, the wall time, and the
matrix-vector hit rate during replay.

Decide from the data:

- Parallel structured branching is worth a separate plan only if one-thread
  branch sampling exceeds the per-shot parallel path on a realistic workload.
- Extending branch sampling to measurements in called functions is worth a
  separate plan only if such programs come from supported frontends.
- For QIR, record the replay hit rate. A low rate links this item to item 4. Do
  not redesign QIR execution.

## Work remaining

- [ ] Baseline numbers for the items 1, 2, 4, and 5 workloads at one recorded
      commit.
- [ ] Item 1 kept or discarded against its criterion; if kept, the tests above
      pass.
- [ ] Item 2 profile recorded; an API change goes ahead only with approval.
- [ ] Item 3 implemented, with QDMI DDSIM tests passing.
- [ ] Item 4 result recorded; any default change backed by its numbers.
- [ ] Item 5 measurements recorded, with a follow-up plan or a closed result.

## Validation

Build and run the native tests with the commands in the
[root guide](../../AGENTS.md#build-and-validation). Focused checks:

- `build/release/test/dd/mqt-core-dd-test`: DD package contracts, including
  Pauli expectations and table behavior.
- `mqt-core-mlir-unittest-qco-utils` with `--gtest_filter='*DD*'`: QCO
  simulation, sampling, and expectation values.
- The JIT and runtime unit tests in `mlir/unittests/Dialect/QIR/Execution/`: QIR
  sampling and branch replay.
- `uv run --no-sync pytest` on `test/python/plugins/qiskit/test_dd_estimator.py`
  and `test/python/test_mlir_expectations.py`: estimator and Python expectation
  values.
- The tests in `test/qdmi/devices/dd/`: worker slot negotiation (item 3).

Run `uvx nox -s cpp-lint` and `uvx nox -s lint` before handoff. Record the
measurements that justify each decision in this plan. Keep the raw data out of
the repository.
