# Consolidate DD circuit execution

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core and MQT DDSIM currently contain overlapping decision-diagram (DD)
circuit-execution loops. A DD is a graph representation of a quantum state or
operation. The overlap has already produced different handling of repeated
measurements, result widths, output permutations, random-number generators, and
final-state ownership.

After this change, Core owns one low-level exact execution kernel because its
QDMI reference device must execute circuits without depending on DDSIM. DDSIM
continues to own the public simulator class and its approximation and noise
policies. Its ordinary exact `CircuitSimulator` delegates to Core and receives
counts, the retained final state, and the number of circuit executions.
High-level Python users import `sample` and `simulate_statevector` from DDSIM;
Core retains the package-aware primitives needed by its QDMI device, QCEC, and
SyReC.

The result can be observed in tests: the same seeded circuits produce stable
counts for terminal and dynamic measurements, repeated calls continue the same
random stream, final DD roots remain balanced, and the moved Python helpers
return the former result types.

## Progress

- [x] (2026-08-27 21:40 UTC) Read issue #2103, repository policy, the ExecPlan
  requirements, and the relevant Core, DDSIM, QCEC, SyReC, and ProblemSolver
  sources.
- [x] (2026-08-27 21:40 UTC) Inventory execution behavior, public Python
  helpers, downstream consumers, tests, and overlapping Core-v4 work.
- [x] (2026-08-28 00:10 CEST) Implement Core's exact result-bearing kernel,
  explicit root ownership, ordered measurement formatting, finalization, and
  private virtual-operation helpers.
- [x] (2026-08-28 00:15 CEST) Add Core regressions for dynamic execution,
  repeated assignments, result widths, permutations, garbage, global phase,
  random-number continuation, zero shots, failures, and root counts.
- [x] (2026-08-28 00:25 CEST) Delegate DDSIM's ordinary exact simulator to Core
      while retaining DDSIM's hook loop for approximation, derived simulators,
      and density-matrix noise.
- [x] (2026-08-28 00:35 CEST) Move the high-level Python helpers to DDSIM,
      migrate ProblemSolver with a Python 3.10 fallback, regenerate Core stubs,
      and update migration documentation.
- [x] (2026-08-28 00:42 CEST) Build the DDSIM Python binding against this Core
  worktree and pass all ten focused Python helper tests.
- [x] (2026-08-28 00:55 CEST) Finish validation and final inspection. Core DD,
      QDMI, stubs, focused Python, and full lint pass. DDSIM focused C++, built
      Python, and full lint pass. ProblemSolver's source checks pass; its lock
      and type session are blocked only by the unreleased DDSIM 2.6 dependency.
- [ ] Add Core and DDSIM changelog entries after pull request numbers exist,
      update DDSIM's Core-v4 pins, and regenerate ProblemSolver's lock file
      after the coordinated releases are available.

## Surprises & Discoveries

- Observation: `DeterministicNoiseSimulator` inherits the old
  `CircuitSimulator::simulate` loop but substitutes a density-matrix DD through
  virtual hooks. Sending every subclass through the exact vector-state kernel
  would silently remove noise. Evidence: its repeated simulation test only
  passes when it explicitly retains the hook path.
- Observation: DDSIM approximation is scheduled at established operation
  boundaries and counts approximation runs. A generic post-operation callback in
  Core changed this schedule. The final Core API is therefore exact-only; DDSIM
  keeps its specialized hook loop for approximation and subclasses.
- Observation: both old samplers stored terminal measurement assignments in a
  map keyed by qubit. Measuring one qubit into two bits dropped an assignment.
  The new ordered assignment vector returns `11` for a prepared `|1>` measured
  into both bits.
- Observation: DDSIM's retained hook path initially kept the same lossy map, so
  exact base and derived simulators disagreed for repeated terminal
  measurements. It now uses the same ordered assignment semantics, and its
  derived-hook regression also returns `11`.
- Observation: implicit sampling used the classical-bit count when an unused
  classical register existed. It must use the qubit count, while explicit
  measurements use the classical-bit count.
- Observation: `dd::applyGlobalPhase` changed an edge weight without
  transferring the package's registered root. The new implementation registers
  the output before releasing the input; the global-phase root test passes.
- Observation: dynamic measurement can collapse a by-value edge before a later
  bounds failure, which invalidates an exception guard. The kernel now validates
  measurement sizes, mapped qubits, and classical bits before executing a shot.
- Observation: DDSIM's deterministic density initialization leaked the prior
  matrix root across repeated jobs. Aligning the density edge before `decRef`
  and tracking initialization leaves one registered root after two jobs.
- Observation: QCEC cannot yet build against Core v4 because it still includes a
  random-Clifford helper removed by separate work. DDSIM's broad test build also
  awaits its active Core-v4 compatibility stack. Neither blocker belongs to
  issue #2103.
- Observation: ProblemSolver still supports Python 3.10, whereas the Core-v4 and
  DDSIM migration requires Python 3.11. A conditional dependency/import uses
  DDSIM on 3.11+ and the released Core 3.x helper on 3.10.
- Observation: Core's Python `VectorDD.get_vector()` already allocates a NumPy
  buffer with independent capsule ownership. Returning that array directly
  avoids a second exponential-size allocation and remains valid after the DD
  root and package are released.

## Decision Log

- Decision: Keep the exact circuit kernel in Core and the public simulator class
  in DDSIM. Rationale: Core's QDMI DD device requires exact execution and cannot
  depend on DDSIM, while public simulator policy belongs in DDSIM. Date/Author:
  2026-08-27 / Codex.
- Decision: Expose `dd::SamplingResult` and an exact `dd::sample` overload that
  accepts a caller-owned generator. Do not expose a callback or execution class
  hierarchy. Rationale: counts, retained state, execution count, and generator
  continuation are the only shared exact contract; hooks changed DDSIM's
  approximation semantics. Date/Author: 2026-08-28 / Codex.
- Decision: Transfer the one registered input root into the exact kernel and
  return exactly one registered output root. Rationale: this matches Core's
  existing package-aware sampling behavior and makes exception cleanup and DDSIM
  state replacement explicit. Date/Author: 2026-08-28 / Codex.
- Decision: Let only the exact, non-approximating base `CircuitSimulator`
  delegate to Core. Derived simulators and approximation use
  `simulateWithHooks`; deterministic noise has an explicit override. Rationale:
  those paths own distinct DDSIM behavior that depends on virtual hooks and an
  established approximation schedule. Date/Author: 2026-08-28 / Codex.
- Decision: Retain Core's package-aware C++ `simulate` and `sample` adapters.
  Rationale: QCEC and SyReC use them with caller-managed DD packages and input
  states. Date/Author: 2026-08-27 / Codex.
- Decision: Move only zero-state Python `sample` and `simulate_statevector` to
  DDSIM. Rationale: they are public circuit-simulator conveniences; arbitrary
  input DD simulation remains a Core package primitive. Date/Author: 2026-08-27
  / Codex.
- Decision: Preserve Core's seed-zero random convention in the DDSIM helper by
  mapping zero to DDSIM's random sentinel and accept non-negative signed 64-bit
  seeds. Rationale: DDSIM's binding accepts signed seeds; a clear `ValueError`
  is preferable to overflow-dependent binding behavior. Date/Author: 2026-08-28
  / Codex.
- Decision: Defer changelog numbers, DDSIM Core-version pins, and the
  ProblemSolver lock update until the coordinated pull requests/releases exist.
  Rationale: inventing unavailable PR numbers or released dependency versions
  would leave invalid metadata. Date/Author: 2026-08-28 / Codex.

## Outcomes & Retrospective

The implementation now establishes the intended ownership boundary. Core has one
exact kernel covering unitary operations, virtual swaps, terminal and dynamic
measurement, reset, classical control, RNG reuse, permutations, garbage, and
global phase. DDSIM's exact base simulator consumes that result; DDSIM alone
retains approximation, subclass hooks, and noisy density execution. The two
high-level Python helpers live in DDSIM, and Core's generated stub no longer
exports them.

Focused and broad validation found and fixed three ownership defects: the
global-phase edge, repeated DDSIM vector roots, and repeated deterministic
density roots. It also found the malformed dynamic-measurement exception path,
which is now prevalidated, and a hook-path measurement-order discrepancy, which
is now covered. Remaining work is release coordination rather than feature
design: the DDSIM branch is stacked on its active Core-v4 migration, Core and
DDSIM changelog entries need eventual PR numbers, DDSIM's dependency pins must
move to Core v4, and ProblemSolver's lock file can only resolve after DDSIM 2.6
exists.

## Context and Orientation

`dd::Package` in `include/mqt-core/dd/Package.hpp` owns DD nodes. A returned
root edge is kept alive by explicit `incRef` and `decRef` calls. The new result
contract must therefore describe whether the input or output owns a registered
root, including on exceptions and zero-shot jobs.

Core declares circuit helpers in `include/mqt-core/dd/Simulation.hpp` and
implements them in `src/dd/Simulation.cpp`. `dd::simulate` applies a unitary
`qc::QuantumComputation` to a caller-supplied state. `dd::sample` has a
caller-supplied package overload and a zero-state compatibility overload. The
Core QDMI DD device calls these functions from `src/qdmi/devices/dd/Device.cpp`.

DDSIM's public `CircuitSimulator` is declared in `include/CircuitSimulator.hpp`
and implemented in `src/CircuitSimulator.cpp`. It stores a circuit, a DD
package, a persistent random-number generator, and the final `rootEdge`. Its
exact base path calls Core. `simulateWithHooks` remains for DDSIM-owned
approximation and subclass behavior. `DeterministicNoiseSimulator` explicitly
selects that hook path and uses density-matrix DDs.

Core Python bindings are registered in `bindings/dd/register_dd.cpp`. DDSIM's
moved helpers are implemented in `python/mqt/ddsim/_simulation.py` and
re-exported by `python/mqt/ddsim/__init__.py`. ProblemSolver imports sampling in
`src/mqt/problemsolver/csp.py` and `src/mqt/problemsolver/tsp.py`.

QCEC uses caller-supplied unitary `dd::simulate`. SyReC uses caller-supplied
`dd::sample` on reversible circuits. Neither repository should acquire a DDSIM
dependency.

## Milestones

### Milestone 1: Establish one exact Core execution contract

Add `dd::SamplingResult` and the generator-aware `dd::sample` overload in
`include/mqt-core/dd/Simulation.hpp`. Implement circuit analysis and exact
execution in `src/dd/Simulation.cpp`. Static circuits execute once even for zero
shots and retain an uncollapsed canonical state. Dynamic circuits execute once
per shot; zero shots return the input unchanged, otherwise the last collapsed
canonical state is retained. The input registered root transfers to the
function, and the returned state owns one root.

Use an ordered list for terminal measurement assignments. Explicit results use
the classical width; implicit sampling uses the qubit width. Finalization
applies the output permutation, removes garbage qubits, and applies global
phase. Move virtual execution helpers into the implementation file and fix
`applyGlobalPhase` ownership in `src/dd/Operations.cpp`.

This milestone is complete because the Core DD tests cover all of these
behaviors, including malformed measurement cleanup.

### Milestone 2: Delegate DDSIM's exact base simulator

In `src/CircuitSimulator.cpp`, call Core only when the object is the exact base
class and approximation is disabled. Preserve the previous root until Core
returns successfully, then release it, install the returned root, and add the
reported execution count. Keep `simulateWithHooks` for behavior that genuinely
depends on DDSIM virtual hooks. Give deterministic noise an explicit override
and release its prior aligned density root on reinitialization.

This milestone is complete because nine focused C++ tests cover exact static and
dynamic execution, zero shots, RNG continuation, failed-job recovery, derived
hooks, approximation scheduling, and deterministic density roots.

### Milestone 3: Move Python ownership and migrate the consumer

Remove Core's `sample` and `simulate_statevector` bindings and regenerate
`python/mqt/core/dd.pyi`. Add equivalent DDSIM helpers in
`python/mqt/ddsim/_simulation.py`. `sample` constructs a `CircuitSimulator`;
`simulate_statevector` uses Core's package-aware unitary primitive, copies the
NumPy view, and releases the final root in `finally`.

Update both upgrade guides and relevant simulator documentation. Add DDSIM as a
conditional ProblemSolver dependency on Python 3.11+ and retain the Core 3.x
import on Python 3.10. Do not change QCEC or SyReC.

This milestone is complete because the built DDSIM binding passes all ten
focused standalone tests against the modified Core package.

### Milestone 4: Validate and coordinate releases

Run Core's DD, QDMI, stub, Python, documentation, formatting, and lint checks.
Build DDSIM against the Core worktree and run the focused C++ and Python tests,
then run formatting and lint. Run ProblemSolver tests after its dependency can
resolve. Inspect every diff and record independent blockers.

Before merge, add changelog entries using actual pull request numbers. Land and
release Core v4 first, update and land DDSIM's Core-v4 migration plus this work,
release DDSIM 2.6, then regenerate the ProblemSolver lock and run its tests.

## Plan of Work

The implementation order is Core, DDSIM, then ProblemSolver. In Core, edit
`include/mqt-core/dd/Simulation.hpp`, `src/dd/Simulation.cpp`,
`include/mqt-core/dd/Operations.hpp`, and `src/dd/Operations.cpp`, followed by
focused DD tests. Remove the high-level bindings, regenerate the stub, and
update the DD guide and upgrade guide.

In DDSIM, change the base simulator's exact path, retain a clearly named hook
fallback, repair repeated density-root ownership, add the two Python helpers,
and cover both exact delegation and specialized fallbacks. In ProblemSolver,
change only the dependency and the two imports needed for the moved helper.

Finally, run validation, search for stale Core Python imports and obsolete
callback design text, and inspect diffs relative to the correct base in each
repository. Do not publish, push, or modify GitHub without separate
authorization.

## Concrete Steps

From the Core repository root, run:

    cmake --build build/release --target mqt-core-dd-test -j2
    build/release/test/dd/mqt-core-dd-test
    cmake --build build/release --target mqt-core-qdmi-ddsim-device-test -j2
    build/release/test/qdmi/devices/dd/mqt-core-qdmi-ddsim-device-test
    uvx nox -s stubs
    uvx nox -s tests -- test/python/dd/test_dd_package.py
    uvx nox --non-interactive -s docs
    uvx nox -s lint

From the DDSIM repository root, configure against the Core source worktree,
build `mqt-ddsim`, manually link the focused test if the broader pending Core-v4
migration prevents the existing test target from configuring, and run:

    /path/to/focused-test --gtest_filter='CircuitExecutionTest.*'

Build `mqt-ddsim-bindings`, stage its extension beside the source package in a
temporary directory, and run:

    python -m pytest test/python/simulator/test_standalone_simulator.py -n 0

Then run DDSIM's Ruff, Markdown, ClangFormat, and diff checks. From the
ProblemSolver repository root, run Ruff immediately; after DDSIM 2.6 is
available, regenerate `uv.lock` and run the CSP and TSP tests.

## Validation and Acceptance

Core is accepted when a seeded generator continues across calls; every success
returns exactly one referenced final state; every failure releases the
transferred input; static circuits report one execution even for zero shots; and
dynamic circuits report one execution per shot. Counts are big-endian, explicit
measurements preserve every assignment at classical width, and implicit results
use qubit width. Layouts, virtual swaps, output permutations, garbage, and
global phase produce a canonical retained state.

DDSIM is accepted when the ordinary exact base simulator delegates to Core,
repeated calls do not leak roots, failed calls preserve the previous state,
fixed seeds continue the same stream, and approximation and derived/noisy
simulators retain their DDSIM-owned hook behavior. Python is accepted when
`mqt.ddsim.sample` and `mqt.ddsim.simulate_statevector` return counts and an
independent NumPy array, while the generated Core stub no longer exports them.

No new Core production dependency is permitted. QCEC and SyReC must not gain a
DDSIM dependency. The only downstream dependency addition is ProblemSolver's
conditional DDSIM requirement.

## Idempotence and Recovery

Source edits, generation, builds, and tests are safe to repeat. Preserve
unrelated user changes and never reset another task's worktree. DDSIM's work is
stacked on a separate Core-v4 migration, so compare it with that migration base
until the prerequisite lands.

The exact kernel owns the input root immediately. A guard releases it on every
throw, and successful callers must eventually release the returned root. If a
root-set test fails, inspect the precise `incRef`/`decRef` transition rather
than suppressing the error. Stub generation is the only permitted way to edit
Core's `.pyi` file.

## Artifacts and Notes

Recorded final evidence:

    Core DD suite: 292 tests passed
    Core QDMI DD device: 51 tests passed
    Core stub generation: passed
    Core focused Python sessions (3.11-3.14): passed
    Core full lint session: passed
    DDSIM focused circuit execution: 9 tests passed
    DDSIM focused Python helper file: 10 tests passed
    DDSIM full lint session: passed
    ProblemSolver Ruff/format/Markdown checks: passed
    ClangFormat and git diff checks: passed for all changed files

The DDSIM focused Python test used an actual locally built extension linked
against this Core worktree. Broad DDSIM configuration remains coupled to its
separate active Core-v4 migration. ProblemSolver's lock cannot include an
unreleased DDSIM 2.6 artifact. QCEC's independent removed-header failure is not
evidence against this execution change.

The Core documentation build reached the changed DD guide but the existing
documentation environment failed independently: generated C++ namespace pages
did not recognize `doxygennamespace`, and the spawned Markdown build loaded a
stale Core shared library with a missing `buildFunctionalityRecursive` symbol.
Markdown lint and all source-level documentation checks pass.

## Interfaces and Dependencies

`include/mqt-core/dd/Simulation.hpp` exposes:

    struct SamplingResult {
      std::map<std::string, std::size_t> counts;
      VectorDD state;
      std::size_t executions;
    };

    [[nodiscard]] SamplingResult sample(
        const qc::QuantumComputation&, VectorDD, Package&, std::size_t,
        std::mt19937_64&);

The `VectorDD` argument owns one registered root that transfers to the function.
`SamplingResult::state` owns one registered root on success. Core continues to
depend only on its IR and DD targets. DDSIM already depends on Core DD and calls
this exact kernel. ProblemSolver adds a direct conditional Python dependency on
`mqt-ddsim`; no other dependency edge changes.

Plan revision note (2026-08-28): Replaced the discarded callback design with the
implemented exact-only contract, corrected input ownership, recorded the DDSIM
specialization fallback, added discovered exception and root-lifetime bugs, and
updated progress, evidence, blockers, and release order.
