# Refocus QCO DD execution and sampling

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds. Maintain it according to `.agent/PLANS.md`.

## Purpose / Big Picture

MQT Core shall execute the supported single-block QCO subset with decision
diagrams (DDs) and expose only functionality construction, simulation, and
zero-state sampling. Sampling returns the entry function's CBit register
results; programs with no CBit result retain final computational-basis sampling.
Compiling the terminal Bell and adaptive examples below to optimized QCO and
sampling them demonstrates the behavior end to end.

## Progress

- [x] (2026-08-26 00:00Z) Preserve the divergent local work and start from PR
  #2077 commit `97f904f6` without rewriting history.
- [x] (2026-08-26 22:08Z) Refactor attributes, direct loops, preparation,
  sampling classification, and the public API.
- [x] (2026-08-26 22:08Z) Add focused C++ coverage and regenerate stubs through
  `uvx nox -s stubs`.
- [x] (2026-08-26 22:17Z) Add the two compiler-to-sampler Python cases.
- [x] (2026-08-26 22:22Z) Run full local validation and prepare two signed
  additive commits.
- [ ] Push normally and inspect hosted checks.

## Surprises & Discoveries

- Observation: `Operation::fold` may update an operation in place. Evidence:
  MLIR's `Operation.h` documents empty successful fold results as in-place
  updates, so execution uses a narrow explicit evaluator.
- Observation: independent quantum and measurement histograms lose their
  correlation. Evidence: one shot cannot be reconstructed from two marginal
  histograms, so `SampleResult` is removed.
- Observation: optimized OpenQASM emits entry-block `qco.alloc` for scalar
  qubits and `qtensor.alloc` for register declarations. The executor maps only
  scalar allocations; the end-to-end test uses scalar qubits and leaves qtensors
  to #2078.

## Decision Log

- Decision: `sample` concatenates returned CBit registers in `func.return`
  order, each from bit `N-1` to bit `0`; no CBit result means `measureAll`.
  Rationale: the program output preserves correlations and matches QCO
  consumers. Date/Author: 2026-08-26, Codex with user approval.
- Decision: interpret `scf.for` directly with one 10,000-step budget per
  execution. Rationale: nested work is bounded without enlarging the IR.
  Date/Author: 2026-08-26, Codex with user approval.
- Decision: keep only zero-state sampling and add entry-block scalar
  `qco.alloc`. Rationale: this is the smallest compiler-to-sampler bridge;
  supplied states and qtensors have no demonstrated #2077 consumer. Date/Author:
  2026-08-26, Codex.

## Outcomes & Retrospective

The focused engine, all 3,872 runnable C++ tests, generated stub, and lint are
green. The compiler-to-sampler file passes all six tests against an installed
extension, including the two new cases. The Python Nox matrix passes 725 tests
on each of Python 3.11 through 3.13 and 736 on Python 3.14; documented skips are
unchanged. The bare root `uv run --no-sync` invocation cannot import the package
and therefore skips collection, while each Nox environment installs and tests
the built extension.

Relative to the original PR tip `97f904f6`, the total PR diff shrank from 1,670
insertions and 238 deletions to 1,658 insertions and 536 deletions. The
production implementation shrank from 1,232 to 1,231 lines. Generated `mlir.pyi`
churn accounts for 55 insertions in the final PR diff. Hosted CI remains to be
inspected after the normal push and will be reported in the PR handoff rather
than requiring a documentation-only commit.

## Context and Orientation

`mlir/lib/Dialect/QCO/Utils/DDFunctionality.cpp` prepares and interprets QCO IR.
A DD compactly represents a quantum state or operator. `ClassicalEnv` stores
runtime scalar attributes and shared CBit register cells.
`mlir/include/mlir/Dialect/QCO/Utils/DDFunctionality.h` is the C++ API;
`bindings/mlir/register_mlir.cpp` and generated `python/mqt/core/mlir.pyi`
provide Python. Focused tests are in
`mlir/unittests/Dialect/QCO/Utils/test_dd_functionality.cpp` and
`test/python/test_qco_dd.py`.

The Python surface is exactly:

    build_functionality(program, dd_package) -> MatrixDD
    simulate(program, initial_state, dd_package, seed=None) -> VectorDD
    sample(program, dd_package, shots=1024, seed=None) -> dict[str, int]

Follow `AGENTS.md` and `docs/ai_usage.md`. Preserve unrelated work, regenerate
stubs only through Nox, sign commits, and do not treat this plan as authority
for GitHub actions.

## Milestones

### Milestone 1: narrow and correct the engine

Store integer and index values as exact-width MLIR attributes. Validate the
containing module once, then reuse prepared wire mappings per shot. Execute
single-block calls and positive-step `scf.for` loops directly. Compute loop
spans in a one-bit-wider `APInt`; decrement the shared budget before each
iteration. Completion is the focused C++ target building with all
`QCODDFunctionalityTest` cases passing.

### Milestone 2: make sampling output-aware

Reject mixed CBit/non-CBit results and undefined returned cells. Defer only
entry-block measurements whose qubit result reaches only a sink or return and
whose bit is unused or stored into a returned register that is never loaded,
called, or otherwise consumed. Static sampling evolves once and maps sampled
physical wires into returned cells. Resets, callees, nested measurements, and
execution-dependent measurements run per shot. Completion includes the narrow
C++/Python APIs and regenerated stub.

### Milestone 3: prove and publish the stack foundation

Compile terminal Bell and adaptive reset programs with
`OutputFormat.QCO_OPTIMIZED`; expect respectively only `00`/`11` and `00`/`01`,
with both outcomes present. Finish full validation, add the test and final plan
outcome in a second signed commit, then push normally and inspect PR №2077
checks.

## Plan of Work

Keep explicit evaluators only for integer constants, logical operations,
add/subtract/multiply, shifts, comparisons, select, and unsigned index casts. Do
not call generic folding. Preserve shared CBit storage across calls. Remove
history sampling, supplied-state sampling, redundant call-arity checks, and
their tests. Cover output ordering, constants, undefined cells, static versus
dynamic measurement classification, extreme loop bounds, shared loop budget, and
DD reference balance.

PR №2078 remains responsible for runtime bindings, additional scalar types,
qtensors, and `scf.while`; PR №2079 remains responsible for multi-block CFG
execution, budgeted CFG transitions, and DD-native deallocation. Neither may
restore encounter histories, generic folding, or independent loop caps without a
compiler-produced consumer.

## Concrete Steps

Run from the repository root:

    cmake --preset release
    cmake --build --preset release --target mqt-core-mlir-unittest-qco-utils
    ctest --preset release -R QCODDFunctionalityTest --output-on-failure
    uvx nox -s stubs
    uv run --no-sync pytest test/python/test_qco_dd.py
    uvx nox -s lint
    cmake --build --preset release
    ctest --preset release --output-on-failure
    uvx nox -s tests

Before publication, compare the production and total diff with `97f904f6`, count
generated stub churn separately, fetch the PR branch, and rebase only if it
advanced. Verify every new commit with `git verify-commit` and push without
force.

## Validation and Acceptance

The focused C++ suite must show exact 10,000-loop success and 10,001 failure,
shared nested-loop accounting, returned-register order and MSB-first encoding,
one evolution for terminal output measurements, per-shot adaptive/callee/nested
execution, conservative unknown-use handling, and balanced DD roots after
success, failure, and zero shots. The Python test must compile optimized QCO,
produce exactly the expected keys, and sum to the requested shots. All commands
above must exit successfully; hosted checks must validate the pushed SHA.

## Idempotence and Recovery

Configuration, builds, tests, stub generation, lint, fetches, and diff checks
are repeatable. The original local work has a backup ref and binary patch
outside this plan. Keep commits additive and use a normal push. If the remote
advanced, fetch, rebase the unpublished commits, rerun validation, and retry; do
not modify #2078 or #2079 in this work.

Revision note (2026-08-26): Replaced the historical first-class-CBit plan with
this focused, self-contained execution and sampling plan; updated it after the
engine and focused validation milestones.
