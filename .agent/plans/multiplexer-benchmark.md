# Add a typed quantum multiplexer benchmark

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core can generate five typed structured benchmark families, but it cannot
currently generate the quantum multiplexer program used by the `jeff`
structured-benchmark collection. After this change, users can configure a
quantum multiplexer by total qubit count through C++, Python, or a strict JSON
instance specification. They can inspect its analytic reference and manifest,
generate a structured QC program, and lower that program to `jeff`.

The observable end-to-end case is a seven-qubit instance. The command-line tool
accepts an instance specification with benchmark ID `multiplexer` and `qubits`
equal to 7, then writes a QC or `jeff` program and its manifest. The program
contains a loop over all control states, nested loops that select each control
pattern, and a uniformly controlled Y rotation.

## Progress

- [x] (2026-08-31 17:07Z) Studied the supplied OpenQASM program, the referenced
      paper, the removed MQT Core emitter, and the current typed benchmark
      extension points.
- [x] (2026-08-31 17:07Z) Chose the instance semantics, angle schedule, result
      order, analytic reference, and representable qubit limit.
- [x] (2026-08-31 18:58Z) Added the typed C++ family, strict JSON forms, and
      semantic tests.
- [x] (2026-08-31 18:58Z) Added the structured MLIR emitter, generation
      dispatch, and emitter tests.
- [x] (2026-08-31 18:58Z) Added Python bindings and tests, then regenerated the
      type stubs.
- [ ] Add the public glossary term and the new pull request number to the
      existing structured-benchmark changelog entry (completed: glossary;
      remaining: pull request number after creation).
- [x] (2026-08-31 19:43Z) Completed the final diff review and focused
      validation. Passed 41 semantic tests, 12 MLIR generation tests, the CLI
      test, 21 focused Python tests, stub generation, manual seven-qubit QC and
      `jeff` generation, maximum-size `jeff` lowering, formatting checks, and
      `git diff --check`. The full documentation build reached the new benchmark
      and then failed because the host has no Graphviz `dot` executable. The
      user asked not to run separate repository or C++ lint sessions.

## Surprises & Discoveries

- Observation: The OpenQASM program exposes an arbitrary angle array, while the
  removed MQT Core benchmark and the prior `multiplexer_7.jeff` artifact use a
  fixed evenly spaced schedule. Evidence: the removed emitter advances the angle
  by pi divided by the number of control states, and the prior seven-qubit
  artifact was generated through the size-only benchmark catalog.
- Observation: QC index values lower to signed 32-bit integers in the current
  QCO-to-jeff conversion. Evidence: the type conversion in
  `mlir/lib/Conversion/QCOToJeff/QCOToJeff.cpp` maps MLIR `index` to `i32`.
- Observation: The supplied OpenQASM input leaves every control qubit and the
  target in the zero state. The selected angle for control state zero is also
  zero. The exact output distribution is therefore a point mass at the all-zero
  outcome, even though the remaining program structure is important for compiler
  benchmarks.
- Observation: The maximum-size QC program contains 104 operations. Evidence:
  the focused emitter test counts the generated module for 31 qubits. The
  operation count grows with the 30 explicit control references but not with the
  2^30 runtime loop iterations.
- Observation: The warning-as-error documentation build reaches and executes the
  structured benchmark notebook, then fails in the unrelated decision diagram
  notebook because the host has no Graphviz `dot` executable. Evidence:
  `uvx nox --non-interactive -s docs` reports
  `ExecutableNotFound: failed to execute PosixPath('dot')` in
  `docs/dd_package.md`.

## Decision Log

- Decision: Expose only the total qubit count as an instance parameter and use
  angle theta(s) = s times pi divided by 2^(qubits - 1). Rationale: this is the
  schedule used to generate the prior `jeff` artifact, keeps instance
  specifications compact, and still represents the uniformly controlled rotation
  defined by the OpenQASM program. Date/Author: 2026-08-31 / Daniel Haag and
  Codex.
- Decision: Support total qubit counts from 2 through 31. Rationale: one qubit
  is the target, at least one is a control, and the largest state-loop bound is
  then 2^30, which fits the signed 32-bit index type used by `jeff` lowering.
  Date/Author: 2026-08-31 / Codex.
- Decision: Return one `result` register with the target at index zero and
  control i at index i + 1. Rationale: MQT Core benchmark outcomes place the
  highest register index at the left, so this layout reads as the control state
  followed by the target bit and combines the two OpenQASM outputs without
  adding a second logical output model. Date/Author: 2026-08-31 / Codex.
- Decision: Use an analytic point distribution at the all-zero outcome and do
  not define a success outcome. Rationale: it is the exact result for the
  supplied zero input and fixed angle schedule; structural tests will protect
  the otherwise unobserved branches. Date/Author: 2026-08-31 / Codex.
- Decision: Keep this work in its dedicated worktree and preserve all changes in
  other worktrees. Rationale: the user requested parallel-safe isolation.
  Date/Author: 2026-08-31 / Daniel Haag and Codex.

## Outcomes & Retrospective

The semantic, generation, and public-integration milestones are complete. MQT
Core now has one typed `multiplexer` family in C++, JSON, MLIR, and Python. The
emitter preserves the source program's control-state loop while fixing the angle
schedule to the historical MQT benchmark definition. Both the representative
seven-qubit case and the 31-qubit boundary lower to `jeff`.

The review found that the all-zero analytic reference cannot detect most emitter
wiring errors. The final structural tests therefore check the flip-loop order,
all control and target operands, measurement bounds and result indices, and the
maximum state-loop bound. This supplements the reference-distribution tests
without exposing internal helper names as a contract.

The only remaining work is administrative. Create the pull request, add its
number to the existing changelog entry, and commit that changelog update. The
documentation limitation is environmental: the new benchmark page executes, but
a later decision-diagram page requires the missing Graphviz `dot` executable.

## Context and Orientation

The installed semantic library lives under `include/mqt-core/bench/` and
`src/bench/`. Each family has an options structure and a validated benchmark
class. The class owns one logical `Output`, calculates ideal probabilities, and
evaluates sampled bit-string counts. `src/bench/JSON.cpp` contains a private
registry that maps stable benchmark IDs to schemas and evaluation callbacks.
`include/mqt-core/bench/JSON.hpp` exposes typed parse, serialization, manifest,
and case-ID overloads.

Structured emitters live in `mlir/bench/programs/`. An emitter uses
`qc::QCProgramBuilder` to create QC dialect operations. A private registry in
`mlir/bench/Generate.cpp` parses an instance specification and calls the typed
emitter. The generated QC program then passes through the existing compiler
pipeline to produce `jeff`, a structured interchange format for quantum
programs.

Python bindings live in `bindings/bench/`. `register_bench.cpp` creates one
direct submodule per family, while a separate registration source defines the
family's explicit classes and functions. Files below `python/mqt/core/bench/`
are generated type stubs and must be regenerated rather than edited.

The quantum multiplexer in this plan has k control qubits and one target. For
each of the 2^k control configurations, it applies a Y-axis rotation selected by
that configuration. The source program realizes the selection with an outer
state loop. Two inner loops temporarily invert controls whose corresponding
state bit is zero, so a single all-ones multi-controlled rotation implements
each configuration. The benchmark then restores the controls before advancing to
the next state.

The related cleanup commit is independent: it removes a brittle Python
namespace-shape test and removes the extra binding implementation namespace. Do
not fold unrelated edits or changes from another worktree into this plan.

## Plan of Work

Create `include/mqt-core/bench/Multiplexer.hpp` and `src/bench/Multiplexer.cpp`.
Define `MultiplexerOptions` with `qubits` and a maximum of 31. Define
`Multiplexer` with `options()`, `output()`, `probability()`, and `evaluate()`.
Validate the qubit range. Set the output name to `result` and its width to the
total qubit count. Validate big-endian outcome strings through the existing
evaluation utility and assign probability one only to the all-zero string.

Extend `include/mqt-core/bench/JSON.hpp` and `src/bench/JSON.cpp` with benchmark
ID `multiplexer`, definition version 1, strict parsing of the required `qubits`
parameter, canonical serialization, manifest serialization and parsing, case
IDs, a JSON Schema with bounds 2 and 31, and generic evaluation. Keep the
registry in lexical order: `bv`, `ghz`, `grover`, `multiplexer`, `qft`, `qpe`.
Add semantic and JSON coverage in `test/bench/test_multiplexer.cpp` and
`test/bench/test_json.cpp`.

Create `mlir/bench/programs/Multiplexer.cpp`. Allocate indexable storage for the
controls, one target qubit, and one classical result register. Do not emit
resets next to allocation because allocation establishes the zero state. Emit an
angle-carrying outer `scf.for` from zero to 2^(qubits - 1). In each iteration,
emit two inner `scf.for` loops. Each inner loop extracts a state bit with signed
shift-right and bitwise-and, compares it with zero, and conditionally applies X
to the dynamically indexed control. Between the two inner loops, apply one
multi-controlled Y rotation. Advance the carried angle by pi divided by the
number of states. Measure the target into result bit zero and each control into
the next result bit.

Declare and register the emitter in `mlir/bench/programs/Programs.h`,
`mlir/bench/programs/CMakeLists.txt`, `mlir/include/mlir/bench/Generate.h`, and
`mlir/bench/Generate.cpp`. Extend
`mlir/unittests/bench/test_benchmark_generate.cpp` with end-to-end QC and `jeff`
generation and focused structure checks. Update the CLI family count and schema
case in `mlir/unittests/bench/test_benchmark_cli.cmake`.

Create `bindings/bench/register_multiplexer.cpp`, then register its direct
submodule in `bindings/bench/register_bench.cpp` and source list in
`bindings/bench/CMakeLists.txt`. Extend `test/python/test_bench.py` and
`test/python/test_cli.py`. Regenerate the stubs with the repository's Nox
session.

Add a glossary entry for quantum multiplexer and uniformly controlled one-qubit
gate in `docs/glossary.md`. Do not create a static family list in
`docs/benchmarks.md`; its executable registry output discovers the new family.
Once the pull request number exists, add it to the existing changelog entry for
the typed structured benchmark library and define the link at the bottom of
`CHANGELOG.md`. Do not add a new changelog bullet or an upgrade-guide section,
because this extends unreleased version 4 functionality.

## Milestones

The first milestone establishes the semantic benchmark family. At its end, C++
users can construct a validated multiplexer, inspect its output and analytic
reference, and round-trip its instance specification and manifest. The semantic
and JSON tests demonstrate the accepted range, canonical seven-qubit form,
stable case identity, and exact zero-input distribution.

The second milestone adds structured program generation. At its end, the MLIR
registry can generate the multiplexer from the same instance specification and
lower representative and maximum-size programs to `jeff`. Structural tests
demonstrate the state loop, the two control-selection loops, the controlled Y
rotation, the register wiring, and the compact host-side representation.

The third milestone completes the public integration. At its end, Python users
have the same typed API and generated stubs, the CLI lists and generates the new
family, and the glossary defines the public term. Focused C++, MLIR, CLI, and
Python tests pass. The existing changelog entry gains the new pull request
number after the pull request exists.

## Concrete Steps

Run all commands from the repository root in the dedicated worktree. Implement
the semantic layer first and run:

    cmake --preset release
    cmake --build --preset release --target mqt-core-bench-test
    ./build/release/test/bench/mqt-core-bench-test

Then implement generation and run:

    cmake --build --preset release --target mqt-core-mlir-unittests-benchmark mqt-core-bench
    ./build/release/mlir/unittests/bench/mqt-core-mlir-unittests-benchmark
    ctest --test-dir build/release -R '^mqt-core-mlir-benchmark-cli$' --output-on-failure

Install the changed bindings, regenerate declarations, and run the focused
Python tests:

    uv sync --inexact --no-dev --no-build-isolation-package mqt-core
    uvx nox -s stubs
    uv run --no-sync pytest test/python/test_bench.py test/python/test_cli.py

Build the documentation because the public glossary changes:

    uvx nox --non-interactive -s docs

Finish with the repository checks that apply to the completed batch:

    uvx nox -s lint
    git diff --check
    git status --short

The user has asked not to run a separate C++ lint session. Record that
`uvx nox -s cpp-lint` was not run for that reason. Do not turn validation into
an unrelated C++ lint cleanup.

## Validation and Acceptance

The semantic tests must accept qubit counts 2 and 31, reject 1 and 32, report
probability one for an all-zero string of the configured width, report zero for
other valid strings, reject malformed outcomes, and calculate exact distance and
fidelity values without a success probability.

JSON tests must prove the canonical seven-qubit instance specification, schema
bounds, registry order, manifest round trip, case-ID stability, tamper
rejection, and generic evaluation. Python must expose
`mqt.core.bench.multiplexer.Options` and `.Multiplexer`, produce the same JSON
and reference values, and return a generated QC program.

The MLIR tests must prove that a representative program contains one
angle-carrying outer loop, two nested bit loops, two classical conditionals,
dynamic control loads, and one multi-controlled Y rotation. They must verify the
target and control result indices, the absence of redundant resets, and
successful conversion to `jeff`. The generated operation count must remain
compact when the configured qubit count grows; the host implementation must not
unroll the exponential state loop.

The CLI test must list six families and accept a multiplexer instance. The
documentation build must accept the new glossary entry. Repository lint and
`git diff --check` must pass, except for any limitation recorded with its exact
output in this plan.

## Idempotence and Recovery

All build, test, stub, and documentation commands are repeatable. CMake output
stays below `build/`; generated stubs are deterministic tracked files. This
worktree uses `/opt/llvm-22` because the updated LLVM 23 API changes are still
uncommitted in another worktree. If a generated stub differs unexpectedly,
inspect the native binding definition and rerun the stub session; never
hand-edit the stub.

The worktree isolates this task. Do not reset, clean, delete, or overwrite files
in another worktree. Preserve unrelated changes found in this worktree and stop
if they overlap this scope.

## Artifacts and Notes

The canonical seven-qubit instance specification is:

    {"benchmark":"multiplexer","parameters":{"qubits":7},"schema_version":1}

For total qubit count n, the implementation uses n - 1 controls, 2^(n - 1)
control states, and angle theta(s) = s*pi/2^(n - 1). The exact zero-input
reference is the n-bit all-zero string.

## Interfaces and Dependencies

The installed C++ interface must add these types and functions under
`mqt::bench`:

    struct MultiplexerOptions {
      static constexpr size_t MAX_QUBITS = 31;
      size_t qubits;
    };

    class Multiplexer final {
    public:
      explicit Multiplexer(MultiplexerOptions options);
      const MultiplexerOptions& options() const noexcept;
      const Output& output() const noexcept;
      double probability(std::string_view outcome) const;
      Evaluation evaluate(const Counts& counts) const;
    };

The semantic JSON interface must add typed `multiplexerFrom...`,
`toInstanceSpecificationJSON`, `toManifestJSON`, and `caseId` overloads. The
source-build MLIR interface must add
`std::optional<mlir::QCProgram> generate(const Multiplexer&)`.

Use only existing MQT Core, LLVM, MLIR, nanobind, nlohmann JSON, GoogleTest, and
Python test infrastructure. Add no dependency. The implementation must follow
`AGENTS.md` and `docs/ai_usage.md`, preserve generated-file rules, and leave all
remote GitHub actions subject to explicit user approval.

Plan created on 2026-08-31 after the benchmark semantics and lowering limits
were established. The plan records the fixed angle schedule and single-output
layout so later implementation work does not have to infer them.
