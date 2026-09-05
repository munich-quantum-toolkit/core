# Add a fixed quantum teleportation benchmark

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core users can generate several typed structured quantum benchmarks, but the
catalog does not include the fixed quantum teleportation program. This change
adds one `teleportation` family to the C++, Python, JSON, command-line, and MLIR
interfaces. A user can construct the family without options, inspect its
analytic reference and manifest, and generate a structured QC or `jeff` program
with measurement-dependent corrections.

The command-line tool demonstrates the result. Given
`{"benchmark":"teleportation","parameters":{},"schema_version":1}`, the tool
generates a three-qubit program. The program measures Alice's two qubits,
applies X and Z corrections to Bob's qubit from those measurement results, and
then measures Bob's qubit.

## Progress

- [x] (2026-09-01 19:28Z) Studied the benchmark source, its paper, the removed
      historical Core emitter, and the current benchmark extension points.
- [x] (2026-09-01 19:28Z) Created a dedicated branch and worktree based on the
      quantum multiplexer work.
- [x] (2026-09-01 19:46Z) Added the fixed C++ family, JSON forms, manifest,
      analytic reference, and semantic tests.
- [x] (2026-09-01 19:50Z) Added structured QC generation, `jeff` lowering,
      command-line registration, and focused MLIR tests.
- [x] (2026-09-01 19:55Z) Added the Python family binding, regenerated stubs,
      and added focused Python tests.
- [x] (2026-09-01 21:05Z) Folded pull request 2324 into the existing unreleased
      benchmark changelog entry.
- [x] (2026-09-01 20:02Z) Passed the focused native, MLIR, command-line, Python,
      stub, direct `jeff` generation, and final diff checks. The broader lint
      session was intentionally omitted by task direction.

## Surprises & Discoveries

- Observation: The source calls the message state unknown, but it prepares the
  fixed state |+> by applying H to a fresh |0> qubit. Evidence: the source
  contains `h msg` after allocation and before the teleportation circuit.
- Observation: Every three-bit outcome has probability 1/8, even if one or both
  corrections are absent. X stabilizes |+>, while Z changes it to |->; a final
  Z-basis measurement distinguishes neither state. The analytic reference
  therefore cannot test the correction data flow.
- Observation: Core previously emitted this program with the first Alice
  measurement at result index zero, the second at index one, and Bob's result at
  index two. The current big-endian convention renders those bits as `b2b1a`.
- Observation: Successful lowering alone did not prove that Bob was measured
  after both corrections because the uniform reference cannot detect an early
  final measurement. The focused MLIR test now checks the operation order.

## Decision Log

- Decision: Make `Teleportation` a fixed family with no empty options type and
  no `options()` accessor. Rationale: the source defines one program and an
  empty type would expose no choice or information. Instance specifications
  still contain a strict empty `parameters` object for the common JSON envelope.
  Date/Author: 2026-09-01 / Daniel Haag.
- Decision: Store Alice's message result `a` at result index zero, Alice's
  entangled-qubit result `b1` at index one, and Bob's final result `b2` at index
  two. Rationale: this preserves the historical Core emitter and records values
  in source declaration order. Date/Author: 2026-09-01 / Daniel Haag.
- Decision: Omit the source's three initial resets. Rationale: allocation
  already creates qubits in |0>, so adjacent resets add no behavior and create
  needless output. Date/Author: 2026-09-01 / Daniel Haag.
- Decision: Use measurement results directly as the two `scf.if` conditions.
  Rationale: the scalar values match the source's `a` and `b1` bits, expose the
  feed-forward dependencies, and avoid redundant loads from the result register.
  Date/Author: 2026-09-01 / Daniel Haag.
- Decision: Use reference model `teleportation`, definition version 1, and no
  success outcome. Rationale: the model name is a short stable family name, all
  eight outcomes are equally likely, and no single result denotes protocol
  success. Date/Author: 2026-09-01 / Daniel Haag.

## Outcomes & Retrospective

The implementation adds one fixed `teleportation` family across the benchmark
library, JSON and command-line surfaces, structured generator, Python bindings,
and generated stubs. The native benchmark tests, all-family QC and `jeff`
generation tests, focused correction-data-flow test, command-line test, focused
Python tests, and stub session pass. A direct command-line generation also wrote
a non-empty `jeff` program with the expected stable case ID. The existing
unreleased benchmark-library changelog entry references pull request 2324. The
broader lint session was intentionally omitted by task direction.

## Context and Orientation

The installed benchmark library lives in `include/mqt-core/bench/` and
`src/bench/`. Each family validates its inputs when it has any, describes one
logical `Output`, and evaluates sampled counts against an analytic reference.
`include/mqt-core/bench/BenchmarkFamilies.inc` contains shared family metadata.
`src/bench/JSON.cpp` contains family-specific schemas, manifests, and generic
evaluation.

Structured generators live in `mlir/bench/programs/`. They use
`qc::QCProgramBuilder` to construct QC dialect operations. The catalog-generated
registry in `mlir/bench/Generate.cpp` parses an instance specification and calls
the typed generator. The normal compiler pipeline can then lower the QC program
to `jeff`, a structured exchange format.

Python bindings live in `bindings/bench/`. `register_bench.cpp` creates a direct
submodule for each family. A family-specific source file in the `mqt` namespace
registers its types and functions. Files below `python/mqt/core/bench/` are
generated stubs and must be regenerated with the repository's stub session.

Quantum teleportation transfers a one-qubit state with a shared entangled pair,
two measurements, and two classical corrections. In this fixed program, Alice
holds the message and one half of the pair; Bob holds the other half. Alice
entangles and measures her qubits. Bob applies X when Alice's entangled-qubit
measurement is one and Z when the message-qubit measurement is one. The source
prepares the message as |+> and measures Bob in the Z basis at the end.

This task builds on the benchmark family catalog and registration cleanup in the
quantum multiplexer work. It must preserve that base and must not modify another
task's worktree.

## Plan of Work

Define the parameterless `Teleportation` class in
`include/mqt-core/bench/Teleportation.hpp` and `src/bench/Teleportation.cpp`.
Use one `result` output of width three. Validate outcome syntax through the
shared helper, assign probability 1/8 to every valid outcome, and evaluate
counts without a success outcome.

Add stable ID `teleportation` and definition version 1 to
`include/mqt-core/bench/BenchmarkFamilies.inc`. Extend
`include/mqt-core/bench/JSON.hpp` and `src/bench/JSON.cpp` with a parser that
rejects every parameter, canonical empty parameters, model `teleportation`, and
an object schema with no properties. Keep catalog rows in lexical order. Cover
the contract in `test/bench/test_teleportation.cpp` and
`test/bench/test_json.cpp`.

Implement generation in `mlir/bench/programs/Teleportation.cpp`. Allocate the
message, Alice, and Bob qubits plus the three-bit result register. Apply H to
the message, create the Alice--Bob Bell pair with H and CX, apply CX and H for
Alice's joint measurement, and store `a` and `b1` at result indices zero and
one. Apply X to Bob under `b1` and Z under `a`, then store Bob's result at index
two. Do not reset freshly allocated qubits.

Register the generator in `mlir/bench/programs/Programs.h`,
`mlir/bench/programs/CMakeLists.txt`, and `mlir/include/mlir/bench/Generate.h`.
The catalog row supplies the implementation wrapper and dispatch entry. Extend
the MLIR unit and command-line tests. In addition to the all-family QC and
`jeff` smoke test, add one focused test that proves the first condition is the
`b1` measurement and contains X, while the second is the `a` measurement and
contains Z. Avoid tests of generic builder internals.

Register `mqt.core.bench.teleportation` through
`bindings/bench/register_teleportation.cpp` and
`bindings/bench/register_bench.cpp`. Expose `Teleportation()` directly, add the
family behavior test, regenerate stubs, and add the family to the command-line
test. Qualify the benchmark documentation's claim about option types because a
fixed family has none.

When the pull request number exists, add its reference to the existing
unreleased structured-benchmark entry in `CHANGELOG.md`; do not add a second
entry for an extension to unreleased functionality. Do not add an upgrade-guide
or glossary entry.

## Milestones

The semantic milestone supplies the C++ family, exact analytic reference, JSON
schema, manifest, and case identity. The focused benchmark test must accept all
eight valid outcomes, reject invalid outcomes, calculate exact evaluation
metrics, and round-trip the empty-parameter instance and manifest.

The generation milestone supplies structured QC generation and `jeff` lowering.
The MLIR tests must generate the family as valid QC and `jeff`, omit redundant
resets, and prove that Bob's X and Z corrections depend on the correct Alice
measurements.

The public-interface milestone supplies the Python submodule, generated stubs,
command-line registration, documentation adjustment, and focused family behavior
test.

## Concrete Steps

Run commands from the repository root. Configure and build the release preset,
then run the focused native and MLIR tests:

    cmake --preset release
    cmake --build --preset release --target mqt-core-bench-test \
        mqt-core-mlir-unittests-benchmark mqt-core-bench
    ./build/release/test/bench/mqt-core-bench-test
    ./build/release/mlir/unittests/bench/mqt-core-mlir-unittests-benchmark
    ctest --test-dir build/release -R '^mqt-core-mlir-benchmark-cli$' \
        --output-on-failure

Install the changed binding without build isolation, regenerate stubs, and run
the focused Python tests:

    uvx nox -s stubs
    uvx nox -s tests-3.13 -- test/python/test_bench.py test/python/test_cli.py

Run the repository checks after focused validation when the broader lint session
is in scope:

    uvx nox -s lint
    git diff --check
    git status --short

## Validation and Acceptance

The C++ tests must report output `result` at width three, assign probability
0.125 to all eight valid outcomes, reject wrong widths and non-binary
characters, and calculate zero total variation distance and unit squared
Hellinger fidelity for uniform counts. Counts concentrated on one valid outcome
must produce total variation distance 0.875 and squared Hellinger fidelity
0.125. No evaluation may report a success probability.

The JSON tests must prove canonical empty parameters, rejection of every
parameter, registry order, schema strictness, manifest integrity, stable case
identity, and generic evaluation. The expected case ID is
`sha256-8abc3c4e4adb4f0fde27c0d3562acddb8c79442fbadf613098878d448b302251`.

The MLIR tests must generate every benchmark as QC and `jeff`. For
teleportation, they must prove the two corrections depend on the intended
measurement results and contain the intended gates. The shared reset test must
show that allocation is not followed by a reset.

Python must expose `mqt.core.bench.teleportation.Teleportation()` with the usual
output, reference, JSON, case-ID, evaluation, and generation methods. It must
not expose a meaningless empty option type. The command-line tool must list
seven families and accept an empty-parameter teleportation instance.

`git diff --check` must pass. The broader repository lint session remains the
normal final check, but task direction excluded it at this stopping point. Any
environment or infrastructure failure must be recorded with its exact command
and output instead of being presented as a product failure.

## Idempotence and Recovery

The build, test, stub, and lint commands are repeatable. Build output remains
below `build/`. Stub generation is deterministic; never edit a generated `.pyi`
file by hand. If a generated stub differs unexpectedly, inspect the native
binding and rerun the stub session.

This task uses a dedicated worktree. Do not reset, clean, delete, or overwrite
another worktree. Preserve unrelated changes and stop if they overlap this
scope.

## Artifacts and Notes

The source program's semantic core is:

    h msg
    h alice
    cx alice, bob
    cx msg, alice
    h msg
    a = measure msg
    b1 = measure alice
    if b1: x bob
    if a: z bob
    b2 = measure bob

The output register stores `a`, `b1`, and `b2` at indices zero, one, and two.
Because logical outcome strings are big-endian, the displayed order is `b2b1a`.

Focused validation produced these results:

    44 native benchmark tests passed
    12 structured generation tests passed
    1 command-line integration test passed
    22 focused Python tests passed
    stub generation passed
    direct teleportation jeff generation returned case ID
      sha256-8abc3c4e4adb4f0fde27c0d3562acddb8c79442fbadf613098878d448b302251

## Interfaces and Dependencies

The installed C++ interface adds a parameterless `mqt::bench::Teleportation`,
plus matching serialization, manifest, case-ID, and generation overloads. The
Python interface adds the direct `mqt.core.bench.teleportation` submodule with
`Teleportation`.

Use only the dependencies already used by the benchmark library: MQT Core, LLVM,
MLIR, nanobind, nlohmann JSON, and GoogleTest. Add no dependency.

Revision note (2026-09-01): Initial plan after source, literature, historical
emitter, API, lowering, and test-scope research.

Revision note (2026-09-01): Recorded the completed implementation, focused
validation evidence, final-measurement ordering test, pending changelog
reference, and omitted broad lint session.

Revision note (2026-09-01): Recorded draft pull request 2324 and its changelog
reference.
