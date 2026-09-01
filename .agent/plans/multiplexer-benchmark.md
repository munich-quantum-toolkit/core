# Add a typed quantum multiplexer benchmark

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core users can generate typed structured benchmarks, but the library does
not yet include a quantum multiplexer. This change adds one `multiplexer` family
to the C++, Python, JSON, command-line, and MLIR interfaces. A user selects the
total qubit count. MQT Core then supplies the fixed angle schedule, analytic
reference, manifest, and compact structured QC program.

The command-line tool demonstrates the result. Given
`{"benchmark":"multiplexer","parameters":{"qubits":7},"schema_version":1}`, the
tool generates a QC or `jeff` program. The program loops over the 64 control
states without expanding them into 64 copies in the host process.

## Progress

- [x] (2026-09-01) Added the typed C++ family, strict JSON forms, manifest,
      analytic reference, and semantic tests.
- [x] (2026-09-01) Added compact structured QC generation, `jeff` lowering,
      command-line registration, and MLIR tests.
- [x] (2026-09-01) Added the Python family binding and its focused behavior
      tests.
- [x] (2026-09-01) Folded this change into the existing structured-benchmark
      changelog entry.
- [x] (2026-09-01) Validated the final implementation on the cleanup base. The
      41 C++ benchmark tests, 12 MLIR generation tests, MLIR CLI test, 21
      focused Python tests, stub session, generated MLIR documentation, and
      complete documentation build passed.
- [x] (2026-09-01) Completed the full repository lint and final diff checks.

## Surprises & Discoveries

- Observation: The benchmark program is compact even though its runtime state
  loop grows exponentially. Evidence: the 31-qubit generation test bounds the
  host-side MLIR operation count and lowers the result to `jeff`.
- Observation: The zero input and fixed angle schedule produce the all-zero
  outcome with probability one. This reference alone cannot detect most wiring
  errors. Execution coverage must wait for a runtime that can execute the
  generated `jeff` program.
- Observation: A fresh worktree must generate the MLIR reference pages before
  the complete documentation build. Evidence: the first documentation build
  reported missing `docs/mlir/Dialects`, `docs/mlir/Conversions`, and
  `docs/mlir/Passes` files; the `mlir-doc` target generated those files and the
  next build passed.

## Decision Log

- Decision: Expose only the total qubit count and use angle theta(s) =
  s*pi/2^(qubits - 1). Rationale: this is the schedule used by the existing
  size-based benchmark and gives each control state one selected Y rotation
  without adding an arbitrary angle payload. Date/Author: 2026-09-01 / Daniel
  Haag.
- Decision: Support 2 through 31 total qubits. Rationale: one qubit is the
  target, at least one qubit is a control, and the largest state-loop bound is
  2^30, which fits the signed 32-bit integer representation used by current
  `jeff` lowering. Date/Author: 2026-09-01 / Daniel Haag.
- Decision: Store the target measurement at result index zero and control i at
  index i + 1. Rationale: this preserves the benchmark outcome order in one
  logical result register. Date/Author: 2026-09-01 / Daniel Haag.
- Decision: Add one benchmark-family catalog row and keep the parameter parser,
  schema, emitter, binding, and tests explicit. Rationale: the catalog supplies
  mechanical JSON and MLIR registration while the family-specific behavior
  remains readable. Date/Author: 2026-09-01 / Daniel Haag.

## Outcomes & Retrospective

The implementation adds one typed `multiplexer` family across the existing
benchmark layers. The family uses the same catalog, serialization, binding, and
generation extension points as the other families. Focused semantic, generation,
command-line, Python, stub, and documentation validation passes on the cleanup
base. The full repository lint and final diff checks also pass.

## Context and Orientation

The installed benchmark library lives in `include/mqt-core/bench/` and
`src/bench/`. Each family owns typed options, validates them, describes one
logical `Output`, and evaluates sampled counts against an analytic reference.
`include/mqt-core/bench/BenchmarkFamilies.inc` contains the shared family
metadata. `src/bench/JSON.cpp` contains the family-specific schemas, manifests,
and generic evaluation.

Structured generators live in `mlir/bench/programs/`. They use
`qc::QCProgramBuilder` to construct QC dialect operations. The catalog-generated
registry in `mlir/bench/Generate.cpp` parses an instance specification and calls
the typed generator. The normal compiler pipeline can then lower the QC program
to other supported forms.

Python bindings live in `bindings/bench/`. `register_bench.cpp` creates a direct
submodule for each family. A family-specific source file in the `mqt` namespace
registers its types and functions. Files below `python/mqt/core/bench/` are
generated stubs and must be regenerated with the repository's stub session.

A quantum multiplexer has k control qubits and one target qubit. Each of the 2^k
control basis states selects one one-qubit operation on the target. This
benchmark selects evenly spaced Y rotations. The generator uses one outer loop
over control states. Before each rotation, an inner loop applies X to controls
whose selected state bit is zero. This turns the selected pattern into the
all-ones pattern required by one multi-controlled rotation. A second inner loop
restores the controls.

## Plan of Work

Define `MultiplexerOptions` and `Multiplexer` in
`include/mqt-core/bench/Multiplexer.hpp` and `src/bench/Multiplexer.cpp`.
Validate 2 through 31 total qubits. Use one `result` output whose width equals
the total qubit count. The analytic reference assigns probability one to the
all-zero outcome and zero to every other valid outcome.

Add the stable ID `multiplexer` and definition version 1 to
`include/mqt-core/bench/BenchmarkFamilies.inc`. Extend
`include/mqt-core/bench/JSON.hpp` and `src/bench/JSON.cpp` with the required
integer `qubits` parameter, canonical instance specifications, manifests, case
IDs, and generic evaluation. Keep catalog rows in lexical order. Cover the
contract in `test/bench/test_multiplexer.cpp` and `test/bench/test_json.cpp`.

Implement generation in `mlir/bench/programs/Multiplexer.cpp`. Allocate the
controls, target, and result register. Emit an angle-carrying outer `scf.for`
from zero to 2^(qubits - 1). Emit two inner loops that test each bit of the
current control state and conditionally apply X. Place one multi-controlled Y
rotation between those loops. Increment the angle by pi/2^(qubits - 1). Measure
the target at result index zero and each control at the next index.

Register the generator in `mlir/bench/programs/Programs.h`,
`mlir/bench/programs/CMakeLists.txt`, and `mlir/include/mlir/bench/Generate.h`.
The catalog row supplies the implementation wrapper and dispatch entry. Extend
the MLIR unit and command-line tests. The tests must cover representative
structure, maximum-size compactness, and successful `jeff` lowering.

Register `mqt.core.bench.multiplexer` through
`bindings/bench/register_multiplexer.cpp` and
`bindings/bench/register_bench.cpp`. Add the family behavior test, regenerate
stubs, and add the family to the command-line test.

Add pull request 2299 to the existing unreleased structured-benchmark entry in
`CHANGELOG.md`; do not add a second entry for an extension to an unreleased
feature.

## Milestones

The semantic milestone supplies the validated C++ family, exact analytic
reference, JSON schema, manifest, and case identity. The focused benchmark test
must accept the boundary sizes, reject invalid sizes and outcomes, and
round-trip a seven-qubit instance and manifest.

The generation milestone supplies compact QC generation and `jeff` lowering. The
MLIR tests must generate the family as valid QC and `jeff`, omit redundant
resets, and keep the 31-qubit operation count bounded.

The public-interface milestone supplies the Python submodule, generated stubs,
command-line registration, changelog update, and focused family behavior test.

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

    uv sync --inexact --no-dev --no-build-isolation-package mqt-core
    uvx nox -s stubs
    uv run --no-sync pytest test/python/test_bench.py test/python/test_cli.py

Build the documentation and run the repository checks:

    uvx nox --non-interactive -s docs
    uvx nox -s lint
    git diff --check
    git status --short

## Validation and Acceptance

The C++ tests must accept qubit counts 2 and 31, reject 1 and 32, validate
outcome widths and characters, and calculate exact evaluation metrics. The JSON
tests must prove canonical serialization, schema bounds, registry order,
manifest integrity, stable case identity, and generic evaluation.

The MLIR tests must generate every benchmark as QC and `jeff`. For the
multiplexer, they must verify the maximum state-loop bound, lack of redundant
resets, compact maximum-size generation, and `jeff` lowering.

Python must expose `mqt.core.bench.multiplexer.Options` and
`mqt.core.bench.multiplexer.Multiplexer` only in the family submodule. The new
family must round-trip its JSON forms, evaluate counts, and generate a QC
program. The command-line tool must list six families and accept a multiplexer
instance.

The complete documentation build, full lint session, and `git diff --check` must
pass. Any environment or infrastructure failure must be recorded with its exact
command and output instead of being presented as a product failure.

## Idempotence and Recovery

The build, test, stub, documentation, and lint commands are repeatable. Build
output remains below `build/`. Stub generation is deterministic; never edit a
generated `.pyi` file by hand. If a generated stub differs unexpectedly, inspect
the native binding and rerun the stub session.

This task uses a dedicated worktree. Do not reset, clean, delete, or overwrite
another worktree. Preserve unrelated changes and stop if they overlap this
scope.

## Interfaces and Dependencies

The installed C++ interface adds `MultiplexerOptions` and `Multiplexer` under
`mqt::bench`, plus matching serialization, manifest, case-ID, and generation
overloads. The Python interface adds the direct `mqt.core.bench.multiplexer`
submodule with `Options` and `Multiplexer`.

Use only the dependencies already used by the benchmark library: MQT Core, LLVM,
MLIR, nanobind, nlohmann JSON, and GoogleTest. Add no dependency.

Revision note (2026-09-01): Rebased the plan on the benchmark-registration
cleanup, removed stale namespace-contract work, and recorded the catalog-based
extension points and current validation.
