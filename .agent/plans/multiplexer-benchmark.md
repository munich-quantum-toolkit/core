# Add a scalable, validated quantum multiplexer benchmark

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core users can generate a typed quantum multiplexer benchmark whose output
is nontrivial and independently checkable. The program prepares every control
state with equal probability and rotates one target qubit by an angle selected
from the control value. The analytic reference predicts the complete sampled
distribution, so an execution test detects incorrect controls, angles, target
placement, and result-bit order.

The fixed angle schedule has an exact linear implementation. A benchmark with
`k` controls executes `k` Hadamard gates and `k` singly controlled Y rotations
instead of iterating over all `2^k` control states. The generated QC program
keeps those operations in structured loops and remains compact at the supported
maximum of 1024 total qubits.

## Progress

- [x] (2026-09-01) Added the typed C++ family, strict JSON forms, manifest, MLIR
      generator, Python binding, command-line registration, and initial tests.
- [x] (2026-09-02 14:20Z) Replaced the trivial reference with the analytic
      uniform-control distribution and raised the maximum to 1024 qubits.
- [x] (2026-09-02 14:20Z) Replaced the exponential selector with an exact
      structured linear circuit.
- [x] (2026-09-02 14:20Z) Added focused native distribution tests, generator
      structure checks, and a maximum-size Jeff byte round-trip.
- [x] (2026-09-02 14:20Z) Added and passed the Python QC-to-QCO DD sampling test
      with 16,384 shots and a fixed seed.
- [x] (2026-09-02 14:40Z) Passed the complete focused suites, full release build
      and CTest suite, generated-stub check, documentation build, C++ lint, and
      full lint; completed the final diff inspection.

## Surprises & Discoveries

- Observation: The fixed schedule is not a general arbitrary-angle multiplexer.
  For control bits `b_i`, its angle is
  `pi * s / 2^k = sum_i b_i * pi / 2^(k-i)`. Controlled Y rotations around the
  same axis add, so one singly controlled rotation per bit implements the exact
  same unitary.
- Observation: Uniform control preparation turns the former all-zero output into
  a distribution that tests every control state in one execution.
- Observation: The 1024-qubit program lowers to Jeff, serializes to bytes, and
  deserializes as valid Jeff in the focused MLIR test.
- Observation: The three-qubit program passes the public Python generation, QCO
  lowering, DD sampling, and analytic evaluation path with total variation
  distance below 0.03.
- Observation: Running `stubs` before `docs` left the shared MinSizeRel Python
  build tree configured with `BUILD_MQT_CORE_DOCUMENTATION=OFF`. The first docs
  attempt therefore reported 25 missing generated MLIR reference files. An
  explicit `mqt-core` reinstall with the docs session's CMake arguments
  regenerated those files, after which the unmodified docs command passed.

## Decision Log

- Decision: Keep the fixed schedule `theta(s) = s*pi/2^k`, where `k` is the
  control count. Rationale: it retains the existing one-parameter family and
  permits an exact linear circuit without an exponential angle payload.
  Date/Author: 2026-09-02 / Daniel Haag.
- Decision: Apply Hadamard gates to all controls and leave the target in `|0>`.
  Rationale: the resulting distribution exercises the multiplexer semantics
  instead of selecting only state zero. Date/Author: 2026-09-02 / Daniel Haag.
- Decision: Traverse controls from most to least significant, start the
  controlled Y angle at `pi/2`, and halve it after each iteration. Rationale:
  this directly implements the binary weights without an integer state count or
  a `2^k` loop. Date/Author: 2026-09-02 / Daniel Haag.
- Decision: Support 2 through 1024 total qubits. Rationale: 1024 is a clear
  power-of-two catalogue ceiling; at 1023 controls both the uniform control
  weight and the smallest scheduled angle remain nonzero as binary64 values, and
  the maximum program round-trips through Jeff. Date/Author: 2026-09-02 / Daniel
  Haag.
- Decision: Store the target measurement at result index zero and control `i` at
  index `i + 1`. Rationale: the displayed big-endian result is `c[k-1]...c[0]t`,
  which permits direct binary interpretation of its control prefix. Date/Author:
  2026-09-02 / Daniel Haag.
- Decision: Keep family ID `multiplexer`, definition version 1, and the sole
  `qubits` parameter. Rationale: the family is unreleased and needs no
  compatibility method or arbitrary-angle variant. Date/Author: 2026-09-02 /
  Daniel Haag.

## Outcomes & Retrospective

The implementation now has a nontrivial analytic reference and an exact linear
generator. The maximum is 1024 qubits, where the compact structured program
lowers to Jeff and survives a stable byte round-trip. The public Python path
executes the generated circuit through QCO and the DD sampler and agrees with
the analytic distribution. All focused and repository-wide validation passes.

## Context and Orientation

The benchmark interface lives in `include/mqt-core/bench/Multiplexer.hpp` and
`src/bench/Multiplexer.cpp`. It validates the qubit count, describes one logical
`result` output, returns ideal probabilities, and evaluates sampled counts.
`src/bench/JSON.cpp` supplies the strict parameter schema, canonical instance
specification, manifest, and generic evaluation integration.

The generator lives in `mlir/bench/programs/Multiplexer.cpp`. It uses
`qc::QCProgramBuilder` and standard `scf.for` operations to construct compact QC
dialect IR. The normal compiler pipeline lowers that program to QCO and Jeff.
The Python binding in `bindings/bench/register_multiplexer.cpp` exposes the same
typed family and calls the shared generator.

For `q` total qubits, let `k = q - 1`. A displayed outcome is `c[k-1]...c[0]t`,
where `t` is the target bit. Interpret the control prefix as the binary fraction
`x = 0.c[k-1]...c[0]`, so the selected rotation is `theta = pi*x`. Its ideal
probability is `2^-k * cos(theta/2)^2` for `t = 0` and `2^-k * sin(theta/2)^2`
for `t = 1`.

## Plan of Work

Set `MultiplexerOptions::MAX_QUBITS` to 1024 and align constructor diagnostics
and the JSON schema. In `Multiplexer::probability`, validate the outcome, build
the binary control fraction without converting the 1023-bit prefix to an
integer, and return the appropriate squared sine or cosine times `2^-k`.

In the MLIR generator, allocate control-register storage without eager loads.
Emit one structured loop that applies Hadamard to each control. Emit a second
structured loop with a carried `f64` angle. Start at `pi/2`, load control
`k - 1 - step`, apply one singly controlled Y rotation to the target, multiply
the angle by `0.5`, and yield it to the next iteration. Keep the existing target
and control measurement order.

Replace tests that accepted the all-zero distribution or a `2^30` loop. Test the
complete three-qubit analytic distribution, normalization, evaluation, bit
order, structured linear generator, 1024-qubit Jeff byte round-trip, and the
public Python DD sampling path. Do not add an arbitrary-angle interface, new
dependency, or a separate benchmark method.

## Milestones

The semantic milestone is complete when the three-qubit probabilities match the
closed form, sum to one, and an intentionally incomplete histogram produces the
expected evaluation metrics. Boundary construction must accept 1024 and reject
1025.

The generation milestone is complete when the QC module contains one
Hadamard-loop body and one single-control Y-rotation-loop body, with a
descending control index and halved angle. The maximum case must lower to Jeff
and round-trip through the binary APIs.

The execution milestone is complete when a generated three-qubit program lowers
to QCO, samples through the DD runtime for 16,384 fixed-seed shots, and
evaluates with total variation distance below 0.03.

## Concrete Steps

Run commands from the repository root. Build and execute the focused native and
MLIR tests:

    cmake --preset release
    cmake --build --preset release --target mqt-core-bench-test \
        mqt-core-mlir-unittests-benchmark mqt-core-bench
    ./build/release/test/bench/mqt-core-bench-test
    ./build/release/mlir/unittests/bench/mqt-core-mlir-unittests-benchmark
    ctest --test-dir build/release -R '^mqt-core-mlir-benchmark-cli$' \
        --output-on-failure

Install the changed binding, verify generated stubs, and run Python tests:

    uv sync --inexact --no-dev --no-build-isolation-package mqt-core
    uvx nox -s stubs
    uv run --no-sync pytest test/python/test_bench.py test/python/test_cli.py

Run full validation and repository checks:

    cmake --build --preset release
    ctest --preset release
    uvx nox -s cpp-lint
    uvx nox --non-interactive -s docs
    uvx nox -s lint
    git diff --check
    git status --short

## Validation and Acceptance

The C++ tests must accept qubit counts 2 and 1024, reject 1 and 1025, validate
outcome widths and characters, and match all eight three-qubit probabilities.
The JSON tests must publish 1024 as the maximum and retain strict parsing,
canonical manifests, and stable case identity.

The MLIR tests must inspect semantic operations rather than a textual snapshot.
They must prove uniform control preparation, one single-control rotation body,
the correct binary angle order, no state-selection X gates, bounded generated
IR, and a successful maximum-size Jeff byte round-trip.

The Python test must exercise the public family, generator, QC-to-QCO lowering,
DD sampler, output order, and analytic evaluator in one path. Compilation alone
does not satisfy this acceptance criterion.

All focused tests, the full CTest suite, stub check, documentation build, C++
lint, full lint, and final diff checks must pass. Record an environment or
infrastructure failure with its command and output instead of presenting it as a
product failure.

## Idempotence and Recovery

The build, test, stub, documentation, and lint commands are repeatable. Build
output remains below `build/`. Stub generation is deterministic; never edit a
generated `.pyi` file by hand. This task uses a dedicated worktree. Preserve
unrelated changes and do not reset, clean, delete, or overwrite another
worktree.

## Artifacts and Notes

Validation on 2026-09-02 produced these results:

    13 tests from Multiplexer and BenchmarkJSON passed.
    2 multiplexer MLIR generation tests passed.
    1 end-to-end Python multiplexer test passed.
    42 native benchmark tests passed.
    12 MLIR benchmark generation tests passed.
    21 Python benchmark and CLI tests passed.
    The MLIR benchmark CLI test passed.
    The full release build completed.
    CTest reported 100% with 0 failures across 3972 tests; one pre-existing test was skipped.
    Stub generation completed without a tracked stub diff.
    C++ lint reported 0 clang-format and 0 clang-tidy findings.
    Documentation and full repository lint completed successfully.
    `git diff --check` completed without diagnostics.

The maximum-size MLIR test lowers the 1024-qubit program to Jeff, verifies that
serialization returns nonempty bytes, deserializes those bytes, verifies the
restored program, and checks that reserialization is stable.

## Interfaces and Dependencies

The installed C++ and Python interfaces remain `MultiplexerOptions`/`Options`
and `Multiplexer`, with `qubits` as their only parameter. The supported maximum
changes from 31 to 1024 and the ideal probability changes from an all-zero
placeholder to the documented uniform-control distribution. Family ID
`multiplexer` and definition version 1 remain unchanged.

Use only MQT Core, LLVM, MLIR, nanobind, nlohmann JSON, GoogleTest, and the
existing DD runtime. Add no dependency.

Revision note (2026-09-02): Replaced the exponential selector and trivial
reference with the exact linear fixed-schedule circuit, uniform-control
distribution, 1024-qubit boundary, semantic execution test, and current
validation evidence.
