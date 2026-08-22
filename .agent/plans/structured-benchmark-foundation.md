# Build a typed structured-benchmark foundation

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core currently contains two incomplete benchmark models. The installed
`MQT::CoreAlgorithms` library builds fixed `QuantumComputation` circuits, while
pull request #2135 adds structured MLIR programs selected only by a name and a
single size. Neither model can record a complete benchmark instance, expose
algorithm-specific parameters, or state how an execution result should be
checked.

After this work, C++ and Python users can configure GHZ, Grover, and quantum
phase estimation with typed options. A command-line tool can read the same
versioned JSON request, generate a structured QC or `jeff` program, and write a
manifest that identifies the exact instance and its analytic reference. The
reference can score measured counts without enumerating an exponential state
vector. The old circuit-building Algorithms library and its obsolete evaluation
program are removed after the new semantic tests cover the retained workloads.

Daniel Haag authored the existing #2135 and #2204 work. The implementation must
preserve those commit objects and author fields. New architecture and migration
work uses separate commits under the person who performs it. No remote branch,
pull request, or discussion is changed by this ExecPlan without separate human
authorization.

## Progress

- [x] (2026-08-22 09:24Z) Archived the exact local heads of #2135 and #2204 and
      created a separate implementation branch.
- [x] (2026-08-22 09:24Z) Merged current `origin/main` without rebasing Daniel's
      commits and verified the signed merge commit.
- [x] (2026-08-22 09:24Z) Read the repository agent rules, AI policy, and
      ExecPlan requirements.
- [x] (2026-08-22 09:31Z) Made QC program cleanup deterministic with ordered
      allocation tracking and passed the focused builder regression test.
- [x] (2026-08-22 09:59Z) Added the MLIR-free benchmark instance and reference
      library and passed all 20 focused C++ tests.
- [x] (2026-08-22 10:00Z) Prototyped returned-classical-register sampling and
      passed all eight focused sampling tests; this change must be restacked
      directly after #2077 before publication.
- [x] (2026-08-22 10:06Z) Added strict request, manifest, count, evaluation,
      JSON Schema, and stable case-ID support; all 29 benchmark-library tests
      pass.
- [x] (2026-08-22 11:07Z) Adapted Daniel's GHZ, Grover, QPE, and IQPE emitters
      to typed instances and passed structural and edge-case tests.
- [x] (2026-08-22 11:30Z) Replaced the size-only CLI with strict requests,
      deterministic manifests, analytic evaluation, and collision-safe output.
- [x] (2026-08-22 11:28Z) Added typed Python bindings and generated stubs
      without duplicating the MLIR runtime across extension modules.
- [x] (2026-08-22 11:27Z) Added structural, CLI, and cross-interface tests.
      Proved the independent DD semantic suite locally and reserved it for the
      follow-up stack above #2077, where its returned-register sampler belongs.
- [x] (2026-08-22 11:44Z) Removed the unowned `MQT::CoreAlgorithms` target and
      legacy DD evaluation after the replacement reference and emitter tests
      passed.
- [x] (2026-08-22 11:45Z) Added one benchmark guide and updated `CHANGELOG.md`
      and `UPGRADING.md` with both human contributors.
- [x] (2026-08-22 12:05Z) Completed independent reference, frontend, and
      packaging audits. Fixed cross-platform Grover iteration selection, exact
      JSON integer parsing, duplicate-key complexity, per-family definition
      versions, and unsafe output rollback.
- [x] (2026-08-22 12:08Z) Passed the full lint session and restacked the
      foundation without the returned-register sampler or nested-loop unroll
      fix. Every rewritten commit is maintainer-authored and signed. Daniel's
      exact commit objects remain unchanged below the signed merge.

## Surprises & Discoveries

- Observation: The worktree started detached at Daniel's exact #2135 head,
  `5e5580eb345249fd1123680994800c7dad85444d`, with no local changes. Evidence:
  `git status --short --branch` reported `HEAD (no branch)` and
  `git rev-parse HEAD` returned that hash.
- Observation: `origin/main` advanced after Daniel's last merge, but a
  merge-tree probe found no conflicts. A signed merge preserves all of Daniel's
  existing commit objects, unlike a rebase.
- Observation: Pull request #2078 already interprets `scf.for`, dynamic QTensor
  allocation, QTensor transport through regions and calls, and bound gate
  parameters. Adding narrower QTensor support or compensating loop-unroll
  behavior in the benchmark branch would duplicate that DD stack.
- Observation: Existing target mapping removes QTensor allocation, extraction,
  insertion, and deallocation before benchmark simulation. The benchmark branch
  therefore does not need #2078. Standard and iterative QPE still need full loop
  unrolling because #2077 does not interpret `tensor.extract` angle tables.
- Observation: Core already defines sampled bitstrings as big-endian: the
  leftmost character is the highest-index qubit. The benchmark result contract
  will reuse that convention.
- Observation: Replacing two `DenseSet<Value>` members with `SetVector<Value>`
  fixes nondeterministic cleanup at the shared builder boundary. Evidence:
  `QCTest.BuilderDeallocatesDynamicResourcesDeterministically` passes and checks
  exact allocation/deallocation order for qubits and registers.
- Observation: QPE reference probabilities can support precision above 1,024
  without a large-integer dependency. Binary long division builds the nearest
  lower outcome and exact remainder in linear time; a wrapped sine ratio then
  evaluates only the requested outcome.
- Observation: The QCO DD simulator already tracked measured CBit registers in
  its internal classical environment. Returning those bits after each shot
  provides independent benchmark outcomes without a second interpreter.
- Observation: Core has no installed stable-hash dependency. A small private
  SHA-256 implementation keeps the public target free of LLVM and has published
  empty, short, and multiblock test vectors.
- Observation: Returning an MLIR object from a second Python extension that
  statically links MLIR creates incompatible process-local MLIR type IDs. The
  benchmark extension now passes canonical request JSON to one private function
  in the existing `mqt.core.mlir` extension, which creates and owns the
  `QCProgram`.
- Observation: Comparing two nearly unit Grover probabilities made automatic
  iteration selection depend on `long double` width. Rounding the first-peak
  iteration directly gives the same width-62 instance on all supported hosts.
- Observation: A two-file rollback cannot safely remove a path after a separate
  ownership check. The CLI now publishes the manifest last and leaves an orphan
  program on a second-file failure. An orphan has no completion marker and no
  final path is deleted during rollback.

## Decision Log

- Decision: Add a new installed `MQT::CoreBenchmarks` target instead of
  repurposing `MQT::CoreAlgorithms`. Rationale: the accepted design removes the
  legacy target and gives semantic benchmark instances a name that does not
  imply circuit construction. Date/Author: 2026-08-22 / Lukas Burgholzer and
  Codex.
- Decision: Keep the MLIR emitter internal to the build and wheel. Rationale:
  exporting it would make the installed ordinary-Core CMake package depend on
  LLVM and MLIR and would require moving the existing install aggregation.
  Date/Author: 2026-08-22 / Lukas Burgholzer and Codex.
- Decision: Ship only GHZ, Grover, and QPE as public benchmark families in the
  foundation. Standard and iterative QPE are methods of one family. Rationale:
  these three have compact analytic references and cover topology, an explicit
  problem instance, exact arithmetic, and dynamic control flow. Date/Author:
  2026-08-22 / Lukas Burgholzer and Codex.
- Decision: Require a reference for every registered benchmark. Rationale: a
  generator without an independent expected result cannot prove algorithmic
  correctness. Date/Author: 2026-08-22 / Lukas Burgholzer and Codex.
- Decision: Use concrete C++ option types and a fixed registry only at the JSON
  boundary. Rationale: this keeps direct APIs typed without a universal map,
  variant of all future algorithms, plugin hierarchy, or public JSON dependency.
  Date/Author: 2026-08-22 / Lukas Burgholzer and Codex.
- Decision: Represent a QPE phase as a reduced rational number of turns.
  Rationale: exact integer arithmetic can reduce controlled powers modulo one
  turn before conversion to `double`, avoiding overflow and loss of exact cases.
  Date/Author: 2026-08-22 / Lukas Burgholzer and Codex.
- Decision: Use the existing big-endian outcome convention and name logical
  output registers in every reference. Rationale: adapters can normalize backend
  layouts once without changing analytic formulas. Date/Author: 2026-08-22 /
  Lukas Burgholzer and Codex.
- Decision: Preserve Daniel's program work in #2135 and park non-foundation
  programs in the archived #2204 line. Rationale: the redesign narrows the first
  merge without discarding or reattributing his implementation and research.
  Date/Author: 2026-08-22 / Lukas Burgholzer and Codex.
- Decision: Identify a semantic case with the full lowercase SHA-256 of its
  canonical manifest payload, prefixed by a versioned domain separator.
  Rationale: the ID is stable across C++ and Python implementations and avoids
  short-hash collisions without adding an installed dependency. Date/Author:
  2026-08-22 / Lukas Burgholzer and Codex.
- Decision: Limit GHZ width and QPE precision to 1,000,000. Rationale: strict
  request parsing must reject tiny inputs that would otherwise request unbounded
  allocation, while retaining ample range for structured-program scaling.
  Date/Author: 2026-08-22 / Lukas Burgholzer and Codex.
- Decision: Keep DD execution changes in the stack that starts at #2077. Add the
  missing returned-CBit outcome adapter directly after #2077. Before simulation,
  use the existing target-mapping pass to replace QTensor values with static
  qubits. Keep the nested-loop unroll correction as a separate optimizer fix
  because QPE angle tables use `tensor.extract`, which #2077 does not interpret.
  Rationale: this reuses the compiler and the DD stack at their existing
  ownership boundaries and adds no parallel QTensor interpreter. Date/Author:
  2026-08-22 / Lukas Burgholzer and Codex.

## Outcomes & Retrospective

The foundation now provides one semantic owner for typed GHZ, Grover, and QPE
instances, analytic references, strict requests and manifests, structured MLIR
generation, a CLI, and Python bindings. The old CoreAlgorithms and DD evaluation
targets are gone. The installed C++ API remains free of MLIR and JSON types.

Daniel's exact #2135 commit objects remain below the signed merge. His emitters
and jeff path evolved in place. New architecture, reference, packaging, and
removal work is in separately authored commits. The archived #2204 head keeps
the remaining catalog available for later typed migrations.

The DD comparison prevented a parallel QTensor interpreter and duplicate
deallocation behavior. The returned-final-CBit sampler and the independent
semantic benchmark suite will live above #2077, not in the benchmark foundation.
No remote state has changed.

## Context and Orientation

Ordinary C++ libraries live below `include/mqt-core/` and `src/`. The new public
headers belong in `include/mqt-core/benchmarks/`, and their implementations
belong in `src/benchmarks/`. This library owns benchmark parameters, validation,
resolved defaults, output names and ordering, reference probabilities, scoring,
JSON requests, and deterministic manifests. It must not include LLVM, MLIR, or
the existing circuit IR.

The structured implementations from #2135 live in `mlir/benchmark/`. They use
`mlir::qc::QCProgramBuilder` to keep loops and classical control flow in the QC
dialect. That subtree becomes an internal emitter: it accepts validated ordinary
C++ benchmark instances and emits `mlir::QCProgram`. It does not choose defaults
or calculate a second reference.

The Python MLIR extension lives in `bindings/mlir/` and is installed as
`mqt.core.mlir`. The small benchmark binding module links only
`MQT::CoreBenchmarks`. Generation crosses into the existing MLIR extension as a
canonical request string so only one extension owns MLIR objects. The installed
pure C++ target stays free of MLIR. Generated `.pyi` files are produced with
`uvx nox -s stubs`; they are never edited by hand.

The old installed target is defined in `src/algorithms/CMakeLists.txt`, with
headers in `include/mqt-core/algorithms/` and tests in `test/algorithms/`.
`eval/` is an optional DD benchmark executable that also consumes it. The final
subtraction removes these components only after the new semantic and emitter
tests pass.

A reference is executable expected-result data. Each reference accepts a
big-endian logical outcome string and returns its ideal probability without
building a dense vector. It also evaluates a measured count map with total
variation distance and squared Hellinger fidelity. Total variation distance is
half the sum of absolute probability differences. Squared Hellinger fidelity is
the square of the sum of square roots of paired ideal and observed
probabilities. Grover also reports the observed probability of the marked
outcome.

## Plan of Work

First add `MQT::CoreBenchmarks`. Define shared `Evaluation` and output metadata,
then three concrete families. GHZ options contain the qubit count, linear or
star topology, and Z or X measurement basis. Grover options contain a required
marked bitstring, whose width is the number of search qubits, and either an
explicit iteration count or automatic selection. QPE options contain the
precision, a rational phase in turns, and the standard or iterative method.
Constructors validate and normalize all values. JSON parsing rejects missing
required values, unknown keys, wrong types, invalid enum strings, and invalid
numeric domains.

Implement references with formulas rather than a class hierarchy. GHZ in Z has
probability one half at all-zero and all-one. GHZ in X is uniform over even
parity outcomes. For Grover, with `N = 2^n`, `theta = asin(1/sqrt(N))`, and `t`
iterations, the marked outcome has `sin((2t+1)theta)^2`; every unmarked outcome
shares the residual mass. Automatic Grover iterations choose the nonnegative
integer that maximizes this expression near the usual closed-form estimate. For
QPE, reduce the phase to `[0,1)` and use the finite geometric-series or stable
sine-ratio form of the Dirichlet-kernel distribution, with exact integer checks
for deterministic cases.

Then replace the size-only MLIR callbacks with typed overloads for GHZ, Grover,
and QPE. Reuse Daniel's builder code where it is correct. GHZ becomes one
emitter with topology and basis branches. Grover removes the flag ancilla,
applies X around zero bits of the explicit marked string, uses a direct
multi-controlled phase oracle, and emits the resolved iteration count only after
checking the MLIR index range. QPE orders controlled powers consistently with
the inverse Fourier transform, computes every phase power with integer modular
arithmetic, and shares the same typed phase and reference between standard and
iterative methods. Avoid eager register element loads when structured indexed
loops are sufficient.

Replace the current registry with three fixed entries. A generic request has
`schema_version`, `benchmark`, and `parameters`. Parsing produces one typed
instance, and generation returns the program plus the manifest serialized from
that same instance. The manifest includes the benchmark definition version,
resolved parameters, named outputs, reference descriptor, and a case ID derived
from canonical semantic JSON. Emitter format and output path do not affect the
case ID.

Rename the executable to `mqt-core-bench`. It provides `list`, `describe`,
`generate`, and `evaluate`, but generation handles one request per invocation.
Output names contain the benchmark ID and case ID. Write a temporary sibling and
rename it only after successful generation. Existing files are rejected unless
the caller passes an explicit overwrite option. Remove the size-only `-n` and
all-program batch behavior that caused stale mixed corpora.

Add Python bindings with keyword-only option constructors. Python uses
`fractions.Fraction` for QPE input and renders the canonical reduced fraction as
`numerator/denominator` in JSON. Expose the generated QC program, manifest,
reference probability, and evaluation result. Keep generic dictionary or JSON
generation as a supplementary path; direct Python calls remain typed.

After all replacement tests pass, delete the legacy Algorithms target, headers,
implementations, tests, optional DD evaluation executable and documentation,
wheel dependency, and the redundant random-Clifford optimizer fixture. Do not
add deprecated wrappers. Update the changelog and upgrade guide with the removed
target and APIs and the new benchmark alternative where one exists.

## Concrete Steps

All commands run from the repository root.

Create and test the ordinary C++ model first:

    cmake --preset release
    cmake --build --preset release --target mqt-core-benchmarks-test
    ./build/release/test/benchmarks/mqt-core-benchmarks-test

Build and test structured emission next:

    cmake --build --preset release --target mqt-core-mlir-unittests-benchmark
    ./build/release/mlir/unittests/benchmark/mqt-core-mlir-unittests-benchmark

Build the CLI and exercise one request after its interface lands:

    cmake --build --preset release --target mqt-core-bench
    ./build/release/mlir/benchmark/mqt-core-bench generate --request test/benchmarks/requests/qpe-inexact.json --format jeff --output build/release/benchmarks

The command must print the generated case ID and paths. A second invocation
without the overwrite option must fail without changing either file.

After bindings land, regenerate stubs and run focused Python tests:

    uvx nox -s stubs
    uv run --no-sync pytest test/python/test_benchmarks.py

At each milestone inspect attribution and scope:

    git status --short
    git diff --check
    git log --format='%h %an <%ae> %s' codex/archive-pr-2135-before-benchmark-redesign..HEAD

At completion run the supported suites:

    cmake --build --preset release
    ctest --preset release
    uvx nox -s tests
    uvx nox -s minimums
    uvx nox --non-interactive -s docs
    uvx nox -s lint

If MLIR is unavailable in the ordinary release preset, configure the documented
LLVM/MLIR-enabled preset used by the worktree and record the exact substitution
and outcome in this plan.

## Validation and Acceptance

For GHZ, exact reference checks must accept only `00...0` and `11...1` in Z, and
every even-parity string with equal probability in X. DD samples of both linear
and star emitters must fit those references.

For Grover, reference probabilities must sum to one for explicit and automatic
iterations. A small DD sample must favor the requested marked bitstring, not
only all ones. The emitted program must have exactly the requested search-qubit
register and no flag ancilla.

For QPE, phases `3/16` at precision four and greater must be deterministic at
the corresponding outcome. An inexact phase such as `1/3` must match the
analytic distribution within a statistically justified finite-shot tolerance.
Precision 55 and larger must not produce an infinite or non-finite rotation.
Standard and iterative methods must expose the same reference and logical output
ordering.

The same typed request through C++, Python, and JSON must produce byte-identical
canonical manifest JSON and the same case ID. Repeated generation must be
byte-reproducible. Different semantic parameters must change the case ID;
changing only the output format must not.

The CLI must reject malformed JSON, unknown parameters, unsupported benchmark
IDs, invalid domains, existing output files, and partial generation without
leaving a final file. Installed C++ consumers must find `MQT::CoreBenchmarks`.
The Python wheel must import `mqt.core.benchmarks` without exposing LLVM types
in its option or reference objects.

After removal, active source and package metadata must have no reference to
`MQT::CoreAlgorithms`, `mqt-core-algorithms`, or the removed public headers.
Historical changelog and upgrade-guide text may retain those names.

## Idempotence and Recovery

Build, test, stub-generation, and documentation commands are repeatable. The
local refs `codex/archive-pr-2135-before-benchmark-redesign` and
`codex/archive-pr-2204-before-benchmark-redesign` preserve the original pull
request heads. Restore or compare any Daniel-authored file from those refs
instead of reconstructing it.

Do not use destructive reset or checkout commands. If a generated output test
fails, write only below `build/` and remove or overwrite the explicit test file,
never a source directory. Do not force-push a remote branch. Any later history
rewrite requires Daniel's explicit agreement, a fresh remote SHA, a backup ref,
and an exact `--force-with-lease`.

## Artifacts and Notes

The initial attribution boundary is:

    #2135 archive: 5e5580eb345249fd1123680994800c7dad85444d
    current main merged: 9c7a2d55a0da4bb577ec1b5bb919a2dc89abbeff
    signed merge: 00f3c8e5dc80cf1ff2829b42dbac6c82fef05088
    pre-restack foundation archive: 93f705bf196d73d23069fde49c39e1f4954131f1

`git verify-commit 00f3c8e5dc80cf1ff2829b42dbac6c82fef05088` reported a good
signature from the configured maintainer key.

The deterministic-builder check ran as:

    ./build/release/mlir/unittests/Dialect/QC/IR/mqt-core-mlir-unittest-qc-ir --gtest_filter='QCTest.BuilderDeallocatesDynamicResourcesDeterministically'
    [  PASSED  ] 1 test.

The semantic-reference check ran as:

    ./build/release/test/benchmarks/mqt-core-benchmarks-test
    [  PASSED  ] 20 tests.

After the strict JSON, case-ID, and portability fixes landed, the same test
binary reported:

    [  PASSED  ] 30 tests.

The returned-register sampler check ran as:

    ./build/release/mlir/unittests/Dialect/QCO/Utils/mqt-core-mlir-unittest-qco-utils --gtest_filter='QCODDFunctionalityTest.Sample*'
    [  PASSED  ] 8 tests.

The complete configured C++ test suite completed all 3,956 tests. One device
query test reported its expected skip. The ordinary Python test sessions passed
on Python 3.10 through 3.14 with 684, 723, 723, 723, and 734 passing tests. The
minimum-dependency sessions passed on the same versions with 495, 534, 534, 534,
and 545 passing tests. The documentation build passed with warnings treated as
errors after generating the MLIR reference pages.

A fresh wheel installed in an isolated environment and ran `mqt-core-bench`
through the packaged Python launcher. A fresh native Runtime install also ran
the CLI. Both layouts found the shared benchmark library through the installed
relative run path.

## Interfaces and Dependencies

The installed C++ target is `MQT::CoreBenchmarks`. Its public headers use only
C++20 standard-library types. Public option and instance types live in namespace
`mqt::benchmarks`. JSON implementation uses the existing private
`nlohmann_json::nlohmann_json` dependency.

The required public concepts are concrete GHZ, Grover, and QPE options and
instances; a named output description; `probability(std::string_view)`;
`evaluate(const std::map<std::string, size_t>&)`; canonical request and manifest
serialization; and strict parsing. No public base class, plugin interface,
`std::any`, universal option map, or variant containing every algorithm is
added.

The internal MLIR target links `MQT::CoreBenchmarks`, the compiler pipeline, and
`MLIRQCProgramBuilder`. Its typed `generateProgram` overloads accept the three
materialized instance types. Only its registry adapter accepts JSON.

The only new external behavior uses dependencies already present in MQT Core:
the C++ standard library, nlohmann JSON, LLVM/MLIR, nanobind, GoogleTest, and
the existing DD package. No new dependency is added.

Revision note, 2026-08-22: Created the initial self-contained plan after Daniel
authorized continued implementation. Recorded the preserved attribution
boundary, accepted API decisions, and current-main merge.

Revision note, 2026-08-22: Recorded the shared deterministic-cleanup fix and its
focused passing test.

Revision note, 2026-08-22: Recorded the completed typed semantic and analytic
reference layer, including the arbitrary-width QPE check.

Revision note, 2026-08-22: Recorded strict JSON storage, self-checking
manifests, JSON Schema descriptions, and stable semantic case IDs.

Revision note, 2026-08-22: Recorded typed structured emitters, the configured
CLI, and the single-MLIR-owner Python binding design.

Revision note, 2026-08-22: Recorded the #2077 ownership boundary, completed
implementation and removal, full validation, independent audits, and the final
portable-reference and no-rollback fixes.
