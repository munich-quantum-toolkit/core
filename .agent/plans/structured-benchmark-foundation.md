# Build a typed structured-benchmark foundation

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core needs benchmark instances that record more than a qubit count. After
this work, C++ and Python users can configure GHZ, Grover, and quantum phase
estimation (QPE) with family-specific types. The `mqt-core-bench` command reads
the same strict JSON requests, emits structured QC or `jeff`, writes a manifest
for the exact instance, and scores measured counts against an analytic
reference.

The foundation deliberately contains only these three families. It does not
retain the earlier size-only callback catalog or publish a generic MLIR program
registry. Daniel Haag authored the original structured emitters and `jeff`
generator work. Moves and adaptations preserve that work and its Git history.
New design and integration work keeps its own attribution.

## Progress

- [x] (2026-08-22 09:24Z) Preserved Daniel's commits and merged the then-current
      main branch without rebasing his work.
- [x] (2026-08-22 10:06Z) Added typed GHZ, Grover, and QPE instances, analytic
      references, strict JSON requests and manifests, and stable case IDs.
- [x] (2026-08-22 11:30Z) Added typed structured emitters, the request-based
      CLI, Python bindings, generated stubs, documentation, and tests.
- [x] (2026-08-22 11:44Z) Prepared the `MQT::CoreAlgorithms` and legacy DD
      evaluation removal after replacement tests passed.
- [x] (2026-08-22 12:08Z) Completed C++, Python, packaging, documentation, and
      lint validation for the first complete foundation.
- [x] (2026-08-23 10:47Z) Removed the remaining size-only callback registry,
      QFT, IQFT, multiplexer, teleportation, GHZ-star wrapper, public emitter
      header, broad pipeline tests, and unused uniform-rotation helper.
- [x] (2026-08-23 10:47Z) Reverted unrelated ResetOp and empty-modifier changes,
      kept the required `jeff-mlir` pin, and restored the scoped Cap'n Proto
      `BUILD_TESTING` guard.
- [x] (2026-08-23 10:47Z) Rebuilt the narrow MLIR benchmark and CLI targets;
      five typed emitter tests, the CLI test, and the deterministic-builder test
      pass.
- [x] (2026-08-23 10:58Z) Extracted the `jeff-mlir` pin, deterministic builder
      finalization, CoreAlgorithms removal, and nested-loop unroll fix into pull
      requests #2212, #2213, #2214, and #2216.
- [x] (2026-08-23 11:10Z) Re-ran the full lint suite, all 4,068 configured C++
      tests, 16 focused Python tests, the bindings-enabled wheel aggregate, and
      the documentation build after pruning.

## Surprises & Discoveries

- Observation: Loading MLIR objects from two Python extensions creates
  incompatible process-local MLIR type IDs. Evidence: benchmark generation now
  passes canonical request JSON into the existing `mqt.core.mlir` extension, so
  one extension owns every returned `QCProgram`.
- Observation: QPE controlled angles can remain finite and structured above
  1,024 bits by reducing the rational phase modulo one turn before conversion to
  `double`. Evidence: the focused emitter test passes at precision 1,025 and
  keeps the module below 150 operations.
- Observation: The DD simulator work that starts at pull request #2077 already
  owns structured execution. Duplicating that interpreter here would couple the
  generator to one backend. The returned-register sampler and semantic DD tests
  therefore remain in that stack.
- Observation: The original MLIR catalog survived the typed redesign only as a
  parallel size-only API. Removing it also removes six source files, two broad
  tests, a public header, and an unused angle-loop variant without reducing the
  typed API.
- Observation: `arith::ConstantIndexOp` and `arith::ConstantOp` already create
  the constants needed by the emitters. Public `QCProgramBuilder` convenience
  methods were unnecessary.

## Decision Log

- Decision: Install `MQT::CoreBenchmarks` as an MLIR-free C++ library.
  Rationale: parameters, references, JSON, and evaluation are useful without a
  compiler dependency. Date/Author: 2026-08-22 / Lukas Burgholzer and Codex.
- Decision: Ship only typed GHZ, Grover, and QPE in the foundation. Standard and
  iterative QPE are methods of one family. Rationale: each family has a compact
  independent reference and together they exercise topology, search instances,
  exact phase input, loops, and classical control. Date/Author: 2026-08-22 /
  Lukas Burgholzer and Codex.
- Decision: Use concrete family option types and dispatch on a fixed benchmark
  ID only at the JSON boundary. Rationale: this avoids a public option map,
  plugin hierarchy, or variant that must change for every future family.
  Date/Author: 2026-08-22 / Lukas Burgholzer and Codex.
- Decision: Keep structured emitters internal and expose only typed
  `generateProgram` overloads within the MLIR build. Rationale: ordinary Core
  consumers must not acquire an LLVM or MLIR dependency. Date/Author: 2026-08-22
  / Lukas Burgholzer and Codex.
- Decision: Require an analytic reference and named logical output for every
  family. Rationale: a generator cannot establish algorithmic correctness by
  comparing against itself. Date/Author: 2026-08-22 / Lukas Burgholzer and
  Codex.
- Decision: Remove the size-only MLIR registry and unmigrated programs instead
  of maintaining two configuration models. Rationale: the old callbacks bypass
  typed parameters, manifests, and references and were no longer used by the CLI
  or Python API. Date/Author: 2026-08-23 / Lukas Burgholzer and Codex.
- Decision: Keep DD execution changes in the stack that begins at #2077.
  Rationale: the compiler and simulator own execution; the benchmark foundation
  owns instances, emission, and references. Date/Author: 2026-08-22 / Lukas
  Burgholzer and Codex.

## Outcomes & Retrospective

The foundation has one semantic owner for typed GHZ, Grover, and QPE instances,
references, strict requests, manifests, structured generation, CLI behavior, and
Python bindings. The installed API stays free of MLIR and JSON types. The
internal emitter now contains only code reached by these three families.

The pruning removed iteration artifacts without discarding Daniel's useful work.
Pull request #2214 owns the separate CoreAlgorithms removal. Simulator
additions, returned-register sampling, and semantic DD checks remain outside
this branch.

## Context and Orientation

Public C++ benchmark headers live in `include/mqt-core/benchmarks/`; their
implementations live in `src/benchmarks/`. These files own option validation,
resolved defaults, output names, analytic probabilities, count evaluation,
canonical JSON, manifests, and case IDs. A case ID is the SHA-256 digest of a
versioned canonical semantic payload, so output paths and formats do not affect
it.

Structured emitters live in `mlir/benchmark/programs/`. The private `Programs.h`
declares only configured GHZ, Grover, and QPE emitters.
`mlir/benchmark/Generate.cpp` creates a compiler context, calls one emitter, and
cleans up the resulting QC program. `mlir/benchmark/MQTCoreBench.cpp` implements
`list`, `describe`, `generate`, and `evaluate`.

Python family bindings live in `bindings/benchmarks/`. The binding passes a
canonical request string to the private `_generate_benchmark` function in
`bindings/mlir/register_mlir.cpp`; this keeps one Python extension responsible
for MLIR objects. Generated stubs live in `python/mqt/core/benchmarks.pyi` and
must be regenerated, not edited by hand.

A reference returns the ideal probability of a big-endian logical bitstring. GHZ
in Z basis assigns one half to all-zero and all-one outcomes. GHZ in X basis is
uniform over even-parity outcomes. Grover uses the usual two-dimensional
rotation formula for the marked and unmarked probabilities. QPE uses the finite
geometric-series distribution, with exact rational arithmetic for deterministic
cases. Evaluation reports total variation distance and squared Hellinger
fidelity; Grover also reports marked-outcome probability.

## Plan of Work

Maintain the typed C++ layer first. Each constructor validates its complete
family-specific options. Strict JSON parsing rejects missing fields, unknown
fields, wrong types, invalid enum values, and out-of-domain integers. A manifest
records the family version, resolved parameters, named outputs, reference
descriptor, and semantic case ID.

Keep the MLIR layer as a small adapter. GHZ emits linear or star entanglement
and optional X-basis measurement. Grover marks the explicit big-endian bitstring
with a direct multi-controlled phase oracle and no flag ancilla. QPE shares
exact phase-power preparation between standard and iterative methods and keeps
controlled powers aligned with result-bit order.

Keep CLI and Python behavior driven by the same canonical requests. Generation
writes the program first and the manifest last; the manifest is the completion
marker. Existing final files require an explicit overwrite option. Do not add a
batch API, size-only compatibility layer, public emitter registry, or second
MLIR-owning Python extension.

## Concrete Steps

Run these commands from the repository root. Build and test the ordinary model:

    cmake --preset release
    cmake --build --preset release --target mqt-core-benchmarks-test
    ./build/release/test/benchmarks/mqt-core-benchmarks-test

Build and test the internal emitter and CLI:

    cmake --build --preset release --target mqt-core-mlir-unittests-benchmark mqt-core-bench
    ./build/release/mlir/unittests/benchmark/mqt-core-mlir-unittests-benchmark
    ctest --test-dir build/release -R '^mqt-core-mlir-benchmark-cli$' --output-on-failure

Regenerate stubs after binding changes and test Python:

    uvx nox -s stubs
    uv run --no-sync pytest test/python/test_benchmarks.py test/python/test_cli.py

At completion run the release tests appropriate to the final split, build the
documentation, and finish with:

    uvx nox -s lint

Always inspect `git status --short`, `git diff --check`, and the author fields
of Daniel's retained commits before publication.

## Validation and Acceptance

All three families must round-trip through typed C++, typed Python, strict JSON,
and the CLI with the same canonical manifest and case ID. Different semantic
parameters must change the ID; changing only output format must not.

The emitter tests must prove that GHZ does not eagerly load large registers,
Grover marks the requested bitstring without a flag ancilla, standard QPE keeps
controlled powers and result order aligned, both QPE methods stay finite and
structured at precision 1,025, and modular phase doubling does not overflow.

The CLI must reject malformed requests, unsupported IDs, invalid formats, and
output collisions without changing an existing final file. Repeated generation
with overwrite must be byte reproducible. Installed C++ consumers must find
`MQT::CoreBenchmarks`, and the wheel must import `mqt.core.benchmarks` without
exposing LLVM types in family options or references.

Active source and package metadata must not refer to `MQT::CoreAlgorithms`,
`mqt-core-algorithms`, the removed public MLIR `Programs.h`, or the size-only
benchmark callbacks. Historical changelog and upgrade text may retain removed
names.

## Idempotence and Recovery

Build, test, stub-generation, and documentation commands are repeatable. Tests
write only below `build/`. Restore Daniel-authored code from its preserved Git
commits rather than reconstructing it. Do not rewrite or force-push a remote
branch without a fresh remote SHA, a backup ref, an exact force-with-lease, and
explicit human authorization.

## Artifacts and Notes

The pruned foundation passes all 4,068 configured C++ tests, 16 focused Python
tests, the bindings-enabled wheel aggregate, the documentation build, and the
full lint suite. The focused results include 30 reference tests, five typed
emitter tests, the CLI scenario, and deterministic builder finalization.

## Interfaces and Dependencies

The installed target is `MQT::CoreBenchmarks`. Public family and option types
live in namespace `mqt::benchmarks` and use C++20 standard-library types. JSON
uses the existing private nlohmann JSON dependency.

The internal `MQTBenchmarkGenerate` target exposes three overloads:

    std::optional<mlir::QCProgram> generateProgram(const benchmarks::GHZ&);
    std::optional<mlir::QCProgram> generateProgram(const benchmarks::Grover&);
    std::optional<mlir::QCProgram> generateProgram(const benchmarks::QPE&);

No public base class, plugin interface, universal option map, algorithm variant,
size-only callback, or new third-party dependency is part of the foundation.

Revision note, 2026-08-23: Updated the completed plan after the Ponytail audit.
Removed the parallel size-only catalog and documented the narrow typed emitter,
attribution boundary, simulator boundary, and post-pruning evidence.
