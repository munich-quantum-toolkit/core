# Complete the structured-benchmark foundation

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core needs structured quantum workloads whose inputs, expected outputs, and
generated programs describe the same exact instance. After this work, C++,
Python, and `mqt-core-bench` users can configure GHZ, Grover, QPE,
Bernstein--Vazirani (BV), and QFT with family-specific parameters. Each instance
has an analytic reference, canonical JSON instance, self-checking manifest,
stable case ID, and structured QC or `jeff` program.

The foundation is also the extension path for Daniel Haag's remaining structured
programs. It keeps his original commits in the branch history and does not copy
simulator support from the pull-request stack that starts at
[#2077](https://github.com/munich-quantum-toolkit/core/pull/2077).

### Progress

- [x] (2026-08-22 09:24Z) Preserved Daniel's commits and merged main without
      rebasing his work.
- [x] (2026-08-23 10:58Z) Extracted and merged the independent dependency,
      deterministic-builder, legacy-library removal, and loop-unroll changes.
- [x] (2026-08-23 15:35Z) Merged current main and removed this branch's
      changelog delta.
- [x] (2026-08-23 16:20Z) Renamed the public component to singular `bench` and
      added typed BV and QFT semantics, schemas, manifests, references, and
      private registry entries.
- [x] (2026-08-23 16:55Z) Added structured BV and QFT emitters, removed
      allocation-adjacent resets and QPE swaps, combined Fourier emission, and
      switched CLI and MLIR Python generation to generic private dispatch.
- [x] (2026-08-23 17:10Z) Added singular Python bindings and executable notebook
      documentation. Focused C++, emitter, `jeff`, CLI, and Python checks pass.
- [x] (2026-08-23 23:20Z) Rebuilt the semantic-test branch above #2077 and added
      BV and QFT distribution checks without copying simulator code into this
      branch.
- [x] (2026-08-23 23:35Z) Ran installation, wheel, notebook, full release, and
      lint validation and completed the stale-name audit.
- [x] (2026-08-27 16:37Z) Merged current main and reread the expanded root,
      MLIR, and binding guidance.
- [x] (2026-08-27 17:20Z) Addressed Damian Rovara's documentation feedback with
      a clearer benchmark definition, direct notebook CLI cells, and method
      documentation. Regenerated stubs; the full release build, all 3,899 native
      tests, the warning-as-error documentation build, and repository lint pass.

### Surprises & Discoveries

- Observation: The current `jeff-mlir` pin already lowers both BV methods and
  both QFT methods. Evidence: focused generation produced nonempty `jeff` files
  for all four cases. The draft pull request #44 is not needed.
- Observation: Loading MLIR objects from two Python extensions creates
  incompatible process-local MLIR type IDs. Evidence: the semantic extension
  passes canonical instance JSON to the existing `mqt.core.mlir` extension, so
  one extension owns every returned `QCProgram`.
- Observation: Exact QPE powers stay finite above 1,024 bits when rational turns
  are reduced before conversion to `double`. Evidence: the precision-1,025
  emitter test keeps the module below 150 operations.
- Observation: The stack that starts at #2077 already owns structured DD
  execution. The benchmark branch therefore owns instances, emission, and
  references only.
- Observation: The combined DD check exposed a shared QFT ordering error in both
  methods. Evidence: the old emitters gave total variation distance above 0.28
  for QFT `(3, 1)`; descending physical targets and matching feed-forward
  reduced it below 0.05 for both methods.
- Observation: The full Python matrix reaches all 570 or 581 tests per version,
  but six unrelated QDMI tests fail because the test installs do not register
  the built-in `mqt.sc.default` and `mqt.sc.iqm.garnet` devices. The focused
  benchmark and CLI tests pass for Python 3.11 through 3.14 with minimum direct
  dependencies.

### Decision Log

- Decision: Use the singular public names `mqt::bench`, `mqt.core.bench`,
  `mqt-core/bench/`, and `MQT::CoreBench`. Rationale: the CLI and component name
  use `bench`, and the old public API is unreleased. Date/Author: 2026-08-23 /
  Lukas Burgholzer and Codex.
- Decision: Ship typed GHZ, Grover, QPE, BV, and QFT families. Standard and
  dynamic or iterative circuit forms are methods of one semantic family.
  Rationale: method changes must not duplicate the instance reference.
  Date/Author: 2026-08-23 / Lukas Burgholzer and Codex.
- Decision: Keep one private semantic registry and one private MLIR registry.
  Rationale: generic JSON dispatch needs a fixed ID table, while public C++ and
  Python remain typed and do not expose a base class, variant, or option map.
  Date/Author: 2026-08-23 / Lukas Burgholzer and Codex.
- Decision: Install only `MQT::CoreBench`. Expose typed generation through the
  source-build target `MQT::CoreBenchGenerate`. Rationale: the wider MLIR C++
  package is not installed yet. Date/Author: 2026-08-23 / Lukas Burgholzer and
  Codex.
- Decision: Qubit allocation establishes the initial zero state. Emitters reset
  only qubits that they reuse after measurement. Rationale: allocation-adjacent
  resets add work without changing the program. Date/Author: 2026-08-23 / Lukas
  Burgholzer and Codex.
- Decision: Mirror QPE inverse-QFT operands and QFT result indices instead of
  emitting final swaps. Rationale: result mapping is cheaper than quantum swaps.
  Date/Author: 2026-08-23 / Lukas Burgholzer and Codex.
- Decision: Never overwrite CLI outputs. Rationale: case-ID paths are stable;
  atomic no-clobber publication and a manifest-last completion marker are
  enough. Date/Author: 2026-08-23 / Lukas Burgholzer and Codex.
- Decision: Keep DD execution changes above #2077. Rationale: the simulator owns
  execution and returned-register sampling. Date/Author: 2026-08-22 / Lukas
  Burgholzer and Codex.
- Decision: Do not add a benchmark option that expands registers into separate
  scalar allocations. Rationale: storage representation does not change the
  benchmark instance or reference, indexed loops require an indexable value, and
  scalar expansion would make large structured programs linear in the qubit
  count. A reusable compiler transformation is the correct owner if a concrete
  consumer needs this representation. Date/Author: 2026-08-27 / Lukas Burgholzer
  and Codex.

### Outcomes & Retrospective

The implementation has one semantic owner and one structured emitter path for
five families. After the current-main merge, the complete release build and all
3,899 configured native tests pass. The warning-as-error documentation build,
including the executable notebook, also passes. The installed C++ consumer,
fresh wheel, launcher, and minimum-version Python benchmark tests pass. The
separate semantic branch validates every family against the #2077-based DD
stack. The full Python matrix is blocked only by the unrelated missing QDMI
device registrations recorded above.

### Context and Orientation

Public C++ types live under `include/mqt-core/bench/`; their implementations
live under `src/bench/`. These files own option validation, resolved defaults,
logical output names, analytic probabilities, count evaluation, canonical JSON,
manifests, and case IDs. `src/bench/JSON.cpp` contains the private semantic
registry: one entry per ID with a definition version, schema callback, and
evaluation callback.

Structured emitters live in `mlir/benchmark/programs/`. `Fourier.cpp` owns QPE,
iterative QPE, standard QFT, and semiclassical QFT so their phase-loop rules and
bit order stay together. `mlir/benchmark/Generate.cpp` contains the private MLIR
instance registry and the typed `generate(...)` overloads declared in
`mlir/include/mlir/Bench/Generate.h`.

`mlir/benchmark/MQTCoreBench.cpp` implements the `list`, `describe`, `generate`,
and `evaluate` commands. Generation stages both outputs, atomically publishes
without replacing existing paths, and publishes the manifest last.

Python types live in `bindings/bench/register_bench.cpp`. Their `generate()`
methods pass canonical instance JSON to `_generate_benchmark` in
`bindings/mlir/register_mlir.cpp`. This boundary keeps MLIR objects in the one
extension that defines them. Generated declarations live in
`python/mqt/core/bench.pyi`.

All references use big-endian strings for the returned `result` register. BV is
a point distribution at the hidden bitstring. QFT with period exponent `k`
assigns probability `2^-k` to outcomes whose final `n-k` bits are zero. GHZ,
Grover, and QPE retain their analytic state, search, and phase-estimation
references.

### Plan of Work

Finish the singular rename without compatibility aliases. Keep semantic JSON
dispatch private, strict, and fixed to the same five IDs as MLIR generation.
Expose each family through typed C++ and explicit Python classes.

Emit BV with one query qubit per hidden bit in static mode and one reused query
qubit in dynamic mode. Emit QFT from the periodic input with a full register in
standard mode and one measured, reset, and reused qubit in semiclassical mode.
Keep loop bodies structured. Do not add allocation/reset canonicalization or a
new helper collection.

Make the notebook discover the catalog through the CLI and execute typed
configuration, instance and manifest inspection, reference evaluation, IR
generation, and a temporary-directory CLI round trip. Document the five explicit
extension points for later families.

After the feature branch passes independently, apply only the semantic-test
commit above #2077 and its returned-register sampling follow-up. Add small BV
and QFT distribution cases there. Do not move DD interpreter code into this
branch.

### Concrete Steps

Run focused semantic and emitter checks from the repository root:

    cmake --preset release
    cmake --build --preset release --target mqt-core-bench-test mqt-core-mlir-unittests-benchmark mqt-core-bench
    ./build/release/test/bench/mqt-core-bench-test
    ./build/release/mlir/unittests/benchmark/mqt-core-mlir-unittests-benchmark
    ctest --test-dir build/release -R '^mqt-core-mlir-benchmark-cli$' --output-on-failure

Regenerate and test Python:

    uv sync --inexact --no-dev --no-build-isolation-package mqt-core
    uvx nox -s stubs
    uv run --no-sync pytest test/python/test_bench.py test/python/test_cli.py

Build the executable notebook and package surfaces:

    uvx nox --non-interactive -s docs
    cmake --build --preset release
    ctest --preset release
    uvx nox -s tests
    uvx nox -s minimums
    uvx nox -s lint

Finish with `git diff --check`, `git status --short`, an installed C++ consumer
of `MQT::CoreBench`, a source-build consumer of `MQT::CoreBenchGenerate`, a
fresh wheel/launcher smoke test, and a search for stale plural public names.

### Validation and Acceptance

All five families must round-trip through typed C++, strict JSON, Python, and
the CLI with one case ID per resolved semantic instance. BV `101` must be
deterministic in both methods. QFT `(3, 1)` must have peaks `000` and `100`, and
QFT `(4, 2)` must have four quarter-probability peaks.

Emitter tests must show that allocation-adjacent resets are absent and reuse
resets remain. Standard QPE and QFT must contain no `qc.swap`. BV must use
structured secret lookup, direct big-endian result indexing, and the specified
static or dynamic resources. Large QPE and QFT programs must stay compact and
finite. Every method must produce valid QC and `jeff` with the existing
dependency pin.

The CLI must reject malformed instances, unknown IDs, invalid formats, and any
output collision without changing an existing file. The executed notebook must
discover all five IDs and complete its CLI round trip. Python must import only
`mqt.core.bench` and generated programs must support `.ir` and `.to_qco()`.

The combined #2077 stack must sample small BV and QFT programs according to the
same references. Those simulator-dependent test commits must remain outside this
feature branch.

### Idempotence and Recovery

Build, test, stub, and documentation commands are repeatable. CLI tests write
below `build/`, and the notebook uses a temporary directory. Existing generated
files are never replaced. Restore Daniel-authored code from its reachable Git
commits rather than reconstructing it. Do not rewrite or push remote history
without a backup ref, a current remote SHA, an exact force-with-lease, and
explicit human authorization.

### Interfaces and Dependencies

The installed semantic target is `MQT::CoreBench`. Public family and option
types use namespace `mqt::bench` and C++20 standard-library types. nlohmann JSON
and the compact SHA-256 implementation remain private.

The source build exposes `MQT::CoreBenchGenerate` and these typed overloads:

    std::optional<mlir::QCProgram> generate(const BV&);
    std::optional<mlir::QCProgram> generate(const GHZ&);
    std::optional<mlir::QCProgram> generate(const Grover&);
    std::optional<mlir::QCProgram> generate(const QFT&);
    std::optional<mlir::QCProgram> generate(const QPE&);

It also has one generic instance overload for the CLI and Python bridge. No
public base class, plugin interface, all-family variant, generic option map,
size-only callback, compatibility alias, or new dependency is part of this
foundation.
