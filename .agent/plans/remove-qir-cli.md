# Remove the standalone QIR runner and internalize QIR execution

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core must stop presenting its LLVM-based Quantum Intermediate Representation
(QIR) executor as a standalone product. After this change, an installation
contains no `mqt-core-qir-runner` command and no public QIR runtime or
just-in-time compiler headers. The build tree keeps the `MQT::CoreQIRJIT` and
`MQT::CoreQIRRuntime` aliases for consistent internal linking. The DDSIM QDMI
device still accepts and executes supported QIR assembly and bitcode jobs. Its
focused tests prove that every job retains independent simulator, runtime,
output, and random number generator state.

## Progress

- [x] (2026-08-25 01:17Z) Read issue #2096, repository policy, and the QIR
      CMake, source, test, and documentation surfaces.
- [x] (2026-08-25 01:20Z) Moved the retained JIT and runtime headers out of the
      public include tree and kept the libraries out of the installed package.
- [x] (2026-08-25 01:20Z) Deleted the native runner implementation, CMake
      target, and subprocess test.
- [x] (2026-08-25 01:21Z) Removed runner documentation.
- [x] (2026-08-25 01:28Z) Configured and built the focused QIR and DDSIM
      targets, ran their tests, checked the install tree, ran documentation
      checks as practical, and ran `uvx nox -s lint`.
- [x] (2026-08-25 01:29Z) Inspected the final diff and working tree and recorded
      the outcome and validation evidence here.
- [x] (2026-08-26 05:59Z) Reviewed the change against issue #1734 and removed
      file loading, named entry-point selection, and other runner-only JIT
      surface.
- [x] (2026-08-26 05:59Z) Exposed deterministic OpenQASM and QIR sampling as
      DDSIM QDMI custom job parameter 1.
- [x] (2026-08-26 06:15Z) Built the full release preset, ran the revised focused
      and full tests, built the documentation, checked the install surface, and
      ran lint.
- [x] (2026-08-26 06:15Z) Prepared the validated, authorized update to pull
      request #2246.
- [x] (2026-08-26 06:43Z) Moved the retained executor into the MLIR QIR subtree,
      restored its build-tree aliases, and removed cross-tree include paths.
- [x] (2026-08-26 06:50Z) Ran final focused and full validation and prepared the
      review update for publication.

## Surprises & Discoveries

- Observation: The JIT and runtime libraries are not members of
  `MQT_CORE_TARGETS`, so the installed CMake package does not export them. The
  build tree can retain namespaced aliases without restoring an installed API.
- Observation: The runner test is only a subprocess smoke test. JIT session
  tests and DDSIM device tests already cover execution, repeated runs,
  ownership, concurrent jobs, and result isolation without requiring a
  production executable. Evidence: `test/qir/runner/test_qir_runner.cpp`,
  `test/qir/jit/test_jit_session.cpp`, and
  `test/qdmi/devices/dd/concurrency_test.cpp`.
- Observation: The compiler and executor repeated QIR attribute, profile,
  schema, and irreversible-operation literals. Both now use
  `mlir/Dialect/QIR/QIRDefinitions.h` without a cross-tree include directory.
- Observation: Release configuration, package installation, and documentation
  need the local LLVM/MLIR package path on this host. Evidence: all three pass
  with `MLIR_DIR=/home/nvidia/.local/opt/mqt-llvm-mlir/22.1.7/lib/cmake/mlir`.
- Observation: The full release build installs to a temporary prefix, although
  bundled Cap'n Proto reports that it cannot create its unrelated
  `/usr/local/bin/capnpc` link. The install-tree scan finds no native QIR
  runner, executor headers, or QIR package targets.
- Observation: Two of issue #1734's three intended outcomes are now covered.
  Base Profile IR and pipeline tests exist, and QIR-Runner provides external
  interoperability checks. The Base Profile conversion still contains its own
  `ConvertMemRefLoadOp`; centralizing dynamic-to-static qubit conversion remains
  separate compiler work.

## Decision Log

- Decision: Keep `MQT::CoreQIRJIT` and `MQT::CoreQIRRuntime` as build-tree
  aliases for the unexported static targets. Rationale: Namespaced aliases match
  other internal links without adding the libraries to the installed package.
  Date/Author: 2026-08-26 / Codex.
- Decision: Place the executor below `mlir/Dialect/QIR/Execution` and expose its
  headers only through build-interface usage requirements. Rationale: The MLIR
  QIR compiler and DDSIM executor now have one owner and no cross-tree include
  dependency. Date/Author: 2026-08-26 / Codex.
- Decision: Remove the subprocess runner test instead of converting it.
  Rationale: The issue explicitly removes subprocess tests, while in-process JIT
  and DDSIM tests own the retained behavioral contracts. Date/Author: 2026-08-25
  / Codex.
- Decision: Store shared QIR attribute, profile, and schema literals in
  `mlir/Dialect/QIR/QIRDefinitions.h`. Rationale: Compiler generation and
  internal execution must agree on these spellings. The header is not part of an
  installed file set. Date/Author: 2026-08-26 / Codex.
- Decision: Keep deterministic seeding as DDSIM QDMI custom job parameter 1
  instead of a JIT session option. Rationale: The job owns sampling policy and
  can apply one seed contract to OpenQASM and QIR. Date/Author: 2026-08-26 /
  Codex.
- Decision: Require exactly one standard QIR entry point and remove named
  selection. Rationale: MQT's compiler emits one entry point, while QIR-Runner
  already serves external modules that need selection. Date/Author: 2026-08-26 /
  Codex.

## Outcomes & Retrospective

The native QIR runner, its subprocess test, its user guide, and all public QIR
runtime/JIT headers are removed. The retained implementation lives below the
MLIR QIR subtree and is shared by DDSIM, the compiler, and focused tests. Its
unexported targets retain consistent build-tree aliases. DDSIM exposes
deterministic OpenQASM and QIR sampling through custom job parameter 1. The
internal JIT accepts only in-memory modules with one standard QIR entry point.
The focused tests, all 4,035 configured C++ tests, documentation build,
install-surface scan, and lint pass.

## Context and Orientation

QIR is an LLVM-based interchange format for quantum programs. MQT Core compiles
programs to QIR and executes QIR inside the decision diagram simulator device in
`src/qdmi/devices/dd/Device.cpp`. The executor lives below
`mlir/lib/Dialect/QIR/Execution`. Its runtime implements quantum instructions
and QIR resource and output functions. Its JIT loads LLVM modules, validates
declarations, binds runtime symbols, and invokes an entry point. Focused tests
live under `test/qir`; DDSIM integration tests live under
`test/qdmi/devices/dd`.

Installed targets are controlled through `MQT_CORE_TARGETS` in
`src/CMakeLists.txt`. The QIR libraries are absent from that list. Plain static
targets with manually defined aliases can provide consistent build-tree links
without exporting the libraries or their headers.

The change is limited to QIR runtime/JIT/runner build files and headers, their
direct consumers and tests, the QIR user guide, `UPGRADING.md`, and this plan.
It must preserve unrelated worktree changes. It must follow `AGENTS.md` and
`docs/ai_usage.md`. The maintainer authorized a fast-forward update to pull
request #2246 after local validation and signed-commit verification.

## Plan of Work

First, place the retained executor below `mlir/lib/Dialect/QIR/Execution` and
its headers below `mlir/include/mlir/Dialect/QIR/Execution`. Use ordinary static
libraries, set position-independent code directly, attach project warning and
option targets, and publish only the MLIR source include directory to build-tree
dependents. Keep the `MQT::CoreQIRJIT` and `MQT::CoreQIRRuntime` build-tree
aliases. Do not add either raw target to `MQT_CORE_TARGETS`.

Second, remove the standalone runner and its subprocess test. Confirm that no
source, CMake, or documentation reference to the runner remains.

Third, revise `docs/qir/index.md`. Remove the native runner build and use
sections. Retain compiler output documentation, the external QIR Alliance runner
example, and the DDSIM QDMI execution contract. Make clear that MQT Core's
retained runtime and JIT are internal to DDSIM. Fold the noteworthy change into
the existing QIR changelog entry. Do not add migration text for unreleased v4
functionality to `UPGRADING.md`.

Fourth, reduce the retained JIT to the in-memory DDSIM contract. Remove file
loading, named entry-point selection, session-owned seeding, and unused
accessors. Require one function with the standard `entry_point` attribute.
Define DDSIM custom job parameter 1 as an optional positive `int` seed and use
it for both OpenQASM and QIR sampling. Document the contract in
`docs/qdmi/ddsim_device.md` and the existing aggregate QIR changelog entry.

Finally, configure the release preset, build the focused QIR JIT/runtime and
DDSIM test targets, run those binaries, and inspect an install tree or generated
target file to prove that no QIR executable, headers, or package target remains.
Build documentation if the environment permits it. Run `uvx nox -s lint`,
inspect the diff and status, and record exact outcomes in this plan.

## Concrete Steps

Run all commands from the repository root.

Move files with `git mv`, then edit CMake and documentation with focused
patches. Search the result with:

    rg -n "mqt-core-qir-runner|src/qir|include/mqt-core/qir" \
      CMakeLists.txt cmake src test include docs UPGRADING.md

The search must return no matches.

Configure and build with:

    cmake --preset release
    cmake --build --preset release --target mqt-core-qir-jit-test mqt-core-qir-runtime-test mqt-core-qdmi-ddsim-device-test

Run the focused binaries with:

    ./build/release/test/qir/jit/mqt-core-qir-jit-test
    ./build/release/test/qir/runtime/mqt-core-qir-runtime-test
    ./build/release/test/qdmi/devices/dd/mqt-core-qdmi-ddsim-device-test

Run documentation and lint checks with:

    uvx nox --non-interactive -s docs
    uvx nox -s lint

If a command cannot run because a dependency or network resource is missing,
record the full failing command and concise diagnostic in
`Surprises & Discoveries` and the final validation report.

## Validation and Acceptance

Configuration must define the `MQT::CoreQIRJIT` and `MQT::CoreQIRRuntime`
build-tree aliases and no runner target. Building the default or focused targets
must not produce `mqt-core-qir-runner`. The install manifest and generated
`mqt-core-targets.cmake` must contain no QIR executable, executor header, or QIR
JIT/runtime target.

The JIT test binary must pass module parsing, ABI validation, repeated session,
ownership, output, and state-extraction cases. The runtime test binary must pass
resource, gate, measurement, output-schema, and state ownership cases. The DDSIM
device test binary must pass QIR assembly and bitcode submission, sampling,
statevector, error, repeated-job, and concurrency cases. A zero exit status and
GoogleTest's complete pass summary are required.

DDSIM custom job parameter 1 must accept positive `int` seeds, reject invalid
values, and reproduce both OpenQASM and QIR sampling results across independent
jobs. Modules with multiple `entry_point` functions must be rejected.

The QIR guide must no longer tell users to build or invoke the removed native
runner. `UPGRADING.md` must not add migration text for the unreleased runner.
`uvx nox -s lint` must pass before handoff unless a documented environment
limitation prevents the run.

## Idempotence and Recovery

CMake configuration, focused builds, tests, documentation builds, searches, and
lint are repeatable. If the existing release build directory is stale, rerun
`cmake --preset release`; do not delete user build output. File moves stay
tracked by Git and can be corrected with additional moves or patches. Do not use
destructive Git commands. Inspect `git status --short` before every broad edit
and preserve any change outside this plan's scope.

## Artifacts and Notes

Issue #2096 requires the native runner and its subprocess test to disappear,
while DDSIM QIR execution and job isolation continue to work.

## Interfaces and Dependencies

At completion, `mqt-core-qir-runtime` and `mqt-core-qir-jit` are internal static
C++20 libraries below `mlir/lib/Dialect/QIR/Execution`. Both use
position-independent code because the shared DDSIM QDMI device links them. Their
`MQT::CoreQIRRuntime` and `MQT::CoreQIRJIT` aliases exist only in the build
tree; neither raw target has an install or export rule.

`src/qdmi/devices/dd/Device.cpp` gains the executor headers through private
links to the aliases. Existing in-process tests use the same aliases. The
existing QDMI custom parameter interface gains a DDSIM-specific contract: custom
parameter 1 is a positive `int` sampling seed. No binding or generated stub
changes are required. `JitSession` accepts only in-memory QIR plus an execution
mode and requires exactly one entry point.

Plan revision note (2026-08-25): Created the initial self-contained plan after
repository and issue inspection.

Plan revision note (2026-08-25): Added the shared internal QIR definitions
header after finding repeated compiler and JIT literals. Recorded the completed
implementation, focused test totals, install-tree evidence, lint result, and
documentation result.

Plan revision note (2026-08-26): Moved deterministic seeding to the DDSIM QDMI
job boundary, removed speculative multiple-entry selection, recorded issue
1734's current status, and authorized the pull-request update.

Plan revision note (2026-08-26): Moved the executor to the MLIR QIR subtree and
retained consistent build-tree target aliases.
