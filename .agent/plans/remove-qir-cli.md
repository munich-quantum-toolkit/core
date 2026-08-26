# Remove the standalone QIR runner and internalize QIR execution

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core must stop presenting its LLVM-based Quantum Intermediate Representation
(QIR) executor as a standalone product. After this change, a source build and
installation contain no `mqt-core-qir-runner` command, no `MQT::CoreQIRJIT` or
`MQT::CoreQIRRuntime` build-tree alias, and no public QIR runtime or
just-in-time compiler headers. The DDSIM QDMI device still accepts and executes
supported QIR assembly and bitcode jobs. Its focused tests prove that every job
retains independent simulator, runtime, output, and random number generator
state.

## Progress

- [x] (2026-08-25 01:17Z) Read issue #2096, repository policy, and the QIR
      CMake, source, test, and documentation surfaces.
- [x] (2026-08-25 01:20Z) Moved the retained JIT and runtime headers out of the
      public include tree and replaced public CMake aliases with internal target
      dependencies.
- [x] (2026-08-25 01:20Z) Deleted the native runner implementation, CMake
      target, and subprocess test.
- [x] (2026-08-25 01:21Z) Removed runner documentation and documented the v4
      removal of the CLI, aliases, and headers.
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

## Surprises & Discoveries

- Observation: The JIT and runtime libraries are not members of
  `MQT_CORE_TARGETS`, so the current installed CMake package does not export
  them. The remaining public surface consists of build-tree aliases and headers
  below `include/mqt-core/qir`. Evidence: `src/qir/jit/CMakeLists.txt` and
  `src/qir/runtime/CMakeLists.txt` create aliases through `add_mqt_core_library`
  but do not append their targets to `MQT_CORE_TARGETS`.
- Observation: The runner test is only a subprocess smoke test. JIT session
  tests and DDSIM device tests already cover execution, repeated runs,
  ownership, concurrent jobs, and result isolation without requiring a
  production executable. Evidence: `test/qir/runner/test_qir_runner.cpp`,
  `test/qir/jit/test_jit_session.cpp`, and
  `test/qdmi/devices/dd/concurrency_test.cpp`.
- Observation: The compiler and JIT repeated QIR attribute, profile, schema, and
  irreversible-operation literals. Evidence: searches found the same strings in
  `mlir/lib/Dialect/QIR` and `src/qir/jit`; both now use
  `src/qir/include/qir/Definitions.hpp`.
- Observation: Release configuration, package installation, and documentation
  need the local LLVM/MLIR package path on this host. Evidence: all three pass
  with `MLIR_DIR=/home/nvidia/.local/opt/mqt-llvm-mlir/22.1.7/lib/cmake/mlir`.
- Observation: The full release build installs cleanly to a temporary prefix.
  Evidence: the final install-tree scan found no native QIR runner, public QIR
  headers, or QIR package targets.
- Observation: Two of issue #1734's three intended outcomes are now covered.
  Base Profile IR and pipeline tests exist, and QIR-Runner provides external
  interoperability checks. The Base Profile conversion still contains its own
  `ConvertMemRefLoadOp`; centralizing dynamic-to-static qubit conversion remains
  separate compiler work.

## Decision Log

- Decision: Keep `mqt-core-qir-jit` and `mqt-core-qir-runtime` as raw static
  CMake targets and remove only their `MQT::` aliases. Rationale: The DDSIM
  device and focused unit tests need linkable targets, while raw unexported
  target names and source-private headers state that these are implementation
  details. Date/Author: 2026-08-25 / Codex.
- Decision: Move retained headers to `src/qir/include/qir` and expose that
  directory only through build-interface usage requirements on the internal
  targets. Rationale: JIT, runtime, DDSIM, and tests can share the same
  declarations without installing them below the public `include/mqt-core` tree.
  Date/Author: 2026-08-25 / Codex.
- Decision: Remove the subprocess runner test instead of converting it.
  Rationale: The issue explicitly removes subprocess tests, while in-process JIT
  and DDSIM tests own the retained behavioral contracts. Date/Author: 2026-08-25
  / Codex.
- Decision: Store shared QIR attribute, profile, schema, and ABI-prefix literals
  in `src/qir/include/qir/Definitions.hpp`. Rationale: Compiler generation and
  internal execution must agree on these spellings, while the source-private
  location avoids creating a replacement public API. Date/Author: 2026-08-25 /
  Codex.
- Decision: Keep deterministic seeding as DDSIM QDMI custom job parameter 1
  instead of a JIT session option. Rationale: The job owns sampling policy and
  can apply one seed contract to OpenQASM and QIR. Date/Author: 2026-08-26 /
  Codex.
- Decision: Require exactly one standard QIR entry point and remove named
  selection. Rationale: MQT's compiler emits one entry point, while QIR-Runner
  already serves external modules that need selection. Date/Author: 2026-08-26 /
  Codex.

## Outcomes & Retrospective

The native QIR runner, its subprocess test, its user guide, the build-tree
`MQT::` JIT/runtime aliases, and all public QIR runtime/JIT headers are removed.
The retained implementation is source-private and shared by DDSIM, the compiler,
and focused tests. DDSIM exposes deterministic OpenQASM and QIR sampling through
custom job parameter 1. The internal JIT accepts only in-memory modules with one
standard QIR entry point. The focused C++ tests, all 4,035 configured C++ tests,
254 Python QDMI/QIR tests, documentation, installation, and lint pass. The
install-tree scan proves that the removed surfaces are absent.

## Context and Orientation

QIR is an LLVM-based interchange format for quantum programs. MQT Core compiles
programs to QIR in `mlir/` and executes QIR inside the decision diagram
simulator device in `src/qdmi/devices/dd/Device.cpp`. The executor consists of a
runtime in `src/qir/runtime`, which implements quantum instructions and QIR
resource/output functions, and a just-in-time compiler in `src/qir/jit`, which
loads LLVM modules, validates declarations, binds runtime symbols, and invokes
an entry point. The standalone command in `src/qir/runner` wraps those two
libraries. Tests mirror these directories under `test/qir`; DDSIM integration
tests live under `test/qdmi/devices/dd`.

`add_mqt_core_library` creates both a raw target and a namespaced `MQT::` alias.
Namespaced aliases look like supported consumer APIs inside an embedding build.
Installed targets are controlled separately through `MQT_CORE_TARGETS` in
`src/CMakeLists.txt`. The QIR libraries are already absent from that install
list. This task must also remove the aliases and public header placement.

The change is limited to QIR runtime/JIT/runner build files and headers, their
direct consumers and tests, the QIR user guide, `UPGRADING.md`, and this plan.
It must preserve unrelated worktree changes. It must follow `AGENTS.md` and
`docs/ai_usage.md`. The maintainer authorized a fast-forward update to pull
request #2246 after local validation and signed-commit verification.

## Plan of Work

First, create an internal header root at `src/qir/include/qir`. Move
`Session.hpp`, `IRRewriter.hpp`, `Runtime.hpp`, and `QIR.h` there without
changing their logical include spellings. Adjust `src/qir/jit/CMakeLists.txt`
and `src/qir/runtime/CMakeLists.txt` to use ordinary static libraries rather
than `add_mqt_core_library`, set position-independent code directly, attach
project warning and option targets, and publish only the source-local build
include directory to internal dependents. The project-wide C++20 setting also
applies to these targets. Link the JIT, DDSIM device, and tests through raw
`mqt-core-qir-*` target names. Do not add either target to `MQT_CORE_TARGETS`
and do not create a namespaced alias.

Second, remove `src/qir/runner`, remove its subdirectory from
`src/qir/CMakeLists.txt`, remove `test/qir/runner`, and remove its subdirectory
from `test/qir/CMakeLists.txt`. Confirm that no source or CMake reference to the
runner or old aliases remains.

Third, revise `docs/qir/index.md`. Remove the native runner build and use
sections. Retain compiler output documentation, the external QIR Alliance runner
example, and the DDSIM QDMI execution contract. Make clear that MQT Core's
retained runtime and JIT are internal to DDSIM. Add an Unreleased v4 upgrade
section to `UPGRADING.md` that lists the removed executable, build-tree aliases,
and headers and points users to DDSIM QDMI or an external QIR runtime. Because
this change removes unreleased v4-facing surfaces and repository policy says not
to add a standalone changelog entry for unreleased v4 functionality, do not add
a new `CHANGELOG.md` item.

Fourth, reduce the retained JIT to the in-memory DDSIM contract. Remove file
loading, named entry-point selection, session-owned seeding, and unused
accessors. Require one function with the standard `entry_point` attribute.
Define DDSIM custom job parameter 1 as an optional positive `int` seed and use
it for both OpenQASM and QIR sampling. Document this migration in
`docs/qdmi/ddsim_device.md`, `UPGRADING.md`, and the existing aggregate QIR
changelog entry.

Finally, configure the release preset, build the focused QIR JIT/runtime and
DDSIM test targets, run those binaries, and inspect an install tree or generated
target file to prove that no QIR executable, headers, or package target remains.
Build documentation if the environment permits it. Run `uvx nox -s lint`,
inspect the diff and status, and record exact outcomes in this plan.

## Concrete Steps

Run all commands from the repository root.

Move files with `git mv`, then edit CMake and documentation with focused
patches. Search the result with:

    rg -n "mqt-core-qir-runner|CoreQIRJIT|CoreQIRRuntime|include/mqt-core/qir" \
      CMakeLists.txt cmake src test include docs UPGRADING.md

The search must return only intentional migration text in `UPGRADING.md`, if
any.

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

Configuration must succeed without defining `MQT::CoreQIRJIT`,
`MQT::CoreQIRRuntime`, or `MQT::CoreQIRRunner`. Building the default or focused
targets must not produce `mqt-core-qir-runner`. The install manifest and
generated `mqt-core-targets.cmake` must contain no QIR executable, QIR header,
or QIR JIT/runtime target.

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
runner. The upgrade guide must name every removed CLI and CMake/header surface
and provide a migration path. `uvx nox -s lint` must pass before handoff unless
a documented environment limitation prevents the run.

## Idempotence and Recovery

CMake configuration, focused builds, tests, documentation builds, searches, and
lint are repeatable. If the existing release build directory is stale, rerun
`cmake --preset release`; do not delete user build output. File moves stay
tracked by Git and can be corrected with additional moves or patches. Do not use
destructive Git commands. Inspect `git status --short` before every broad edit
and preserve any change outside this plan's scope.

## Artifacts and Notes

Initial evidence from the source tree:

    src/qir/CMakeLists.txt: add_subdirectory(runner)
    src/qir/jit/CMakeLists.txt: add_mqt_core_library(... ALIAS_NAME QIRJIT)
    src/qir/runtime/CMakeLists.txt: add_mqt_core_library(... ALIAS_NAME QIRRuntime)
    src/qdmi/devices/dd/CMakeLists.txt: MQT::CoreQIRJIT MQT::CoreQIRRuntime

Issue #2096 requires the native runner and its subprocess test to disappear,
while DDSIM QIR execution and job isolation continue to work.

## Interfaces and Dependencies

At completion, `mqt-core-qir-runtime` is an internal static C++20 library that
contains `src/qir/runtime/QIR.cpp` and `src/qir/runtime/Runtime.cpp`. It links
the public Core IR and DD targets and exposes `src/qir/include` only to build
tree dependents. `mqt-core-qir-jit` is an internal static C++20 library that
contains `src/qir/jit/Session.cpp` and `src/qir/jit/IRRewriter.cpp`. It links
`mqt-core-qir-runtime` and the existing LLVM execution engine components. Both
targets use position-independent code because the shared DDSIM QDMI device links
them. Neither target has an `MQT::` alias or install/export rule.

`src/qdmi/devices/dd/Device.cpp` continues to include `qir/jit/Session.hpp` and
`qir/runtime/Runtime.hpp` and gains those includes through its private link to
the internal targets. Existing in-process tests use the same internal targets.
The existing QDMI custom parameter interface gains a DDSIM-specific contract:
custom parameter 1 is a positive `int` sampling seed. No binding or generated
stub changes are required. `JitSession` accepts only in-memory QIR plus an
execution mode and requires exactly one entry point.

Plan revision note (2026-08-25): Created the initial self-contained plan after
repository and issue inspection.

Plan revision note (2026-08-25): Added the shared internal QIR definitions
header after finding repeated compiler and JIT literals. Recorded the completed
implementation, focused test totals, install-tree evidence, lint result, and
documentation result.

Plan revision note (2026-08-26): Moved deterministic seeding to the DDSIM QDMI
job boundary, removed speculative multiple-entry selection, recorded issue
1734's current status, and authorized the pull-request update.
