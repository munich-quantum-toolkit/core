# Reduce MQT Core build and CI time

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core's native code now takes long enough to compile that routine pull
requests occupy all available macOS runners and delay unrelated work. This
change reduces repeated compilation without reducing the supported operating
system or architecture coverage. A developer can observe the result by using the
existing CMake presets: normal builds use unity compilation and skip link time
optimization, while explicit LTO presets remain available for performance
builds. CI retains Release and Debug coverage on Linux, macOS, and Windows but
runs only one of each configuration per operating system.

## Progress

- [x] (2026-08-13 15:56Z) Create an isolated worktree from current `origin/main`
  and read repository policy.
- [x] (2026-08-13 16:24Z) Apply and validate the build-system defaults and unity
      collision fixes.
- [x] (2026-08-13 16:24Z) Apply and validate the CI matrix cleanup.
- [x] (2026-08-13 16:24Z) Run targeted C++, Python, preset, and lint validation.
- [x] (2026-08-13 16:31Z) Perform a fresh adversarial review and correct all
  in-scope findings.
- [x] (2026-08-13 16:31Z) Inspect the final diff and record the outcome.

## Surprises & Discoveries

- Observation: The existing `ENABLE_IPO` default is already off in CI because CI
  forces the deployment configuration on. It is on for local Release builds.
  Evidence: `cmake/StandardProjectSettings.cmake` derives the option default
  from `DEPLOY` and the build type.
- Observation: Header-set verification creates useful compile database entries
  but its aggregate target is not part of the default build. Evidence: a fresh
  Ninja graph did not make `all_verify_interface_header_sets` a dependency of
  `all`.
- Observation: A full unity build at the default batch size of eight exposes
  four test-only name collisions. Evidence: the DD test has two `vecNear`
  helpers, and three MLIR test targets use namespace names that become ambiguous
  when source files share one translation unit.
- Observation: The QC translation unity target additionally exposed a
  source-order dependency that was invisible in separate translation units. Two
  test files had file-scope `using namespace mlir` directives, which made later
  public-header references to `qc` ambiguous. Evidence: the generated unity
  source included the OpenQASM emission and translation tests before the
  quantum-computation translation test. Wrapping each test file in a named
  namespace contained the directive and made both unity and non-unity builds
  compile.
- Observation: Header verification remains available to clang-tidy without being
  part of normal builds. Evidence: the non-unity lint compile database contains
  1,112 lines referring to `_verify_interface_header_sets` commands and contains
  no `/Unity/` source paths.
- Observation: The build-system-only CI revision exposed a Windows-specific
  third-party collision. Enabling unity globally also enabled it for Cap'n
  Proto's KJ targets, while the former opt-out covered only `capnp`. MSVC then
  combined platform sources with duplicate anonymous-namespace helpers.
  Evidence: Windows Python job `94522290286` in run `31722197876` failed while
  compiling `_deps/capnproto-build/.../kj.dir/Unity/unity_2_cxx.cxx`.

## Decision Log

- Decision: Enable unity builds through first-party presets and Python package
  configuration instead of changing the global CMake default. Rationale: An
  embedded consumer must retain control over compilation of its surrounding
  project. Date/Author: 2026-08-13 / Codex.
- Decision: Keep the lint preset non-unity and enable header-set verification
  only there. Rationale: Lint must detect missing direct includes and needs
  compile commands for public headers. Date/Author: 2026-08-13 / Codex.
- Decision: Keep Python tests in Release. Rationale: scikit-build-core already
  selects Release, and these jobs provide Release coverage for architectures
  omitted from the smaller C++ matrix. Date/Author: 2026-08-13 / Codex.
- Decision: Remove all dormant extensive CI jobs. Rationale: They are not used,
  their Cartesian matrices consume excessive capacity, and the macOS matrix
  includes unsupported GCC combinations. Date/Author: 2026-08-13 / Codex.
- Decision: Contain file-scope MLIR using directives with named test namespaces
  instead of qualifying a large set of otherwise valid QC builder references.
  Rationale: The namespace leak was the root cause, and containing it preserves
  source readability while making unity source order irrelevant. Date/Author:
  2026-08-13 / Codex.
- Decision: Disable unity recursively for every target in the transitive Cap'n
  Proto build directory. Rationale: The incompatibility applies to KJ as well as
  `capnp`, while Jeff and first-party MQT targets must remain eligible. A
  directory-based traversal follows future Cap'n Proto target additions without
  hard-coding its current target list. Date/Author: 2026-08-13 / Codex.

## Outcomes & Retrospective

Implemented the build-system and CI portions of the approved plan in the
isolated `agent/ci-build-quick-wins` worktree. Normal first-party CMake presets
and scikit-build-core builds now use unity batches of eight with IPO and header
verification disabled. The lint preset remains non-unity and owns header-set
verification. Explicit Unix and Windows LTO presets retain intentional
performance builds. The routine C++ workflow now has six rows and no extensive
CI plumbing; the five Python platform rows and all coverage, minimums,
packaging, and wheel behavior are unchanged.

Fresh Release and Debug unity builds succeeded. Both complete native suites
passed, 4,488 tests each; the two existing QDMI job-ID tests were skipped in
each configuration. A complete non-unity lint build succeeded. Its cache and
compile database confirm unity off and header verification on. An existing
Release cache was deliberately switched to IPO on and then reconfigured with the
normal preset; both `ENABLE_IPO` and `CMAKE_INTERPROCEDURAL_OPTIMIZATION`
returned to off. The LTO preset resolved both values to on.

The Python package built as Release with unity on and IPO/header verification
off. The direct Python suite passed with 600 tests and three expected Qiskit
skips. Representative `tests-3.14` and `minimums-3.14` nox sessions passed; the
minimums session ran 603 tests successfully. Repository-wide lint and
`git diff --check` passed. The signed build-system commit was pushed to draft PR

## 2083 and its CI run was allowed to start before the matrix cleanup commit. The

PR is assigned to `burgholzer` and uses existing CI, tooling, code-quality, and
skip-changelog labels. The baseline passed Ubuntu ARM Python and Release C++ but
exposed the Cap'n Proto KJ unity failure on Windows. The follow-up build-system
commit disables unity for the complete transitive dependency; a local
reconfigure built `kj` and `capnp` as separate sources, and lint passed. No
GitHub label was deleted. Final cross-platform timing and terminal CI state
remain publication-stage evidence.

### Context and Orientation

`cmake/StandardProjectSettings.cmake` owns project-wide CMake defaults such as
header verification and interprocedural optimization. Interprocedural
optimization, also called link time optimization or LTO, lets the compiler
optimize across object-file boundaries but makes linking and compilation more
expensive. `CMakePresets.json` defines the supported local and CI CMake build
configurations. `pyproject.toml` supplies CMake definitions to scikit-build-core
when it builds the Python package.

CMake unity builds combine several source files into one generated translation
unit. This avoids repeatedly parsing common headers. The batch size limits how
many source files CMake combines. Normal builds will use the repository's
validated batch size of eight. `cmake/ExternalDependencies.cmake` already keeps
the incompatible Cap'n Proto target out of unity builds.

`.github/workflows/ci.yml` defines the repository's routine platform matrices
and delegates each matrix row to a shared reusable workflow. The Python matrix
already covers both Linux and macOS architectures plus x64 Windows. The C++
matrix can therefore retain one Release and one Debug row per operating system
without losing architecture coverage across the aggregate CI system.

The implementation is confined to this task worktree. It must not edit another
worktree, shared workflow repository, or GitHub label. Publication is authorized
through draft PR #2083.

### Plan of Work

First, update `cmake/StandardProjectSettings.cmake` so header verification and
IPO default to off. Ensure the CMake interprocedural optimization variable is
also forced off when `ENABLE_IPO` is off so an existing cache cannot retain the
old behavior.

Next, update `CMakePresets.json`. Enable unity builds with batch size eight and
explicitly disable IPO and header verification in the common normal presets.
Override the lint preset so unity is off and header verification is on. Add Unix
and Windows Release LTO configure, build, and test presets. Each LTO preset must
use a distinct build directory through its preset name.

Add the same normal-build definitions to `pyproject.toml` so Python package
builds use unity, skip IPO, and skip header verification while retaining
scikit-build-core's Release default. Keep the existing docs and stubs MinSizeRel
configuration.

Repair the four test-only unity collisions with narrow source changes. Rename
the duplicate DD helper to express its local purpose. Qualify affected MQT
namespace references from the global namespace and contain file-scope MLIR using
directives so they cannot affect later unity sources. Do not exclude a
first-party target from unity compilation.

Finally, rewrite each routine C++ CI matrix in `.github/workflows/ci.yml` as two
explicit rows: Linux ARM Release and x64 Debug, macOS Intel Release and ARM
Debug, and Windows ARM Release and x64 Debug. Remove all extensive C++ and
Python jobs and remove their references from the required-check aggregation. Do
not change Python tests, coverage, minimum-version sessions, packaging, or
shared workflows.

### Concrete Steps

Run all commands from the repository root through `.agent/run.sh` when they
create caches.

After the build-system edit, configure from a fresh cache:

    ./.agent/run.sh cmake --preset release
    ./.agent/run.sh cmake --build --preset release
    ./.agent/run.sh ctest --preset release

Repeat with the Debug preset. Inspect `build/release/CMakeCache.txt` and the
lint and LTO preset caches to confirm the intended values. Configure the lint
preset without running a rewriting formatter before the source changes are
complete.

Install and test the Python package with:

    ./.agent/run.sh uv sync --inexact --only-group build --only-group test
    ./.agent/run.sh uv sync --inexact --no-dev --no-build-isolation-package mqt-core
    ./.agent/run.sh uv run --no-sync pytest

Finish with:

    ./.agent/run.sh uvx nox -s lint
    git diff --check

### Validation and Acceptance

A fresh Release and Debug build must complete with unity enabled at batch size
eight. All CTest tests must pass. The four prior collision sites must compile
without target exclusions. The lint build must remain non-unity and its compile
database must contain header verification entries.

The normal Release cache must contain `CMAKE_UNITY_BUILD=ON`,
`CMAKE_UNITY_BUILD_BATCH_SIZE=8`, `ENABLE_IPO=OFF`,
`CMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF`, and
`CMAKE_VERIFY_INTERFACE_HEADER_SETS=OFF`. The lint cache must contain unity off
and header verification on. The Release LTO cache must contain `ENABLE_IPO=ON`
and interprocedural optimization on when the compiler supports it.

The Python package cache must use Release and the same normal unity, IPO, and
header-verification values. Python tests must pass without changing coverage or
minimum-version behavior.

The CI workflow must expose exactly six routine C++ rows, five unchanged Python
rows, and no extensive jobs. The required-check job must not refer to a removed
job. Repository lint and `git diff --check` must pass.

### Idempotence and Recovery

CMake configuration, builds, CTest, Python tests, and lint are safe to repeat.
If a stale cache obscures a default, use a new preset build directory or the
repository's cache-cleaning helper after no build process is active. Do not
delete another worktree or shared cache. If a unity collision appears, inspect
the generated `Unity` source and correct the smallest first-party name or
linkage conflict instead of disabling unity for the target.

### Artifacts and Notes

The original performance baseline is GitHub Actions run `31686027627`. The
build-system-only revision is `7fecfc4ad1a051edcd7e56559edb2acb43002978`, whose
GitHub Actions run `31722197876` provides the post-unity baseline before the
matrix cleanup is pushed. The local design experiments measured about 34 percent
less wall time for a representative unity build and about 38 percent more wall
time when ThinLTO was enabled. These values are evidence for the configuration
direction, not fixed acceptance thresholds for every compiler and operating
system.

### Interfaces and Dependencies

This change does not alter a C++ or Python runtime API. It adds the
`release-lto` and `release-lto-windows` developer presets. It uses only standard
CMake variables and the existing GitHub reusable workflows. It adds no package,
compiler, action, or runtime dependency.

Revision note: Initial plan created from the approved implementation plan and
the exact `origin/main` source state on 2026-08-13. Updated after implementation
and local validation on 2026-08-13.
