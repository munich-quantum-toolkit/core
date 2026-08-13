# Reduce MQT Core build and CI time

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core's native code takes long enough to compile that routine pull requests
occupy all available macOS runners and delay unrelated work. This change reduces
repeated compilation without reducing supported operating-system or architecture
coverage. Normal first-party CMake and Python package builds use unity
compilation. The lint build remains non-unity so it can detect missing includes
and source-order dependencies. CI retains Release and Debug coverage on Linux,
macOS, and Windows but runs one of each configuration per operating system.

The change does not alter link-time optimization, coverage, minimum-version
testing, packaging, wheel builds, or shared workflows. Alternative linkers,
identical code folding, and persistent compiler caching remain separate
experiments that require measurements before adoption.

## Progress

- [x] (2026-08-13 15:56Z) Create an isolated worktree from current `origin/main`
      and read repository policy.
- [x] (2026-08-13 16:24Z) Enable unity builds and repair the first-party test
  collisions exposed by combined translation units.
- [x] (2026-08-13 16:24Z) Reduce the routine C++ matrix and remove dormant
  extensive-CI paths.
- [x] (2026-08-13 19:23Z) Publish the build and CI changes to draft PR #2083.
- [x] (2026-08-13 19:40Z) Simplify the dependency opt-out and remove explicit
  defaults and LTO changes from the final design.
- [x] (2026-08-13 20:07Z) Run fresh Release, Debug, lint, and Python validation
  for the simplified design.
- [ ] Perform an adversarial review, publish the cleanup commit, update the PR
  description, and monitor exact-head CI.

## Surprises & Discoveries

- Observation: CI already disables interprocedural optimization because CI
  forces deployment configuration on. Local non-deployment Release builds enable
  it by default. Evidence: `cmake/StandardProjectSettings.cmake` derives the
  option default from `DEPLOY` and the build type.
- Observation: Header-set verification creates compile database entries needed
  by clang-tidy, but its aggregate target is not part of a normal default build.
  Evidence: a fresh Ninja graph did not make `all_verify_interface_header_sets`
  a dependency of `all`.
- Observation: A unity build at CMake's default batch size exposes four
  first-party test collisions. Evidence: two DD test files define a file-local
  helper with the same name, and three MLIR test targets leak namespace names
  when sources share one translation unit.
- Observation: The transitive Cap'n Proto KJ targets cannot use unity builds on
  Windows. Evidence: a Windows Python build combined platform sources with
  duplicate anonymous-namespace helpers and macros.
- Observation: CMake initializes `UNITY_BUILD_BATCH_SIZE` to eight when the
  project does not set `CMAKE_UNITY_BUILD_BATCH_SIZE`. An explicit value adds no
  behavior.
- Observation: The Jeff MLIR dependency contains eight C++ source files. It is
  simpler to compile this small dependency without unity than to discover and
  modify every target in its transitive Cap'n Proto build.
- Observation: Linux CI already selects mold through the pinned shared
  workflows. The portable toolchain bundles mold on Linux and LLD on Windows,
  but not on macOS. Linker changes therefore need platform-specific A/B tests in
  the shared workflows.

## Decision Log

- Decision: Enable unity through first-party presets and scikit-build-core
  configuration instead of changing the global CMake default. Rationale:
  embedded consumers retain control over their surrounding builds. Date/Author:
  2026-08-13 / Codex.
- Decision: Use CMake's default unity batch size. Rationale: the default is
  eight, which is the tested value, and an explicit override adds no behavior.
  Date/Author: 2026-08-13 / Codex.
- Decision: Keep the lint preset non-unity and enable header verification only
  there. Rationale: lint must detect direct-include defects and needs compile
  commands for public headers. Date/Author: 2026-08-13 / Codex.
- Decision: Set `CMAKE_UNITY_BUILD` to `OFF` in the function that makes Jeff
  available. Rationale: function scope contains the setting to Jeff and its
  dependencies, avoids target discovery, and restores unity for MQT Core.
  Date/Author: 2026-08-13 / Codex.
- Decision: Leave interprocedural optimization unchanged. Rationale: deployment
  configuration already disables it in CI, so changing local Release behavior
  does not improve CI time. Date/Author: 2026-08-13 / Codex.
- Decision: Keep Python tests in Release. Rationale: scikit-build-core already
  selects Release, and these jobs provide Release coverage for architectures
  omitted from the smaller C++ matrix. Date/Author: 2026-08-13 / Codex.
- Decision: Remove dormant extensive CI jobs. Rationale: the unused Cartesian
  matrices consume excessive capacity and include unsupported macOS GCC
  combinations. Date/Author: 2026-08-13 / Codex.
- Decision: Defer compiler-cache, linker, and identical-code-folding changes.
  Rationale: their value and compatibility depend on the runner, compiler, and
  output binary and require separate measurements. Date/Author: 2026-08-13 /
  Codex.

## Outcomes & Retrospective

The final implementation enables unity in normal first-party CMake presets and
scikit-build-core builds. CMake supplies the default batch size of eight. The
lint preset disables unity and enables interface-header verification. Jeff and
its complete dependency subtree compile without unity through one
function-scoped variable. Header verification defaults to off for ordinary
builds. Interprocedural optimization retains its pre-existing behavior.

The routine C++ workflow has six rows: Linux ARM Release and x64 Debug, macOS
Intel Release and ARM Debug, and Windows ARM Release and x64 Debug. The five
Python platform jobs continue to provide Release coverage. Python coverage,
minimum-version sessions, packaging, and wheel matrices are unchanged. All
dormant extensive jobs, label conditions, and required-check references are
removed.

Fresh Release and Debug builds completed. Each native suite passed all 4,488
tests, with the two expected QDMI job-ID skips. The fresh Release graph used
first-party unity units with at most eight sources and compiled Jeff and Cap'n
Proto from separate source files. The lint build completed without unity. Its
compile database contains 1,112 interface-header verification entries and no
unity source. `tests-3.14` passed 598 tests with three expected Qiskit skips.
`minimums-3.14` passed all 601 tests. Repository lint and `git diff --check`
passed. Exact-head CI results will be added after they reach terminal states.

## Context and Orientation

`CMakePresets.json` defines supported local and CI CMake configurations.
`pyproject.toml` supplies definitions to scikit-build-core when it builds the
Python package. `cmake/StandardProjectSettings.cmake` owns ordinary project
defaults. `cmake/ExternalDependencies.cmake` fetches Jeff MLIR, which fetches
Cap'n Proto. `.github/workflows/ci.yml` defines MQT Core's platform matrices and
delegates each row to pinned shared workflows.

A unity build combines source files into generated translation units. This
avoids repeated parsing of common headers. CMake initializes the target batch
size to eight when the project does not override it. A function scope in CMake
applies variables to functions and subdirectories called from that function,
then restores the caller's value when the function returns. The Jeff setup uses
this behavior to keep only that dependency subtree non-unity.

## Plan of Work

Keep `CMAKE_UNITY_BUILD=ON` in the shared first-party preset and the
scikit-build-core definitions. Do not set the batch size, header-verification
default, or IPO option at those call sites. Keep the lint preset's explicit
unity-off and header-verification-on overrides.

Default interface-header verification to off in
`cmake/StandardProjectSettings.cmake`. Preserve the rest of that file's IPO
logic exactly. In `cmake/ExternalDependencies.cmake`, set unity off immediately
before making Jeff available. Do not inspect or mutate dependency targets after
creation.

Retain the narrow first-party source corrections needed for unity builds. Use
distinct file-local DD helper names. Contain file-scope MLIR using directives
and qualify ambiguous MQT namespaces. Do not exclude first-party targets from
unity compilation.

Keep one Release and one Debug C++ row on each operating system. Remove the
extensive C++ and Python jobs and their workflow conditions. Do not change the
five Python jobs or the coverage, packaging, minimum-version, wheel, and shared
workflow behavior.

## Concrete Steps

Run cache-producing commands from the repository root through `.agent/run.sh`.
Use fresh build directories for the native acceptance builds:

    ./.agent/run.sh cmake --preset release
    ./.agent/run.sh cmake --build --preset release
    ./.agent/run.sh ctest --preset release

Repeat those commands with `debug`. Configure and build `lint`, then inspect its
cache and compile database. Run the representative Python sessions:

    ./.agent/run.sh uvx nox -s tests-3.14
    ./.agent/run.sh uvx nox -s minimums-3.14

Finish with:

    ./.agent/run.sh uvx nox -s lint
    git diff --check

## Validation and Acceptance

Fresh Release and Debug builds must complete with unity enabled and no explicit
batch-size cache entry. Generated first-party unity files must combine at most
eight sources. Jeff and Cap'n Proto build directories must contain no generated
unity sources. All configured CTest tests must pass.

The lint cache must contain `CMAKE_UNITY_BUILD=OFF` and
`CMAKE_VERIFY_INTERFACE_HEADER_SETS=ON`. Its compile database must contain
interface-header verification entries and no generated unity sources. A normal
deployment configuration must retain `ENABLE_IPO=OFF`; a local non-deployment
Release configuration must retain the established default of `ENABLE_IPO=ON`.

The Python package cache must use Release and unity. The `tests-3.14` and
`minimums-3.14` sessions must pass. The CI workflow must expose exactly six
routine C++ rows, five unchanged Python rows, and no extensive jobs. Repository
lint and `git diff --check` must pass.

All revised C++, Python, coverage, lint, and packaging jobs on the published
head must reach successful terminal states. Windows Python proves the scoped
Cap'n Proto fix. All four macOS jobs prove the most capacity-constrained
platform.

## Idempotence and Recovery

CMake configuration, builds, CTest, Python tests, and lint are safe to repeat.
Use a new preset build directory when a stale cache can obscure a default. Do
not remove another worktree or shared cache. If a new first-party unity
collision appears, inspect the generated unity source and correct the smallest
linkage or namespace conflict instead of excluding the target.

## Artifacts and Notes

GitHub Actions run `31686027627` is the original timing baseline. Cold local
AppleClang measurements reduced a representative native and MLIR build from
111.2 seconds to 73.2 seconds with unity compilation, about 34 percent. Compare
the final build steps and total job times with that baseline, then monitor later
natural runs for variance.

Linux already uses mold in the pinned shared workflows. A future shared-workflow
experiment can select the mold binary shipped by the portable toolchain and
remove the separate setup action. A separate Windows experiment can compare the
default generator and linker against Ninja and the bundled LLD. Persistent
compiler caching should be tested on macOS first, then Windows. Identical code
folding needs binary-size, link-time, runtime, and function-address tests; it is
not part of this change.

## Interfaces and Dependencies

This change does not alter a C++ or Python runtime API. It adds no preset,
package, compiler, action, or runtime dependency. Direct embedded CMake
consumers do not receive a forced unity default.

Revision note: Created from the approved implementation plan and updated on
2026-08-13 to record the simplified final design.
