# Reduce MQT Core build and CI time

Status: historical implementation record.

## Goal and scope

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

## Constraints

- CI already disables interprocedural optimization because CI forces deployment
  configuration on. Local non-deployment Release builds enable it by default.
  Evidence: `cmake/StandardProjectSettings.cmake` derives the option default
  from `DEPLOY` and the build type.

- Header-set verification creates compile database entries needed by clang-tidy,
  but its aggregate target is not part of a normal default build. Evidence: a
  fresh Ninja graph did not make `all_verify_interface_header_sets` a dependency
  of `all`.

- A unity build at CMake's default batch size exposes four first-party test
  collisions. Evidence: two DD test files define a file-local helper with the
  same name, and three MLIR test targets leak namespace names when sources share
  one translation unit.

- The transitive Cap'n Proto KJ targets cannot use unity builds on Windows.
  Evidence: a Windows Python build combined platform sources with duplicate
  anonymous-namespace helpers and macros.

- CMake initializes `UNITY_BUILD_BATCH_SIZE` to eight when the project does not
  set `CMAKE_UNITY_BUILD_BATCH_SIZE`. An explicit value adds no behavior.

- The Jeff MLIR dependency contains eight C++ source files. It is simpler to
  compile this small dependency without unity than to discover and modify every
  target in its transitive Cap'n Proto build.

- Linux CI already selects mold through the pinned shared workflows. The
  portable toolchain bundles mold on Linux and LLD on Windows, but not on macOS.
  Linker changes therefore need platform-specific A/B tests in the shared
  workflows.

## Decisions

- Enable unity through first-party presets and scikit-build-core configuration
  instead of changing the global CMake default. Rationale: embedded consumers
  retain control over their surrounding builds.

- Use CMake's default unity batch size. Rationale: the default is eight, which
  is the tested value, and an explicit override adds no behavior.

- Keep the lint preset non-unity and enable header verification only there.
  Rationale: lint must detect direct-include defects and needs compile commands
  for public headers.

- Set `CMAKE_UNITY_BUILD` to `OFF` in the function that makes Jeff available.
  Rationale: function scope contains the setting to Jeff and its dependencies,
  avoids target discovery, and restores unity for MQT Core.

- Leave interprocedural optimization unchanged. Rationale: deployment
  configuration already disables it in CI, so changing local Release behavior
  does not improve CI time.

- Keep Python tests in Release. Rationale: scikit-build-core already selects
  Release, and these jobs provide Release coverage for architectures omitted
  from the smaller C++ matrix.

- Remove dormant extensive CI jobs. Rationale: the unused Cartesian matrices
  consume excessive capacity and include unsupported macOS GCC combinations.

- Defer compiler-cache, linker, and identical-code-folding changes. Rationale:
  their value and compatibility depend on the runner, compiler, and output
  binary and require separate measurements.

## Outcome and validation

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
passed. Final hosted CI was not recorded.

## Code and ownership

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

## Acceptance

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

## Interfaces

This change does not alter a C++ or Python runtime API. It adds no preset,
package, compiler, action, or runtime dependency. Direct embedded CMake
consumers do not receive a forced unity default.
