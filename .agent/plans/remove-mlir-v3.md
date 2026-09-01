# Remove the v3 MLIR and QIR stacks

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core v3 contains an optional compiler subsystem based on MLIR and a QIR
execution stack consisting of a runtime, an LLVM-based JIT, a command-line
runner, and optional QIR input support in the DDSIM QDMI device. After this
change, the v3 source tree, normal build, tests, continuous integration, and
current documentation no longer provide or exercise either stack. Users can
configure, build, test, package, and document MQT Core without installing MLIR,
LLVM, or their Python documentation and test helpers.

A reviewer can observe the result by confirming that the `mlir/`,
`include/mqt-core/qir/`, `src/qir/`, `test/qir/`, `docs/mlir/`, and `docs/qir/`
trees are absent; no CMake option or preset enables MLIR or QIR execution;
dependency resolution omits `lit` and `mlir-pygments`; and the normal build,
tests, and documentation pass without an MLIR or LLVM package.

## Progress

- [x] (2026-09-01 09:28Z) Created a separate worktree from the current
  `origin/v3.x` tip and confirmed that its starting tree is clean.
- [x] (2026-09-01 09:28Z) Started exhaustive audits of code and tests, build and
      dependency files, and documentation and template inputs.
- [x] (2026-09-01 09:58Z) Deleted the MLIR implementation, tests, tools,
  documentation, and helper modules: 130 files and 24,797 lines.
- [x] (2026-09-01 09:58Z) Removed MLIR build switches, presets, CI setup,
      package metadata, dependencies, coverage paths, and release-category
      inputs.
- [x] (2026-09-01 09:58Z) Decoupled DDSIM QIR support from MLIR by finding LLVM
  directly when the optional feature is enabled.
- [x] (2026-09-01 09:58Z) Removed current documentation links and MLIR-dependent
      QIR instructions, and added release and upgrade notes.
- [x] (2026-09-01 09:58Z) Regenerated `uv.lock` and completed release, optional
      QIR, Python, source-distribution, structural documentation, and lint
      checks.
- [x] (2026-09-01 09:58Z) Audited the final tree and diff. No active MLIR path,
  setting, dependency, test, documentation page, or template input remains.
- [x] (2026-09-01 10:35Z) Removed the standalone QIR runner, its build option,
      tests, and documentation while preserving the runtime and DDSIM
      integration.
- [x] (2026-09-01 11:10Z) Expanded the requested removal to the complete
      Core-owned QIR execution stack and audited the boundary with generic QDMI
      program-format support.
- [x] (2026-09-01 11:18Z) Removed the QIR runtime, JIT, public headers, test
      fixtures, LLVM setup, DDSIM integration, build option, current
      documentation, and release-category source input.
- [x] (2026-09-01 11:42Z) Rebuilt, retested, linted, regenerated clean
      documentation, and audited the expanded removal.

## Surprises & Discoveries

- Observation: The v3 templating workflow already uses the non-MLIR `c++-python`
  project type. Evidence: `.github/workflows/templating.yml` contains
  `project-type: c++-python` at the fetched `origin/v3.x` tip. The custom
  release-drafter category input still asks the template to emit an MLIR
  category.
- Observation: The QIR runner and DDSIM QIR support use LLVM APIs, but v3 ties
  both options to `BUILD_MQT_CORE_MLIR`. Evidence: the top-level
  `CMakeLists.txt` defines both with `cmake_dependent_option`, while
  `src/qir/jit/CMakeLists.txt` links LLVM components and contains no MLIR API.
- Observation: Default QIR runtime tests require the LLVM `llc` executable even
  when the MLIR option is off. Evidence: `test/qir/runtime/CMakeLists.txt` fails
  configuration when `llc` is absent.
- Observation: When LLVM is not enabled, GoogleTest rejects the empty
  parameterized QIR executable suite. Evidence: the first full release CTest run
  failed only the uninstantiated `QIRFilesTest`; guarding that suite with
  `MQT_CORE_QIR_EXECUTABLE_TESTS` made all 1,531 release tests pass.
- Observation: LLVM exposes its package version as `LLVM_PACKAGE_VERSION`.
  Evidence: an explicit minimum of version 99 now rejects the installed LLVM
  22.1.8 package before configuration continues.
- Observation: CMake caches `find_program` results by default. Evidence: a build
  tree retained the old `LLC_EXECUTABLE` cache entry after LLVM was disabled.
  The new non-cached variable and explicit LLVM gate correctly removed all `.ll`
  executable targets on reconfiguration.
- Observation: A fully executed documentation notebook build needs the external
  Graphviz `dot` executable, which is absent on this host. Evidence:
  `docs/dd_package.md` stopped at `ExecutableNotFound: dot`. The structural HTML
  build passed with notebook execution disabled.
- Observation: The remaining QIR executor was the project's only LLVM package
  consumer. Evidence: all `find_package(LLVM)`, LLVM component mapping, LLVM
  headers, and `llc` use were confined to the QIR JIT, DDSIM QIR path, and their
  tests. Removing those paths makes `cmake/SetupLLVM.cmake` obsolete.
- Observation: QDMI itself defines four QIR program formats. Evidence: the
  pinned QDMI headers expose Base and Adaptive Profile string and module enum
  values. Generic FoMaC and Python QDMI clients must keep classifying and
  exposing these values so external devices can use the upstream protocol.

## Decision Log

- Decision: Base the work on the fetched `origin/v3.x` tip rather than the stale
  local `v3.x` branch. Rationale: the requested branch update must include all
  remote v3 changes while leaving the user's existing branches untouched.
  Date/Author: 2026-09-01 / Codex.
- Decision: Initially preserve the independent QIR runtime and DDSIM QIR
  formats, replace their accidental MLIR gate with direct LLVM discovery, and
  remove only the standalone runner. This decision was superseded when the user
  expanded the scope to the entire QIR stack. Date/Author: 2026-09-01 / Codex.
- Decision: Remove the complete Core-owned QIR execution stack, including its
  public libraries and DDSIM implementation. Keep generic QDMI QIR format
  identifiers and payload classification because they belong to the upstream
  protocol and support external devices. Rationale: this removes every QIR
  executor owned by MQT Core without breaking unrelated generic QDMI client
  semantics. Date/Author: 2026-09-01 / Codex.
- Decision: Do not hand-edit files that declare themselves generated from the
  external templates repository. Keep `project-type: c++-python`, which is
  already the required type, and change only source inputs owned by this
  repository. Rationale: this follows the repository policy and the user's
  explicit instruction. Date/Author: 2026-09-01 / Codex.
- Decision: Keep accurate historical changelog entries and unrelated research
  comparisons, but remove every current feature page, navigation link, setup
  instruction, and supported-activity statement for MLIR. Rationale: release
  history must remain accurate, while current documentation must not advertise
  removed behavior. Date/Author: 2026-09-01 / Codex.

## Outcomes & Retrospective

The v3 tree no longer contains the MLIR source, tools, tests, documentation, or
CMake helpers. Build presets, CI, Read the Docs, package metadata, lint rules,
coverage configuration, and release-category source data no longer enable or
name the removed subsystem. The generated release-drafter file remains unchanged
by policy; its repository-owned category input is corrected and the templating
workflow will update it.

The Core-owned QIR runtime, LLVM JIT, standalone runner, DDSIM execution path,
public headers, tests, fixtures, and documentation are also absent. No CMake
path locates LLVM. Generic QDMI clients still expose the QIR program-format enum
values required by the upstream QDMI interface, but MQT Core no longer executes
those formats itself.

The release build passed. CTest passed 1,688 tests with two existing skips. The
Python 3.13 suite passed 574 tests with three expected skips. The source
distribution built and contains no removed implementation path. All repository
lint hooks passed with Python 3.13. A clean structural Sphinx build passed with
602 existing warnings and generated no QIR namespace page or XML. Notebook
execution remains limited by the sandbox's ban on local Jupyter port binding.

The final scan found only removal notes, accurate historical records, generic
QDMI format handling, this plan, and the generated release-drafter output. None
is an active Core QIR executor, LLVM dependency, build option, test, package, or
current feature page.

## Context and Orientation

The top-level `CMakeLists.txt` owns feature options and adds the `mlir/` source
tree. `cmake/SetupMLIR.cmake` locates both MLIR and LLVM, which caused the
independent QIR JIT in `src/qir/jit/` to rely on MLIR setup. `CMakePresets.json`
contains separate MLIR variants and makes coverage and lint inherit an MLIR
preset. The `.github/workflows/ci.yml`, `.github/workflows/slurm.yml`, and
`.readthedocs.yaml` files install or configure MLIR.

The full compiler implementation, command-line tool, CMake tests, lit tests, and
unit tests live under `mlir/`. Handwritten compiler documentation lives under
`docs/mlir/`, and `docs/index.md` adds it to the Sphinx navigation. Python
development and documentation dependencies live in `pyproject.toml`, with
resolved versions in `uv.lock`. Repository-wide support files include
MLIR-specific paths in `.gitignore`, `.license-tools-config.json`,
`.github/codecov.yml`, and
`.github/workflow_inputs/release_drafter_categories.json`.

Before removal, `src/qir/runtime/` provided the DD-based runtime and
`src/qir/jit/` provided the LLVM executor. Public headers lived under
`include/mqt-core/qir/`; tests and LLVM IR fixtures lived under `test/qir/` and
`test/circuits/`. The optional DDSIM integration connected these libraries to
four QDMI program formats. No other project component needs the LLVM package.

## Plan of Work

First, remove the `mlir/`, `docs/mlir/`, `cmake/SetupMLIR.cmake`, and
`cmake/CleanMLIRDocs.cmake` trees and files. Remove the two Python tests that
compile input through the absent `mqt.core.mlir` module. Clean all top-level
CMake branches that add the deleted tree or documentation target.

Next, remove the complete QIR source, public headers, tests, LLVM IR fixtures,
and documentation. Remove the QIR runner and DDSIM option, delete the LLVM setup
module, and remove the DDSIM QIR execution and test paths. Keep only generic
QDMI format identifiers and binary-payload classification used by external
devices.

Then, collapse `CMakePresets.json` to the normal Unix and Windows matrix. Make
coverage and lint inherit the normal Unix base. Change CI jobs to those presets
and remove all MLIR setup inputs. Remove the standalone MLIR setup from the
Slurm workflow and Read the Docs build. Remove MLIR paths from Codecov, source
distribution checks, type-check exclusions, license rules, ignore rules, and
release-drafter input data.

Remove `lit` and `mlir-pygments` from `pyproject.toml` and regenerate `uv.lock`
with `uv lock`. Do not edit the lock file by hand. Remove the compiler and QIR
pages from the documentation navigation. Add a breaking change to `CHANGELOG.md`
and migration guidance to `UPGRADING.md`; preserve older release records because
they describe versions that still shipped the subsystem. Update `AGENTS.md`
because it is maintained directly on v3 and still instructs agents to build and
test the removed tree.

Finally, search tracked files by path and content for active MLIR and QIR
executor artifacts. Inspect every remaining match and classify it as an accurate
historical record, an unrelated external comparison, or a defect to remove. Run
the release configure, build, CTest, Python tests, documentation build, lock
check, CMake preset listing, and full lint.

## Concrete Steps

Run all commands from the repository root. Use the worktree-local wrapper for
cache-producing tools.

    git status --short --branch
    rg -n -i 'mlir|mqtopt|mqtref|quantum-opt|qir' --hidden -g '!.git'
    ./.agent/run.sh uv lock
    ./.agent/run.sh cmake --list-presets=all
    ./.agent/run.sh cmake --preset release
    ./.agent/run.sh cmake --build --preset release
    ./.agent/run.sh ctest --preset release
    ./.agent/run.sh uv run --no-sync pytest
    ./.agent/run.sh uvx nox --non-interactive -s docs
    ./.agent/run.sh uvx nox -s lint

Record short command outcomes in `Artifacts and Notes` as validation completes.

## Validation and Acceptance

Acceptance requires `cmake --list-presets=all` to show no MLIR preset and a
normal `release` configuration to succeed without `MLIR_DIR` or `LLVM_DIR`. The
release build and CTest run must pass.

`uv lock --check` must report that `uv.lock` is current. The full Python test
suite must contain no test that imports `mqt.core.mlir`. The Sphinx
documentation must build without downloading or locating MLIR, and its source
navigation must contain no removed compiler page. `uvx nox -s lint` must pass. A
final tracked file search must find no MLIR or Core-owned QIR implementation,
build switch, preset, CI setup, live dependency, test, current documentation
page, or release-category input. Generated files, generic upstream QDMI format
support, and historical records may remain only when repository policy forbids
direct edits or when the text accurately describes an older release.

## Idempotence and Recovery

File deletion and text edits are repeatable through the patch history. CMake
uses a separate `build/release` tree, so failed configurations can be retried
after corrections without changing source files. `uv lock` is deterministic from
`pyproject.toml`. If a generated file still contains removed text, do not patch
that file; correct its repository-owned template input and record that a later
template render will update it. Preserve unrelated user changes and never modify
another worktree.

## Artifacts and Notes

Initial evidence:

    origin/v3.x = 46b9b0f2d (v3.9.2)
    .github/workflows/templating.yml: project-type: c++-python
    pyproject.toml docs group: mlir-pygments>=1.0.0
    pyproject.toml dev group: lit>=18.1.8

Expanded final validation:

    fresh CMake configure: passed without MLIR_DIR or LLVM_DIR
    cmake --preset release: passed
    cmake --build --preset release --parallel 4: passed, 68 steps
    ctest --preset release --parallel 4: 1,688 passed, 2 skipped
    focused DDSIM device test: 37 passed
    nox tests-3.13: 574 passed, 3 skipped
    uv lock --check: passed
    uv build --sdist: passed
    clean structural Sphinx HTML: passed, no QIR output generated
    prek run --all-files with Python 3.13: passed
    git diff --check: passed

The v3 `noxfile.py` does not define the `cpp-lint` session named by `AGENTS.md`.
The local manual `clang-tidy` fallback could not resolve the macOS SDK standard
headers. The release build, focused DDSIM tests, `clang-format`, and all other
lint hooks passed; the reusable CI C++ linter remains the authoritative
changed-line check.

## Interfaces and Dependencies

No MLIR C++ interface, dialect, pass, conversion, tool, test target, CMake
option, or Python helper remains. No Core QIR runtime, JIT, runner, public
header, test, fixture, DDSIM execution path, or CMake option remains. The
project no longer discovers or links LLVM. Generic QDMI format identifiers and
payload classification remain because they are part of the upstream QDMI
interface and allow external devices to accept QIR programs.

Revision note (2026-09-01): Created the initial plan from the clean v3.9.2 tree
after the first cross-cutting inventory.

Revision note (2026-09-01): Recorded the completed removal, validation results,
and the Graphviz limitation after the final audit.

Revision note (2026-09-01): Extended the removal to the standalone QIR runner
and retained only the independent runtime and DDSIM QIR integration.

Revision note (2026-09-01): Superseded that boundary after the user requested
removal of the entire Core-owned QIR stack; retained only generic upstream QDMI
format awareness.
