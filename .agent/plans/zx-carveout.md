# Move the ZX-calculus implementation from MQT Core to MQT QCEC

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core currently builds, installs, documents, and tests a general ZX-calculus
library. MQT QCEC is its only consumer. This change moves the implementation and
its tests into QCEC, where the equivalence checker can own its cancellation
behavior directly. MQT Core v4 then stops shipping the unused `MQT::CoreZX`
target and also stops carrying Boost.Multiprecision and GMP configuration that
exists only for ZX.

The result is visible in two repositories. QCEC continues to pass its ZX
equivalence-checking tests while depending on released MQT Core 3.8. MQT Core
configures, builds, installs, and documents without any ZX,
Boost.Multiprecision, or GMP artifact.

## Progress

- [x] (2026-08-13 10:00Z) Refreshed both repositories and created isolated task
      branches from current `main`.
- [x] (2026-08-13 10:40Z) Imported the exact MQT Core ZX implementation and
      complete test suite into QCEC under `ec::zx`.
- [x] (2026-08-13 11:05Z) Replaced QCEC's duplicate simplification driver with
      cancellation-aware shared routines.
- [x] (2026-08-13 11:25Z) Gave QCEC ownership of the Boost.Multiprecision
      dependency and removed its `MQT::CoreZX` link and packaging exclusions.
- [x] (2026-08-13 11:50Z) Removed ZX and its exclusive dependency/configuration
      surface from MQT Core.
- [x] (2026-08-13 14:20Z) Validated focused behavior, full debug/release C++
      suites, Python packaging, documentation, installation, and Core/QCEC
      integration.
- [x] (2026-08-13 14:30Z) Ran an adversarial internal review and corrected stale
      Core metadata and composite simplification counting.

## Surprises & Discoveries

- Observation: QCEC duplicates the Core simplification driver only to poll the
  checker's `isDone()` state. Evidence: `include/checker/zx/ZXChecker.hpp` and
  `src/checker/zx/ZXChecker.cpp` repeat the loops from Core's `zx/Simplify.hpp`
  and `zx/Simplify.cpp`.
- Observation: Boost.Multiprecision, the `MQT::Multiprecision` target, GMP, and
  the related package-config values are exclusive to Core ZX. Evidence:
  repository-wide searches find production references only in `src/zx`,
  `cmake/ExternalDependencies.cmake`, `cmake/mqt-core-config.cmake.in`, and the
  corresponding README text.
- Observation: The unreleased changelog still described a ZX optimization that
  will not ship in v4 after this removal. Evidence: the `#1984` entry appeared
  under `[Unreleased]`, not in a released historical section.
- Observation: Core's package keywords and release-drafter categories still
  advertised ZX after the code removal. Evidence: `pyproject.toml`,
  `.github/release-drafter.yml`, and
  `.github/workflow_inputs/release_drafter_categories.json` retained ZX entries
  until the internal review.
- Observation: The imported composite simplifiers did not count rewrites from
  the initial spider pass and counted later iterations rather than rewrites.
  Evidence: a three-spider regression simplified the diagram but originally
  reported zero; QCEC now counts every completed rewrite, including work
  completed before cancellation.

## Decision Log

- Decision: Import from Core commit `e56bab0360cac0c3a57db0730a253e97d5fb65c6`.
  Rationale: This was current `origin/main` when implementation started and
  gives the transfer an exact provenance boundary. Date/Author: 2026-08-13,
  Codex.
- Decision: Put imported symbols in `ec::zx` and compile them into `MQT::QCEC`.
  Rationale: QCEC still consumes Core 3.8, which exports global `zx` symbols; a
  distinct namespace prevents source and linker collisions without creating
  another public library. Date/Author: 2026-08-13, user and Codex.
- Decision: Keep only Boost.Multiprecision's `cpp_rational` backend in QCEC.
  Rationale: QCEC does not need Core's optional GMP path, and removing that path
  reduces the maintenance surface. Date/Author: 2026-08-13, user and Codex.
- Decision: Keep QCEC's Core dependency at `~=3.8.0`. Rationale: Core v4 is not
  released, so compatibility declarations for it would be speculative.
  Date/Author: 2026-08-13, user.
- Decision: Do not publish branches or create pull requests as part of local
  implementation. Rationale: The task plan explicitly keeps external actions
  separately authorized. Date/Author: 2026-08-13, Codex.
- Decision: Remove the unreleased `#1984` ZX optimization entry while retaining
  all released ZX history. Rationale: The v4 changelog should describe the final
  released surface rather than a feature removed before release. Date/Author:
  2026-08-13, Codex.

## Outcomes & Retrospective

The local Core branch removes the complete ZX implementation and all exclusive
Boost/GMP integration. Clean debug and release builds each passed 4,377 tests,
lint and HTML documentation passed, and an install smoke test contained no ZX,
Boost.Multiprecision, GMP, or stale CMake artifact. Link checking reached the
documentation successfully but retained only pre-existing external 403/404 and
stale upstream-link failures.

The local QCEC branch imports the implementation from Core commit
`e56bab0360cac0c3a57db0730a253e97d5fb65c6`, isolates it in `ec::zx`, and keeps
the released Core 3.8 dependency declarations unchanged. Debug and release each
passed 510 C++ tests with Core 3.8; the Core-removal integration build passed
509 tests because the conditional coexistence test correctly disappears. The
normal Python suite passed 58 tests with 12 optional tests skipped. The wheel
retains `mqt.core~=3.8.0` and has no CoreZX, Boost, or GMP dynamic dependency.
The optional profile-regeneration run has six unrelated failures because the
environment resolved Qiskit 2.5.1 while the checked profiles record 2.5.0.

The adversarial review removed stale Core packaging/release metadata and added a
regression ensuring composite ZX simplification counts include every applied
rewrite. No branches were pushed and no pull requests were created.

## Context and Orientation

MQT Core's ZX public headers are in `include/mqt-core/zx`, implementations are
in `src/zx`, tests are in `test/zx`, and user documentation is in
`docs/zx_package.md`. The CMake target `mqt-core-zx`, exported as `MQT::CoreZX`,
owns a helper `MQT::Multiprecision` target and optional GMP support. Core's
external dependency and installed package configuration also carry Boost and GMP
settings solely for this target.

MQT QCEC currently has `include/checker/zx/ZXChecker.hpp` and
`src/checker/zx/ZXChecker.cpp`. The checker constructs Core `zx::ZXDiagram`
objects and duplicates Core's simplification traversal so it can stop when the
equivalence-checking manager signals cancellation. QCEC links `MQT::CoreZX` and
excludes the corresponding Core shared library during wheel repair.

The transferred implementation is internal to QCEC. Its headers live below
`include/checker/zx` because QCEC sources and tests need them, but QCEC does not
install or promise a standalone ZX API or target.

## Plan of Work

First copy the seven ZX headers, six implementation files, and five test files
from the recorded Core revision into QCEC. Rewrite their local includes to
`checker/zx/...` and their namespace to `ec::zx`. Integrate the sources into the
existing QCEC target and the tests into the existing test executable. Add a
small coexistence test that includes both the Core 3.8 and QCEC diagram types.

Next extend QCEC's imported `Simplify` interface with an optional cancellation
predicate. Each outer pass and each individual rewrite checks the predicate. The
no-argument default never cancels, so the transferred Core tests retain their
exact behavior and simplification counts. Make `ZXEquivalenceChecker` call these
routines with `isDone()` and remove its duplicate traversal and driver methods.

Then configure Boost.Multiprecision in QCEC. A system-Boost mode links the Boost
headers; the default mode fetches the same pinned standalone
Boost.Multiprecision source used by Core. Link this internal dependency through
`MQT::QCEC`, remove `MQT::CoreZX`, and remove the obsolete wheel exclusions.
Keep all MQT Core version requirements and the lockfile unchanged.

Finally delete Core's ZX trees and registrations. Remove the exclusive Boost and
GMP discovery, cache options, installed-package configuration, CMake module,
wheel target, README statements, and documentation page. Add concise v4
changelog and upgrade-guide entries in both repositories where applicable.

## Concrete Steps

From each repository root, inspect the branch and status before each batch with
`git branch --show-current` and `git status --short`. Build QCEC first with
`cmake --preset debug`, `cmake --build --preset debug`, and the focused ZX test
filter in `build/debug/test/mqt-qcec-test`. Then run `ctest --preset debug` and
repeat in release mode.

Run Core cache-producing commands through `.agent/run.sh`. Configure and build
with `./.agent/run.sh cmake --preset debug` and
`./.agent/run.sh cmake --build --preset debug`, then use
`./.agent/run.sh ctest --preset debug`. Repeat with `release`. Build docs with
`./.agent/run.sh uvx nox --non-interactive -s docs` and check links with
`./.agent/run.sh uvx nox -s docs -- -b linkcheck`.

Run each repository's required `uvx nox -s lint` last. Inspect
`git diff --check`, the complete diff, and status after every formatter-induced
change.

## Validation and Acceptance

QCEC acceptance requires every transferred Core ZX test and every existing QCEC
ZX-checker test to pass. Cancellation tests must show that a predicate can stop
a traversal before any rewrite and during a traversal without corrupting the
diagram. A coexistence test must construct both `::zx::ZXDiagram` from Core 3.8
and `ec::zx::ZXDiagram` from QCEC in one executable. QCEC's final CMake and
wheel metadata must contain no `CoreZX` or `mqt-core-zx` reference, while its
Core dependency declarations remain at 3.8.

Core acceptance requires a clean configure and build with no ZX target. Its
install and package metadata must contain no ZX, Boost.Multiprecision, GMP,
`MQT_CORE_WITH_GMP`, or `MQT_CORE_ZX_SYSTEM_BOOST` reference. Full C++ tests,
documentation, link checks, and lint must pass. Historical changelog entries
remain intact.

Cross-repository acceptance requires a clean QCEC source build using the Core
removal worktree through `FETCHCONTENT_SOURCE_DIR_MQT-CORE`. This build is only
integration evidence. QCEC's final committed dependency remains released Core
3.8.

## Idempotence and Recovery

All configure, build, test, documentation, and lint commands are repeatable.
Build outputs remain in worktree-local ignored directories. The two task
branches and worktrees isolate the changes from `base` and all other tasks. If a
mechanical import must be repeated, restore only the imported files from the
recorded Core commit and reapply the namespace rewrite; never reset or clean an
unrelated worktree. External publication requires separate authorization.
