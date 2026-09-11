# Move the ZX-calculus implementation from MQT Core to MQT QCEC

Status: historical implementation record.

## Goal and scope

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

## Constraints

- QCEC duplicates the Core simplification driver only to poll the checker's
  `isDone()` state. Evidence: `include/checker/zx/ZXChecker.hpp` and
  `src/checker/zx/ZXChecker.cpp` repeat the loops from Core's `zx/Simplify.hpp`
  and `zx/Simplify.cpp`.

- Boost.Multiprecision, the `MQT::Multiprecision` target, GMP, and the related
  package-config values are exclusive to Core ZX. Evidence: repository-wide
  searches find production references only in `src/zx`,
  `cmake/ExternalDependencies.cmake`, `cmake/mqt-core-config.cmake.in`, and the
  corresponding README text.

- The unreleased changelog still described a ZX optimization that will not ship
  in v4 after this removal. Evidence: the `#1984` entry appeared under
  `[Unreleased]`, not in a released historical section.

- Core's package keywords and release-drafter categories still advertised ZX
  after the code removal. Evidence: `pyproject.toml`,
  `.github/release-drafter.yml`, and
  `.github/workflow_inputs/release_drafter_categories.json` retained ZX entries
  until the internal review.

- The imported composite simplifiers did not count rewrites from the initial
  spider pass and counted later iterations rather than rewrites. Evidence: a
  three-spider regression simplified the diagram but originally reported zero;
  QCEC now counts every completed rewrite, including work completed before
  cancellation.

## Decisions

- Import from Core commit `e56bab0360cac0c3a57db0730a253e97d5fb65c6`. Rationale:
  This was current `origin/main` when implementation started and gives the
  transfer an exact provenance boundary.

- Put imported symbols in `ec::zx` and compile them into `MQT::QCEC`. Rationale:
  QCEC still consumes Core 3.8, which exports global `zx` symbols; a distinct
  namespace prevents source and linker collisions without creating another
  public library.

- Keep only Boost.Multiprecision's `cpp_rational` backend in QCEC. Rationale:
  QCEC does not need Core's optional GMP path, and removing that path reduces
  the maintenance surface.

- Keep QCEC's Core dependency at `~=3.8.0`. Rationale: Core v4 is not released,
  so compatibility declarations for it would be speculative.

## Outcome and validation

Core removes ZX and its exclusive Boost/GMP integration. QCEC owns the
transferred implementation in `ec::zx`, with provenance at Core commit
`e56bab0360cac0c3a57db0730a253e97d5fb65c6`, and retains its released Core 3.8
dependency.

Core debug/release, packaging, documentation, and lint checks passed. QCEC
passed debug/release against Core 3.8, an integration build against the removal
tree, and its normal Python suite. External link checks and optional Qiskit-
version-sensitive profile regeneration did not fully pass. A regression
preserves composite simplification counts. Hosted publication was outside the
recorded work.

## Code and ownership

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

## Acceptance

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
