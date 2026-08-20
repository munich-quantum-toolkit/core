# Consolidate shared MQT compiler infrastructure

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

The compiler collection currently stores MQT-owned passes and quantum-specific
helpers below the broad path `mlir/Dialect/Utils`. That path overlaps MLIR's own
shared dialect utilities and obscures which code owns each semantic contract.
After this change, MQT-owned cross-dialect transformations and quantum semantics
live below `mlir/Dialect/MQT`, MQT-owned constant-folding helpers live below
`mlir/Support/MQT`, and the quantum-aware module equivalence checker is built
only for tests. Existing pass command names and behavior remain unchanged.

The result is observable by building the same compiler and tests with no source
file below the project-owned `mlir/Dialect/Utils` directory. The generated MQT
documentation lists both the MQT metadata dialect and its cross-dialect passes.

## Progress

- [x] (2026-08-20 15:13Z) Create a dedicated worktree and branch from the
      current head of the symbolic Qiskit parameter pull request.
- [x] (2026-08-20 15:13Z) Audit the shared transform, utility, verifier, test,
      and build dependencies.
- [x] (2026-08-20 15:53Z) Move the existing MQT transform package and its
      normalization tests.
- [x] (2026-08-20 15:53Z) Split quantum semantics and generic folding helpers by
      owner.
- [x] (2026-08-20 15:53Z) Move the module equivalence checker into test support.
- [x] (2026-08-20 15:53Z) Update all includes, namespaces, CMake targets, and
      documentation.
- [x] (2026-08-20 15:53Z) Run focused tests, the release build, the
      documentation build, and lint.
- [x] (2026-08-20 15:53Z) Inspect the final diff and prepare signed, scoped
      local commits.
- [x] (2026-08-20 17:30Z) Rebase the two follow-up commits onto the merged
      symbolic Qiskit parameter change and repeat the publication checks.
- [x] (2026-08-20 19:30Z) Place MQT-owned folding support in `mlir::mqt` below
      `mlir/Support/MQT` and address all review comments.
- [x] (2026-08-20 19:30Z) Repeat the release and non-unity builds, focused and
      full tests, repository lint, and the full pull request Clang-Tidy diff.

## Surprises & Discoveries

- Observation: The legacy transform package is already MQT-owned except for its
  path. Its target is `MLIRMQTTransforms`, its generated pass group is `MQT`,
  and its C++ namespace is `mlir::mqt`. Evidence: the files
  `mlir/include/mlir/Dialect/MQT/Transforms/Passes.td` and
  `mlir/lib/Dialect/Utils/Transforms/CMakeLists.txt`.
- Observation: The project adds custom headers to MLIR's existing
  `mlir/Dialect/Utils` include hierarchy, which also supplies upstream headers
  such as `StaticValueUtils.h`. A clean split must remove only project-owned
  files and must continue using upstream headers from that path.
- Observation: `mlir/Support/IRVerification` is production-built but every
  caller is a unit test. The implementation directly depends on QC, QCO,
  QTensor, SCF, and LLVM IR details, so it is test infrastructure rather than a
  generic production verifier.
- Observation: Exposing support helpers through `mlir::mqt` revealed unqualified
  references to the repository's top-level `mqt::test` namespace in unit tests.
  Root-qualifying those references as `::mqt::test` removes the ambiguity and
  keeps the support helpers in their owning namespace.
- Observation: Pull request #2189 merged while this work was in progress and
  expanded the global-phase tests. The follow-up branch was rebased onto the
  updated pull request #2150 head, which already includes #2189. The moved test
  retains those semantic checks.
- Observation: The release preset does not build the Python MLIR bindings. The
  strict Sphinx build found stale Qiskit import and export includes after the
  utility split. The corrected binding target and the full Sphinx build now
  pass.

## Decision Log

- Decision: Keep `MLIRMQTDialect` and `MLIRMQTTransforms` as separate targets.
  Rationale: an IR dialect library must not pull pass infrastructure and
  cross-dialect rewrite dependencies into every metadata consumer. Date/Author:
  2026-08-20 / Codex.
- Decision: Move both `NormalizeGlobalPhases` and `UnrollModifiers` as one
  transform package. Rationale: both passes already share the MQT namespace,
  generated pass group, target, and QC/QCO ownership. Moving only one would
  retain an artificial package split. Date/Author: 2026-08-20 / Codex.
- Decision: Preserve the pass arguments, target names, and dependent dialect
  lists. Rationale: ownership does not change runtime behavior, and MLIR pass
  dependencies describe dialect entities that a pass may create. The MQT
  transforms create Arith, QC, and QCO entities but no MQT entities.
  Date/Author: 2026-08-20 / Codex.
- Decision: Do not introduce a shared QC/QCO unitary operation interface.
  Rationale: QC has reference semantics and QCO has value semantics. The user
  explicitly excluded that abstraction from this refactor. Date/Author:
  2026-08-20 / Codex.
- Decision: Split the former `Utils.h` by semantic responsibility instead of
  renaming the monolith. Rationale: a path move alone would preserve unclear
  ownership and excessive include dependencies. Date/Author: 2026-08-20 / Codex.
- Decision: Do not preserve forwarding headers below the old project-owned
  `mlir/Dialect/Utils` path. Rationale: the compiler collection and these APIs
  are part of the unreleased general launch, and the requested cleanup should
  remove the ambiguous project-owned include surface. Date/Author: 2026-08-20 /
  Codex.
- Decision: Put the dialect-independent constant-folding helpers below
  `mlir/Support/MQT` in `mlir::mqt`. Rationale: the include path describes the
  dependency layer, while the namespace records that MQT Core owns these
  helpers. Root-qualified `::mqt::test` references avoid ambiguity with the
  repository's separate top-level namespace. Date/Author: 2026-08-20 / Codex.
- Decision: Build the module equivalence checker in one concrete test-support
  library and expose that library through `MLIRTestCaseUtils`. Rationale: every
  caller is a unit test, while a single target avoids repeating its quantum
  dialect dependencies across test directories. Date/Author: 2026-08-20 / Codex.

## Outcomes & Retrospective

The follow-up now has explicit ownership boundaries. MQT owns the cross-dialect
passes, quantum semantics, and project-specific constant-folding support. Unit
test support owns the module equivalence checker. The implementation does not
add a shared QC/QCO unitary interface.

After pull request #2150 merged, the follow-up commits were rebased onto its
squash commit. The final review update passed the release build and the
non-unity debug build, including the Python MLIR binding. Five focused binaries
passed 322 tests. CTest reported 100% success across 4,301 configured tests; one
QDMI test was skipped by its own condition. The repository lint suite and the
full pull request Clang-Tidy diff passed. Generated MLIR documentation and the
strict Sphinx documentation build passed before the final review update, which
does not change generated documentation or Sphinx inputs. The signed commits are
ready for review.

## Context and Orientation

MQT Core embeds an MLIR-based compiler collection below `mlir/`. QC represents
mutable qubit references. QCO represents linear qubit values. The MQT dialect,
defined in `mlir/include/mlir/Dialect/MQT/IR/MQTDialect.td`, owns metadata and
other infrastructure shared across those representations.

The files below `mlir/include/mlir/Dialect/MQT/Transforms` and
`mlir/lib/Dialect/Utils/Transforms` define two module passes. Global-phase
normalization combines and moves `qc.gphase` and `qco.gphase` operations while
preserving modifier semantics. Modifier unrolling splits multi-operation QC and
QCO control, inverse, and power regions. Their library is already named
`MLIRMQTTransforms`.

The header `mlir/include/mlir/Dialect/Utils/Utils.h` mixes numeric helpers,
constant builders and folders, parameter validation, assembly parsing, and
modifier-region rewrites. `DenseUnitary.h` verifies the common dense-matrix
contract of QC and QCO unitary operations. `UGateUtils.h` implements shared
binary64 U-gate powering. These are project files in a path also owned by
upstream MLIR.

The function `areModulesEquivalentWithPermutations` is declared in
`mlir/include/mlir/Support/IRVerification.h`, implemented in
`mlir/lib/Support/IRVerification.cpp`, and called only by tests. The test root
already provides an interface target named `MLIRTestCaseUtils`; a new concrete
`MLIRTestSupport` target can carry the checker implementation without exposing
it through the installed `MLIRSupportMQT` library.

## Plan of Work

Move the transform headers, TableGen file, generated include locations, source
files, and CMake files to matching `Dialect/MQT/Transforms` directories. Update
all includes and parent CMake files. Move the global-phase normalization test to
`mlir/unittests/Dialect/MQT/Transforms`. Leave QC- and QCO-specific modifier
tests with their respective dialect tests.

Create small headers below `mlir/include/mlir/Dialect/MQT/Utils`. A math header
will own angle normalization, exponent checks, and the shared numeric tolerance.
A global-phase header will own the supported angle range and one verifier used
by both QC and QCO `GPhaseOp` implementations. A parameter header will own
constant construction, variant-to-value conversion, and finite parameter
validation. A modifier header will own common modifier parsing, printing,
building, and region-rewrite helpers. Dense-unitary verification and U-gate
powering will move into focused MQT utility headers without creating a shared
unitary operation interface.

Create `mlir/include/mlir/Support/MQT/ConstantFolding.h` for the generic
attribute and SSA constant-folding helpers. Move the corresponding tests to
`mlir/unittests/Support`. Keep the helpers in `mlir::mqt` to record project
ownership without placing dialect-independent code in a dialect library.

Move the quantum module equivalence header and implementation below
`mlir/unittests/Support`. Build them in a test-only `MLIRTestSupport` target and
make `MLIRTestCaseUtils` expose that target to configured test executables.
Remove the implementation and its test-only dependencies from `MLIRSupportMQT`.
Update every test include.

Remove the now-empty project-owned `Dialect/Utils` CMake subdirectories and
parent `add_subdirectory(Utils)` entries. Continue using upstream includes such
as `mlir/Dialect/Utils/StaticValueUtils.h`; those files are not part of this
repository and must not be changed.

Extend `docs/mlir/MQT.md` with the generated MQT pass reference. Do not add a
changelog entry because the change reorganizes unreleased compiler collection
internals without changing supported behavior or command-line interfaces.

## Concrete Steps

Run all commands from the repository root. Use `git status --short` before each
edit batch. Apply source edits with `apply_patch`; use ordinary `mv` only for
pure file relocation, followed by explicit content patches.

Configure and build the release preset:

    cmake --preset release
    cmake --build --preset release

During iteration, build and run the MQT transform, support, QC IR, QCO IR, and
compiler tests. The exact target names will be recorded after the CMake split.
Generate the pass and dialect documentation with:

    cmake --build --preset release --target mlir-doc
    uvx nox --non-interactive -s docs

Finish with:

    ctest --preset release
    uvx nox -s lint

Before each commit, inspect `git diff --check`, the staged diff, and the
complete commit message. Sign each commit and verify it with
`git verify-commit HEAD`. Do not push or create a pull request without separate
authorization.

## Validation and Acceptance

The repository contains no project-owned files below `mlir/Dialect/Utils`, and
no project source includes the removed custom paths. Includes of upstream MLIR
headers in that hierarchy remain valid.

The textual pass names `normalize-global-phases` and `unroll-modifiers` still
parse and run. Their focused QC and QCO tests pass with unchanged semantic
expectations. QC and QCO reject the same invalid global-phase values through one
shared helper, and both unitary interfaces apply one shared finite-parameter
validator. Dense unitary and U-gate power tests pass from their new include
locations. Constant-folding tests pass from the support test suite.

The production `MLIRSupportMQT` target no longer compiles or installs the
quantum module equivalence checker. All existing tests that compare modules
still link and pass through `MLIRTestSupport`.

The release build, configured CTest suite, generated MLIR documentation, full
Sphinx documentation, and repository lint suite complete successfully. The
worktree is clean after signed local commits, and no remote state changes are
made for this follow-up branch.

## Idempotence and Recovery

The include replacements, formatting, builds, and tests are repeatable. CMake
may retain stale generated include paths after the move; rerun
`cmake --preset release` before diagnosing missing generated files. If a move is
interrupted, use `git status --short` to identify source and destination files,
then complete the move without deleting unrelated work. Never reset the worktree
or discard changes from another task.
