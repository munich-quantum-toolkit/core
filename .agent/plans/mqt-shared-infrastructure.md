# Consolidate shared MQT compiler infrastructure

Status: historical implementation record.

## Goal and scope

The compiler collection currently stores MQT-owned passes and quantum-specific
helpers below the broad path `mlir/Dialect/Utils`. That path overlaps MLIR's own
shared dialect utilities and obscures which code owns each semantic contract.
After this change, MQT-owned cross-dialect transformations and IR-aware quantum
helpers live below `mlir/Dialect/MQT`, and the quantum-aware module equivalence
checker is built only for tests. The utility implementations form one leaf
`MLIRMQTUtils` library. Existing pass command names and behavior remain
unchanged.

The result is observable by building the same compiler and tests with no source
file below the project-owned `mlir/Dialect/Utils` directory. The generated MQT
documentation lists both the MQT metadata dialect and its cross-dialect passes.

## Constraints

- The legacy transform package is already MQT-owned except for its path. Its
  target is `MLIRMQTTransforms`, its generated pass group is `MQT`, and its C++
  namespace is `mlir::mqt`. Evidence: the files
  `mlir/include/mlir/Dialect/MQT/Transforms/Passes.td` and
  `mlir/lib/Dialect/Utils/Transforms/CMakeLists.txt`.

- The project adds custom headers to MLIR's existing `mlir/Dialect/Utils`
  include hierarchy, which also supplies upstream headers such as
  `StaticValueUtils.h`. A clean split must remove only project-owned files and
  must continue using upstream headers from that path.

- `mlir/Support/IRVerification` is production-built but every caller is a unit
  test. The implementation directly depends on QC, QCO, QTensor, SCF, and LLVM
  IR details, so it is test infrastructure rather than a generic production
  verifier.

- Exposing support helpers through `mlir::mqt` revealed unqualified references
  to the repository's top-level `mqt::test` namespace in unit tests.
  Root-qualifying those references as `::mqt::test` removes the ambiguity and
  keeps the support helpers in their owning namespace.

- The release preset does not build the Python MLIR bindings. The strict Sphinx
  build found stale Qiskit import and export includes after the utility split.
  The corrected binding target and the full Sphinx build now pass.

- Current upstream MLIR keeps `mlir/Support` independent of IR, dialect, and
  interface libraries. IR-aware shared helpers instead use dedicated utility
  libraries, such as `MLIRDialectUtils`, below the IR or dialect hierarchy.

- `add_mlir_dialect_library` appends its target to `MLIR_DIALECT_LIBS`.
  `MLIRMQTUtils` is not a dialect registration target and must remain below the
  MQT, QC, and QCO dialect libraries, so `add_mlir_library` describes its role
  and dependencies more accurately.

## Decisions

- Keep `MLIRMQTDialect` and `MLIRMQTTransforms` as separate targets. Rationale:
  an IR dialect library must not pull pass infrastructure and cross-dialect
  rewrite dependencies into every metadata consumer.

- Move both `NormalizeGlobalPhases` and `UnrollModifiers` as one transform
  package. Rationale: both passes already share the MQT namespace, generated
  pass group, target, and QC/QCO ownership. Moving only one would retain an
  artificial package split.

- Preserve the pass arguments, target names, and dependent dialect lists.
  Rationale: ownership does not change runtime behavior, and MLIR pass
  dependencies describe dialect entities that a pass may create. The MQT
  transforms create Arith, QC, and QCO entities but no MQT entities.

- Do not introduce a shared QC/QCO unitary operation interface. Rationale: QC
  has reference semantics and QCO has value semantics. The user explicitly
  excluded that abstraction from this refactor.

- Split the former `Utils.h` by semantic responsibility instead of renaming the
  monolith. Rationale: a path move alone would preserve unclear ownership and
  excessive include dependencies.

- Do not preserve forwarding headers below the old project-owned
  `mlir/Dialect/Utils` path. Rationale: the compiler collection and these APIs
  are part of the unreleased general launch, and the requested cleanup should
  remove the ambiguous project-owned include surface.

- Put all IR-aware shared helpers below `mlir/Dialect/MQT/Utils` in `mlir::mqt`,
  including constant folding. Rationale: the helpers depend on MLIR IR,
  dialects, or interfaces, so `mlir/Support` is the wrong dependency layer. A
  named owner and a dedicated library match current MLIR organization without
  creating loose headers or an umbrella include.

- Split the utility surface into `Angles`, `ConstantFolding`, `DenseUnitary`,
  `GatePowering`, `Modifiers`, and `Parameters`. Rationale: each header has one
  semantic responsibility and enough related declarations to justify the file.
  `Math.h` and an umbrella `Utils.h` would hide those contracts. Non-template
  implementations belong in matching source files.

- Build the helper package with `add_mlir_library(MLIRMQTUtils)` and link only
  `MLIRArithDialect`, `MLIRIR`, and `MLIRSideEffectInterfaces`. Rationale: the
  target is an IR-aware leaf library, not a dialect registration library. It
  must not link the MQT, QC, or QCO dialects because those dialects consume it.

- Build the module equivalence checker in one concrete test-support library and
  expose that library through `MLIRTestCaseUtils`. Rationale: every caller is a
  unit test, while a single target avoids repeating its quantum dialect
  dependencies across test directories.

## Outcome and validation

MQT owns cross-dialect passes, IR-aware quantum semantics, and project-specific
constant folding. Test support owns module equivalence. No shared QC/QCO unitary
interface was added.

The final utility package built all six source files independently and built the
release tree. Focused Clang-Tidy, 20 utility tests, the configured CTest suite
with one expected QDMI skip, lint, and diff checks passed. Generated MLIR and
strict Sphinx documentation passed before the final code-only update.

## Code and ownership

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

The MQT utility headers define angle and global-phase contracts, constant
folding, dense-unitary validation, gate powering, parameter handling, and
modifier-region rewrites. QC, QCO, builders, transforms, conversions, and the
Qiskit binding consume different subsets of those contracts. They need a shared
leaf target that does not depend on any consuming dialect.

The function `areModulesEquivalentWithPermutations` is declared in
`mlir/include/mlir/Support/IRVerification.h`, implemented in
`mlir/lib/Support/IRVerification.cpp`, and called only by tests. The test root
already provides an interface target named `MLIRTestCaseUtils`; a new concrete
`MLIRTestSupport` target can carry the checker implementation without exposing
it through the installed `MLIRSupportMQT` library.

## Acceptance

The repository contains no project-owned files below `mlir/Dialect/Utils`, and
no project source includes the removed custom paths. Includes of upstream MLIR
headers in that hierarchy remain valid.

The textual pass names `normalize-global-phases` and `unroll-modifiers` still
parse and run. Their focused QC and QCO tests pass with unchanged semantic
expectations. QC and QCO reject the same invalid global-phase values through one
shared helper, and both unitary interfaces apply one shared finite-parameter
validator. Dense-unitary and gate-power tests pass from their new include
locations. Constant-folding tests pass from the MQT utility test suite.

`MLIRMQTUtils` builds as an ordinary MLIR library. It links only MLIR Arith, IR,
and side-effect interfaces. It does not register as a dialect library and does
not link MQT, QC, or QCO. All direct consumers declare the utility target.

The production `MLIRSupportMQT` target no longer compiles or installs the
quantum module equivalence checker. All existing tests that compare modules
still link and pass through `MLIRTestSupport`.

Acceptance covers the release build, configured CTest suite, generated MLIR
documentation, strict Sphinx documentation, and repository lint.
