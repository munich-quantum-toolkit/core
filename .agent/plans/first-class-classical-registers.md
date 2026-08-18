# Replace implicit classical-register memrefs with CBit IR

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core currently represents a source-language classical-bit register as a
generic one-dimensional `memref` whose elements have type `i1`. A `memref` says
how to store bits, but it does not say that the storage is a classical register,
whether its initial value is zero or undefined, or whether the register is a
public program result. Qiskit, OpenQASM, QIR, jeff, and the decision-diagram
executor therefore infer those facts independently from operation order,
attributes, and placeholder values.

After this change, QC and QCO programs use a shared CBit dialect for classical
registers. The textual IR makes initialization and public outputs explicit. A
user can import a Qiskit or OpenQASM program, inspect `!cbit.reg<N>`,
`cbit.alloc`, `cbit.load`, and `cbit.store`, convert between QC and QCO without
changing those operations, export the result, execute QCO, lower to QIR or jeff,
or choose an explicit late lowering to `memref`. Tests demonstrate that a Qiskit
condition can read a zero before the first measurement, OpenQASM 3 no longer
needs poison placeholders, and internal CBit registers do not become public
outputs.

## Progress

- [x] (2026-08-18 20:31Z) Confirmed that the compatibility fix from pull request
      #2140 is the current `main` revision and opened architectural issue #2155
      with the agreed contract.
- [x] (2026-08-18 20:35Z) Created a clean implementation branch from that `main`
      revision and added this ExecPlan before production edits.
- [x] (2026-08-18 21:50Z) Added and registered the CBit dialect, its generated
      and narrative documentation, and six focused IR tests for syntax, types,
      effects, initialization, and constant bounds.
- [x] (2026-08-18 21:12Z) Replaced the QC and QCO builder-wide memref policy
      with per-allocation CBit APIs. Both focused builder suites pass six tests,
      including independent zero and undefined registers.
- [x] (2026-08-18 21:12Z) Added `convert-cbit-to-memref`, preserved CBit through
      QC-to-QCO and QCO-to-QC, and proved operation, function, call, return,
      name, initialization, and output conversion behavior.
- [x] (2026-08-18 21:14Z) Committed and published the compiling dialect,
      builder, and conversion foundation as draft pull request #2158. Commit
      `fe432ee83` is signed and passes `git verify-commit`.
- [x] (2026-08-18 21:50Z) Migrated Qiskit import and export to returned CBit
      registers and removed the temporary memref and UB recognizers. All 119
      focused Python translation tests pass.
- [x] (2026-08-18 21:50Z) Migrated OpenQASM import and export to CBit and
      removed parallel bit SSA state and classical poison placeholders. All 173
      focused translation tests pass.
- [x] (2026-08-18 21:50Z) Migrated QIR lowering, QCO decision-diagram execution,
      and both jeff conversion directions. The focused QIR, QCO utility, and
      jeff suites pass.
- [x] (2026-08-18 23:45Z) Completed cross-component tests, documentation,
      changelog, and generated stubs. The full native suite passes 4,288 tests.
      The required Python, stub, documentation, and Python-version matrix
      sessions pass.
- [x] (2026-08-18 23:31Z) Validated published integration commit `07df0c738`.
      Every platform and coverage job passed, but extensive C++ lint found
      indirect CBit includes plus helper linkage and pass visibility issues.
      Added direct provider includes and fixed those diagnostics for a signed
      follow-up commit.
- [x] (2026-08-18 23:47Z) The remote Clang 22.1.8 lint run on follow-up
      `56c65d078` reduced the result to three test-context include diagnostics.
      Fixed each reported include and reran the three affected test binaries and
      repository lint successfully.
- [ ] Publish and validate the signed final head. Keep the pull request in draft
      until link checking passes: the required link-check session currently
      fails only on unrelated external DOI, Qiskit policy, and old Read the Docs
      links.

## Surprises & Discoveries

- Observation: The fix merged in #2140 had to teach Qiskit export that an
  allocation followed by one false store per element is initialization rather
  than classical execution. Evidence: `bindings/mlir/qiskit/QiskitExport.cpp`
  contains `initialClassicalZeroStoreIndex` and a block-order recognizer.
- Observation: Qiskit import needed a builder-wide initialization policy only
  because the register type did not carry initialization. Evidence:
  `bindings/mlir/qiskit/QiskitImport.cpp` constructs `QCProgramBuilder` with
  `ClassicalRegisterInitialization::Zero`.
- Observation: QIR and OpenQASM contain separate reconstructions of the same
  missing semantics. Evidence:
  `mlir/lib/Conversion/QCToQIR/QIRCommon/QIRCommon.cpp` scans result memrefs,
  while `mlir/lib/Dialect/QC/Translation/OpenQASMToQCEmitter.cpp` tracks both
  SSA bit values and classical-register memrefs.
- Observation: This LLVM 22 TableGen setup requires explicit enum and attribute
  generation in addition to operation and type generation. Generated enum
  declarations also require the MLIR operation-implementation headers before
  inclusion. Evidence: the first focused build failed before
  `CBitOpsEnums.h.inc` and `CBitOpsAttributes.h.inc` were added to
  `MLIRCBitOpsIncGen`; the corrected target builds all ten generated fragments.
- Observation: `register` is a C++ keyword and cannot be an ODS operand name
  because TableGen uses operand names in generated builder parameters. Evidence:
  the generated C++ failed until the operand identifier became `reg`; the
  user-facing operand description remains "register".
- Observation: MLIR verifies custom operations while parsing. A textual module
  with a constant out-of-bounds CBit index therefore fails to parse instead of
  returning an invalid module for a separate `verify` call. Evidence: the
  focused test now asserts parse failure and receives the
  `index exceeds register width` diagnostic.
- Observation: Generated dialect and enum declarations require their MLIR and
  LLVM support headers to precede the generated includes. The repository's
  include sorter groups quoted includes first unless a section comment separates
  the generated include. Evidence: a clean focused rebuild failed in the
  generated headers until the CBit wrappers added explicit dependencies and a
  generated-code section; the rebuilt targets then passed.
- Observation: Qiskit export must validate CBit stores before walking the
  instruction stream. A dynamic index expression appears before its store and
  would otherwise fail first as an unsupported scalar operation. Evidence: the
  exporter now inventories direct measurement stores and gives destination
  errors before it constructs instructions.
- Observation: Dialect conversion adapts a CBit operand to its tensor value
  before the QCO-to-jeff load or store pattern runs. The state map must retain
  the original CBit identity for that adapted value. Evidence: mapping each
  converted allocation result back to its source register fixed structured and
  dynamic-index jeff round trips; all 142 focused tests pass.
- Observation: OpenQASM no longer needs to carry bit registers as SCF iterated
  values. CBit operations have explicit memory effects, and nested regions can
  capture the same non-aliasing register. Evidence: removing bit vectors from
  the frontend state preserved dynamic indexing and structured control across
  all 173 translation tests.
- Observation: QIR Adaptive needs ordinary register storage for internal CBit
  allocations, while returned CBit allocations use QIR result arrays. Evidence:
  the final lowering distinguishes these cases, supports dynamic internal
  indices, and passes all 146 Adaptive tests.
- Observation: CBit loads must become SSA values before conversion pipelines can
  lower structured conditions. Evidence: a conservative same-block canonicalizer
  forwards a known prior store or zero initialization and stops at dynamic
  aliasing, nested regions, or unknown register users. The compiler and CBit
  tests cover both forwarding and unsafe cases.
- Observation: Combining a measurement with its only CBit store can move the
  measurement past intervening quantum operations if the store is not adjacent.
  Evidence: OpenQASM emission now combines only adjacent operations, and a
  regression test preserves the order of a delayed store.
- Observation: Unity builds hid indirect dependencies on the generated CBit
  enum, type, and dialect declarations. Evidence: the extensive C++ lint job
  found 87 diagnostics on the published head even though the release build and
  local lint session passed. Direct includes and explicit helper linkage remove
  the branch-related diagnostics, and all focused binaries still pass.

## Decision Log

- Decision: Add CBit as a neutral dialect, not as part of QC or QCO. Rationale:
  The same register must pass unchanged between reference-semantics QC and
  value-semantics QCO, and a neutral dialect avoids conversion-owned copies.
  Date/Author: 2026-08-18 / Codex, based on the approved design.
- Decision: Model a register as `!cbit.reg<N>` with a static positive width and
  no aliasing operations. Rationale: Every current frontend has a known register
  width, and excluding casts, subviews, aliases, and deallocation makes output
  identity and effect analysis precise. Date/Author: 2026-08-18 / Codex, based
  on the approved design.
- Decision: Put zero or undefined initialization on each `cbit.alloc`.
  Rationale: Qiskit and OpenQASM 2 require zero, OpenQASM 3 permits undefined
  bits, and a builder-wide policy cannot represent both in one program.
  Date/Author: 2026-08-18 / Codex, based on the approved design.
- Decision: Define public result registers only by return values of the entry
  function. Rationale: Allocation and a source name are insufficient to
  distinguish user-visible output from internal state. Date/Author: 2026-08-18 /
  Codex, based on the approved design.
- Decision: Do not retain a reverse recognizer for arbitrary `memref<Nxi1>`
  programs. Rationale: Compatibility inference recreates the ambiguity that CBit
  removes. Clients that require memref use the explicit one-way late conversion.
  Date/Author: 2026-08-18 / Codex, based on the approved design.
- Decision: Do not add `UPGRADING.md` instructions. Rationale: The affected
  Qiskit, OpenQASM, builder, and classical-register interfaces were added after
  v3.8.0 and are not part of a released compatibility contract. Date/Author:
  2026-08-18 / Codex, based on the approved design.

## Outcomes & Retrospective

The compatibility fix is merged and issue #2155 records the replacement
contract. Draft pull request #2158 contains the neutral dialect, per-allocation
builder APIs, identity preservation across QC and QCO, the explicit late memref
lowering, and all planned producer and consumer migrations. The focused suites,
the complete 4,288-test native suite, the required Python tests, all supported
Python-version matrices, generated stubs, and documentation build pass. The
link-check session fails on pre-existing external links that this change does
not touch. The pull request remains a draft until that required check passes.
Publishing and remote validation of the signed final head remain.

## Context and Orientation

MLIR is the compiler infrastructure used under `mlir/`. A dialect is a named set
of MLIR types, attributes, and operations. QC is the reference-semantics quantum
dialect in `mlir/include/mlir/Dialect/QC` and `mlir/lib/Dialect/QC`. QCO is the
value-semantics optimization dialect in the matching QCO directories. QTensor is
an existing neutral dialect whose TableGen, CMake, documentation, and unit-test
layout provides a repository-local example for adding CBit.

The new dialect belongs under `mlir/include/mlir/Dialect/CBit/IR` and
`mlir/lib/Dialect/CBit/IR`. Its public type is `!cbit.reg<N>`, where `N` is a
strictly positive compile-time integer. `#cbit.init<zero>` means every element
starts false. `#cbit.init<undefined>` means reading an element before a store is
undefined behavior. `cbit.alloc` creates the register and accepts an optional
source-level name. `cbit.load` reads an `i1` at an `index`. `cbit.store` writes
an `i1` at an `index`. Allocation, load, and store report explicit allocation,
read, and write effects through MLIR's memory-effect interface. Constant
negative and out-of-bounds indices fail verification. There is no deallocation,
cast, subview, or alias operation.

The QC and QCO builders are declared in
`mlir/include/mlir/Dialect/QC/Builder/QCProgramBuilder.h` and
`mlir/include/mlir/Dialect/QCO/Builder/QCOProgramBuilder.h`, with definitions in
the corresponding `mlir/lib` paths. They currently allocate `memref<Nxi1>` and
QC owns a builder-wide `ClassicalRegisterInitialization` enum. Both builders
must instead expose the same per-register operations:

    Value allocClassicalBitRegister(
        int64_t size, StringRef name,
        cbit::Initialization initialization);
    Value loadClassicalBit(Value reg,
                           const std::variant<int64_t, Value>& index);
    void storeClassicalBit(Value value, Value reg,
                           const std::variant<int64_t, Value>& index);

Use the final generated enum name if TableGen produces a more precise spelling,
but keep the public behavior and call-site order `size, name, initialization`.
Measurement helpers continue to return `i1` and call `storeClassicalBit` when a
destination is supplied. Conditional builder helpers use `loadClassicalBit`.
Remove the overloaded QC constructor, overloaded static `build` helpers, and the
builder member that apply one policy to every allocation.

Qiskit conversion lives in `bindings/mlir/qiskit/QiskitImport.cpp` and
`bindings/mlir/qiskit/QiskitExport.cpp`. Import must allocate Qiskit classical
registers with zero initialization, load conditions through CBit, store
measurements through CBit, and return all Qiskit-visible registers. Export must
discover output registers only from the entry function return, not from all
allocations. It accepts static direct stores of measurement results,
zero-initialized elements that are never written, and undefined registers only
when every returned element is written. It rejects dynamic output destinations,
loads or classical control flow, non-measurement stores, a measurement written
more than once, and returned undefined elements. Delete the leading-zero memref
recognizer and the UB dialect dependency.

OpenQASM analysis and emission are implemented in
`mlir/lib/Dialect/QC/Translation/OpenQASMToQCEmitter.cpp`; QC-to-OpenQASM output
is implemented in `mlir/lib/Dialect/QC/Translation/TranslateQCToOpenQASM3.cpp`.
OpenQASM 2 allocates zero-initialized CBit registers. OpenQASM 3 allocates
undefined CBit registers. Loads and stores must replace the parallel `bitValues`
and result-memref state through structured control flow and dynamic indices.
Remove per-element poison placeholders. Export declarations and assignments from
returned CBit registers and CBit operations without scanning generic memrefs.

QC-to-QCO and QCO-to-QC live in `mlir/lib/Conversion/QCToQCO/QCToQCO.cpp` and
`mlir/lib/Conversion/QCOToQC/QCOToQC.cpp`. Their conversion targets must mark
CBit legal and their type converters must preserve `!cbit.reg<N>` exactly. QIR
result handling is shared in
`mlir/lib/Conversion/QCToQIR/QIRCommon/QIRCommon.cpp`; replace its generic
classical memref scan and leading-zero recognition with direct CBit handling.
The Base profile retains static-index restrictions. The Adaptive profile keeps
dynamic-index behavior. Unsupported non-measurement output stores must name the
unsupported operation in a precise diagnostic.

QCO decision-diagram execution is in
`mlir/lib/Dialect/QCO/Utils/DDFunctionality.cpp`. Add register storage plus an
initialization mask so conditions and measurement updates follow CBit state and
an undefined read fails instead of inventing a value. jeff conversions are in
`mlir/lib/Conversion/QCOToJeff/QCOToJeff.cpp` and
`mlir/lib/Conversion/JeffToQCO/JeffToQCO.cpp`; convert CBit directly to the jeff
integer-array representation and back. An undefined CBit allocation may use any
jeff initial value because reading it before a store is outside the CBit
contract.

Add the explicit one-way conversion under
`mlir/include/mlir/Conversion/CBitToMemRef` and
`mlir/lib/Conversion/CBitToMemRef`, with tests under
`mlir/unittests/Conversion/CBitToMemRef`. It converts CBit types in function
signatures and return operations as well as alloc, load, and store operations.
Zero allocation emits one false store for each element; undefined allocation
emits only `memref.alloc`. Do not infer CBit from existing memrefs.

Register CBit in compiler programs, the `mqt-cc` tool, Python bindings, and all
test contexts that parse or build programs containing it. Update
`docs/mlir/CBit.md`, `docs/mlir/index.md`, `docs/mlir/Conversions.md`, and
`docs/mlir/OpenQASM.md`. Generate the dialect reference through CMake rather
than editing generated output. Add an unreleased `CHANGELOG.md` entry after the
draft pull request number exists, with every contributing author and the PR link
definition.

## Plan of Work

First, add the CBit TableGen definitions, generated targets, C++ dialect
initialization, operation verifiers, effects, and focused parsing and verifier
tests. Wire the new library through the dialect CMake trees and the main
registries. A textual module containing one zero-initialized register, a store,
a load, and a returned register must parse, print, and verify.

Second, migrate QCProgramBuilder and QCOProgramBuilder to the per-allocation
API. Replace every in-tree builder call so each allocation states zero or
undefined. Add builder tests with two registers that use different
initialization in the same module. At this point no builder-global policy or
generic classical-register-name memref attribute remains.

Third, add the late CBit-to-memref conversion and mark CBit legal in both QC and
QCO conversions. Test type, signature, return, initialization, load, store,
name, and output preservation. Build this foundation and its focused tests,
commit it with a signed commit, push it, and open the required draft pull
request. Record the draft number before adding the changelog reference in a
separate signed commit.

Fourth, migrate the two frontend pairs. Start with Qiskit because its importer
and exporter have narrow Python tests. Then change OpenQASM import and export,
including structured control flow and dynamic indices. Remove compatibility
logic only after equivalent CBit tests pass.

Fifth, migrate QIR, decision-diagram execution, and jeff. Run the smallest test
binary after each subsystem. Keep CBit legal until the consumer that owns the
next representation converts it explicitly.

Finally, finish documentation and the changelog, regenerate stubs, run every
focused and full command below, inspect the entire diff, verify every commit,
publish the exact validated head, update the ExecPlan outcome, mark the pull
request ready, and request human review.

## Concrete Steps

Run all commands from the repository root. During development, configure and
build the release tree as needed:

    cmake --preset release
    cmake --build --preset release --target <focused-target>

The first milestone runs the new CBit IR test binary and builder tests. The
exact binary name must follow the established unit-test convention and be
recorded here after CMake defines it. The conversion milestone runs the CBit
conversion binary plus the existing QC-to-QCO, QCO-to-QC, and round-trip
binaries. The consumer milestones run the QIR Base and Adaptive binaries, the
QCO utility binary that owns decision-diagram execution, the jeff round-trip
binary, and the QC translation binary.

The required Python checks are:

    uv run --no-sync pytest test/python/test_mlir_qiskit_translation.py test/python/test_mlir.py
    uvx nox -s stubs
    uvx nox -s tests
    uvx nox -s minimums

The required build, C++, documentation, and lint checks are:

    cmake --build --preset release --target mlir-doc
    cmake --build --preset release
    ctest --preset release
    uvx nox -s docs
    uvx nox -s docs -- -b linkcheck
    uvx nox -s lint

Record each command, result, and test count in `Progress` or
`Artifacts and Notes`. If a command fails because of the environment, preserve
the exact failure, repair only an in-scope environment problem, and rerun it.
Source failures must be fixed before publication.

## Validation and Acceptance

Parsing and printing a program must preserve `!cbit.reg<N>`, both initialization
attributes, an optional source name, `cbit.load`, and `cbit.store`. Width zero,
negative or too-large constant indices, non-`index` indices, non-`i1` stored
values, and mismatched register operands must fail verification. Effect queries
must report allocation on `cbit.alloc`, read on `cbit.load`, and write on
`cbit.store`.

A single builder invocation must allocate one zero-initialized and one undefined
register. QC-to-QCO-to-QC and QCO-to-QC-to-QCO must preserve both values, their
names and initialization, all loads and stores, and entry-function returns. Only
returned registers appear in Qiskit or OpenQASM output.

Qiskit tests must cover zero-before-measurement conditions, measurement round
trips, anonymous clbits, exclusion of internal registers, undefined returned
bits, duplicate destinations, dynamic destinations, and unsupported classical
execution. OpenQASM tests must show zero initialization for version 2, undefined
initialization for version 3, no `ub.poison`, assignments, dynamic indices, and
structured control flow.

QIR tests must cover Base and Adaptive output mapping, repeated measurements,
unsupported stores, and profile-specific dynamic-index rules. QCO execution
tests must show conditions reading stored bits and measurements updating CBit
state. jeff tests must round-trip supported registers. The late conversion must
show zero stores only for zero initialization and must convert signatures and
returned registers.

Final source inspection must find no leading-zero memref recognizer in Qiskit or
QIR, no UB dialect dependency in Qiskit export, no classical-output poison
placeholder in OpenQASM, and no use of the generic classical-register-name
memref convention. Every commit in the pull-request range must pass
`git verify-commit`. Every required local check and every remote check must pass
on the exact submitted head before the pull request is marked ready.

## Idempotence and Recovery

Configuration, builds, tests, documentation generation, and lint commands are
repeatable. Generated stubs and documentation must come from their generators;
never edit generated files by hand. Before each commit, inspect `git status`,
stage explicit paths, inspect the staged diff, create a signed commit, and run
`git verify-commit HEAD`.

Preserve unrelated changes. If `main` advances before publication, fetch and
rebase signed commits only after making a backup ref and recording the old
remote head. Use an exact `--force-with-lease` only when a published branch must
be rewritten. Do not use destructive reset or checkout commands. A failed
conversion can be retried after fixing the owning pattern because the source
CBit operations remain explicit and the conversion is one-way.

## Artifacts and Notes

Initial repository evidence:

    origin/main ed1d6e3f9 ✨ Export OpenQASM measurements to Qiskit (#2140)
    issue #2155       ✨ Add a first-class classical-bit register dialect
    draft PR #2158    ♻️ Replace implicit classical-register memrefs with CBit IR

Final local validation evidence:

    cmake --build --preset release --target mlir-doc             passed
    cmake --build --preset release                               passed
    ctest --preset release                                       4,288 passed, 1 skipped
    uv run --no-sync pytest test/python/test_mlir_qiskit_translation.py test/python/test_mlir.py
                                                                  167 passed
    uvx nox -s stubs                                             passed
    uvx nox -s tests                                             passed on Python 3.10-3.14
    uvx nox -s minimums                                          passed on Python 3.10-3.14
    uvx nox -s docs                                              passed
    uvx nox -s docs -- -b linkcheck                              failed on unrelated external links

The focused CBit, CBit-to-memref, QC-to-QCO, QCO-to-QC, QC/QCO round-trip, QIR
Base, QIR Adaptive, QCO utility, jeff, QC translation, compiler, QCO IR, and
OpenQASM test binaries all pass. `uvx nox -s lint` passes on the formatted final
tree.

## Interfaces and Dependencies

The final CBit C++ namespace is `mlir::cbit`. TableGen must generate a dialect
class, `RegisterType`, an initialization attribute or generated enum that prints
exactly as `#cbit.init<zero>` and `#cbit.init<undefined>`, and `AllocOp`,
`LoadOp`, and `StoreOp`. The operations use only MLIR builtin `index` and `i1`
types plus MLIR's side-effect interfaces. CBit must not depend on QC, QCO,
Qiskit, OpenQASM, QIR, jeff, or the UB dialect.

The builders depend on CBit and expose per-allocation initialization, load, and
store methods. Frontends and consumers use those APIs or CBit operations
directly. QC-to-QCO and QCO-to-QC treat CBit as identity-preserved legal IR.
QIR, jeff, DD execution, Qiskit, OpenQASM, and CBit-to-memref are the only
components that interpret CBit semantics. No generic memref analysis is a
supported fallback.

Revision note: This first revision records the approved full migration after
pull request #2140 merged and issue #2155 was opened. Update every section when
concrete target names, implementation discoveries, or validation results change.
