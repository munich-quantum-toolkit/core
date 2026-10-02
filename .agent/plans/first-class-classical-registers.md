# Replace implicit classical-register memrefs with CBit IR

Status: historical implementation record.

## Goal and scope

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

## Constraints

- The fix merged in #2140 had to teach Qiskit export that an allocation followed
  by one false store per element is initialization rather than classical
  execution. Evidence: `bindings/mlir/qiskit/QiskitExport.cpp` contains
  `initialClassicalZeroStoreIndex` and a block-order recognizer.

- Qiskit import needed a builder-wide initialization policy only because the
  register type did not carry initialization. Evidence:
  `bindings/mlir/qiskit/QiskitImport.cpp` constructs `QCProgramBuilder` with
  `ClassicalRegisterInitialization::Zero`.

- QIR and OpenQASM contain separate reconstructions of the same missing
  semantics. Evidence: `mlir/lib/Conversion/QCToQIR/QIRCommon/QIRCommon.cpp`
  scans result memrefs, while
  `mlir/lib/Dialect/QC/Translation/OpenQASMToQCEmitter.cpp` tracks both SSA bit
  values and classical-register memrefs.

- This LLVM 22 TableGen setup requires explicit enum and attribute generation in
  addition to operation and type generation. Generated enum declarations also
  require the MLIR operation-implementation headers before inclusion. Evidence:
  the first focused build failed before `CBitOpsEnums.h.inc` and
  `CBitOpsAttributes.h.inc` were added to `MLIRCBitOpsIncGen`; the corrected
  target builds all ten generated fragments.

- `register` is a C++ keyword and cannot be an ODS operand name because TableGen
  uses operand names in generated builder parameters. Evidence: the generated
  C++ failed until the operand identifier became `reg`; the user-facing operand
  description remains "register".

- MLIR verifies custom operations while parsing. A textual module with a
  constant out-of-bounds CBit index therefore fails to parse instead of
  returning an invalid module for a separate `verify` call. Evidence: the
  focused test now asserts parse failure and receives the
  `index exceeds register width` diagnostic.

- Generated dialect and enum declarations require their MLIR and LLVM support
  headers to precede the generated includes. The repository's include sorter
  groups quoted includes first unless a section comment separates the generated
  include. Evidence: a clean focused rebuild failed in the generated headers
  until the CBit wrappers added explicit dependencies and a generated-code
  section; the rebuilt targets then passed.

- Qiskit export must validate CBit stores before walking the instruction stream.
  A dynamic index expression appears before its store and would otherwise fail
  first as an unsupported scalar operation. Evidence: the exporter now
  inventories direct measurement stores and gives destination errors before it
  constructs instructions.

- Dialect conversion adapts a CBit operand to its tensor value before the
  QCO-to-jeff load or store pattern runs. The state map must retain the original
  CBit identity for that adapted value. Evidence: mapping each converted
  allocation result back to its source register fixed structured and
  dynamic-index jeff round trips; all 142 focused tests pass.

- OpenQASM no longer needs to carry bit registers as SCF iterated values. CBit
  operations have explicit memory effects, and nested regions can capture the
  same non-aliasing register. Evidence: removing bit vectors from the frontend
  state preserved dynamic indexing and structured control across all 173
  translation tests.

- QIR Adaptive needs ordinary register storage for internal CBit allocations,
  while returned CBit allocations use QIR result arrays. Evidence: the final
  lowering distinguishes these cases, supports dynamic internal indices, and
  passes all 146 Adaptive tests.

- CBit loads must become SSA values before conversion pipelines can lower
  structured conditions. Evidence: a conservative same-block canonicalizer
  forwards a known prior store or zero initialization and stops at dynamic
  aliasing, nested regions, or unknown register users. The compiler and CBit
  tests cover both forwarding and unsafe cases.

- Combining a measurement with its only CBit store can move the measurement past
  intervening quantum operations if the store is not adjacent. Evidence:
  OpenQASM emission now combines only adjacent operations, and a regression test
  preserves the order of a delayed store.

- Unity builds hid indirect dependencies on the generated CBit enum, type, and
  dialect declarations. Evidence: the extensive C++ lint job found 87
  diagnostics on the published head even though the release build and local lint
  session passed. Direct includes and explicit helper linkage remove the
  branch-related diagnostics, and all focused binaries still pass.

- The QC and QCO modifier verifiers rejected quantum-register and generic memref
  access but did not know about CBit. Evidence: CBit allocation, load, and store
  operations nested under `inv`, `ctrl`, or `pow` passed the old checks even
  though modifier regions must remain unitary.

- QCO-to-jeff can share CBit register state and type conversion with a neutral
  CBit-to-tensor layer. Its same-pass operation materialization must still
  create jeff arrays directly so that dialect conversion keeps adapted register
  values stable. Evidence: the shared state and type layer passes all 142 jeff
  tests, while the independent generic operation patterns pass their focused
  CBit-to-tensor test.

- Clang-Tidy's include cleaner requires the header that declares a generated
  CBit enum or type and the MLIR header that imports LLVM casting functions.
  Evidence: `CBitAttributes.h`, `CBitDialect.h`, and `mlir/Support/LLVM.h`
  remove the diagnostics for `Initialization`, `RegisterType`, and `mlir::isa`;
  the exact focused checks pass.

- A successful QCO-to-jeff-to-QCO round trip did not prove that the jeff
  function returned the updated register value. The reverse conversion
  reconstructed CBit stores even when jeff returned the initial array. Evidence:
  an assertion on the intermediate jeff return exposed the stale value, and the
  QCO-to-jeff return pattern now replaces returned aliases with the latest jeff
  array value.

- The generic native-to-jeff conversion cannot lower a static `tensor.empty`, so
  the staged QCO-to-jeff path still needed a jeff-specific allocation pattern
  and declared temporary `arith` and `tensor` dialect dependencies. Evidence:
  direct `cbit.alloc`, `cbit.load`, and `cbit.store` patterns remove both
  temporary dialect dependencies from the pass.

- The old jeff-to-QCO reconstruction inferred classical registers by scanning
  generic `tensor.empty`, `tensor.insert`, `tensor.extract`, and `scf.for`
  operations after conversion. Evidence: converting jeff i1 array operations
  directly removes the post-pass scan and preserves structured control-flow
  round trips across all 143 jeff tests.

- After direct CBit-to-jeff patterns replaced the tensor stage,
  `populateCBitToTensorConversionPatterns` and `updateRegisterReturns` had no
  production callers. Only QCO-to-jeff used the public library, and it used only
  the state tracker and the trivial register type conversion. Evidence:
  localizing those two pieces removes the library, header, and standalone test
  target while all 143 jeff tests still pass.

## Decisions

- Add CBit as a neutral dialect, not as part of QC or QCO. Rationale: The same
  register must pass unchanged between reference-semantics QC and
  value-semantics QCO, and a neutral dialect avoids conversion-owned copies.

- Model a register as `!cbit.reg<N>` with a static positive width and no
  aliasing operations. Rationale: Every current frontend has a known register
  width, and excluding casts, subviews, aliases, and deallocation makes output
  identity and effect analysis precise.

- Put zero or undefined initialization on each `cbit.alloc`. Rationale: Qiskit
  and OpenQASM 2 require zero, OpenQASM 3 permits undefined bits, and a
  builder-wide policy cannot represent both in one program.

- Default builder allocations to an unnamed, zero-initialized register while
  keeping initialization explicit on `cbit.alloc`. Rationale: zero is the common
  circuit-builder behavior and preserves concise existing call sites; OpenQASM 3
  and tests of undefined behavior still request `Undefined` per allocation.

- Define public result registers only by return values of the entry function.
  Rationale: Allocation and a source name are insufficient to distinguish
  user-visible output from internal state.

- Do not retain a reverse recognizer for arbitrary `memref<Nxi1>` programs.
  Rationale: Compatibility inference recreates the ambiguity that CBit removes.
  Clients that require memref use the explicit one-way late conversion.

- Do not add `UPGRADING.md` instructions. Rationale: The affected Qiskit,
  OpenQASM, builder, and classical-register interfaces were added after v3.8.0
  and are not part of a released compatibility contract.

- Reject CBit allocation, load, and store recursively in QC and QCO modifier
  regions and during QC-to-QCO preflight. Rationale: modifiers model unitary
  transformations; classical state access makes a modifier body non-unitary.

- Convert CBit operations directly to jeff integer-array operations and convert
  static one-dimensional jeff i1 arrays directly to CBit. Keep the neutral state
  and type helpers for structured SSA threading, but do not expose temporary
  native tensor operations as either conversion's register contract. Rationale:
  both directions now identify classical registers from source dialect
  operations, QCO-to-jeff declares only the dialect that it creates, and
  jeff-to-QCO no longer recognizes arbitrary tensor programs.

- Keep the mutable-register-to-SSA tracker private to QCO-to-jeff and remove the
  public CBit-to-tensor conversion library. Rationale: direct jeff lowering is
  its only production consumer, so a public operation-lowering API and
  independent target duplicate behavior without providing reuse. The jeff
  round-trip suite exercises allocation, loads, stores, returns, and structured
  control flow through the localized tracker.

- Keep static register-index validation in the CBit dialect instead of a generic
  type template in dialect utilities. Rationale: every caller uses
  `cbit::RegisterType`, so the template adds an unused abstraction and obscures
  ownership.

## Outcome and validation

The neutral CBit dialect, per-allocation builders, QC/QCO identity preservation,
late memref lowering, and producer/consumer migrations were implemented. CBit
and `jeff` arrays map directly in both directions; QCO-to-jeff owns its private
SSA state, and the unused CBit-to-tensor library was removed. The recorded
native, Python, stub, documentation, and 143 jeff round-trip tests passed. Final
hosted validation was not recorded.

## Code and ownership

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
        int64_t size, StringRef name = {},
        cbit::Initialization initialization = cbit::Initialization::Zero);
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
than editing generated output. Add the draft pull request number to the existing
unreleased MLIR infrastructure changelog entry and keep its link definition.

## Acceptance

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
memref convention.

## Interfaces

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
