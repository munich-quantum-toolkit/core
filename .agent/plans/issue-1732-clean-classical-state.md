# Integrate zero-initialized classical state across QC construction and QIR lowering

Status: historical implementation record.

Later register representation:
[first-class CBit registers](first-class-classical-registers.md).

## Goal and scope

OpenQASM 2 and `qc::QuantumComputation` define classical registers to contain
zero before the first measurement. OpenQASM 3 deliberately does not provide that
guarantee. After this work, both zero-initialized source paths produce the same
canonical QC memory pattern through `QCProgramBuilder`, OpenQASM 3 remains
uninitialized, and Base and Adaptive QIR consume the canonical classical-result
memory representation in one preparation phase. A reader can observe the result
by translating conditions that read unmeasured bits, compiling them to Adaptive
QIR, and running the full QC translation, OpenQASM target, QC-to-QIR, and
compiler test binaries.

The refactor also removes the test-only
`mlir/unittests/Dialect/QC/Translation/ClassicalRegisterTestUtils.h`. Reference
programs select the same builder initialization policy as the source format
instead of constructing an uninitialized module and rewriting it afterward.

## Constraints

- the typed OpenQASM emitter already keeps every classical bit as an SSA value
  for structured control flow, but it also materializes classical result
  registers as `memref<...xi1>` so measurements and returned outputs can be
  consumed by later QC and QIR conversions. The two representations have
  different jobs and neither can simply be deleted. Evidence: `bitValues` is
  threaded through `scf` regions in
  `mlir/lib/Dialect/QC/Translation/OpenQASMToQCEmitter.cpp`, while
  `classicalRegisters` is returned from the entry function.

- `stripReturnedMeasurements` already performs the complete pre-conversion
  inventory of classical registers, measurement destinations, and returned
  outputs. Carrying a second `cregInitializations` operation set into both QIR
  profile conversions duplicates that phase boundary. Evidence: both
  `ConvertMemRefStoreOp` implementations only erase operations that the common
  scan has already classified.

- the test helper reconstructs production semantics by walking arbitrary modules
  after construction. This makes exact-reference tests know the layout of the
  translator output and is a sign that construction policy is missing from
  `QCProgramBuilder`.

- preserving the existing one-argument constructor and static `build` symbols is
  inexpensive and keeps the builder change additive. Evidence: delegating
  overloads retain every existing caller while source translators and reference
  builders opt into `Zero` explicitly.

- making initialization an argument of each allocation looks more local, but it
  makes source-wide semantics harder to state and forces every reference-program
  allocation to repeat the same policy. Evidence: OpenQASM 2 and
  `QuantumComputation` apply one rule to all their classical registers, and the
  parameterized reference builders are intentionally source-agnostic.

## Decisions

- add a descriptive classical-register initialization policy to
  `QCProgramBuilder`, defaulting to uninitialized for compatibility. The
  OpenQASM emitter selects zero only for OpenQASM 2, and the
  `QuantumComputation` translator always selects zero. Rationale: allocation is
  the one point shared by production and reference builders, while source
  language semantics remain explicit at builder construction.

- keep explicit false stores in QC IR. Rationale: `memref.alloc` is
  uninitialized by MLIR semantics, and QC-to-QCO and other consumers must see a
  real memory write; an ad-hoc attribute or QIR-only assumption would make the
  QC module itself incorrect.

- have the common QIR preparation phase erase all supported classical-result
  stores after recording measurement destinations. Rationale: QIR profiles
  represent result storage themselves, so the memory stores are source
  representation consumed by analysis rather than operations that each profile
  should rediscover. This removes the initializer operation side table and both
  profile-specific store conversion patterns.

- continue rejecting a false store that occurs after an observable use or
  measurement. Rationale: QIR can omit only initial writes into its
  already-false result slots; erasing a later reset would change program
  behavior.

## Outcome and validation

The builder owns classical zero initialization. QIR uses shared lowering without
an initializer side table, duplicated Base/Adaptive patterns, or test- only IR
rewriting. All six affected QC, translation, OpenQASM, QIR, and compiler
binaries passed, together with clang-tidy and repository lint.

## Code and ownership

`QCProgramBuilder` in `mlir/include/mlir/Dialect/QC/Builder/QCProgramBuilder.h`
and `mlir/lib/Dialect/QC/Builder/QCProgramBuilder.cpp` constructs QC dialect
modules. Classical registers are ordinary one-dimensional MLIR memory references
whose elements are one-bit integers. Ordinary `memref.alloc` leaves memory
uninitialized, so zero initialization must be represented by `arith.constant`
and `memref.store` operations.

`mlir/lib/Dialect/QC/Translation/OpenQASMToQCEmitter.cpp` consumes the typed
OpenQASM frontend. It tracks source-level bit values as SSA values for
structured control flow and creates a memref only for registers that must be
returned. OpenQASM 2 declarations start at false; OpenQASM 3 declarations use
`ub.poison` until assigned.

`mlir/lib/Dialect/QC/Translation/TranslateQuantumComputationToQC.cpp` converts
the C++ `qc::QuantumComputation` representation. Measurements write into
classical-register memrefs, and conditions reload those memrefs. All registers
therefore need zero initialization before operations are translated.

Base and Adaptive QIR conversions live under `mlir/lib/Conversion/QCToQIR/`. The
common function currently named `stripReturnedMeasurements` inventories
classical result registers and maps each `qc.measure` to the memref slot
receiving its result. Base QIR creates static result pointers, while Adaptive
QIR allocates result-pointer arrays initialized to the false/null result. In
both profiles, a leading false store is therefore redundant after its semantics
have been validated.

## Acceptance

An entirely unmeasured OpenQASM 2 register must compare equal to zero, and a
partially measured register must use false for untouched bits. OpenQASM 3 must
continue producing the existing uninitialized-value diagnostic. The
`QuantumComputation` translator must support single-bit and all six register
comparisons before or after partial measurement, including nonzero register
start indices and measurements inside conditional branches.

Both QIR profiles must accept the canonical leading zero initialization and must
reject a false store after a measurement. Adaptive QIR must compile a
zero-initialized register condition end to end. The five binaries named above
must pass in full, lint and clang-tidy must report no changed-source errors, and
the final worktree must contain only the cohesive refactor.

## Interfaces

The public builder type gains the nested enum
`QCProgramBuilder::ClassicalRegisterInitialization` and additional constructor
and static-build overloads. Existing constructor and build entry points retain
uninitialized behavior. No QC or QIR dialect operation, type, attribute,
external runtime function, or serialization format changes. QIR continues to use
existing MLIR `memref`, `arith`, `qc.measure`, and LLVM/QIR runtime operations.
