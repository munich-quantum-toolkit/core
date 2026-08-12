# Integrate zero-initialized classical state across QC construction and QIR lowering

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

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

## Progress

- [x] (2026-08-09 20:07Z) Mapped the current OpenQASM-to-QC,
  `QuantumComputation`-to-QC, Base QIR, Adaptive QIR, and compiler-test
  initialization paths.
- [x] (2026-08-09 20:07Z) Selected a target architecture: construction policy
  belongs to `QCProgramBuilder`; QIR preparation consumes supported
  classical-result stores before dialect conversion.
- [x] (2026-08-09 20:14Z) Added the builder initialization policy and migrated
      both production translators and their reference builders.
- [x] (2026-08-09 20:14Z) Refactored QIR preparation to erase accepted
      initializer and measurement stores, then remove profile-specific store
      patterns and initializer pointer tracking.
- [x] (2026-08-09 20:14Z) Deleted the test-only module-rewriting helper and
      adapted regression tests.
- [x] (2026-08-09 20:14Z) Built all affected targets and passed the six complete
      affected test binaries.
- [x] (2026-08-09 20:42Z) Passed the focused regressions, all six affected test
      binaries, repository-wide lint, and LLVM 22 clang-tidy for every changed
      C++ source.
- [x] (2026-08-09 20:49Z) Confirmed `origin/main` at `7cef69d7` is already an
  ancestor, the draft PR still points to published head `34289e83`, and the
  final diff passes `git diff --check`.
- [x] (2026-08-09 20:52Z) Prepared and verified a signed local follow-up commit
  without publishing it.

## Surprises & Discoveries

- Observation: the typed OpenQASM emitter already keeps every classical bit as
  an SSA value for structured control flow, but it also materializes classical
  result registers as `memref<...xi1>` so measurements and returned outputs can
  be consumed by later QC and QIR conversions. The two representations have
  different jobs and neither can simply be deleted. Evidence: `bitValues` is
  threaded through `scf` regions in
  `mlir/lib/Dialect/QC/Translation/OpenQASMToQCEmitter.cpp`, while
  `classicalRegisters` is returned from the entry function.
- Observation: `stripReturnedMeasurements` already performs the complete
  pre-conversion inventory of classical registers, measurement destinations, and
  returned outputs. Carrying a second `cregInitializations` operation set into
  both QIR profile conversions duplicates that phase boundary. Evidence: both
  `ConvertMemRefStoreOp` implementations only erase operations that the common
  scan has already classified.
- Observation: the test helper reconstructs production semantics by walking
  arbitrary modules after construction. This makes exact-reference tests know
  the layout of the translator output and is a sign that construction policy is
  missing from `QCProgramBuilder`.
- Observation: preserving the existing one-argument constructor and static
  `build` symbols is inexpensive and keeps the builder change additive.
  Evidence: delegating overloads retain every existing caller while source
  translators and reference builders opt into `Zero` explicitly.
- Observation: making initialization an argument of each allocation looks more
  local, but it makes source-wide semantics harder to state and forces every
  reference-program allocation to repeat the same policy. Evidence: OpenQASM 2
  and `QuantumComputation` apply one rule to all their classical registers, and
  the parameterized reference builders are intentionally source-agnostic.

## Decision Log

- Decision: add a descriptive classical-register initialization policy to
  `QCProgramBuilder`, defaulting to uninitialized for compatibility. The
  OpenQASM emitter selects zero only for OpenQASM 2, and the
  `QuantumComputation` translator always selects zero. Rationale: allocation is
  the one point shared by production and reference builders, while source
  language semantics remain explicit at builder construction. Date/Author:
  2026-08-09 / Codex.
- Decision: keep explicit false stores in QC IR. Rationale: `memref.alloc` is
  uninitialized by MLIR semantics, and QC-to-QCO and other consumers must see a
  real memory write; an ad-hoc attribute or QIR-only assumption would make the
  QC module itself incorrect. Date/Author: 2026-08-09 / Codex.
- Decision: have the common QIR preparation phase erase all supported
  classical-result stores after recording measurement destinations. Rationale:
  QIR profiles represent result storage themselves, so the memory stores are
  source representation consumed by analysis rather than operations that each
  profile should rediscover. This removes the initializer operation side table
  and both profile-specific store conversion patterns. Date/Author: 2026-08-09 /
  Codex.
- Decision: continue rejecting a false store that occurs after an observable use
  or measurement. Rationale: QIR can omit only initial writes into its
  already-false result slots; erasing a later reset would change program
  behavior. Date/Author: 2026-08-09 / Codex.

## Outcomes & Retrospective

The implementation now emits initialization in one production abstraction, has
no test-only IR rewrite, no QIR initializer side table, and no duplicated
Base/Adaptive store patterns. The implementation-and-test delta against the
published revision removes 15 lines overall while adding a focused builder
contract test. LLVM 22 clang-tidy and repository-wide lint pass, as do all six
affected binaries: QC IR 316/316, QC translation 287/287, OpenQASM target
163/163, Base QIR 121/121, Adaptive QIR 145/145, and compiler end-to-end
230/230. The implementation-and-test diff is 159 insertions and 174 deletions,
so the refactor removes 15 lines while eliminating four duplicated ownership
mechanisms. The live base and PR audit is clean, and the redesign is recorded in
a signed local commit. Publication is intentionally deferred until the user
approves this exact revision.

## Context and Orientation

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

## Plan of Work

First, introduce `QCProgramBuilder::ClassicalRegisterInitialization` with
`Uninitialized` and `Zero` alternatives. Store the selected policy in the
builder. When `allocClassicalBitRegister` runs under the zero policy, emit a
false constant and one constant-indexed store for every bit. Preserve the
uninitialized default for all existing callers. Extend the two static `build`
helpers so reference tests can select the policy without rewriting a completed
module.

Next, construct the OpenQASM emitter's builder with `Zero` for an OpenQASM 2
typed program and `Uninitialized` otherwise. Remove its explicit memref store
loop but keep the SSA bit-value initialization and adjust operation-budget
accounting for the builder-emitted constant. Construct the `QuantumComputation`
translator's builder with `Zero` and remove its explicit store loop.

Then rename the common QIR preparation function to describe its broader job.
During its existing store scan, collect every accepted leading false store and
direct measurement-result store. If validation succeeds, erase those stores
before func and QC dialect conversion. Remove `cregInitializations` from
`LoweringState`, remove `ConvertMemRefStoreOp` from Base and Adaptive QIR, and
simplify Adaptive measurement lowering now that the consumed store use is gone.

Finally, delete
`mlir/unittests/Dialect/QC/Translation/ClassicalRegisterTestUtils.h`. Build QC
reference modules with `Zero` only when the corresponding input is OpenQASM 2 or
`QuantumComputation`. Keep the tests that reject a zero store after a
measurement, because they define the boundary between initialization and a
state-changing reset.

## Concrete Steps

All commands run from the repository root.

Build the affected targets after editing:

    ./.agent/run.sh cmake --build build/debug --target \
      mqt-core-mlir-unittest-qc-translation \
      mqt-core-mlir-unittest-qc-ir \
      mqt-core-mlir-unittest-openqasm-target \
      mqt-core-mlir-unittest-qc-to-qir-base \
      mqt-core-mlir-unittest-qc-to-qir-adaptive \
      mqt-core-mlir-unittests-compiler -j 8

Run the six complete binaries:

    ./build/debug/mlir/unittests/Dialect/QC/IR/mqt-core-mlir-unittest-qc-ir --gtest_brief=1
    ./build/debug/mlir/unittests/Dialect/QC/Translation/mqt-core-mlir-unittest-qc-translation --gtest_brief=1
    ./build/debug/mlir/unittests/Target/OpenQASM/mqt-core-mlir-unittest-openqasm-target --gtest_brief=1
    ./build/debug/mlir/unittests/Conversion/QCToQIR/QCToQIRBase/mqt-core-mlir-unittest-qc-to-qir-base --gtest_brief=1
    ./build/debug/mlir/unittests/Conversion/QCToQIR/QCToQIRAdaptive/mqt-core-mlir-unittest-qc-to-qir-adaptive --gtest_brief=1
    ./build/debug/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler --gtest_brief=1

Run repository validation and inspect the final state:

    ./.agent/run.sh uvx nox -s lint
    git diff --check
    git status --short

Run the configured LLVM 22 clang-tidy surface for every changed C++ source using
`build/debug/compile_commands.json`. On macOS, pass the active SDK sysroot and
Xcode libc++ include directory if the Homebrew LLVM binary otherwise cannot find
standard headers.

## Validation and Acceptance

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

## Idempotence and Recovery

Build, test, lint, and inspection commands are repeatable. The refactor is made
on the existing task branch and must preserve unrelated files. If QIR store
erasure exposes an ordering problem, keep collection and validation separate
from mutation: no store is erased until the whole preparation scan succeeds. Do
not reset or clean the worktree; use the focused diff to repair an intermediate
compile failure.

## Artifacts and Notes

Before this refactor, zero initialization is emitted separately in both source
translators, reconstructed by a test-only module walk, and carried into QIR via
`LoweringState::cregInitializations` plus duplicate Base and Adaptive store
patterns. The target architecture has one emission point and one consumption
point.

## Interfaces and Dependencies

The public builder type gains the nested enum
`QCProgramBuilder::ClassicalRegisterInitialization` and additional constructor
and static-build overloads. Existing constructor and build entry points retain
uninitialized behavior. No QC or QIR dialect operation, type, attribute,
external runtime function, or serialization format changes. QIR continues to use
existing MLIR `memref`, `arith`, `qc.measure`, and LLVM/QIR runtime operations.

Revision note: created after tracing the published `34289e83c` implementation.
It replaces translator-local initialization and profile-local initializer
tracking with shared construction and preparation boundaries.

Revision note: updated after the first complete build and test pass. QC IR
passed 316 tests, QC translation 287, OpenQASM target 163, Base QIR 121,
Adaptive QIR 145, and compiler end-to-end 230.

Revision note: finalized after the exact-source test rerun, repository-wide
lint, LLVM 22 clang-tidy, live base/PR audit, and signed local commit.
