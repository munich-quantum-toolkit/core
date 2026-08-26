# Move classic OpenQASM serialization into one serializer

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core currently makes every classic intermediate-representation operation
format itself as OpenQASM. After this change, `QuantumComputation::dumpOpenQASM`
and `QuantumComputation::toQASM` keep producing the same text, but one concrete
`OpenQASMSerializer` owns all formatting. Operation classes no longer expose or
implement OpenQASM methods. Code that must format an individual operation can
use the serializer with the same qubit and classical-bit register maps that the
old methods accepted.

The existing exact-output and import/export round-trip tests demonstrate that
the public circuit behavior remains unchanged. A focused serializer test
demonstrates the replacement for direct operation dumping.

## Progress

- [x] (2026-08-26 09:58Z) Read `AGENTS.md`, `.agent/PLANS.md`,
  `docs/ai_usage.md`, and the ponytail coding guidance.
- [x] (2026-08-26 09:58Z) Traced the current circuit and operation formatting
  flow and identified all in-repository implementations.
- [x] (2026-08-26 09:58Z) Found the direct downstream operation-level callers in
      MQT Debugger and QMAP that require a public replacement.
- [x] (2026-08-26 10:18Z) Created `codex/2098-openqasm-serializer` from
      `origin/main` and confirmed the focused pre-change compatibility suite
      passes apart from two tests whose relative fixture path requires the IR
      build directory as the working directory.
- [x] (2026-08-26 10:20Z) Added `OpenQASMSerializer`, centralized the existing
  formatting behavior, and made `QuantumComputation` delegate to it.
- [x] (2026-08-26 10:20Z) Removed OpenQASM methods and formatting helpers from
  the operation hierarchy and moved the register-map aliases into the
  serializer header.
- [x] (2026-08-26 10:20Z) Added exact direct-operation coverage plus Python
  string/file parity coverage; retained all existing golden, round-trip, and
  symbolic-error behavior.
- [x] (2026-08-26 10:20Z) Documented the direct-caller and custom-operation
  migration in the v4 upgrade guide. A changelog reference remains deferred
  until this work has a PR number, because project policy requires changelog
  entries to cite the PR and author.
- [x] (2026-08-26 10:20Z) Built the release IR target and Python bindings, ran
      all 281 IR tests, the focused Python tests, full lint, standalone checks
      for all changed public headers, and final whitespace checks.
- [x] (2026-08-26 10:48Z) Built the downstream Debugger draft against the local
      Core source. This exposed and fixed a missing direct `<iomanip>` include
      in `NonUnitaryOperation.cpp`; the full 281-test IR suite still passes.
- [x] (2026-08-26 11:03Z) Applied the final Ponytail review: unified the two
      whole-register predicates and inlined compound dispatch, removing 16
      source lines. The 52 focused tests and full lint pass.

## Surprises & Discoveries

- Observation: The register-map aliases in `include/mqt-core/ir/Register.hpp`
  are used only by OpenQASM operation formatting inside Core. Evidence:
  repository-wide search finds their Core uses only in the current dump methods
  and `QuantumComputation::dumpOpenQASM`.
- Observation: Direct operation formatting is a real downstream contract, not
  dead API. Evidence: MQT Debugger formats inverted operations with
  `Operation::dumpOpenQASM2`, and QMAP's superconducting data logger formats a
  `CompoundOperation` with `dumpOpenQASM3`.
- Observation: `SymbolicOperation` currently rejects both OpenQASM 2 and 3
  serialization with distinct error messages. The serializer must keep these
  failures unchanged.
- Observation: QMAP draft PR #1111 adds operation subclasses with their own
  extended OpenQASM syntax and routes mixed circuits through Core's exporter.
  Evidence: the draft adds `NeutralAtomOperation::dumpOpenQASM` and
  `AodOperation::dumpOpenQASM`. QMAP must instead own serialization for those
  extensions; Core's serializer deliberately handles only Core operation
  classes.
- Observation: Running the pre-change `IO.*` tests from the repository root
  produces two fixture lookup failures for `../circuits/test.qasm`; the other 49
  selected tests pass. The final IR suite must be run from
  `build/release/test/ir` so the existing relative fixture paths resolve.
- Observation: CMake's aggregate header-set verification target also treats
  `OpType.inc` as a standalone header and fails on its required `HANDLE_OP_TYPE`
  macro. This is unrelated to the refactor. Every changed public header,
  including `OpenQASMSerializer.hpp`, was compiled through its individual
  generated header-set object target successfully.
- Observation: Debugger's build enables `_LIBCPP_REMOVE_TRANSITIVE_INCLUDES`.
  Its local Core integration build revealed that `NonUnitaryOperation.cpp` uses
  `std::setw` without directly including `<iomanip>` after the serialization
  includes were removed. Adding that standard-library include makes the
  downstream integration build self-contained.

## Decision Log

- Decision: Add one concrete `qc::OpenQASMSerializer` in
  `include/mqt-core/ir/OpenQASMSerializer.hpp` and
  `src/ir/OpenQASMSerializer.cpp`. It stores only the destination stream and
  selected format; register maps are supplied to the operation entry point and
  are never retained. Rationale: One serializer removes formatting from the
  operation hierarchy without adding a visitor interface, factory, extension
  registry, or lifetime hazards around reference-bearing maps. Date/Author:
  2026-08-26 / Codex.
- Decision: Keep an operation-level serializer entry point that accepts the
  existing register maps. Rationale: MQT Debugger and QMAP need to format
  operations without exporting a complete `QuantumComputation`. Date/Author:
  2026-08-26 / Codex.
- Decision: Make register-map entries own their register metadata instead of
  retaining references. Rationale: QMAP stores its qubit map beyond the lifetime
  of the local combined-register map used to build it. Owning the small
  descriptors removes that dangling-reference hazard without changing how
  callers construct or access map entries. Date/Author: 2026-08-26 / Codex.
- Decision: Dispatch over the closed operation hierarchy in the serializer and
  use only public operation getters. Rationale: This avoids friends and new
  virtual methods while keeping ownership of formatting in one file.
  Date/Author: 2026-08-26 / Codex.
- Decision: Preserve emitted bytes, warnings, and exceptions before improving
  any OpenQASM behavior. Rationale: Issue #2098 is an ownership refactor, not a
  syntax or feature change. Date/Author: 2026-08-26 / Codex.

## Outcomes & Retrospective

Core now has one concrete serializer with two entry points:

    OpenQASMSerializer(output, format).serialize(computation)
    OpenQASMSerializer(output, format)
        .serialize(operation, qubitMap, bitMap, indent)

`QuantumComputation` retains its existing C++ and Python export API. The five
operation subclasses contain no OpenQASM formatting code, and the generic
register header no longer exposes serialization-only aliases. The ponytail
constraint kept the design to one class and implementation-local dispatch; there
is no visitor hierarchy, extension registry, friend access, or new dependency.

Validation completed successfully: the release IR target and MinSizeRel Python
bindings build; all 281 IR tests pass; all three focused Python IR tests pass;
the rebuilt Python extension produces identical string and file exports for
OpenQASM 2 and 3; the downstream Debugger integration build passes with
transitive standard-library includes disabled; and `uvx nox -s lint` passes.
`git diff --check` is clean.

Follow-up work remains in downstream repositories. MQT Debugger must replace its
direct `dumpOpenQASM2` calls with the serializer (and should serialize the
inverted clone it already creates). QMAP's superconducting logger can use the
Core operation entry point, while QMAP draft PR #1111 must move neutral-atom and
AOD circuit serialization into QMAP. Add the required changelog entry when the
Core pull request number is known.

## Context and Orientation

Classic IR means the `qc::QuantumComputation` circuit representation and the
operation classes under `include/mqt-core/ir/operations/`; it is separate from
the MLIR OpenQASM translation code under `mlir/`. The public circuit export
methods are declared in `include/mqt-core/ir/QuantumComputation.hpp` and
implemented in `src/ir/QuantumComputation.cpp`.

`QuantumComputation::dumpOpenQASM` currently writes layout comments, the
OpenQASM header, and register declarations. It then builds two maps from global
qubit or bit indices to register names and calls the virtual
`Operation::dumpOpenQASM` method. `StandardOperation`, `NonUnitaryOperation`,
`CompoundOperation`, `IfElseOperation`, and `SymbolicOperation` implement that
method in their own source files. The aliases `QubitIndexToRegisterMap` and
`BitIndexToRegisterMap` live in the generic `Register.hpp` header only because
the virtual interface exposes them.

The new `OpenQASMSerializer` owns the complete circuit header and every
operation form. Its circuit entry point can use the public register getters,
public layout permutations, and public circuit iterators. Its operation entry
point can use `Operation` getters plus the public subclass getters for
measurement destinations, compound children, and if/else state. No operation
class needs to grant friendship.

The relevant behavior tests are in `test/ir/test_io.cpp`,
`test/ir/test_qasm3_parser.cpp`, and `test/ir/test_symbolic.cpp`. The IR test
binary is `build/release/test/ir/mqt-core-ir-test`.

## Plan of Work

Add `include/mqt-core/ir/OpenQASMSerializer.hpp`. Move the two register-map
aliases from `Register.hpp` into this header. Declare a concrete serializer
whose two `serialize` overloads accept a complete circuit or one operation. The
instance stores only a reference to the output stream and the selected OpenQASM
version. Keep implementation-only dispatch in the source file.

Add `src/ir/OpenQASMSerializer.cpp`. Move the existing register sorting, layout
comment, header, register declaration, gate, measurement, reset, compound,
if/else, and symbolic-error logic into this file without changing emitted text.
Dispatch to the known concrete operation types. Keep recursive compound and
if/else serialization inside the serializer.

Replace the body of `QuantumComputation::dumpOpenQASM` with a delegation to the
serializer. Remove `dumpOpenQASM`, `dumpOpenQASM2`, and `dumpOpenQASM3` from
`Operation` and every subclass. Remove the now-unused formatting helpers and
includes from the operation headers and source files. Remove the register-map
aliases from `Register.hpp`.

Add a focused test in `test/ir/test_io.cpp` that constructs register maps,
serializes an operation through `OpenQASMSerializer`, and checks the exact text.
Run all existing IO, parser, and symbolic tests so their golden output, round
trips, conditions, compounds, measurements, layout comments, and error messages
continue to pass.

Update the existing v4 cleanup entry in `UPGRADING.md`. Name the removed
operation methods and register-map include move. Document `OpenQASMSerializer`
as the direct-operation replacement. Defer the matching `CHANGELOG.md` entry
until a pull request number is available, because the project requires that
reference. Do not change Python stubs because the Python-facing
`QuantumComputation` API does not change.

## Concrete Steps

Run all commands from the repository root.

Inspect the focused diff while implementing:

    git diff -- include/mqt-core/ir src/ir test/ir/test_io.cpp CHANGELOG.md UPGRADING.md

Configure and build the release IR test target after adding the new source:

    cmake --preset release
    cmake --build --preset release --target mqt-core-ir-test

Run the narrow serializer and output tests first, followed by all IR tests:

    ./build/release/test/ir/mqt-core-ir-test --gtest_filter='IO.*:SymbolicTest.failPrintingQASM*'
    ./build/release/test/ir/mqt-core-ir-test

Run repository checks at completion:

    uvx nox -s lint
    git diff --check
    git status --short

Expected test output reports all selected tests as passed. Lint and
`git diff --check` must exit with status zero.

## Validation and Acceptance

`QuantumComputation::toQASM(false)` and `toQASM(true)` must match all existing
exact expected strings. Exported OpenQASM must still import to an equivalent
circuit in the existing round-trip tests. OpenQASM 2 and 3 conditions,
whole-register and single-bit measurements, compound operations, negative
controls, layout comments, and register declaration order must remain unchanged.
Symbolic operations must still raise the existing version-specific errors.

The focused new test must serialize an operation without calling a method on
that operation. It must use `OpenQASMSerializer` and the relocated register-map
aliases. A repository search must find no OpenQASM formatting method or helper
in any operation header or source file.

The public `QuantumComputation` methods must remain source compatible. The
upgrade guide must give direct operation callers one exact replacement call.

## Idempotence and Recovery

All source edits and test commands are repeatable. CMake configuration may be
rerun safely when the glob discovers the new source and header. If a behavior
test changes output, compare the serializer code with the deleted operation
implementation and restore the old byte sequence rather than updating the golden
expectation. Preserve unrelated work and revert only task-owned lines if an
experiment is discarded.

## Artifacts and Notes

The initial direct-consumer audit found these old calls:

    MQT Debugger: Operation::dumpOpenQASM2(stream, qubitMap, {})
    QMAP DataLogger: CompoundOperation::dumpOpenQASM3(stream, qregs, cregs)

The planned replacement is an `OpenQASMSerializer` constructed with the output
stream and version, followed by `serialize(operation, qubitMap, bitMap)`.

## Interfaces and Dependencies

The final public header `include/mqt-core/ir/OpenQASMSerializer.hpp` must define
`qc::QubitIndexToRegisterMap`, `qc::BitIndexToRegisterMap`, and
`qc::OpenQASMSerializer`. The serializer must provide a circuit entry point used
by `QuantumComputation::dumpOpenQASM` and an operation entry point used by
direct consumers. It must depend only on the existing Core IR library and the
C++20 standard library.

No operation class may declare an OpenQASM formatting method. No new virtual
interface, generic visitor framework, factory, callback, or external dependency
is part of this change.

Revision note: Created the initial self-contained plan after tracing the current
implementation and downstream direct callers. Updated it after the compiling
milestone with the final API, compatibility evidence, downstream coordination,
and the deferred changelog requirement.
