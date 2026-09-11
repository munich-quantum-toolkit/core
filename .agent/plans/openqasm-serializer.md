# Move circuit IR OpenQASM serialization into one serializer

Status: historical implementation record.

## Goal and scope

MQT Core currently makes every circuit IR operation format itself as OpenQASM.
After this change, `QuantumComputation::dumpOpenQASM` and
`QuantumComputation::toQASM` keep producing the same text, but one concrete
`OpenQASMSerializer` owns all formatting. Operation classes no longer expose or
implement OpenQASM methods. Code that must format an individual operation can
use the serializer with the same qubit and classical-bit register maps that the
old methods accepted.

The existing exact-output and import/export round-trip tests demonstrate that
the public circuit behavior remains unchanged. A focused serializer test
demonstrates the replacement for direct operation dumping.

## Constraints

- The register-map aliases in `include/mqt-core/ir/Register.hpp` are used only
  by OpenQASM operation formatting inside Core. Evidence: repository-wide search
  finds their Core uses only in the current dump methods and
  `QuantumComputation::dumpOpenQASM`.

- Direct operation formatting is a real downstream contract, not dead API.
  Evidence: MQT Debugger formats inverted operations with
  `Operation::dumpOpenQASM2`, and QMAP's superconducting data logger formats a
  `CompoundOperation` with `dumpOpenQASM3`.

- `SymbolicOperation` currently rejects both OpenQASM 2 and 3 serialization with
  distinct error messages. The serializer must keep these failures unchanged.

- QMAP draft PR #1111 adds operation subclasses with their own extended OpenQASM
  syntax and routes mixed circuits through Core's exporter. Evidence: the draft
  adds `NeutralAtomOperation::dumpOpenQASM` and `AodOperation::dumpOpenQASM`.
  QMAP must instead own serialization for those extensions; Core's serializer
  deliberately handles only Core operation classes.

- CMake's aggregate header-set verification target also treats `OpType.inc` as a
  standalone header and fails on its required `HANDLE_OP_TYPE` macro. This is
  unrelated to the refactor. Every changed public header, including
  `OpenQASMSerializer.hpp`, was compiled through its individual generated
  header-set object target successfully.

- Debugger's build enables `_LIBCPP_REMOVE_TRANSITIVE_INCLUDES`. Its local Core
  integration build revealed that `NonUnitaryOperation.cpp` uses `std::setw`
  without directly including `<iomanip>` after the serialization includes were
  removed. Adding that standard-library include makes the downstream integration
  build self-contained.

## Decisions

- Add one concrete `qc::OpenQASMSerializer` in
  `include/mqt-core/ir/OpenQASMSerializer.hpp` and
  `src/ir/OpenQASMSerializer.cpp`. It stores the destination stream, selected
  format, and an optional callback for otherwise unsupported leaf operations;
  register maps are supplied to the callback and are never retained. Rationale:
  One serializer removes formatting from the operation hierarchy while letting
  downstream formats reuse Core's compound and conditional traversal without a
  visitor hierarchy or extension registry.

- Keep an operation-level serializer entry point that accepts the existing
  register maps. Rationale: MQT Debugger and QMAP need to format operations
  without exporting a complete `QuantumComputation`.

- Make register-map entries own their register metadata instead of retaining
  references. Rationale: QMAP stores its qubit map beyond the lifetime of the
  local combined-register map used to build it. Owning the small descriptors
  removes that dangling-reference hazard without changing how callers construct
  or access map entries.

- Dispatch built-in operations in the serializer, then offer otherwise
  unsupported leaves to the optional callback. Rationale: This avoids friends
  and new virtual methods while keeping built-in formatting in one file and
  supporting downstream operation types.

- Preserve emitted bytes, warnings, and exceptions before improving any OpenQASM
  behavior. Rationale: Issue #2098 is an ownership refactor, not a syntax or
  feature change.

## Outcome and validation

Core now has one concrete serializer with two entry points:

    OpenQASMSerializer(output, format).serialize(computation)
    OpenQASMSerializer(output, format)
        .serialize(operation, qubitMap, bitMap, indent)

`QuantumComputation` retains its existing C++ and Python export API. The five
operation subclasses contain no OpenQASM formatting code, and the generic
register header no longer exposes serialization-only aliases. The ponytail
constraint kept the design to one class and implementation-local dispatch; there
is no visitor hierarchy, extension registry, friend access, or new dependency.
An optional leaf callback lets downstream serializers reuse the same traversal.

Validation completed successfully: the release IR target and MinSizeRel Python
bindings build; all 287 IR tests pass; all three focused Python IR tests pass;
the rebuilt Python extension produces identical string and file exports for
OpenQASM 2 and 3; the downstream Debugger integration build passes with
transitive standard-library includes disabled; and `uvx nox -s lint` passes.
`git diff --check` is clean.

Follow-up work remains in downstream repositories. MQT Debugger must replace its
direct `dumpOpenQASM2` calls with the serializer (and should serialize the
inverted clone it already creates). QMAP's superconducting logger can use the
Core operation entry point, while QMAP draft PR #1111 must move neutral-atom and
AOD circuit serialization into QMAP.

## Code and ownership

Circuit IR means the `qc::QuantumComputation` circuit representation and the
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

## Acceptance

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

## Interfaces

The final public header `include/mqt-core/ir/OpenQASMSerializer.hpp` must define
`qc::QubitIndexToRegisterMap`, `qc::BitIndexToRegisterMap`, and
`qc::OpenQASMSerializer`. The serializer must provide a circuit entry point used
by `QuantumComputation::dumpOpenQASM` and an operation entry point used by
direct consumers. It must depend only on the existing Core IR library and the
C++20 standard library.

No operation class may declare an OpenQASM formatting method. No new virtual
interface, generic visitor framework, factory, extension registry, or external
dependency is part of this change. The serializer may accept one optional
custom-leaf callback.
