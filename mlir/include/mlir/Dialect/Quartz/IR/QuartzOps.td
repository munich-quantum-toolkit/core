// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef QUARTZ_OPS
#define QUARTZ_OPS

include "mlir/Dialect/Quartz/IR/QuartzInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Quartz Dialect Definition
//===----------------------------------------------------------------------===//

def QuartzDialect : Dialect {
    let name = "quartz";

    let summary = "The Quartz (reference semantics) dialect for quantum computing.";

    let description = [{
        The Quartz dialect uses **reference semantics** where quantum operations
        modify qubits in place, similar to how hardware physically transforms
        quantum states. This model provides:

        - Natural mapping to hardware execution models
        - Intuitive representation for circuit descriptions
        - Direct compatibility with imperative quantum programming languages
        - Straightforward backend code generation

        The name "Quartz" reflects the crystalline, structured nature of
        hardware-oriented representations—operations have fixed positions and
        transform states in place, like atoms in a crystal lattice.

        Example:
        ```mlir
        quartz.h %q              // Applies Hadamard to qubit %q in place
        quartz.swap %q0, %q1     // Applies SWAP using %q0, %q1 as targets
        ```
    }];

    let cppNamespace = "::mlir::quartz";

    let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Quartz Type Definitions
//===----------------------------------------------------------------------===//

class QuartzType<string name, string typeMnemonic, list<Trait> traits = []>
        : TypeDef<QuartzDialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def QubitType : QuartzType<"Qubit", "qubit"> {
    let summary = "Quartz qubit reference type";
    let description = [{
        The `!quartz.qubit` type represents a reference to a quantum bit in the
        Quartz dialect. Operations using this type modify qubits in place using
        reference semantics, similar to how classical imperative languages handle
        mutable references.
    }];
}

//===----------------------------------------------------------------------===//
// Base Operation Classes
//===----------------------------------------------------------------------===//

class QuartzOp<string mnemonic, list<Trait> traits = []> :
    Op<QuartzDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Resource Operations
//===----------------------------------------------------------------------===//

def AllocOp : QuartzOp<"alloc", [MemoryEffects<[MemAlloc]>]> {
    let summary = "Allocate a qubit dynamically";
    let description = [{
        Allocates a new qubit dynamically and returns a reference to it.
        The qubit is initialized to the |0⟩ state.

        Optionally, the qubit can be part of a register by specifying:
        - `register_name`: The name of the register this qubit belongs to
        - `register_size`: The total size of the register
        - `register_index`: The index of this qubit within the register

        Example (single qubit):
        ```mlir
        %q = quartz.alloc : !quartz.qubit
        ```

        Example (qubits in a register):
        ```mlir
        %q0 = quartz.alloc("q", 3, 0) : !quartz.qubit
        %q1 = quartz.alloc("q", 3, 1) : !quartz.qubit
        %q2 = quartz.alloc("q", 3, 2) : !quartz.qubit
        ```
    }];

    let arguments = (ins OptionalAttr<StrAttr>:$register_name,
                     OptionalAttr<ConfinedAttr<I64Attr, [IntPositive]>>:$register_size,
                     OptionalAttr<ConfinedAttr<I64Attr, [IntNonNegative]>>:$register_index);
    let results = (outs QubitType:$result);
    let assemblyFormat = [{
        (`(` $register_name^ `,` $register_size `,` $register_index `)`)?
        attr-dict `:` type($result)
    }];

    let builders = [
        OpBuilder<(ins), [{
            build($_builder, $_state, QubitType::get($_builder.getContext()), nullptr, nullptr, nullptr);
        }]>,
        OpBuilder<(ins "::mlir::StringAttr":$register_name,
                       "::mlir::IntegerAttr":$register_size,
                       "::mlir::IntegerAttr":$register_index), [{
            build($_builder, $_state, QubitType::get($_builder.getContext()),
                  register_name, register_size, register_index);
        }]>
    ];

    let hasVerifier = 1;
}

def DeallocOp : QuartzOp<"dealloc", [MemoryEffects<[MemFree]>]> {
    let summary = "Deallocate a qubit";
    let description = [{
        Deallocates a qubit, releasing its resources.

        Example:
        ```mlir
        quartz.dealloc %q : !quartz.qubit
        ```
    }];

    let arguments = (ins QubitType:$qubit);
    let assemblyFormat = "$qubit attr-dict `:` type($qubit)";
}

def StaticOp : QuartzOp<"static", [Pure]> {
    let summary = "Retrieve a static qubit by index";
    let description = [{
        The `quartz.static` operation produces an SSA value representing a qubit
        identified by a static index. This is useful for referring to fixed
        qubits in a quantum program or to hardware-mapped qubits.

        Example:
        ```mlir
        %q = quartz.static 0 : !quartz.qubit
        ```
    }];

    let arguments = (ins ConfinedAttr<I64Attr, [IntNonNegative]>:$index);
    let results = (outs QubitType:$qubit);
    let assemblyFormat = "$index attr-dict `:` type($qubit)";
}

//===----------------------------------------------------------------------===//
// Measurement and Reset Operations
//===----------------------------------------------------------------------===//

def MeasureOp : QuartzOp<"measure"> {
    let summary = "Measure a qubit in the computational basis";
    let description = [{
        Measures a qubit in the computational (Z) basis, collapsing the state
        and returning a classical bit result.

        Optionally, the measurement can be recorded to an output register by
        specifying:
        - `register_name`: Name of the classical register (e.g., "c")
        - `register_size`: Total size of the register
        - `register_index`: Index within the register for this measurement

        Example (simple measurement):
        ```mlir
        %result = quartz.measure %q : !quartz.qubit -> i1
        ```

        Example (measurement with output recording):
        ```mlir
        %result = quartz.measure("c", 2, 0) %q : !quartz.qubit -> i1
        ```
    }];

    let arguments = (ins Arg<QubitType, "the qubit to measure", [MemRead, MemWrite]>:$qubit,
                     OptionalAttr<StrAttr>:$register_name,
                     OptionalAttr<ConfinedAttr<I64Attr, [IntPositive]>>:$register_size,
                     OptionalAttr<ConfinedAttr<I64Attr, [IntNonNegative]>>:$register_index);
    let results = (outs I1:$result);
    let assemblyFormat = [{
        (`(` $register_name^ `,` $register_size `,` $register_index `)`)?
        $qubit `:` type($qubit) `->` type($result) attr-dict
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit), [{
            build($_builder, $_state, $_builder.getI1Type(), qubit, nullptr, nullptr, nullptr);
        }]>
    ];

    let hasVerifier = 1;
}

def ResetOp : QuartzOp<"reset"> {
    let summary = "Reset a qubit to |0⟩ state";
    let description = [{
        Resets a qubit to the |0⟩ state, regardless of its current state.

        Example:
        ```mlir
        quartz.reset %q : !quartz.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the qubit to reset", [MemRead, MemWrite]>:$qubit);
    let assemblyFormat = "$qubit attr-dict `:` type($qubit)";
}

//===----------------------------------------------------------------------===//
// Unitary Operations
//===----------------------------------------------------------------------===//

def XOp : QuartzOp<"x", traits = [Quartz_UnitaryOpInterface]> {
    let arguments = (ins QubitType:$qubit_in);
    let assemblyFormat = "$qubit_in attr-dict";

    let extraClassDeclaration = [{
        size_t getNumQubits() { return getNumTargets() + getNumControls(); }
        static size_t getNumTargets() { return 1; }
        size_t getNumControls() { return getNumPosControls() + getNumNegControls(); }
        static size_t getNumPosControls() { return 0; }
        static size_t getNumNegControls() { return 0; }
        Value getQubit(size_t i);
        Value getTarget(size_t i);
        static Value getPosControl(size_t i);
        static Value getNegControl(size_t i);
        static size_t getNumParams() { return 0; }
        static ParameterDescriptor getParameter(size_t i);
        static bool hasStaticUnitary() { return true; }
        DenseElementsAttr tryGetStaticMatrix();
        static StringRef getBaseSymbol() { return "x"; }
    }];
}

def RXOp : QuartzOp<"rx", traits = [Quartz_UnitaryOpInterface]> {
    let arguments = (ins QubitType:$qubit_in,
                     OptionalAttr<F64Attr>:$theta_attr,
                     Optional<F64>:$theta_operand);
    let assemblyFormat = "`(` ($theta_attr^)? ($theta_operand^)? `)` $qubit_in attr-dict";

    let extraClassDeclaration = [{
        size_t getNumQubits() { return getNumTargets() + getNumControls(); }
        static size_t getNumTargets() { return 1; }
        size_t getNumControls() { return getNumPosControls() + getNumNegControls(); }
        static size_t getNumPosControls() { return 0; }
        static size_t getNumNegControls() { return 0; }
        Value getQubit(size_t i);
        Value getTarget(size_t i);
        static Value getPosControl(size_t i);
        static Value getNegControl(size_t i);
        static size_t getNumParams() { return 1; }
        ParameterDescriptor getParameter(size_t i);
        bool hasStaticUnitary();
        DenseElementsAttr tryGetStaticMatrix();
        static StringRef getBaseSymbol() { return "rx"; }
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit_in, "double":$theta_double), [{
            auto theta_attr = FloatAttr::get(Float64Type::get($_builder.getContext()), theta_double);
            build($_builder, $_state, qubit_in, theta_attr, nullptr);
        }]>,
        OpBuilder<(ins "Value":$qubit_in, "FloatAttr":$theta_attr), [{
            build($_builder, $_state, qubit_in, theta_attr, nullptr);
        }]>,
        OpBuilder<(ins "Value":$qubit_in, "Value":$theta_operand), [{
            build($_builder, $_state, qubit_in, nullptr, theta_operand);
        }]>,
    ];

    let hasVerifier = 1;
}

#endif // QUARTZ_OPS
