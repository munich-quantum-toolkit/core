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
include "mlir/IR/RegionKindInterface.td"
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
// Traits
//===----------------------------------------------------------------------===//

class TargetAndParameterArityTrait<int T, int P>
        : ParamNativeOpTrait<"TargetAndParameterArityTrait", !strconcat(!cast<string>(T), ",", !cast<string>(P))> {
    let cppNamespace = "::mlir::quartz";
}

def OneTargetZeroParameter : TargetAndParameterArityTrait<1, 0>;
def OneTargetOneParameter : TargetAndParameterArityTrait<1, 1>;
def OneTargetTwoParameter : TargetAndParameterArityTrait<1, 2>;
def TwoTargetZeroParameter : TargetAndParameterArityTrait<2, 0>;

//===----------------------------------------------------------------------===//
// Unitary Operations
//===----------------------------------------------------------------------===//

def XOp : QuartzOp<"x", traits = [UnitaryOpInterface, OneTargetZeroParameter]> {
    let summary = "Apply an X gate to a qubit";
    let description = [{
        Applies an X gate to a qubit, modifying it in place.

        Example:
        ```mlir
        quartz.x %q : !flux.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead, MemWrite]>:$qubit_in);
    let assemblyFormat = "$qubit_in attr-dict";

    let extraClassDeclaration = [{
        DenseElementsAttr tryGetStaticMatrix();
        static StringRef getBaseSymbol() { return "x"; }
    }];
}

def SOp : QuartzOp<"s", traits = [UnitaryOpInterface, OneTargetZeroParameter]> {
    let summary = "Apply an S gate to a qubit";
    let description = [{
        Applies an S gate to a qubit, modifying it in place.

        Example:
        ```mlir
        quartz.s %q : !flux.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead, MemWrite]>:$qubit_in);
    let assemblyFormat = "$qubit_in attr-dict";

    let extraClassDeclaration = [{
        DenseElementsAttr tryGetStaticMatrix();
        static StringRef getBaseSymbol() { return "s"; }
    }];
}

def SdgOp : QuartzOp<"sdg", traits = [UnitaryOpInterface, OneTargetZeroParameter]> {
    let summary = "Apply an Sdg gate to a qubit";
    let description = [{
        Applies an Sdg gate to a qubit, modifying it in place.

        Example:
        ```mlir
        quartz.sdg %q : !flux.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead, MemWrite]>:$qubit_in);
    let assemblyFormat = "$qubit_in attr-dict";

    let extraClassDeclaration = [{
        DenseElementsAttr tryGetStaticMatrix();
        static StringRef getBaseSymbol() { return "sdg"; }
    }];
}

def RXOp : QuartzOp<"rx", traits = [UnitaryOpInterface, OneTargetOneParameter]> {
    let summary = "Apply an RX gate to a qubit";
    let description = [{
        Applies an RX gate to a qubit, modifying it in place.

        Example:
        ```mlir
        quartz.rx(%theta) %q : !flux.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead, MemWrite]>:$qubit_in,
                     Arg<F64, "the rotation angle theta in radians", [MemRead]>:$theta);
    let assemblyFormat = "`(` $theta `)` $qubit_in attr-dict";

    let extraClassDeclaration = [{
        DenseElementsAttr tryGetStaticMatrix();
        static StringRef getBaseSymbol() { return "rx"; }
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit_in, "const std::variant<double, Value>&":$theta)>
    ];
}

def U2Op : QuartzOp<"u2", traits = [UnitaryOpInterface, OneTargetTwoParameter]> {
    let summary = "Apply a U2 gate to a qubit";
    let description = [{
        Applies a U2 gate to a qubit, modifying it in place.

        Example:
        ```mlir
        quartz.u2(%phi, %lambda) %q : !flux.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead, MemWrite]>:$qubit_in,
                     Arg<F64, "the rotation angle phi in radians", [MemRead]>:$phi,
                     Arg<F64, "the rotation angle lambda in radians", [MemRead]>:$lambda);
    let assemblyFormat = "`(` $phi `,` $lambda `)` $qubit_in attr-dict";
    let extraClassDeclaration = [{
        DenseElementsAttr tryGetStaticMatrix();
        static StringRef getBaseSymbol() { return "u2"; }
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit_in, "const std::variant<double, Value>&":$phi, "const std::variant<double, Value>&":$lambda)>
    ];
}

def SWAPOp : QuartzOp<"swap", traits = [UnitaryOpInterface, TwoTargetZeroParameter]> {
    let summary = "Apply a SWAP gate to two qubits";
    let description = [{
        Applies a SWAP gate to two qubits, modifying them in place.

        Example:
        ```mlir
        quartz.swap %q0, %q1 : !flux.qubit, !flux.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the first target qubit", [MemRead, MemWrite]>:$qubit0,
                     Arg<QubitType, "the second target qubit", [MemRead, MemWrite]>:$qubit1);
    let assemblyFormat = "$qubit0 $qubit1 attr-dict";

    let extraClassDeclaration = [{
        DenseElementsAttr tryGetStaticMatrix();
        static StringRef getBaseSymbol() { return "swap"; }
    }];
}

//===----------------------------------------------------------------------===//
// Modifiers
//===----------------------------------------------------------------------===//

def YieldOp : QuartzOp<"yield", traits = [Terminator]> {
    let summary = "Yield from a modifier region";
    let description = [{
        Terminates a modifier region, yielding control back to the enclosing operation.
    }];

    let assemblyFormat = "attr-dict";
}

def CtrlOp : QuartzOp<"ctrl",
            traits = [
                UnitaryOpInterface,
                SingleBlockImplicitTerminator<"::mlir::quartz::YieldOp">
            ]> {
    let summary = "Add control qubits to a unitary operation";
    let description = [{
        A modifier operation that adds control qubits to the unitary operation
        defined in its body region. The controlled operation applies the
        underlying unitary only when all control qubits are in the |1⟩ state.
        The control qubits are not modified by this operation.

        Example:
        ```mlir
        quartz.ctrl(%q0) {
          quartz.x %q1 : !quartz.qubit
        }
        ```
    }];

    let arguments = (ins Arg<Variadic<QubitType>, "the control qubits", [MemRead,MemWrite]>:$controls);
    let regions = (region SizedRegion<1>:$body);
    let assemblyFormat = "`(` $controls `)` $body attr-dict";

    let builders = [
        OpBuilder<(ins "ValueRange":$controls, "UnitaryOpInterface":$bodyUnitary)>,
        OpBuilder<(ins "ValueRange":$controls, "const std::function<void(OpBuilder &)>&":$bodyBuilder)>
    ];

    let extraClassDeclaration = [{
        [[nodiscard]] UnitaryOpInterface getBodyUnitary();
        size_t getNumQubits();
        size_t getNumTargets();
        size_t getNumControls();
        size_t getNumPosControls();
        size_t getNumNegControls();
        Value getQubit(size_t i);
        Value getTarget(size_t i);
        Value getPosControl(size_t i);
        Value getNegControl(size_t i);
        size_t getNumParams();
        Value getParameter(size_t i);
        bool hasStaticUnitary();
        DenseElementsAttr tryGetStaticMatrix();
        static StringRef getBaseSymbol() { return "ctrl"; }
    }];

    let hasCanonicalizer = 1;
    let hasVerifier = 1;
}

#endif // QUARTZ_OPS
