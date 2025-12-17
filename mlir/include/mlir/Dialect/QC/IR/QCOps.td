// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef QC_OPS
#define QC_OPS

include "mlir/Dialect/QC/IR/QCInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// QC Dialect Definition
//===----------------------------------------------------------------------===//

def QCDialect : Dialect {
    let name = "qc";

    let summary = "The QC (reference semantics) dialect for quantum computing.";

    let description = [{
        The QC dialect uses **reference semantics** where quantum operations
        modify qubits in place, similar to how hardware physically transforms
        quantum states. This model provides:

        - Natural mapping to hardware execution models
        - Intuitive representation for circuit descriptions
        - Direct compatibility with imperative quantum programming languages
        - Straightforward backend code generation

        The name "QC" stands for "Quantum Circuit."

        Example:
        ```mlir
        qc.h %q          // Applies Hadamard to qubit %q in place
        qc.swap %q0, %q1 // Applies SWAP using %q0, %q1 as targets
        ```
    }];

    let cppNamespace = "::mlir::qc";

    let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// QC Type Definitions
//===----------------------------------------------------------------------===//

class QCType<string name, string typeMnemonic, list<Trait> traits = []>
        : TypeDef<QCDialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def QubitType : QCType<"Qubit", "qubit"> {
    let summary = "QC qubit reference type";
    let description = [{
        The `!qc.qubit` type represents a reference to a quantum bit in the
        QC dialect. Operations using this type modify qubits in place using
        reference semantics, similar to how classical imperative languages handle
        mutable references.
    }];
}

//===----------------------------------------------------------------------===//
// Base Operation Classes
//===----------------------------------------------------------------------===//

class QCOp<string mnemonic, list<Trait> traits = []> :
    Op<QCDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Resource Operations
//===----------------------------------------------------------------------===//

def AllocOp : QCOp<"alloc", [MemoryEffects<[MemAlloc]>]> {
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
        %q = qc.alloc : !qc.qubit
        ```

        Example (qubits in a register):
        ```mlir
        %q0 = qc.alloc("q", 3, 0) : !qc.qubit
        %q1 = qc.alloc("q", 3, 1) : !qc.qubit
        %q2 = qc.alloc("q", 3, 2) : !qc.qubit
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

def DeallocOp : QCOp<"dealloc", [MemoryEffects<[MemFree]>]> {
    let summary = "Deallocate a qubit";
    let description = [{
        Deallocates a qubit, releasing its resources.

        Example:
        ```mlir
        qc.dealloc %q : !qc.qubit
        ```
    }];

    let arguments = (ins QubitType:$qubit);
    let assemblyFormat = "$qubit attr-dict `:` type($qubit)";

    let hasCanonicalizer = 1;
}

def StaticOp : QCOp<"static", [Pure]> {
    let summary = "Retrieve a static qubit by index";
    let description = [{
        The `qc.static` operation produces an SSA value representing a qubit
        identified by a static index. This is useful for referring to fixed
        qubits in a quantum program or to hardware-mapped qubits.

        Example:
        ```mlir
        %q = qc.static 0 : !qc.qubit
        ```
    }];

    let arguments = (ins ConfinedAttr<I64Attr, [IntNonNegative]>:$index);
    let results = (outs QubitType:$qubit);
    let assemblyFormat = "$index attr-dict `:` type($qubit)";
}

//===----------------------------------------------------------------------===//
// Measurement and Reset Operations
//===----------------------------------------------------------------------===//

def MeasureOp : QCOp<"measure"> {
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
        %result = qc.measure %q : !qc.qubit -> i1
        ```

        Example (measurement with output recording):
        ```mlir
        %result = qc.measure("c", 2, 0) %q : !qc.qubit -> i1
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

def ResetOp : QCOp<"reset"> {
    let summary = "Reset a qubit to |0⟩ state";
    let description = [{
        Resets a qubit to the |0⟩ state, regardless of its current state.

        Example:
        ```mlir
        qc.reset %q : !qc.qubit
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
    let cppNamespace = "::mlir::qc";
}

def ZeroTargetOneParameter : TargetAndParameterArityTrait<0, 1>;
def OneTargetZeroParameter : TargetAndParameterArityTrait<1, 0>;
def OneTargetOneParameter : TargetAndParameterArityTrait<1, 1>;
def OneTargetTwoParameter : TargetAndParameterArityTrait<1, 2>;
def OneTargetThreeParameter : TargetAndParameterArityTrait<1, 3>;
def TwoTargetZeroParameter : TargetAndParameterArityTrait<2, 0>;
def TwoTargetOneParameter : TargetAndParameterArityTrait<2, 1>;
def TwoTargetTwoParameter : TargetAndParameterArityTrait<2, 2>;

//===----------------------------------------------------------------------===//
// Unitary Operations
//===----------------------------------------------------------------------===//

def GPhaseOp : QCOp<"gphase", traits = [UnitaryOpInterface, ZeroTargetOneParameter, MemoryEffects<[MemWrite]>]> {
    let summary = "Apply a global phase to the state";
    let description = [{
        Applies a global phase to the state.

        Example:
        ```mlir
        qc.gphase(%theta)
        ```
    }];

    let arguments = (ins Arg<F64, "the rotation angle theta in radians">:$theta);
    let assemblyFormat = "`(` $theta `)` attr-dict";

    let extraClassDeclaration = [{
        static StringRef getBaseSymbol() { return "gphase"; }
    }];

    let builders = [
        OpBuilder<(ins "const std::variant<double, Value>&":$theta)>
    ];
}

def IdOp : QCOp<"id", traits = [UnitaryOpInterface, OneTargetZeroParameter]> {
    let summary = "Apply an Id gate to a qubit";
    let description = [{
        Applies an Id gate to a qubit, modifying it in place.

        Example:
        ```mlir
        qc.id %q : !qc.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead, MemWrite]>:$qubit_in);
    let assemblyFormat = "$qubit_in attr-dict `:` type($qubit_in)";

    let extraClassDeclaration = [{
        static StringRef getBaseSymbol() { return "id"; }
    }];
}

def XOp : QCOp<"x", traits = [UnitaryOpInterface, OneTargetZeroParameter]> {
    let summary = "Apply an X gate to a qubit";
    let description = [{
        Applies an X gate to a qubit, modifying it in place.

        Example:
        ```mlir
        qc.x %q : !qc.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead, MemWrite]>:$qubit_in);
    let assemblyFormat = "$qubit_in attr-dict `:` type($qubit_in)";

    let extraClassDeclaration = [{
        static StringRef getBaseSymbol() { return "x"; }
    }];
}

def YOp : QCOp<"y", traits = [UnitaryOpInterface, OneTargetZeroParameter]> {
    let summary = "Apply a Y gate to a qubit";
    let description = [{
        Applies a Y gate to a qubit, modifying it in place.

        Example:
        ```mlir
        qc.y %q : !qc.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead, MemWrite]>:$qubit_in);
    let assemblyFormat = "$qubit_in attr-dict `:` type($qubit_in)";

    let extraClassDeclaration = [{
        static StringRef getBaseSymbol() { return "y"; }
    }];
}

def ZOp : QCOp<"z", traits = [UnitaryOpInterface, OneTargetZeroParameter]> {
    let summary = "Apply a Z gate to a qubit";
    let description = [{
        Applies a Z gate to a qubit, modifying it in place.

        Example:
        ```mlir
        qc.z %q : !qc.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead, MemWrite]>:$qubit_in);
    let assemblyFormat = "$qubit_in attr-dict `:` type($qubit_in)";

    let extraClassDeclaration = [{
        static StringRef getBaseSymbol() { return "z"; }
    }];
}

def HOp : QCOp<"h", traits = [UnitaryOpInterface, OneTargetZeroParameter]> {
    let summary = "Apply an H gate to a qubit";
    let description = [{
        Applies an H gate to a qubit, modifying it in place.

        Example:
        ```mlir
        qc.h %q : !qc.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead, MemWrite]>:$qubit_in);
    let assemblyFormat = "$qubit_in attr-dict `:` type($qubit_in)";

    let extraClassDeclaration = [{
        static StringRef getBaseSymbol() { return "h"; }
    }];
}

def SOp : QCOp<"s", traits = [UnitaryOpInterface, OneTargetZeroParameter]> {
    let summary = "Apply an S gate to a qubit";
    let description = [{
        Applies an S gate to a qubit, modifying it in place.

        Example:
        ```mlir
        qc.s %q : !qc.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead, MemWrite]>:$qubit_in);
    let assemblyFormat = "$qubit_in attr-dict `:` type($qubit_in)";

    let extraClassDeclaration = [{
        static StringRef getBaseSymbol() { return "s"; }
    }];
}

def SdgOp : QCOp<"sdg", traits = [UnitaryOpInterface, OneTargetZeroParameter]> {
    let summary = "Apply an Sdg gate to a qubit";
    let description = [{
        Applies an Sdg gate to a qubit, modifying it in place.

        Example:
        ```mlir
        qc.sdg %q : !qc.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead, MemWrite]>:$qubit_in);
    let assemblyFormat = "$qubit_in attr-dict `:` type($qubit_in)";

    let extraClassDeclaration = [{
        static StringRef getBaseSymbol() { return "sdg"; }
    }];
}

def TOp : QCOp<"t", traits = [UnitaryOpInterface, OneTargetZeroParameter]> {
    let summary = "Apply a T gate to a qubit";
    let description = [{
        Applies a T gate to a qubit, modifying it in place.

        Example:
        ```mlir
        qc.t %q : !qc.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead, MemWrite]>:$qubit_in);
    let assemblyFormat = "$qubit_in attr-dict `:` type($qubit_in)";

    let extraClassDeclaration = [{
        static StringRef getBaseSymbol() { return "t"; }
    }];
}

def TdgOp : QCOp<"tdg", traits = [UnitaryOpInterface, OneTargetZeroParameter]> {
    let summary = "Apply a Tdg gate to a qubit";
    let description = [{
        Applies a Tdg gate to a qubit, modifying it in place.

        Example:
        ```mlir
        qc.tdg %q : !qc.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead, MemWrite]>:$qubit_in);
    let assemblyFormat = "$qubit_in attr-dict `:` type($qubit_in)";

    let extraClassDeclaration = [{
        static StringRef getBaseSymbol() { return "tdg"; }
    }];
}

def SXOp : QCOp<"sx", traits = [UnitaryOpInterface, OneTargetZeroParameter]> {
    let summary = "Apply an SX gate to a qubit";
    let description = [{
        Applies an SX gate to a qubit, modifying it in place.

        Example:
        ```mlir
        qc.sx %q : !qc.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead, MemWrite]>:$qubit_in);
    let assemblyFormat = "$qubit_in attr-dict `:` type($qubit_in)";

    let extraClassDeclaration = [{
        static StringRef getBaseSymbol() { return "sx"; }
    }];
}

def SXdgOp : QCOp<"sxdg", traits = [UnitaryOpInterface, OneTargetZeroParameter]> {
    let summary = "Apply an SXdg gate to a qubit";
    let description = [{
        Applies an SXdg gate to a qubit, modifying it in place.

        Example:
        ```mlir
        qc.sxdg %q : !qc.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead, MemWrite]>:$qubit_in);
    let assemblyFormat = "$qubit_in attr-dict `:` type($qubit_in)";

    let extraClassDeclaration = [{
        static StringRef getBaseSymbol() { return "sxdg"; }
    }];
}

def RXOp : QCOp<"rx", traits = [UnitaryOpInterface, OneTargetOneParameter]> {
    let summary = "Apply an RX gate to a qubit";
    let description = [{
        Applies an RX gate to a qubit, modifying it in place.

        Example:
        ```mlir
        qc.rx(%theta) %q : !qc.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead, MemWrite]>:$qubit_in,
                     Arg<F64, "the rotation angle theta in radians">:$theta);
    let assemblyFormat = "`(` $theta `)` $qubit_in attr-dict `:` type($qubit_in)";

    let extraClassDeclaration = [{
        static StringRef getBaseSymbol() { return "rx"; }
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit_in, "const std::variant<double, Value>&":$theta)>
    ];
}

def RYOp : QCOp<"ry", traits = [UnitaryOpInterface, OneTargetOneParameter]> {
    let summary = "Apply an RY gate to a qubit";
    let description = [{
        Applies an RY gate to a qubit, modifying it in place.

        Example:
        ```mlir
        qc.ry(%theta) %q : !qc.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead, MemWrite]>:$qubit_in,
                     Arg<F64, "the rotation angle theta in radians">:$theta);
    let assemblyFormat = "`(` $theta `)` $qubit_in attr-dict `:` type($qubit_in)";

    let extraClassDeclaration = [{
        static StringRef getBaseSymbol() { return "ry"; }
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit_in, "const std::variant<double, Value>&":$theta)>
    ];
}

def RZOp : QCOp<"rz", traits = [UnitaryOpInterface, OneTargetOneParameter]> {
    let summary = "Apply an RZ gate to a qubit";
    let description = [{
        Applies an RZ gate to a qubit, modifying it in place.

        Example:
        ```mlir
        qc.rz(%theta) %q : !qc.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead, MemWrite]>:$qubit_in,
                     Arg<F64, "the rotation angle theta in radians">:$theta);
    let assemblyFormat = "`(` $theta `)` $qubit_in attr-dict `:` type($qubit_in)";

    let extraClassDeclaration = [{
        static StringRef getBaseSymbol() { return "rz"; }
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit_in, "const std::variant<double, Value>&":$theta)>
    ];
}

def POp : QCOp<"p", traits = [UnitaryOpInterface, OneTargetOneParameter]> {
    let summary = "Apply a P gate to a qubit";
    let description = [{
        Applies a P gate to a qubit, modifying it in place.

        Example:
        ```mlir
        qc.p(%theta) %q : !qc.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead, MemWrite]>:$qubit_in,
                     Arg<F64, "the rotation angle theta in radians">:$theta);
    let assemblyFormat = "`(` $theta `)` $qubit_in attr-dict `:` type($qubit_in)";

    let extraClassDeclaration = [{
        static StringRef getBaseSymbol() { return "p"; }
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit_in, "const std::variant<double, Value>&":$theta)>
    ];
}

def ROp : QCOp<"r", traits = [UnitaryOpInterface, OneTargetTwoParameter]> {
    let summary = "Apply an R gate to a qubit";
    let description = [{
        Applies an R gate to a qubit, modifying it in place.

        Example:
        ```mlir
        qc.r(%theta, %phi) %q : !qc.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead, MemWrite]>:$qubit_in,
                     Arg<F64, "the rotation angle theta in radians">:$theta,
                     Arg<F64, "the rotation angle phi in radians">:$phi);
    let assemblyFormat = "`(` $theta `,` $phi `)` $qubit_in attr-dict `:` type($qubit_in)";

    let extraClassDeclaration = [{
        static StringRef getBaseSymbol() { return "r"; }
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit_in, "const std::variant<double, Value>&":$theta, "const std::variant<double, Value>&":$phi)>
    ];
}

def U2Op : QCOp<"u2", traits = [UnitaryOpInterface, OneTargetTwoParameter]> {
    let summary = "Apply a U2 gate to a qubit";
    let description = [{
        Applies a U2 gate to a qubit, modifying it in place.

        Example:
        ```mlir
        qc.u2(%phi, %lambda) %q : !qc.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead, MemWrite]>:$qubit_in,
                     Arg<F64, "the rotation angle phi in radians">:$phi,
                     Arg<F64, "the rotation angle lambda in radians">:$lambda);
    let assemblyFormat = "`(` $phi `,` $lambda `)` $qubit_in attr-dict `:` type($qubit_in)";

    let extraClassDeclaration = [{
        static StringRef getBaseSymbol() { return "u2"; }
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit_in, "const std::variant<double, Value>&":$phi, "const std::variant<double, Value>&":$lambda)>
    ];
}

def UOp : QCOp<"u", traits = [UnitaryOpInterface, OneTargetThreeParameter]> {
    let summary = "Apply a U gate to a qubit";
    let description = [{
        Applies a U gate to a qubit, modifying it in place.

        Example:
        ```mlir
        qc.u(%theta, %phi, %lambda) %q : !qc.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead, MemWrite]>:$qubit_in,
                     Arg<F64, "the rotation angle theta in radians">:$theta,
                     Arg<F64, "the rotation angle phi in radians">:$phi,
                     Arg<F64, "the rotation angle lambda in radians">:$lambda);
    let assemblyFormat = "`(` $theta `,` $phi `,` $lambda `)` $qubit_in attr-dict `:` type($qubit_in)";

    let extraClassDeclaration = [{
        static StringRef getBaseSymbol() { return "u"; }
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit_in, "const std::variant<double, Value>&":$theta, "const std::variant<double, Value>&":$phi, "const std::variant<double, Value>&":$lambda)>
    ];
}

def SWAPOp : QCOp<"swap", traits = [UnitaryOpInterface, TwoTargetZeroParameter]> {
    let summary = "Apply a SWAP gate to two qubits";
    let description = [{
        Applies a SWAP gate to two qubits, modifying them in place.

        Example:
        ```mlir
        qc.swap %q0, %q1 : !qc.qubit, !qc.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the first target qubit", [MemRead, MemWrite]>:$qubit0_in,
                     Arg<QubitType, "the second target qubit", [MemRead, MemWrite]>:$qubit1_in);
    let assemblyFormat = "$qubit0_in `,` $qubit1_in attr-dict `:` type($qubit0_in) `,` type($qubit1_in)";

    let extraClassDeclaration = [{
        static StringRef getBaseSymbol() { return "swap"; }
    }];
}

def iSWAPOp : QCOp<"iswap", traits = [UnitaryOpInterface, TwoTargetZeroParameter]> {
    let summary = "Apply a iSWAP gate to two qubits";
    let description = [{
        Applies a iSWAP gate to two qubits, modifying them in place.

        Example:
        ```mlir
        qc.iswap %q0, %q1 : !qc.qubit, !qc.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the first target qubit", [MemRead, MemWrite]>:$qubit0_in,
                     Arg<QubitType, "the second target qubit", [MemRead, MemWrite]>:$qubit1_in);
    let assemblyFormat = "$qubit0_in `,` $qubit1_in attr-dict `:` type($qubit0_in) `,` type($qubit1_in)";

    let extraClassDeclaration = [{
        static StringRef getBaseSymbol() { return "iswap"; }
    }];
}

def DCXOp : QCOp<"dcx", traits = [UnitaryOpInterface, TwoTargetZeroParameter]> {
    let summary = "Apply a DCX gate to two qubits";
    let description = [{
        Applies a DCX gate to two qubits, modifying them in place.

        Example:
        ```mlir
        qc.dcx %q0, %q1 : !qc.qubit, !qc.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the first target qubit", [MemRead, MemWrite]>:$qubit0_in,
                     Arg<QubitType, "the second target qubit", [MemRead, MemWrite]>:$qubit1_in);
    let assemblyFormat = "$qubit0_in `,` $qubit1_in attr-dict `:` type($qubit0_in) `,` type($qubit1_in)";

    let extraClassDeclaration = [{
        static StringRef getBaseSymbol() { return "dcx"; }
    }];
}

def ECROp : QCOp<"ecr", traits = [UnitaryOpInterface, TwoTargetZeroParameter]> {
    let summary = "Apply an ECR gate to two qubits";
    let description = [{
        Applies an ECR gate to two qubits, modifying them in place.

        Example:
        ```mlir
        qc.ecr %q0, %q1 : !qc.qubit, !qc.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the first target qubit", [MemRead, MemWrite]>:$qubit0_in,
                     Arg<QubitType, "the second target qubit", [MemRead, MemWrite]>:$qubit1_in);
    let assemblyFormat = "$qubit0_in `,` $qubit1_in attr-dict `:` type($qubit0_in) `,` type($qubit1_in)";

    let extraClassDeclaration = [{
        static StringRef getBaseSymbol() { return "ecr"; }
    }];
}

def RXXOp : QCOp<"rxx", traits = [UnitaryOpInterface, TwoTargetOneParameter]> {
    let summary = "Apply an RXX gate to two qubits";
    let description = [{
        Applies an RXX gate to two qubits, modifying them in place.

        Example:
        ```mlir
        qc.rxx(%theta) %q0, %q1 : !qc.qubit, !qc.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the first target qubit", [MemRead, MemWrite]>:$qubit0_in,
                     Arg<QubitType, "the second target qubit", [MemRead, MemWrite]>:$qubit1_in,
                     Arg<F64, "the rotation angle theta in radians">:$theta);
    let assemblyFormat = "`(` $theta `)` $qubit0_in `,` $qubit1_in attr-dict `:` type($qubit0_in) `,` type($qubit1_in)";

    let extraClassDeclaration = [{
        static StringRef getBaseSymbol() { return "rxx"; }
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit0_in, "Value":$qubit1_in, "const std::variant<double, Value>&":$theta)>
    ];
}

def RYYOp : QCOp<"ryy", traits = [UnitaryOpInterface, TwoTargetOneParameter]> {
    let summary = "Apply an RYY gate to two qubits";
    let description = [{
        Applies an RYY gate to two qubits, modifying them in place.

        Example:
        ```mlir
        qc.ryy(%theta) %q0, %q1 : !qc.qubit, !qc.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the first target qubit", [MemRead, MemWrite]>:$qubit0_in,
                     Arg<QubitType, "the second target qubit", [MemRead, MemWrite]>:$qubit1_in,
                     Arg<F64, "the rotation angle theta in radians">:$theta);
    let assemblyFormat = "`(` $theta `)` $qubit0_in `,` $qubit1_in attr-dict `:` type($qubit0_in) `,` type($qubit1_in)";

    let extraClassDeclaration = [{
        static StringRef getBaseSymbol() { return "ryy"; }
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit0_in, "Value":$qubit1_in, "const std::variant<double, Value>&":$theta)>
    ];
}

def RZXOp : QCOp<"rzx", traits = [UnitaryOpInterface, TwoTargetOneParameter]> {
    let summary = "Apply an RZX gate to two qubits";
    let description = [{
        Applies an RZX gate to two qubits, modifying them in place.

        Example:
        ```mlir
        qc.rzx(%theta) %q0, %q1 : !qc.qubit, !qc.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the first target qubit", [MemRead, MemWrite]>:$qubit0_in,
                     Arg<QubitType, "the second target qubit", [MemRead, MemWrite]>:$qubit1_in,
                     Arg<F64, "the rotation angle theta in radians">:$theta);
    let assemblyFormat = "`(` $theta `)` $qubit0_in `,` $qubit1_in attr-dict `:` type($qubit0_in) `,` type($qubit1_in)";

    let extraClassDeclaration = [{
        static StringRef getBaseSymbol() { return "rzx"; }
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit0_in, "Value":$qubit1_in, "const std::variant<double, Value>&":$theta)>
    ];
}

def RZZOp : QCOp<"rzz", traits = [UnitaryOpInterface, TwoTargetOneParameter]> {
    let summary = "Apply an RZZ gate to two qubits";
    let description = [{
        Applies an RZZ gate to two qubits, modifying them in place.

        Example:
        ```mlir
        qc.rzz(%theta) %q0, %q1 : !qc.qubit, !qc.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the first target qubit", [MemRead, MemWrite]>:$qubit0_in,
                     Arg<QubitType, "the second target qubit", [MemRead, MemWrite]>:$qubit1_in,
                     Arg<F64, "the rotation angle theta in radians">:$theta);
    let assemblyFormat = "`(` $theta `)` $qubit0_in `,` $qubit1_in attr-dict `:` type($qubit0_in) `,` type($qubit1_in)";

    let extraClassDeclaration = [{
        static StringRef getBaseSymbol() { return "rzz"; }
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit0_in, "Value":$qubit1_in, "const std::variant<double, Value>&":$theta)>
    ];
}

def XXPlusYYOp : QCOp<"xx_plus_yy", traits = [UnitaryOpInterface, TwoTargetTwoParameter]> {
    let summary = "Apply an XX+YY gate to two qubits";
    let description = [{
        Applies an XX+YY gate to two qubits, modifying them in place.

        Example:
        ```mlir
        qc.xx_plus_yy(%theta, %beta) %q0, %q1 : !qc.qubit, !qc.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the first target qubit", [MemRead, MemWrite]>:$qubit0_in,
                     Arg<QubitType, "the second target qubit", [MemRead, MemWrite]>:$qubit1_in,
                     Arg<F64, "the rotation angle theta in radians">:$theta,
                     Arg<F64, "the rotation angle beta in radians">:$beta);
    let assemblyFormat = "`(` $theta `,` $beta `)` $qubit0_in `,` $qubit1_in attr-dict `:` type($qubit0_in) `,` type($qubit1_in)";

    let extraClassDeclaration = [{
        static StringRef getBaseSymbol() { return "xx_plus_yy"; }
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit0_in, "Value":$qubit1_in, "const std::variant<double, Value>&":$theta, "const std::variant<double, Value>&":$beta)>
    ];
}

def XXMinusYYOp : QCOp<"xx_minus_yy", traits = [UnitaryOpInterface, TwoTargetTwoParameter]> {
    let summary = "Apply an XX-YY gate to two qubits";
    let description = [{
        Applies an XX-YY gate to two qubits, modifying them in place.

        Example:
        ```mlir
        qc.xx_minus_yy(%theta, %beta) %q0, %q1 : !qc.qubit, !qc.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the first target qubit", [MemRead, MemWrite]>:$qubit0_in,
                     Arg<QubitType, "the second target qubit", [MemRead, MemWrite]>:$qubit1_in,
                     Arg<F64, "the rotation angle theta in radians">:$theta,
                     Arg<F64, "the rotation angle beta in radians">:$beta);
    let assemblyFormat = "`(` $theta `,` $beta `)` $qubit0_in `,` $qubit1_in attr-dict `:` type($qubit0_in) `,` type($qubit1_in)";

    let extraClassDeclaration = [{
        static StringRef getBaseSymbol() { return "xx_minus_yy"; }
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit0_in, "Value":$qubit1_in, "const std::variant<double, Value>&":$theta, "const std::variant<double, Value>&":$beta)>
    ];
}

def BarrierOp : QCOp<"barrier", traits = [UnitaryOpInterface]> {
    let summary = "Apply a barrier gate to a set of qubits";
    let description = [{
        Applies a barrier gate to a set of qubits, modifying them in place.

        Example:
        ```mlir
        qc.barrier %q : !qc.qubit
        ```
    }];

    let arguments = (ins Arg<Variadic<QubitType>, "the target qubits", [MemRead, MemWrite]>:$qubits);
    let assemblyFormat = "$qubits attr-dict `:` type($qubits)";

    let extraClassDeclaration = [{
        size_t getNumQubits();
        size_t getNumTargets();
        static size_t getNumControls();
        Value getQubit(size_t i);
        Value getTarget(size_t i);
        static Value getControl(size_t i);
        static size_t getNumParams();
        static Value getParameter(size_t i);
        static StringRef getBaseSymbol() { return "barrier"; }
    }];
}

//===----------------------------------------------------------------------===//
// Modifiers
//===----------------------------------------------------------------------===//

def YieldOp : QCOp<"yield", traits = [Terminator]> {
    let summary = "Yield from a modifier region";
    let description = [{
        Terminates a modifier region, yielding control back to the enclosing operation.
    }];

    let assemblyFormat = "attr-dict";
}

def CtrlOp : QCOp<"ctrl",
            traits = [
                UnitaryOpInterface,
                SingleBlockImplicitTerminator<"::mlir::qc::YieldOp">,
                RecursiveMemoryEffects
            ]> {
    let summary = "Add control qubits to a unitary operation";
    let description = [{
        A modifier operation that adds control qubits to the unitary operation
        defined in its body region. The controlled operation applies the
        underlying unitary only when all control qubits are in the |1⟩ state.

        Note that control qubits are logically unmodified by this operation in that
        their quantum state remains unchanged. However, the `controls` argument
        is marked with `MemWrite` to ensure correct dependency tracking in MLIR.

        Example:
        ```mlir
        qc.ctrl(%q0) {
          qc.x %q1 : !qc.qubit
        }
        ```
    }];

    let arguments = (ins Arg<Variadic<QubitType>, "the control qubits", [MemRead, MemWrite]>:$controls);
    let regions = (region SizedRegion<1>:$body);
    let assemblyFormat = "`(` $controls `)` $body attr-dict `:` type($controls)";

    let extraClassDeclaration = [{
        [[nodiscard]] UnitaryOpInterface getBodyUnitary();
        size_t getNumQubits();
        size_t getNumTargets();
        size_t getNumControls();
        Value getQubit(size_t i);
        Value getTarget(size_t i);
        Value getControl(size_t i);
        size_t getNumParams();
        Value getParameter(size_t i);
        static StringRef getBaseSymbol() { return "ctrl"; }
    }];

    let builders = [
        OpBuilder<(ins "ValueRange":$controls, "UnitaryOpInterface":$bodyUnitary)>,
        OpBuilder<(ins "ValueRange":$controls, "const std::function<void(OpBuilder &)>&":$bodyBuilder)>
    ];

    let hasCanonicalizer = 1;
    let hasVerifier = 1;
}

#endif // QC_OPS
