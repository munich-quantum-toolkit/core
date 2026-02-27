// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef QCOOPS
#define QCOOPS

include "mlir/Dialect/QCO/IR/QCOInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def QCODialect : Dialect {
    let name = "qco";

    let summary = "The QCO (value semantics) dialect for quantum computing.";

    let description = [{
        The QCO dialect uses **value semantics** where quantum operations
        consume input qubits and produce new output values, following the
        functional programming and SSA paradigm. This model enables:

        - Powerful compiler optimizations through clear dataflow
        - Safe reordering and parallelization analysis
        - Advanced transformation passes
        - Explicit dependency tracking

        The name "QCO" stands for "Quantum Circuit Optimization."

        Example:
        ```mlir
        %q_out = qco.h %q_in                       // Consumes %q_in, produces %q_out
        %q0_out, %q1_out = qco.swap %q0_in, %q1_in // Consumes inputs, produces outputs
        ```
    }];

    let cppNamespace = "::mlir::qco";

    let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// QCO Type Definitions
//===----------------------------------------------------------------------===//

class QCOType<string name, string typeMnemonic, list<Trait> traits = []>
        : TypeDef<QCODialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def QubitType : QCOType<"Qubit", "qubit"> {
    let summary = "QCO qubit value type";
    let description = [{
        The `!qco.qubit` type represents an SSA value holding a quantum bit
        in the QCO dialect. Operations using this type consume input qubits
        and produce new output qubits following value semantics and the SSA
        paradigm, enabling powerful dataflow analysis and optimization.

        Example:
        ```mlir
        %q0 = qco.alloc : !qco.qubit
        %q1 = qco.h %q0 : !qco.qubit -> !qco.qubit
        %q2 = qco.x %q1 : !qco.qubit -> !qco.qubit
        ```
    }];
}

//===----------------------------------------------------------------------===//
// Base Operation Classes
//===----------------------------------------------------------------------===//

class QCOOp<string mnemonic, list<Trait> traits = []> :
    Op<QCODialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Resource Operations
//===----------------------------------------------------------------------===//

def AllocOp : QCOOp<"alloc", [MemoryEffects<[MemAlloc]>]> {
    let summary = "Allocate a qubit dynamically";
    let description = [{
        Allocates a new qubit dynamically and returns an SSA value representing it.
        The qubit is initialized to the |0⟩ state.

        Optionally, the qubit can be part of a register by specifying:
        - `register_name`: The name of the register this qubit belongs to
        - `register_size`: The total size of the register
        - `register_index`: The index of this qubit within the register

        Example (single qubit):
        ```mlir
        %q = qco.alloc : !qco.qubit
        ```

        Example (qubits in a register):
        ```mlir
        %q0 = qco.alloc("q", 3, 0) : !qco.qubit
        %q1 = qco.alloc("q", 3, 1) : !qco.qubit
        %q2 = qco.alloc("q", 3, 2) : !qco.qubit
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

def DeallocOp : QCOOp<"dealloc", [MemoryEffects<[MemFree]>]> {
    let summary = "Deallocate a qubit";
    let description = [{
        Deallocates a qubit, releasing its resources.

        Example:
        ```mlir
        qco.dealloc %q : !qco.qubit
        ```
    }];

    let arguments = (ins QubitType:$qubit);
    let assemblyFormat = "$qubit attr-dict `:` type($qubit)";
    let hasCanonicalizer = 1;
}

def StaticOp : QCOOp<"static", [Pure]> {
    let summary = "Retrieve a static qubit by index";
    let description = [{
        The `qco.static` operation produces an SSA value representing a qubit
        identified by a static index. This is useful for referring to fixed
        qubits in a quantum program or to hardware-mapped qubits.

        Example:
        ```mlir
        %q = qco.static 0 : !qco.qubit
        ```
    }];

    let arguments = (ins ConfinedAttr<I64Attr, [IntNonNegative]>:$index);
    let results = (outs QubitType:$qubit);
    let assemblyFormat = "$index attr-dict `:` type($qubit)";
}

//===----------------------------------------------------------------------===//
// Measurement and Reset Operations
//===----------------------------------------------------------------------===//

def MeasureOp : QCOOp<"measure"> {
    let summary = "Measure a qubit in the computational basis";
    let description = [{
        Measures a qubit in the computational (Z) basis, collapsing the state
        and returning both the output qubit and a classical bit result.

        Optionally, the measurement can be recorded to an output register by
        specifying:
        - `register_name`: Name of the classical register (e.g., "c")
        - `register_size`: Total size of the register
        - `register_index`: Index within the register for this measurement

        Example (simple measurement):
        ```mlir
        %q_out, %result = qco.measure %q_in : !qco.qubit
        ```

        Example (measurement with output recording):
        ```mlir
        %q_out, %result = qco.measure("c", 2, 0) %q_in : !qco.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the qubit to measure", [MemRead]>:$qubit_in,
                     OptionalAttr<StrAttr>:$register_name,
                     OptionalAttr<ConfinedAttr<I64Attr, [IntPositive]>>:$register_size,
                     OptionalAttr<ConfinedAttr<I64Attr, [IntNonNegative]>>:$register_index);
    let results = (outs QubitType:$qubit_out, I1:$result);
    let assemblyFormat = [{
        (`(` $register_name^ `,` $register_size `,` $register_index `)`)?
        $qubit_in `:` type($qubit_in) attr-dict
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit_in), [{
            build($_builder, $_state, QubitType::get($_builder.getContext()), $_builder.getI1Type(),
                  qubit_in, nullptr, nullptr, nullptr);
        }]>
    ];

    let hasVerifier = 1;
}

def ResetOp : QCOOp<"reset", [Idempotent, SameOperandsAndResultType]> {
    let summary = "Reset a qubit to |0⟩ state";
    let description = [{
        Resets a qubit to the |0⟩ state, regardless of its current state,
        and returns the reset qubit.

        Example:
        ```mlir
        %q_out = qco.reset %q_in : !qco.qubit -> !qco.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the qubit to reset", [MemRead]>:$qubit_in);
    let results = (outs QubitType:$qubit_out);
    let assemblyFormat = "$qubit_in attr-dict `:` type($qubit_in) `->` type($qubit_out)";
    let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Traits
//===----------------------------------------------------------------------===//

class TargetAndParameterArityTrait<int T, int P>
        : ParamNativeOpTrait<"TargetAndParameterArityTrait", !cast<string>(T) # ", " # !cast<string>(P)> {
    let cppNamespace = "::mlir::qco";
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

def GPhaseOp : QCOOp<"gphase", traits = [UnitaryOpInterface, ZeroTargetOneParameter, MemoryEffects<[MemWrite]>]> {
    let summary = "Apply a global phase to the state";
    let description = [{
        Applies a global phase to the state.

        Example:
        ```mlir
        qco.gphase(%theta)
        ```
    }];

    let arguments = (ins Arg<F64, "the rotation angle theta in radians">:$theta);
    let assemblyFormat = "`(` $theta `)` attr-dict";

    let extraClassDeclaration = [{
        [[nodiscard]] static StringRef getBaseSymbol() { return "gphase"; }
        [[nodiscard]] std::optional<Eigen::Matrix<std::complex<double>, 1, 1>> getUnitaryMatrix();
    }];

    let builders = [
        OpBuilder<(ins "const std::variant<double, Value>&":$theta)>
    ];

    let hasCanonicalizer = 1;
}

def IdOp : QCOOp<"id", traits = [UnitaryOpInterface, OneTargetZeroParameter]> {
    let summary = "Apply an Id gate to a qubit";
    let description = [{
        Applies an Id gate to a qubit and returns the transformed qubit.

        Example:
        ```mlir
        %q_out = qco.id %q_in : !qco.qubit -> !qco.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead]>:$qubit_in);
    let results = (outs QubitType:$qubit_out);
    let assemblyFormat = "$qubit_in attr-dict `:` type($qubit_in) `->` type($qubit_out)";

    let extraClassDeclaration = [{
        [[nodiscard]] static StringRef getBaseSymbol() { return "id"; }
        [[nodiscard]] static Eigen::Matrix2cd getUnitaryMatrix();
    }];

    let hasCanonicalizer = 1;
}

def XOp : QCOOp<"x", traits = [UnitaryOpInterface, OneTargetZeroParameter]> {
    let summary = "Apply an X gate to a qubit";
    let description = [{
        Applies an X gate to a qubit and returns the transformed qubit.

        Example:
        ```mlir
        %q_out = qco.x %q_in : !qco.qubit -> !qco.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead]>:$qubit_in);
    let results = (outs QubitType:$qubit_out);
    let assemblyFormat = "$qubit_in attr-dict `:` type($qubit_in) `->` type($qubit_out)";

    let extraClassDeclaration = [{
        [[nodiscard]] static StringRef getBaseSymbol() { return "x"; }
        [[nodiscard]] static Eigen::Matrix2cd getUnitaryMatrix();
    }];

    let hasCanonicalizer = 1;
}

def YOp : QCOOp<"y", traits = [UnitaryOpInterface, OneTargetZeroParameter]> {
    let summary = "Apply a Y gate to a qubit";
    let description = [{
        Applies a Y gate to a qubit and returns the transformed qubit.

        Example:
        ```mlir
        %q_out = qco.y %q_in : !qco.qubit -> !qco.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead]>:$qubit_in);
    let results = (outs QubitType:$qubit_out);
    let assemblyFormat = "$qubit_in attr-dict `:` type($qubit_in) `->` type($qubit_out)";

    let extraClassDeclaration = [{
        [[nodiscard]] static StringRef getBaseSymbol() { return "y"; }
        [[nodiscard]] static Eigen::Matrix2cd getUnitaryMatrix();
    }];

    let hasCanonicalizer = 1;
}

def ZOp : QCOOp<"z", traits = [UnitaryOpInterface, OneTargetZeroParameter]> {
    let summary = "Apply a Z gate to a qubit";
    let description = [{
        Applies a Z gate to a qubit and returns the transformed qubit.

        Example:
        ```mlir
        %q_out = qco.z %q_in : !qco.qubit -> !qco.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead]>:$qubit_in);
    let results = (outs QubitType:$qubit_out);
    let assemblyFormat = "$qubit_in attr-dict `:` type($qubit_in) `->` type($qubit_out)";

    let extraClassDeclaration = [{
        [[nodiscard]] static StringRef getBaseSymbol() { return "z"; }
        [[nodiscard]] static Eigen::Matrix2cd getUnitaryMatrix();
    }];

    let hasCanonicalizer = 1;
}

def HOp : QCOOp<"h", traits = [UnitaryOpInterface, OneTargetZeroParameter]> {
    let summary = "Apply a H gate to a qubit";
    let description = [{
        Applies a H gate to a qubit and returns the transformed qubit.

        Example:
        ```mlir
        %q_out = qco.h %q_in : !qco.qubit -> !qco.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead]>:$qubit_in);
    let results = (outs QubitType:$qubit_out);
    let assemblyFormat = "$qubit_in attr-dict `:` type($qubit_in) `->` type($qubit_out)";

    let extraClassDeclaration = [{
        [[nodiscard]] static StringRef getBaseSymbol() { return "h"; }
        [[nodiscard]] static Eigen::Matrix2cd getUnitaryMatrix();
    }];

    let hasCanonicalizer = 1;
}

def SOp : QCOOp<"s", traits = [UnitaryOpInterface, OneTargetZeroParameter]> {
    let summary = "Apply an S gate to a qubit";
    let description = [{
        Applies an S gate to a qubit and returns the transformed qubit.

        Example:
        ```mlir
        %q_out = qco.s %q_in : !qco.qubit -> !qco.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead]>:$qubit_in);
    let results = (outs QubitType:$qubit_out);
    let assemblyFormat = "$qubit_in attr-dict `:` type($qubit_in) `->` type($qubit_out)";

    let extraClassDeclaration = [{
        [[nodiscard]] static StringRef getBaseSymbol() { return "s"; }
        [[nodiscard]] static Eigen::Matrix2cd getUnitaryMatrix();
    }];

    let hasCanonicalizer = 1;
}

def SdgOp : QCOOp<"sdg", traits = [UnitaryOpInterface, OneTargetZeroParameter]> {
    let summary = "Apply an Sdg gate to a qubit";
    let description = [{
        Applies an Sdg gate to a qubit and returns the transformed qubit.

        Example:
        ```mlir
        %q_out = qco.sdg %q_in : !qco.qubit -> !qco.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead]>:$qubit_in);
    let results = (outs QubitType:$qubit_out);
    let assemblyFormat = "$qubit_in attr-dict `:` type($qubit_in) `->` type($qubit_out)";

    let extraClassDeclaration = [{
        [[nodiscard]] static StringRef getBaseSymbol() { return "sdg"; }
        [[nodiscard]] static Eigen::Matrix2cd getUnitaryMatrix();
    }];

    let hasCanonicalizer = 1;
}

def TOp : QCOOp<"t", traits = [UnitaryOpInterface, OneTargetZeroParameter]> {
    let summary = "Apply a T gate to a qubit";
    let description = [{
        Applies a T gate to a qubit and returns the transformed qubit.

        Example:
        ```mlir
        %q_out = qco.t %q_in : !qco.qubit -> !qco.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead]>:$qubit_in);
    let results = (outs QubitType:$qubit_out);
    let assemblyFormat = "$qubit_in attr-dict `:` type($qubit_in) `->` type($qubit_out)";

    let extraClassDeclaration = [{
        [[nodiscard]] static StringRef getBaseSymbol() { return "t"; }
        [[nodiscard]] static Eigen::Matrix2cd getUnitaryMatrix();
    }];

    let hasCanonicalizer = 1;
}

def TdgOp : QCOOp<"tdg", traits = [UnitaryOpInterface, OneTargetZeroParameter]> {
    let summary = "Apply a Tdg gate to a qubit";
    let description = [{
        Applies a Tdg gate to a qubit and returns the transformed qubit.

        Example:
        ```mlir
        %q_out = qco.tdg %q_in : !qco.qubit -> !qco.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead]>:$qubit_in);
    let results = (outs QubitType:$qubit_out);
    let assemblyFormat = "$qubit_in attr-dict `:` type($qubit_in) `->` type($qubit_out)";

    let extraClassDeclaration = [{
        [[nodiscard]] static StringRef getBaseSymbol() { return "tdg"; }
        [[nodiscard]] static Eigen::Matrix2cd getUnitaryMatrix();
    }];

    let hasCanonicalizer = 1;
}

def SXOp : QCOOp<"sx", traits = [UnitaryOpInterface, OneTargetZeroParameter]> {
    let summary = "Apply an SX gate to a qubit";
    let description = [{
        Applies an SX gate to a qubit and returns the transformed qubit.

        Example:
        ```mlir
        %q_out = qco.sx %q_in : !qco.qubit -> !qco.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead]>:$qubit_in);
    let results = (outs QubitType:$qubit_out);
    let assemblyFormat = "$qubit_in attr-dict `:` type($qubit_in) `->` type($qubit_out)";

    let extraClassDeclaration = [{
        [[nodiscard]] static StringRef getBaseSymbol() { return "sx"; }
        [[nodiscard]] static Eigen::Matrix2cd getUnitaryMatrix();
    }];

    let hasCanonicalizer = 1;
}

def SXdgOp : QCOOp<"sxdg", traits = [UnitaryOpInterface, OneTargetZeroParameter]> {
    let summary = "Apply an SXdg gate to a qubit";
    let description = [{
        Applies an SXdg gate to a qubit and returns the transformed qubit.

        Example:
        ```mlir
        %q_out = qco.sxdg %q_in : !qco.qubit -> !qco.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead]>:$qubit_in);
    let results = (outs QubitType:$qubit_out);
    let assemblyFormat = "$qubit_in attr-dict `:` type($qubit_in) `->` type($qubit_out)";

    let extraClassDeclaration = [{
        [[nodiscard]] static StringRef getBaseSymbol() { return "sxdg"; }
        [[nodiscard]] static Eigen::Matrix2cd getUnitaryMatrix();
    }];

    let hasCanonicalizer = 1;
}

def RXOp : QCOOp<"rx", traits = [UnitaryOpInterface, OneTargetOneParameter]> {
    let summary = "Apply an RX gate to a qubit";
    let description = [{
        Applies an RX gate to a qubit and returns the transformed qubit.

        Example:
        ```mlir
        %q_out = qco.rx(%theta) %q_in : !qco.qubit -> !qco.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead]>:$qubit_in,
                     Arg<F64, "the rotation angle theta in radians">:$theta);
    let results = (outs QubitType:$qubit_out);
    let assemblyFormat = "`(` $theta `)` $qubit_in attr-dict `:` type($qubit_in) `->` type($qubit_out)";

    let extraClassDeclaration = [{
        [[nodiscard]] static StringRef getBaseSymbol() { return "rx"; }
        [[nodiscard]] std::optional<Eigen::Matrix2cd> getUnitaryMatrix();
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit_in, "const std::variant<double, Value>&":$theta)>
    ];

    let hasCanonicalizer = 1;
}

def RYOp : QCOOp<"ry", traits = [UnitaryOpInterface, OneTargetOneParameter]> {
    let summary = "Apply an RY gate to a qubit";
    let description = [{
        Applies an RY gate to a qubit and returns the transformed qubit.

        Example:
        ```mlir
        %q_out = qco.ry(%theta) %q_in : !qco.qubit -> !qco.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead]>:$qubit_in,
                     Arg<F64, "the rotation angle theta in radians">:$theta);
    let results = (outs QubitType:$qubit_out);
    let assemblyFormat = "`(` $theta `)` $qubit_in attr-dict `:` type($qubit_in) `->` type($qubit_out)";

    let extraClassDeclaration = [{
        [[nodiscard]] static StringRef getBaseSymbol() { return "ry"; }
        [[nodiscard]] std::optional<Eigen::Matrix2cd> getUnitaryMatrix();
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit_in, "const std::variant<double, Value>&":$theta)>
    ];

    let hasCanonicalizer = 1;
}

def RZOp : QCOOp<"rz", traits = [UnitaryOpInterface, OneTargetOneParameter]> {
    let summary = "Apply an RZ gate to a qubit";
    let description = [{
        Applies an RZ gate to a qubit and returns the transformed qubit.

        Example:
        ```mlir
        %q_out = qco.rz(%theta) %q_in : !qco.qubit -> !qco.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead]>:$qubit_in,
                     Arg<F64, "the rotation angle theta in radians">:$theta);
    let results = (outs QubitType:$qubit_out);
    let assemblyFormat = "`(` $theta `)` $qubit_in attr-dict `:` type($qubit_in) `->` type($qubit_out)";

    let extraClassDeclaration = [{
        [[nodiscard]] static StringRef getBaseSymbol() { return "rz"; }
        [[nodiscard]] std::optional<Eigen::Matrix2cd> getUnitaryMatrix();
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit_in, "const std::variant<double, Value>&":$theta)>
    ];

    let hasCanonicalizer = 1;
}

def POp : QCOOp<"p", traits = [UnitaryOpInterface, OneTargetOneParameter]> {
    let summary = "Apply a P gate to a qubit";
    let description = [{
        Applies a P gate to a qubit and returns the transformed qubit.

        Example:
        ```mlir
        %q_out = qco.p(%theta) %q_in : !qco.qubit -> !qco.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead]>:$qubit_in,
                     Arg<F64, "the rotation angle theta in radians">:$theta);
    let results = (outs QubitType:$qubit_out);
    let assemblyFormat = "`(` $theta `)` $qubit_in attr-dict `:` type($qubit_in) `->` type($qubit_out)";

    let extraClassDeclaration = [{
        [[nodiscard]] static StringRef getBaseSymbol() { return "p"; }
        [[nodiscard]] std::optional<Eigen::Matrix2cd> getUnitaryMatrix();
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit_in, "const std::variant<double, Value>&":$theta)>
    ];

    let hasCanonicalizer = 1;
}

def ROp : QCOOp<"r", traits = [UnitaryOpInterface, OneTargetTwoParameter]> {
    let summary = "Apply an R gate to a qubit";
    let description = [{
        Applies an R gate to a qubit and returns the transformed qubit.

        Example:
        ```mlir
        %q_out = qco.r(%theta, %phi) %q_in : !qco.qubit -> !qco.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead]>:$qubit_in,
                     Arg<F64, "the rotation angle theta in radians">:$theta,
                     Arg<F64, "the rotation angle phi in radians">:$phi);
    let results = (outs QubitType:$qubit_out);
    let assemblyFormat = "`(` $theta `,` $phi `)` $qubit_in attr-dict `:` type($qubit_in) `->` type($qubit_out)";

    let extraClassDeclaration = [{
        [[nodiscard]] static StringRef getBaseSymbol() { return "r"; }
        [[nodiscard]] std::optional<Eigen::Matrix2cd> getUnitaryMatrix();
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit_in, "const std::variant<double, Value>&":$theta, "const std::variant<double, Value>&":$phi)>
    ];

    let hasCanonicalizer = 1;
}

def U2Op : QCOOp<"u2", traits = [UnitaryOpInterface, OneTargetTwoParameter]> {
    let summary = "Apply a U2 gate to a qubit";
    let description = [{
        Applies a U2 gate to a qubit and returns the transformed qubit.

        Example:
        ```mlir
        %q_out = qco.u2(%phi, %lambda) %q_in : !qco.qubit -> !qco.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead]>:$qubit_in,
                     Arg<F64, "the rotation angle phi in radians">:$phi,
                     Arg<F64, "the rotation angle lambda in radians">:$lambda);
    let results = (outs QubitType:$qubit_out);
    let assemblyFormat = "`(` $phi `,` $lambda `)` $qubit_in attr-dict `:` type($qubit_in) `->` type($qubit_out)";

    let extraClassDeclaration = [{
        [[nodiscard]] static StringRef getBaseSymbol() { return "u2"; }
        [[nodiscard]] std::optional<Eigen::Matrix2cd> getUnitaryMatrix();
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit_in, "const std::variant<double, Value>&":$phi, "const std::variant<double, Value>&":$lambda)>
    ];

    let hasCanonicalizer = 1;
}

def UOp : QCOOp<"u", traits = [UnitaryOpInterface, OneTargetThreeParameter]> {
    let summary = "Apply a U gate to a qubit";
    let description = [{
        Applies a U gate to a qubit and returns the transformed qubit.

        Example:
        ```mlir
        %q_out = qco.u(%theta, %phi, %lambda) %q_in : !qco.qubit -> !qco.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the target qubit", [MemRead]>:$qubit_in,
                     Arg<F64, "the rotation angle theta in radians">:$theta,
                     Arg<F64, "the rotation angle phi in radians">:$phi,
                     Arg<F64, "the rotation angle lambda in radians">:$lambda);
    let results = (outs QubitType:$qubit_out);
    let assemblyFormat = "`(` $theta `,` $phi `,` $lambda `)` $qubit_in attr-dict `:` type($qubit_in) `->` type($qubit_out)";

    let extraClassDeclaration = [{
        [[nodiscard]] static StringRef getBaseSymbol() { return "u"; }
        [[nodiscard]] std::optional<Eigen::Matrix2cd> getUnitaryMatrix();
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit_in, "const std::variant<double, Value>&":$theta, "const std::variant<double, Value>&":$phi, "const std::variant<double, Value>&":$lambda)>
    ];

    let hasCanonicalizer = 1;
}

def SWAPOp : QCOOp<"swap", traits = [UnitaryOpInterface, TwoTargetZeroParameter]> {
    let summary = "Apply a SWAP gate to two qubits";
    let description = [{
        Applies a SWAP gate to two qubits and returns the transformed qubits.

        Example:
        ```mlir
        %q0_out, %q1_out = qco.swap %q0_in, %q1_in : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the first target qubit", [MemRead]>:$qubit0_in,
                     Arg<QubitType, "the second target qubit", [MemRead]>:$qubit1_in);
    let results = (outs QubitType:$qubit0_out, QubitType:$qubit1_out);
    let assemblyFormat = "$qubit0_in `,` $qubit1_in attr-dict `:` type($qubit0_in) `,` type($qubit1_in) `->` type($qubit0_out) `,` type($qubit1_out)";

    let extraClassDeclaration = [{
        [[nodiscard]] static StringRef getBaseSymbol() { return "swap"; }
        [[nodiscard]] static Eigen::Matrix4cd getUnitaryMatrix();
    }];

    let hasCanonicalizer = 1;
}

def iSWAPOp : QCOOp<"iswap", traits = [UnitaryOpInterface, TwoTargetZeroParameter]> {
    let summary = "Apply a iSWAP gate to two qubits";
    let description = [{
        Applies a iSWAP gate to two qubits and returns the transformed qubits.

        Example:
        ```mlir
        %q0_out, %q1_out = qco.iswap %q0_in, %q1_in : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the first target qubit", [MemRead]>:$qubit0_in,
                     Arg<QubitType, "the second target qubit", [MemRead]>:$qubit1_in);
    let results = (outs QubitType:$qubit0_out, QubitType:$qubit1_out);
    let assemblyFormat = "$qubit0_in `,` $qubit1_in attr-dict `:` type($qubit0_in) `,` type($qubit1_in) `->` type($qubit0_out) `,` type($qubit1_out)";

    let extraClassDeclaration = [{
        [[nodiscard]] static StringRef getBaseSymbol() { return "iswap"; }
        [[nodiscard]] static Eigen::Matrix4cd getUnitaryMatrix();
    }];
}

def DCXOp : QCOOp<"dcx", traits = [UnitaryOpInterface, TwoTargetZeroParameter]> {
    let summary = "Apply a DCX gate to two qubits";
    let description = [{
        Applies a DCX gate to two qubits and returns the transformed qubits.

        Example:
        ```mlir
        %q0_out, %q1_out = qco.dcx %q0_in, %q1_in : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the first target qubit", [MemRead]>:$qubit0_in,
                     Arg<QubitType, "the second target qubit", [MemRead]>:$qubit1_in);
    let results = (outs QubitType:$qubit0_out, QubitType:$qubit1_out);
    let assemblyFormat = "$qubit0_in `,` $qubit1_in attr-dict `:` type($qubit0_in) `,` type($qubit1_in) `->` type($qubit0_out) `,` type($qubit1_out)";

    let extraClassDeclaration = [{
        [[nodiscard]] static StringRef getBaseSymbol() { return "dcx"; }
        [[nodiscard]] static Eigen::Matrix4cd getUnitaryMatrix();
    }];
}

def ECROp : QCOOp<"ecr", traits = [UnitaryOpInterface, TwoTargetZeroParameter]> {
    let summary = "Apply an ECR gate to two qubits";
    let description = [{
        Applies an ECR gate to two qubits and returns the transformed qubits.

        Example:
        ```mlir
        %q0_out, %q1_out = qco.ecr %q0_in, %q1_in : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the first target qubit", [MemRead]>:$qubit0_in,
                     Arg<QubitType, "the second target qubit", [MemRead]>:$qubit1_in);
    let results = (outs QubitType:$qubit0_out, QubitType:$qubit1_out);
    let assemblyFormat = "$qubit0_in `,` $qubit1_in attr-dict `:` type($qubit0_in) `,` type($qubit1_in) `->` type($qubit0_out) `,` type($qubit1_out)";

    let extraClassDeclaration = [{
        [[nodiscard]] static StringRef getBaseSymbol() { return "ecr"; }
        [[nodiscard]] static Eigen::Matrix4cd getUnitaryMatrix();
    }];

    let hasCanonicalizer = 1;
}

def RXXOp : QCOOp<"rxx", traits = [UnitaryOpInterface, TwoTargetOneParameter]> {
    let summary = "Apply an RXX gate to two qubits";
    let description = [{
        Applies an RXX gate to two qubits and returns the transformed qubits.

        Example:
        ```mlir
        %q0_out, %q1_out = qco.rxx(%theta) %q0_in, %q1_in : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the first target qubit", [MemRead]>:$qubit0_in,
                     Arg<QubitType, "the second target qubit", [MemRead]>:$qubit1_in,
                     Arg<F64, "the rotation angle theta in radians">:$theta);
    let results = (outs QubitType:$qubit0_out, QubitType:$qubit1_out);
    let assemblyFormat = "`(` $theta `)` $qubit0_in `,` $qubit1_in attr-dict `:` type($qubit0_in) `,` type($qubit1_in) `->` type($qubit0_out) `,` type($qubit1_out)";

    let extraClassDeclaration = [{
        [[nodiscard]] static StringRef getBaseSymbol() { return "rxx"; }
        [[nodiscard]] std::optional<Eigen::Matrix4cd> getUnitaryMatrix();
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit0_in, "Value":$qubit1_in, "const std::variant<double, Value>&":$theta)>
    ];

    let hasCanonicalizer = 1;
}

def RYYOp : QCOOp<"ryy", traits = [UnitaryOpInterface, TwoTargetOneParameter]> {
    let summary = "Apply an RYY gate to two qubits";
    let description = [{
        Applies an RYY gate to two qubits and returns the transformed qubits.

        Example:
        ```mlir
        %q0_out, %q1_out = qco.ryy(%theta) %q0_in, %q1_in : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the first target qubit", [MemRead]>:$qubit0_in,
                     Arg<QubitType, "the second target qubit", [MemRead]>:$qubit1_in,
                     Arg<F64, "the rotation angle theta in radians">:$theta);
    let results = (outs QubitType:$qubit0_out, QubitType:$qubit1_out);
    let assemblyFormat = "`(` $theta `)` $qubit0_in `,` $qubit1_in attr-dict `:` type($qubit0_in) `,` type($qubit1_in) `->` type($qubit0_out) `,` type($qubit1_out)";

    let extraClassDeclaration = [{
        [[nodiscard]] static StringRef getBaseSymbol() { return "ryy"; }
        [[nodiscard]] std::optional<Eigen::Matrix4cd> getUnitaryMatrix();
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit0_in, "Value":$qubit1_in, "const std::variant<double, Value>&":$theta)>
    ];

    let hasCanonicalizer = 1;
}

def RZXOp : QCOOp<"rzx", traits = [UnitaryOpInterface, TwoTargetOneParameter]> {
    let summary = "Apply an RZX gate to two qubits";
    let description = [{
        Applies an RZX gate to two qubits and returns the transformed qubits.

        Example:
        ```mlir
        %q0_out, %q1_out = qco.rzx(%theta) %q0_in, %q1_in : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the first target qubit", [MemRead]>:$qubit0_in,
                     Arg<QubitType, "the second target qubit", [MemRead]>:$qubit1_in,
                     Arg<F64, "the rotation angle theta in radians">:$theta);
    let results = (outs QubitType:$qubit0_out, QubitType:$qubit1_out);
    let assemblyFormat = "`(` $theta `)` $qubit0_in `,` $qubit1_in attr-dict `:` type($qubit0_in) `,` type($qubit1_in) `->` type($qubit0_out) `,` type($qubit1_out)";

    let extraClassDeclaration = [{
        [[nodiscard]] static StringRef getBaseSymbol() { return "rzx"; }
        [[nodiscard]] std::optional<Eigen::Matrix4cd> getUnitaryMatrix();
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit0_in, "Value":$qubit1_in, "const std::variant<double, Value>&":$theta)>
    ];

    let hasCanonicalizer = 1;
}

def RZZOp : QCOOp<"rzz", traits = [UnitaryOpInterface, TwoTargetOneParameter]> {
    let summary = "Apply an RZZ gate to two qubits";
    let description = [{
        Applies an RZZ gate to two qubits and returns the transformed qubits.

        Example:
        ```mlir
        %q0_out, %q1_out = qco.rzz(%theta) %q0_in, %q1_in : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the first target qubit", [MemRead]>:$qubit0_in,
                     Arg<QubitType, "the second target qubit", [MemRead]>:$qubit1_in,
                     Arg<F64, "the rotation angle theta in radians">:$theta);
    let results = (outs QubitType:$qubit0_out, QubitType:$qubit1_out);
    let assemblyFormat = "`(` $theta `)` $qubit0_in `,` $qubit1_in attr-dict `:` type($qubit0_in) `,` type($qubit1_in) `->` type($qubit0_out) `,` type($qubit1_out)";

    let extraClassDeclaration = [{
        [[nodiscard]] static StringRef getBaseSymbol() { return "rzz"; }
        [[nodiscard]] std::optional<Eigen::Matrix4cd> getUnitaryMatrix();
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit0_in, "Value":$qubit1_in, "const std::variant<double, Value>&":$theta)>
    ];

    let hasCanonicalizer = 1;
}

def XXPlusYYOp : QCOOp<"xx_plus_yy", traits = [UnitaryOpInterface, TwoTargetTwoParameter]> {
    let summary = "Apply an XX+YY gate to two qubits";
    let description = [{
        Applies an XX+YY gate to two qubits and returns the transformed qubits.

        Example:
        ```mlir
        %q0_out, %q1_out = qco.xx_plus_yy(%theta, %beta) %q0_in, %q1_in : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the first target qubit", [MemRead]>:$qubit0_in,
                     Arg<QubitType, "the second target qubit", [MemRead]>:$qubit1_in,
                     Arg<F64, "the rotation angle theta in radians">:$theta,
                     Arg<F64, "the rotation angle beta in radians">:$beta);
    let results = (outs QubitType:$qubit0_out, QubitType:$qubit1_out);
    let assemblyFormat = "`(` $theta `,` $beta `)` $qubit0_in `,` $qubit1_in attr-dict `:` type($qubit0_in) `,` type($qubit1_in) `->` type($qubit0_out) `,` type($qubit1_out)";

    let extraClassDeclaration = [{
        [[nodiscard]] static StringRef getBaseSymbol() { return "xx_plus_yy"; }
        [[nodiscard]] std::optional<Eigen::Matrix4cd> getUnitaryMatrix();
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit0_in, "Value":$qubit1_in, "const std::variant<double, Value>&":$theta, "const std::variant<double, Value>&":$beta)>
    ];

    let hasCanonicalizer = 1;
}

def XXMinusYYOp : QCOOp<"xx_minus_yy", traits = [UnitaryOpInterface, TwoTargetTwoParameter]> {
    let summary = "Apply an XX-YY gate to two qubits";
    let description = [{
        Applies an XX-YY gate to two qubits and returns the transformed qubits.

        Example:
        ```mlir
        %q0_out, %q1_out = qco.xx_minus_yy(%theta, %beta) %q0_in, %q1_in : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
        ```
    }];

    let arguments = (ins Arg<QubitType, "the first target qubit", [MemRead]>:$qubit0_in,
                     Arg<QubitType, "the second target qubit", [MemRead]>:$qubit1_in,
                     Arg<F64, "the rotation angle theta in radians">:$theta,
                     Arg<F64, "the rotation angle beta in radians">:$beta);
    let results = (outs QubitType:$qubit0_out, QubitType:$qubit1_out);
    let assemblyFormat = "`(` $theta `,` $beta `)` $qubit0_in `,` $qubit1_in attr-dict `:` type($qubit0_in) `,` type($qubit1_in) `->` type($qubit0_out) `,` type($qubit1_out)";

    let extraClassDeclaration = [{
        [[nodiscard]] static StringRef getBaseSymbol() { return "xx_minus_yy"; }
        [[nodiscard]] std::optional<Eigen::Matrix4cd> getUnitaryMatrix();
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit0_in, "Value":$qubit1_in, "const std::variant<double, Value>&":$theta, "const std::variant<double, Value>&":$beta)>
    ];

    let hasCanonicalizer = 1;
}

def BarrierOp : QCOOp<"barrier", traits = [UnitaryOpInterface]> {
    let summary = "Apply a barrier gate to a set of qubits";
    let description = [{
        Applies a barrier gate to a set of qubits and returns the transformed qubits.

        Example:
        ```mlir
        %q_out = qco.barrier %q_in : !qco.qubit -> !qco.qubit
        ```
    }];

    let arguments = (ins Arg<Variadic<QubitType>, "the target qubits", [MemRead]>:$qubits_in);
    let results = (outs Variadic<QubitType>:$qubits_out);
    let assemblyFormat = "$qubits_in attr-dict `:` type($qubits_in) `->` type($qubits_out)";

    let extraClassDeclaration = [{
        size_t getNumQubits() { return getNumTargets(); }
        size_t getNumTargets() { return getQubitsIn().size(); }
        static size_t getNumControls() { return 0; }
        Value getInputQubit(size_t i) { return getInputTarget(i); }
        OperandRange getInputQubits() { return getQubitsIn(); }
        Value getOutputQubit(size_t i) { return getOutputTarget(i); }
        ResultRange getOutputQubits() { return getQubitsOut(); }
        Value getInputTarget(size_t i);
        Value getOutputTarget(size_t i);
        static Value getInputControl(size_t i) { llvm::reportFatalUsageError("BarrierOp cannot be controlled"); }
        static Value getOutputControl(size_t i) { llvm::reportFatalUsageError("BarrierOp cannot be controlled"); }
        Value getInputForOutput(Value output);
        Value getOutputForInput(Value input);
        static size_t getNumParams() { return 0; }
        static Value getParameter(size_t i) { llvm::reportFatalUsageError("BarrierOp has no parameters"); }
        [[nodiscard]] static StringRef getBaseSymbol() { return "barrier"; }
    }];

    let builders = [
        OpBuilder<(ins "ValueRange":$qubits)>
    ];

    let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Modifiers
//===----------------------------------------------------------------------===//

def YieldOp : QCOOp<"yield", traits = [Terminator, ReturnLike]> {
    let summary = "Yield from a modifier region";
    let description = [{
        Terminates a modifier region, yielding the transformed target qubits back to the enclosing modifier operation.
        The targets must match the expected output signature of the modifier region.

        Example:
        ```mlir
        %res_ctrl, %res_tgt:2 = qco.ctrl(%ctrl) targets(%a0 = %q0, %a1 = %q1) {
            %a0_1, %a1_1 = qco.swap %a0, %a1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
            qco.yield %a0_1, %a1_1
        } : ({!qco.qubit}, {!qco.qubit, !qco.qubit}) -> ({!qco.qubit}, {!qco.qubit, !qco.qubit})
        ```
    }];

    let arguments = (ins Variadic<QubitType>:$targets);
    let assemblyFormat = "$targets attr-dict";
}

def CtrlOp : QCOOp<"ctrl", traits =
            [
                UnitaryOpInterface,
                AttrSizedOperandSegments,
                AttrSizedResultSegments,
                SameOperandsAndResultType,
                SameOperandsAndResultShape,
                SingleBlockImplicitTerminator<"::mlir::qco::YieldOp">,
                RecursiveMemoryEffects
            ]> {
    let summary = "Add control qubits to a unitary operation";
    let description = [{
        A modifier operation that adds control qubits to the unitary operation
        defined in its body region. The controlled operation applies the
        underlying unitary only when all control qubits are in the |1⟩ state.
        The operation takes a variadic number of control and target qubits as
        inputs and produces corresponding output qubits. Control qubits are not
        modified by the operation and simply pass through to the outputs.

        Example:
        ```mlir
        %res_ctrl, %res_tgt:2 = qco.ctrl(%ctrl) targets(%a0 = %q0, %a1 = %q1) {
            %a0_1, %a1_1 = qco.swap %a0, %a1 : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
            qco.yield %a0_1, %a1_1
        } : ({!qco.qubit}, {!qco.qubit, !qco.qubit}) -> ({!qco.qubit}, {!qco.qubit, !qco.qubit})
        ```
    }];

    let arguments = (ins Arg<Variadic<QubitType>, "the control qubits", [MemRead]>:$controls_in,
                     Arg<Variadic<QubitType>, "the target qubits", [MemRead]>:$targets_in);
    let results = (outs Variadic<QubitType>:$controls_out, Variadic<QubitType>:$targets_out);
    let regions = (region SizedRegion<1>:$region);
    let assemblyFormat = [{
        `(` $controls_in `)`
        `targets`
        custom<TargetAliasing>($region, $targets_in)
        attr-dict `:`
        `(` `{` type($controls_in) `}` ( `,` `{` type($targets_in)^ `}` )? `)`
        `->`
        `(` `{` type($controls_out) `}` ( `,` `{` type($targets_out)^ `}` )? `)`
    }];

    let extraClassDeclaration = [{
        UnitaryOpInterface getBodyUnitary();
        size_t getNumQubits() { return getNumControls() + getNumTargets(); }
        size_t getNumTargets() { return getTargetsIn().size(); }
        size_t getNumControls() { return getControlsIn().size(); }
        Value getInputQubit(size_t i);
        OperandRange getInputQubits() { return getOperands(); }
        Value getOutputQubit(size_t i);
        ResultRange getOutputQubits() { return getResults(); }
        Value getInputTarget(size_t i);
        Value getOutputTarget(size_t i);
        Value getInputControl(size_t i);
        Value getOutputControl(size_t i);
        Value getInputForOutput(Value output);
        Value getOutputForInput(Value input);
        size_t getNumParams() { return getBodyUnitary().getNumParams(); }
        Value getParameter(size_t i) { return getBodyUnitary().getParameter(i); }
        [[nodiscard]] static StringRef getBaseSymbol() { return "ctrl"; }
        [[nodiscard]] std::optional<Eigen::MatrixXcd> getUnitaryMatrix();
    }];

    let builders = [
        OpBuilder<(ins "ValueRange":$controls, "ValueRange":$targets), [{
           build($_builder, $_state, controls.getTypes(), targets.getTypes(), controls, targets);
        }]>,
        OpBuilder<(ins "ValueRange":$controls, "ValueRange":$targets, "llvm::function_ref<llvm::SmallVector<Value>(ValueRange)>":$bodyBuilder)>
    ];

    let hasCanonicalizer = 1;
    let hasVerifier = 1;
}

def InvOp : QCOOp<"inv",
            traits = [
                UnitaryOpInterface,
                SingleBlockImplicitTerminator<"::mlir::qco::YieldOp">,
                RecursiveMemoryEffects
            ]> {
    let summary = "Invert a unitary operation";
    let description = [{
        A modifier operation that inverts the unitary operation defined in its body
        region. The operation takes a variadic number of qubits as inputs and
        produces corresponding output qubits.

        Example:
        ```mlir
        %q_out = qco.inv (%q = %q_in) {
            %q_1 = qco.s %q : !qco.qubit -> !qco.qubit
            qco.yield %q_1
        } : {!qco.qubit} -> {!qco.qubit}
        ```
    }];

    let arguments = (ins Arg<Variadic<QubitType>, "the qubits involved in the operation", [MemRead]>:$qubits_in);
    let results = (outs Variadic<QubitType>:$qubits_out);
    let regions = (region SizedRegion<1>:$region);
    let assemblyFormat = [{
        custom<TargetAliasing>($region, $qubits_in)
        attr-dict `:`
        `{` type($qubits_in) `}`
        `->`
        `{` type($qubits_out) `}`
    }];

    let extraClassDeclaration = [{
        UnitaryOpInterface getBodyUnitary();
        size_t getNumQubits() { return getNumTargets(); }
        size_t getNumTargets() { return getQubitsIn().size(); }
        static size_t getNumControls() { return 0; }
        Value getInputQubit(size_t i);
        OperandRange getInputQubits() { return getOperands(); }
        Value getOutputQubit(size_t i);
        ResultRange getOutputQubits() { return getResults(); }
        Value getInputTarget(size_t i) { return getInputQubit(i); }
        Value getOutputTarget(size_t i) { return getOutputQubit(i); }
        static Value getInputControl(size_t i) { llvm::reportFatalUsageError("Operation does not have controls"); }
        static Value getOutputControl(size_t i) { llvm::reportFatalUsageError("Operation does not have controls"); }
        Value getInputForOutput(Value output);
        Value getOutputForInput(Value input);
        size_t getNumParams() { return getBodyUnitary().getNumParams(); }
        Value getParameter(size_t i) { return getBodyUnitary().getParameter(i); }
        [[nodiscard]] static StringRef getBaseSymbol() { return "inv"; }
        [[nodiscard]] std::optional<Eigen::MatrixXcd> getUnitaryMatrix();
    }];

    let builders = [
        OpBuilder<(ins "ValueRange":$qubits), [{
           build($_builder, $_state, qubits.getTypes(), qubits);
        }]>,
        OpBuilder<(ins "ValueRange":$qubits, "llvm::function_ref<llvm::SmallVector<Value>(ValueRange)>":$bodyBuilder)>
    ];

    let hasCanonicalizer = 1;
    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SCF operations
//===----------------------------------------------------------------------===//

def IfOp : QCOOp<"if", traits =
            [
                DeclareOpInterfaceMethods<RegionBranchOpInterface, [
                "getNumRegionInvocations", "getRegionInvocationBounds",
                "getEntrySuccessorRegions"]>,
                SingleBlock,
                SingleBlockImplicitTerminator<"::mlir::qco::YieldOp">,
                RecursiveMemoryEffects
            ]> {

    let summary = "If-then-else operation for linear (qubit) types";
    let description = [{
        The `qco.if` operation is an if-then-else construct similar to the standard scf.if operation.
        In addition to the condition, the operation takes a variadic number of qubits as inputs that are
        required in the bodies of both branches. These qubits are passed down to the individual regions
        as block arguments. The number of results and the type of the results must be equivalent to the
        number and types of the input qubits.

        Example:
        ```mlir
        %result = qco.if %condition qubits(%arg0 = %q0) {
            %q1 = qco.h %arg0 : !qco.qubit -> !qco.qubit
            qco.yield %q1
         } else qubits(%arg0 = %q0) {
            qco.yield %arg0
        } : {i1, !qco.qubit} -> {!qco.qubit}
        ```
    }];

    let arguments = (ins I1:$condition,  Variadic<QubitType>:$qubits);
    let results = (outs Variadic<QubitType>:$results);
    let regions = (region SizedRegion<1>:$thenRegion,
                          SizedRegion<1>:$elseRegion);

    let assemblyFormat = [{
        $condition
         custom<IfOpAliasing>($thenRegion, $elseRegion, $qubits)
         attr-dict `:`
        `{` type($condition) `,` type($qubits) `}`
        `->`
        `{` type($results) `}`
    }];

    let builders = [
        OpBuilder<(ins "Value":$condition, "ValueRange":$qubits), [{
           build($_builder, $_state, qubits.getTypes(), condition, qubits);
        }]>,
        OpBuilder<(ins "Value":$condition, "ValueRange":$qubits, "llvm::function_ref<llvm::SmallVector<Value>(ValueRange)>":$thenBuilder,
        CArg<"llvm::function_ref<llvm::SmallVector<Value>(ValueRange)>","nullptr">:$elseBuilder)>
    ];

    let extraClassDeclaration = [{
        Block *thenBlock() {
           return &getThenRegion().back();
        }
        YieldOp thenYield();
        Block* elseBlock() {
           return &getElseRegion().back();
        }
        YieldOp elseYield();
    }];

    let extraClassDefinition = [{
        YieldOp $cppClass::thenYield() {
           return cast<YieldOp>(&thenBlock()->back());
        }
        YieldOp $cppClass::elseYield() {
           return cast<YieldOp>(&elseBlock()->back());
        }
    }];

    let hasCanonicalizer = 1;
    let hasVerifier = 1;
}

#endif // QCOOPS
