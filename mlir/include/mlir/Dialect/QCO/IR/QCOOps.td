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

class MatrixDefinitionLambda<code MatrixSize, code MatrixDefinitionBody> {
  code MatrixType = "Eigen::Matrix<std::complex<double>, " # MatrixSize # ", " # MatrixSize # ">";
  code Code = !cond(!empty(MatrixDefinitionBody): "nullptr", true: "[](UnitaryOpInterface op) -> " # MatrixType # " { " # MatrixDefinitionBody # " }");
}
class FixedSizeMatrixDefinitionLambda<int T, code MatrixDefinitionBody> : MatrixDefinitionLambda<"1 << " # !cast<string>(T), MatrixDefinitionBody>;

class TargetAndParameterArityTrait<int T, int P, MatrixDefinitionLambda MatrixDefinition = FixedSizeMatrixDefinitionLambda<T, [{}]>>
        : ParamNativeOpTrait<"TargetAndParameterArityTrait", !cast<string>(T) # ", " # !cast<string>(P) # ", " # MatrixDefinition.MatrixType # ", " # MatrixDefinition.Code> {
    let cppNamespace = "::mlir::qco";
}

def ZeroTargetZeroParameter : TargetAndParameterArityTrait<0, 0>;
def ZeroTargetOneParameter : TargetAndParameterArityTrait<0, 1>;
def OneTargetZeroParameter : TargetAndParameterArityTrait<1, 0>;
def OneTargetOneParameter : TargetAndParameterArityTrait<1, 1>;
def OneTargetTwoParameter : TargetAndParameterArityTrait<1, 2>;
def OneTargetThreeParameter : TargetAndParameterArityTrait<1, 3>;
def TwoTargetZeroParameter : TargetAndParameterArityTrait<2, 0>;
def TwoTargetOneParameter : TargetAndParameterArityTrait<2, 1>;
def TwoTargetTwoParameter : TargetAndParameterArityTrait<2, 2>;

class DynamicTargetZeroParameterUnitaryMatrix<code MatrixDefinitionBody> : TargetAndParameterArityTrait<0, 0, MatrixDefinitionLambda<"Eigen::Dynamic", MatrixDefinitionBody>>;

class ZeroTargetOneParameterUnitaryMatrix<code MatrixDefinitionBody> : TargetAndParameterArityTrait<0, 1, FixedSizeMatrixDefinitionLambda<0, MatrixDefinitionBody>>;
class OneTargetZeroParameterUnitaryMatrix<code MatrixDefinitionBody> : TargetAndParameterArityTrait<1, 0, FixedSizeMatrixDefinitionLambda<1, MatrixDefinitionBody>>;
class OneTargetOneParameterUnitaryMatrix<code MatrixDefinitionBody> : TargetAndParameterArityTrait<1, 1, FixedSizeMatrixDefinitionLambda<1, MatrixDefinitionBody>>;
class OneTargetTwoParameterUnitaryMatrix<code MatrixDefinitionBody> : TargetAndParameterArityTrait<1, 2, FixedSizeMatrixDefinitionLambda<1, MatrixDefinitionBody>>;
class OneTargetThreeParameterUnitaryMatrix<code MatrixDefinitionBody> : TargetAndParameterArityTrait<1, 3, FixedSizeMatrixDefinitionLambda<1, MatrixDefinitionBody>>;
class TwoTargetZeroParameterUnitaryMatrix<code MatrixDefinitionBody> : TargetAndParameterArityTrait<2, 0, FixedSizeMatrixDefinitionLambda<2, MatrixDefinitionBody>>;
class TwoTargetOneParameterUnitaryMatrix<code MatrixDefinitionBody> : TargetAndParameterArityTrait<2, 1, FixedSizeMatrixDefinitionLambda<2, MatrixDefinitionBody>>;
class TwoTargetTwoParameterUnitaryMatrix<code MatrixDefinitionBody> : TargetAndParameterArityTrait<2, 2, FixedSizeMatrixDefinitionLambda<2, MatrixDefinitionBody>>;

//===----------------------------------------------------------------------===//
// Unitary Operations
//===----------------------------------------------------------------------===//

def GPhaseOp : QCOOp<"gphase", traits = [UnitaryOpInterface, ZeroTargetOneParameterUnitaryMatrix<[{ return utils::getMatrixGPhase(tryGetParameterAsDouble(op, 0).value()); }]>, MemoryEffects<[MemWrite]>]> {
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
        static StringRef getBaseSymbol() { return "gphase"; }
    }];

    let builders = [
        OpBuilder<(ins "const std::variant<double, Value>&":$theta)>
    ];

    let hasCanonicalizer = 1;
}

def IdOp : QCOOp<"id", traits = [UnitaryOpInterface, OneTargetZeroParameterUnitaryMatrix<[{ return utils::getMatrixId(); }]>]> {
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
        static StringRef getBaseSymbol() { return "id"; }
    }];

    let hasCanonicalizer = 1;
}

def XOp : QCOOp<"x", traits = [UnitaryOpInterface, OneTargetZeroParameterUnitaryMatrix<[{ return utils::getMatrixX(); }]>]> {
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
        static StringRef getBaseSymbol() { return "x"; }
    }];

    let hasCanonicalizer = 1;
}

def YOp : QCOOp<"y", traits = [UnitaryOpInterface, OneTargetZeroParameterUnitaryMatrix<[{ return utils::getMatrixY(); }]>]> {
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
        static StringRef getBaseSymbol() { return "y"; }
    }];

    let hasCanonicalizer = 1;
}

def ZOp : QCOOp<"z", traits = [UnitaryOpInterface, OneTargetZeroParameterUnitaryMatrix<[{ return utils::getMatrixZ(); }]>]> {
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
        static StringRef getBaseSymbol() { return "z"; }
    }];

    let hasCanonicalizer = 1;
}

def HOp : QCOOp<"h", traits = [UnitaryOpInterface, OneTargetZeroParameterUnitaryMatrix<[{ return utils::getMatrixH(); }]>]> {
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
        static StringRef getBaseSymbol() { return "h"; }
    }];

    let hasCanonicalizer = 1;
}

def SOp : QCOOp<"s", traits = [UnitaryOpInterface, OneTargetZeroParameterUnitaryMatrix<[{ return utils::getMatrixS(); }]>]> {
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
        static StringRef getBaseSymbol() { return "s"; }
    }];

    let hasCanonicalizer = 1;
}

def SdgOp : QCOOp<"sdg", traits = [UnitaryOpInterface, OneTargetZeroParameterUnitaryMatrix<[{ return utils::getMatrixSdg(); }]>]> {
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
        static StringRef getBaseSymbol() { return "sdg"; }
    }];

    let hasCanonicalizer = 1;
}

def TOp : QCOOp<"t", traits = [UnitaryOpInterface, OneTargetZeroParameterUnitaryMatrix<[{ return utils::getMatrixT(); }]>]> {
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
        static StringRef getBaseSymbol() { return "t"; }
    }];

    let hasCanonicalizer = 1;
}

def TdgOp : QCOOp<"tdg", traits = [UnitaryOpInterface, OneTargetZeroParameterUnitaryMatrix<[{ return utils::getMatrixTdg(); }]>]> {
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
        static StringRef getBaseSymbol() { return "tdg"; }
    }];

    let hasCanonicalizer = 1;
}

def SXOp : QCOOp<"sx", traits = [UnitaryOpInterface, OneTargetZeroParameterUnitaryMatrix<[{ return utils::getMatrixSX(); }]>]> {
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
        static StringRef getBaseSymbol() { return "sx"; }
    }];

    let hasCanonicalizer = 1;
}

def SXdgOp : QCOOp<"sxdg", traits = [UnitaryOpInterface, OneTargetZeroParameterUnitaryMatrix<[{ return utils::getMatrixSXdg(); }]>]> {
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
        static StringRef getBaseSymbol() { return "sxdg"; }
    }];

    let hasCanonicalizer = 1;
}

def RXOp : QCOOp<"rx", traits = [UnitaryOpInterface, OneTargetOneParameterUnitaryMatrix<[{ return utils::getMatrixRX(tryGetParameterAsDouble(op, 0).value()); }]>]> {
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
        static StringRef getBaseSymbol() { return "rx"; }
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit_in, "const std::variant<double, Value>&":$theta)>
    ];

    let hasCanonicalizer = 1;
}

def RYOp : QCOOp<"ry", traits = [UnitaryOpInterface, OneTargetOneParameterUnitaryMatrix<[{ return utils::getMatrixRY(tryGetParameterAsDouble(op, 0).value()); }]>]> {
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
        static StringRef getBaseSymbol() { return "ry"; }
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit_in, "const std::variant<double, Value>&":$theta)>
    ];

    let hasCanonicalizer = 1;
}

def RZOp : QCOOp<"rz", traits = [UnitaryOpInterface, OneTargetOneParameterUnitaryMatrix<[{ return utils::getMatrixRZ(tryGetParameterAsDouble(op, 0).value()); }]>]> {
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
        static StringRef getBaseSymbol() { return "rz"; }
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit_in, "const std::variant<double, Value>&":$theta)>
    ];

    let hasCanonicalizer = 1;
}

def POp : QCOOp<"p", traits = [UnitaryOpInterface, OneTargetOneParameterUnitaryMatrix<[{ return utils::getMatrixP(tryGetParameterAsDouble(op, 0).value()); }]>]> {
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
        static StringRef getBaseSymbol() { return "p"; }
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit_in, "const std::variant<double, Value>&":$theta)>
    ];

    let hasCanonicalizer = 1;
}

def ROp : QCOOp<"r", traits = [UnitaryOpInterface, OneTargetTwoParameterUnitaryMatrix<[{ return utils::getMatrixR(tryGetParameterAsDouble(op, 0).value(), tryGetParameterAsDouble(op, 1).value()); }]>]> {
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
        static StringRef getBaseSymbol() { return "r"; }
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit_in, "const std::variant<double, Value>&":$theta, "const std::variant<double, Value>&":$phi)>
    ];

    let hasCanonicalizer = 1;
}

def U2Op : QCOOp<"u2", traits = [UnitaryOpInterface, OneTargetTwoParameterUnitaryMatrix<[{ return utils::getMatrixU2(tryGetParameterAsDouble(op, 0).value(), tryGetParameterAsDouble(op, 1).value()); }]>]> {
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
        static StringRef getBaseSymbol() { return "u2"; }
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit_in, "const std::variant<double, Value>&":$phi, "const std::variant<double, Value>&":$lambda)>
    ];

    let hasCanonicalizer = 1;
}

def UOp : QCOOp<"u", traits = [UnitaryOpInterface, OneTargetThreeParameterUnitaryMatrix<[{ return utils::getMatrixU(tryGetParameterAsDouble(op, 0).value(), tryGetParameterAsDouble(op, 1).value(), tryGetParameterAsDouble(op, 2).value()); }]>]> {
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
        static StringRef getBaseSymbol() { return "u"; }
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit_in, "const std::variant<double, Value>&":$theta, "const std::variant<double, Value>&":$phi, "const std::variant<double, Value>&":$lambda)>
    ];

    let hasCanonicalizer = 1;
}

def SWAPOp : QCOOp<"swap", traits = [UnitaryOpInterface, TwoTargetZeroParameterUnitaryMatrix<[{ return utils::getMatrixSWAP(); }]>]> {
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
        static StringRef getBaseSymbol() { return "swap"; }
    }];

    let hasCanonicalizer = 1;
}

def iSWAPOp : QCOOp<"iswap", traits = [UnitaryOpInterface, TwoTargetZeroParameterUnitaryMatrix<[{ return utils::getMatrixiSWAP(); }]>]> {
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
        static StringRef getBaseSymbol() { return "iswap"; }
    }];
}

def DCXOp : QCOOp<"dcx", traits = [UnitaryOpInterface, TwoTargetZeroParameterUnitaryMatrix<[{ return utils::getMatrixDCX(); }]>]> {
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
        static StringRef getBaseSymbol() { return "dcx"; }
    }];
}

def ECROp : QCOOp<"ecr", traits = [UnitaryOpInterface, TwoTargetZeroParameterUnitaryMatrix<[{ return utils::getMatrixECR(); }]>]> {
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
        static StringRef getBaseSymbol() { return "ecr"; }
    }];

    let hasCanonicalizer = 1;
}

def RXXOp : QCOOp<"rxx", traits = [UnitaryOpInterface, TwoTargetOneParameterUnitaryMatrix<[{ return utils::getMatrixRXX(tryGetParameterAsDouble(op, 0).value()); }]>]> {
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
        static StringRef getBaseSymbol() { return "rxx"; }
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit0_in, "Value":$qubit1_in, "const std::variant<double, Value>&":$theta)>
    ];

    let hasCanonicalizer = 1;
}

def RYYOp : QCOOp<"ryy", traits = [UnitaryOpInterface, TwoTargetOneParameterUnitaryMatrix<[{ return utils::getMatrixRYY(tryGetParameterAsDouble(op, 0).value()); }]>]> {
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
        static StringRef getBaseSymbol() { return "ryy"; }
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit0_in, "Value":$qubit1_in, "const std::variant<double, Value>&":$theta)>
    ];

    let hasCanonicalizer = 1;
}

def RZXOp : QCOOp<"rzx", traits = [UnitaryOpInterface, TwoTargetOneParameterUnitaryMatrix<[{ return utils::getMatrixRZX(tryGetParameterAsDouble(op, 0).value()); }]>]> {
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
        static StringRef getBaseSymbol() { return "rzx"; }
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit0_in, "Value":$qubit1_in, "const std::variant<double, Value>&":$theta)>
    ];

    let hasCanonicalizer = 1;
}

def RZZOp : QCOOp<"rzz", traits = [UnitaryOpInterface, TwoTargetOneParameterUnitaryMatrix<[{ return utils::getMatrixRZZ(tryGetParameterAsDouble(op, 0).value()); }]>]> {
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
        static StringRef getBaseSymbol() { return "rzz"; }
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit0_in, "Value":$qubit1_in, "const std::variant<double, Value>&":$theta)>
    ];

    let hasCanonicalizer = 1;
}

def XXPlusYYOp : QCOOp<"xx_plus_yy", traits = [UnitaryOpInterface, TwoTargetTwoParameterUnitaryMatrix<[{ return utils::getMatrixXXPlusYY(tryGetParameterAsDouble(op, 0).value(), tryGetParameterAsDouble(op, 1).value()); }]>]> {
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
        static StringRef getBaseSymbol() { return "xx_plus_yy"; }
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit0_in, "Value":$qubit1_in, "const std::variant<double, Value>&":$theta, "const std::variant<double, Value>&":$beta)>
    ];

    let hasCanonicalizer = 1;
}

def XXMinusYYOp : QCOOp<"xx_minus_yy", traits = [UnitaryOpInterface, TwoTargetTwoParameterUnitaryMatrix<[{ return utils::getMatrixXXMinusYY(tryGetParameterAsDouble(op, 0).value(), tryGetParameterAsDouble(op, 1).value()); }]>]> {
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
        static StringRef getBaseSymbol() { return "xx_minus_yy"; }
    }];

    let builders = [
        OpBuilder<(ins "Value":$qubit0_in, "Value":$qubit1_in, "const std::variant<double, Value>&":$theta, "const std::variant<double, Value>&":$beta)>
    ];

    let hasCanonicalizer = 1;
}

def BarrierOp : QCOOp<"barrier", traits = [UnitaryOpInterface, ZeroTargetZeroParameter]> {
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
        size_t getNumQubits();
        size_t getNumTargets();
        static size_t getNumControls();
        Value getInputQubit(size_t i);
        Value getOutputQubit(size_t i);
        Value getInputTarget(size_t i);
        Value getOutputTarget(size_t i);
        static Value getInputControl(size_t i);
        static Value getOutputControl(size_t i);
        Value getInputForOutput(Value output);
        Value getOutputForInput(Value input);
        static size_t getNumParams();
        static Value getParameter(size_t i);
        static StringRef getBaseSymbol() { return "barrier"; }
    }];

    let builders = [
        OpBuilder<(ins "ValueRange":$qubits)>
    ];

    let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Modifiers
//===----------------------------------------------------------------------===//

def YieldOp : QCOOp<"yield", traits = [Terminator]> {
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
                DynamicTargetZeroParameterUnitaryMatrix<[{ return utils::getMatrixCtrl(op.getNumControls(), getControlledOp(op).getUnitaryMatrix()); }]>,
                AttrSizedOperandSegments,
                AttrSizedResultSegments,
                SameOperandsAndResultType,
                SameOperandsAndResultShape,
                SingleBlock,
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
        size_t getNumQubits();
        size_t getNumTargets();
        size_t getNumControls();
        Value getInputQubit(size_t i);
        Value getOutputQubit(size_t i);
        Value getInputTarget(size_t i);
        Value getOutputTarget(size_t i);
        Value getInputControl(size_t i);
        Value getOutputControl(size_t i);
        Value getInputForOutput(Value output);
        Value getOutputForInput(Value input);
        size_t getNumParams();
        Value getParameter(size_t i);
        static StringRef getBaseSymbol() { return "ctrl"; }
    }];

    let builders = [
        OpBuilder<(ins "ValueRange":$controls, "ValueRange":$targets), [{
           build($_builder, $_state, controls.getTypes(), targets.getTypes(), controls, targets);
        }]>,
        OpBuilder<(ins "ValueRange":$controls, "ValueRange":$targets, "UnitaryOpInterface":$bodyUnitary)>,
        OpBuilder<(ins "ValueRange":$controls, "ValueRange":$targets, "llvm::function_ref<llvm::SmallVector<Value>(ValueRange)>":$bodyBuilder)>
    ];

    let hasCanonicalizer = 1;
    let hasVerifier = 1;
}

#endif // QCOOPS
