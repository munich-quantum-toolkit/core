// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MQTOPT_OPS
#define MQTOPT_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/DialectBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "mlir/Dialect/Common/IR/CommonTraits.td"
include "mlir/Dialect/MQTOpt/IR/MQTOptInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def MQTOptDialect : Dialect {
    // The dialect name used in the MLIR file to prefix operations.
    let name = "mqtopt";

    let summary = "The MQT optimization (mqtopt) dialect.";

    let description = [{
        This dialect is using value semantics for qubits. This means that
        variables of type qubit and the elements of qubit registers do not
        identify a physical, they rather refer to the state of a qubit. A qubit
        state can be assigned only once and can also be used only once.
        Otherwise, the no-cloning theorem would be violated.

        The first requirement is enforced by the SSA form of MLIR. The second
        requirement is not explicitly enforced. However, when converting the
        input dialect mqt that is using reference semantics to the mqtopt dialect,
        qubit state values will only be used once by construction. This is also
        the reason why the mqtopt dialect is not meant to be used by hand, but
        rather as an intermediate representation for optimization passes.

        Due to the value semantics, the mqtopt dialect allows dataflow analysis
        and optimizations. Those are very common in classical compiler
        optimizations. Hence, the existing classical compiler optimizations can
        easier be reused for quantum programs.

        For more information, see the paper "QIRO:A Static Single Assignment
        based Quantum Program Representation for Optimization"
        (https://doi.org/10.1145/3491247)
    }];

    // The C++ namespace that the dialect, and all sub-components, get placed
    // in. Here, `mlir::` must either appear at the front or not at all.
    let cppNamespace = "::mqt::ir::opt";

    // This dialect defines its own types for qubits and qubit registers. To
    // parse and print these, respective hooks must be defined. With this flag
    // set to 1, the default hooks are created and used.
    let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

class MQTOptType<string name, string typeMnemonic, list<Trait> traits = []>
        : TypeDef<MQTOptDialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def QubitType : MQTOptType<"Qubit", "Qubit"> {
    let summary = "A value-semantic qubit (state).";
}

def QuregType : MQTOptType<"QubitRegister", "QubitRegister"> {
    let summary = "An array of value-semantic qubits (i.e. quantum register).";
}

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

// This is the base class for all operations in the MQTOpt dialect. It is a
// template that takes the operation mnemonic and a list of traits. The MQTOptOp
// class is a subclass of the Op class defined in the MLIR core.
class MQTOptOp<string mnemonic, list<Trait> traits = []> :
        Op<MQTOptDialect, mnemonic, traits>;

class GateOp<string mnemonic, list<Trait> traits = [NoMemoryEffect]> :
    MQTOptOp<mnemonic, traits> {
}

class UnitaryOp<string mnemonic, list<Trait> traits = []> :
    GateOp<mnemonic, traits # [AttrSizedOperandSegments, UnitaryInterface]> {
    let arguments = (ins
        OptionalAttr<DenseF64ArrayAttr>:$static_params,
        OptionalAttr<DenseBoolArrayAttr>:$params_mask,
        Variadic<F64>:$params,
        Variadic<QubitType>:$in_qubits,
        Variadic<QubitType>:$pos_ctrl_qubits,
        Variadic<QubitType>:$neg_ctrl_qubits
    );

    let results = (outs
        Variadic<QubitType>:$out_qubits
    );

    let assemblyFormat = [{
        `(` $params ( `static` $static_params^ )? ( `mask` $params_mask^ )? `)` attr-dict $in_qubits ( `ctrl` $pos_ctrl_qubits^ )? ( `nctrl` $neg_ctrl_qubits^ )?
        `:` type($out_qubits)
    }];
}

def GPhaseOp : UnitaryOp<"GPhase", [NoTarget, OneParameter]> {
    let summary = "GPhase operation";

    let description = [{
        This class represents a general phase gate. It accepts a parameter indicating
        the degree of the rotation angle. The global phase gate does not have
        any input and output qubits. It might still be controlled by arbitrarily
        many qubits.

        An analogous example can be seen in the description of XOp.
    }];

    // This operation has a verifier that checks if the gate has no input or output
    // qubits. It is implemented in the corresponding .cpp file.
    let hasVerifier = 1;
}

def IOp : UnitaryOp<"I", [OneTarget, NoParameter]> {
    let summary = "I operation";

    let description = [{
        This class represents an identity gate. It takes a qubit and a variadic
        list of positive/negative controls as an input.

        An analogous example can be seen in the description of XOp.
    }];
}

def BarrierOp : UnitaryOp<"barrier", [NoParameter]> {
    let summary = "Barrier operation";

    let description = [{
        This class represents the barrier operation. It has a variadic number
        of input qubits and no control qubits.

        An analogous example can be seen in the description of XOp.
    }];

    // This operation has a verifier that checks if the gate does not have any
    // control qubits. It is implemented in the corresponding .cpp file.
    let hasVerifier = 1;
}

def HOp : UnitaryOp<"H", [OneTarget, NoParameter]> {
    let summary = "H operation";

    let description = [{
        This class represents a Hadamard gate. It takes a qubit and a variadic
        list of positive/negative controls as an input.

        An analogous example can be seen in the description of XOp.
    }];
}

def XOp : UnitaryOp<"x", [OneTarget, NoParameter]> {
    let summary = "X operation";

    let description = [{
        This class represents a Pauli-X gate. It takes a qubit and a variadic
        list of positive/negative controls as an input.

        Example:
        ```mlir
        %1 = mqtopt.x() %0 : (!mqtopt.Qubit) -> (!mqtopt.Qubit)
        %3, %4 = mqtopt.x() %1 ctrl(%2) : (!mqtopt.Qubit, !mqtopt.Qubit) -> (!mqtopt.Qubit, !mqtopt.Qubit)
        %7, %8, %9, %10 = mqtopt.x() %3 ctrl(%4) nctrl(%5, %6) : (!mqtopt.Qubit, !mqtopt.Qubit, !mqtopt.Qubit, !mqtopt.Qubit) -> (!mqtopt.Qubit, !mqtopt.Qubit, !mqtopt.Qubit, !mqtopt.Qubit)
        ```
    }];
}

def YOp : UnitaryOp<"y", [OneTarget, NoParameter]> {
    let summary = "Y operation";

    let description = [{
        This class represents a Pauli-Y gate. It takes a qubit and a variadic
        list of positive/negative controls as an input.

        An analogous example can be seen in the description of XOp.
    }];
}

def ZOp : UnitaryOp<"z", [OneTarget, NoParameter]> {
    let summary = "Z operation";

    let description = [{
        This class represents a Pauli-Z gate. It takes a qubit and a variadic
        list of positive/negative controls as an input.

        An analogous example can be seen in the description of XOp.
    }];
}

def SOp : UnitaryOp<"s", [OneTarget, NoParameter]> {
    let summary = "S operation";

    let description = [{
        This class represents an S gate. It takes a qubit as an input and a
        variadic list of positive/negative controls.

        An analogous example can be seen in the description of XOp.
    }];
}

def SdgOp : UnitaryOp<"sdg", [OneTarget, NoParameter]> {
    let summary = "Sdg operation";

    let description = [{
        This class represents an inverse S gate. It takes a qubit and a variadic
        list of positive/negative controls as an input.

        An analogous example can be seen in the description of XOp.
    }];
}

def TOp : UnitaryOp<"t", [OneTarget, NoParameter]> {
    let summary = "T operation";

    let description = [{
        This class represents a T gate. It takes a qubit and a variadic list of
        positive/negative controls as an input.

        An analogous example can be seen in the description of XOp.
    }];
}

def TdgOp : UnitaryOp<"tdg", [OneTarget, NoParameter]> {
    let summary = "Tdg operation";

    let description = [{
        This class represents an inverse T gate. It takes a qubit and a variadic
        list of positive/negative controls as an input.

        An analogous example can be seen in the description of XOp.
    }];
}

def VOp : UnitaryOp<"v", [OneTarget, NoParameter]> {
    let summary = "V operation";

    let description = [{
        This class represents a V gate. It takes a qubit and a variadic
        list of positive/negative controls as an input.

        An analogous example can be seen in the description of XOp.
    }];
}

def VdgOp : UnitaryOp<"vdg", [OneTarget, NoParameter]> {
    let summary = "Vdg operation";

    let description = [{
        This class represents an inverse V gate. It takes a qubit and a variadic
        list of positive/negative controls as an input.

        An analogous example can be seen in the description of XOp.
    }];
}

def UOp : UnitaryOp<"u", [OneTarget, ThreeParameters]> {
    let summary = "U operation";

    let description = [{
        This class represents a U gate. It takes a qubit and a variadic
        list of positive/negative controls as an input. Additionally, it accepts
        three parameters indicating the degree of the rotation angles.

        An analogous example can be seen in the description of XOp.
    }];
}

def U2Op : UnitaryOp<"u2", [OneTarget, TwoParameters]> {
    let summary = "U2 operation";

    let description = [{
        This class represents a U2 gate. It takes a qubit and a variadic
        list of positive/negative controls as an input. Additionally, it accepts
        two parameters indicating the degree of the rotation angles.

        An analogous example can be seen in the description of XOp.
    }];
}

def POp : UnitaryOp<"p", [OneTarget, OneParameter]> {
    let summary = "P operation";

    let description = [{
        This class represents a phase gate. It takes a qubit and a variadic
        list of positive/negative controls as an input. Additionally, it accepts
        a parameter indicating the degree of the rotation angle.

        An analogous example can be seen in the description of XOp.
    }];
}

def SXOp : UnitaryOp<"sx", [OneTarget, NoParameter]> {
    let summary = "SX operation";

    let description = [{
        This class represents an SX gate. It takes a qubit and a variadic
        list of positive/negative controls as an input.

        An analogous example can be seen in the description of XOp.
    }];
}

def SXdgOp : UnitaryOp<"sxdg", [OneTarget, NoParameter]> {
    let summary = "SXdg operation";

    let description = [{
        This class represents an inverse SX gate. It takes a qubit and a
        variadic list of positive/negative controls as an input.

        An analogous example can be seen in the description of XOp.
    }];
}

def RXOp : UnitaryOp<"rx", [OneTarget, OneParameter]> {
    let summary = "RX operation";

    let description = [{
        This class represents an RX gate. It takes a qubit and a variadic
        list of positive/negative controls as input. Additionally, it accepts
        a parameter indicating the degree of the rotation angle.

        An analogous example can be seen in the description of XOp.
    }];
}

def RYOp : UnitaryOp<"ry", [OneTarget, OneParameter]> {
    let summary = "RY operation";

    let description = [{
        This class represents an RY gate. It takes a qubit and a variadic
        list of positive/negative controls as input. Additionally, it accepts
        a parameter indicating the degree of the rotation angle.

        An analogous example can be seen in the description of XOp.
    }];
}

def RZOp : UnitaryOp<"rz", [OneTarget, OneParameter]> {
    let summary = "RZ operation";

    let description = [{
        This class represents an RZ gate. It takes a qubit and a variadic
        list of positive/negative controls as input. Additionally, it accepts
        a parameter indicating the degree of the rotation angle.

        An analogous example can be seen in the description of XOp.
    }];
}

def SWAPOp : UnitaryOp<"swap", [TwoTarget, NoParameter]> {
    let summary = "SWAP operation";

    let description = [{
        This class represents a SWAP gate. It takes two qubits and a variadic
        list of positive/negative controls as input.

        An analogous example can be seen in the description of XOp.
    }];
}

def iSWAPOp : UnitaryOp<"iswap", [TwoTarget, NoParameter]> {
    let summary = "iSWAP operation";

    let description = [{
        This class represents an iSWAP gate. It takes two qubits and a variadic
        list of positive/negative controls as input.

        An analogous example can be seen in the description of XOp.
    }];
}

def iSWAPdgOp : UnitaryOp<"iswapdg", [TwoTarget, NoParameter]> {
    let summary = "iSWAPdg operation";

    let description = [{
        This class represents an inverse iSWAP gate. It takes two qubits and a
        variadic list of positive/negative controls as input.

        An analogous example can be seen in the description of XOp.
    }];
}

def PeresOp : UnitaryOp<"peres", [TwoTarget, NoParameter]> {
    let summary = "Peres operation";

    let description = [{
        This class represents a Peres gate. It takes two qubits and a variadic
        list of positive/negative controls as input.

        An analogous example can be seen in the description of XOp.
    }];
}

def PeresdgOp : UnitaryOp<"peresdg", [TwoTarget, NoParameter]> {
    let summary = "Peresdg operation";

    let description = [{
        This class represents an inverse Peres gate. It takes two qubits and a
        variadic list of positive/negative controls as input.

        An analogous example can be seen in the description of XOp.
    }];
}

def DCXOp : UnitaryOp<"dcx", [TwoTarget, NoParameter]> {
    let summary = "DCX operation";

    let description = [{
        This class represents a DCX gate. It takes two qubits and a variadic
        list of positive/negative controls as input.

        An analogous example can be seen in the description of XOp.
    }];
}

def ECROp : UnitaryOp<"ecr", [TwoTarget, NoParameter]> {
    let summary = "DCX operation";

    let description = [{
        This class represents an ECR gate. It takes two qubits and a variadic
        list of positive/negative controls as input.

        An analogous example can be seen in the description of XOp.
    }];
}

def RXXOp : UnitaryOp<"rxx", [TwoTarget, OneParameter]> {
    let summary = "RXX operation";

    let description = [{
        This class represents an RXX gate. It takes two qubits and a variadic
        list of positive/negative controls as input. Additionally, it accepts a
        parameter indicating the degree of the rotation angle.

        An analogous example can be seen in the description of XOp.
    }];
}

def RYYOp : UnitaryOp<"ryy", [TwoTarget, OneParameter]> {
    let summary = "RYY operation";

    let description = [{
        This class represents an RYY gate. It takes two qubits and a variadic
        list of positive/negative controls as input. Additionally, it accepts a
        parameter indicating the degree of the rotation angle.

        An analogous example can be seen in the description of XOp.
    }];
}

def RZZOp : UnitaryOp<"rzz", [TwoTarget, OneParameter]> {
    let summary = "RZZ operation";

    let description = [{
        This class represents an RZZ gate. It takes two qubits and a variadic
        list of positive/negative controls as input. Additionally, it accepts a
        parameter indicating the degree of the rotation angle.

        An analogous example can be seen in the description of XOp.
    }];
}

def RZXOp : UnitaryOp<"rzx", [TwoTarget, OneParameter]> {
    let summary = "RZX operation";

    let description = [{
        This class represents an RZX gate. It takes two qubits and a variadic
        list of positive/negative controls as input. Additionally, it accepts a
        parameter indicating the degree of the rotation angle.

        An analogous example can be seen in the description of XOp.
    }];
}

def XXminusYY : UnitaryOp<"xxminusyy", [TwoTarget, TwoParameters]> {
    let summary = "XXminusYY operation";

    let description = [{
        This class represents an XXminusYY gate. It takes two qubits and a
        variadic list of positive/negative controls as input. Additionally, it
        accepts two parameters indicating the degree of the rotation angles.

        An analogous example can be seen in the description of XOp.
    }];
}

def XXplusYY : UnitaryOp<"xxplusyy", [TwoTarget, TwoParameters]> {
    let summary = "XXplusYY operation";

    let description = [{
        This class represents an XXplusYY gate. It takes two qubits and a
        variadic list of positive/negative controls as input. Additionally, it
        accepts two parameters indicating the degree of the rotation angles.

        An analogous example can be seen in the description of XOp.
    }];
}

def MeasureOp : GateOp<"measure", [SameVariadicResultSize]> {
    let summary = "A measure operation";

    let description = [{
        This class represents a measure operation. It takes a single qubit as
        input and returns a qubit and a bit. After the measurement, the returned
        qubit is either in the state |0> or |1> and is indicated by the returned
        bit.

        Example:
        ```mlir
        %0_1, %1 = "mqtopt.measure"(%0) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        ```
    }];

    let arguments = (ins
        Variadic<QubitType>:$in_qubits
    );

    let results = (outs
        Variadic<QubitType>:$out_qubits,
        Variadic<I1>:$out_bits
    );

    // This operation has a verifier that checks if the number of input qubits
    // is equal to the number of output qubits. It is implemented in the
    // corresponding .cpp file.
    let hasVerifier = 1;
}

class ResourceOp<string mnemonic, list<Trait> traits = []> : MQTOptOp<mnemonic, traits>;

def AllocOp : ResourceOp<"allocQubitRegister"> {
    let summary = "Allocates a qubit register";

    let description = [{
        This class represents an operation that allocates a qubit register. The
        size can either be specified during runtime as an operand or as an
        attribute when the size is known at compile time. Before the qubits in
        the register can be used, they need to be extracted from the register
        with the extract operation.

        Example:
        ```mlir
        %0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

        %c0_i64 = arith.constant 2 : i64
        %1 = "mqtopt.allocQubitRegister"(%c0_i64) : (i64) -> !mqtopt.QubitRegister
        ```
    }];

    let arguments = (ins
        Optional<I64>:$size,
        OptionalAttr<ConfinedAttr<I64Attr, [IntNonNegative]>>:$size_attr
    );

    let results = (outs
        QuregType:$qureg
    );

    // This operation has a verifier that checks if the size is given either as
    // operand or attribute and not both. It is implemented in the corresponding
    // .cpp file.
    let hasVerifier = 1;
}

def DeallocOp : ResourceOp<"deallocQubitRegister"> {
    let summary = "Deallocates a qubit register";

    let description = [{
        This class represents an operation that deallocates a qubit register.
        The qubit register is no longer valid after this operation. To be valid
        all extracted qubits had to be inserted back into the register. However,
        this property is not enforced by the dialect.

        Example:
        ```
        "mqtopt.deallocQubitRegister"(%0) : (!mqtopt.QubitRegister) -> ()
        ```
    }];

    let arguments = (ins
        QuregType:$qureg
    );
}

def ExtractOp : ResourceOp<"extractQubit"> {
    let summary = "Extract a specific qubit from a given register";

    let description = [{
        This class represents an operation that extracts a specific qubit from
        a given register. Before, the register must be allocated with the
        allocateQubitRegister operation. Upon invocation of this operation, the
        qubit is removed from the register and can be used in operations. A gap
        is left in the register that can be filled with the insert operation.

        Extract:

        %out_qureg, %out_qubit = "mqtopt.extractQubit"(%0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    }];

    let arguments = (ins
        QuregType:$in_qureg,
        Optional<I64>:$index,
        OptionalAttr<ConfinedAttr<I64Attr, [IntNonNegative]>>:$index_attr
    );

    let results = (outs
        QuregType:$out_qureg,
        QubitType:$out_qubit
    );

    // This operation has a verifier that checks if the index is given either as
    // operand or attribute and not both. It is implemented in the corresponding
    // .cpp file.
    let hasVerifier = 1;
}

def InsertOp : ResourceOp<"insertQubit"> {

    let description = [{
        This class represents an operation that inserts a qubit back into a qubit
        register. Before, the same underlying qubit must have been extracted
        from the same register at the same index. However, this property is not
        enforced by the dialect. It is naturally satisfied when a program in the
        mqt input dialect is converted to the mqto dialect.
    }];

    let arguments = (ins
        QuregType:$in_qureg,
        QubitType:$in_qubit,
        Optional<I64>:$index,
        OptionalAttr<ConfinedAttr<I64Attr, [IntNonNegative]>>:$index_attr
    );

    let results = (outs
        QuregType:$out_qureg
    );

    // This operation has a verifier that checks if the index is given either as
    // operand or attribute and not both. It is implemented in the corresponding
    // .cpp file.
    let hasVerifier = 1;
}

#endif // MQTOPT_OPS
