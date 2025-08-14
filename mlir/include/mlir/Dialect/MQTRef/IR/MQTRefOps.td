// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MQTREF_OPS
#define MQTREF_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/DialectBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "mlir/Dialect/Common/IR/CommonTraits.td"
include "mlir/Dialect/MQTRef/IR/MQTRefInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def MQTRefDialect : Dialect {
    // The dialect name used in the MLIR file to prefix operations.
    let name = "mqtref";

    let summary = "The MQT reference semantics (mqtref) dialect.";

    let description = [{
        This dialect uses memory semantics to represent and work with qubits.
        Similar to many existing quantum programming languages, operations can
        just be applied to qubits without needing to track state updates.

        This dialect is designed as a 'compatibility' dialect that simplifies
        translations from and to existing languages such as QASM or QIR.
        Its optimization capabilities are more limited compared to the
        `mqtopt` dialect, which is designed for optimization.

        The `mqtref` dialect supports dynamic as well as static qubit
        addressing.
    }];

    // The C++ namespace that the dialect, and all sub-components, get placed
    // in. Here, `mlir::` must either appear at the front or not at all.
    let cppNamespace = "::mqt::ir::ref";

    // This dialect defines its own types for qubits and qubit registers. To
    // parse and print these, respective hooks must be defined. With this flag
    // set to 1, the default hooks are created and used.
    let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

class MQTRefType<string name, string typeMnemonic, list<Trait> traits = []>
        : TypeDef<MQTRefDialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def QubitType : MQTRefType<"Qubit", "Qubit"> {
    let summary = "A (dynamic or static) qubit reference.";
}

def QregType : MQTRefType<"QubitRegister", "QubitRegister"> {
    let summary = "An array of qubit references.";
}

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

// This is the base class for all operations in the MQTRef dialect. It is a
// template that takes the operation mnemonic and a list of traits. The MQTRefOp
// class is a subclass of the Op class defined in the MLIR core.
class MQTRefOp<string mnemonic, list<Trait> traits = []> :
        Op<MQTRefDialect, mnemonic, traits>;

class GateOp<string mnemonic, list<Trait> traits = []> :
    MQTRefOp<mnemonic, traits> {
}

class UnitaryOp<string mnemonic, list<Trait> traits = []> :
    GateOp<mnemonic, traits # [AttrSizedOperandSegments, UnitaryInterface]> {
    let arguments = (ins
        OptionalAttr<DenseF64ArrayAttr>:$static_params,
        OptionalAttr<DenseBoolArrayAttr>:$params_mask,
        Variadic<F64>:$params,
        Variadic<QubitType>:$in_qubits,
        Variadic<QubitType>:$pos_ctrl_in_qubits,
        Variadic<QubitType>:$neg_ctrl_in_qubits
    );

    let results = (outs);

    let assemblyFormat = [{
        `(` $params ( `static` $static_params^ )? ( `mask` $params_mask^ )? `)` attr-dict $in_qubits ( `ctrl` $pos_ctrl_in_qubits^ )? ( `nctrl` $neg_ctrl_in_qubits^ )?
    }];
}

include "mlir/Dialect/MQTRef/IR/MQTRefStdOps.td"

def MeasureOp : GateOp<"measure", []> {
    let summary = "A measure operation";

    let description = [{
        This class represents a measure operation. It takes a single qubit as
        input and returns a bit. After the measurement, the referenced qubit
        is either in the state |0> or |1> and is indicated by the returned bit.

        Example:
        ```mlir
        %1 = mqtref.measure %0
        ```
    }];

    let arguments = (ins
        QubitType:$in_qubit
    );

    let results = (outs
        I1:$out_bit
    );

    let assemblyFormat = [{
        attr-dict $in_qubit
    }];
}

def ResetOp : GateOp<"reset", []> {
    let summary = "A reset operation";

    let description = [{
        This class represents a reset operation. It takes a single qubit as
        input and and resets its value |0>.

        Example:
        ```mlir
        "mqtref.reset"(%0) : (!mqtopt.Qubit) -> ()
        ```
    }];

    let arguments = (ins
        QubitType:$in_qubit
    );

    let results = (outs);
}

class ResourceOp<string mnemonic, list<Trait> traits = []> : MQTRefOp<mnemonic, traits>;

def AllocOp : ResourceOp<"allocQubitRegister", [UniqueSizeDefinition]> {
    let summary = "Allocates a qubit register";

    let description = [{
        This class represents an operation that allocates a qubit register. The
        size can either be specified during runtime as an operand or as an
        attribute when the size is known at compile time. Before the qubits in
        the register can be used, they need to be extracted from the register
        with the extract operation.

        Example:
        ```mlir
        %0 = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtref.QubitRegister

        %c0_i64 = arith.constant 2 : i64
        %1 = "mqtref.allocQubitRegister"(%c0_i64) : (i64) -> !mqtref.QubitRegister
        ```
    }];

    let arguments = (ins
        Optional<I64>:$size,
        OptionalAttr<ConfinedAttr<I64Attr, [IntNonNegative]>>:$size_attr
    );

    let results = (outs
        QregType:$qreg
    );
}

def DeallocOp : ResourceOp<"deallocQubitRegister"> {
    let summary = "Deallocates a qubit register";

    let description = [{
        This class represents an operation that deallocates a qubit register.
        The qubit register is no longer valid after this operation.

        Example:
        ```mlir
        "mqtref.deallocQubitRegister"(%0) : (!mqtref.QubitRegister) -> ()
        ```
    }];

    let arguments = (ins
        QregType:$qreg
    );
}

def ExtractOp : ResourceOp<"extractQubit", [UniqueIndexDefinition]> {
    let summary = "Extract a specific qubit reference from a given register";

    let description = [{
        This class represents an operation that extracts a specific qubit
        reference from a given register. Before, the register must be
        allocated with the allocateQubitRegister operation. Upon invocation
        of this operation, the returned qubit reference can be used in
        operations.

        Example:
        ```mlir
        %out_qubit = "mqtref.extractQubit"(%0) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        ```
    }];

    let arguments = (ins
        QregType:$in_qreg,
        Optional<I64>:$index,
        OptionalAttr<ConfinedAttr<I64Attr, [IntNonNegative]>>:$index_attr
    );

    let results = (outs
        QubitType:$out_qubit
    );
}

def QubitOp : ResourceOp<"qubit"> {
    let summary = "Assign static qubit reference";
    let description = [{
        The `mqtref.qubit` operation produces an SSA value from the given index
        to a static (hardware) qubit.

        Example:
        ```mlir
        // Static (hardware) qubit with address '0'.
        %q = "mqtref.qubit"() <{index = 0 : i64}> : () -> !mqtref.Qubit

        // Using custom assembly.
        %q = mqtref.qubit 0
        ```
    }];
    let arguments = (ins ConfinedAttr<I64Attr, [IntNonNegative]>:$index);
    let results = (outs QubitType:$qubit);
    let assemblyFormat = " attr-dict $index ";
}

#endif // MQTREF_OPS
