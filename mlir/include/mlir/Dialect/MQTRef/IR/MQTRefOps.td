// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MQTREF_OPS
#define MQTREF_OPS

include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "mlir/Dialect/Common/IR/CommonTraits.td"
include "mlir/Dialect/MQTRef/IR/MQTRefInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def MQTRefDialect : Dialect {
    // The dialect name used in the MLIR file to prefix operations.
    let name = "mqtref";

    let summary = "The MQT reference semantics (`mqtref`) dialect.";

    let description = [{
        This dialect uses memory semantics to represent and work with qubits.
        Similar to many existing quantum programming languages, operations can
        just be applied to qubits without needing to track state updates.

        This dialect is designed as a "compatibility" dialect that simplifies
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

def QubitType : MQTRefType<"Qubit", "Qubit", [MemRefElementTypeInterface]> {
    let summary = "qubit reference";
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
        `(` custom<RefParams>($params, $static_params, $params_mask) `)`
        attr-dict
        $in_qubits ( `ctrl` $pos_ctrl_in_qubits^ )? ( `nctrl` $neg_ctrl_in_qubits^ )?
    }];

    int qcType;
    int isInverted;
}

include "mlir/Dialect/MQTRef/IR/MQTRefStdOps.td"

def MeasureOp : GateOp<"measure", []> {
    let summary = "A measure operation";

    let description = [{
        This class represents a measure operation.
        It takes a single qubit as input and returns a bit.
        After the measurement, the referenced qubit is in state "0" or "1".
        The value of the state is indicated by the returned bit.

        Example:
        ```mlir
        %1 = mqtref.measure %0
        ```
    }];

    let arguments = (ins QubitType:$in_qubit);
    let results = (outs I1:$out_bit);
    let assemblyFormat = "$in_qubit attr-dict";
}

def ResetOp : GateOp<"reset", []> {
    let summary = "A reset operation";

    let description = [{
        This class represents a reset operation.
        It takes a single qubit as input and and resets its state to "0".

        Example:
        ```mlir
        mqtref.reset %0
        ```
    }];

    let arguments = (ins QubitType:$in_qubit);
    let results = (outs);
    let assemblyFormat = "$in_qubit attr-dict";
}

class ResourceOp<string mnemonic, list<Trait> traits = []> : MQTRefOp<mnemonic, traits>;

def AllocQubitOp : ResourceOp<"allocQubit"> {
    let summary = "Allocates a single qubit";

    let description = [{
        Allocates a single qubit in the "0" state.
        The qubit can be used in operations after allocation.
        It must be deallocated with the `deallocQubit` operation.

        Example:
        ```mlir
        %q = mqtref.allocQubit
        ```
    }];

    let arguments = (ins);
    let results = (outs QubitType:$qubit);
    let assemblyFormat = "attr-dict";
}

def DeallocQubitOp : ResourceOp<"deallocQubit"> {
    let summary = "Deallocates a single qubit";

    let description = [{
        Deallocates a single qubit that was previously allocated with the
        `allocQubit` operation. After this operation, the qubit is no longer
        valid and cannot be used in further operations.

        Example:
        ```mlir
        mqtref.deallocQubit %q
        ```
    }];

    let arguments = (ins QubitType:$qubit);
    let results = (outs);
    let assemblyFormat = "$qubit attr-dict";
}

def QubitOp : ResourceOp<"qubit"> {
    let summary = "Assign static qubit reference";
    let description = [{
        The `mqtref.qubit` operation produces an SSA value from the given index
        to a static (hardware) qubit.

        Example:
        ```mlir
        %q = mqtref.qubit 0
        ```
    }];

    let arguments = (ins ConfinedAttr<I64Attr, [IntNonNegative]>:$index);
    let results = (outs QubitType:$qubit);
    let assemblyFormat = "$index attr-dict";
}

#endif // MQTREF_OPS
