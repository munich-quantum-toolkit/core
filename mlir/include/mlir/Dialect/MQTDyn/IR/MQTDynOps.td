// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MQTDYN_OPS
#define MQTDYN_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/DialectBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "mlir/Dialect/Common/IR/CommonTraits.td"
include "mlir/Dialect/MQTDyn/IR/MQTDynInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def MQTDynDialect : Dialect {
    // The dialect name used in the MLIR file to prefix operations.
    let name = "mqtdyn";

    let summary = "The MQT dynamic qubit allocation (mqtdyn) dialect.";

    let description = [{
        This dialect uses memory semantics to represent and work with qubits.
        Similar to many existing quantum programming languages, operations can
        just be applied to qubits without needing to track state updates.

        This dialect is designed as a 'compatibility' dialect that simplifies
        translations from and to existing languages such as QASM or QIR.
        Its optimization capabilities are more limited compared to the
        `mqtopt` dialect, which is designed for optimization.
    }];

    // The C++ namespace that the dialect, and all sub-components, get placed
    // in. Here, `mlir::` must either appear at the front or not at all.
    let cppNamespace = "::mqt::ir::dyn";

    // This dialect defines its own types for qubits and qubit registers. To
    // parse and print these, respective hooks must be defined. With this flag
    // set to 1, the default hooks are created and used.
    let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

class MQTDynType<string name, string typeMnemonic, list<Trait> traits = []>
        : TypeDef<MQTDynDialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def QubitType : MQTDynType<"Qubit", "Qubit"> {
    let summary = "A qubit reference.";
}

def QregType : MQTDynType<"QubitRegister", "QubitRegister"> {
    let summary = "An array of qubit references.";
}

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

// This is the base class for all operations in the MQTDyn dialect. It is a
// template that takes the operation mnemonic and a list of traits. The MQTDynOp
// class is a subclass of the Op class defined in the MLIR core.
class MQTDynOp<string mnemonic, list<Trait> traits = []> :
        Op<MQTDynDialect, mnemonic, traits>;

class GateOp<string mnemonic, list<Trait> traits = []> :
    MQTDynOp<mnemonic, traits> {
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

include "mlir/Dialect/MQTDyn/IR/MQTDynStdOps.td"

def MeasureOp : GateOp<"measure", [MatchingMeasureInOuts]> {
    let summary = "A measure operation";

    let description = [{
        This class represents a measure operation. It takes a single qubit as
        input and returns a bit. After the measurement, the referenced qubit
        is either in the state |0> or |1> and is indicated by the returned bit.

        Example:
        ```mlir
        %1 = "mqtdyn.measure"(%0) : (!mqtdyn.Qubit) -> (i1)
        ```
    }];

    let arguments = (ins
        Variadic<QubitType>:$in_qubits
    );

    let results = (outs
        Variadic<I1>:$out_bits
    );
}

class ResourceOp<string mnemonic, list<Trait> traits = []> : MQTDynOp<mnemonic, traits>;

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
        %0 = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister

        %c0_i64 = arith.constant 2 : i64
        %1 = "mqtdyn.allocQubitRegister"(%c0_i64) : (i64) -> !mqtdyn.QubitRegister
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
        ```
        "mqtdyn.deallocQubitRegister"(%0) : (!mqtdyn.QubitRegister) -> ()
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

        Extract:

        %out_qubit = "mqtdyn.extractQubit"(%0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> (!mqtdyn.QubitRegister, !mqtdyn.Qubit)

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

#endif // MQTDYN_OPS
