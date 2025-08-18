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

        The `mqtopt` dialect supports dynamic as well as static qubit
        addressing.
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
    let summary = "A value-semantic (dynamic or static) qubit (state).";
}

def QregType : MQTOptType<"QubitRegister", "QubitRegister"> {
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
    GateOp<mnemonic, traits # [AttrSizedOperandSegments, AttrSizedResultSegments, UnitaryInterface]> {
    let arguments = (ins
        OptionalAttr<DenseF64ArrayAttr>:$static_params,
        OptionalAttr<DenseBoolArrayAttr>:$params_mask,
        Variadic<F64>:$params,
        Variadic<QubitType>:$in_qubits,
        Variadic<QubitType>:$pos_ctrl_in_qubits,
        Variadic<QubitType>:$neg_ctrl_in_qubits
    );

    let results = (outs
        Variadic<QubitType>:$out_qubits,
        Variadic<QubitType>:$pos_ctrl_out_qubits,
        Variadic<QubitType>:$neg_ctrl_out_qubits
    );

    let assemblyFormat = [{
        `(` $params ( `static` $static_params^ )? ( `mask` $params_mask^ )? `)`
        attr-dict ( $in_qubits^ )? ( `ctrl` $pos_ctrl_in_qubits^ )? ( `nctrl` $neg_ctrl_in_qubits^ )?
        custom<OptOutputTypes>(type($out_qubits), type($pos_ctrl_out_qubits), type($neg_ctrl_out_qubits))
    }];
}

include "mlir/Dialect/MQTOpt/IR/MQTOptStdOps.td"

def MeasureOp : GateOp<"measure", []> {
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
        QubitType:$in_qubit
    );

    let results = (outs
        QubitType:$out_qubit,
        I1:$out_bit
    );
}

def ResetOp : GateOp<"reset", []> {
    let summary = "A reset operation";

    let description = [{
        This class represents a reset operation. It takes a single qubit as
        input and returns the same qubit after it was reset to the value |0>.

        Example:
        ```mlir
        %0_1 = "mqtopt.reset"(%0) : (!mqtopt.Qubit) -> (!mqtopt.Qubit)
        ```
    }];

    let arguments = (ins
        QubitType:$in_qubit
    );

    let results = (outs
        QubitType:$out_qubit
    );
}

class ResourceOp<string mnemonic, list<Trait> traits = []> : MQTOptOp<mnemonic, traits>;

def AllocQubitOp : ResourceOp<"allocQubit"> {
    let summary = "Allocates a single qubit";

    let description = [{
        This class represents an operation that allocates a single qubit.
        The qubit can be used immediately and must be deallocated at a
        later point using the corresponding operation.

        Example:
        ```mlir
        %0 = "mqtopt.allocQubit"() : () -> !mqtopt.Qubit
        ```
    }];

    let arguments = (ins);

    let results = (outs
        QubitType:$qubit
    );
}

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
        QregType:$qreg
    );
}

def DeallocQubitOp : ResourceOp<"deallocQubit"> {
    let summary = "Deallocates a single qubit";

    let description = [{
        This class represents an operation that deallocates a single qubit.
        The qubit must have previously been created using some allocation operation
        and is consumed by this operation, so it cannot be used further.

        Example:
        ```
        "mqtopt.deallocQubit"(%0) : (!mqtopt.Qubit) -> ()
        ```
    }];

    let arguments = (ins
        QubitType:$qubit
    );
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
        QregType:$qreg
    );
}

def ExtractOp : ResourceOp<"extractQubit", [UniqueIndexDefinition]> {
    let summary = "Extract a specific qubit from a given register";

    let description = [{
        This class represents an operation that extracts a specific qubit from
        a given register. Before, the register must be allocated with the
        allocateQubitRegister operation. Upon invocation of this operation, the
        qubit is removed from the register and can be used in operations. A gap
        is left in the register that can be filled with the insert operation.

        Extract:

        %out_qreg, %out_qubit = "mqtopt.extractQubit"(%0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    }];

    let arguments = (ins
        QregType:$in_qreg,
        Optional<I64>:$index,
        OptionalAttr<ConfinedAttr<I64Attr, [IntNonNegative]>>:$index_attr
    );

    let results = (outs
        QregType:$out_qreg,
        QubitType:$out_qubit
    );
}

def InsertOp : ResourceOp<"insertQubit", [UniqueIndexDefinition]> {

    let description = [{
        This class represents an operation that inserts a qubit back into a qubit
        register. Before, the same underlying qubit must have been extracted
        from the same register at the same index. However, this property is not
        enforced by the dialect. Moreover, the dialect permits the (invalid) insertion
        of static qubits into registers. The dialect avoids these issues by relying on
        the fact that the conversion from `mqtref` to `mqtopt` will never produce these
        invalid operations.
    }];

    let arguments = (ins
        QregType:$in_qreg,
        QubitType:$in_qubit,
        Optional<I64>:$index,
        OptionalAttr<ConfinedAttr<I64Attr, [IntNonNegative]>>:$index_attr
    );

    let results = (outs
        QregType:$out_qreg
    );
}

def QubitOp : ResourceOp<"qubit"> {
    let summary = "Retrieve static qubit";
    let description = [{
        The `mqtopt.qubit` operation produces an SSA value from the given index
        to a static (hardware) qubit.

        Example:
        ```mlir
        // Static (hardware) qubit with address '0'.
        %q = "mqtopt.qubit"() <{index = 0 : i64}> : () -> !mqtopt.Qubit

        // Using custom assembly.
        %q = mqtopt.qubit 0
        ```
    }];
    let arguments = (ins ConfinedAttr<I64Attr, [IntNonNegative]>:$index);
    let results = (outs QubitType:$qubit);
    let assemblyFormat = " attr-dict $index ";
}

#endif // MQTOPT_OPS
