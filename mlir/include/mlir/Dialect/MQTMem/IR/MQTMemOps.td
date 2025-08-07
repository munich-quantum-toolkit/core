// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MQTMEM_OPS
#define MQTMEM_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/DialectBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "mlir/Dialect/Common/IR/CommonTraits.td"
include "mlir/Dialect/MQTMem/IR/MQTMemInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def MQTMemDialect : Dialect {
    // The dialect name used in the MLIR file to prefix operations.
    let name = "mqtmem";

    let summary = "The MQT memory (mqtmem) dialect.";

    let description = [{
        TODO
    }];

    // The C++ namespace that the dialect, and all sub-components, get placed
    // in. Here, `mlir::` must either appear at the front or not at all.
    let cppNamespace = "::mqt::ir::mem";

    // This dialect defines its own types for qubits and qubit registers. To
    // parse and print these, respective hooks must be defined. With this flag
    // set to 1, the default hooks are created and used.
    let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

class MQTMemType<string name, string typeMnemonic, list<Trait> traits = []>
        : TypeDef<MQTMemDialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def DeviceQubitType : MQTMemType<"DeviceQubit", "DeviceQubit"> {
    let summary = "A statically addressed (device) qubit.";
}

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

// This is the base class for all operations in the MQTMem dialect. It is a
// template that takes the operation mnemonic and a list of traits. The MQTMemOp
// class is a subclass of the Op class defined in the MLIR core.
class MQTMemOp<string mnemonic, list<Trait> traits = []> :
        Op<MQTMemDialect, mnemonic, traits>;

class GateOp<string mnemonic, list<Trait> traits = [NoMemoryEffect]> :
    MQTMemOp<mnemonic, traits> {
}

class UnitaryOp<string mnemonic, list<Trait> traits = []> :
    GateOp<mnemonic, traits # [AttrSizedOperandSegments, UnitaryInterface]> {
    let arguments = (ins
        OptionalAttr<DenseF64ArrayAttr>:$static_params,
        OptionalAttr<DenseBoolArrayAttr>:$params_mask,
        Variadic<F64>:$params,
        Optional<DeviceQubitType>:$primary,
        Optional<DeviceQubitType>:$secondary,
        Optional<DeviceQubitType>:$ctrl_primary
    );

    let results = (outs);

    let assemblyFormat = [{
        `(` $params ( `static` $static_params^ )? ( `mask` $params_mask^ )? `)`
        attr-dict ( $primary^ )? ( `,` $secondary^ )? ( `ctrl` $ctrl_primary^ )?
    }];

    let hasVerifier = 1;
    let extraClassDefinition = [{
        ::llvm::LogicalResult mqt::ir::mem::$cppClass::verify() {
            if (getCtrlPrimary() && getSecondary()) {
                return emitOpError() << "expects either a secondary OR a control qubit.";
            }
            return mlir::success();
        }
    }];

}

include "mlir/Dialect/MQTMem/IR/MQTMemStdOps.td"

def MeasureOp : GateOp<"measure", [SameVariadicResultSize, MatchingMeasureInOuts]> {
    let summary = "A measure operation";

    let description = [{
        The `mqtmem.measure` operation represents the measurement operation. It takes a
        variable number of qubits as input and returns the measured bits. The returned
        bit indicates whether the respective qubit collapsed to |0> or |1>.

        Example:
        ```mlir
        // Measuring a single qubit
        %b = "mqtmem.measure"(%q) : (!mqtmem.DeviceQubit) -> (i1)

        // Measuring multiple qubits
        %b0, %b1 = "mqtmem.measure"(%q0, %q1) : (!mqtmem.DeviceQubit, !mqtmem.DeviceQubit) -> (i1, i1)
        ```
    }];

    let arguments = (ins Variadic<DeviceQubitType>:$in_qubits);
    let results = (outs Variadic<I1>:$out_bits);
}

class ResourceOp<string mnemonic, list<Trait> traits = []> : MQTMemOp<mnemonic, traits>;

def QubitOp : ResourceOp<"qubit"> {
    let summary = "Assign static (device) qubit address";
    let description = [{
        The `mqtmem.qubit` operation produces an SSA value from the given index
        to a static (device) qubit.

        Example:
        ```mlir
        // Static (device) qubit with address '0'.
        %q = "mqtmem.qubit"() <{index = 0 : i64}> : () -> !mqtmem.DeviceQubit

        // Using custom assembly.
        %q = mqtmem.qubit 0
        ```
    }];
    let arguments = (ins I64Attr:$index);
    let results = (outs DeviceQubitType:$qubit);
    let assemblyFormat = "attr-dict $index";
}

#endif // MQTMEM_OPS
