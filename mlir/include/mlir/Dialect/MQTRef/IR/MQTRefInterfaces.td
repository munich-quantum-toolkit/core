// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MQTREF_INTERFACES
#define MQTREF_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

def UnitaryInterface : OpInterface<"UnitaryInterface"> {
    let description = [{
        This interface provides a generic way to interact with unitary
        operations in the MQTRef dialect. Unitary operations are quantum
        operations that are reversible and can be represented by a unitary
        matrix. The interface provides methods to access the operands and
        results of the operation.
    }];

    let cppNamespace = "::mqt::ir::ref";

    let methods = [
        InterfaceMethod<
            /*desc=*/        "Returns all qubits of the operation excl. controlling qubits.",
            /*returnType=*/  "mlir::OperandRange",
            /*methodName=*/  "getInQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                return $_op.getInQubits();
            }]>,
        InterfaceMethod<
            /*desc=*/        "Returns all positively-controlling qubits of the operation.",
            /*returnType=*/  "mlir::OperandRange",
            /*methodName=*/  "getPosCtrlInQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                return $_op.getPosCtrlInQubits();
            }]>,
        InterfaceMethod<
            /*desc=*/        "Returns all negatively-controlling qubits of the operation.",
            /*returnType=*/  "mlir::OperandRange",
            /*methodName=*/  "getNegCtrlInQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                return $_op.getNegCtrlInQubits();
            }]>,
        InterfaceMethod<
            /*desc=*/        "Returns all controlling qubits of the operation.",
            /*returnType=*/  "std::vector<mlir::Value>",
            /*methodName=*/  "getCtrlInQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                const auto& posCtrlInQubits = $_op.getPosCtrlInQubits();
                const auto& negCtrlInQubits = $_op.getNegCtrlInQubits();
                std::vector<mlir::Value> controls{};
                controls.reserve(posCtrlInQubits.size() + negCtrlInQubits.size());
                controls.insert(controls.end(), posCtrlInQubits.begin(), posCtrlInQubits.end());
                controls.insert(controls.end(), negCtrlInQubits.begin(), negCtrlInQubits.end());
                return controls;
            }]>,
        InterfaceMethod<
            /*desc=*/        "Returns true if the operation has any control qubits, otherwise false.",
            /*returnType=*/  "bool",
            /*methodName=*/  "isControlled",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                return !$_op.getPosCtrlInQubits().empty() || !$_op.getNegCtrlInQubits().empty();
            }]>,
        InterfaceMethod<
            /*desc=*/        "Returns all input qubits of the operation incl. all controlling qubits.",
            /*returnType=*/  "std::vector<mlir::Value>",
            /*methodName=*/  "getAllInQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                const auto& inQubits = $_op.getInQubits();
                const auto& controls = $_op.getCtrlInQubits();
                std::vector<mlir::Value> operands{};
                operands.reserve(inQubits.size() + controls.size());
                operands.insert(operands.end(), inQubits.begin(), inQubits.end());
                operands.insert(operands.end(), controls.begin(), controls.end());
                return operands;
            }]>,
        InterfaceMethod<
            /*desc=*/        "Returns the name of the gate.",
            /*returnType=*/  "llvm::StringRef",
            /*methodName=*/  "getIdentifier",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
               return  $_op->getName().getStringRef().split('.').second;
            }]>
    ];
}

#endif // MQTREF_INTERFACES
