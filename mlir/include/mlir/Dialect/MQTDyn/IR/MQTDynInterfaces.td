// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MQTDYN_INTERFACES
#define MQTDYN_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

def UnitaryInterface : OpInterface<"UnitaryInterface"> {
    let description = [{
        This interface provides a generic way to interact with unitary
        operations in the MQTDyn dialect. Unitary operations are quantum
        operations that are reversible and can be represented by a unitary
        matrix. The interface provides methods to access the operands and
        results of the operation.
    }];

    let cppNamespace = "::mqt::ir::dyn";

    let methods = [
        InterfaceMethod<
            /*desc=*/        "Returns all input qubits of the operation excl. controlling qubits.",
            /*returnType=*/  "mlir::OperandRange",
            /*methodName=*/  "getInQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                return $_op.getInQubits();
            }]>,
        InterfaceMethod<
            /*desc=*/        "Returns all positively-controllingling qubits of the operation.",
            /*returnType=*/  "mlir::OperandRange",
            /*methodName=*/  "getPosCtrlQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                return $_op.getPosCtrlQubits();
            }]>,
        InterfaceMethod<
            /*desc=*/        "Returns all negatively-controllingling qubits of the operation.",
            /*returnType=*/  "mlir::OperandRange",
            /*methodName=*/  "getNegCtrlQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                return $_op.getNegCtrlQubits();
            }]>,
        InterfaceMethod<
            /*desc=*/        "Returns all controlling qubits of the operation incl. positive and negative controls.",
            /*returnType=*/  "std::vector<mlir::Value>",
            /*methodName=*/  "getCtrlQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                const auto& posCtrlQubits = $_op.getPosCtrlQubits();
                const auto& negCtrlQubits = $_op.getNegCtrlQubits();
                std::vector<mlir::Value> controls{};
                controls.reserve(posCtrlQubits.size() + negCtrlQubits.size());
                controls.insert(controls.end(), posCtrlQubits.begin(), posCtrlQubits.end());
                controls.insert(controls.end(), negCtrlQubits.begin(), negCtrlQubits.end());
                return controls;
            }]>,
        InterfaceMethod<
            /*desc=*/        "Returns all input qubits of the operation incl. all controlling qubits.",
            /*returnType=*/  "std::vector<mlir::Value>",
            /*methodName=*/  "getAllInQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                const auto& inQubits = $_op.getInQubits();
                const auto& controls = $_op.getCtrlQubits();
                std::vector<mlir::Value> operands{};
                operands.reserve(inQubits.size() + controls.size());
                operands.insert(operands.end(), inQubits.begin(), inQubits.end());
                operands.insert(operands.end(), controls.begin(), controls.end());
                return operands;
            }]>
    ];
}

#endif // MQTDYN_INTERFACES
