// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MQTMEM_INTERFACES
#define MQTMEM_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

def UnitaryInterface : OpInterface<"UnitaryInterface"> {
    let description = [{
        This interface provides a generic way to interact with unitary
        operations in the MQTMem dialect. Unitary operations are quantum
        operations that are reversible and can be represented by a unitary
        matrix. The interface provides methods to access the operands and
        results of the operation.
    }];

    let cppNamespace = "::mqt::ir::mem";

    let methods = [
        InterfaceMethod<
            /*desc=*/        "Returns all qubits of the operation excl. controlling qubits.",
            /*returnType=*/  "std::vector<::mlir::Value>",
            /*methodName=*/  "getInQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                std::vector<::mlir::Value> in_qubits;
                if (auto p = $_op.getPrimary()) { in_qubits.push_back(p); }
                if (auto s = $_op.getSecondary()) { in_qubits.push_back(s); }
                return in_qubits;
            }]>,
        InterfaceMethod<
            /*desc=*/        "Returns all positively-controlling qubits of the operation.",
            /*returnType=*/  "std::vector<::mlir::Value>",
            /*methodName=*/  "getPosCtrlInQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                return { $_op.getCtrlPrimary() };
            }]>,
        InterfaceMethod<
            /*desc=*/        "Returns all negatively-controlling qubits of the operation.",
            /*returnType=*/  "std::vector<::mlir::Value>",
            /*methodName=*/  "getNegCtrlInQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                return {}; // These must already be decomposed in the mem dialect, hence empty.
            }]>,
        InterfaceMethod<
            /*desc=*/        "Returns all controlling qubits of the operation.",
            /*returnType=*/  "std::vector<::mlir::Value>",
            /*methodName=*/  "getCtrlInQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                return $_op.getPosCtrlInQubits();;
            }]>,
        InterfaceMethod<
            /*desc=*/        "Returns all input qubits of the operation incl. all controlling qubits.",
            /*returnType=*/  "std::vector<::mlir::Value>",
            /*methodName=*/  "getAllInQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                const auto& inQubits = $_op.getInQubits();
                const auto& controls = $_op.getCtrlInQubits();
                std::vector<::mlir::Value> operands{};
                operands.insert(operands.end(), inQubits.begin(), inQubits.end());
                operands.insert(operands.end(), controls.begin(), controls.end());
                return operands;
            }]>
    ];
}

#endif // MQTMEM_INTERFACES
