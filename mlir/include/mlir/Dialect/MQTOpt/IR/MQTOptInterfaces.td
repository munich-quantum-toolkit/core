// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MQTOPT_INTERFACES
#define MQTOPT_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

def UnitaryInterface : OpInterface<"UnitaryInterface"> {
    let description = [{
        This interface provides a generic way to interact with unitary
        operations in the MQTOpt dialect. Unitary operations are quantum
        operations that are reversible and can be represented by a unitary
        matrix. The interface provides methods to access the operands and
        results of the operation.
    }];

    let cppNamespace = "::mqt::ir::opt";

    let methods = [
        InterfaceMethod<
            /*desc=*/        "Returns all input qubits of the operation excl. control qubits.",
            /*returnType=*/  "mlir::OperandRange",
            /*methodName=*/  "getInQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                return $_op.getInQubits();
            }]>,
        InterfaceMethod<
            /*desc=*/        "Returns all positively-controlling input qubits of the operation.",
            /*returnType=*/  "mlir::OperandRange",
            /*methodName=*/  "getPosCtrlInQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                return $_op.getPosCtrlInQubits();
            }]>,
        InterfaceMethod<
            /*desc=*/        "Returns all negatively-controlling input qubits of the operation.",
            /*returnType=*/  "mlir::OperandRange",
            /*methodName=*/  "getNegCtrlInQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                return $_op.getNegCtrlInQubits();
            }]>,
        InterfaceMethod<
            /*desc=*/        "Returns all output qubits of the operation excl. control qubits.",
            /*returnType=*/  "mlir::ResultRange",
            /*methodName=*/  "getOutQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                return $_op.getOutQubits();
            }]>,
        InterfaceMethod<
            /*desc=*/        "Returns all positively-controlling output qubits of the operation.",
            /*returnType=*/  "mlir::ResultRange",
            /*methodName=*/  "getPosCtrlOutQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                return $_op.getPosCtrlOutQubits();
            }]>,
        InterfaceMethod<
            /*desc=*/        "Returns all negatively-controlling output qubits of the operation.",
            /*returnType=*/  "mlir::ResultRange",
            /*methodName=*/  "getNegCtrlOutQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                return $_op.getNegCtrlOutQubits();
            }]>,
        InterfaceMethod<
            /*desc=*/        "Returns all controlling input qubits of the operation.",
            /*returnType=*/  "std::vector<mlir::Value>",
            /*methodName=*/  "getAllCtrlInQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                const auto& posCtrlQubits = $_op.getPosCtrlInQubits();
                const auto& negCtrlQubits = $_op.getNegCtrlInQubits();
                std::vector<mlir::Value> controls{};
                controls.reserve(posCtrlQubits.size() + negCtrlQubits.size());
                controls.insert(controls.end(), posCtrlQubits.begin(), posCtrlQubits.end());
                controls.insert(controls.end(), negCtrlQubits.begin(), negCtrlQubits.end());
                return controls;
            }]>,
        InterfaceMethod<
            /*desc=*/        "Returns all control output qubits of the operation.",
            /*returnType=*/  "std::vector<mlir::Value>",
            /*methodName=*/  "getAllCtrlOutQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                const auto& posCtrlQubits = $_op.getPosCtrlOutQubits();
                const auto& negCtrlQubits = $_op.getNegCtrlOutQubits();
                std::vector<mlir::Value> controls{};
                controls.reserve(posCtrlQubits.size() + negCtrlQubits.size());
                controls.insert(controls.end(), posCtrlQubits.begin(), posCtrlQubits.end());
                controls.insert(controls.end(), negCtrlQubits.begin(), negCtrlQubits.end());
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
                const auto& controls = $_op.getAllCtrlInQubits();
                std::vector<mlir::Value> operands{};
                operands.reserve(inQubits.size() + controls.size());
                operands.insert(operands.end(), inQubits.begin(), inQubits.end());
                operands.insert(operands.end(), controls.begin(), controls.end());
                return operands;
            }]>,
        InterfaceMethod<
            /*desc=*/        "Returns all output qubits of the operation incl. all controlling qubits.",
            /*returnType=*/  "std::vector<mlir::Value>",
            /*methodName=*/  "getAllOutQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                const auto& outQubits = $_op.getOutQubits();
                const auto& controls = $_op.getAllCtrlOutQubits();
                std::vector<mlir::Value> operands{};
                operands.reserve(outQubits.size() + controls.size());
                operands.insert(operands.end(), outQubits.begin(), outQubits.end());
                operands.insert(operands.end(), controls.begin(), controls.end());
                return operands;
            }]>,
        InterfaceMethod<
            /*desc=*/        "Get params.",
            /*returnType=*/  "mlir::ValueRange",
            /*methodName=*/  "getParams",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                return $_op.getParams();
            }]>
    ];

    let verify = [{
        auto unitaryOp = mlir::cast<ConcreteOp>($_op);
        if (unitaryOp.getInQubits().size() != unitaryOp.getOutQubits().size()) {
            return $_op->emitError() <<
                "number of input qubits (" << unitaryOp.getInQubits().size() << ") "
                << "and output qubits (" << unitaryOp.getOutQubits().size() << ") must be the same";
        } else if (unitaryOp.getPosCtrlInQubits().size() != unitaryOp.getPosCtrlOutQubits().size()) {
            return $_op->emitError() <<
                "number of positively-controlling input qubits (" << unitaryOp.getPosCtrlInQubits().size() << ") "
                << "and positively-controlling output qubits (" << unitaryOp.getPosCtrlOutQubits().size()
                << ") must be the same";
        } else if (unitaryOp.getNegCtrlInQubits().size() != unitaryOp.getNegCtrlOutQubits().size()) {
            return $_op->emitError() <<
                "number of negatively-controlling input qubits (" << unitaryOp.getNegCtrlInQubits().size() << ") "
                << "and negatively-controlling output qubits (" << unitaryOp.getNegCtrlOutQubits().size()
                << ") must be the same";
        }
        return mlir::success();
    }];
}

#endif // MQTOPT_INTERFACES
