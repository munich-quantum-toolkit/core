// Copyright (c) 2025 Chair for Design Automation, TUM
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

def UnitaryInterface : OpInterface<"UnitaryInterface"> {
    let description = [{.
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
            /*desc=*/        "Returns all positively control qubits of the operation.",
            /*returnType=*/  "mlir::OperandRange",
            /*methodName=*/  "getPosCtrlQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                return $_op.getPosCtrlQubits();
            }]>,
        InterfaceMethod<
            /*desc=*/        "Returns all negatively control qubits of the operation.",
            /*returnType=*/  "mlir::OperandRange",
            /*methodName=*/  "getNegCtrlQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                return $_op.getNegCtrlQubits();
            }]>,
        InterfaceMethod<
            /*desc=*/        "Returns all output qubits of the operation incl. all control qubits.",
            /*returnType=*/  "mlir::ResultRange",
            /*methodName=*/  "getOutQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                return $_op.getOutQubits();
            }]>,
        InterfaceMethod<
            /*desc=*/        "Returns all control qubits of the operation incl. positive and negative controls.",
            /*returnType=*/  "std::vector<mlir::Value>",
            /*methodName=*/  "getCtrlQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                std::vector<mlir::Value> controls{};
                const auto& posCtrlQubits = $_op.getPosCtrlQubits();
                controls.insert(controls.end(), posCtrlQubits.begin(), posCtrlQubits.end());
                const auto& negCtrlQubits = $_op.getNegCtrlQubits();
                controls.insert(controls.end(), negCtrlQubits.begin(), negCtrlQubits.end());
                return controls;
            }]>,
        InterfaceMethod<
            /*desc=*/        "Returns all input qubits of the operation incl. all control qubits.",
            /*returnType=*/  "std::vector<mlir::Value>",
            /*methodName=*/  "getAllInQubits",
            /*args=*/        (ins),
            /*methodBody=*/  [{}],
            /*defaultImpl=*/ [{
                std::vector<mlir::Value> operands{};
                const auto& inQubits = $_op.getInQubits();
                operands.insert(operands.end(), inQubits.begin(), inQubits.end());
                const auto& controls = $_op.getCtrlQubits();
                operands.insert(operands.end(), controls.begin(), controls.end());
                return operands;
            }]>
    ];

    let verify = [{
        auto unitaryOp = mlir::cast<ConcreteOp>($_op);
        if (const auto in_qubits = unitaryOp.getInQubits().size() + unitaryOp.getPosCtrlQubits().size() + unitaryOp.getNegCtrlQubits().size();
            in_qubits != unitaryOp.getOutQubits().size()) {
            return $_op->emitError() <<
                "number of input qubits (" << in_qubits << ") " <<
                "and output qubits (" << unitaryOp.getOutQubits().size() << ") must be the same";
        }
        return mlir::success();
    }];
}
