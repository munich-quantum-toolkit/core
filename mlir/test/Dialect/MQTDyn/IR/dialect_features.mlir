// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// -----
// This test checks if the AllocOp is parsed and handled correctly using a static attribute.
module {
    // CHECK-LABEL: func.func @testAllocOpAttribute
    func.func @testAllocOpAttribute() {
        // CHECK: %[[QREQ_0:.*]] = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}>
        // CHECK-NOT: "mqtdyn.allocQubitRegister"

        %qreq_0 = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        return
    }
}

// -----
// This test checks if the AllocOp is parsed and handled correctly using a dynamic operand.
module {
    // CHECK-LABEL: func.func @testAllocOpOperand
    func.func @testAllocOpOperand() {
        // CHECK: %[[SIZE:.*]] = arith.constant 4
        // CHECK: %[[QREQ_0:.*]] = "mqtdyn.allocQubitRegister"(%[[SIZE]]) : (i64) -> !mqtdyn.QubitRegister
        // CHECK-NOT: "mqtdyn.allocQubitRegister"

        %size = arith.constant 4 : i64
        %qreg_0 = "mqtdyn.allocQubitRegister"(%size) : (i64) -> !mqtdyn.QubitRegister
        return
    }
}

// -----
// This test checks if the DeallocOp is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testDeallocOp
    func.func @testDeallocOp() {
        // CHECK: "mqtdyn.deallocQubitRegister"(%[[ANY:.*]])
        // CHECK-NOT: "mqtdyn.deallocQubitRegister"

        %qreg_0 = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        "mqtdyn.deallocQubitRegister"(%qreg_0) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if the ExtractOp is parsed and handled correctly using a static attribute.
module {
    // CHECK-LABEL: func.func @testExtractOpAttribute
    func.func @testExtractOpAttribute() {
        // CHECK: %[[Q_0:.*]] = "mqtdyn.extractQubit"(%[[ANY:.*]]) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        // CHECK-NOT: "mqtdyn.extractQubit"([[ANY:.*]])

        %reg_0 = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister
        %q_0 = "mqtdyn.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        return
    }
}

// -----
// This test checks if the ExtractOp is parsed and handled correctly using a dynamic operand.
module {
    // CHECK-LABEL: func.func @testExtractOpOperand
    func.func @testExtractOpOperand() {
        // CHECK: %[[INDEX:.*]] = arith.constant 0
        // CHECK: %[[Q_0:.*]] = "mqtdyn.extractQubit"(%[[ANY:.*]], %[[INDEX]]) : (!mqtdyn.QubitRegister, i64) -> !mqtdyn.Qubit
        // CHECK-NOT: "mqtdyn.extractQubit"([[ANY:.*]])

        %index = arith.constant 0 : i64
        %qreg_0 = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister
        %q_0 = "mqtdyn.extractQubit"(%qreg_0, %index) : (!mqtdyn.QubitRegister, i64) -> !mqtdyn.Qubit
        return
    }
}

// -----
// This test checks that all resources defined in the MQTDyn dialect are parsed and handled correctly using dynamic operands.
module {
    // CHECK-LABEL: func.func @testAllResourcesUsingOperands
    func.func @testAllResourcesUsingOperands() {
        // CHECK: %[[SIZE:.*]] = arith.constant 1 : i64
        // CHECK: %[[INDEX:.*]] = arith.constant 0 : i64
        // CHECK: %[[QREQ_0:.*]] = "mqtdyn.allocQubitRegister"(%[[SIZE]]) : (i64) -> !mqtdyn.QubitRegister
        // CHECK: %[[Q_0:.*]] = "mqtdyn.extractQubit"(%[[QREQ_0]], %[[INDEX]]) : (!mqtdyn.QubitRegister, i64) -> !mqtdyn.Qubit
        // CHECK: "mqtdyn.deallocQubitRegister"(%[[QREQ_0]])
        // CHECK-NOT: "mqtdyn.allocQubitRegister"
        // CHECK-NOT: "mqtdyn.extractQubit"
        // CHECK-NOT: "mqtdyn.deallocQubitRegister"

        %size = arith.constant 1 : i64
        %index = arith.constant 0 : i64
        %qreg_0 = "mqtdyn.allocQubitRegister"(%size) : (i64) -> !mqtdyn.QubitRegister
        %q_0 = "mqtdyn.extractQubit"(%qreg_0, %index) : (!mqtdyn.QubitRegister, i64) -> !mqtdyn.Qubit
        "mqtdyn.deallocQubitRegister"(%qreg_0) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}
