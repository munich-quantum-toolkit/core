// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --constant-folding | FileCheck %s


// -----
// Tests that constant indices passed to `mqtdyn.extractQubit` are transformed into static attributes correctly.

module {
  // CHECK-LABEL: @foldExtractQubitIndex
  func.func @foldExtractQubitIndex() {
    // CHECK: %[[Reg_0:.*]] = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}>
    %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister

    %i = arith.constant 0 : i64
    %q0 = "mqtdyn.extractQubit"(%r0, %i) : (!mqtdyn.QubitRegister, i64) -> !mqtdyn.Qubit
    // CHECK-NOT: arith.constant
    // CHECK: %[[Q0:.*]] = "mqtdyn.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

    // CHECK: mqtdyn.x() %[[Q0]]
    mqtdyn.x () %q0
    return
  }
}


// -----
// Tests that nothing is done with `mqtdyn.extractQubit` if index is already given as an attribute.

module {
  // CHECK-LABEL: @extractQubitIndexDoNothing
  func.func @extractQubitIndexDoNothing() {
    // CHECK: %[[Reg_0:.*]] = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}>
    %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister

    %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
    // CHECK: %[[Q0:.*]] = "mqtdyn.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

    // CHECK: mqtdyn.x() %[[Q0]]
    mqtdyn.x () %q0
    return
  }
}
