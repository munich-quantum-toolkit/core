// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --constant-folding | FileCheck %s


// -----
// Tests that constant indices passed to `mqtref.extractQubit` are transformed into static attributes correctly.

module {
  // CHECK-LABEL: @foldExtractQubitIndex
  func.func @foldExtractQubitIndex() {
    // CHECK: %[[Reg_0:.*]] = "mqtref.allocQubitRegister"() <{size_attr = 1 : i64}>
    %r0 = "mqtref.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtref.QubitRegister

    %i = arith.constant 0 : i64
    %q0 = "mqtref.extractQubit"(%r0, %i) : (!mqtref.QubitRegister, i64) -> !mqtref.DynamicQubit
    // CHECK-NOT: arith.constant
    // CHECK: %[[Q0:.*]] = "mqtref.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.DynamicQubit

    // CHECK: mqtref.x() %[[Q0]]
    mqtref.x () %q0 : !mqtref.DynamicQubit
    return
  }
}


// -----
// Tests that nothing is done with `mqtref.extractQubit` if index is already given as an attribute.

module {
  // CHECK-LABEL: @extractQubitIndexDoNothing
  func.func @extractQubitIndexDoNothing() {
    // CHECK: %[[Reg_0:.*]] = "mqtref.allocQubitRegister"() <{size_attr = 1 : i64}>
    %r0 = "mqtref.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtref.QubitRegister

    %q0 = "mqtref.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.DynamicQubit
    // CHECK: %[[Q0:.*]] = "mqtref.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.DynamicQubit

    // CHECK: mqtref.x() %[[Q0]]
    mqtref.x () %q0 : !mqtref.DynamicQubit
    return
  }
}

// -----
// Tests that nothing is done with `mqtref.extractQubit` if index is not a constant.

module {
  // CHECK-LABEL: @noConstantDontFold
  func.func @noConstantDontFold() {
    %r0 = "mqtref.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtref.QubitRegister

    %q0 = "mqtref.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.DynamicQubit

    %m = "mqtref.measure"(%q0) : (!mqtref.DynamicQubit) -> i1
    %i = arith.extui %m : i1 to i64
    // CHECK: %[[Q0:.*]] = "mqtref.extractQubit"(%[[Reg_0:.*]], %[[i:.*]]) : (!mqtref.QubitRegister, i64) -> !mqtref.DynamicQubit
    %q1 = "mqtref.extractQubit"(%r0, %i) : (!mqtref.QubitRegister, i64) -> !mqtref.DynamicQubit

    mqtref.x () %q1 : !mqtref.DynamicQubit
    return
  }
}
