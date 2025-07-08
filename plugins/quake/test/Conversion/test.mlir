// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s --quake-to-mqtdyn | FileCheck %s

module {
  // CHECK-LABEL: func @testAlloca()
  func.func @testAlloca() {
    // CHECK: %[[ANY:.*]] = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}>
    %qubit = quake.alloca !quake.ref

    return
  }
}
