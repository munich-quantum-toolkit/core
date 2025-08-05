// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-convert %s -split-input-file --mqtdyn-to-quake | FileCheck %s

// -----
// This test checks that the AllocOp is converted correctly.

module {
    // CHECK-LABEL: func.func @testConvertAllocOp()
    func.func @testConvertAllocOp() {
        %r0 = "mqtdyn.allocQubitRegister" () {"size_attr" = 2 : i64} : () -> !mqtdyn.QubitRegister
        return
    }
}
