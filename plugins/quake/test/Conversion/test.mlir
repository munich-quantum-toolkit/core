// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-convert %s -split-input-file | FileCheck %s

// -----
// This is an empty dummy test.

module {
    // CHECK-LABEL: func.func @test()
    func.func @test() {
        return
    }
}
