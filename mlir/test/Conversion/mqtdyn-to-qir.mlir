// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --mqtdyn-to-qir | FileCheck %s

// -----
// TODO
module {
    // CHECK-LABEL: llvm.func @testConvertAllocRegister()
    llvm.func @testConvertAllocRegister() attributes {passthrough = ["entry_point"]}  {
        %0 = llvm.mlir.zero : !llvm.ptr
        llvm.br ^bb1
      ^bb1:

        llvm.br ^bb2
      ^bb2:
        llvm.return
    }
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate_array(i64) -> !llvm.ptr
}

// -----
// TODO
module {
    // CHECK-LABEL: llvm.func @testConvertAllocRegister()
    llvm.func @testConvertAllocRegister() attributes {passthrough = ["entry_point"]}  {
        %0 = llvm.mlir.zero : !llvm.ptr
        llvm.br ^bb1
      ^bb1:

        llvm.br ^bb2
      ^bb2:
        llvm.return
    }
    llvm.func @__quantum__rt__initialize(!llvm.ptr)
    llvm.func @__quantum__rt__qubit_allocate_array(i64) -> !llvm.ptr
}
