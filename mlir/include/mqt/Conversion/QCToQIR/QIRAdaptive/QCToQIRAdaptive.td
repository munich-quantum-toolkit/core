// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def QCToQIRAdaptive : Pass<"qc-to-qir-adaptive", "mlir::ModuleOp"> {
  let summary = "Lower the QC dialect to the LLVM dialect compliant with the "
                "QIR Adaptive Profile 2.1";

  let description = [{
    Lower supported QC operations to QIR Adaptive Profile runtime calls and
    LLVM dialect operations. Run `set-qir-attributes-and-metadata` separately
    to attach QIR attributes and module flags.

    Requirements:

    - Input is a valid module in the QC dialect.
    - The entry function must be marked with `mqt.entry_point` and have a single return.
    - Stores to returned CBit registers must share a block with their measurement,
      use an index available at measurement (or a constant), and have no
      intervening effects that can access CBit storage. Stores to provably
      distinct constant indices of the same register are allowed.

    Behavior:

    - Each QC quantum operation is replaced by a call to the corresponding QIR function in the LLVM dialect.
    - Operations from the scf dialect are lowered into the cf dialect before they are lowered to the LLVM dialect.
    - The pass extends the existing block structure of the entry function with an entry block and an epilogue block:
      0. Initialization block: Sets up the execution environment and performs required runtime initialization.
      1. Epilogue block: Records measurement results and returns from the entry function.
      Any blocks in-between have no restrictions regarding their operations as long as they are supported.
    - Quantum releases retain their original control-flow positions.
    - Local qubit-reference buffers support loads, stores, and shape casts without
      allocating qubits.
    - Scalar and register results are dynamically allocated and released after output recording.
    - Measurement results may be used as classical values to drive conditional branches.
    - Non-quantum dialects are lowered via MLIR's built-in conversions.
    - Constant classical tensors and element reads are lowered to constant LLVM arrays and loads, including reads with runtime indices.
      Other classical tensor operations must be lowered before this pass.
  }];

  let dependentDialects = ["mlir::LLVM::LLVMDialect",
                           "mlir::cf::ControlFlowDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect", "mlir::qc::QCDialect",
                           "mlir::arith::ArithDialect"];
}
