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
                "QIR Adaptive Profile";

  let description = [{
    Lower supported QC operations to QIR Adaptive Profile runtime calls and
    LLVM dialect operations. Run `set-qir-attributes-and-metadata` separately
    to attach QIR attributes and module flags.

    Requirements:

    - Input is a valid module in the QC dialect.
    - The entry function must be marked with `mqt.entry_point` and have a single return.
    - Measurements and CBit allocations must be in the entry function.
      Returns may contain direct measurements, direct CBit allocations, and
      boolean scalar values. Other scalar program outputs are not supported.

    Behavior:

    - Outputs retain return order and repetitions. Every record has a unique label.
      Computed boolean scalars use boolean output records without register storage.
    - Returned registers use result arrays when measurement stores can fuse safely
      and preserve returned measurement snapshots. With writable results enabled,
      other stores write boolean values into result slots. Otherwise they use boolean storage,
      including runtime-indexed accesses. Loads of boolean outputs precede output recording.
      A target with complete optional capability metadata also enables writable
      results when it permits `qir.writable-results` without constraints.
    - The entry function returns an `i64` zero execution status after output recording.
    - Each QC quantum operation is replaced by a call to the corresponding QIR function in the LLVM dialect.
    - Operations from the scf dialect are lowered into the cf dialect before they are lowered to the LLVM dialect.
    - The pass extends the existing block structure of the entry function with an entry block and an epilogue block:
      0. Initialization block: Sets up the execution environment and performs required runtime initialization.
      1. Epilogue block: Records measurement results and returns from the entry function.
      Any blocks in-between have no restrictions regarding their operations as long as they are supported.
    - Quantum releases retain their original control-flow positions.
    - Local qubit-reference buffers support loads, stores, and shape casts without
      allocating qubits.
    - Measurement results are dynamically allocated and released after output recording.
    - Measurement results may be used as classical values to drive conditional branches.
    - Non-quantum dialects are lowered via MLIR's built-in conversions.
    - Constant classical tensors and element reads are lowered to constant LLVM arrays and loads, including reads with runtime indices.
      Other classical tensor operations must be lowered before this pass.
    - Statically shaped classical stack memrefs lower to native LLVM arrays, loads, and stores.
      Classical heap allocation is rejected. These operations require the
      Adaptive array capability and the stored element types.
    - Classical array indices must be in bounds. No runtime checks are inserted.
      The metadata pass conservatively requires loop support for residual
      dynamic array indices.
  }];

  let dependentDialects = ["mlir::LLVM::LLVMDialect",
                           "mlir::cf::ControlFlowDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect", "mlir::qc::QCDialect",
                           "mlir::arith::ArithDialect"];
}
