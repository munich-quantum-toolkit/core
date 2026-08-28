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
    This pass lowers all operations from the QC dialect to their equivalent operations in the LLVM dialect, ensuring compliance with the QIR Adaptive Profile 2.1 standard.
    It translates quantum operations and types from QC to their corresponding representations in QIR, facilitating interoperability with quantum computing frameworks that utilize the QIR standard.

    Requirements:

    - Input is a valid module in the QC dialect.
    - The entry function must be marked with `mqt.entry_point`.
    - The entry function must be defined and have no arguments.
    - QC, CBit, and MemRef operations must occur only in the entry function.
    - Input may contain at most 64 region-owning operations along any IR path.
    - The total width of all `cbit.alloc` operations must not exceed 1,048,576
      (`1 << 20`) classical result slots per module.
    - Existing `__quantum__` runtime symbols must be Func or LLVM function
      declarations.
    - Static and dynamic qubit allocation modes must not be mixed.
    - Qubit MemRef registers and loads must be one-dimensional.

    Behavior:

    - Each QC quantum operation is replaced by a call to the corresponding QIR function in the LLVM dialect.
    - Operations from the scf dialect are lowered into the cf dialect before they are lowered to the LLVM dialect.
    - QIR module flags and entry-point passthrough metadata are attached by the
      separate QIR metadata pass, normally through the QIR cleanup pipeline.
    - The pass extends the existing block structure of the entry function with an entry block and an epilogue block:
      0. Initialization block: Sets up the execution environment and performs required runtime initialization.
      1. Epilogue block: Records measurement results and returns from the entry function.
      Any blocks in-between have no restrictions regarding their operations as long as they are supported.
    - Measurement results may be used as classical values to drive conditional branches.
    - Supported Func, SCF, Arith, CF, and Math operations are lowered via
      MLIR's built-in conversions; other dialects are preserved.
  }];

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::cf::ControlFlowDialect",
                           "mlir::LLVM::LLVMDialect", "mlir::qc::QCDialect",
                           "mlir::qco::QCODialect"];
}
