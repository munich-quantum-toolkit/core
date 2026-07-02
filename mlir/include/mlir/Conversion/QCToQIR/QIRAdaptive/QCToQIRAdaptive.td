// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def QCToQIRAdaptive : Pass<"qc-to-qir-adaptive"> {
  let summary = "Lower the QC dialect to the LLVM dialect compliant with the "
                "QIR Adaptive Profile 2.1";

  let description = [{
    This pass lowers all operations from the QC dialect to their equivalent operations in the LLVM dialect, ensuring compliance with the QIR Adaptive Profile 2.1 standard.
    It translates quantum operations and types from QC to their corresponding representations in QIR, facilitating interoperability with quantum computing frameworks that utilize the QIR standard.

    Requirements:

    - Input is a valid module in the QC dialect.
    - The entry function must be marked with the `entry_point` attribute.

    Behavior:

    - Each QC quantum operation is replaced by a call to the corresponding QIR function in the LLVM dialect.
    - Operations from the scf dialect are lowered into the cf dialect before they are lowered to the LLVM dialect.
    - Required QIR module flags are attached to the MLIR module.
    - Required attributes are attached to the entry function.
    - The pass extends the existing block structure of the entry function with an entry block and an epilogue block:
      0. Initialization block: Sets up the execution environment and performs required runtime initialization.
      1. Epilogue block: Records measurement results and returns from the entry function.
      Any blocks in-between have no restrictions regarding their operations as long as they are supported.
    - Measurement results may be used as classical values to drive conditional branches.
    - Non-quantum dialects are lowered via MLIR's built-in conversions.
  }];

  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}
