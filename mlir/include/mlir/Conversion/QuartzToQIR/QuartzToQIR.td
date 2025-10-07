// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def QuartzToQIR : Pass<"quartz-to-qir"> {
  let summary = "Lower the Quartz dialect to the LLVM dialect compliant with QIR 2.0";

  let description = [{
      This pass lowers all operations from the Quartz dialect to their equivalent
      operations in the LLVM dialect, ensuring compliance with the QIR 2.0 standard.
      It translates quantum operations and types from Quartz to their corresponding
      representations in QIR, facilitating interoperability with quantum computing
      frameworks that utilize the QIR standard.

      Requirements:
      - Input is a valid module in the Quartz dialect.
      - The entry function must be marked with the `entry_point` attribute.
      - The program must have straight-line control flow (i.e., Base Profile QIR).

      Behavior:
      - Each Quartz quantum operation is replaced by a call to the corresponding QIR function in the LLVM dialect.
      - Required QIR module flags are attached as attributes to the entry function.
      - The entry function is split into four blocks to satisfy QIR Base Profile constraints:
        0. Initialization block: Sets up the execution environment and performs required runtime initialization.
        1. Reversible operations block: Contains only void-returning calls to reversible quantum operations.
        2. Irreversible operations block: Contains only void-returning calls to operations marked irreversible, e.g.:
           `__quantum__qis__mz__body`, `__quantum__rt__qubit_release_array`,
           `__quantum__rt__qubit_release`, `__quantum__qis__reset__body`.
        3. Epilogue block: Records measurement results and returns from the entry function.
      - Blocks are connected via unconditional branches in the order listed above.
      - Non-quantum dialects are lowered via MLIR's built-in conversions.

     Producing LLVM IR:
     - After conversion to the LLVM dialect, produce LLVM IR with:
       mlir-translate --mlir-to-llvmir input.mlir > output.ll
   }];


  // Define dependent dialects
  let dependentDialects = [
     "mlir::memref::MemRefDialect",
     "mlir::LLVM::LLVMDialect",
     "mlir::QuartzDialect",
  ];
}
