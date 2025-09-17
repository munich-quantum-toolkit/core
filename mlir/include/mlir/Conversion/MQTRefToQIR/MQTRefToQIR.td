// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def MQTRefToQIR : Pass<"mqtref-to-qir"> {
   let summary = "Lower the `MQTRef` dialect to the LLVM dialect compliant with QIR 2.0 (Base Profile)";

   let description = [{
      Overview
      Converts the MQTRef dialect to the LLVM dialect following the QIR 2.0 Base Profile. Quantum operations are mapped to calls
      into the QIR instruction set, while non-quantum operations are lowered using MLIR's standard LLVM conversions.

      Requirements
      - Input is a valid module in the MQTRef dialect.
      - Static or dynamic qubit addressing is supported.
      - The entry function must be marked with the `entry_point` attribute.
      - The program must have straight-line control flow (no complex branching).

      Behavior
      - Each MQTRef quantum operation is replaced by a call to the corresponding QIR function in the LLVM dialect.
      - Required QIR module flags are attached as attributes to the entry function.
      - The entry function is split into four blocks to satisfy QIR Base Profile constraints.
      - Non-quantum dialects are lowered via MLIR's built-in conversions.

      Block structure (Base Profile)
      0. Initialization block
         - Sets up the execution environment and performs required runtime initialization.
      1. Reversible operations block
         - Contains only void-returning calls to reversible quantum operations.
      2. Irreversible operations block
         - Contains only void-returning calls to operations marked irreversible, e.g.:
           `__quantum__qis__mz__body`, `__quantum__rt__qubit_release_array`,
           `__quantum__rt__qubit_release`, `__quantum__qis__reset__body`.
      3. Epilogue block
         - Records measurement results and returns from the entry function.
      - Blocks are connected via unconditional branches in the order listed above.

      Operation mapping
      - `mqtref.measure` → call to `__quantum__qis__mz__body` followed by
        `__quantum__rt__result_record_output` to record the measurement result.

      Lowering of non-quantum operations
      - Dialects lowered via MLIR conversions: `func`, `arith`, `cf` → LLVM dialect.

      Limitations
      - Only straight-line control flow is supported.
      - Adheres to QIR Base Profile requirements.
      - Unsupported operations or control flow patterns may cause the conversion to fail.

      Producing LLVM IR
      - After conversion to the LLVM dialect, produce LLVM IR with:
        mlir-translate --mlir-to-llvmir input.mlir > output.ll
   }];

  // Define dependent dialects
  let dependentDialects = [
     "::mqt::ir::ref::MQTRefDialect",
     "mlir::LLVM::LLVMDialect"
  ];
}
