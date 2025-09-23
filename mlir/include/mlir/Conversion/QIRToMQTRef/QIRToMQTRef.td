// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def QIRToMQTRef : Pass<"qir-to-mqtref"> {
    let summary = "Raise LLVM/QIR (Base Profile) to the `MQTRef` dialect by converting QIR calls into `mqtref` ops";

    let description = [{
      Overview
      Converts LLVM dialect modules that use QIR into MQTRef IR by pattern-matching and replacing calls to QIR functions with
      equivalent operations from the MQTRef dialect.

      Requirements
      - Input is a valid MLIR module in the LLVM dialect that follows QIR Base Profile conventions.
      - Static or dynamic qubit addressing is supported.
      - The program is expected to have straight-line control flow.

      Behavior
      - Replaces `llvm.call` operations targeting QIR functions with corresponding `mqtref` operations.
      - The conversion is driven by the QIR function name used by the call.

      Operation mapping
      - `__quantum__qis__mz__body` → `mqtref.measure` returning an `i1` measurement result.
      - The paired `__quantum__rt__result_record_output` for measurements is removed.

      Non-converted constructs
      - `llvm.mlir.constant` and function declarations are currently not converted and remain in the module.

      Limitations
      - Only simple, straight-line programs that follow the QIR Base Profile are supported.
      - QIR calls without a known mapping in `mqtref` cause the conversion to fail.

      Obtaining the input module
      - Import an LLVM IR file as MLIR with:
        mlir-translate --import-llvm input.ll -o output.mlir
   }];

  // Define dependent dialects
  let dependentDialects = [
    "::mqt::ir::ref::MQTRefDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect"
  ];
}
