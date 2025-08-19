// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def QIRToMQTRef : Pass<"qir-to-mqtref"> {
    let summary = "Convert llvm dialect that adheres to the QIR specification to MQT's `Ref` dialect";

    let description = [{
      This pass converts all llvm.mlir.call operation from the qir instruction set into quantum operations in the mqtref dialect.
      The pass expects a valid mlir module in the llvm dialect that uses dynamic addressing for the qubit allocation and uses the
      QIR base profile. At the moment the pass only supports simple quantum programs with a straight controlflow. The conversion
      of the quantum operation is done checking the function name of the call operation and replacing it with an equivalent operation
      from the mqtref dialect. Other operations from the llvm dialect like llvm.mlir.constants or function declarations are currently not
      converted and persist after the conversion. At the moment quantum operations that do not have an equivalent operation in
      the mqtref dialect are deleted during the conversion.

      The initial .mlir module with the llvm dialect can be obtained by converting a .ll file with the following command:
      mlir-translate --import-llvm input.ll -o output.mlir
   }];

  // Define dependent dialects
  let dependentDialects = [
    "::mqt::ir::ref::MQTRefDialect",
    "mlir::LLVM::LLVMDialect"
  ];
}
