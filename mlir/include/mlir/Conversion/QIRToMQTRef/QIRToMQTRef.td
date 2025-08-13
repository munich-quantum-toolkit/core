// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def QIRToMQTRef : Pass<"qir-to-mqtref"> {
    let summary = "Convert qir to MQT's `Ref` ";

    let description = [{
   }];

  // Define dependent dialects
  let dependentDialects = [
    "::mqt::ir::ref::MQTRefDialect",
    "mlir::LLVM::LLVMDialect"
  ];
}
