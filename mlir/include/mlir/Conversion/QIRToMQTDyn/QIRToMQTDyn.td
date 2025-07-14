// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def QIRToMQTDyn : Pass<"qir-to-mqtdyn"> {
    let summary = "Convert qir to MQT's `Dyn` ";

    let description = [{
   }];

  // Define dependent dialects
  let dependentDialects = [
    "::mqt::ir::dyn::MQTDynDialect",
    "mlir::func::FuncDialect",
    "mlir::LLVM::LLVMDialect"
  ];
}
