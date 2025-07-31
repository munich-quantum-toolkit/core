// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def MQTDynToQIR : Pass<"mqtdyn-to-qir"> {
    let summary = "Convert MQT's `Dyn` to qir";

    let description = [{
   }];

  // Define dependent dialects
  let dependentDialects = [
     "::mqt::ir::dyn::MQTDynDialect",
     "mlir::LLVM::LLVMDialect"
  ];
}
