// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def MQTOptToMQTDyn : Pass<"mqtopt-to-mqtdyn"> {
  let summary = "Convert MQT's `MQTOpt` to MQT's `MQTDyn` dialect.";

  let description = [{

   }];

  // Define dependent dialects
  let dependentDialects = [
    "::mqt::ir::dyn::MQTDynDialect",
    "::mqt::ir::opt::MQTOptDialect"
  ];
}
