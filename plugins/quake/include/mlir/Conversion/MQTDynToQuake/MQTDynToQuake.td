// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def MQTDynToQuake : Pass<"mqtdyn-to-quake"> {
  let summary = "Convert MQT's `MQTDyn` to NVIDIA's `Quake` dialect.";

  let description = [{
    TODO.
   }];

  // Define dependent dialects
  let dependentDialects = [
    "quake::QuakeDialect",
    "::mqt::ir::dyn::MQTDynDialect"
  ];
}
