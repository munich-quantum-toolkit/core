// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def QuakeToMQTDyn : Pass<"quake-to-mqtdyn"> {
  let summary = "Convert NVIDIA's `Quake` to MQT's `MQTDyn` dialect.";

  let description = [{
    TODO.
   }];

  // Define dependent dialects
  let dependentDialects = [
    "cudaq::quake::QuakeDialect",
    "::mqt::ir::dyn::MQTDynDialect"
  ];
}

