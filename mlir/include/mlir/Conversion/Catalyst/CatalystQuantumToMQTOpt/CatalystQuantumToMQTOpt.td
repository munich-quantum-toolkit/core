// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def CatalystQuantumToMQTOpt : Pass<"quantum-to-mqtopt"> {
  let summary = "Lower `quantum` to `mqtopt` dialect.";

  let description = [{
    This pass lowers the `quantum` to `mqtopt` dialect.
  }];

  // Define dependent dialects
  let dependentDialects = [
    "catalyst::quantum::QuantumDialect",
    "::mqt::ir::opt::MQTOptDialect"
  ];
}
