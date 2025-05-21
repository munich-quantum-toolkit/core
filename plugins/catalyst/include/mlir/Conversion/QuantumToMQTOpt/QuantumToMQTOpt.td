// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def QuantumToMQTOpt : Pass<"catalystquantum-to-mqtopt"> {
  let summary = "Convert Catalyst's `Quantum` to MQT's `MQTOpt` dialect.";

  let description = [{
    This pass converts Catalyst's `Quantum` to MQT's `MQTOpt` dialect.
    The following operations are currently NOT converted (and instead marked legal):
    - DeviceInitOp
    - DeviceReleaseOp
    - NamedObsOp
    - ExpvalOp
    - FinalizeOp
    - ComputationalBasisOp
    - StateOp
    - InitializeOp
   }];

  // Define dependent dialects
  let dependentDialects = [
    "catalyst::quantum::QuantumDialect",
    "::mqt::ir::opt::MQTOptDialect"
  ];
}
