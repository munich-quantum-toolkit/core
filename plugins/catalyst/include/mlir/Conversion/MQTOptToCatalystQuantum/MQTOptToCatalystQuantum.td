// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def MQTOptToCatalystQuantum : Pass<"mqtopt-to-catalystquantum"> {
  let summary = "Convert MQT's `MQTOpt` to Catalyst's `Quantum` dialect.";

  let description = [{
    This pass converts MQT's `MQTOpt` to Catalyst's `Quantum` dialect.
    The following operations are currently NOT converted (and instead marked legal):
    - IOp
    - GPhaseOp
    - BarrierOp
    - SOp, SdgOp
    - TOp, TdgOp
    - VOp, VdgOp
    - UOp, U2Op
    - SXOp, SXdgOp
    - iSWAPOp, iSWAPdgOp
    - PeresOp, PeresdgOp
    - DCXOp
    - ECROp
    - RXXOp, RYYOp, RZZOp, RZXOp
    - XXminusYY, XXplusYY
  }];
  let dependentDialects = [
    "catalyst::quantum::QuantumDialect",
    "::mqt::ir::opt::MQTOptDialect"
  ];
}
