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
      This pass converts all current mqtopt operations to an equivalent mqtdyn operation.
      The result of the converted mqtdyn alloc operation acts as a qubit register reference and replaces the operand of the operation
      that uses the initial mqtopt qubit register. This reference is then propagated to all subsequent operations that use the state
      of the initial mqtopt register and replaces their operand. The same applies to the result of the extract operation that acts as a qubit
      reference and is propagated to all later uses of this qubit. The mqtopt insert operation is deleted as there is no equivalent
      operation in the mqtdyn dialect and any subsequent operation that uses the result of this operation acts on the qubit register reference.
   }];

  // Define dependent dialects
  let dependentDialects = [
    "::mqt::ir::dyn::MQTDynDialect",
    "::mqt::ir::opt::MQTOptDialect"
  ];
}
