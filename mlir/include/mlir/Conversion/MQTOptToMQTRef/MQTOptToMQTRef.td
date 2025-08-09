// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def MQTOptToMQTRef : Pass<"mqtopt-to-mqtref"> {
  let summary = "Convert MQT's `Opt` to MQT's `Ref` dialect.";

  let description = [{
      This pass converts all current MQTOpt operations to an equivalent MQTRef operation.
      The result of the converted MQTRef alloc operation acts as a qubit register reference and replaces the operand of the operation
      that uses the initial MQTOpt qubit register. This reference is then propagated to all subsequent operations that use the state
      of the initial MQTOpt register and replaces their operand. The same applies to the result of the extract operation that acts as a qubit
      reference and is propagated to all later uses of this qubit. The MQTOpt insert operation is deleted as there is no equivalent
      operation in the MQTRef dialect. The use of the insert operation's result in any subsequent operation is replaced by the qubit register reference.
   }];

  // Define dependent dialects
  let dependentDialects = [
    "::mqt::ir::ref::MQTRefDialect",
    "::mqt::ir::opt::MQTOptDialect"
  ];
}
