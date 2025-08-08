// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def MQTRefToMQTOpt : Pass<"mqtref-to-mqtopt"> {
    let summary = "Convert MQT's `Ref` to MQT's `Opt` dialect.";

    let description = [{
      This pass converts all current MQTRef operations to an equivalent MQTRef operation.
      To convert them to the opt dialect, the conversion pass maintains a map that matches each ref qubit
      and ref qubit register to its latest state value when replacing them with an opt operation. These state values
      are obtained by updating the map with the return values of the newly created opt operations. The MQTOpt insert
      operation is added when matching the dealloc operation of any qubit register. An MQTOpt insert operation is then
      created for each qubit that was extracted from the qubit register and placed before the dealloc operation.
   }];

  // Define dependent dialects
  let dependentDialects = [
    "::mqt::ir::ref::MQTRefDialect",
    "::mqt::ir::opt::MQTOptDialect"
  ];
}
