// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def MQTDynToMQTOpt : Pass<"mqtdyn-to-mqtopt"> {
  let summary = "Convert MQT's `MQTDyn` to MQT's `MQTOpt` dialect.";

  let description = [{
      This pass converts all current mqtdyn operations to an equivalent mqtopt operation.
      To convert them to the opt dialect, the conversion pass keeps an updated map that tracks each dyn qubit
      and dyn qubit register to its latest state value when replacing them with an opt operation. These state values
      are obtained by updating the map with the return values of the newly created opt operations. The mqtopt insert
      operation is added when matching the dealloc operation of any qubit register. A mqtopt insert operation is then
      created for each qubit that was extracted from the qubit register and placed before the dealloc operation.
   }];

  // Define dependent dialects
  let dependentDialects = [
    "::mqt::ir::dyn::MQTDynDialect",
    "::mqt::ir::opt::MQTOptDialect"
  ];
}
