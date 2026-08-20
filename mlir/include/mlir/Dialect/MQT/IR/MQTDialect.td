// Copyright (c) 2026 Chair for Design Automation, TUM
// Copyright (c) 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MLIR_DIALECT_MQT_IR_MQTDIALECT_TD
#define MLIR_DIALECT_MQT_IR_MQTDIALECT_TD

include "mlir/IR/DialectBase.td"

def MQTDialect : Dialect {
  let name = "mqt";
  let cppNamespace = "::mlir::mqt";

  let summary = "Shared metadata for MQT quantum programs.";
  let description = [{
    The MQT dialect owns frontend-neutral metadata that must remain meaningful
    across quantum dialect conversions. It defines no operations or types.

    `mqt.input_name` records the source-level name of a function input.
    `mqt.qubit_register_name` records the source-level name of a rank-one qubit
    register allocation.
  }];

  let discardableAttrs = (ins "::mlir::StringAttr":$input_name,
      "::mlir::StringAttr":$qubit_register_name);

  let hasOperationAttrVerify = 1;
  let hasRegionArgAttrVerify = 1;
  let hasRegionResultAttrVerify = 1;
}

#endif // MLIR_DIALECT_MQT_IR_MQTDIALECT_TD
