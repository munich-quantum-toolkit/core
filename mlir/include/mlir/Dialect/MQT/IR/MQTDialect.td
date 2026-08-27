// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MLIR_DIALECT_MQT_IR_MQTDIALECT_TD
#define MLIR_DIALECT_MQT_IR_MQTDIALECT_TD

include "mlir/IR/OpBase.td"

def MQTDialect : Dialect {
  let name = "mqt";
  let cppNamespace = "::mlir::mqt";

  let summary = "Shared metadata for MQT quantum programs.";
  let description = [{
    The MQT dialect owns frontend-neutral metadata that must remain meaningful
    across quantum dialect conversions. It defines no operations or types.

    `mqt.input_name` records the source-level name of a function input.
    `mqt.parameter_group` optionally preserves the source-level vector
    identity, name, element index, and size of a function input or a lexically
    bound `scf.for` parameter.
    `mqt.register_name` records the source-level name of a quantum or classical
    register allocation. Input and register names share one function-wide
    namespace.
    `mqt.entry_point` marks the single defined program entry function in a
    module.
  }];

  let discardableAttrs = (ins "::mlir::StringAttr":$input_name,
      "::mlir::DictionaryAttr":$parameter_group,
      "::mlir::StringAttr":$register_name, "::mlir::UnitAttr":$entry_point);

  let hasOperationAttrVerify = 1;
  let hasRegionArgAttrVerify = 1;
  let hasRegionResultAttrVerify = 1;
}

#endif // MLIR_DIALECT_MQT_IR_MQTDIALECT_TD
