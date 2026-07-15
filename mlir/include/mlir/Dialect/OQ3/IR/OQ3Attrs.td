// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MLIR_DIALECT_OQ3_IR_OQ3ATTRS_TD
#define MLIR_DIALECT_OQ3_IR_OQ3ATTRS_TD

include "mlir/Dialect/OQ3/IR/OQ3Dialect.td"
include "mlir/IR/EnumAttr.td"

def GateModifierKind : I32EnumAttr<"GateModifierKind", "OpenQASM gate modifier",
                                   [I32EnumAttrCase<"inv", 0, "Inv">,
                                    I32EnumAttrCase<"ctrl", 1, "Ctrl">,
                                    I32EnumAttrCase<"negctrl", 2, "NegCtrl">,
                                    I32EnumAttrCase<"pow", 3, "Pow">]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::oq3";
}

#endif // MLIR_DIALECT_OQ3_IR_OQ3ATTRS_TD
