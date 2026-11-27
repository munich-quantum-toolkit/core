// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MQT_DELAY_UNIT_TD
#define MQT_DELAY_UNIT_TD

include "mlir/IR/EnumAttr.td"

// Shared by QC and QCO delays; the attribute prints as a bare keyword.
def MQT_DelayUnit
    : I32EnumAttr<"DelayUnit", "Time unit of a delay duration",
                  [I32EnumAttrCase<"dt", 0>, I32EnumAttrCase<"s", 1>,
                   I32EnumAttrCase<"ms", 2>, I32EnumAttrCase<"us", 3>,
                   I32EnumAttrCase<"ns", 4>, I32EnumAttrCase<"ps", 5>]> {
  let cppNamespace = "::mlir::mqt";
}

#endif // MQT_DELAY_UNIT_TD
