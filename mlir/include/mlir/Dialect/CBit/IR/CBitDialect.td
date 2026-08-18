// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MLIR_DIALECT_CBIT_IR_CBITDIALECT_TD
#define MLIR_DIALECT_CBIT_IR_CBITDIALECT_TD

include "mlir/IR/DialectBase.td"

def CBitDialect : Dialect {
  let name = "cbit";
  let cppNamespace = "::mlir::cbit";
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;

  let summary = "Classical-bit register storage shared by quantum dialects";
  let description = [{
    The CBit dialect represents non-aliasing classical-bit registers whose
    width, initialization, and public identity remain explicit across quantum
    IR conversions. It separates source-level classical storage from generic
    memory lowering.
  }];
}

#endif // MLIR_DIALECT_CBIT_IR_CBITDIALECT_TD
