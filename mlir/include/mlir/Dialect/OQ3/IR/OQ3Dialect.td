// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MLIR_DIALECT_OQ3_IR_OQ3DIALECT_TD
#define MLIR_DIALECT_OQ3_IR_OQ3DIALECT_TD

include "mlir/IR/DialectBase.td"

def OQ3Dialect : Dialect {
  let name = "oq3";
  let summary = "Experimental typed OpenQASM 3.1 semantic dialect";
  let description = [{
    OQ3 is a typed semantic intermediate representation for OpenQASM 3.1.
    It deliberately reuses builtin MLIR dialects for classical computation
    and only models language concepts that do not have a faithful builtin
    representation. The dialect is experimental and is not yet a stable
    textual interface.
  }];
  let cppNamespace = "::mlir::oq3";
  let useDefaultTypePrinterParser = 1;
}

#endif // MLIR_DIALECT_OQ3_IR_OQ3DIALECT_TD
