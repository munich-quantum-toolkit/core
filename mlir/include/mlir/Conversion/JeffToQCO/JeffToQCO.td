// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def JeffToQCO : Pass<"jeff-to-qco"> {
  let summary = "Convert Jeff operations to QCO operations";

  let description = [{
      TODO
   }];

  let dependentDialects = [
     "jeff::JeffDialect",
     "mlir::QCODialect",
  ];
}
