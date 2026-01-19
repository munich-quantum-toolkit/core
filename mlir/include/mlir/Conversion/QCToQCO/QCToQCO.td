// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def QCToQCO : Pass<"qc-to-qco"> {
  let summary = "Convert QC dialect to QCO dialect.";

  let description = [{
      This pass converts all operations from the QC dialect to their equivalent
      operations in the QCO dialect. It handles the transformation of qubit
      references in QC to qubit values in QCO, ensuring that the semantics
      of quantum operations are preserved during the conversion process.
   }];

  // Define dependent dialects
  let dependentDialects = [
    "mlir::qc::QCDialect",
    "mlir::qco::QCODialect"
  ];
}
