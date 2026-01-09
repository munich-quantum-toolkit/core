// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def QCOToQC : Pass<"qco-to-qc"> {
  let summary = "Convert QCO dialect to QC dialect.";

  let description = [{
      This pass converts all operations from the QCO dialect to their equivalent
      operations in the QC dialect. It handles the transformation of qubit
      values in QCO to qubit references in QC, ensuring that the semantics
      of quantum operations are preserved during the conversion process.
   }];

  // Define dependent dialects
  let dependentDialects = [
    "mlir::qco::QCODialect",
    "mlir::qc::QCDialect"
  ];
}
