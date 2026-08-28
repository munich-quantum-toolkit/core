// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def QCOToQC : Pass<"qco-to-qc", "mlir::ModuleOp"> {
  let summary = "Convert QCO dialect to QC dialect.";

  let description = [{
    This pass converts all operations from the QCO dialect to their equivalent operations in the QC dialect.
    It handles the transformation of qubit values in QCO to qubit references in QC, ensuring that the semantics of quantum operations are preserved during the conversion process.

    Every physical static-qubit identity must have at most one live QCO wire.
    A static index may be reacquired after its previous wire is explicitly sunk
    earlier in the same block.
    Input may contain at most 64 region-owning operations along any IR path.
  }];

  let dependentDialects = ["mlir::cf::ControlFlowDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect", "mlir::qc::QCDialect",
                           "mlir::scf::SCFDialect"];
}
