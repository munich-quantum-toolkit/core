// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MLIR_DIALECT_OQ3_TRANSFORMS_PASSES_TD
#define MLIR_DIALECT_OQ3_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def LowerOQ3ToQC : Pass<"lower-oq3-to-qc", "::mlir::ModuleOp"> {
  let summary = "Lower supported typed OpenQASM operations to QC";
  let constructor = "mlir::oq3::createLowerOQ3ToQCPass()";
  let dependentDialects = ["mlir::oq3::OQ3Dialect", "mlir::qc::QCDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect", "mlir::scf::SCFDialect"];
}

#endif // MLIR_DIALECT_OQ3_TRANSFORMS_PASSES_TD
