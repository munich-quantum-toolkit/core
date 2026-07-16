// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MLIR_CONVERSION_OQ3TOQC_OQ3TOQC_TD
#define MLIR_CONVERSION_OQ3TOQC_OQ3TOQC_TD

include "mlir/Pass/PassBase.td"

def OQ3ToQC : Pass<"oq3-to-qc", "::mlir::ModuleOp"> {
  let summary = "Convert supported typed OpenQASM operations to QC";
  let description = [{
    Resolves the target-neutral OQ3 gate contract into QC operations. The pass
    converts reachable custom gate bodies before their applications, rejects
    reachable recursion and excessive expansion, and makes OQ3 illegal in the
    conversion target so that success proves no OQ3 operations remain.

    Conversion can fail for valid source semantics that QC cannot represent
    faithfully. This includes power modifiers and inverse or control modifiers
    on custom gates whose bodies contain structured control flow.
  }];
  let constructor = "mlir::oq3::createOQ3ToQCPass()";
  let dependentDialects = ["mlir::oq3::OQ3Dialect", "mlir::qc::QCDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect", "mlir::scf::SCFDialect"];
}

#endif // MLIR_CONVERSION_OQ3TOQC_OQ3TOQC_TD
