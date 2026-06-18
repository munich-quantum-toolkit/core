/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Translation/Translation.h"

#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QC/Translation/TranslateQASM3ToQC.h"

#include <llvm/Support/SourceMgr.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Tools/mlir-translate/Translation.h>

namespace mlir {

void registerQASM3ToQCTranslation() {
  static TranslateToMLIRRegistration registration(
      "qasm3-to-qc", "translate an OpenQASM 3 program to a QC program",
      [](llvm::SourceMgr& sourceMgr,
         MLIRContext* context) -> OwningOpRef<Operation*> {
        context->loadDialect<arith::ArithDialect, func::FuncDialect,
                             memref::MemRefDialect, qc::QCDialect,
                             scf::SCFDialect>();
        return OwningOpRef<Operation*>(
            qc::translateQASM3ToQC(sourceMgr, context).release());
      });
}

} // namespace mlir
