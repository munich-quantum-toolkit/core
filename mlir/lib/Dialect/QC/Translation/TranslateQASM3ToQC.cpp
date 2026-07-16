/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QC/Translation/TranslateQASM3ToQC.h"

#include "mlir/Conversion/OQ3ToQC/OQ3ToQC.h"
#include "mlir/Target/OpenQASM/OpenQASM.h"

#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

namespace mlir::qc {

OwningOpRef<ModuleOp> translateQASM3ToQC(llvm::SourceMgr& sourceMgr,
                                         MLIRContext* context) {
  auto module = oq3::translateOpenQASMToOQ3(sourceMgr, *context);
  if (!module) {
    return nullptr;
  }

  PassManager manager(context);
  manager.addPass(oq3::createOQ3ToQCPass());
  if (failed(manager.run(*module))) {
    llvm::errs() << "OpenQASM target lowering failed.\n";
    return nullptr;
  }
  if (failed(verify(*module))) {
    llvm::errs() << "OpenQASM target lowering produced invalid QC IR.\n";
    return nullptr;
  }
  return module;
}

OwningOpRef<ModuleOp> translateQASM3ToQC(const StringRef source,
                                         MLIRContext* context) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(source),
                               llvm::SMLoc());
  return translateQASM3ToQC(sourceMgr, context);
}

} // namespace mlir::qc
