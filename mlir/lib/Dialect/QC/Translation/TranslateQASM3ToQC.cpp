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

#include "OpenQASMToQCEmitter.h"
#include "mlir/Target/OpenQASM/Frontend.h"

#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Support/LLVM.h>

namespace mlir::qc {

static void
emitDiagnostics(const ArrayRef<oq3::frontend::Diagnostic> diagnostics,
                MLIRContext& context) {
  for (const auto& diagnostic : diagnostics) {
    emitError(detail::getOpenQASMLocation(diagnostic.location, context))
        << "OpenQASM frontend error: " << diagnostic.message;
  }
}

OwningOpRef<ModuleOp> translateQASM3ToQC(llvm::SourceMgr& sourceMgr,
                                         MLIRContext* context) {
  auto analyzed = oq3::frontend::analyzeOpenQASM(sourceMgr);
  if (!analyzed) {
    emitDiagnostics(analyzed.diagnostics, *context);
    return nullptr;
  }
  auto moduleOp = detail::emitOpenQASMToQC(*analyzed.program, *context);
  if (!moduleOp) {
    return nullptr;
  }
  if (failed(verify(*moduleOp))) {
    return nullptr;
  }
  return moduleOp;
}

OwningOpRef<ModuleOp> translateQASM3ToQC(const StringRef source,
                                         MLIRContext* context) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(source, "<input>"), llvm::SMLoc());
  return translateQASM3ToQC(sourceMgr, context);
}

} // namespace mlir::qc
