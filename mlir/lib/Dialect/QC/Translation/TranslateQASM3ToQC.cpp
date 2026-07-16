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
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>

#include <vector>

namespace mlir::qc {
namespace {

void printDiagnostics(
    const std::vector<oq3::frontend::Diagnostic>& diagnostics) {
  for (const auto& diagnostic : diagnostics) {
    llvm::errs() << diagnostic.location.filename << ':'
                 << diagnostic.location.line << ':'
                 << diagnostic.location.column
                 << ": OpenQASM frontend error: " << diagnostic.message << '\n';
  }
}

} // namespace

OwningOpRef<ModuleOp> translateQASM3ToQC(llvm::SourceMgr& sourceMgr,
                                         MLIRContext* context) {
  auto analyzed = oq3::frontend::analyzeOpenQASM(sourceMgr);
  if (!analyzed) {
    printDiagnostics(analyzed.diagnostics);
    return nullptr;
  }
  auto module = detail::emitOpenQASMToQC(*analyzed.program, *context);
  if (!module) {
    return nullptr;
  }
  if (failed(verify(*module))) {
    llvm::errs() << "OpenQASM emission produced invalid QC IR.\n";
    return nullptr;
  }
  return module;
}

OwningOpRef<ModuleOp> translateQASM3ToQC(const StringRef source,
                                         MLIRContext* context) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(source, "<input>"), llvm::SMLoc());
  return translateQASM3ToQC(sourceMgr, context);
}

} // namespace mlir::qc
