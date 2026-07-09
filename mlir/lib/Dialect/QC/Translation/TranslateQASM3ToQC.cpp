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

#include "mlir/Dialect/QC/Translation/qasm3/QASM3Emitter.h"

#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Support/LLVM.h>

#include <utility>

namespace mlir::qc {

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

OwningOpRef<ModuleOp> translateQASM3ToQC(llvm::SourceMgr& sourceMgr,
                                         MLIRContext* context) {
  // Route diagnostics through the source manager so errors carry source
  // locations and snippets.
  const SourceMgrDiagnosticHandler handler(sourceMgr, context);
  return detail::importQASM3(sourceMgr, context);
}

OwningOpRef<ModuleOp> translateQASM3ToQC(StringRef source,
                                         MLIRContext* context) {
  llvm::SourceMgr sourceMgr;
  auto buffer = llvm::MemoryBuffer::getMemBufferCopy(source);
  sourceMgr.AddNewSourceBuffer(std::move(buffer), SMLoc());
  return translateQASM3ToQC(sourceMgr, context);
}

} // namespace mlir::qc
