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

#include "QASM3Parser.h"

#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Support/LLVM.h>

#include <exception>
#include <string_view>
#include <utility>

namespace mlir::qc {

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

OwningOpRef<ModuleOp> translateQASM3ToQC(llvm::SourceMgr& sourceMgr,
                                         MLIRContext* context) {
  try {
    const auto buffer =
        sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID())->getBuffer();
    return detail::parseQASM3(std::string_view(buffer.data(), buffer.size()),
                              context);
  } catch (const std::exception& e) {
    llvm::errs() << "Import error: " << e.what() << "\n";
    return nullptr;
  }
}

OwningOpRef<ModuleOp> translateQASM3ToQC(StringRef source,
                                         MLIRContext* context) {
  llvm::SourceMgr sourceMgr;
  auto buffer = llvm::MemoryBuffer::getMemBufferCopy(source);
  sourceMgr.AddNewSourceBuffer(std::move(buffer), SMLoc());
  return translateQASM3ToQC(sourceMgr, context);
}

} // namespace mlir::qc
