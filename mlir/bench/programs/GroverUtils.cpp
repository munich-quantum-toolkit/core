/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "GroverUtils.h"

#include "mqt/Dialect/QC/Builder/QCProgramBuilder.h"

#include "mlir/IR/Value.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstddef>
#include <string_view>

namespace mqt::bench::detail {

void maskMarkedState(mlir::qc::QCProgramBuilder& builder,
                     llvm::ArrayRef<mlir::Value> search,
                     std::string_view markedBitstring) {
  for (size_t index = 0; index < search.size(); ++index) {
    if (markedBitstring[search.size() - 1 - index] == '0') {
      builder.x(search[index]);
    }
  }
}

void markPhase(mlir::qc::QCProgramBuilder& builder,
               llvm::ArrayRef<mlir::Value> search,
               std::string_view markedBitstring) {
  maskMarkedState(builder, search, markedBitstring);
  builder.mcz(search.drop_back(), search.back());
  maskMarkedState(builder, search, markedBitstring);
}

} // namespace mqt::bench::detail
