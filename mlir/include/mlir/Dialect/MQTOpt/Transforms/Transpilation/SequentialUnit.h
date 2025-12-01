/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Layout.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Unit.h"

#include <cstddef>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Region.h>
#include <mlir/Support/LLVM.h>

namespace mqt::ir::opt {

/// @brief A SequentialUnit traverses a program sequentially.
class SequentialUnit : public Unit {
public:
  static SequentialUnit fromEntryPointFunction(mlir::func::FuncOp func,
                                               const std::size_t nqubits) {
    Layout layout(nqubits);
    for_each(func.getOps<QubitOp>(), [&](QubitOp op) {
      layout.add(op.getIndex(), op.getIndex(), op.getQubit());
    });
    return {std::move(layout), &func.getBody()};
  }

  SequentialUnit(Layout layout, mlir::Region* region,
                 mlir::Region::OpIterator start, bool restore = false);

  SequentialUnit(Layout layout, mlir::Region* region, bool restore = false)
      : SequentialUnit(std::move(layout), region, region->op_begin(), restore) {
  }

  [[nodiscard]] mlir::SmallVector<SequentialUnit, 3> next();
  [[nodiscard]] mlir::Region::OpIterator begin() const { return start_; }
  [[nodiscard]] mlir::Region::OpIterator end() const { return end_; }

private:
  mlir::Region::OpIterator start_;
  mlir::Region::OpIterator end_;
};
} // namespace mqt::ir::opt
