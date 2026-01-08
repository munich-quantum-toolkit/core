/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Layout.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Unit.h"

#include <cstddef>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Region.h>
#include <mlir/Support/LLVM.h>

namespace mqt::ir::opt {

/// @brief A SequentialUnit traverses a program sequentially.
class SequentialUnit : public Unit<SequentialUnit> {
public:
  [[nodiscard]] static SequentialUnit
  fromEntryPointFunction(mlir::func::FuncOp func, std::size_t nqubits);

  SequentialUnit(Layout layout, mlir::Region* region,
                 mlir::Region::OpIterator start);

  SequentialUnit(Layout layout, mlir::Region* region)
      : SequentialUnit(std::move(layout), region, region->op_begin()) {}

private:
  friend class Unit<SequentialUnit>;

  [[nodiscard]] mlir::SmallVector<SequentialUnit, 3> nextImpl();
  [[nodiscard]] mlir::Region::OpIterator beginImpl() const { return start_; }
  [[nodiscard]] mlir::Region::OpIterator endImpl() const { return end_; }

  mlir::Region::OpIterator start_;
  mlir::Region::OpIterator end_;
};
} // namespace mqt::ir::opt
