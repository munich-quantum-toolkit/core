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

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>

namespace mqt::test {

inline void zeroInitializeClassicalRegisters(mlir::ModuleOp moduleOp) {
  mlir::SmallVector<mlir::memref::AllocOp> registers;
  moduleOp.walk([&](mlir::memref::AllocOp allocation) {
    const auto type = allocation.getType();
    if (type.getRank() == 1 && type.getElementType().isInteger(1)) {
      registers.push_back(allocation);
    }
  });
  if (registers.empty()) {
    return;
  }

  mlir::OpBuilder builder(registers.front());
  const auto location = builder.getUnknownLoc();
  auto zero = mlir::arith::ConstantOp::create(builder, location,
                                              builder.getBoolAttr(false));
  for (auto allocation : registers) {
    builder.setInsertionPointAfter(allocation);
    const auto width = allocation.getType().getShape().front();
    for (int64_t bit = 0; bit < width; ++bit) {
      auto index = mlir::arith::ConstantIndexOp::create(builder, location, bit);
      mlir::memref::StoreOp::create(builder, location, zero.getResult(),
                                    allocation, index.getResult());
    }
  }
}

} // namespace mqt::test
