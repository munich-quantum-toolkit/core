/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/WState.hpp"

#include "mqt/Dialect/QC/Builder/QCProgramBuilder.h"

#include "Programs.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mqt::bench {
using namespace mlir;

SmallVector<Value> wState(qc::QCProgramBuilder& b, const WState& benchmark) {
  const auto size = static_cast<int64_t>(benchmark.options().qubits);
  auto q = b.allocQubitRegisterStorage(size, "q");
  auto result = b.allocClassicalBitRegister(size, benchmark.output().name);
  b.x(b.loadQubit(q, b.indexConstant(0)));
  if (size > 1) {
    std::vector<double> angles(static_cast<size_t>(size - 1));
    for (size_t i = 0; i < angles.size(); ++i) {
      angles[i] = 2. * std::acos(1. / std::sqrt(static_cast<double>(size) -
                                                static_cast<double>(i)));
    }
    const auto type = RankedTensorType::get({size - 1}, b.getF64Type());
    auto table = arith::ConstantOp::create(
        b, DenseElementsAttr::get(type, ArrayRef<double>(angles)));
    auto one = b.indexConstant(1);
    b.scfFor(0, size - 1, 1, [&](Value index) {
      auto next = arith::AddIOp::create(b, index, one);
      auto left = b.loadQubit(q, index);
      auto right = b.loadQubit(q, next);
      auto angle = tensor::ExtractOp::create(b, table, ValueRange{index});
      b.cry(angle, left, right);
      b.cx(right, left);
    });
  }
  b.measureQubitRegister(q, result, size);
  return {result};
}
} // namespace mqt::bench
