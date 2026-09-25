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
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include <cstdint>

namespace mqt::bench {
using namespace mlir;

SmallVector<Value> wState(qc::QCProgramBuilder& b, const WState& benchmark) {
  const auto size = static_cast<int64_t>(benchmark.options().qubits);
  auto q = b.allocQubitRegisterStorage(size, "q");
  auto result = b.allocClassicalBitRegister(size, benchmark.output().name);
  b.x(b.loadQubit(q, b.indexConstant(0)));
  auto count = b.indexConstant(size);
  auto step = b.indexConstant(1);
  auto one = b.floatConstant(1.);
  auto two = b.floatConstant(2.);
  b.scfFor(0, size - 1, 1, [&](Value index) {
    auto remaining = arith::SubIOp::create(b, count, index);
    auto integer = arith::IndexCastOp::create(b, b.getI64Type(), remaining);
    auto floating = arith::SIToFPOp::create(b, b.getF64Type(), integer);
    auto root = math::SqrtOp::create(b, floating);
    auto cosine = arith::DivFOp::create(b, one, root);
    auto angle = arith::MulFOp::create(b, two, math::AcosOp::create(b, cosine));
    auto next = arith::AddIOp::create(b, index, step);
    auto left = b.loadQubit(q, index);
    auto right = b.loadQubit(q, next);
    b.cry(angle, left, right);
    b.cx(right, left);
  });
  b.measureQubitRegister(q, result, size);
  return {result};
}
} // namespace mqt::bench
