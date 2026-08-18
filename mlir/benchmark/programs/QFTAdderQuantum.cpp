/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Benchmark/Programs.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <numbers>

namespace mqt::benchmark {

using namespace mlir;

namespace {
/// Applies a quantum Fourier transform, or its inverse, to @p reg.
void transform(qc::QCProgramBuilder& b, Value reg, const int64_t size,
               const double sign) {
  auto one = b.indexConstant(1);
  auto last = b.indexConstant(size - 1);

  b.scfFor(0, size, 1, [&](Value step) {
    // The inverse runs the same rotations in the opposite order.
    auto i = sign > 0.0 ? step : Value{arith::SubIOp::create(b, last, step)};
    if (sign < 0.0) {
      b.h(b.loadQubit(reg, i));
    }

    auto lower = arith::AddIOp::create(b, i, one);
    auto upper = b.indexConstant(size);
    auto start = b.floatConstant(sign * std::numbers::pi / 2.0);
    auto half = b.floatConstant(0.5);

    auto loop = scf::ForOp::create(b, lower, upper, one, ValueRange{start});
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(loop.getBody());
      auto angle = loop.getRegionIterArg(0);
      b.cp(angle, b.loadQubit(reg, loop.getInductionVar()),
           b.loadQubit(reg, i));
      auto next = arith::MulFOp::create(b, angle, half);
      scf::YieldOp::create(b, ValueRange{next});
    }

    if (sign > 0.0) {
      b.h(b.loadQubit(reg, i));
    }
  });
}
} // namespace

SmallVector<Value> qftAdderQuantum(qc::QCProgramBuilder& b, const uint64_t n) {
  const auto size = static_cast<int64_t>(n) / 2;
  auto x = b.allocQubitRegister(size, "x");
  auto y = b.allocQubitRegister(size, "y");
  auto c = b.allocClassicalBitRegister(size, "c");

  b.scfFor(0, size, 1, [&](Value iv) { b.reset(b.loadQubit(x.value, iv)); });
  b.scfFor(0, size, 1, [&](Value iv) { b.reset(b.loadQubit(y.value, iv)); });
  b.scfFor(0, size, 1, [&](Value iv) { b.h(b.loadQubit(x.value, iv)); });

  // The Draper adder writes the sum into the Fourier basis of `y`. The phase
  // that qubit `x[j - d]` contributes to `y[j]` halves as the distance grows.
  transform(b, y.value, size, 1.0);

  auto one = b.indexConstant(1);
  b.scfFor(0, size, 1, [&](Value j) {
    auto count = arith::AddIOp::create(b, j, one);
    auto lower = b.indexConstant(0);
    auto start = b.floatConstant(std::numbers::pi);
    auto half = b.floatConstant(0.5);

    auto loop = scf::ForOp::create(b, lower, count, one, ValueRange{start});
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loop.getBody());
    auto angle = loop.getRegionIterArg(0);
    auto source = arith::SubIOp::create(b, j, loop.getInductionVar());
    b.cp(angle, b.loadQubit(x.value, source), b.loadQubit(y.value, j));
    auto next = arith::MulFOp::create(b, angle, half);
    scf::YieldOp::create(b, ValueRange{next});
  });

  transform(b, y.value, size, -1.0);

  b.scfFor(0, size, 1,
           [&](Value iv) { b.measure(b.loadQubit(y.value, iv), c, iv); });

  return {c};
}

} // namespace mqt::benchmark
