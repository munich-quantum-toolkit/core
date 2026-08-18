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

#include <llvm/ADT/APFloat.h>
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
/// The constant that the adder adds to the register.
constexpr int64_t ADDEND = 5;
} // namespace

SmallVector<Value> qftAdderClassical(qc::QCProgramBuilder& b,
                                     const uint64_t n) {
  const auto size = static_cast<int64_t>(n);
  auto q = b.allocQubitRegister(size, "q");
  auto c = b.allocClassicalBitRegister(size, "c");

  b.scfFor(0, size, 1, [&](Value iv) { b.reset(b.loadQubit(q.value, iv)); });
  b.scfFor(0, size, 1, [&](Value iv) { b.h(b.loadQubit(q.value, iv)); });

  auto one = arith::ConstantIndexOp::create(b, 1);
  auto last = arith::ConstantIndexOp::create(b, size - 1);

  // Adding a classical constant in the Fourier basis needs no controls. Each
  // qubit takes one phase that depends only on the constant, so the whole
  // addition is a single layer of phase gates between the two transforms.
  b.scfFor(0, size, 1, [&](Value i) {
    auto lower = arith::AddIOp::create(b, i, one);
    auto upper = arith::ConstantIndexOp::create(b, size);
    auto start = arith::ConstantFloatOp::create(
        b, b.getF64Type(), llvm::APFloat(std::numbers::pi / 2.0));
    auto half =
        arith::ConstantFloatOp::create(b, b.getF64Type(), llvm::APFloat(0.5));

    auto loop = scf::ForOp::create(b, lower, upper, one, ValueRange{start});
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(loop.getBody());
      auto angle = loop.getRegionIterArg(0);
      b.cp(angle, b.loadQubit(q.value, loop.getInductionVar()),
           b.loadQubit(q.value, i));
      auto next = arith::MulFOp::create(b, angle, half);
      scf::YieldOp::create(b, ValueRange{next});
    }
    b.h(b.loadQubit(q.value, i));
  });

  for (int64_t i = 0; i < size; ++i) {
    const auto angle = std::numbers::pi * static_cast<double>(ADDEND) /
                       static_cast<double>(int64_t{1} << i);
    b.p(angle, q[size - 1 - i]);
  }

  b.scfFor(0, size, 1, [&](Value step) {
    auto i = arith::SubIOp::create(b, last, step);
    b.h(b.loadQubit(q.value, i));

    auto lower = arith::AddIOp::create(b, i, one);
    auto upper = arith::ConstantIndexOp::create(b, size);
    auto start = arith::ConstantFloatOp::create(
        b, b.getF64Type(), llvm::APFloat(-std::numbers::pi / 2.0));
    auto half =
        arith::ConstantFloatOp::create(b, b.getF64Type(), llvm::APFloat(0.5));

    auto loop = scf::ForOp::create(b, lower, upper, one, ValueRange{start});
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loop.getBody());
    auto angle = loop.getRegionIterArg(0);
    b.cp(angle, b.loadQubit(q.value, loop.getInductionVar()),
         b.loadQubit(q.value, i));
    auto next = arith::MulFOp::create(b, angle, half);
    scf::YieldOp::create(b, ValueRange{next});
  });

  b.scfFor(0, size, 1,
           [&](Value iv) { b.measure(b.loadQubit(q.value, iv), c, iv); });

  return {c};
}

} // namespace mqt::benchmark
