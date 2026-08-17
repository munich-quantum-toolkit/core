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

namespace mqt::benchmarks {

using namespace mlir;

namespace {
/// The phase of the unitary whose eigenvalue is estimated.
constexpr double QPE_PHASE = 3.0 * std::numbers::pi / 8.0;
} // namespace

SmallVector<Value> qpe(qc::QCProgramBuilder& b, const uint64_t n) {
  const auto counting = static_cast<int64_t>(n) - 1;
  auto q = b.allocQubitRegister(counting, "q");
  auto anc = b.allocQubit();
  auto c = b.allocClassicalBitRegister(counting, "c");

  b.scfFor(0, counting, 1,
           [&](Value iv) { b.reset(b.loadQubit(q.value, iv)); });
  b.reset(anc);

  b.scfFor(0, counting, 1, [&](Value iv) { b.h(b.loadQubit(q.value, iv)); });
  b.x(anc);

  // Repeating a phase gate 2^i times multiplies its angle by 2^i, so each
  // controlled power is one rotation whose angle doubles. The loop carries the
  // angle because `QCProgramBuilder::scfFor` takes no loop-carried values.
  {
    auto lower = arith::ConstantIndexOp::create(b, 0);
    auto upper = arith::ConstantIndexOp::create(b, counting);
    auto step = arith::ConstantIndexOp::create(b, 1);
    const Value start = arith::ConstantFloatOp::create(
        b, b.getF64Type(), llvm::APFloat(QPE_PHASE));
    const Value two =
        arith::ConstantFloatOp::create(b, b.getF64Type(), llvm::APFloat(2.0));

    auto loop = scf::ForOp::create(b, lower, upper, step, ValueRange{start});
    const OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loop.getBody());
    auto angle = loop.getRegionIterArg(0);
    b.cp(angle, b.loadQubit(q.value, loop.getInductionVar()), anc);
    const Value next = arith::MulFOp::create(b, angle, two);
    scf::YieldOp::create(b, ValueRange{next});
  }

  // Inverse quantum Fourier transform on the counting register.
  b.scfFor(0, counting / 2, 1, [&](Value i) {
    auto last = arith::ConstantIndexOp::create(b, counting - 1);
    auto mirrored = arith::SubIOp::create(b, last, i);
    b.swap(b.loadQubit(q.value, i), b.loadQubit(q.value, mirrored));
  });

  b.scfFor(0, counting, 1, [&](Value i) {
    auto one = arith::ConstantIndexOp::create(b, 1);
    auto lower = arith::AddIOp::create(b, i, one);
    auto upper = arith::ConstantIndexOp::create(b, counting);
    const Value start = arith::ConstantFloatOp::create(
        b, b.getF64Type(), llvm::APFloat(-std::numbers::pi / 2.0));
    const Value half =
        arith::ConstantFloatOp::create(b, b.getF64Type(), llvm::APFloat(0.5));

    {
      auto loop = scf::ForOp::create(b, lower, upper, one, ValueRange{start});
      const OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(loop.getBody());
      auto angle = loop.getRegionIterArg(0);
      b.cp(angle, b.loadQubit(q.value, loop.getInductionVar()),
           b.loadQubit(q.value, i));
      const Value next = arith::MulFOp::create(b, angle, half);
      scf::YieldOp::create(b, ValueRange{next});
    }
    b.h(b.loadQubit(q.value, i));
  });

  b.scfFor(0, counting, 1,
           [&](Value iv) { b.measure(b.loadQubit(q.value, iv), c, iv); });

  return {c};
}

} // namespace mqt::benchmarks
