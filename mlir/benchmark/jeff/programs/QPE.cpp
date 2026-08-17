/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "Programs.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <llvm/ADT/APFloat.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/ValueRange.h>

#include <cmath>
#include <cstdint>

namespace mqt::jeff::benchmarks {

using mlir::Value;
using mlir::qc::QCProgramBuilder;

namespace {
/// The phase of the unitary whose eigenvalue is estimated.
constexpr double QPE_PHASE = 3.0 * M_PI / 8.0;
} // namespace

llvm::SmallVector<Value> qpe(QCProgramBuilder& b, const uint64_t n) {
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
    auto lower = mlir::arith::ConstantIndexOp::create(b, 0);
    auto upper = mlir::arith::ConstantIndexOp::create(b, counting);
    auto step = mlir::arith::ConstantIndexOp::create(b, 1);
    const Value start = mlir::arith::ConstantFloatOp::create(
        b, b.getF64Type(), llvm::APFloat(QPE_PHASE));
    const Value two = mlir::arith::ConstantFloatOp::create(b, b.getF64Type(),
                                                           llvm::APFloat(2.0));

    auto loop = mlir::scf::ForOp::create(b, lower, upper, step,
                                         mlir::ValueRange{start});
    const mlir::OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loop.getBody());
    auto angle = loop.getRegionIterArg(0);
    b.cp(angle, b.loadQubit(q.value, loop.getInductionVar()), anc);
    const Value next = mlir::arith::MulFOp::create(b, angle, two);
    mlir::scf::YieldOp::create(b, mlir::ValueRange{next});
  }

  // Inverse quantum Fourier transform on the counting register.
  b.scfFor(0, counting / 2, 1, [&](Value i) {
    auto last = mlir::arith::ConstantIndexOp::create(b, counting - 1);
    auto mirrored = mlir::arith::SubIOp::create(b, last, i);
    b.swap(b.loadQubit(q.value, i), b.loadQubit(q.value, mirrored));
  });

  b.scfFor(0, counting, 1, [&](Value i) {
    auto one = mlir::arith::ConstantIndexOp::create(b, 1);
    auto lower = mlir::arith::AddIOp::create(b, i, one);
    auto upper = mlir::arith::ConstantIndexOp::create(b, counting);
    const Value start = mlir::arith::ConstantFloatOp::create(
        b, b.getF64Type(), llvm::APFloat(-M_PI / 2.0));
    const Value half = mlir::arith::ConstantFloatOp::create(b, b.getF64Type(),
                                                            llvm::APFloat(0.5));

    {
      auto loop = mlir::scf::ForOp::create(b, lower, upper, one,
                                           mlir::ValueRange{start});
      const mlir::OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(loop.getBody());
      auto angle = loop.getRegionIterArg(0);
      b.cp(angle, b.loadQubit(q.value, loop.getInductionVar()),
           b.loadQubit(q.value, i));
      const Value next = mlir::arith::MulFOp::create(b, angle, half);
      mlir::scf::YieldOp::create(b, mlir::ValueRange{next});
    }
    b.h(b.loadQubit(q.value, i));
  });

  b.scfFor(0, counting, 1,
           [&](Value iv) { b.measure(b.loadQubit(q.value, iv), c, iv); });

  return {c};
}

} // namespace mqt::jeff::benchmarks
