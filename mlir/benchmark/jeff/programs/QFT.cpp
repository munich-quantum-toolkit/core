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
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <numbers>

namespace mqt::jeff::benchmarks {

using namespace mlir;

SmallVector<Value> qft(qc::QCProgramBuilder& b, const uint64_t n) {
  const auto size = static_cast<int64_t>(n);
  auto q = b.allocQubitRegister(size, "q");
  auto c = b.allocClassicalBitRegister(size, "c");

  b.scfFor(0, size, 1, [&](Value iv) { b.reset(b.loadQubit(q.value, iv)); });

  // Every qubit takes a Hadamard and then a controlled phase from each qubit
  // above it. The angle halves as the distance grows, so the inner loop carries
  // it. `QCProgramBuilder::scfFor` takes no loop-carried values, so the inner
  // loop is built directly.
  b.scfFor(0, size, 1, [&](Value i) {
    b.h(b.loadQubit(q.value, i));

    auto one = arith::ConstantIndexOp::create(b, 1);
    auto lower = arith::AddIOp::create(b, i, one);
    auto upper = arith::ConstantIndexOp::create(b, size);
    const Value start = arith::ConstantFloatOp::create(
        b, b.getF64Type(), llvm::APFloat(std::numbers::pi / 2.0));
    const Value half =
        arith::ConstantFloatOp::create(b, b.getF64Type(), llvm::APFloat(0.5));

    auto loop = scf::ForOp::create(b, lower, upper, one, ValueRange{start});
    const OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loop.getBody());
    auto angle = loop.getRegionIterArg(0);
    b.cp(angle, b.loadQubit(q.value, loop.getInductionVar()),
         b.loadQubit(q.value, i));
    const Value next = arith::MulFOp::create(b, angle, half);
    scf::YieldOp::create(b, ValueRange{next});
  });

  // Reverse the bit order. The count is floor(size / 2).
  b.scfFor(0, size / 2, 1, [&](Value i) {
    auto last = arith::ConstantIndexOp::create(b, size - 1);
    auto mirrored = arith::SubIOp::create(b, last, i);
    b.swap(b.loadQubit(q.value, i), b.loadQubit(q.value, mirrored));
  });

  b.scfFor(0, size, 1,
           [&](Value iv) { b.measure(b.loadQubit(q.value, iv), c, iv); });

  return {c};
}

} // namespace mqt::jeff::benchmarks
