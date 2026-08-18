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

#include <cmath>
#include <cstdint>
#include <numbers>

namespace mqt::benchmark {

using namespace mlir;

namespace {
/// The phase of the unitary whose eigenvalue is estimated.
constexpr double IQPE_PHASE = 3.0 * std::numbers::pi / 8.0;
} // namespace

SmallVector<Value> iqpe(qc::QCProgramBuilder& b, const uint64_t n) {
  const auto precision = static_cast<int64_t>(n);
  auto q = b.allocQubit();
  auto anc = b.allocQubit();
  auto res = b.allocClassicalBitRegister(precision, "res");

  b.reset(q);
  b.reset(anc);
  b.x(anc);

  // The bits are produced from the most significant one downwards, so the loop
  // runs over the distance from the last bit. Each controlled power is one
  // rotation whose angle halves on every step. Both loops carry their angle
  // because `QCProgramBuilder::scfFor` takes no loop-carried values.
  auto lower = arith::ConstantIndexOp::create(b, 0);
  auto upper = arith::ConstantIndexOp::create(b, precision);
  auto one = arith::ConstantIndexOp::create(b, 1);
  auto last = arith::ConstantIndexOp::create(b, precision - 1);
  const Value first = arith::ConstantFloatOp::create(
      b, b.getF64Type(),
      llvm::APFloat(std::pow(2.0, static_cast<double>(precision - 1)) *
                    IQPE_PHASE));
  const Value half =
      arith::ConstantFloatOp::create(b, b.getF64Type(), llvm::APFloat(0.5));

  auto outer = scf::ForOp::create(b, lower, upper, one, ValueRange{first});
  const OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(outer.getBody());
  auto power = outer.getRegionIterArg(0);
  auto index = arith::SubIOp::create(b, last, outer.getInductionVar());

  b.h(q);
  b.cp(power, q, anc);

  // Correct against the bits that were already measured.
  {
    auto start = arith::ConstantFloatOp::create(
        b, b.getF64Type(), llvm::APFloat(std::numbers::pi / 2.0));
    auto innerLower = arith::AddIOp::create(b, index, one);
    auto inner =
        scf::ForOp::create(b, innerLower, upper, one, ValueRange{start});
    const OpBuilder::InsertionGuard innerGuard(b);
    b.setInsertionPointToStart(inner.getBody());
    auto angle = inner.getRegionIterArg(0);
    b.scfIf(res, inner.getInductionVar(), [&] { b.p(angle, q); });
    const Value next = arith::MulFOp::create(b, angle, half);
    scf::YieldOp::create(b, ValueRange{next});
  }

  b.h(q);
  b.measure(q, res, index);
  b.reset(q);
  const Value next = arith::MulFOp::create(b, power, half);
  scf::YieldOp::create(b, ValueRange{next});

  return {res};
}

} // namespace mqt::benchmark
