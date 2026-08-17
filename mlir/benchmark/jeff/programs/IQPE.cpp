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
constexpr double IQPE_PHASE = 3.0 * M_PI / 8.0;
} // namespace

llvm::SmallVector<Value> iqpe(QCProgramBuilder& b, const uint64_t n) {
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
  auto lower = mlir::arith::ConstantIndexOp::create(b, 0);
  auto upper = mlir::arith::ConstantIndexOp::create(b, precision);
  auto one = mlir::arith::ConstantIndexOp::create(b, 1);
  auto last = mlir::arith::ConstantIndexOp::create(b, precision - 1);
  const Value first = mlir::arith::ConstantFloatOp::create(
      b, b.getF64Type(),
      llvm::APFloat(std::pow(2.0, static_cast<double>(precision - 1)) *
                    IQPE_PHASE));
  const Value half = mlir::arith::ConstantFloatOp::create(b, b.getF64Type(),
                                                          llvm::APFloat(0.5));

  auto outer =
      mlir::scf::ForOp::create(b, lower, upper, one, mlir::ValueRange{first});
  const mlir::OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(outer.getBody());
  auto power = outer.getRegionIterArg(0);
  auto index = mlir::arith::SubIOp::create(b, last, outer.getInductionVar());

  b.h(q);
  b.cp(power, q, anc);

  // Correct against the bits that were already measured.
  {
    auto start = mlir::arith::ConstantFloatOp::create(
        b, b.getF64Type(), llvm::APFloat(M_PI / 2.0));
    auto innerLower = mlir::arith::AddIOp::create(b, index, one);
    auto inner = mlir::scf::ForOp::create(b, innerLower, upper, one,
                                          mlir::ValueRange{start});
    const mlir::OpBuilder::InsertionGuard innerGuard(b);
    b.setInsertionPointToStart(inner.getBody());
    auto angle = inner.getRegionIterArg(0);
    b.scfIf(res, inner.getInductionVar(), [&] { b.p(angle, q); });
    const Value next = mlir::arith::MulFOp::create(b, angle, half);
    mlir::scf::YieldOp::create(b, mlir::ValueRange{next});
  }

  b.h(q);
  b.measure(q, res, index);
  b.reset(q);
  const Value next = mlir::arith::MulFOp::create(b, power, half);
  mlir::scf::YieldOp::create(b, mlir::ValueRange{next});

  return {res};
}

} // namespace mqt::jeff::benchmarks
