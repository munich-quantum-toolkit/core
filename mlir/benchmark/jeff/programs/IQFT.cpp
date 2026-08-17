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

llvm::SmallVector<Value> iqft(QCProgramBuilder& b, const uint64_t n) {
  const auto bits = static_cast<int64_t>(n);
  auto q = b.allocQubit();
  auto res = b.allocClassicalBitRegister(bits, "res");

  b.reset(q);

  // One qubit is measured and reset once per result bit. The correction angle
  // depends on the distance to an earlier bit. Walking the earlier bits from
  // the closest one outwards halves the angle on every step, which avoids
  // raising two to a loop-dependent power. The inner loop carries the angle
  // because `QCProgramBuilder::scfFor` takes no loop-carried values.
  b.scfFor(0, bits, 1, [&](Value i) {
    auto total = mlir::arith::ConstantIndexOp::create(b, bits);
    auto one = mlir::arith::ConstantIndexOp::create(b, 1);
    auto lower = mlir::arith::ConstantIndexOp::create(b, 0);
    auto offset = mlir::arith::SubIOp::create(b, total, i);
    const Value start = mlir::arith::ConstantFloatOp::create(
        b, b.getF64Type(), llvm::APFloat(M_PI / 2.0));
    const Value half = mlir::arith::ConstantFloatOp::create(b, b.getF64Type(),
                                                            llvm::APFloat(0.5));

    {
      auto loop =
          mlir::scf::ForOp::create(b, lower, i, one, mlir::ValueRange{start});
      const mlir::OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(loop.getBody());
      auto angle = loop.getRegionIterArg(0);
      auto index =
          mlir::arith::AddIOp::create(b, offset, loop.getInductionVar());
      b.scfIf(res, index, [&] { b.p(angle, q); });
      const Value next = mlir::arith::MulFOp::create(b, angle, half);
      mlir::scf::YieldOp::create(b, mlir::ValueRange{next});
    }

    b.h(q);
    auto last = mlir::arith::ConstantIndexOp::create(b, bits - 1);
    auto target = mlir::arith::SubIOp::create(b, last, i);
    b.measure(q, res, target);
    b.reset(q);
  });

  return {res};
}

} // namespace mqt::jeff::benchmarks
