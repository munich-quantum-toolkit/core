/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "BenchmarkUtils.h"
#include "mlir/Benchmark/Programs.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <numbers>

namespace mqt::benchmark {

using namespace mlir;

SmallVector<Value> iqft(qc::QCProgramBuilder& b, const uint64_t n) {
  const auto bits = static_cast<int64_t>(n);
  auto q = b.allocQubit();
  auto res = b.allocClassicalBitRegister(bits, "res");

  b.reset(q);

  // One qubit is measured and reset once per result bit. The correction angle
  // depends on the distance to an earlier bit. Walking the earlier bits from
  // the closest one outwards halves the angle on every step, which avoids
  // raising two to a loop-dependent power.
  auto zero = b.indexConstant(0);
  auto total = b.indexConstant(bits);
  auto last = b.indexConstant(bits - 1);

  b.scfFor(0, bits, 1, [&](Value i) {
    auto offset = arith::SubIOp::create(b, total, i);

    phaseRotationLoop(b, zero, i, std::numbers::pi / 2.0, 0.5,
                      [&](Value angle, Value step) {
                        auto index = arith::AddIOp::create(b, offset, step);
                        b.scfIf(res, index, [&] { b.p(angle, q); });
                      });

    b.h(q);
    auto target = arith::SubIOp::create(b, last, i);
    b.measure(q, res, target);
    b.reset(q);
  });

  return {res};
}

} // namespace mqt::benchmark
