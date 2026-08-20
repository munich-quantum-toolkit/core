/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Benchmark/BenchmarkUtils.h"
#include "mlir/Benchmark/Programs.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Value.h>
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
  // rotation whose angle halves on every step.
  auto lower = b.indexConstant(0);
  auto upper = b.indexConstant(precision);
  auto one = b.indexConstant(1);
  auto last = b.indexConstant(precision - 1);
  const auto first =
      std::pow(2.0, static_cast<double>(precision - 1)) * IQPE_PHASE;

  scfForWithAngle(b, lower, upper, first, 0.5, [&](Value power, Value step) {
    auto index = arith::SubIOp::create(b, last, step);

    b.h(q);
    b.cp(power, q, anc);

    // Remove the phase that the bits measured so far contribute.
    auto innerLower = arith::AddIOp::create(b, index, one);
    scfForWithAngle(b, innerLower, upper, -std::numbers::pi / 2.0, 0.5,
                    [&](Value angle, Value bit) {
                      b.scfIf(res, bit, [&] { b.p(angle, q); });
                    });

    b.h(q);
    b.measure(q, res, index);
    b.reset(q);
  });

  return {res};
}

} // namespace mqt::benchmark
