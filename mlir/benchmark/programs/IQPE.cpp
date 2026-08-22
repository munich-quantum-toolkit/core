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
#include "benchmarks/QPE.hpp"
#include "mlir/Benchmark/Programs.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <cstdint>
#include <numbers>

namespace mqt::benchmark {

using namespace mlir;

SmallVector<Value> iterativeQPE(qc::QCProgramBuilder& b,
                                const benchmarks::QPE& benchmark) {
  const auto precision = static_cast<int64_t>(benchmark.options().precision);
  auto q = b.allocQubit();
  auto anc = b.allocQubit();
  auto result = b.allocClassicalBitRegister(precision, benchmark.output().name);

  b.reset(q);
  b.reset(anc);
  b.x(anc);

  /// Descending powers produce the requested phase bits least-significant
  /// first.
  auto lower = b.indexConstant(0);
  auto upper = b.indexConstant(precision);
  auto one = b.indexConstant(1);
  auto last = b.indexConstant(precision - 1);
  auto angles = controlledPhaseAngles(b, benchmark);

  b.scfFor(lower, upper, 1, [&](Value step) {
    auto power = arith::SubIOp::create(b, last, step);
    auto angle =
        tensor::ExtractOp::create(b, angles, ValueRange{power}).getResult();

    b.h(q);
    b.cp(angle, q, anc);

    /// Remove the phase that the bits measured so far contribute.
    auto previous = arith::SubIOp::create(b, step, one);
    phaseRotationLoop(b, lower, step, -std::numbers::pi / 2.0, 0.5,
                      [&](Value angle, Value distance) {
                        auto bit = arith::SubIOp::create(b, previous, distance);
                        b.scfIf(result, bit, [&] { b.p(angle, q); });
                      });

    b.h(q);
    b.measure(q, result, step);
    b.reset(q);
  });

  return {result};
}

SmallVector<Value> iqpe(qc::QCProgramBuilder& b, const uint64_t n) {
  return qpe(b, benchmarks::QPE({.precision = static_cast<size_t>(n),
                                 .phase = benchmarks::Phase(3, 16),
                                 .method = benchmarks::QPEMethod::Iterative}));
}

} // namespace mqt::benchmark
