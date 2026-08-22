/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "benchmarks/QPE.hpp"

#include "BenchmarkUtils.h"
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

SmallVector<Value> qpe(qc::QCProgramBuilder& b,
                       const benchmarks::QPE& benchmark) {
  if (benchmark.options().method == benchmarks::QPEMethod::Iterative) {
    return iterativeQPE(b, benchmark);
  }

  const auto counting = static_cast<int64_t>(benchmark.options().precision);
  auto q = b.allocQubitRegisterStorage(counting, "q");
  auto anc = b.allocQubit();
  auto result = b.allocClassicalBitRegister(counting, benchmark.output().name);

  resetRegister(b, q, counting);
  b.reset(anc);

  b.scfFor(0, counting, 1, [&](Value iv) { b.h(b.loadQubit(q, iv)); });
  b.x(anc);

  auto zero = b.indexConstant(0);
  auto one = b.indexConstant(1);
  auto upper = b.indexConstant(counting);
  auto last = b.indexConstant(counting - 1);
  auto angles = controlledPhaseAngles(b, benchmark);

  b.scfFor(zero, upper, 1, [&](Value i) {
    auto angle =
        tensor::ExtractOp::create(b, angles, ValueRange{i}).getResult();
    auto control = arith::SubIOp::create(b, last, i);
    b.cp(angle, b.loadQubit(q, control), anc);
  });

  /// Inverse quantum Fourier transform on the counting register.
  b.scfFor(0, counting / 2, 1, [&](Value i) {
    auto mirrored = arith::SubIOp::create(b, last, i);
    b.swap(b.loadQubit(q, i), b.loadQubit(q, mirrored));
  });

  /// The transform runs the forward circuit backwards.
  b.scfFor(0, counting, 1, [&](Value step) {
    auto i = arith::SubIOp::create(b, last, step);
    auto lower = arith::AddIOp::create(b, i, one);
    phaseRotationLoop(b, lower, upper, -std::numbers::pi / 2.0, 0.5,
                      [&](Value angle, Value j) {
                        b.cp(angle, b.loadQubit(q, j), b.loadQubit(q, i));
                      });
    b.h(b.loadQubit(q, i));
  });

  b.scfFor(0, counting, 1, [&](Value i) {
    auto resultIndex = arith::SubIOp::create(b, last, i);
    b.measure(b.loadQubit(q, i), result, resultIndex);
  });

  return {result};
}

SmallVector<Value> qpe(qc::QCProgramBuilder& b, const uint64_t n) {
  return qpe(b, benchmarks::QPE({.precision = static_cast<size_t>(n - 1),
                                 .phase = benchmarks::Phase(3, 16)}));
}

} // namespace mqt::benchmark
