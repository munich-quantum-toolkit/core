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

#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <numbers>
#include <variant>

namespace mqt::benchmark {

using namespace mlir;

void resetRegister(qc::QCProgramBuilder& b, Value reg, const int64_t size) {
  b.scfFor(0, size, 1, [&](Value i) { b.reset(b.loadQubit(reg, i)); });
}

void measureRegister(qc::QCProgramBuilder& b, Value reg, const int64_t size,
                     Value bits) {
  b.scfFor(0, size, 1,
           [&](Value i) { b.measure(b.loadQubit(reg, i), bits, i); });
}

void scfForWithAngle(qc::QCProgramBuilder& b, Value lower, Value upper,
                     const std::variant<double, Value>& start,
                     const double factor,
                     const function_ref<void(Value, Value)>& body) {
  auto one = b.indexConstant(1);
  auto first = std::holds_alternative<Value>(start)
                   ? std::get<Value>(start)
                   : b.floatConstant(std::get<double>(start));
  auto scale = b.floatConstant(factor);

  auto loop = scf::ForOp::create(b, lower, upper, one, ValueRange{first});
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(loop.getBody());
  auto angle = loop.getRegionIterArg(0);
  body(angle, loop.getInductionVar());
  auto next = arith::MulFOp::create(b, angle, scale);
  scf::YieldOp::create(b, ValueRange{next});
}

void fourierTransform(qc::QCProgramBuilder& b, Value reg, const int64_t size,
                      const double sign) {
  auto one = b.indexConstant(1);
  auto last = b.indexConstant(size - 1);

  b.scfFor(0, size, 1, [&](Value step) {
    // The inverse runs the same rotations in the opposite order, so it walks
    // the register backwards and takes its Hadamard after the rotations.
    auto i = sign > 0.0 ? step : Value{arith::SubIOp::create(b, last, step)};
    if (sign > 0.0) {
      b.h(b.loadQubit(reg, i));
    }

    auto lower = arith::AddIOp::create(b, i, one);
    auto upper = b.indexConstant(size);
    scfForWithAngle(b, lower, upper, sign * std::numbers::pi / 2.0, 0.5,
                    [&](Value angle, Value j) {
                      b.cp(angle, b.loadQubit(reg, j), b.loadQubit(reg, i));
                    });

    if (sign < 0.0) {
      b.h(b.loadQubit(reg, i));
    }
  });
}

} // namespace mqt::benchmark
