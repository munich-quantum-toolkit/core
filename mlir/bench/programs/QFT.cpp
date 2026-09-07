/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/QFT.hpp"

#include "Programs.h"
#include "QFTUtils.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <numbers>

namespace mqt::bench {

using namespace mlir;

[[nodiscard]] static SmallVector<Value>
standardQFT(qc::QCProgramBuilder& builder, const QFT& benchmark) {
  const auto& options = benchmark.options();
  const auto qubits = static_cast<int64_t>(options.qubits);
  const auto period = static_cast<int64_t>(options.periodExponent);
  auto query = builder.allocQubitRegisterStorage(qubits, "query");
  auto result =
      builder.allocClassicalBitRegister(qubits, benchmark.output().name);

  builder.scfFor(period, qubits, 1, [&](Value index) {
    builder.h(builder.loadQubit(query, index));
  });
  detail::forwardQFT(builder, query, qubits);

  auto last = builder.indexConstant(qubits - 1);
  builder.scfFor(0, qubits, 1, [&](Value index) {
    auto resultIndex = arith::SubIOp::create(builder, last, index);
    builder.measure(builder.loadQubit(query, index), result, resultIndex);
  });
  return {result};
}

[[nodiscard]] static SmallVector<Value>
semiclassicalQFT(qc::QCProgramBuilder& builder, const QFT& benchmark) {
  const auto& options = benchmark.options();
  const auto qubits = static_cast<int64_t>(options.qubits);
  const auto period = static_cast<int64_t>(options.periodExponent);
  auto query = builder.allocQubit();
  auto result =
      builder.allocClassicalBitRegister(qubits, benchmark.output().name);
  auto zero = builder.indexConstant(0);
  auto total = builder.indexConstant(qubits);
  auto one = builder.indexConstant(1);
  auto active = builder.indexConstant(qubits - period);
  auto firstAngle = builder.floatConstant(std::numbers::pi / 2.);
  auto half = builder.floatConstant(0.5);

  const auto round = [&](Value step, const bool preparePlus) {
    if (preparePlus) {
      builder.h(query);
    }
    auto previous = arith::SubIOp::create(builder, step, one);
    detail::phaseRotationLoop(
        builder, zero, step, one, firstAngle, half,
        [&](Value angle, Value distance) {
          auto bit = arith::SubIOp::create(builder, previous, distance);
          builder.scfIf(result, bit, [&] { builder.p(angle, query); });
        });
    builder.h(query);
    builder.measure(query, result, step);
    builder.reset(query);
  };

  builder.scfFor(zero, active, 1, [&](Value step) { round(step, true); });
  builder.scfFor(active, total, 1, [&](Value step) { round(step, false); });
  return {result};
}

SmallVector<Value> qft(qc::QCProgramBuilder& builder, const QFT& benchmark) {
  return benchmark.options().method == QFTMethod::Standard
             ? standardQFT(builder, benchmark)
             : semiclassicalQFT(builder, benchmark);
}

} // namespace mqt::bench
