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
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <numbers>

namespace mqt::bench {

using namespace mlir;

static void
qftPhaseRotationLoop(qc::QCProgramBuilder& builder, Value lower, Value upper,
                     const double start, const double factor,
                     const function_ref<void(Value angle, Value index)>& body) {
  auto one = builder.indexConstant(1);
  auto first = builder.floatConstant(start);
  auto scale = builder.floatConstant(factor);
  auto loop = scf::ForOp::create(builder, lower, upper, one, ValueRange{first});
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(loop.getBody());
  auto angle = loop.getRegionIterArg(0);
  body(angle, loop.getInductionVar());
  auto next = arith::MulFOp::create(builder, angle, scale).getResult();
  scf::YieldOp::create(builder, ValueRange{next});
}

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
  auto zero = builder.indexConstant(0);
  auto one = builder.indexConstant(1);
  auto last = builder.indexConstant(qubits - 1);
  builder.scfFor(0, qubits, 1, [&](Value step) {
    auto target = arith::SubIOp::create(builder, last, step);
    builder.h(builder.loadQubit(query, target));
    auto previous = arith::SubIOp::create(builder, target, one);
    qftPhaseRotationLoop(builder, zero, target, std::numbers::pi / 2.0, 0.5,
                         [&](Value angle, Value distance) {
                           auto control = arith::SubIOp::create(
                               builder, previous, distance);
                           builder.cp(angle, builder.loadQubit(query, control),
                                      builder.loadQubit(query, target));
                         });
  });
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

  const auto round = [&](Value step, const bool preparePlus) {
    if (preparePlus) {
      builder.h(query);
    }
    auto previous = arith::SubIOp::create(builder, step, one);
    qftPhaseRotationLoop(
        builder, zero, step, std::numbers::pi / 2.0, 0.5,
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
