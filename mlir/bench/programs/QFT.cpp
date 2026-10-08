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

#include "mqt/Dialect/QC/Builder/QCProgramBuilder.h"

#include "Programs.h"
#include "QFTUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

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
  auto initialCorrection = builder.floatConstant(0.);
  auto firstAngle = builder.floatConstant(std::numbers::pi / 2.);
  auto half = builder.floatConstant(0.5);

  const auto rounds = [&](Value lower, Value upper, Value correction,
                          const bool preparePlus) {
    auto loop =
        scf::ForOp::create(builder, lower, upper, one, ValueRange{correction});
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    if (preparePlus) {
      builder.h(query);
    }
    builder.p(loop.getRegionIterArg(0), query);
    builder.h(query);
    auto bit = builder.measure(query, result, loop.getInductionVar());
    builder.reset(query);
    auto next = detail::advancePhaseCorrection(
        builder, loop.getRegionIterArg(0), bit, half, firstAngle);
    scf::YieldOp::create(builder, ValueRange{next});
    return loop.getResult(0);
  };

  auto correction = rounds(zero, active, initialCorrection, true);
  rounds(active, total, correction, false);
  return {result};
}

SmallVector<Value> qft(qc::QCProgramBuilder& builder, const QFT& benchmark) {
  return benchmark.options().method == QFTMethod::Standard
             ? standardQFT(builder, benchmark)
             : semiclassicalQFT(builder, benchmark);
}

} // namespace mqt::bench
