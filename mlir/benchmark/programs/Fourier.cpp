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
#include "bench/QFT.hpp"
#include "bench/QPE.hpp"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <llvm/ADT/ArrayRef.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <cstdint>
#include <numbers>
#include <vector>

namespace mqt::bench {

using namespace mlir;

namespace {

void phaseRotationLoop(
    qc::QCProgramBuilder& builder, Value lower, Value upper, const double start,
    const double factor,
    const function_ref<void(Value angle, Value index)>& body) {
  auto one = arith::ConstantIndexOp::create(builder, 1).getResult();
  auto first =
      arith::ConstantOp::create(builder, builder.getF64FloatAttr(start))
          .getResult();
  auto scale =
      arith::ConstantOp::create(builder, builder.getF64FloatAttr(factor))
          .getResult();
  auto loop = scf::ForOp::create(builder, lower, upper, one, ValueRange{first});
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(loop.getBody());
  auto angle = loop.getRegionIterArg(0);
  body(angle, loop.getInductionVar());
  auto next = arith::MulFOp::create(builder, angle, scale).getResult();
  scf::YieldOp::create(builder, ValueRange{next});
}

[[nodiscard]] Value controlledPhaseAngles(qc::QCProgramBuilder& builder,
                                          const QPE& benchmark) {
  const auto& options = benchmark.options();
  const auto denominator = options.phase.denominator();
  auto remainder = options.phase.numerator();

  std::vector<double> angles;
  angles.reserve(options.precision);
  for (size_t i = 0; i < options.precision; ++i) {
    const auto turns = static_cast<long double>(remainder) /
                       static_cast<long double>(denominator);
    angles.emplace_back(
        static_cast<double>(2.L * std::numbers::pi_v<long double> * turns));
    if (remainder >= denominator - remainder) {
      remainder -= denominator - remainder;
    } else {
      remainder += remainder;
    }
  }

  const auto type = RankedTensorType::get(
      {static_cast<int64_t>(options.precision)}, builder.getF64Type());
  const auto value = DenseElementsAttr::get(type, ArrayRef<double>(angles));
  return arith::ConstantOp::create(builder, value).getResult();
}

SmallVector<Value> iterativeQPE(qc::QCProgramBuilder& builder,
                                const QPE& benchmark) {
  const auto precision = static_cast<int64_t>(benchmark.options().precision);
  auto query = builder.allocQubit();
  auto ancilla = builder.allocQubit();
  auto result =
      builder.allocClassicalBitRegister(precision, benchmark.output().name);
  builder.x(ancilla);

  auto lower = arith::ConstantIndexOp::create(builder, 0).getResult();
  auto upper = arith::ConstantIndexOp::create(builder, precision).getResult();
  auto one = arith::ConstantIndexOp::create(builder, 1).getResult();
  auto last =
      arith::ConstantIndexOp::create(builder, precision - 1).getResult();
  auto angles = controlledPhaseAngles(builder, benchmark);

  builder.scfFor(lower, upper, 1, [&](Value step) {
    auto power = arith::SubIOp::create(builder, last, step);
    auto angle = tensor::ExtractOp::create(builder, angles, ValueRange{power})
                     .getResult();
    builder.h(query);
    builder.cp(angle, query, ancilla);

    auto previous = arith::SubIOp::create(builder, step, one);
    phaseRotationLoop(
        builder, lower, step, -std::numbers::pi / 2.0, 0.5,
        [&](Value correction, Value distance) {
          auto bit = arith::SubIOp::create(builder, previous, distance);
          builder.scfIf(result, bit, [&] { builder.p(correction, query); });
        });

    builder.h(query);
    builder.measure(query, result, step);
    builder.reset(query);
  });
  return {result};
}

SmallVector<Value> standardQPE(qc::QCProgramBuilder& builder,
                               const QPE& benchmark) {
  const auto precision = static_cast<int64_t>(benchmark.options().precision);
  auto query = builder.allocQubitRegisterStorage(precision, "query");
  auto ancilla = builder.allocQubit();
  auto result =
      builder.allocClassicalBitRegister(precision, benchmark.output().name);
  builder.scfFor(0, precision, 1, [&](Value index) {
    builder.h(builder.loadQubit(query, index));
  });
  builder.x(ancilla);

  auto zero = arith::ConstantIndexOp::create(builder, 0).getResult();
  auto upper = arith::ConstantIndexOp::create(builder, precision).getResult();
  auto one = arith::ConstantIndexOp::create(builder, 1).getResult();
  auto last =
      arith::ConstantIndexOp::create(builder, precision - 1).getResult();
  auto angles = controlledPhaseAngles(builder, benchmark);
  builder.scfFor(zero, upper, 1, [&](Value index) {
    auto angle = tensor::ExtractOp::create(builder, angles, ValueRange{index})
                     .getResult();
    auto control = arith::SubIOp::create(builder, last, index);
    builder.cp(angle, builder.loadQubit(query, control), ancilla);
  });

  builder.scfFor(zero, upper, 1, [&](Value step) {
    auto previous = arith::SubIOp::create(builder, step, one);
    phaseRotationLoop(builder, zero, step, -std::numbers::pi / 2.0, 0.5,
                      [&](Value angle, Value distance) {
                        auto control =
                            arith::SubIOp::create(builder, previous, distance);
                        builder.cp(angle, builder.loadQubit(query, control),
                                   builder.loadQubit(query, step));
                      });
    builder.h(builder.loadQubit(query, step));
  });
  builder.measureQubitRegister(query, result, precision);
  return {result};
}

SmallVector<Value> standardQFT(qc::QCProgramBuilder& builder,
                               const QFT& benchmark) {
  const auto& options = benchmark.options();
  const auto qubits = static_cast<int64_t>(options.qubits);
  const auto period = static_cast<int64_t>(options.periodExponent);
  auto query = builder.allocQubitRegisterStorage(qubits, "query");
  auto result =
      builder.allocClassicalBitRegister(qubits, benchmark.output().name);

  builder.scfFor(period, qubits, 1, [&](Value index) {
    builder.h(builder.loadQubit(query, index));
  });
  auto zero = arith::ConstantIndexOp::create(builder, 0).getResult();
  auto one = arith::ConstantIndexOp::create(builder, 1).getResult();
  auto last = arith::ConstantIndexOp::create(builder, qubits - 1).getResult();
  builder.scfFor(0, qubits, 1, [&](Value step) {
    auto target = arith::SubIOp::create(builder, last, step);
    builder.h(builder.loadQubit(query, target));
    auto previous = arith::SubIOp::create(builder, target, one);
    phaseRotationLoop(builder, zero, target, std::numbers::pi / 2.0, 0.5,
                      [&](Value angle, Value distance) {
                        auto control =
                            arith::SubIOp::create(builder, previous, distance);
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

SmallVector<Value> semiclassicalQFT(qc::QCProgramBuilder& builder,
                                    const QFT& benchmark) {
  const auto& options = benchmark.options();
  const auto qubits = static_cast<int64_t>(options.qubits);
  const auto period = static_cast<int64_t>(options.periodExponent);
  auto query = builder.allocQubit();
  auto result =
      builder.allocClassicalBitRegister(qubits, benchmark.output().name);
  auto zero = arith::ConstantIndexOp::create(builder, 0).getResult();
  auto total = arith::ConstantIndexOp::create(builder, qubits).getResult();
  auto one = arith::ConstantIndexOp::create(builder, 1).getResult();
  auto active =
      arith::ConstantIndexOp::create(builder, qubits - period).getResult();

  const auto round = [&](Value step, const bool preparePlus) {
    if (preparePlus) {
      builder.h(query);
    }
    auto previous = arith::SubIOp::create(builder, step, one);
    phaseRotationLoop(
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

} // namespace

SmallVector<Value> qpe(qc::QCProgramBuilder& builder, const QPE& benchmark) {
  return benchmark.options().method == QPEMethod::Standard
             ? standardQPE(builder, benchmark)
             : iterativeQPE(builder, benchmark);
}

SmallVector<Value> qft(qc::QCProgramBuilder& builder, const QFT& benchmark) {
  return benchmark.options().method == QFTMethod::Standard
             ? standardQFT(builder, benchmark)
             : semiclassicalQFT(builder, benchmark);
}

} // namespace mqt::bench
