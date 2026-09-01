/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/QPE.hpp"

#include "Programs.h"
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

static void
qpePhaseRotationLoop(qc::QCProgramBuilder& builder, Value lower, Value upper,
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

[[nodiscard]] static Value controlledPhaseAngles(qc::QCProgramBuilder& builder,
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

[[nodiscard]] static SmallVector<Value>
iterativeQPE(qc::QCProgramBuilder& builder, const QPE& benchmark) {
  const auto precision = static_cast<int64_t>(benchmark.options().precision);
  auto query = builder.allocQubit();
  auto ancilla = builder.allocQubit();
  auto result =
      builder.allocClassicalBitRegister(precision, benchmark.output().name);
  builder.x(ancilla);

  auto lower = builder.indexConstant(0);
  auto upper = builder.indexConstant(precision);
  auto one = builder.indexConstant(1);
  auto last = builder.indexConstant(precision - 1);
  auto angles = controlledPhaseAngles(builder, benchmark);

  builder.scfFor(lower, upper, 1, [&](Value step) {
    auto power = arith::SubIOp::create(builder, last, step);
    auto angle = tensor::ExtractOp::create(builder, angles, ValueRange{power})
                     .getResult();
    builder.h(query);
    builder.cp(angle, query, ancilla);

    auto previous = arith::SubIOp::create(builder, step, one);
    qpePhaseRotationLoop(
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

[[nodiscard]] static SmallVector<Value>
standardQPE(qc::QCProgramBuilder& builder, const QPE& benchmark) {
  const auto precision = static_cast<int64_t>(benchmark.options().precision);
  auto query = builder.allocQubitRegisterStorage(precision, "query");
  auto ancilla = builder.allocQubit();
  auto result =
      builder.allocClassicalBitRegister(precision, benchmark.output().name);
  builder.scfFor(0, precision, 1, [&](Value index) {
    builder.h(builder.loadQubit(query, index));
  });
  builder.x(ancilla);

  auto zero = builder.indexConstant(0);
  auto upper = builder.indexConstant(precision);
  auto one = builder.indexConstant(1);
  auto last = builder.indexConstant(precision - 1);
  auto angles = controlledPhaseAngles(builder, benchmark);
  builder.scfFor(zero, upper, 1, [&](Value index) {
    auto angle = tensor::ExtractOp::create(builder, angles, ValueRange{index})
                     .getResult();
    auto control = arith::SubIOp::create(builder, last, index);
    builder.cp(angle, builder.loadQubit(query, control), ancilla);
  });

  builder.scfFor(zero, upper, 1, [&](Value step) {
    auto previous = arith::SubIOp::create(builder, step, one);
    qpePhaseRotationLoop(builder, zero, step, -std::numbers::pi / 2.0, 0.5,
                         [&](Value angle, Value distance) {
                           auto control = arith::SubIOp::create(
                               builder, previous, distance);
                           builder.cp(angle, builder.loadQubit(query, control),
                                      builder.loadQubit(query, step));
                         });
    builder.h(builder.loadQubit(query, step));
  });
  builder.measureQubitRegister(query, result, precision);
  return {result};
}

SmallVector<Value> qpe(qc::QCProgramBuilder& builder, const QPE& benchmark) {
  return benchmark.options().method == QPEMethod::Standard
             ? standardQPE(builder, benchmark)
             : iterativeQPE(builder, benchmark);
}

} // namespace mqt::bench
