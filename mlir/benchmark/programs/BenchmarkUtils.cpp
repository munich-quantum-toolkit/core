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

#include "mlir/Dialect/MQT/Utils/Parameters.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

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
#include <variant>
#include <vector>

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

Value controlledPhaseAngles(qc::QCProgramBuilder& b,
                            const benchmarks::QPE& benchmark) {
  const auto& options = benchmark.options();
  const auto precision = options.precision;
  const auto denominator = options.phase.denominator();
  auto remainder = options.phase.numerator();

  std::vector<double> angles;
  angles.reserve(precision);
  for (size_t i = 0; i < precision; ++i) {
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

  const auto type =
      RankedTensorType::get({static_cast<int64_t>(precision)}, b.getF64Type());
  const auto value =
      DenseElementsAttr::get(type, llvm::ArrayRef<double>(angles));
  return arith::ConstantOp::create(b, value).getResult();
}

/// Runs @p body over [@p lower, @p upper) with an angle that @p advance carries
/// from one step to the next.
static void angleLoop(qc::QCProgramBuilder& b, Value lower, Value upper,
                      Value first, const function_ref<Value(Value)>& advance,
                      const function_ref<void(Value, Value)>& body) {
  auto one = b.indexConstant(1);

  auto loop = scf::ForOp::create(b, lower, upper, one, ValueRange{first});
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(loop.getBody());
  auto angle = loop.getRegionIterArg(0);
  body(angle, loop.getInductionVar());
  scf::YieldOp::create(b, ValueRange{advance(angle)});
}

void phaseRotationLoop(qc::QCProgramBuilder& b, Value lower, Value upper,
                       const std::variant<double, Value>& start,
                       const double factor,
                       const function_ref<void(Value, Value)>& body) {
  auto first = mlir::mqt::variantToValue(b, b.getLoc(), start);
  auto scale = b.floatConstant(factor);

  const auto advance = [&](Value angle) {
    return arith::MulFOp::create(b, angle, scale).getResult();
  };
  angleLoop(b, lower, upper, first, advance, body);
}

void uniformRotationLoop(qc::QCProgramBuilder& b, Value lower, Value upper,
                         const std::variant<double, Value>& start,
                         const double increment,
                         const function_ref<void(Value, Value)>& body) {
  auto first = mlir::mqt::variantToValue(b, b.getLoc(), start);
  auto step = b.floatConstant(increment);

  const auto advance = [&](Value angle) {
    return arith::AddFOp::create(b, angle, step).getResult();
  };
  angleLoop(b, lower, upper, first, advance, body);
}

} // namespace mqt::benchmark
