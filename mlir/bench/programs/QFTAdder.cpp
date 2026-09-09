/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/QFTAdder.hpp"

#include "Programs.h"
#include "QFTUtils.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <llvm/ADT/ArrayRef.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <cstdint>
#include <numbers>
#include <ranges>
#include <string_view>

namespace mqt::bench {

using namespace mlir;

static void addQuantumRegister(qc::QCProgramBuilder& builder, Value addend,
                               Value sum, int64_t qubits, bool carry) {
  auto zero = builder.indexConstant(0);
  auto one = builder.indexConstant(1);
  auto last = builder.indexConstant(qubits - 1);
  auto firstAngle = builder.floatConstant(std::numbers::pi);
  auto half = builder.floatConstant(0.5);
  builder.scfFor(0, qubits, 1, [&](Value step) {
    auto target = arith::SubIOp::create(builder, last, step).getResult();
    auto upper = arith::AddIOp::create(builder, target, one).getResult();
    detail::phaseRotationLoop(
        builder, zero, upper, one, firstAngle, half,
        [&](Value angle, Value distance) {
          auto control =
              arith::SubIOp::create(builder, target, distance).getResult();
          builder.cp(angle, builder.loadQubit(addend, control),
                     builder.loadQubit(sum, target));
        });
  });
  if (carry) {
    auto target = builder.loadQubit(sum, builder.indexConstant(qubits));
    detail::phaseRotationLoop(
        builder, zero, builder.indexConstant(qubits), one,
        builder.floatConstant(std::numbers::pi / 2.), half,
        [&](Value angle, Value distance) {
          auto control = arith::SubIOp::create(builder, last, distance);
          builder.cp(angle, builder.loadQubit(addend, control), target);
        });
  }
}

[[nodiscard]] static Value phaseAngles(qc::QCProgramBuilder& builder,
                                       std::string_view addend, bool carry) {
  SmallVector<double> angles;
  angles.reserve(addend.size() + 1U);
  long double angle = 0.L;
  for (const char bit : addend | std::views::reverse) {
    angle /= 2.L;
    if (bit == '1') {
      angle += std::numbers::pi_v<long double>;
    }
    angles.push_back(static_cast<double>(angle));
  }
  if (carry) {
    angles.push_back(static_cast<double>(angle / 2.L));
  }

  const auto type = RankedTensorType::get({static_cast<int64_t>(angles.size())},
                                          builder.getF64Type());
  const auto value = DenseElementsAttr::get(type, ArrayRef<double>(angles));
  return arith::ConstantOp::create(builder, value).getResult();
}

SmallVector<Value> qftAdder(qc::QCProgramBuilder& builder,
                            const QFTAdder& benchmark) {
  const auto& options = benchmark.options();
  const auto qubits = static_cast<int64_t>(options.addend.size());
  const auto carry = options.overflow == QFTAdderOverflow::Carry;
  const auto sumQubits = qubits + static_cast<int64_t>(carry);
  Value addend;
  if (options.method == QFTAdderMethod::Register) {
    addend = builder.allocQubitRegisterStorage(qubits, "addend");
    detail::prepareRegister(builder, addend, options.addend);
  }
  auto sum = builder.allocQubitRegisterStorage(sumQubits, "sum");
  detail::prepareRegister(builder, sum, options.accumulator);
  auto result = builder.allocClassicalBitRegister(
      static_cast<int64_t>(benchmark.output().width), benchmark.output().name);

  detail::forwardQFT(builder, sum, sumQubits);
  if (addend) {
    addQuantumRegister(builder, addend, sum, qubits, carry);
  } else {
    auto angles = phaseAngles(builder, options.addend, carry);
    builder.scfFor(0, sumQubits, 1, [&](Value target) {
      auto angle =
          tensor::ExtractOp::create(builder, angles, ValueRange{target});
      builder.p(angle, builder.loadQubit(sum, target));
    });
  }
  detail::inverseQFT(builder, sum, sumQubits);

  builder.measureQubitRegister(sum, result, sumQubits);
  if (addend) {
    auto offset = builder.indexConstant(sumQubits);
    builder.scfFor(0, qubits, 1, [&](Value index) {
      auto resultIndex = arith::AddIOp::create(builder, offset, index);
      builder.measure(builder.loadQubit(addend, index), result, resultIndex);
    });
  }
  return {result};
}

} // namespace mqt::bench
