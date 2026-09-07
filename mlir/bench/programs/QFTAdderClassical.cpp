/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/QFTAdderClassical.hpp"

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

#include <cstdint>
#include <numbers>
#include <ranges>
#include <string_view>

namespace mqt::bench {

using namespace mlir;

[[nodiscard]] static Value phaseAngles(qc::QCProgramBuilder& builder,
                                       const std::string_view addend) {
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
  angles.push_back(static_cast<double>(angle / 2.L));

  const auto type = RankedTensorType::get({static_cast<int64_t>(angles.size())},
                                          builder.getF64Type());
  const auto value = DenseElementsAttr::get(type, ArrayRef<double>(angles));
  return arith::ConstantOp::create(builder, value).getResult();
}

SmallVector<Value> qftAdderClassical(qc::QCProgramBuilder& builder,
                                     const QFTAdderClassical& benchmark) {
  const auto qubits =
      static_cast<int64_t>(benchmark.options().addend.size() + 1U);
  auto sum = builder.allocQubitRegisterStorage(qubits, "sum");
  auto result = builder.allocClassicalBitRegister(
      static_cast<int64_t>(benchmark.output().width), benchmark.output().name);

  auto zero = builder.indexConstant(0);
  builder.x(builder.loadQubit(sum, zero));

  detail::forwardQFT(builder, sum, qubits);
  auto angles = phaseAngles(builder, benchmark.options().addend);
  builder.scfFor(0, qubits, 1, [&](Value target) {
    auto angle = tensor::ExtractOp::create(builder, angles, ValueRange{target})
                     .getResult();
    builder.p(angle, builder.loadQubit(sum, target));
  });
  detail::inverseQFT(builder, sum, qubits);

  builder.measureQubitRegister(sum, result, qubits);
  return {result};
}

} // namespace mqt::bench
