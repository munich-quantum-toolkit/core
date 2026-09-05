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
#include "QFTAdderUtils.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"

#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <cstdint>
#include <numbers>
#include <ranges>
#include <string_view>

namespace mqt::bench {

using namespace mlir;

[[nodiscard]] static SmallVector<double>
phaseAngles(const std::string_view addend) {
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
  return angles;
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
  const auto angles = phaseAngles(benchmark.options().addend);
  for (size_t target = 0; target < angles.size(); ++target) {
    auto angle = builder.floatConstant(angles[target]);
    auto index = builder.indexConstant(static_cast<int64_t>(target));
    builder.p(angle, builder.loadQubit(sum, index));
  }
  detail::inverseQFT(builder, sum, qubits);

  builder.measureQubitRegister(sum, result, qubits);
  return {result};
}

} // namespace mqt::bench
