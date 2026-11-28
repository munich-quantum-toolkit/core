/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/QAE.hpp"

#include "mqt/Dialect/QC/Builder/QCProgramBuilder.h"

#include "GroverUtils.h"
#include "Programs.h"
#include "QPEUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <string>
#include <string_view>

namespace mqt::bench {
using namespace mlir;

// A conditional Y rotation splits a prefix's probability between its children.
// Prefixes with only bad states split evenly, so one π/2 rotation per level
// covers them all without enumerating the complement of the good set.
static void prepareQAEState(qc::QCProgramBuilder& builder, ArrayRef<Value> data,
                            const QAEProblem& problem, bool inverse = false) {
  const auto& states = problem.goodStates;
  const auto space = uint64_t{1} << data.size();
  const auto goodWeight = std::sqrt(problem.amplitude) /
                          std::sqrt(static_cast<double>(states.size()));
  const auto badWeight = std::sqrt(1. - problem.amplitude) /
                         std::sqrt(static_cast<double>(space - states.size()));
  const auto sign = inverse ? -1. : 1.;
  for (size_t level = 0; level < data.size(); ++level) {
    const auto depth = inverse ? data.size() - 1 - level : level;
    const auto index = data.size() - 1 - depth;
    auto target = data[index];
    auto controls = data.drop_front(index + 1);
    const auto childSize = uint64_t{1} << index;
    if (!controls.empty()) {
      builder.ry(sign * std::numbers::pi / 2., target);
    }
    for (auto first = states.begin(); first != states.end();) {
      const auto prefix = std::string_view(*first).substr(0, depth);
      const auto last =
          std::partition_point(first, states.end(), [&](const auto& state) {
            return state.starts_with(prefix);
          });
      const auto middle = std::partition_point(
          first, last, [&](const auto& state) { return state[depth] == '0'; });
      const auto weight = [&](uint64_t good) {
        return std::hypot(std::sqrt(static_cast<double>(good)) * goodWeight,
                          std::sqrt(static_cast<double>(childSize - good)) *
                              badWeight);
      };
      const auto left = weight(static_cast<uint64_t>(middle - first));
      const auto right = weight(static_cast<uint64_t>(last - middle));
      if (controls.empty()) {
        builder.ry(sign * 2. * std::atan2(right, left), target);
      } else if (left != right) {
        // Subtract the uniform π/2 rotation. All rotations at this level
        // share the Y axis and commute, including when applying the adjoint.
        const auto correction = 2. * std::atan2(right - left, right + left);
        detail::maskMarkedState(builder, controls, prefix);
        builder.mcry(sign * correction, controls, target);
        detail::maskMarkedState(builder, controls, prefix);
      }
      first = last;
    }
  }
}

SmallVector<Value> qae(qc::QCProgramBuilder& builder, const QAE& benchmark) {
  const auto& options = benchmark.options();
  const auto& states = options.problem.goodStates;
  const auto width = static_cast<int64_t>(states.front().size());
  auto data = builder.allocQubitRegister(width, "data");
  auto result = builder.allocClassicalBitRegister(
      static_cast<int64_t>(benchmark.output().width), benchmark.output().name);
  prepareQAEState(builder, data.qubits, options.problem);
  const std::string zeroState(states.front().size(), '0');
  const auto iterate = [&](Value control) {
    for (const auto& state : states) {
      detail::markPhase(builder, data.qubits, state, control);
    }
    prepareQAEState(builder, data.qubits, options.problem, true);
    detail::markPhase(builder, data.qubits, zeroState, control);
    prepareQAEState(builder, data.qubits, options.problem);
    // The reflection about the prepared state is -A S₀ A†. Under control
    // the minus sign becomes a relative phase on the query qubit.
    if (control) {
      builder.z(control);
    }
  };
  if (options.iterations) {
    builder.scfFor(0, static_cast<int64_t>(*options.iterations), 1,
                   [&](Value) { iterate({}); });
    builder.measureQubitRegister(data.value, result, width);
  } else {
    const auto precision = static_cast<int64_t>(*options.precision);
    detail::measureEigenphase(
        builder, precision, result, builder.indexConstant(1),
        [&](Value control, Value power) {
          builder.scfFor(0, power, 1, [&](Value) { iterate(control); });
          return arith::AddIOp::create(builder, power, power).getResult();
        });
  }
  return {result};
}
} // namespace mqt::bench
