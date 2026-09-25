/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "bench/Evaluation.hpp"

#include "support/Diagnostics.hpp"

#include "mlir/Support/LogicalResult.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <string_view>

namespace mqt::bench::detail {

inline mlir::LogicalResult validateOutcome(const std::string_view outcome,
                                           const size_t width) {
  if (outcome.size() != width) {
    return ::mqt::emitError("outcome width does not match the benchmark output",
                            ::mqt::ErrorCategory::InvalidArgument);
  }
  if (!std::ranges::all_of(
          outcome, [](const char bit) { return bit == '0' || bit == '1'; })) {
    return ::mqt::emitError("outcome must contain only '0' and '1'",
                            ::mqt::ErrorCategory::InvalidArgument);
  }
  return mlir::success();
}

[[nodiscard]] inline mlir::FailureOr<size_t>
validateCounts(const Output& output, const Counts& counts) {
  if (counts.empty()) {
    return ::mqt::emitError("counts must not be empty",
                            ::mqt::ErrorCategory::InvalidArgument);
  }

  size_t totalShots = 0;
  for (const auto& [outcome, count] : counts) {
    if (mlir::failed(validateOutcome(outcome, output.width))) {
      return mlir::failure();
    }
    if (count > std::numeric_limits<size_t>::max() - totalShots) {
      return ::mqt::emitError("total shot count exceeds size_t",
                              ::mqt::ErrorCategory::Overflow);
    }
    totalShots += count;
  }
  if (totalShots == 0) {
    return ::mqt::emitError("total shot count must be positive",
                            ::mqt::ErrorCategory::InvalidArgument);
  }

  return totalShots;
}

template <class Probability>
[[nodiscard]] mlir::FailureOr<Evaluation>
evaluate(const Output& output, const Counts& counts,
         const Probability& probability,
         const std::optional<std::string_view> successOutcome = std::nullopt) {
  const auto totalShots = validateCounts(output, counts);
  if (mlir::failed(totalShots)) {
    return mlir::failure();
  }
  size_t successShots = 0;

  // Extended precision prevents avoidable loss while summing distributions.
  // NOLINTBEGIN(google-runtime-float)
  long double observedDistance = 0.L;
  long double observedIdealMass = 0.L;
  long double coefficient = 0.L;
  for (const auto& [outcome, count] : counts) {
    if (successOutcome && outcome == *successOutcome) {
      successShots = count;
    }
    auto reference = probability(outcome, count);
    if (mlir::failed(reference)) {
      return mlir::failure();
    }
    const auto ideal = static_cast<long double>((*reference));
    const auto observed =
        static_cast<long double>(count) / static_cast<long double>(*totalShots);
    observedDistance += std::abs(observed - ideal);
    observedIdealMass += ideal;
    coefficient += std::sqrt(observed * ideal);
  }

  const auto missingIdealMass = std::max(0.L, 1.L - observedIdealMass);
  const auto totalVariation =
      std::clamp((observedDistance + missingIdealMass) / 2.L, 0.L, 1.L);
  const auto fidelity = std::clamp(coefficient * coefficient, 0.L, 1.L);
  const auto success =
      successOutcome ? std::optional<double>{static_cast<double>(successShots) /
                                             static_cast<double>(*totalShots)}
                     : std::nullopt;
  // NOLINTEND(google-runtime-float)
  return Evaluation{
      .totalVariationDistance = static_cast<double>(totalVariation),
      .squaredHellingerFidelity = static_cast<double>(fidelity),
      .successProbability = success,
  };
}

template <class Benchmark>
[[nodiscard]] mlir::FailureOr<Evaluation>
evaluate(const Benchmark& benchmark, const Counts& counts,
         const std::optional<std::string_view> successOutcome = std::nullopt) {
  return evaluate(
      benchmark.output(), counts,
      [&benchmark](const std::string_view outcome, size_t) {
        return benchmark.probability(outcome);
      },
      successOutcome);
}

} // namespace mqt::bench::detail
