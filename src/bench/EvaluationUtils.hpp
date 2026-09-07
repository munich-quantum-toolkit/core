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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace mqt::bench::detail {

inline void validateOutcome(const std::string_view outcome,
                            const size_t width) {
  if (outcome.size() != width) {
    throw std::invalid_argument(
        "outcome width does not match the benchmark output");
  }
  if (!std::ranges::all_of(
          outcome, [](const char bit) { return bit == '0' || bit == '1'; })) {
    throw std::invalid_argument("outcome must contain only '0' and '1'");
  }
}

template <class Probability>
[[nodiscard]] Evaluation
evaluate(const Output& output, const Counts& counts,
         const Probability& probability,
         const std::optional<std::string_view> successOutcome = std::nullopt) {
  if (counts.empty()) {
    throw std::invalid_argument("counts must not be empty");
  }

  size_t totalShots = 0;
  size_t successShots = 0;
  for (const auto& [outcome, count] : counts) {
    validateOutcome(outcome, output.width);
    if (count > std::numeric_limits<size_t>::max() - totalShots) {
      throw std::overflow_error("total shot count exceeds size_t");
    }
    totalShots += count;
    if (successOutcome && outcome == *successOutcome) {
      successShots = count;
    }
  }
  if (totalShots == 0) {
    throw std::invalid_argument("total shot count must be positive");
  }

  // Extended precision prevents avoidable loss while summing distributions.
  // NOLINTBEGIN(google-runtime-float)
  long double observedDistance = 0.L;
  long double observedIdealMass = 0.L;
  long double coefficient = 0.L;
  for (const auto& [outcome, count] : counts) {
    const auto ideal = static_cast<long double>(probability(outcome));
    const auto observed =
        static_cast<long double>(count) / static_cast<long double>(totalShots);
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
                                             static_cast<double>(totalShots)}
                     : std::nullopt;
  // NOLINTEND(google-runtime-float)
  return {
      .totalVariationDistance = static_cast<double>(totalVariation),
      .squaredHellingerFidelity = static_cast<double>(fidelity),
      .successProbability = success,
  };
}

} // namespace mqt::bench::detail
