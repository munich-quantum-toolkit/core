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

#include "bench/Error.hpp"
#include "bench/Evaluation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <string_view>
#include <utility>
#include <variant>

namespace mqt::bench::detail {

inline std::optional<Error> validateOutcome(const std::string_view outcome,
                                            const size_t width) {
  if (outcome.size() != width) {
    return Error{
        .message = "outcome width does not match the benchmark output",
    };
  }
  if (!std::ranges::all_of(
          outcome, [](const char bit) { return bit == '0' || bit == '1'; })) {
    return Error{.message = "outcome must contain only '0' and '1'"};
  }
  return std::nullopt;
}

template <class Probability>
[[nodiscard]] Result<Evaluation>
evaluate(const Output& output, const Counts& counts,
         const Probability& probability,
         const std::optional<std::string_view> successOutcome = std::nullopt) {
  if (counts.empty()) {
    return Error{.message = "counts must not be empty"};
  }

  size_t totalShots = 0;
  size_t successShots = 0;
  for (const auto& [outcome, count] : counts) {
    if (auto error = validateOutcome(outcome, output.width)) {
      return std::move(*error);
    }
    if (count > std::numeric_limits<size_t>::max() - totalShots) {
      return Error{
          .message = "total shot count exceeds size_t",
          .kind = Error::Kind::Overflow,
      };
    }
    totalShots += count;
    if (successOutcome && outcome == *successOutcome) {
      successShots = count;
    }
  }
  if (totalShots == 0) {
    return Error{.message = "total shot count must be positive"};
  }

  // Extended precision prevents avoidable loss while summing distributions.
  // NOLINTBEGIN(google-runtime-float)
  long double observedDistance = 0.L;
  long double observedIdealMass = 0.L;
  long double coefficient = 0.L;
  for (const auto& [outcome, count] : counts) {
    auto reference = probability(outcome);
    if (auto* error = std::get_if<Error>(&reference)) {
      return std::move(*error);
    }
    const auto ideal = static_cast<long double>(std::get<double>(reference));
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
  return Evaluation{
      .totalVariationDistance = static_cast<double>(totalVariation),
      .squaredHellingerFidelity = static_cast<double>(fidelity),
      .successProbability = success,
  };
}

template <class Benchmark>
[[nodiscard]] Result<Evaluation>
evaluate(const Benchmark& benchmark, const Counts& counts,
         const std::optional<std::string_view> successOutcome = std::nullopt) {
  return evaluate(
      benchmark.output(), counts,
      [&benchmark](const std::string_view outcome) {
        return benchmark.probability(outcome);
      },
      successOutcome);
}

} // namespace mqt::bench::detail
