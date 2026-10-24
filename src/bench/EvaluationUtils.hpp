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

#include "llvm/Support/Error.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <string_view>
#include <system_error>

namespace mqt::bench::detail {

inline llvm::Error validateOutcome(const std::string_view outcome,
                                   const size_t width) {
  if (outcome.size() != width) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "outcome width does not match the benchmark output");
  }
  if (!std::ranges::all_of(
          outcome, [](const char bit) { return bit == '0' || bit == '1'; })) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "outcome must contain only '0' and '1'");
  }
  return llvm::Error::success();
}

[[nodiscard]] inline llvm::Expected<size_t> countShots(const Counts& counts) {
  if (counts.empty()) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "counts must not be empty");
  }

  size_t totalShots = 0;
  for (const auto& [outcome, count] : counts) {
    if (count > std::numeric_limits<size_t>::max() - totalShots) {
      return llvm::createStringError(std::errc::value_too_large,
                                     "total shot count exceeds size_t");
    }
    totalShots += count;
  }
  if (totalShots == 0) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "total shot count must be positive");
  }

  return totalShots;
}

template <class Probability>
[[nodiscard]] llvm::Expected<Evaluation>
evaluate(const Counts& counts, const Probability& probability,
         const std::optional<std::string_view> successOutcome = std::nullopt) {
  auto totalShots = countShots(counts);
  if (!totalShots) {
    return totalShots.takeError();
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
    auto value = probability(outcome, count);
    if (!value) {
      return value.takeError();
    }
    const auto ideal = static_cast<long double>(*value);
    const auto observed =
        static_cast<long double>(count) / static_cast<long double>(*totalShots);
    observedDistance += std::abs(observed - ideal);
    observedIdealMass += ideal;
    // Normalize after summation so a positive ideal does not underflow first.
    coefficient += std::sqrt(static_cast<long double>(count) * ideal);
  }

  const auto missingIdealMass = std::max(0.L, 1.L - observedIdealMass);
  const auto totalVariation =
      std::clamp((observedDistance + missingIdealMass) / 2.L, 0.L, 1.L);
  const auto fidelity = std::clamp(coefficient * coefficient /
                                       static_cast<long double>(*totalShots),
                                   0.L, 1.L);
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
[[nodiscard]] llvm::Expected<Evaluation>
evaluate(const Benchmark& benchmark, const Counts& counts,
         const std::optional<std::string_view> successOutcome = std::nullopt) {
  return evaluate(
      counts,
      [&benchmark](const std::string_view outcome, size_t) {
        return benchmark.probability(outcome);
      },
      successOutcome);
}

} // namespace mqt::bench::detail
