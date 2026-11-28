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

#include "EvaluationUtils.hpp"

#include "llvm/Support/Error.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <numeric>
#include <optional>
#include <system_error>
#include <utility>
#include <vector>

namespace mqt::bench {
namespace {

constexpr double HALF_PI = std::numbers::pi / 2.;

struct AmplificationCounts {
  size_t good;
  size_t total;
};

llvm::Expected<AmplificationCounts> samplePower(const QAEProblem& problem,
                                                size_t iterations, QAERun run) {
  auto experiment = QAE::create({.problem = problem, .iterations = iterations});
  if (!experiment) {
    return experiment.takeError();
  }
  if (!run) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "QAE requires an execution callback");
  }
  auto counts = run(*experiment);
  if (!counts) {
    return counts.takeError();
  }
  auto total = detail::validateCounts(*counts, experiment->output().width);
  if (!total) {
    return total.takeError();
  }
  size_t good = 0;
  for (const auto& [bits, count] : *counts) {
    if (std::ranges::binary_search(experiment->options().problem.goodStates,
                                   bits)) {
      good += count;
    }
  }
  return AmplificationCounts{.good = good, .total = *total};
}

struct LikelihoodTerm {
  double frequency;
  AmplificationCounts counts;
  // Next multiple of π/2 in this term's amplified angle.
  size_t boundary = 1;
};

double fitLikelihood(std::vector<LikelihoodTerm>& terms) {
  if (std::ranges::all_of(terms,
                          [](const auto& t) { return t.counts.good == 0; })) {
    return 0.;
  }
  if (std::ranges::all_of(terms, [](const auto& t) {
        return t.counts.good == t.counts.total;
      })) {
    return 1.;
  }
  double bestScore = -std::numeric_limits<double>::infinity();
  double bestAngle = 0.;
  double left = 0.;
  while (left < 1.) {
    double right = 1.;
    for (const auto& t : terms) {
      right = std::min(right, static_cast<double>(t.boundary) / t.frequency);
    }
    // Between consecutive zeros of sine and cosine the joint log likelihood
    // is concave. Its decreasing derivative locates the maximum on this
    // interval.
    double low = left;
    double high = right;
    for (size_t step = 0; step < std::numeric_limits<double>::digits; ++step) {
      const auto mid = std::midpoint(low, high);
      if (mid == low || mid == high) {
        break;
      }
      double slope = 0.;
      for (const auto& t : terms) {
        const auto tangent =
            std::tan(HALF_PI * std::remainder(mid * t.frequency, 2.));
        if (t.counts.good != 0) {
          slope += t.frequency * static_cast<double>(t.counts.good) / tangent;
        }
        if (t.counts.total != t.counts.good) {
          slope -= t.frequency *
                   static_cast<double>(t.counts.total - t.counts.good) *
                   tangent;
        }
      }
      if (slope > 0.) {
        low = mid;
      } else {
        high = mid;
      }
    }
    const auto candidate = std::midpoint(low, high);
    double score = 0.;
    for (auto& t : terms) {
      const auto angle = HALF_PI * std::remainder(candidate * t.frequency, 2.);
      if (t.counts.good != 0) {
        score += static_cast<double>(t.counts.good) *
                 std::log(std::abs(std::sin(angle)));
      }
      if (t.counts.good != t.counts.total) {
        score += static_cast<double>(t.counts.total - t.counts.good) *
                 std::log(std::abs(std::cos(angle)));
      }
      if (static_cast<double>(t.boundary) / t.frequency == right) {
        ++t.boundary;
      }
    }
    if (score > bestScore) {
      bestScore = score;
      bestAngle = candidate;
    }
    left = right;
  }
  const auto sine = std::sin(HALF_PI * bestAngle);
  return sine * sine;
}

double amplitudeAt(double angle) {
  const auto sine = std::sin(angle);
  return sine * sine;
}

} // namespace

llvm::Expected<double> maximumLikelihood(const QAEProblem& problem,
                                         size_t rounds, QAERun run) {
  if (rounds == 0 || rounds > QAEOptions::MAX_PRECISION) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "QAE rounds must be between 1 and 31");
  }
  std::vector<LikelihoodTerm> terms;
  for (size_t round = 0; round < rounds; ++round) {
    const auto power = size_t{1} << round;
    auto counts = samplePower(problem, power, run);
    if (!counts) {
      return counts.takeError();
    }
    terms.push_back({
        .frequency = (2. * static_cast<double>(power)) + 1.,
        .counts = *counts,
    });
  }
  return fitLikelihood(terms);
}

llvm::Expected<std::pair<double, double>>
iterativeAmplitudeEstimation(const QAEProblem& problem, double epsilon,
                             double confidence, size_t maxBatches, QAERun run) {
  if (!std::isfinite(epsilon) || epsilon <= 0. || epsilon >= 0.5 ||
      !std::isfinite(confidence) || confidence <= 0. || confidence >= 1. ||
      maxBatches == 0) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "IQAE requires 0 < epsilon < 0.5, 0 < confidence < 1, and a positive "
        "batch budget");
  }
  // Budget one Hoeffding interval for every possible start and length of a
  // sequence of batches at a fixed power, including repeated looks at its
  // counts.
  const auto batches = static_cast<double>(maxBatches);
  const auto logBound = std::log(batches * (batches + 1.) / (1. - confidence));
  double low = 0.;
  double high = HALF_PI;
  int64_t frequency = 1;
  int64_t lobe = 0;
  AmplificationCounts pooled{};
  std::optional<size_t> batchSize;
  for (size_t batch = 0; batch < maxBatches; ++batch) {
    // Find the largest odd frequency that at least doubles the previous one
    // and keeps the amplified interval on one monotone branch of sin².
    constexpr auto maxFrequency =
        (2 * static_cast<int64_t>(QAEOptions::MAX_ITERATIONS)) + 1;
    auto candidate = static_cast<int64_t>(
        std::min(static_cast<double>(maxFrequency), HALF_PI / (high - low)));
    if (candidate % 2 == 0) {
      --candidate;
    }
    for (; candidate >= 2 * frequency; candidate -= 2) {
      const auto scale = static_cast<double>(candidate);
      const auto branch =
          static_cast<int64_t>(std::floor(scale * low / HALF_PI));
      if (scale * high <= static_cast<double>(branch + 1) * HALF_PI) {
        frequency = candidate;
        lobe = branch;
        pooled = {};
        break;
      }
    }
    auto counts = samplePower(problem, static_cast<size_t>(frequency / 2), run);
    if (!counts) {
      return counts.takeError();
    }
    if (batchSize && *batchSize != counts->total) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "IQAE requires the same shot count in each batch");
    }
    batchSize = counts->total;
    if (counts->total > std::numeric_limits<size_t>::max() - pooled.total) {
      return llvm::createStringError(std::errc::value_too_large,
                                     "pooled IQAE shot count exceeds size_t");
    }
    pooled.good += counts->good;
    pooled.total += counts->total;
    const auto shots = static_cast<double>(pooled.total);
    const auto fraction = static_cast<double>(pooled.good) / shots;
    const auto radius = std::sqrt(logBound / (2. * shots));
    auto lowerAngle = std::asin(std::sqrt(std::max(0., fraction - radius)));
    auto upperAngle = std::asin(std::sqrt(std::min(1., fraction + radius)));
    if (lobe % 2 != 0) {
      std::swap(lowerAngle, upperAngle);
      lowerAngle = HALF_PI - lowerAngle;
      upperAngle = HALF_PI - upperAngle;
    }
    const auto offset = static_cast<double>(lobe) * HALF_PI;
    low = std::max(low, (offset + lowerAngle) / static_cast<double>(frequency));
    high =
        std::min(high, (offset + upperAngle) / static_cast<double>(frequency));
    if (low > high) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "IQAE observations give disjoint confidence intervals");
    }
    const std::pair bounds{amplitudeAt(low), amplitudeAt(high)};
    if (bounds.second - bounds.first <= 2. * epsilon) {
      return bounds;
    }
  }
  return llvm::createStringError(
      std::errc::result_out_of_range,
      "IQAE exhausted its batch budget before reaching the requested accuracy");
}

} // namespace mqt::bench
