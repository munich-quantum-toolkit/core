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

#include "bench/Evaluation.hpp"

#include "EvaluationUtils.hpp"

#include "llvm/Support/Error.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <numbers>
#include <string_view>
#include <system_error>
#include <utility>

namespace mqt::bench {
namespace {

uint64_t binIndex(std::string_view bits) {
  uint64_t index = 0;
  for (const char bit : bits) {
    index = (index << 1U) + static_cast<unsigned>(bit - '0');
  }
  return index;
}

double squared(double x) { return x * x; }

// Squared Fourier amplitude at a distance measured in phase bins.
double phasePeak(double peak, double bin, double bins) {
  const auto distance = std::remainder(peak - bin, bins);
  if (distance == 0.) {
    return 1.;
  }
  return squared(std::sin(std::numbers::pi * std::remainder(peak, 1.)) /
                 (bins * std::sin(std::numbers::pi * distance / bins)));
}

} // namespace

QAE::QAE(QAEOptions options)
    : options_(std::move(options)),
      output_{
          .name = "result",
          .width = options_.precision.value_or(
              options_.problem.goodStates.front().size()),
      } {}

llvm::Expected<QAE> QAE::create(QAEOptions options) {
  auto& [states, amplitude] = options.problem;
  if (!std::isfinite(amplitude) || amplitude < 0. || amplitude > 1.) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "amplitude must be a probability in [0, 1]");
  }
  // Both signs of zero describe the same experiment and case ID.
  amplitude = std::abs(amplitude);
  if (states.empty() || states.front().empty() ||
      states.front().size() > QAEOptions::MAX_QUBITS) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "good_states must contain bitstrings of width 1 through 63");
  }
  const auto width = states.front().size();
  for (const auto& state : states) {
    if (auto error = detail::validateOutcome(state, width)) {
      return error;
    }
  }
  std::ranges::sort(states);
  if (std::ranges::adjacent_find(states) != states.end() ||
      states.size() == (uint64_t{1} << width)) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "good_states must be distinct and leave at least one bad state");
  }
  if (options.precision.has_value() == options.iterations.has_value()) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "specify exactly one of precision and iterations");
  }
  if (options.precision && (*options.precision == 0 ||
                            *options.precision > QAEOptions::MAX_PRECISION)) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "QAE precision must be between 1 and 31");
  }
  if (options.iterations && *options.iterations > QAEOptions::MAX_ITERATIONS) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "QAE iterations must fit a signed 32-bit integer");
  }
  return QAE(std::move(options));
}

const QAEOptions& QAE::options() const noexcept { return options_; }
const Output& QAE::output() const noexcept { return output_; }

double QAE::probability(std::string_view outcome) const {
  assert(outcome.size() == output_.width &&
         outcome.find_first_not_of("01") == std::string_view::npos);
  const auto& [states, amplitude] = options_.problem;
  // Work with the smaller class probability to retain precision near either
  // endpoint.
  const auto reflected = amplitude > 0.5;
  const auto theta =
      std::asin(std::sqrt(reflected ? 1. - amplitude : amplitude));
  if (options_.precision) {
    const auto bins = uint64_t{1} << *options_.precision;
    const auto index = binIndex(outcome) ^ (reflected ? bins / 2 : 0);
    const auto scale = static_cast<double>(bins);
    const auto peak =
        amplitude == 0.5 ? scale / 4. : scale * (theta / std::numbers::pi);
    return 0.5 * (phasePeak(peak, static_cast<double>(index), scale) +
                  phasePeak(peak, -static_cast<double>(index), scale));
  }
  const auto good = std::ranges::binary_search(states, outcome);
  const auto count =
      good ? states.size() : (uint64_t{1} << output_.width) - states.size();
  const auto angle =
      ((2. * static_cast<double>(*options_.iterations)) + 1.) * theta;
  const auto mass =
      amplitude == 0.5
          ? 0.5
          : squared(good != reflected ? std::sin(angle) : std::cos(angle));
  return mass / static_cast<double>(count);
}

llvm::Expected<Evaluation> QAE::evaluate(const Counts& counts) const {
  return detail::evaluate(*this, counts);
}

llvm::Expected<double> phaseEstimation(const QAEProblem& problem,
                                       size_t precision, QAERun run) {
  auto benchmark = QAE::create({.problem = problem, .precision = precision});
  if (!benchmark) {
    return benchmark.takeError();
  }
  if (!run) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "QAE requires an execution callback");
  }
  auto counts = run(*benchmark);
  if (!counts) {
    return counts.takeError();
  }
  if (auto shots = detail::validateCounts(*counts, precision); !shots) {
    return shots.takeError();
  }
  const auto bins = uint64_t{1} << precision;
  std::map<uint64_t, size_t> folded;
  for (const auto& [outcome, count] : *counts) {
    const auto index = binIndex(outcome);
    folded[std::min(index, bins - index)] += count;
  }
  const auto peak = std::ranges::max_element(
      folded, {}, [](const auto& bin) { return bin.second; });
  return squared(std::sin(std::numbers::pi * static_cast<double>(peak->first) /
                          static_cast<double>(bins)));
}

} // namespace mqt::bench
