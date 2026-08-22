/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "benchmarks/Grover.hpp"

#include "EvaluationUtils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace mqt::benchmarks {
namespace {

[[nodiscard]] long double successProbability(const size_t iterations,
                                             const long double theta) {
  const auto angle = (2.L * static_cast<long double>(iterations) + 1.L) * theta;
  const auto amplitude = std::sin(angle);
  return amplitude * amplitude;
}

[[nodiscard]] size_t resolveIterations(const size_t qubits) {
  const auto states = std::ldexp(1.L, static_cast<int>(qubits));
  const auto theta = std::asin(1.L / std::sqrt(states));
  const auto target = std::numbers::pi_v<long double> / (4.L * theta) - 0.5L;
  return static_cast<size_t>(std::floor(target + 0.5L));
}

} // namespace

Grover::Grover(GroverOptions options)
    : options_(std::move(options)),
      output_{"result", options_.markedBitstring.size()} {
  const auto width = options_.markedBitstring.size();
  if (width < 2 || width > 62) {
    throw std::invalid_argument(
        "Grover requires a marked bitstring of width 2 through 62");
  }
  detail::validateOutcome(options_.markedBitstring, width);

  if (!options_.iterations) {
    options_.iterations = resolveIterations(width);
  }
  if (*options_.iterations >
      static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw std::invalid_argument(
        "Grover iterations must fit a signed 32-bit integer");
  }

  const auto states = std::ldexp(1.L, static_cast<int>(width));
  const auto theta = std::asin(1.L / std::sqrt(states));
  const auto marked =
      std::clamp(successProbability(*options_.iterations, theta), 0.L, 1.L);
  markedProbability_ = static_cast<double>(marked);
  otherProbability_ = static_cast<double>((1.L - marked) / (states - 1.L));
}

const GroverOptions& Grover::options() const noexcept { return options_; }

size_t Grover::qubits() const noexcept { return output_.width; }

const Output& Grover::output() const noexcept { return output_; }

double Grover::markedProbability() const noexcept { return markedProbability_; }

double Grover::otherProbability() const noexcept { return otherProbability_; }

double Grover::probability(const std::string_view outcome) const {
  detail::validateOutcome(outcome, output_.width);
  return outcome == options_.markedBitstring ? markedProbability_
                                             : otherProbability_;
}

Evaluation Grover::evaluate(const Counts& counts) const {
  return detail::evaluate(
      output_, counts,
      [this](const std::string_view outcome) { return probability(outcome); },
      options_.markedBitstring);
}

} // namespace mqt::benchmarks
