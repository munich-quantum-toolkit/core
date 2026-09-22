/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/WeakMeasurementGrover.hpp"

#include "bench/Evaluation.hpp"

#include "EvaluationUtils.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace mqt::bench {
namespace {

[[nodiscard]] double defaultMeasurementStrength(const size_t qubits) {
  return std::exp2(-static_cast<double>(qubits) / 2.);
}

} // namespace

WeakMeasurementGrover::WeakMeasurementGrover(
    WeakMeasurementGroverOptions options)
    : options_(std::move(options)),
      output_{.name = "result", .width = options_.markedBitstring.size()} {
  const auto width = options_.markedBitstring.size();
  if (width < 2 || width > WeakMeasurementGroverOptions::MAX_QUBITS) {
    throw std::invalid_argument(
        "weak-measurement Grover requires a marked bitstring of width 2 "
        "through 62");
  }
  detail::validateOutcome(options_.markedBitstring, width);

  const auto maximumStrength = defaultMeasurementStrength(width);
  if (!options_.measurementStrength) {
    options_.measurementStrength = maximumStrength;
  }
  if (!std::isfinite(*options_.measurementStrength) ||
      *options_.measurementStrength <= 0. ||
      *options_.measurementStrength > maximumStrength) {
    throw std::invalid_argument(
        "weak-measurement Grover measurement strength must be finite and in "
        "(0, 2^(-n/2)]");
  }
}

const WeakMeasurementGroverOptions&
WeakMeasurementGrover::options() const noexcept {
  return options_;
}

size_t WeakMeasurementGrover::qubits() const noexcept { return output_.width; }

const Output& WeakMeasurementGrover::output() const noexcept { return output_; }

double
WeakMeasurementGrover::probability(const std::string_view outcome) const {
  detail::validateOutcome(outcome, output_.width);
  return outcome == options_.markedBitstring ? 1. : 0.;
}

Evaluation WeakMeasurementGrover::evaluate(const Counts& counts) const {
  return detail::evaluate(*this, counts, options_.markedBitstring);
}

} // namespace mqt::bench
