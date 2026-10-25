/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/GroverWeakMeasurement.hpp"

#include "bench/Evaluation.hpp"

#include "EvaluationUtils.hpp"

#include "llvm/Support/Error.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <string_view>
#include <system_error>
#include <utility>

namespace mqt::bench {
namespace {

[[nodiscard]] double defaultMeasurementStrength(const size_t qubits) {
  return std::exp2(-static_cast<double>(qubits) / 2.);
}

} // namespace

GroverWeakMeasurement::GroverWeakMeasurement(
    GroverWeakMeasurementOptions options)
    : options_(std::move(options)),
      output_{.name = "result", .width = options_.markedBitstring.size()} {}

llvm::Expected<GroverWeakMeasurement>
GroverWeakMeasurement::create(GroverWeakMeasurementOptions options) {
  const auto width = options.markedBitstring.size();
  if (width < 2 || width > GroverWeakMeasurementOptions::MAX_QUBITS) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "weak-measurement Grover requires a marked bitstring of width 2 "
        "through 2044");
  }
  if (auto error = detail::validateOutcome(options.markedBitstring, width)) {
    return error;
  }

  const auto maximumStrength = defaultMeasurementStrength(width);
  if (!options.measurementStrength) {
    options.measurementStrength = maximumStrength;
  }
  if (!std::isfinite(*options.measurementStrength) ||
      *options.measurementStrength <= 0. ||
      *options.measurementStrength > maximumStrength) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "weak-measurement Grover measurement strength must be finite and in "
        "(0, 2^(-n/2)]");
  }
  return GroverWeakMeasurement(std::move(options));
}

const GroverWeakMeasurementOptions&
GroverWeakMeasurement::options() const noexcept {
  return options_;
}

size_t GroverWeakMeasurement::qubits() const noexcept { return output_.width; }

const Output& GroverWeakMeasurement::output() const noexcept { return output_; }

double
GroverWeakMeasurement::probability(const std::string_view outcome) const {
  assert(outcome.size() == output_.width &&
         outcome.find_first_not_of("01") == std::string_view::npos);
  return outcome == options_.markedBitstring ? 1. : 0.;
}

llvm::Expected<Evaluation>
GroverWeakMeasurement::evaluate(const Counts& counts) const {
  return detail::evaluate(*this, counts, options_.markedBitstring);
}

} // namespace mqt::bench
