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
#include "support/Diagnostics.hpp"

#include "llvm/Support/LogicalResult.h"

#include <cmath>
#include <cstddef>
#include <string_view>
#include <utility>

namespace mqt::bench {
namespace {

[[nodiscard]] double defaultMeasurementStrength(const size_t qubits) {
  return std::exp2(-static_cast<double>(qubits) / 2.);
}

} // namespace

llvm::FailureOr<WeakMeasurementGrover>
WeakMeasurementGrover::create(WeakMeasurementGroverOptions options) {
  const auto width = options.markedBitstring.size();
  if (width < 2 || width > WeakMeasurementGroverOptions::MAX_QUBITS) {
    return ::mqt::emitError(
        "weak-measurement Grover requires a marked bitstring of width 2 "
        "through 2044",
        ::mqt::ErrorCategory::InvalidArgument);
  }
  if (llvm::failed(detail::validateOutcome(options.markedBitstring, width))) {
    return llvm::failure();
  }

  const auto maximumStrength = defaultMeasurementStrength(width);
  if (!options.measurementStrength) {
    options.measurementStrength = maximumStrength;
  }
  if (!std::isfinite(*options.measurementStrength) ||
      *options.measurementStrength <= 0. ||
      *options.measurementStrength > maximumStrength) {
    return ::mqt::emitError(
        "weak-measurement Grover measurement strength must be finite and in "
        "(0, 2^(-n/2)]",
        ::mqt::ErrorCategory::InvalidArgument);
  }
  return WeakMeasurementGrover(std::move(options));
}

WeakMeasurementGrover::WeakMeasurementGrover(
    WeakMeasurementGroverOptions options)
    : options_(std::move(options)),
      output_{.name = "result", .width = options_.markedBitstring.size()} {}

const WeakMeasurementGroverOptions&
WeakMeasurementGrover::options() const noexcept {
  return options_;
}

size_t WeakMeasurementGrover::qubits() const noexcept { return output_.width; }

const Output& WeakMeasurementGrover::output() const noexcept { return output_; }

llvm::FailureOr<double>
WeakMeasurementGrover::probability(const std::string_view outcome) const {
  if (llvm::failed(detail::validateOutcome(outcome, output_.width))) {
    return llvm::failure();
  }
  return outcome == options_.markedBitstring ? 1. : 0.;
}

llvm::FailureOr<Evaluation>
WeakMeasurementGrover::evaluate(const Counts& counts) const {
  return detail::evaluate(*this, counts, options_.markedBitstring);
}

} // namespace mqt::bench
