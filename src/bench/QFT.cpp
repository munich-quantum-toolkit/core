/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/QFT.hpp"

#include "EvaluationUtils.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace mqt::bench {

QFT::QFT(QFTOptions options)
    : options_(std::move(options)), output_{"result", options_.qubits} {
  if (options_.qubits == 0 || options_.qubits > QFTOptions::MAX_QUBITS) {
    throw std::invalid_argument("QFT qubits must be between 1 and 1000000");
  }
  if (options_.periodExponent > options_.qubits ||
      options_.periodExponent > QFTOptions::MAX_PERIOD_EXPONENT) {
    throw std::invalid_argument(
        "QFT period exponent must be at most the qubit count and 1074");
  }
  if (options_.method != QFTMethod::Standard &&
      options_.method != QFTMethod::Semiclassical) {
    throw std::invalid_argument("unknown QFT method");
  }
}

const QFTOptions& QFT::options() const noexcept { return options_; }

const Output& QFT::output() const noexcept { return output_; }

double QFT::probability(const std::string_view outcome) const {
  detail::validateOutcome(outcome, output_.width);
  if (!std::ranges::all_of(outcome.substr(options_.periodExponent),
                           [](const char bit) { return bit == '0'; })) {
    return 0.;
  }
  return std::ldexp(1., -static_cast<int>(options_.periodExponent));
}

Evaluation QFT::evaluate(const Counts& counts) const {
  return detail::evaluate(
      output_, counts,
      [this](const std::string_view outcome) { return probability(outcome); });
}

} // namespace mqt::bench
