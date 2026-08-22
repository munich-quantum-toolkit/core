/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "benchmarks/GHZ.hpp"

#include "EvaluationUtils.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace mqt::benchmarks {

GHZ::GHZ(GHZOptions options)
    : options_(std::move(options)), output_{"result", options_.qubits} {
  if (options_.qubits == 0) {
    throw std::invalid_argument("GHZ requires at least one qubit");
  }
  if (options_.topology != GHZTopology::Linear &&
      options_.topology != GHZTopology::Star) {
    throw std::invalid_argument("unknown GHZ topology");
  }
  if (options_.basis != GHZBasis::Z && options_.basis != GHZBasis::X) {
    throw std::invalid_argument("unknown GHZ measurement basis");
  }
}

const GHZOptions& GHZ::options() const noexcept { return options_; }

const Output& GHZ::output() const noexcept { return output_; }

double GHZ::probability(const std::string_view outcome) const {
  detail::validateOutcome(outcome, output_.width);

  if (options_.basis == GHZBasis::Z) {
    const auto allZero = outcome.find('1') == std::string_view::npos;
    const auto allOne = outcome.find('0') == std::string_view::npos;
    return allZero || allOne ? 0.5 : 0.;
  }

  size_t ones = 0;
  for (const auto bit : outcome) {
    ones += bit == '1' ? 1U : 0U;
  }
  if (ones % 2U != 0U) {
    return 0.;
  }
  if (options_.qubits > 1075) {
    return 0.;
  }
  return std::ldexp(1., 1 - static_cast<int>(options_.qubits));
}

Evaluation GHZ::evaluate(const Counts& counts) const {
  return detail::evaluate(
      output_, counts,
      [this](const std::string_view outcome) { return probability(outcome); });
}

} // namespace mqt::benchmarks
