/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/Multiplexer.hpp"

#include "EvaluationUtils.hpp"
#include "bench/Evaluation.hpp"

#include <stdexcept>
#include <string_view>

namespace mqt::bench {

Multiplexer::Multiplexer(MultiplexerOptions options)
    : options_(options), output_{.name = "result", .width = options_.qubits} {
  if (options_.qubits < 2 || options_.qubits > MultiplexerOptions::MAX_QUBITS) {
    throw std::invalid_argument("multiplexer qubits must be between 2 and 31");
  }
}

const MultiplexerOptions& Multiplexer::options() const noexcept {
  return options_;
}

const Output& Multiplexer::output() const noexcept { return output_; }

double Multiplexer::probability(const std::string_view outcome) const {
  detail::validateOutcome(outcome, output_.width);
  return outcome.find('1') == std::string_view::npos ? 1. : 0.;
}

Evaluation Multiplexer::evaluate(const Counts& counts) const {
  return detail::evaluate(
      output_, counts,
      [this](const std::string_view outcome) { return probability(outcome); });
}

} // namespace mqt::bench
