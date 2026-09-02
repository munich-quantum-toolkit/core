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

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string_view>

namespace mqt::bench {

Multiplexer::Multiplexer(MultiplexerOptions options)
    : options_(options), output_{.name = "result", .width = options_.qubits} {
  if (options_.qubits < 2 || options_.qubits > MultiplexerOptions::MAX_QUBITS) {
    throw std::invalid_argument(
        "multiplexer qubits must be between 2 and 1024");
  }
}

const MultiplexerOptions& Multiplexer::options() const noexcept {
  return options_;
}

const Output& Multiplexer::output() const noexcept { return output_; }

double Multiplexer::probability(const std::string_view outcome) const {
  detail::validateOutcome(outcome, output_.width);

  double state = 0.;
  double weight = 0.5;
  for (const auto bit : outcome.substr(0, outcome.size() - 1)) {
    state += bit == '1' ? weight : 0.;
    weight *= 0.5;
  }
  const auto angle = std::numbers::pi * state;
  const auto amplitude =
      outcome.back() == '0' ? std::cos(angle / 2.) : std::sin(angle / 2.);
  return std::ldexp(amplitude * amplitude,
                    1 - static_cast<int>(options_.qubits));
}

Evaluation Multiplexer::evaluate(const Counts& counts) const {
  return detail::evaluate(
      output_, counts,
      [this](const std::string_view outcome) { return probability(outcome); });
}

} // namespace mqt::bench
