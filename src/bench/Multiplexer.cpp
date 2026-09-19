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

#include "bench/Error.hpp"
#include "bench/Evaluation.hpp"

#include "EvaluationUtils.hpp"

#include <cmath>
#include <numbers>
#include <string_view>
#include <utility>

namespace mqt::bench {

Result<Multiplexer> Multiplexer::create(MultiplexerOptions options) {
  if (options.qubits < 2 || options.qubits > MultiplexerOptions::MAX_QUBITS) {
    return Error{.message = "multiplexer qubits must be between 2 and 1024"};
  }
  return Multiplexer(options);
}

Multiplexer::Multiplexer(MultiplexerOptions options)
    : options_(options), output_{.name = "result", .width = options_.qubits} {}

const MultiplexerOptions& Multiplexer::options() const noexcept {
  return options_;
}

const Output& Multiplexer::output() const noexcept { return output_; }

Result<double> Multiplexer::probability(const std::string_view outcome) const {
  if (auto error = detail::validateOutcome(outcome, output_.width)) {
    return std::move(*error);
  }

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

Result<Evaluation> Multiplexer::evaluate(const Counts& counts) const {
  return detail::evaluate(*this, counts);
}

} // namespace mqt::bench
