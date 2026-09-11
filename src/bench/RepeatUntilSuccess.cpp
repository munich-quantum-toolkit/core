/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/RepeatUntilSuccess.hpp"

#include "bench/Evaluation.hpp"

#include "EvaluationUtils.hpp"

#include <numbers>
#include <stdexcept>
#include <string_view>

namespace mqt::bench {

RepeatUntilSuccess::RepeatUntilSuccess(RepeatUntilSuccessOptions options)
    : options_(options), output_{.name = "result", .width = 1} {
  if (options_.dataQubits == 0 ||
      options_.dataQubits > RepeatUntilSuccessOptions::MAX_DATA_QUBITS) {
    throw std::invalid_argument(
        "repeat-until-success data qubits must be between 1 and 1000000");
  }
}

const RepeatUntilSuccessOptions& RepeatUntilSuccess::options() const noexcept {
  return options_;
}

const Output& RepeatUntilSuccess::output() const noexcept { return output_; }

double RepeatUntilSuccess::probability(const std::string_view outcome) const {
  detail::validateOutcome(outcome, output_.width);
  constexpr auto bias = std::numbers::sqrt2 / 3.;
  return outcome == "0" ? 0.5 + bias : 0.5 - bias;
}

Evaluation RepeatUntilSuccess::evaluate(const Counts& counts) const {
  return detail::evaluate(*this, counts);
}

} // namespace mqt::bench
