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

#include "EvaluationUtils.hpp"
#include "bench/Evaluation.hpp"

#include <numbers>
#include <string_view>

namespace mqt::bench {

RepeatUntilSuccess::RepeatUntilSuccess()
    : output_{.name = "result", .width = 1} {}

const Output& RepeatUntilSuccess::output() const noexcept { return output_; }

double RepeatUntilSuccess::probability(const std::string_view outcome) const {
  detail::validateOutcome(outcome, output_.width);
  constexpr auto bias = std::numbers::sqrt2 / 3.;
  return outcome == "0" ? 0.5 + bias : 0.5 - bias;
}

Evaluation RepeatUntilSuccess::evaluate(const Counts& counts) const {
  return detail::evaluate(
      output_, counts,
      [this](const std::string_view outcome) { return probability(outcome); });
}

} // namespace mqt::bench
