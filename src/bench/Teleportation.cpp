/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/Teleportation.hpp"

#include "EvaluationUtils.hpp"
#include "bench/Evaluation.hpp"

#include <string_view>

namespace mqt::bench {

Teleportation::Teleportation() : output_{.name = "result", .width = 3} {}

const Output& Teleportation::output() const noexcept { return output_; }

double Teleportation::probability(const std::string_view outcome) const {
  detail::validateOutcome(outcome, output_.width);
  return 1. / 8.;
}

Evaluation Teleportation::evaluate(const Counts& counts) const {
  return detail::evaluate(
      output_, counts,
      [this](const std::string_view outcome) { return probability(outcome); });
}

} // namespace mqt::bench
