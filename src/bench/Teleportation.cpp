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

#include "bench/Evaluation.hpp"

#include "EvaluationUtils.hpp"

#include <string_view>

namespace mqt::bench {

Teleportation::Teleportation() : output_{.name = "result", .width = 1} {}

const Output& Teleportation::output() const noexcept { return output_; }

double Teleportation::probability(const std::string_view outcome) const {
  detail::validateOutcome(outcome, output_.width);
  return outcome == "0" ? 1. : 0.;
}

Evaluation Teleportation::evaluate(const Counts& counts) const {
  return detail::evaluate(*this, counts, "0");
}

} // namespace mqt::bench
