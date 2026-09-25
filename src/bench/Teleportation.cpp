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

#include "mlir/Support/LogicalResult.h"

#include <string_view>

namespace mqt::bench {

Teleportation::Teleportation() : output_{.name = "result", .width = 1} {}

const Output& Teleportation::output() const noexcept { return output_; }

mlir::FailureOr<double>
Teleportation::probability(const std::string_view outcome) const {
  if (mlir::failed(detail::validateOutcome(outcome, output_.width))) {
    return mlir::failure();
  }
  return outcome == "0" ? 1. : 0.;
}

mlir::FailureOr<Evaluation>
Teleportation::evaluate(const Counts& counts) const {
  return detail::evaluate(*this, counts, "0");
}

} // namespace mqt::bench
