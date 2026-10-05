/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/MagicStateDistillation.hpp"

#include "bench/Evaluation.hpp"

#include "EvaluationUtils.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace mqt::bench {

MagicStateDistillation::MagicStateDistillation(
    MagicStateDistillationOptions options)
    : options_(options), output_{.name = "result", .width = 2} {
  if (options_.levels == 0 ||
      options_.levels >
          static_cast<size_t>(std::numeric_limits<int64_t>::max()) / 5) {
    throw std::invalid_argument("magic-state-distillation levels must be "
                                "positive and fit circuit dimensions");
  }
}

const MagicStateDistillationOptions&
MagicStateDistillation::options() const noexcept {
  return options_;
}

const Output& MagicStateDistillation::output() const noexcept {
  return output_;
}

double
MagicStateDistillation::probability(const std::string_view outcome) const {
  detail::validateOutcome(outcome, output_.width);
  return outcome == "00" ? 1. : 0.;
}

Evaluation MagicStateDistillation::evaluate(const Counts& counts) const {
  return detail::evaluate(*this, counts, "00");
}

} // namespace mqt::bench
