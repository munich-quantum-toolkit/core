/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/WState.hpp"

#include "bench/Evaluation.hpp"

#include "EvaluationUtils.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace mqt::bench {

WState::WState(const WStateOptions options)
    : options_(options), output_{.name = "result", .width = options.qubits} {
  if (options.qubits == 0 ||
      options.qubits >
          static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
      options.qubits - 1 > std::vector<double>{}.max_size()) {
    throw std::invalid_argument("W-state qubits must be positive and fit "
                                "circuit dimensions and angle storage");
  }
}
const WStateOptions& WState::options() const noexcept { return options_; }
const Output& WState::output() const noexcept { return output_; }
double WState::probability(const std::string_view outcome) const {
  detail::validateOutcome(outcome, output_.width);
  return std::ranges::count(outcome, '1') == 1
             ? 1. / static_cast<double>(options_.qubits)
             : 0.;
}
Evaluation WState::evaluate(const Counts& counts) const {
  return detail::evaluate(*this, counts);
}

} // namespace mqt::bench
