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

#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <system_error>

namespace mqt::bench {

WState::WState(const WStateOptions options)
    : options_(options), output_{.name = "result", .width = options.qubits} {}

llvm::Expected<WState> WState::create(WStateOptions options) {
  if (options.qubits == 0 ||
      options.qubits >
          static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "W-state qubits must be positive and fit "
                                   "circuit dimensions");
  }
  return WState(options);
}
const WStateOptions& WState::options() const noexcept { return options_; }
const Output& WState::output() const noexcept { return output_; }
llvm::Expected<double>
WState::probability(const std::string_view outcome) const {
  if (auto error = detail::validateOutcome(outcome, output_.width)) {
    return error;
  }
  return std::ranges::count(outcome, '1') == 1
             ? 1. / static_cast<double>(options_.qubits)
             : 0.;
}
llvm::Expected<Evaluation> WState::evaluate(const Counts& counts) const {
  return detail::evaluate(*this, counts);
}

} // namespace mqt::bench
