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
#include "support/Diagnostics.hpp"

#include "mlir/Support/LogicalResult.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

namespace mqt::bench {

mlir::FailureOr<WState> WState::create(const WStateOptions options) {
  if (options.qubits == 0 ||
      options.qubits >
          static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    return ::mqt::emitError("W-state qubits must be positive and fit "
                            "circuit dimensions",
                            ::mqt::ErrorCategory::InvalidArgument);
  }
  return WState(options);
}
WState::WState(const WStateOptions options)
    : options_(options), output_{.name = "result", .width = options.qubits} {}
const WStateOptions& WState::options() const noexcept { return options_; }
const Output& WState::output() const noexcept { return output_; }
mlir::FailureOr<double>
WState::probability(const std::string_view outcome) const {
  if (mlir::failed(detail::validateOutcome(outcome, output_.width))) {
    return mlir::failure();
  }
  return std::ranges::count(outcome, '1') == 1
             ? 1. / static_cast<double>(options_.qubits)
             : 0.;
}
mlir::FailureOr<Evaluation> WState::evaluate(const Counts& counts) const {
  return detail::evaluate(*this, counts);
}

} // namespace mqt::bench
