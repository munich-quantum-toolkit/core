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
#include "support/Diagnostics.hpp"

#include "llvm/Support/LogicalResult.h"

#include <numbers>
#include <string_view>

namespace mqt::bench {

llvm::FailureOr<RepeatUntilSuccess>
RepeatUntilSuccess::create(RepeatUntilSuccessOptions options) {
  if (options.dataQubits == 0 ||
      options.dataQubits > RepeatUntilSuccessOptions::MAX_DATA_QUBITS) {
    return ::mqt::emitError(
        "repeat-until-success data qubits must be between 1 and 1000000",
        ::mqt::ErrorCategory::InvalidArgument);
  }
  return RepeatUntilSuccess(options);
}

RepeatUntilSuccess::RepeatUntilSuccess(RepeatUntilSuccessOptions options)
    : options_(options), output_{.name = "result", .width = 1} {}

const RepeatUntilSuccessOptions& RepeatUntilSuccess::options() const noexcept {
  return options_;
}

const Output& RepeatUntilSuccess::output() const noexcept { return output_; }

llvm::FailureOr<double>
RepeatUntilSuccess::probability(const std::string_view outcome) const {
  if (llvm::failed(detail::validateOutcome(outcome, output_.width))) {
    return llvm::failure();
  }
  constexpr auto bias = std::numbers::sqrt2 / 3.;
  return outcome == "0" ? 0.5 + bias : 0.5 - bias;
}

llvm::FailureOr<Evaluation>
RepeatUntilSuccess::evaluate(const Counts& counts) const {
  return detail::evaluate(*this, counts);
}

} // namespace mqt::bench
