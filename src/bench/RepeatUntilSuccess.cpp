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

#include "llvm/Support/Error.h"

#include <numbers>
#include <string_view>
#include <system_error>

namespace mqt::bench {

RepeatUntilSuccess::RepeatUntilSuccess(RepeatUntilSuccessOptions options)
    : options_(options), output_{.name = "result", .width = 1} {}

llvm::Expected<RepeatUntilSuccess>
RepeatUntilSuccess::create(RepeatUntilSuccessOptions options) {
  if (options.dataQubits == 0 ||
      options.dataQubits > RepeatUntilSuccessOptions::MAX_DATA_QUBITS) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "repeat-until-success data qubits must be between 1 and 1000000");
  }
  return RepeatUntilSuccess(options);
}

const RepeatUntilSuccessOptions& RepeatUntilSuccess::options() const noexcept {
  return options_;
}

const Output& RepeatUntilSuccess::output() const noexcept { return output_; }

llvm::Expected<double>
RepeatUntilSuccess::probability(const std::string_view outcome) const {
  if (auto error = detail::validateOutcome(outcome, output_.width)) {
    return error;
  }
  constexpr auto bias = std::numbers::sqrt2 / 3.;
  return outcome == "0" ? 0.5 + bias : 0.5 - bias;
}

llvm::Expected<Evaluation>
RepeatUntilSuccess::evaluate(const Counts& counts) const {
  return detail::evaluate(*this, counts);
}

} // namespace mqt::bench
