/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/FanOut.hpp"

#include "bench/Evaluation.hpp"

#include "EvaluationUtils.hpp"

#include "llvm/Support/Error.h"

#include <algorithm>
#include <cmath>
#include <string_view>
#include <system_error>

namespace mqt::bench {

FanOut::FanOut(FanOutOptions options)
    : options_(options),
      output_{.name = "result", .width = options_.controls + 1} {}

llvm::Expected<FanOut> FanOut::create(FanOutOptions options) {
  if (options.controls == 0 || options.controls > FanOutOptions::MAX_CONTROLS) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "fan-out controls must be between 1 and 1073");
  }
  return FanOut(options);
}

const FanOutOptions& FanOut::options() const noexcept { return options_; }

const Output& FanOut::output() const noexcept { return output_; }

llvm::Expected<double>
FanOut::probability(const std::string_view outcome) const {
  if (auto error = detail::validateOutcome(outcome, output_.width)) {
    return error;
  }

  const auto weight =
      std::ranges::count(outcome.substr(0, options_.controls), '1') % 4;
  const auto exponent = -static_cast<int>(options_.controls);
  if (weight % 2 != 0) {
    return std::ldexp(1., exponent - 1);
  }
  const auto expectedTarget = weight == 0 ? '0' : '1';
  return outcome.back() == expectedTarget ? std::ldexp(1., exponent) : 0.;
}

llvm::Expected<Evaluation> FanOut::evaluate(const Counts& counts) const {
  return detail::evaluate(*this, counts);
}

} // namespace mqt::bench
