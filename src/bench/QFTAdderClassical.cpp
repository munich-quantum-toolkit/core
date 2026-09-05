/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/QFTAdderClassical.hpp"

#include "EvaluationUtils.hpp"
#include "bench/Evaluation.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace mqt::bench {

[[nodiscard]] static std::string increment(const std::string_view addend) {
  auto result = std::string{"0"} + std::string{addend};
  auto carry = true;
  for (size_t index = result.size(); index > 0 && carry; --index) {
    auto& bit = result[index - 1];
    carry = bit == '1';
    bit = carry ? '0' : '1';
  }
  return result;
}

QFTAdderClassical::QFTAdderClassical(QFTAdderClassicalOptions options)
    : options_(std::move(options)),
      output_{.name = "result", .width = options_.addend.size() + 1U} {
  const auto width = options_.addend.size();
  if (width == 0 || width > QFTAdderClassicalOptions::MAX_ADDEND_BITS) {
    throw std::invalid_argument(
        "classical QFT adder addend must contain between 1 and 1023 bits");
  }
  if (!std::ranges::all_of(options_.addend, [](const char bit) {
        return bit == '0' || bit == '1';
      })) {
    throw std::invalid_argument(
        "classical QFT adder addend must contain only '0' and '1'");
  }
  expectedResult_ = increment(options_.addend);
}

const QFTAdderClassicalOptions& QFTAdderClassical::options() const noexcept {
  return options_;
}

const Output& QFTAdderClassical::output() const noexcept { return output_; }

const std::string& QFTAdderClassical::expectedResult() const noexcept {
  return expectedResult_;
}

double QFTAdderClassical::probability(const std::string_view outcome) const {
  detail::validateOutcome(outcome, output_.width);
  return outcome == expectedResult_ ? 1. : 0.;
}

Evaluation QFTAdderClassical::evaluate(const Counts& counts) const {
  return detail::evaluate(
      output_, counts,
      [this](const std::string_view outcome) { return probability(outcome); },
      expectedResult_);
}

} // namespace mqt::bench
