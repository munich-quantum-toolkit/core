/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/QFTAdder.hpp"

#include "bench/Evaluation.hpp"

#include "EvaluationUtils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace mqt::bench {
namespace {

[[nodiscard]] std::string sumBits(const std::string_view addend,
                                  const std::string_view accumulator,
                                  const QFTAdderOverflow overflow) {
  const auto offset = overflow == QFTAdderOverflow::Carry ? 1U : 0U;
  auto result = std::string(addend.size() + offset, '0');
  auto carry = 0;
  for (size_t i = addend.size(); i > 0; --i) {
    const auto sum = (addend[i - 1] - '0') + (accumulator[i - 1] - '0') + carry;
    result[i - 1 + offset] = static_cast<char>('0' + (sum % 2));
    carry = sum / 2;
  }
  if (offset != 0) {
    result[0] = static_cast<char>('0' + carry);
  }
  return result;
}

} // namespace

QFTAdder::QFTAdder(QFTAdderOptions options)
    : options_(std::move(options)), output_{.name = "result", .width = 0} {
  if (options_.method != QFTAdderMethod::Register &&
      options_.method != QFTAdderMethod::Constant) {
    throw std::invalid_argument(
        "QFT adder method must be register or constant");
  }
  if (options_.overflow != QFTAdderOverflow::Wrap &&
      options_.overflow != QFTAdderOverflow::Carry) {
    throw std::invalid_argument("QFT adder overflow must be wrap or carry");
  }
  const auto width = options_.addend.size();
  const auto carry = options_.overflow == QFTAdderOverflow::Carry;
  if (width == 0 ||
      width > QFTAdderOptions::MAX_QUBITS - static_cast<size_t>(carry) ||
      options_.accumulator.size() != width) {
    throw std::invalid_argument("QFT adder operands must have equal nonzero "
                                "width, with at most 1024 sum bits");
  }
  const auto isRegister = options_.method == QFTAdderMethod::Register;
  if (options_.addend.find_first_not_of(isRegister ? "01+" : "01") !=
          std::string::npos ||
      options_.accumulator.find_first_not_of("01") != std::string::npos) {
    throw std::invalid_argument("QFT adder operands must be binary; only "
                                "register addends may contain '+'");
  }
  output_.width =
      width + static_cast<size_t>(carry) + (isRegister ? width : 0U);
  if (options_.addend.find('+') == std::string::npos) {
    expectedResult_ =
        (isRegister ? options_.addend : std::string{}) +
        sumBits(options_.addend, options_.accumulator, options_.overflow);
  }
}

const QFTAdderOptions& QFTAdder::options() const noexcept { return options_; }

const Output& QFTAdder::output() const noexcept { return output_; }

const std::optional<std::string>& QFTAdder::expectedResult() const noexcept {
  return expectedResult_;
}

double QFTAdder::probability(const std::string_view outcome) const {
  detail::validateOutcome(outcome, output_.width);
  if (expectedResult_) {
    return outcome == *expectedResult_ ? 1. : 0.;
  }
  const auto addend = outcome.substr(0, options_.addend.size());
  for (size_t i = 0; i < addend.size(); ++i) {
    if (options_.addend[i] != '+' && options_.addend[i] != addend[i]) {
      return 0.;
    }
  }
  if (outcome.substr(addend.size()) !=
      sumBits(addend, options_.accumulator, options_.overflow)) {
    return 0.;
  }
  return std::ldexp(
      1., -static_cast<int>(std::ranges::count(options_.addend, '+')));
}

Evaluation QFTAdder::evaluate(const Counts& counts) const {
  return detail::evaluate(*this, counts, expectedResult_);
}

} // namespace mqt::bench
