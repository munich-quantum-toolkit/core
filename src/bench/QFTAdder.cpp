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

#include "bench/Error.hpp"
#include "bench/Evaluation.hpp"

#include "EvaluationUtils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
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

Result<QFTAdder> QFTAdder::create(QFTAdderOptions options) {
  if (options.method != QFTAdderMethod::Register &&
      options.method != QFTAdderMethod::Constant) {
    return Error{.message = "QFT adder method must be register or constant"};
  }
  if (options.overflow != QFTAdderOverflow::Wrap &&
      options.overflow != QFTAdderOverflow::Carry) {
    return Error{.message = "QFT adder overflow must be wrap or carry"};
  }
  const auto width = options.addend.size();
  const auto carry = options.overflow == QFTAdderOverflow::Carry;
  if (width == 0 ||
      width > QFTAdderOptions::MAX_QUBITS - static_cast<size_t>(carry) ||
      options.accumulator.size() != width) {
    return Error{
        .message = "QFT adder operands must have equal nonzero "
                   "width, with at most 1024 sum bits",
    };
  }
  const auto isRegister = options.method == QFTAdderMethod::Register;
  if (options.addend.find_first_not_of(isRegister ? "01+" : "01") !=
          std::string::npos ||
      options.accumulator.find_first_not_of("01") != std::string::npos) {
    return Error{
        .message = "QFT adder operands must be binary; only "
                   "register addends may contain '+'",
    };
  }
  return QFTAdder(std::move(options));
}

QFTAdder::QFTAdder(QFTAdderOptions options)
    : options_(std::move(options)), output_{.name = "result", .width = 0} {
  const auto width = options_.addend.size();
  const auto carry = options_.overflow == QFTAdderOverflow::Carry;
  const auto isRegister = options_.method == QFTAdderMethod::Register;
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

Result<double> QFTAdder::probability(const std::string_view outcome) const {
  if (auto error = detail::validateOutcome(outcome, output_.width)) {
    return std::move(*error);
  }
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

Result<Evaluation> QFTAdder::evaluate(const Counts& counts) const {
  return detail::evaluate(*this, counts, expectedResult_);
}

} // namespace mqt::bench
