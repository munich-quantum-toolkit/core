/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/ControlledMultiplicationModuloN.hpp"

#include "EvaluationUtils.hpp"
#include "bench/Evaluation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace mqt::bench {

[[nodiscard]] static bool isBitstring(const std::string_view value) {
  return std::ranges::all_of(
      value, [](const char bit) { return bit == '0' || bit == '1'; });
}

[[nodiscard]] static std::string subtract(const std::string_view minuend,
                                          const std::string_view subtrahend) {
  auto difference = std::string(minuend);
  auto borrow = 0;
  for (size_t index = difference.size(); index > 0; --index) {
    const auto position = index - 1U;
    auto bit =
        (minuend[position] - '0') - (subtrahend[position] - '0') - borrow;
    if (bit < 0) {
      bit += 2;
      borrow = 1;
    } else {
      borrow = 0;
    }
    difference[position] = static_cast<char>('0' + bit);
  }
  return difference;
}

[[nodiscard]] static std::string addModulo(const std::string_view lhs,
                                           const std::string_view rhs,
                                           const std::string_view modulus) {
  const auto width = lhs.size();
  auto sum = std::string(width + 1U, '0');
  auto carry = 0;
  for (size_t index = width; index > 0; --index) {
    const auto position = index - 1U;
    const auto bit = (lhs[position] - '0') + (rhs[position] - '0') + carry;
    sum[index] = static_cast<char>('0' + (bit & 1));
    carry = bit >> 1;
  }
  sum[0] = static_cast<char>('0' + carry);

  const auto extendedModulus = std::string{"0"} + std::string{modulus};
  if (sum >= extendedModulus) {
    sum = subtract(sum, extendedModulus);
  }
  return sum.substr(1);
}

[[nodiscard]] static std::string
multiplyModulo(const std::string_view multiplier,
               const std::string_view multiplicand,
               const std::string_view modulus) {
  auto result = std::string(multiplier.size(), '0');
  for (const auto bit : multiplicand) {
    result = addModulo(result, result, modulus);
    if (bit == '1') {
      result = addModulo(result, multiplier, modulus);
    }
  }
  return result;
}

ControlledMultiplicationModuloN::ControlledMultiplicationModuloN(
    ControlledMultiplicationModuloNOptions options)
    : options_(std::move(options)),
      output_{.name = "result", .width = 2U * options_.multiplier.size() + 2U} {
  const auto width = options_.multiplier.size();
  if (width < 2U || width > ControlledMultiplicationModuloNOptions::MAX_BITS) {
    throw std::invalid_argument(
        "controlled multiplication modulo N inputs must contain between 2 and "
        "63 bits");
  }
  if (options_.modulus.size() != width) {
    throw std::invalid_argument(
        "controlled multiplication modulo N inputs must have equal widths");
  }
  if (!isBitstring(options_.multiplier) || !isBitstring(options_.modulus)) {
    throw std::invalid_argument(
        "controlled multiplication modulo N inputs must contain only '0' and "
        "'1'");
  }
  if (options_.modulus.front() != '1') {
    throw std::invalid_argument(
        "controlled multiplication modulo N modulus must be canonical");
  }
  if (std::ranges::all_of(options_.multiplier,
                          [](const char bit) { return bit == '0'; }) ||
      options_.multiplier >= options_.modulus) {
    throw std::invalid_argument(
        "controlled multiplication modulo N multiplier must satisfy 0 < a < "
        "N");
  }
}

const ControlledMultiplicationModuloNOptions&
ControlledMultiplicationModuloN::options() const noexcept {
  return options_;
}

const Output& ControlledMultiplicationModuloN::output() const noexcept {
  return output_;
}

double ControlledMultiplicationModuloN::probability(
    const std::string_view outcome) const {
  detail::validateOutcome(outcome, output_.width);
  const auto width = options_.multiplier.size();
  const auto control = outcome.front();
  const auto multiplicand = outcome.substr(1U, width);
  const auto accumulator = outcome.substr(width + 1U);
  const auto expected =
      control == '0'
          ? std::string(width + 1U, '0')
          : std::string{"0"} + multiplyModulo(options_.multiplier, multiplicand,
                                              options_.modulus);
  if (accumulator != expected) {
    return 0.;
  }
  return std::ldexp(1., -static_cast<int>(width + 1U));
}

Evaluation
ControlledMultiplicationModuloN::evaluate(const Counts& counts) const {
  return detail::evaluate(
      output_, counts,
      [this](const std::string_view outcome) { return probability(outcome); });
}

} // namespace mqt::bench
