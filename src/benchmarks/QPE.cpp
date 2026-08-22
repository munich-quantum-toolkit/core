/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "benchmarks/QPE.hpp"

#include "EvaluationUtils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace mqt::benchmarks {
namespace {

struct SignedMagnitude {
  bool negative = false;
  std::string magnitude;
};

[[nodiscard]] SignedMagnitude subtractModulo(const std::string_view left,
                                             const std::string_view right) {
  std::string difference(left.size(), '0');
  unsigned int borrow = 0;
  for (size_t index = left.size(); index > 0; --index) {
    auto bit = static_cast<int>(left[index - 1] - '0') -
               static_cast<int>(right[index - 1] - '0') -
               static_cast<int>(borrow);
    if (bit < 0) {
      bit += 2;
      borrow = 1;
    } else {
      borrow = 0;
    }
    difference[index - 1] = static_cast<char>('0' + bit);
  }

  if (difference.find('1') == std::string::npos) {
    return {false, std::move(difference)};
  }
  if (difference.front() == '0') {
    return {false, std::move(difference)};
  }

  unsigned int carry = 1;
  for (size_t index = difference.size(); index > 0; --index) {
    const auto bit = static_cast<unsigned int>(difference[index - 1] - '0');
    const auto complement = (1U - bit) + carry;
    difference[index - 1] = static_cast<char>('0' + (complement & 1U));
    carry = complement >> 1U;
  }
  return {true, std::move(difference)};
}

[[nodiscard]] size_t bitLength(const std::string_view bits) {
  const auto first = bits.find('1');
  return first == std::string_view::npos ? 0 : bits.size() - first;
}

[[nodiscard]] long double toLongDouble(const std::string_view bits) {
  long double value = 0.L;
  for (const auto bit : bits) {
    value = value * 2.L + static_cast<long double>(bit - '0');
  }
  return value;
}

[[nodiscard]] long double sinc(const long double value) {
  if (value == 0.L) {
    return 1.L;
  }
  const auto argument = std::numbers::pi_v<long double> * value;
  return std::sin(argument) / argument;
}

} // namespace

Phase::Phase(uint64_t numerator, const uint64_t denominator) {
  if (denominator == 0) {
    throw std::invalid_argument("phase denominator must not be zero");
  }
  numerator %= denominator;
  const auto divisor = std::gcd(numerator, denominator);
  numerator_ = numerator / divisor;
  denominator_ = denominator / divisor;
}

uint64_t Phase::numerator() const noexcept { return numerator_; }

uint64_t Phase::denominator() const noexcept { return denominator_; }

QPE::QPE(QPEOptions options)
    : options_(std::move(options)), output_{"result", options_.precision},
      lowerOutcome_(options_.precision, '0'),
      scaledRemainder_(options_.phase.numerator()) {
  if (options_.precision == 0) {
    throw std::invalid_argument("QPE precision must be positive");
  }
  if (options_.method != QPEMethod::Standard &&
      options_.method != QPEMethod::Iterative) {
    throw std::invalid_argument("unknown QPE method");
  }

  const auto denominator = options_.phase.denominator();
  for (auto& bit : lowerOutcome_) {
    if (scaledRemainder_ >= denominator - scaledRemainder_) {
      bit = '1';
      scaledRemainder_ -= denominator - scaledRemainder_;
    } else {
      scaledRemainder_ += scaledRemainder_;
    }
  }
}

const QPEOptions& QPE::options() const noexcept { return options_; }

const Output& QPE::output() const noexcept { return output_; }

double QPE::probability(const std::string_view outcome) const {
  detail::validateOutcome(outcome, output_.width);
  const auto difference = subtractModulo(lowerOutcome_, outcome);
  const auto integerBits = bitLength(difference.magnitude);
  if (integerBits > 600) {
    return 0.;
  }

  const auto integerMagnitude = toLongDouble(difference.magnitude);
  const auto denominator = options_.phase.denominator();
  long double magnitude = 0.L;
  if (!difference.negative) {
    magnitude = integerMagnitude + static_cast<long double>(scaledRemainder_) /
                                       static_cast<long double>(denominator);
  } else if (integerMagnitude == 1.L) {
    magnitude = static_cast<long double>(denominator - scaledRemainder_) /
                static_cast<long double>(denominator);
  } else {
    magnitude = integerMagnitude - 1.L +
                static_cast<long double>(denominator - scaledRemainder_) /
                    static_cast<long double>(denominator);
  }

  if (magnitude == 0.L) {
    return 1.;
  }

  const auto reflectedRemainder =
      std::min(scaledRemainder_, denominator - scaledRemainder_);
  const auto sine = std::sin(std::numbers::pi_v<long double> *
                             static_cast<long double>(reflectedRemainder) /
                             static_cast<long double>(denominator));
  const auto numerator = sine / (std::numbers::pi_v<long double> * magnitude);

  long double scaledMagnitude = 0.L;
  if (options_.precision <=
      static_cast<size_t>(std::numeric_limits<int>::max())) {
    scaledMagnitude =
        std::ldexp(magnitude, -static_cast<int>(options_.precision));
  }
  const auto denominatorFactor = sinc(scaledMagnitude);
  const auto ratio = numerator / denominatorFactor;
  return static_cast<double>(std::clamp(ratio * ratio, 0.L, 1.L));
}

Evaluation QPE::evaluate(const Counts& counts) const {
  return detail::evaluate(
      output_, counts,
      [this](const std::string_view outcome) { return probability(outcome); });
}

} // namespace mqt::benchmarks
