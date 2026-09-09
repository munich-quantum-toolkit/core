/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/ModularMultiplier.hpp"

#include "EvaluationUtils.hpp"
#include "bench/Evaluation.hpp"

#include <algorithm>
#include <bitset>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>

namespace mqt::bench {
namespace {

static_assert(ModularMultiplierOptions::MAX_BITS <
              static_cast<size_t>(std::numeric_limits<uint64_t>::digits));

[[nodiscard]] std::optional<uint64_t>
binaryValue(const std::string_view bitstring) {
  uint64_t value = 0;
  const auto* const begin = std::to_address(bitstring.begin());
  const auto* const end = std::to_address(bitstring.end());
  const auto [parsedEnd, error] =
      std::from_chars(begin, end, value, /*base=*/2);
  if (error != std::errc{} || parsedEnd != end) {
    return std::nullopt;
  }
  return value;
}

[[nodiscard]] uint64_t multiplyModulo(const uint64_t multiplier,
                                      uint64_t multiplicand,
                                      const uint64_t modulus) {
  uint64_t result = 0;
  auto addend = multiplier;
  while (multiplicand != 0) {
    if ((multiplicand & uint64_t{1}) != 0) {
      result = (result + addend) % modulus;
    }
    addend = (addend + addend) % modulus;
    multiplicand >>= 1U;
  }
  return result;
}

} // namespace

ModularMultiplier::ModularMultiplier(ModularMultiplierOptions options)
    : options_(std::move(options)),
      output_{
          .name = "result",
          .width = (2U * options_.multiplier.size()) + 2U,
      } {
  const auto width = options_.multiplier.size();
  if (width < 2U || width > ModularMultiplierOptions::MAX_BITS) {
    throw std::invalid_argument(
        "modular multiplier inputs must contain between 2 and "
        "63 bits");
  }
  if (options_.modulus.size() != width) {
    throw std::invalid_argument(
        "modular multiplier inputs must have equal widths");
  }
  const auto multiplier = binaryValue(options_.multiplier);
  const auto modulus = binaryValue(options_.modulus);
  if (!multiplier || !modulus) {
    throw std::invalid_argument(
        "modular multiplier inputs must contain only '0' and "
        "'1'");
  }
  if (options_.modulus.front() != '1') {
    throw std::invalid_argument("modular multiplier modulus must be canonical");
  }
  if (*multiplier == 0 || *multiplier >= *modulus) {
    throw std::invalid_argument(
        "modular multiplier multiplier must satisfy 0 < a < "
        "N");
  }
  if (options_.multiplicand.size() != width ||
      options_.multiplicand.find_first_not_of("01+") != std::string::npos ||
      std::string_view("01+").find(options_.control) ==
          std::string_view::npos) {
    throw std::invalid_argument(
        "modular multiplier multiplicand must match the input width and "
        "multiplicand and control must contain only '0', '1', or '+'");
  }
  if (options_.control != '+' &&
      options_.multiplicand.find('+') == std::string::npos) {
    const auto product =
        options_.control == '0'
            ? uint64_t{0}
            : multiplyModulo(*multiplier,
                             binaryValue(options_.multiplicand).value(),
                             *modulus);
    expectedResult_ = options_.control + options_.multiplicand +
                      std::bitset<64>(product).to_string().substr(63U - width);
  }
  multiplierValue_ = *multiplier;
  modulusValue_ = *modulus;
}

const ModularMultiplierOptions& ModularMultiplier::options() const noexcept {
  return options_;
}

const Output& ModularMultiplier::output() const noexcept { return output_; }

const std::optional<std::string>&
ModularMultiplier::expectedResult() const noexcept {
  return expectedResult_;
}

double ModularMultiplier::probability(const std::string_view outcome) const {
  detail::validateOutcome(outcome, output_.width);
  const auto width = options_.multiplier.size();
  const auto control = outcome.front();
  const auto multiplicand = outcome.substr(1U, width);
  const auto accumulator = outcome.substr(width + 1U);
  if (options_.control != '+' && control != options_.control) {
    return 0.;
  }
  for (size_t i = 0; i < width; ++i) {
    if (options_.multiplicand[i] != '+' &&
        options_.multiplicand[i] != multiplicand[i]) {
      return 0.;
    }
  }
  const auto expected =
      control == '0'
          ? uint64_t{0}
          : multiplyModulo(multiplierValue_, binaryValue(multiplicand).value(),
                           modulusValue_);
  if (binaryValue(accumulator).value() != expected) {
    return 0.;
  }
  return std::ldexp(
      1., -static_cast<int>(std::ranges::count(options_.multiplicand, '+') +
                            (options_.control == '+' ? 1 : 0)));
}

Evaluation ModularMultiplier::evaluate(const Counts& counts) const {
  auto result = detail::evaluate(*this, counts);
  size_t totalShots = 0;
  size_t successShots = 0;
  for (const auto& [outcome, count] : counts) {
    totalShots += count;
    if (probability(outcome) > 0.) {
      successShots += count;
    }
  }
  result.successProbability =
      static_cast<double>(successShots) / static_cast<double>(totalShots);
  return result;
}

} // namespace mqt::bench
