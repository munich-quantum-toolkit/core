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

static_assert(ControlledMultiplicationModuloNOptions::MAX_BITS <
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

ControlledMultiplicationModuloN::ControlledMultiplicationModuloN(
    ControlledMultiplicationModuloNOptions options)
    : options_(std::move(options)),
      output_{
          .name = "result",
          .width = (2U * options_.multiplier.size()) + 2U,
      } {
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
  const auto multiplier = binaryValue(options_.multiplier);
  const auto modulus = binaryValue(options_.modulus);
  if (!multiplier || !modulus) {
    throw std::invalid_argument(
        "controlled multiplication modulo N inputs must contain only '0' and "
        "'1'");
  }
  if (options_.modulus.front() != '1') {
    throw std::invalid_argument(
        "controlled multiplication modulo N modulus must be canonical");
  }
  if (*multiplier == 0 || *multiplier >= *modulus) {
    throw std::invalid_argument(
        "controlled multiplication modulo N multiplier must satisfy 0 < a < "
        "N");
  }
  multiplierValue_ = *multiplier;
  modulusValue_ = *modulus;
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
          ? uint64_t{0}
          : multiplyModulo(multiplierValue_, binaryValue(multiplicand).value(),
                           modulusValue_);
  if (binaryValue(accumulator).value() != expected) {
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
