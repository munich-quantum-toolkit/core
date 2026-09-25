/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/Shor.hpp"

#include "bench/Evaluation.hpp"

#include "EvaluationUtils.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>

namespace mqt::bench {
namespace {

// Products of reduced residues fit because the modulus is at most 31 bits.
[[nodiscard]] uint64_t powerModulo(uint64_t base, uint64_t exponent,
                                   uint64_t modulus) {
  uint64_t result = 1;
  while (exponent != 0) {
    if ((exponent & 1U) != 0) {
      result = (result * base) % modulus;
    }
    base = (base * base) % modulus;
    exponent >>= 1U;
  }
  return result;
}

[[nodiscard]] std::optional<FactorPair> factorPair(uint64_t number,
                                                   uint64_t divisor) {
  if (divisor <= 1 || divisor >= number || number % divisor != 0) {
    return std::nullopt;
  }
  const auto other = number / divisor;
  return FactorPair{std::min(divisor, other), std::max(divisor, other)};
}

[[nodiscard]] std::optional<FactorPair>
recoverFactors(const ShorOptions& options, uint64_t numerator,
               uint64_t denominator) {
  uint64_t previous = 0;
  uint64_t beforePrevious = 1;
  while (denominator != 0) {
    const auto coefficient = numerator / denominator;
    if (previous != 0 &&
        coefficient > (options.number - 1 - beforePrevious) / previous) {
      break;
    }
    const auto candidate = (coefficient * previous) + beforePrevious;
    if (candidate % 2 == 0 &&
        powerModulo(options.base, candidate, options.number) == 1) {
      const auto halfPower =
          powerModulo(options.base, candidate / 2, options.number);
      for (const auto residue : {halfPower - 1, halfPower + 1}) {
        if (const auto factors =
                factorPair(options.number, std::gcd(residue, options.number))) {
          return factors;
        }
      }
    }
    beforePrevious = previous;
    previous = candidate;
    const auto remainder = numerator % denominator;
    numerator = denominator;
    denominator = remainder;
  }
  return std::nullopt;
}

// Odd inputs are at most 31 bits, so at most 23,170 trial divisions suffice.
[[nodiscard]] bool isPrime(uint64_t number) {
  for (uint64_t divisor = 3; divisor * divisor <= number; divisor += 2) {
    if (number % divisor == 0) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] uint64_t boundedPower(uint64_t base, unsigned exponent,
                                    uint64_t limit) {
  uint64_t result = 1;
  for (unsigned i = 0; i < exponent; ++i) {
    if (result > limit / base) {
      return limit + 1;
    }
    result *= base;
  }
  return result;
}

[[nodiscard]] std::optional<FactorPair> perfectPowerFactors(uint64_t number) {
  const auto bits = static_cast<unsigned>(std::bit_width(number));
  for (unsigned exponent = 2; exponent < bits; ++exponent) {
    uint64_t lower = 2;
    uint64_t upper = uint64_t{1} << ((bits + exponent - 1) / exponent);
    while (lower <= upper) {
      const auto base = std::midpoint(lower, upper);
      const auto power = boundedPower(base, exponent, number);
      if (power == number) {
        return factorPair(number, base);
      }
      if (power < number) {
        lower = base + 1;
      } else {
        upper = base - 1;
      }
    }
  }
  return std::nullopt;
}

} // namespace

Shor::Shor(ShorOptions options)
    : options_(options),
      output_{
          .name = "result",
          .width = size_t{2} * std::bit_width(options_.number),
      } {
  if (options_.number < 3 || options_.number > ShorOptions::MAX_NUMBER ||
      options_.number % 2 == 0) {
    throw std::invalid_argument(
        "Shor number must be odd and between 3 and 2^31 - 1");
  }
  if (options_.base <= 1 || options_.base >= options_.number ||
      std::gcd(options_.base, options_.number) != 1) {
    throw std::invalid_argument(
        "Shor base must satisfy 1 < base < number and be coprime to number");
  }
}

const ShorOptions& Shor::options() const noexcept { return options_; }
const Output& Shor::output() const noexcept { return output_; }

ShorEvaluation Shor::evaluate(const Counts& counts) const {
  const auto total = detail::validateCounts(output_, counts);
  size_t successes = 0;
  ShorEvaluation result;
  for (const auto& [outcome, count] : counts) {
    if (count == 0) {
      continue;
    }
    uint64_t phase = 0;
    std::from_chars(outcome.data(), std::to_address(outcome.end()), phase, 2);
    const auto factors =
        recoverFactors(options_, phase, uint64_t{1} << output_.width);
    if (factors) {
      successes += count;
      if (!result.factors) {
        result.factors = factors;
      }
    }
  }
  result.successProbability =
      static_cast<double>(successes) / static_cast<double>(total);
  return result;
}

FactorResult factor(uint64_t number,
                    const std::function<Counts(const Shor&)>& run,
                    const FactorOptions& options) {
  if (number < 2 || number > ShorOptions::MAX_NUMBER) {
    throw std::invalid_argument(
        "factoring number must be between 2 and 2^31 - 1");
  }
  if (options.maxAttempts == 0 || !run) {
    throw std::invalid_argument(
        "factoring requires a callback and a positive attempt limit");
  }
  if (number == 2) {
    return {.status = FactorStatus::Prime, .factors = std::nullopt};
  }
  if (number % 2 == 0) {
    return {.status = FactorStatus::Success, .factors = factorPair(number, 2)};
  }
  if (isPrime(number)) {
    return {.status = FactorStatus::Prime, .factors = std::nullopt};
  }
  if (const auto factors = perfectPowerFactors(number)) {
    return {.status = FactorStatus::Success, .factors = factors};
  }

  std::mt19937_64 random(options.seed);
  std::uniform_int_distribution<uint64_t> bases(2, number - 2);
  for (size_t attempt = 0; attempt < options.maxAttempts; ++attempt) {
    const auto base = attempt == 0 ? uint64_t{2} : bases(random);
    if (const auto factors = factorPair(number, std::gcd(base, number))) {
      return {
          .status = FactorStatus::Success,
          .factors = factors,
          .attempts = attempt + 1,
      };
    }
    const Shor benchmark({.number = number, .base = base});
    if (const auto evaluation = benchmark.evaluate(run(benchmark));
        evaluation.factors) {
      return {
          .status = FactorStatus::Success,
          .factors = evaluation.factors,
          .attempts = attempt + 1,
      };
    }
  }
  return {
      .status = FactorStatus::AttemptsExhausted,
      .factors = std::nullopt,
      .attempts = options.maxAttempts,
  };
}

} // namespace mqt::bench
