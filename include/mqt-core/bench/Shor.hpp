/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "bench/Evaluation.hpp"
#include "bench/mqt_core_bench_export.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <utility>

namespace mqt::bench {

/// Parameters for semiclassical order finding with 2n phase bits and 2n+3
/// qubits.
struct ShorOptions {
  static constexpr uint64_t MAX_NUMBER = (uint64_t{1} << 31U) - 1U;
  uint64_t number;
  uint64_t base = 2;
  /// Largest retained controlled-rotation distance; omitted means exact.
  std::optional<size_t> qftCutoff;
};

/// A nontrivial factor pair, sorted in ascending order.
using FactorPair = std::pair<uint64_t, uint64_t>;

/// Classical verification of measured Shor phases.
struct ShorEvaluation {
  /// Fraction of shots that independently yield a verified factor pair.
  double successProbability = 0.;
  std::optional<FactorPair> factors;
};

/// Structured semiclassical order finding for an odd modulus of at most 31
/// bits.
///
/// The base must satisfy 1 < base < number and be coprime to number.
/// Evaluation uses exact continued fractions and verifies factors by division.
/// It does not compute an ideal phase distribution or the order classically.
class MQT_CORE_BENCH_EXPORT Shor final {
public:
  explicit Shor(ShorOptions options);
  [[nodiscard]] const ShorOptions& options() const noexcept;
  [[nodiscard]] const Output& output() const noexcept;
  [[nodiscard]] ShorEvaluation evaluate(const Counts& counts) const;

private:
  ShorOptions options_;
  Output output_;
};

/// Outcome of a bounded factoring workflow.
enum class FactorStatus : uint8_t { Success, Prime, AttemptsExhausted };

struct FactorOptions {
  size_t maxAttempts = 16;
  uint64_t seed = 0;
  std::optional<size_t> qftCutoff;
};

struct FactorResult {
  FactorStatus status;
  std::optional<FactorPair> factors;
  /// Number of attempted bases; classical prechecks use zero attempts.
  size_t attempts = 0;
};

/// Find one nontrivial factor pair, using the callback for order-finding runs.
///
/// Accepts 2 <= number <= ShorOptions::MAX_NUMBER. Even numbers, primes, and
/// perfect powers are handled classically. Other inputs try base 2 first, then
/// seeded random bases. The callback owns device selection, shots, and
/// execution seeds. Invalid counts and callback errors propagate to the caller.
[[nodiscard]] MQT_CORE_BENCH_EXPORT FactorResult
factor(uint64_t number, const std::function<Counts(const Shor&)>& run,
       const FactorOptions& options = {});

} // namespace mqt::bench
