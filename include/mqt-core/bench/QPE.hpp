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
#include <string>
#include <string_view>

namespace mqt::bench {

/// An exact phase in turns, reduced modulo one turn.
class MQT_CORE_BENCH_EXPORT Phase final {
public:
  /// Construct `numerator / denominator` turns.
  Phase(uint64_t numerator, uint64_t denominator);

  [[nodiscard]] uint64_t numerator() const noexcept;
  [[nodiscard]] uint64_t denominator() const noexcept;

  friend bool operator==(const Phase&, const Phase&) = default;

private:
  uint64_t numerator_;
  uint64_t denominator_;
};

/// Circuit method used for QPE.
enum class QPEMethod : uint8_t {
  /// Estimate all phase bits with one query qubit per bit.
  Standard,
  /// Measure and reset one reused query qubit, with feed-forward corrections.
  Iterative
};

/// Parameters for one quantum phase-estimation benchmark instance.
struct QPEOptions {
  static constexpr size_t MAX_PRECISION = 1'000'000;

  /// Number of measured phase bits. Must be in `[1, MAX_PRECISION]`.
  size_t precision;
  /// Eigenphase in turns.
  Phase phase;
  /// Full-register or iterative circuit method.
  QPEMethod method = QPEMethod::Standard;
};

/// A validated QPE benchmark instance and its analytic reference.
class MQT_CORE_BENCH_EXPORT QPE final {
public:
  explicit QPE(QPEOptions options);

  [[nodiscard]] const QPEOptions& options() const noexcept;
  [[nodiscard]] const Output& output() const noexcept;
  /// Return the ideal probability of a big-endian logical outcome.
  [[nodiscard]] double probability(std::string_view outcome) const;
  /// Compare sampled logical outcomes with the ideal distribution.
  [[nodiscard]] Evaluation evaluate(const Counts& counts) const;

private:
  QPEOptions options_;
  Output output_;
  std::string lowerOutcome_;
  uint64_t scaledRemainder_;
};

} // namespace mqt::bench
