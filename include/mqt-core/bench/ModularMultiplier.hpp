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
#include <optional>
#include <string>
#include <string_view>

namespace mqt::bench {

/// Parameters for one modular multiplier benchmark instance.
struct ModularMultiplierOptions {
  static constexpr size_t MAX_BITS = 63;

  /// Big-endian classical multiplier. Leading zeros define its width.
  std::string multiplier;
  /// Big-endian canonical modulus with the same width as the multiplier.
  std::string modulus;
  /// Big-endian multiplicand of the same width; '+' prepares a
  /// \f$|+\rangle\f$ qubit.
  std::string multiplicand;
  /// Initial control bit: '0', '1', or '+' for \f$|+\rangle\f$.
  char control = '1';
};

/// A validated modular multiplier benchmark.
///
/// The initially zero product register stores
/// \f$c \cdot a \cdot x \bmod N\f$, where \f$c\f$ is the control, \f$a\f$ is
/// the classical multiplier, \f$x\f$ is the multiplicand, and \f$N\f$ is the
/// modulus.
class MQT_CORE_BENCH_EXPORT ModularMultiplier final {
public:
  explicit ModularMultiplier(ModularMultiplierOptions options);

  [[nodiscard]] const ModularMultiplierOptions& options() const noexcept;
  [[nodiscard]] const Output& output() const noexcept;
  /// Return the unique outcome, or no value for superposed inputs.
  [[nodiscard]] const std::optional<std::string>&
  expectedResult() const noexcept;
  /// Return the ideal probability of a big-endian logical outcome.
  [[nodiscard]] double probability(std::string_view outcome) const;
  /// Compare sampled logical outcomes with the ideal distribution.
  [[nodiscard]] Evaluation evaluate(const Counts& counts) const;

private:
  ModularMultiplierOptions options_;
  Output output_;
  std::optional<std::string> expectedResult_;
  uint64_t multiplierValue_ = 0;
  uint64_t modulusValue_ = 0;
};

} // namespace mqt::bench
