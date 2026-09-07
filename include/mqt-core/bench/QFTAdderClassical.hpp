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
#include <string>
#include <string_view>

namespace mqt::bench {

/// Parameters for one classical-input QFT adder benchmark instance.
struct QFTAdderClassicalOptions {
  static constexpr size_t MAX_ADDEND_BITS = 1'023;

  /// Big-endian classical addend. Leading zeros define its width.
  std::string addend;
};

/// A QFT adder that adds the classical addend to an accumulator in |1>.
/// Leading zeros define the n-bit input width. The n+1-bit big-endian result
/// is addend + 1; the extra bit preserves carry.
class MQT_CORE_BENCH_EXPORT QFTAdderClassical final {
public:
  explicit QFTAdderClassical(QFTAdderClassicalOptions options);

  [[nodiscard]] const QFTAdderClassicalOptions& options() const noexcept;
  [[nodiscard]] const Output& output() const noexcept;
  /// Return the deterministic big-endian result.
  [[nodiscard]] const std::string& expectedResult() const noexcept;
  /// Return the ideal probability of a big-endian logical outcome.
  [[nodiscard]] double probability(std::string_view outcome) const;
  /// Compare sampled logical outcomes with the ideal distribution.
  [[nodiscard]] Evaluation evaluate(const Counts& counts) const;

private:
  QFTAdderClassicalOptions options_;
  Output output_;
  std::string expectedResult_;
};

} // namespace mqt::bench
