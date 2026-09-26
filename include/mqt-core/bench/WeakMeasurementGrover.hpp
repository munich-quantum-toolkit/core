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
#include <optional>
#include <string>
#include <string_view>

namespace mqt::bench {

/// Parameters for one weak-measurement Grover benchmark instance.
struct WeakMeasurementGroverOptions {
  /// Largest supported search-register width.
  static constexpr size_t MAX_QUBITS = 62;

  /// Big-endian marked outcome with 2 through 62 search qubits.
  std::string markedBitstring;
  /// Finite strength in (0, 2^(-n/2)], or no value to use the upper bound.
  std::optional<double> measurementStrength = std::nullopt;
};

/// A validated weak-measurement Grover benchmark.
class MQT_CORE_BENCH_EXPORT WeakMeasurementGrover final {
public:
  explicit WeakMeasurementGrover(WeakMeasurementGroverOptions options);

  [[nodiscard]] const WeakMeasurementGroverOptions& options() const noexcept;
  /// Return the number of search qubits.
  [[nodiscard]] size_t qubits() const noexcept;
  [[nodiscard]] const Output& output() const noexcept;
  /// Return the ideal probability of a big-endian logical outcome.
  [[nodiscard]] double probability(std::string_view outcome) const;
  /// Compare sampled logical outcomes with the ideal distribution.
  [[nodiscard]] Evaluation evaluate(const Counts& counts) const;

private:
  WeakMeasurementGroverOptions options_;
  Output output_;
};

} // namespace mqt::bench
