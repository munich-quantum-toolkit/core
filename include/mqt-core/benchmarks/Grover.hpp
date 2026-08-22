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

#include "benchmarks/Evaluation.hpp"
#include "benchmarks/mqt_core_benchmarks_export.h"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace mqt::benchmarks {

/// Parameters for one single-solution Grover benchmark instance.
struct GroverOptions {
  /// Big-endian marked outcome. Its width is the number of search qubits.
  std::string markedBitstring;
  /// Iteration count, or no value to select the optimal count.
  std::optional<size_t> iterations = std::nullopt;
};

/// A validated Grover benchmark instance and its analytic reference.
class MQT_CORE_BENCHMARKS_EXPORT Grover final {
public:
  explicit Grover(GroverOptions options);

  [[nodiscard]] const GroverOptions& options() const noexcept;
  [[nodiscard]] size_t qubits() const noexcept;
  [[nodiscard]] const Output& output() const noexcept;
  [[nodiscard]] double markedProbability() const noexcept;
  [[nodiscard]] double otherProbability() const noexcept;
  /// Return the ideal probability of a big-endian logical outcome.
  [[nodiscard]] double probability(std::string_view outcome) const;
  /// Compare sampled logical outcomes with the ideal distribution.
  [[nodiscard]] Evaluation evaluate(const Counts& counts) const;

private:
  GroverOptions options_;
  Output output_;
  double markedProbability_;
  double otherProbability_;
};

} // namespace mqt::benchmarks
