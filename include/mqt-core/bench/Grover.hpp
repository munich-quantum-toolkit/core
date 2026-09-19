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

#include "bench/Error.hpp"
#include "bench/Evaluation.hpp"
#include "bench/mqt_core_bench_export.h"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace mqt::bench {

/// Parameters for one single-solution Grover benchmark instance.
struct GroverOptions {
  /// Big-endian marked outcome. Its width is the number of search qubits.
  std::string markedBitstring;
  /// Iteration count, or no value to select the optimal count.
  std::optional<size_t> iterations = std::nullopt;
};

/// A validated single-solution Grover benchmark.
class MQT_CORE_BENCH_EXPORT Grover final {
public:
  [[nodiscard]] static Result<Grover> create(GroverOptions options);

  [[nodiscard]] const GroverOptions& options() const noexcept;
  /// Return the number of search qubits.
  [[nodiscard]] size_t qubits() const noexcept;
  [[nodiscard]] const Output& output() const noexcept;
  /// Return the ideal probability of a big-endian logical outcome.
  [[nodiscard]] Result<double> probability(std::string_view outcome) const;
  /// Compare sampled logical outcomes with the ideal distribution.
  [[nodiscard]] Result<Evaluation> evaluate(const Counts& counts) const;

private:
  explicit Grover(GroverOptions options);

  GroverOptions options_;
  Output output_;
  double markedProbability_;
  double otherProbability_;
};

} // namespace mqt::bench
