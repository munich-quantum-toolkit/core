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
#include <string_view>

namespace mqt::bench {

/// Parameters for one quantum multiplexer benchmark instance.
struct MultiplexerOptions {
  static constexpr size_t MAX_QUBITS = 1'024;

  /// Total number of control and target qubits.
  size_t qubits;
};

/// A validated quantum multiplexer benchmark and its analytic reference.
class MQT_CORE_BENCH_EXPORT Multiplexer final {
public:
  explicit Multiplexer(MultiplexerOptions options);

  [[nodiscard]] const MultiplexerOptions& options() const noexcept;
  [[nodiscard]] const Output& output() const noexcept;
  /// Return the ideal probability of a big-endian logical outcome.
  [[nodiscard]] double probability(std::string_view outcome) const;
  /// Compare sampled logical outcomes with the ideal distribution.
  [[nodiscard]] Evaluation evaluate(const Counts& counts) const;

private:
  MultiplexerOptions options_;
  Output output_;
};

} // namespace mqt::bench
