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

/// Parameters for W-state preparation.
struct WStateOptions {
  /// Positive number of qubits; circuit dimensions must fit signed 64-bit
  /// indices.
  size_t qubits;
};

/// Prepare the equal, positive-amplitude superposition of single excitations.
class MQT_CORE_BENCH_EXPORT WState final {
public:
  explicit WState(WStateOptions options);
  [[nodiscard]] const WStateOptions& options() const noexcept;
  [[nodiscard]] const Output& output() const noexcept;
  [[nodiscard]] double probability(std::string_view outcome) const;
  [[nodiscard]] Evaluation evaluate(const Counts& counts) const;

private:
  WStateOptions options_;
  Output output_;
};

} // namespace mqt::bench
