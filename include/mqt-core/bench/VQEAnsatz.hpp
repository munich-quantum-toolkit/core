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

namespace mqt::bench {

/// Parameters for a VQE ansatz benchmark.
struct VQEAnsatzOptions {
  /// Number of qubits, at least one.
  size_t qubits;
  /// Number of rotation-entanglement layers.
  size_t depth = 1;
};

/// A validated VQE ansatz benchmark.
class MQT_CORE_BENCH_EXPORT VQEAnsatz final {
public:
  explicit VQEAnsatz(VQEAnsatzOptions options);

  [[nodiscard]] const VQEAnsatzOptions& options() const noexcept;
  [[nodiscard]] const Output& output() const noexcept;
  /// Number of rotation angles, including the final rotation layer.
  [[nodiscard]] size_t numParameters() const noexcept;

private:
  VQEAnsatzOptions options_;
  Output output_;
};

} // namespace mqt::bench
