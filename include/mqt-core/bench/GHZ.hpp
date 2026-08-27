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
#include <string_view>

namespace mqt::bench {

/// Entangling topology for GHZ-state preparation.
enum class GHZTopology : uint8_t {
  /// Entangle each qubit with its next neighbor.
  Linear,
  /// Entangle every other qubit directly with the first qubit.
  Star
};

/// Measurement basis used to verify the prepared GHZ state.
enum class GHZBasis : uint8_t {
  /// Measure the two computational-basis outcomes.
  Z,
  /// Measure the uniform distribution over even-parity outcomes.
  X
};

/// Parameters for one GHZ benchmark instance.
struct GHZOptions {
  static constexpr size_t MAX_QUBITS = 1'000'000;
  static constexpr size_t MAX_X_BASIS_QUBITS = 1'075;

  /// Number of qubits. Must be in `[1, MAX_QUBITS]`.
  size_t qubits;
  /// Entangling topology.
  GHZTopology topology = GHZTopology::Linear;
  /// Measurement basis used by the program and reference.
  GHZBasis basis = GHZBasis::Z;
};

/// A validated GHZ benchmark instance and its analytic reference.
class MQT_CORE_BENCH_EXPORT GHZ final {
public:
  explicit GHZ(GHZOptions options);

  [[nodiscard]] const GHZOptions& options() const noexcept;
  [[nodiscard]] const Output& output() const noexcept;
  /// Return the ideal probability of a big-endian logical outcome.
  [[nodiscard]] double probability(std::string_view outcome) const;
  /// Compare sampled logical outcomes with the ideal distribution.
  [[nodiscard]] Evaluation evaluate(const Counts& counts) const;

private:
  GHZOptions options_;
  Output output_;
};

} // namespace mqt::bench
