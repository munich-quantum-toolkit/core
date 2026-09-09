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

/// Parameters for a Pauli-string repeat-until-success benchmark.
struct RepeatUntilSuccessOptions {
  static constexpr size_t MAX_DATA_QUBITS = 1'000'000;

  /// Number of data qubits in `[1, MAX_DATA_QUBITS]`, excluding the ancilla.
  size_t dataQubits = 1;
};

/// A validated Pauli-string repeat-until-success benchmark.
///
/// The circuit applies
/// \f$(I+i\sqrt{2}X^{\otimes n})/\sqrt{3}\f$.
class MQT_CORE_BENCH_EXPORT RepeatUntilSuccess final {
public:
  explicit RepeatUntilSuccess(RepeatUntilSuccessOptions options = {});

  [[nodiscard]] const RepeatUntilSuccessOptions& options() const noexcept;

  [[nodiscard]] const Output& output() const noexcept;
  /// Return the ideal probability of a big-endian logical outcome.
  [[nodiscard]] double probability(std::string_view outcome) const;
  /// Compare sampled logical outcomes with the ideal distribution.
  [[nodiscard]] Evaluation evaluate(const Counts& counts) const;

private:
  RepeatUntilSuccessOptions options_;
  Output output_;
};

} // namespace mqt::bench
