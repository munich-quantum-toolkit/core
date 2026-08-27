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

/// Circuit method used for the quantum Fourier transform.
enum class QFTMethod : uint8_t {
  /// Transform one qubit for each input bit before measurement.
  Standard,
  /// Measure and reset one reused qubit, with feed-forward from prior results.
  Semiclassical
};

/// Parameters for one quantum Fourier-transform benchmark instance.
struct QFTOptions {
  static constexpr size_t MAX_QUBITS = 1'000'000;
  static constexpr size_t MAX_PERIOD_EXPONENT = 1'074;

  /// Number of transformed qubits.
  size_t qubits;
  /// The input period is two raised to this exponent.
  size_t periodExponent;
  /// Full-register or semiclassical circuit method.
  QFTMethod method = QFTMethod::Standard;
};

/// A validated QFT benchmark instance and its analytic reference.
class MQT_CORE_BENCH_EXPORT QFT final {
public:
  explicit QFT(QFTOptions options);

  [[nodiscard]] const QFTOptions& options() const noexcept;
  [[nodiscard]] const Output& output() const noexcept;
  /// Return the ideal probability of a big-endian logical outcome.
  [[nodiscard]] double probability(std::string_view outcome) const;
  /// Compare sampled logical outcomes with the ideal distribution.
  [[nodiscard]] Evaluation evaluate(const Counts& counts) const;

private:
  QFTOptions options_;
  Output output_;
};

} // namespace mqt::bench
