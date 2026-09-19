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
#include <cstdint>
#include <string_view>

namespace mqt::bench {

/// Circuit method used for the QFT.
enum class QFTMethod : uint8_t {
  /// Transform one qubit for each input bit before measurement.
  Standard,
  /// Measure and reset one reused qubit, with feed-forward from prior results.
  Semiclassical,
};

/// Parameters for one QFT benchmark instance.
struct QFTOptions {
  static constexpr size_t MAX_QUBITS = 1'000'000;
  static constexpr size_t MAX_PERIOD_EXPONENT = 1'074;

  /// Number of transformed qubits.
  size_t qubits;
  /// Exponent \f$k\f$ of the input period \f$2^k\f$.
  size_t periodExponent;
  /// Full-register or semiclassical circuit method.
  QFTMethod method = QFTMethod::Standard;
};

/// A validated QFT benchmark.
class MQT_CORE_BENCH_EXPORT QFT final {
public:
  [[nodiscard]] static Result<QFT> create(QFTOptions options);

  [[nodiscard]] const QFTOptions& options() const noexcept;
  [[nodiscard]] const Output& output() const noexcept;
  /// Return the ideal probability of a big-endian logical outcome.
  [[nodiscard]] Result<double> probability(std::string_view outcome) const;
  /// Compare sampled logical outcomes with the ideal distribution.
  [[nodiscard]] Result<Evaluation> evaluate(const Counts& counts) const;

private:
  explicit QFT(QFTOptions options);

  QFTOptions options_;
  Output output_;
};

} // namespace mqt::bench
