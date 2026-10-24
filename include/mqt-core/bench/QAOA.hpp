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

#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

namespace mqt::bench {

/// Parameters for a QAOA benchmark for MaxCut on a cycle graph.
struct QAOAOptions {
  /// Maximum supported depth.
  static constexpr size_t MAX_DEPTH =
      static_cast<size_t>(std::numeric_limits<int64_t>::max());

  /// Number of vertices and qubits, at least three and within signed 64-bit
  /// circuit dimensions.
  size_t qubits;
  /// Number of cost-mixer layers.
  size_t depth = 1;
};

/// Classical scores of sampled QAOA cuts.
struct QAOAEvaluation {
  /// Mean number of edges whose endpoints have different measured bits.
  double meanCut = 0.;
  /// Mean cut size divided by the cycle graph's maximum cut size.
  double approximationRatio = 0.;
};

/// A validated QAOA benchmark for MaxCut.
class MQT_CORE_BENCH_EXPORT QAOA final {
public:
  [[nodiscard]] static llvm::Expected<QAOA> create(QAOAOptions options);

  [[nodiscard]] const QAOAOptions& options() const noexcept;
  [[nodiscard]] const Output& output() const noexcept;
  /// Return the cut size of a big-endian logical outcome on the cycle.
  [[nodiscard]] llvm::Expected<size_t> cutSize(std::string_view outcome) const;
  /// Score sampled cuts without computing an ideal output distribution.
  [[nodiscard]] llvm::Expected<QAOAEvaluation>
  evaluate(const Counts& counts) const;

private:
  explicit QAOA(QAOAOptions options);

  QAOAOptions options_;
  Output output_;
};

} // namespace mqt::bench
