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
#include <string_view>

namespace mqt::bench {

/// Parameters for one quantum fan-out benchmark instance.
struct FanOutOptions {
  /// Maximum control width with representable nonzero output probabilities.
  static constexpr size_t MAX_CONTROLS = 1'073;

  /// Number of input control qubits.
  ///
  /// Must be in `[1, MAX_CONTROLS]`. The circuit uses twice as many qubits.
  size_t controls;
};

/// A validated quantum fan-out benchmark.
class MQT_CORE_BENCH_EXPORT FanOut final {
public:
  [[nodiscard]] static llvm::Expected<FanOut> create(FanOutOptions options);

  [[nodiscard]] const FanOutOptions& options() const noexcept;
  [[nodiscard]] const Output& output() const noexcept;
  /// Return the ideal probability of a big-endian logical outcome.
  [[nodiscard]] llvm::Expected<double>
  probability(std::string_view outcome) const;
  /// Compare sampled logical outcomes with the ideal distribution.
  [[nodiscard]] llvm::Expected<Evaluation> evaluate(const Counts& counts) const;

private:
  explicit FanOut(FanOutOptions options);

  FanOutOptions options_;
  Output output_;
};

} // namespace mqt::bench
