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
#include <string>
#include <string_view>

namespace mqt::bench {

/// Parameters for one controlled multiplication modulo N benchmark instance.
struct ControlledMultiplicationModuloNOptions {
  static constexpr size_t MAX_BITS = 63;

  /// Big-endian classical multiplier. Leading zeros define its width.
  std::string multiplier;
  /// Big-endian canonical modulus with the same width as the multiplier.
  std::string modulus;
};

/// A validated controlled multiplication modulo N and its analytic reference.
class MQT_CORE_BENCH_EXPORT ControlledMultiplicationModuloN final {
public:
  explicit ControlledMultiplicationModuloN(
      ControlledMultiplicationModuloNOptions options);

  [[nodiscard]] const ControlledMultiplicationModuloNOptions&
  options() const noexcept;
  [[nodiscard]] const Output& output() const noexcept;
  /// Return the ideal probability of a big-endian logical outcome.
  [[nodiscard]] double probability(std::string_view outcome) const;
  /// Compare sampled logical outcomes with the ideal distribution.
  [[nodiscard]] Evaluation evaluate(const Counts& counts) const;

private:
  ControlledMultiplicationModuloNOptions options_;
  Output output_;
};

} // namespace mqt::bench
