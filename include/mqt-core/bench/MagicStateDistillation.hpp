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

/// Parameters for concatenated 15-to-1 Reed--Muller distillation.
struct MagicStateDistillationOptions {
  /// Concatenated levels in [1, 4], using exactly 15^levels qubits.
  size_t levels = 1;
};

/// Distill ideal |T> = T|+> inputs, consuming retained outputs at each level.
/// Bit 1 flags any rejected block; bit 0 checks the root output in the T basis.
class MQT_CORE_BENCH_EXPORT MagicStateDistillation final {
public:
  explicit MagicStateDistillation(MagicStateDistillationOptions options = {});
  [[nodiscard]] const MagicStateDistillationOptions& options() const noexcept;
  [[nodiscard]] const Output& output() const noexcept;
  /// Return the ideal probability of a big-endian logical outcome.
  [[nodiscard]] double probability(std::string_view outcome) const;
  /// Compare sampled logical outcomes with the ideal distribution.
  [[nodiscard]] Evaluation evaluate(const Counts& counts) const;

private:
  MagicStateDistillationOptions options_;
  Output output_;
};

} /* namespace mqt::bench */
