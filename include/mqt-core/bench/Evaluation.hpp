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

#include <cstddef>
#include <map>
#include <optional>
#include <string>

namespace mqt::bench {

/// Describes one logical classical output register.
struct Output {
  /// Register name used by generated programs and manifests.
  std::string name;
  /// Number of bits, encoded as `width - 1` through `0` in outcome strings.
  size_t width;

  friend bool operator==(const Output&, const Output&) = default;
};

using Counts = std::map<std::string, size_t>;

/// Comparison of sampled counts with a benchmark's ideal reference.
struct Evaluation {
  /// Total variation distance. Zero is an exact match.
  double totalVariationDistance;
  /// Squared Hellinger fidelity. One is an exact match.
  double squaredHellingerFidelity;
  /// Observed success probability when the benchmark defines success.
  std::optional<double> successProbability;
};

} // namespace mqt::bench
