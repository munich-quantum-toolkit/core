/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include <cstddef>

namespace dd {
enum ApproximationStrategy { FidelityDriven, MemoryDriven };

template <const ApproximationStrategy strategy> struct Approximation {};

template <> struct Approximation<FidelityDriven> {
  constexpr explicit Approximation(double finalFidelity) noexcept
      : finalFidelity(finalFidelity) {}
  double finalFidelity;
};

template <> struct Approximation<MemoryDriven> {
  constexpr Approximation(std::size_t maxNodes, double roundFidelity,
                          double factor = 2.) noexcept
      : maxNodes(maxNodes), roundFidelity(roundFidelity), factor(factor) {}

  /**
   * @brief Multiplies `roundFidelity` by `factor`.
   * @details Used after each approx. round to increase `maxNodes` s.t. too many
   * approximations are avoided.
   */
  void increaseFidelity() noexcept { roundFidelity *= factor; }

  std::size_t maxNodes;
  double roundFidelity;
  double factor;
};
} // namespace dd
