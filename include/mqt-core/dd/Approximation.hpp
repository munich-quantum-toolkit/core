/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "dd/ComplexValue.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"

#include <cstddef>

namespace dd {
enum ApproximationStrategy { None, FidelityDriven, MemoryDriven };

template <const ApproximationStrategy stgy> struct Approximation {};

template <> struct Approximation<None> {};

template <> struct Approximation<FidelityDriven> {
  constexpr explicit Approximation(double fidelity) noexcept
      : fidelity(fidelity) {}
  double fidelity;
};

template <> struct Approximation<MemoryDriven> {
  constexpr Approximation(std::size_t maxNodes, double fidelity,
                          std::size_t factor = 2) noexcept
      : maxNodes(maxNodes), fidelity(fidelity), factor(factor) {}

  std::size_t maxNodes;
  double fidelity;
  std::size_t factor;
};

class NodeContributions {
public:
  explicit NodeContributions(const vEdge& root) {
    Map lookup{};
    compute(root, ComplexValue{root.w}, lookup);
  }

  double operator[](const vNode* key) noexcept { return contributions[key]; }

private:
  using Map = std::unordered_map<const vNode*, double>;

  /**
   * @brief   Recursively compute contributions for each node.
   * @details Propagates downwards until it reaches a terminal. Then, return
   *          squared magnitude of amplitudes or 0. Sum these values for all
   *          edges of node. Since nodes can be visited multiple times by edges
   *          going inwards we add the sums to the value stored in the map for
   *          the respective node
   *
   *          The accumulator computes the amplitude.
   *
   *          Uses a lookup table to avoid computing the contributions of the
   *          same node twice.
   */
  double compute(const vEdge& edge, const ComplexValue acc, Map& lookup) {
    // Reached the end of a path. Either return 0 or squared magnitude.
    if (edge.isZeroTerminal()) {
      return 0.;
    }
    if (edge.isTerminal()) {
      return acc.mag2();
    }

    double sum = 0.;
    const vNode* node = edge.p;

    // If the node has already been visited once, reuse value from lookup table.
    //
    // Otherwise compute the contribution of each node, which is the sum of
    // squared magnitudes of amplitudes for each path passing through that node.
    if (lookup.count(node) > 0) {
      sum = lookup[node];
    } else {
      for (const auto& e : node->e) {
        sum += compute(e, acc * e.w, lookup);
      }
      lookup[node] = sum;
    }

    contributions[node] += sum;

    return sum;
  }

  Map contributions{};
};

template <const ApproximationStrategy stgy>
void applyApproximation(VectorDD& v, const Approximation<stgy>& approx,
                        Package& dd);
} // namespace dd
