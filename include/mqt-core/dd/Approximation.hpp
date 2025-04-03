/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "dd/ComplexValue.hpp"
#include "dd/Node.hpp"

#include <algorithm>
#include <cstddef>

namespace dd {
enum ApproximationStrategy { FidelityDriven, MemoryDriven };

template <const ApproximationStrategy strategy> struct Approximation {};

template <> struct Approximation<FidelityDriven> {
  explicit Approximation(double finalFidelity) noexcept
      : finalFidelity(finalFidelity) {}
  double finalFidelity;
};

template <> struct Approximation<MemoryDriven> {
  Approximation(std::size_t maxNodes, double roundFidelity,
                std::size_t factor = 2) noexcept
      : maxNodes(maxNodes), roundFidelity(roundFidelity), factor(factor) {}

  /**
   * @brief Multiplies `maxNodes` by `factor`.
   * @details Used after each approx. round to increase `maxNodes` s.t. too many
   * approximations are avoided.
   */
  void increaseMaxNodes() noexcept { maxNodes *= factor; }

  std::size_t maxNodes;
  double roundFidelity;
  std::size_t factor;
};

struct NodeContributions {
  using Map = std::unordered_map<const dd::vNode*, double>;
  using Vector = std::vector<std::pair<const dd::vNode*, double>>;

  /**
   * @brief Recursively compute contributions for each node.
   * @return Vector of node-contribution pairs with ascending contribution.
   */
  Vector operator()(const dd::vEdge& root) {
    compute(root, dd::ComplexValue{root.w});
    return vectorize();
  }

private:
  /**
   * @brief Recursively compute contributions for each node.
   * @details Propagates downwards until it reaches a terminal. Then, return
   *          squared magnitude of amplitudes or 0. Sum these values for all
   *          edges of node. Since nodes can be visited multiple times by edges
   *          going inwards we add the sums to the value stored in the map for
   *          the respective node
   *
   *          Uses a lookup table to avoid computing the contributions of the
   *          same node twice.
   * @return Vector of node-contribution pairs with ascending contribution.
   */
  double compute(const dd::vEdge& edge, const dd::ComplexValue acc) {
    // Reached the end of a path. Either return 0 or squared magnitude.
    if (edge.isZeroTerminal()) {
      return 0.;
    }
    if (edge.isTerminal()) {
      return acc.mag2();
    }

    double sum = 0.;
    const dd::vNode* node = edge.p;

    // If the node has already been visited once, reuse value from lookup table.
    //
    // Otherwise compute the contribution of each node, which is the sum of
    // squared magnitudes of amplitudes for each path passing through that node.
    if (lookup.count(node) > 0) {
      sum = lookup[node];
    } else {
      for (const auto& e : node->e) {
        sum += compute(e, acc * e.w);
      }
      lookup[node] = sum;
    }

    contributions[node] += sum;

    return sum;
  }

  /**
   * @brief Vectorize `contributions` map and sort ascendingly by contribution.
   */
  Vector vectorize() {
    Vector v{};
    v.reserve(contributions.size());
    for (const auto& pair : contributions) {
      v.emplace_back(pair);
    }

    const auto comp = [](auto& a, auto& b) { return a.second < b.second; };
    std::sort(v.begin(), v.end(), comp);

    return v;
  }

  Map contributions{};
  Map lookup{};
};
} // namespace dd
