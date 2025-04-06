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
#include <deque>

namespace dd {
enum ApproximationStrategy { None, FidelityDriven, MemoryDriven };

template <const ApproximationStrategy stgy> struct Approximation {};
template <> struct Approximation<None> {};
template <> struct Approximation<FidelityDriven> {
  explicit Approximation(double fidelity) noexcept : fidelity(fidelity) {}
  double fidelity;
};
template <> struct Approximation<MemoryDriven> {
  Approximation(std::size_t maxNodes, double fidelity,
                std::size_t factor = 2) noexcept
      : maxNodes(maxNodes), fidelity(fidelity), factor(factor) {}

  /**
   * @brief   Multiplies `maxNodes` by `factor`.
   * @details Used after each approx. round to increase `maxNodes` s.t. too
   *          many approximations are avoided.
   */
  void increaseMaxNodes() noexcept { maxNodes *= factor; }

  std::size_t maxNodes;
  double fidelity;
  std::size_t factor;
};

struct NodeContributions {
  struct Pair {
    const vNode* node;
    double contribution;
  };

  using Vector = std::vector<Pair>;

  /**
   * @brief  Recursively compute contributions for each node.
   * @return Vector of node-contribution pairs with ascending contribution.
   */
  Vector operator()(const vEdge& root) {
    compute(root, ComplexValue{root.w});
    return vectorize(root.p);
  }

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
   * @return  Vector of node-contribution pairs with ascending contribution.
   */
  double compute(const vEdge& edge, const ComplexValue acc) {
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
        sum += compute(e, acc * e.w);
      }
      lookup[node] = sum;
    }

    contributions[node] += sum;

    return sum;
  }

  /**
   * @brief Vectorize `contributions` map breadth-first.
   * @details Uses iterative deepening search.
   */
  Vector vectorize(const vNode* node) {
    Vector v{};
    v.reserve(contributions.size()); // The # of nodes.
    v.push_back({node, contributions[node]});

    std::deque<const vNode*> q;
    q.push_back(node);
    while (!q.empty()) {
      const vNode* curr = q.front();
      q.pop_front();

      for (const auto& edge : curr->e) {
        const vNode* nxt = edge.p;
        if (!edge.isTerminal() &&
            std::find(q.begin(), q.end(), nxt) == q.end()) {
          v.push_back({nxt, contributions[nxt]});
          q.push_back(nxt);
        }
      }
    }

    return v;
  }

  Map contributions{};
  Map lookup{};
};

template <const ApproximationStrategy stgy>
void applyApproximation(VectorDD& v, Approximation<stgy>& approx);
} // namespace dd
