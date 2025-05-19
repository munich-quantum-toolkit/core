/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/Approximation.hpp"

#include "dd/Complex.hpp"
#include "dd/ComplexNumbers.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <queue>
#include <unordered_map>

namespace dd {
namespace {
/**
 * @brief Holds a node-contribution pair and distance value for
 * prioritization.
 */
struct LayerNode {
  explicit LayerNode(vNode* node)
      : ptr(node), contribution(1.),
        distance(std::numeric_limits<double>::max()) {}

  LayerNode(vNode* node, double contribution, double budget)
      : ptr(node), contribution(contribution),
        distance(std::max<double>(0, contribution - budget)) {}

  bool operator<(const LayerNode& other) const {
    if (distance == 0 && other.distance == 0) {
      return contribution < other.contribution;
    }
    return distance > other.distance;
  }

  vNode* ptr;
  double contribution;

private:
  double distance;
};

/// @brief Priority queue of nodes, sorted by their distance to a given budget.
using Layer = std::priority_queue<LayerNode>;
/// @brief Maps nodes to their respective contribution.
using Contributions = std::unordered_map<vNode*, double>;
///  @brief Maps old nodes to rebuilt edges.
using Lookup = std::unordered_map<const vNode*, vEdge>;

/**
 * @brief Search and mark nodes for deletion until the budget 1 - @p fidelity is
 * exhausted.
 * @details Uses a prioritized iterative-deepening search.
 * Iterating layer by layer ensures that each node is only visited once.
 */
std::pair<double, Qubit> mark(VectorDD& state, const double fidelity) {
  Layer curr{};
  curr.emplace(state.p);

  double budget = 1 - fidelity;
  Qubit min{std::numeric_limits<Qubit>::max()};

  while (budget > 0) {
    Contributions c; // Stores contributions of the next layer.
    while (!curr.empty()) {
      LayerNode n = curr.top();
      curr.pop();

      // If possible, flag a node for deletion and decrease the budget
      // accordingly.
      // If necessary, reset the lowest qubit number effected.
      if (budget >= n.contribution) {
        n.ptr->flags = 1U;
        budget -= n.contribution;
        min = std::min(min, n.ptr->v);
        continue;
      }

      // Compute contributions of the next layer.
      for (const vEdge& eRef : n.ptr->e) {
        if (eRef.isTerminal()) {
          continue;
        }
        c[eRef.p] += n.contribution * ComplexNumbers::mag2(eRef.w);
      }
    }

    Layer next{}; // Prioritize nodes for next iteration.
    for (auto& [n, contribution] : c) {
      next.emplace(n, contribution, budget);
    }

    if (next.empty()) { // Break early. Avoid std::move.
      break;
    }

    curr = std::move(next);
  }

  // The final fidelity is the desired fidelity plus the unused budget.
  return {fidelity + budget, min};
}

vEdge sweep(const vEdge& curr, const Qubit min, Lookup& l, Package& dd) {
  vNode* n = curr.p;

  // Nodes below v_{min} don't require rebuilding.
  if (n->v < min) {
    return curr;
  }

  // If a node is flagged, reset the flag and return a zero edge.
  if (n->flags == 1U) {
    n->flags = 0U;
    return vEdge::zero();
  }

  // If a node has been visited once, return the already rebuilt node
  // and set the edge weight accordingly.
  if (l.find(n) != l.end()) {
    vEdge eR = l.at(n);
    eR.w = curr.w;
    return eR;
  }

  // Otherwise traverse down to rebuild each non-terminal edge.
  std::array<vEdge, RADIX> edges{};
  for (std::size_t i = 0; i < RADIX; ++i) {
    vEdge& eRef = n->e[i];
    if (eRef.isTerminal()) {
      edges[i] = eRef;
      continue;
    }

    edges[i] = sweep(eRef, min, l, dd);
  }

  // Rebuild node and set its edge weight accordingly.
  vEdge eR = dd.makeDDNode(n->v, edges);
  eR.w = curr.w;
  l[n] = eR;
  return eR;
}

/**
 * @brief Recursively rebuild DD depth-first.
 * @details A lookup table ensures that each node is only visited once.
 */
vEdge sweep(const vEdge& e, const Qubit min, Package& dd) {
  Lookup l{};
  return sweep(e, min, l, dd);
}
}; // namespace

double approximate(VectorDD& state, const double fidelity, Package& dd) {
  const auto& [finalFidelity, min] = mark(state, fidelity);
  const vEdge& approx = sweep(state, min, dd);

  dd.incRef(approx);
  dd.decRef(state);
  dd.garbageCollect();

  state = approx;

  return finalFidelity;
}

} // namespace dd
