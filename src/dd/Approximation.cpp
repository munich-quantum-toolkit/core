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

#include "dd/ComplexNumbers.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"

#include <array>
#include <forward_list>
#include <unordered_map>
#include <utility>

namespace dd {
namespace {
using UpwardsItem = std::pair<std::forward_list<vEdge*>, std::size_t>;
using Upwards = std::unordered_map<vEdge*, UpwardsItem>;
using Layer = std::unordered_map<vEdge*, double>;

/**
 * @brief Sets the edge specified by the @p item to `vEdge::zero()` and
 * multiplies the weights of the parent's edges with the weight of the
 * new edge.
 * @note The resulting DD is not normalized.
 */
void zeroEdge(UpwardsItem& item, Package& dd) {
  const auto& [parents, i] = item;
  const vNode* parentNode = parents.front()->p;

  std::array<vEdge, RADIX> edges = parentNode->e;
  edges[i] = vEdge::zero();

  vEdge newEdge = dd.makeDDNode(parentNode->v, edges);
  Complex w = newEdge.w;

  for (const auto& edge : parents) {
    newEdge.w = dd.cn.lookup(w * edge->w);

    dd.decRef(*edge);
    dd.incRef(newEdge);
    *edge = newEdge;
  }
}
}; // namespace

double approximate(VectorDD& state, const double fidelity, Package& dd) {
  const Complex phase = state.w;

  Upwards upwards{};
  Layer curr{{&state, ComplexNumbers::mag2(state.w)}};

  double budget = 1 - fidelity;
  while (!curr.empty() && budget > 0) {
    Layer next{};
    Upwards upwardsNext{};

    for (const auto& [edge, contribution] : curr) {
      if (contribution <= budget) {
        zeroEdge(upwards[edge], dd);
        budget -= contribution;
        continue;
      }

      if (edge->isTerminal()) {
        continue;
      }

      for (std::size_t i = 0; i < RADIX; ++i) {
        vEdge* eChild = &edge->p->e[i];
        if (eChild->w.exactlyZero()) {
          continue;
        }

        // Implicit: If `next` doesn't contain `eChild`, it will be initialized
        // with 0. See `operator[]`.
        next[eChild] += contribution * ComplexNumbers::mag2(eChild->w);

        // Link the child with the parent's edge and its associated index.
        if (upwardsNext.find(eChild) == upwardsNext.end()) {
          upwardsNext[eChild] = {{edge}, i};
          continue;
        }
        upwardsNext[eChild].first.emplace_front(edge);
      }
    }

    curr = std::move(next);
    upwards = std::move(upwardsNext);
  }

  // Rebuild only the root state to normalize all nodes of the DD.
  // Then: Apply global phase.
  VectorDD approx = dd.makeDDNode(state.p->v, state.p->e);
  approx.w = phase;

  // Make sure to correctly update the reference counts and clean up.
  dd.incRef(approx);
  dd.decRef(state);
  dd.garbageCollect();

  // Finally, apply approximation to source state.
  state = approx;

  return fidelity + budget;
}

} // namespace dd
