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

#include <array>
#include <cstddef>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace dd {
namespace {
using Layer = std::unordered_map<vEdge*, double>;

/**
 * @brief Given the l-th layer, compute the contributions for the (l-1)-th
 * layer.
 * @return The (l-1)-th layer.
 */
Layer computeLayer(const Layer& l) {
  Layer lNext;
  for (const auto& [edge, contribution] : l) {
    if (edge->isTerminal()) {
      continue;
    }

    for (std::size_t i = 0; i < RADIX; ++i) {
      vEdge* eChild = &edge->p->e[i];
      if (eChild->w.exactlyZero()) {
        continue;
      }

      lNext[eChild] += contribution * ComplexNumbers::mag2(eChild->w);
    }
  }

  return lNext;
}
}; // namespace

double approximate(VectorDD& state, const double fidelity, Package& dd) {
  Layer curr{{&state, 1.}};

  double budget = 1 - fidelity;
  while (!curr.empty() && budget > 0) {
    Layer candidates = computeLayer(curr);

    std::unordered_set<vEdge*> m{};
    for (const auto& [edge, contribution] : candidates) {
      if (contribution <= budget) {
        budget -= contribution;
        m.emplace(edge);
      }
    }

    Layer next{};
    for (const auto& [edge, _] : curr) {
      if (edge->isTerminal()) {
        continue;
      }

      vNode* node = edge->p;
      std::array<vEdge, RADIX> edges{};
      for (std::size_t i = 0; i < RADIX; ++i) {
        vEdge* eChild = &(node->e[i]);
        if (m.find(eChild) != m.end()) {
          edges[i] = vEdge::zero();
          continue;
        }

        edges[i] = *eChild;
      }
      vEdge replacement = dd.makeDDNode(node->v, edges);
      replacement.w = edge->w;
      dd.incRef(replacement);
      dd.decRef(*edge);
      *edge = replacement;

      for (std::size_t i = 0; i < RADIX; ++i) {
        next[&replacement.p->e[i]] = candidates[&(node->e[i])];
      }
    }

    curr = std::move(next);
  }

  // Rebuild only the root state to normalize all nodes of the DD.
  // TODO: Necessary?
  VectorDD approx = dd.makeDDNode(state.p->v, state.p->e);
  approx.w = state.w;

  // Make sure to correctly update the reference counts and clean up.
  // Finally, apply approximation to source state.
  if (approx != state) {
    dd.incRef(approx);
    dd.decRef(state);
    state = approx;
  }

  dd.garbageCollect();

  return fidelity + budget;
}

} // namespace dd
