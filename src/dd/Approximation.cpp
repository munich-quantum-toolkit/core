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

#include <algorithm>
#include <deque>

namespace dd {
namespace {
/**
 * @brief Find next node to remove and return its ingoing edge.
 * @details Traverses breadth-first using the iterative deepening algorithm.
 * @return Edge to remove or nullptr.
 */
vEdge* findNext(const VectorDD& state, const double budget,
                NodeContributions& contributions) {
  std::deque<vNode*> q = {state.p};
  while (!q.empty()) {
    vNode* curr = q.front();
    q.pop_front();

    for (std::size_t i = 0; i < 2; i++) {
      vEdge* edge = &curr->e[i];
      if (edge->isTerminal()) {
        continue;
      }

      vNode* nxt = edge->p;
      if (std::find(q.begin(), q.end(), nxt) == q.end()) {
        if (budget > contributions[nxt]) {
          return edge;
        }
        q.push_back(nxt);
      }
    }
  }

  return nullptr;
}

/**
 * @brief Recursively rebuild `state` without `edge` edge.
 * @return Edge to new `state`.
 */
vEdge rebuildWithout(const VectorDD& state, const vEdge& edge, Package& dd) {
  if (state.isTerminal()) {
    return state;
  }

  if (state == edge) {
    return vEdge::zero();
  }

  std::array<vEdge, RADIX> edges{rebuildWithout(state.p->e[0], edge, dd),
                                 rebuildWithout(state.p->e[1], edge, dd)};

  return dd.makeDDNode(state.p->v, edges);
}
}; // namespace

template <>
void applyApproximation<dd::FidelityDriven>(
    VectorDD& state, const Approximation<FidelityDriven>& approx, Package& dd) {
  if (state.isTerminal()) {
    return;
  }

  double budget = 1 - approx.fidelity;
  while (true) {
    NodeContributions contributions(state);

    vEdge* edge = findNext(state, budget, contributions);
    if (edge == nullptr) {
      break;
    }

    state = rebuildWithout(state, *edge, dd);
    state.w = dd.cn.lookup(state.w / std::sqrt(ComplexNumbers::mag2(state.w)));

    dd.incRef(state);
    dd.decRef(*edge);

    budget -= contributions[edge->p];
  }

  dd.garbageCollect();
};

/**
 * @brief Apply Memory-Driven Approximation.
 * @details If the threshold is exceeded, apply Fidelity-Driven approximation.
 * Inheritance allows us to simply downcast the Memory-Driven object.
 */
template <>
void applyApproximation<dd::MemoryDriven>(
    VectorDD& state, const Approximation<MemoryDriven>& approx, Package& dd) {
  if (state.size() > approx.threshold) {
    applyApproximation<FidelityDriven>(state, approx, dd);
  }
};

} // namespace dd
