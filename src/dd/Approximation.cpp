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

#include "dd/Node.hpp"
#include "dd/Package.hpp"

#include <algorithm>
#include <deque>

namespace dd {
namespace {
/**
 * @brief Find next node to remove and delete its edge.
 * @details Traverses breadth-first with iterative deepening algorithm.
 * @return Fidelity lost due to removal of node.
 */
double findAndRemoveNext(VectorDD& v, Package& dd,
                         const Approximation<FidelityDriven>& approx,
                         NodeContributions& contributions) {
  std::deque<vNode*> q = {v.p};
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
        if (approx.fidelity < (1. - contributions[nxt])) {
          v = dd.deleteEdge(*edge, curr->v, i);
          return contributions[nxt];
        }
        q.push_back(nxt);
      }
    }
  }

  return 0;
}
}; // namespace

template <>
void applyApproximation<dd::FidelityDriven>(
    VectorDD& v, const Approximation<FidelityDriven>& approx, Package& dd) {
  if (v.isTerminal()) {
    return;
  }

  NodeContributions contributions(v);

  double fidelityBudget = 1 - approx.fidelity;
  while (fidelityBudget > 0) {
    double fidelityLost = findAndRemoveNext(v, dd, approx, contributions);
    if (fidelityLost == 0) {
      break;
    }
    fidelityBudget -= fidelityLost;
  }
};

/**
 * @brief Apply Memory-Driven Approximation.
 * @details If the threshold is exceeded, apply Fidelity-Driven approximation.
 * Inheritance allows us to simply downcast the Memory-Driven object.
 */
template <>
void applyApproximation<dd::MemoryDriven>(
    VectorDD& v, const Approximation<MemoryDriven>& approx, Package& dd) {
  if (v.size() > approx.threshold) {
    applyApproximation<FidelityDriven>(v, approx, dd);
  }
};

} // namespace dd
