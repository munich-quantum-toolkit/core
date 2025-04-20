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
#include "dd/Node.hpp"
#include "dd/Package.hpp"

#include <cmath>
#include <deque>
#include <unordered_map>
#include <vector>

namespace dd {

void approximate(VectorDD& state, const double fidelity, Package& dd) {
  constexpr auto mag2 = ComplexNumbers::mag2;

  double budget = 1 - fidelity;

  std::unordered_map<const vEdge*, double> probs{{&state, mag2(state.w)}};
  std::deque<vEdge*> q{&state};
  while (!q.empty() && budget > 0) {
    const std::vector<vEdge*> layer(q.begin(), q.end());

    q.clear();
    for (vEdge* lEdge : layer) {
      vNode* node = lEdge->p;
      const double parent = probs[lEdge];

      if (parent <= budget) {
        dd.decRef(*lEdge);
        *lEdge = vEdge::zero();
        budget -= parent;
      } else {
        for (auto& edge : node->e) {
          if (!edge.isTerminal() && !edge.w.exactlyZero()) {
            if (probs.find(&edge) == probs.end()) {
              q.push_back(&edge);
            }

            probs[&edge] += parent * mag2(edge.w);
          }
        }
      }
    }
  }

  auto& mm = dd.getMemoryManager<vNode>();
  state = vEdge::normalize(state.p, state.p->e, mm, dd.cn);
  state.w = dd.cn.lookup(state.w / std::sqrt(mag2(state.w)));
}

} // namespace dd
