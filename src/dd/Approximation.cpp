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
#include <forward_list>
#include <unordered_map>

namespace dd {

void approximate(VectorDD& state, const double fidelity, Package& dd) {
  using ContributionMap = std::unordered_map<const vEdge*, double>;
  using Layer = std::forward_list<vEdge*>;

  constexpr auto mag2 = ComplexNumbers::mag2;

  Layer l{&state};
  ContributionMap m{{&state, mag2(state.w)}};

  double budget = 1 - fidelity;
  while (!l.empty() && budget > 0) {
    Layer nextL{};
    ContributionMap nextM{};

    for (vEdge* edge : l) {
      const double contribution = m[edge];

      if (contribution <= budget) {
        dd.decRef(*edge);
        *edge = vEdge::zero();
        budget -= contribution;
      } else if (!edge->isTerminal()) {
        vNode* node = edge->p;
        for (auto& nextEdge : node->e) {
          if (!nextEdge.w.exactlyZero()) {

            if (nextM.find(&nextEdge) == nextM.end()) {
              nextL.emplace_front(&nextEdge);
            }

            nextM[&nextEdge] += contribution * mag2(nextEdge.w);
          }
        }
      }
    }

    l = std::move(nextL);
    m = std::move(nextM);
  }

  auto& mm = dd.getMemoryManager<vNode>();
  state = vEdge::normalize(state.p, state.p->e, mm, dd.cn);
  state.w = dd.cn.lookup(state.w / std::sqrt(mag2(state.w)));
}

} // namespace dd
