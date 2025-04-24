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
  constexpr auto mag2 = ComplexNumbers::mag2;

  std::unordered_map<const vEdge*, double> contributions{
      {&state, mag2(state.w)}};
  std::forward_list<vEdge*> layer{&state};

  double budget = 1 - fidelity;
  while (!layer.empty() && budget > 0) {
    std::forward_list<vEdge*> nextLayer{};

    for (vEdge* lEdge : layer) {
      vNode* node = lEdge->p;
      const double contribution = contributions[lEdge];

      if (contribution <= budget) {
        dd.decRef(*lEdge);
        *lEdge = vEdge::zero();
        budget -= contribution;
      } else {
        for (auto& edge : node->e) {
          if (!edge.isTerminal() && !edge.w.exactlyZero()) {
            if (contributions.find(&edge) == contributions.end()) {
              nextLayer.emplace_front(&edge);
            }

            contributions[&edge] += contribution * mag2(edge.w);
          }
        }
      }
    }

    layer = std::move(nextLayer);
  }

  auto& mm = dd.getMemoryManager<vNode>();
  state = vEdge::normalize(state.p, state.p->e, mm, dd.cn);
  state.w = dd.cn.lookup(state.w / std::sqrt(mag2(state.w)));
}

} // namespace dd
