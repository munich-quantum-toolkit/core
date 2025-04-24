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
namespace {
/**
 * @brief Recursively rebuild @p state. Exclude edges contained in @p exclude.
 * @return Rebuilt VectorDD.
 */
VectorDD rebuild(const VectorDD& state,
                 const std::forward_list<vEdge*>& exclude, Package& dd) {
  const auto it = std::find(exclude.begin(), exclude.end(), &state);
  if (it != exclude.end()) {
    return vEdge::zero();
  }

  if (state.isTerminal()) {
    return state;
  }

  std::array<vEdge, RADIX> edges{rebuild(state.p->e[0], exclude, dd),
                                 rebuild(state.p->e[1], exclude, dd)};

  auto e = dd.makeDDNode(state.p->v, edges);
  e.w = dd.cn.lookup(e.w * state.w);
  return e;
}
}; // namespace

VectorDD approximate(VectorDD& state, const double fidelity, Package& dd) {
  using ContributionMap = std::unordered_map<const vEdge*, double>;
  using Layer = std::forward_list<vEdge*>;

  constexpr auto mag2 = ComplexNumbers::mag2;

  Layer l{&state};
  Layer exclude{};
  ContributionMap m{{&state, mag2(state.w)}};

  double budget = 1 - fidelity;
  while (!l.empty() && budget > 0) {
    Layer nextL{};
    ContributionMap nextM{};

    for (vEdge* edge : l) {
      const double contribution = m[edge];
      if (contribution <= budget) {
        exclude.emplace_front(edge);
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

  auto approx = rebuild(state, exclude, dd);
  approx.w = dd.cn.lookup(approx.w / std::sqrt(mag2(approx.w)));

  dd.incRef(approx);
  dd.decRef(state);
  dd.garbageCollect();

  return approx;
}

} // namespace dd
