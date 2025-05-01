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
#include <array>
#include <cmath>
#include <forward_list>
#include <utility>

namespace dd {
namespace {
/**
 * @brief Recursively rebuild @p state. Exclude edges contained in @p exclude.
 * @return Rebuilt VectorDD.
 */
VectorDD rebuild(const VectorDD& state,
                 const std::forward_list<const vEdge*>& exclude, Package& dd) {
  const auto it = std::find(exclude.begin(), exclude.end(), &state);
  if (it != exclude.end()) {
    return vEdge::zero();
  }

  if (state.isTerminal()) {
    return state;
  }

  const std::array<vEdge, RADIX> edges{rebuild(state.p->e[0], exclude, dd),
                                       rebuild(state.p->e[1], exclude, dd)};

  VectorDD edge = dd.makeDDNode(state.p->v, edges);
  edge.w = dd.cn.lookup(edge.w * state.w);
  return edge;
}
}; // namespace

std::pair<VectorDD, double> approximate(const VectorDD& state,
                                        const double fidelity, Package& dd) {
  using Layer = std::forward_list<std::pair<const vEdge*, double>>;

  double budget = 1 - fidelity;
  std::forward_list<const vEdge*> exclude{};

  Layer curr{{&state, ComplexNumbers::mag2(state.w)}};
  while (!curr.empty() && budget > 0) {
    Layer next{};

    for (const auto& [e, contribution] : curr) {
      if (contribution <= budget) {
        exclude.emplace_front(e);
        budget -= contribution;
        continue;
      }

      if (e->isTerminal()) {
        continue;
      }

      const vNode* n = e->p;
      for (const vEdge& eChildRef : n->e) {
        const vEdge* eChild = &eChildRef;

        if (eChild->w.exactlyZero()) {
          continue;
        }

        const double childContribution =
            contribution * ComplexNumbers::mag2(eChild->w);
        const auto it =
            std::find_if(next.begin(), next.end(), [&eChild](const auto& p) {
              return p.first == eChild;
            });
        if (it == next.end()) {
          next.emplace_front(eChild, childContribution);
        } else {
          (*it).second += childContribution;
        }
      }
    }

    curr = std::move(next);
  }

  VectorDD approx = rebuild(state, exclude, dd);
  approx.w = dd.cn.lookup(approx.w / std::sqrt(ComplexNumbers::mag2(approx.w)));

  dd.incRef(approx);
  dd.decRef(state);
  dd.garbageCollect();

  return {approx, fidelity + budget};
}

} // namespace dd
