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
#include <unordered_map>
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
  std::forward_list<const vEdge*> exclude{};
  std::forward_list<const vEdge*> layer{&state};
  std::unordered_map<const vEdge*, double> contributions{
      {&state, ComplexNumbers::mag2(state.w)}};

  double budget = 1 - fidelity;
  while (!layer.empty() && budget > 0) {
    std::forward_list<const vEdge*> nextLayer{};

    for (const vEdge* edge : layer) {
      const double contribution = contributions[edge];
      if (contribution <= budget) {
        exclude.emplace_front(edge);
        budget -= contribution;
      } else if (!edge->isTerminal()) {
        const vNode* node = edge->p;
        for (const auto& nextEdge : node->e) {
          if (!nextEdge.w.exactlyZero()) {
            if (std::find(nextLayer.begin(), nextLayer.end(), &nextEdge) ==
                nextLayer.end()) {
              nextLayer.emplace_front(&nextEdge);
            }
            contributions[&nextEdge] =
                contributions[&nextEdge] +
                contribution * ComplexNumbers::mag2(nextEdge.w);
          }
        }
      }
    }

    layer = std::move(nextLayer);
  }

  VectorDD approx = rebuild(state, exclude, dd);
  approx.w = dd.cn.lookup(approx.w / std::sqrt(ComplexNumbers::mag2(approx.w)));

  dd.incRef(approx);
  dd.decRef(state);
  dd.garbageCollect();

  return {approx, fidelity + budget};
}

} // namespace dd
