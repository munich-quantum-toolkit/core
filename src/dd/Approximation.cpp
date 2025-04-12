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

namespace dd {
namespace {
struct Approx {
  vEdge edge;
  double contrib;
};
Approx approximate(const vEdge& curr, const ComplexValue& amplitude,
                   double budget, Package& dd) {
  if (curr.isTerminal()) {
    return {curr, curr.w.exactlyZero() ? 0 : amplitude.mag2()};
  }

  const vNode* node = curr.p;

  double sum{};
  std::array<vEdge, RADIX> edges{};
  for (std::size_t i = 0; i < edges.size(); i++) {
    const vEdge& edge = node->e[i];

    const Approx& ap = approximate(edge, amplitude * edge.w, budget, dd);
    if (ap.edge.isTerminal() || ap.contrib > budget) {
      edges[i] = ap.edge;
      sum += ap.contrib;
    } else {
      edges[i] = vEdge::zero();
      budget -= ap.contrib;
    }
  }

  vEdge next = dd.makeDDNode(node->v, edges);
  next.w = dd.cn.lookup(next.w * curr.w);
  return {next, sum};
}
}; // namespace

VectorDD approximate(const VectorDD& state, const double fidelity,
                     Package& dd) {
  const ComplexValue amplitude{state.w};
  const double budget = 1 - fidelity;
  Approx ap = approximate(state, amplitude, budget, dd);
  ap.edge.w =
      dd.cn.lookup(ap.edge.w / std::sqrt(ComplexNumbers::mag2(ap.edge.w)));
  return ap.edge;
};

} // namespace dd
