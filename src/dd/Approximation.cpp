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

  double sum{};
  const vNode* node = curr.p;

  std::array<vEdge, RADIX> edges{};
  for (std::size_t i = 0; i < edges.size(); i++) {
    const auto& edge = node->e[i];

    const auto& p = approximate(edge, amplitude * edge.w, budget, dd);
    if (p.edge.isTerminal() || p.contrib > budget) {
      edges[i] = p.edge;
    } else {
      edges[i] = vEdge::zero();
      std::cout << "budget: " << budget << " contrib: " << p.contrib << '\n';
      budget -= p.contrib;
    }
    sum += p.contrib;
  }

  return {dd.makeDDNode(node->v, edges), sum};
}
}; // namespace

VectorDD approximate(const VectorDD& state, const double fidelity,
                     Package& dd) {
  const ComplexValue amplitude{state.w};
  const double budget = 1 - fidelity;
  const Approx result = approximate(state, amplitude, budget, dd);

  VectorDD out = result.edge;
  out.w = dd.cn.lookup(out.w / std::sqrt(ComplexNumbers::mag2(out.w)));
  return out;
};

} // namespace dd
