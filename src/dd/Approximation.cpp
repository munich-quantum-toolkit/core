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

#include "dd/Complex.hpp"
#include "dd/ComplexNumbers.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"

#include <array>
#include <forward_list>
#include <unordered_map>
#include <utility>

namespace dd {
namespace {
vEdge rebuild(const vEdge& curr,
              std::unordered_map<const vEdge*, bool>& removables, Package& dd) {
  if (removables.find(&curr) == removables.end()) {
    return curr;
  }

  if (removables[&curr]) {
    return vEdge::zero();
  }

  std::array<vEdge, RADIX> edges{
      rebuild(curr.p->e[0], removables, dd),
      rebuild(curr.p->e[1], removables, dd),
  };

  vEdge e = dd.makeDDNode(curr.p->v, edges);
  e.w = curr.w;

  return e;
}
}; // namespace

double approximate(VectorDD& state, const double fidelity, Package& dd) {
  using Path = std::forward_list<const vEdge*>;
  using Paths = std::forward_list<Path>;
  using Queue = std::unordered_map<const vEdge*, std::pair<double, Paths>>;

  std::unordered_map<const vEdge*, bool> removables{};

  double budget = 1 - fidelity;

  Queue l{{&state, {1., {{}}}}};
  while (!l.empty() && budget > 0) {
    Queue lNext{};

    for (const auto& [edge, pair] : l) {
      const auto [contribution, paths] = pair;

      if (contribution <= budget) {
        for (const auto& path : paths) {
          for (const auto& e : path) {
            removables[e] = false;
          }
        }
        removables[edge] = true;
        budget -= contribution;
        continue;
      }

      if (edge->isTerminal()) {
        continue;
      }

      for (const auto& eRef : edge->p->e) {
        if (eRef.w.exactlyZero()) {
          continue;
        }

        lNext[&eRef].first += contribution * ComplexNumbers::mag2(eRef.w);

        Paths extended{paths};
        for (auto& path : extended) {
          path.emplace_front(edge);
          lNext[&eRef].second.emplace_front(path);
        }
      }
    }

    l = std::move(lNext);
  }

  vEdge approx = rebuild(state, removables, dd);
  dd.incRef(approx);
  dd.decRef(state);
  dd.garbageCollect();

  state = approx;

  return fidelity + budget;
}

} // namespace dd
