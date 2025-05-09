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
#include <queue>
#include <unordered_map>

namespace dd {
namespace {
// Queue for iterative-deepening search.
using Queue = std::queue<const vEdge*>;
// Map that holds edges required for rebuilding the DD.
// If the value of an edge is true it will be deleted.
using PathFlags = std::unordered_map<const vEdge*, bool>;
// Maps edges to their respective contribution.
using Contributions = std::unordered_map<const vEdge*, double>;
// Maps edges to their respective parent edges.
using Parents =
    std::unordered_map<const vEdge*, std::forward_list<const vEdge*>>;

/**
 * @brief Recursively rebuild DD depth-first.

 * @details Only visits the paths from the root edge
 * to ∀e ∈ {e | f[e] = true} by using @p f.
 */
vEdge rebuild(const vEdge& e, const PathFlags& f, Package& dd) {
  // If the edge isn't contained in the pathflags,
  // we keep the edge as it is.
  if (f.find(&e) == f.end()) {
    return e;
  }

  // If the pathflag is true, delete the edge.
  if (f.at(&e)) {
    return vEdge::zero();
  }

  // Otherwise, if the pathflag is false, traverse down.
  const std::array<vEdge, RADIX> edges{
      rebuild(e.p->e[0], f, dd),
      rebuild(e.p->e[1], f, dd),
  };

  vEdge eNew = dd.makeDDNode(e.p->v, edges);
  eNew.w = e.w;

  return eNew;
}

/**
 * @brief Flag (or mark) the path from edge @p e to the root node.
 */
void markParentEdges(const vEdge* e, const Parents& m, PathFlags& f) {
  Queue q{};
  q.emplace(e);
  while (!q.empty()) {
    const vEdge* eX = q.front();
    q.pop();
    for (const vEdge* eP : m.at(eX)) {
      f[eP] = false;
      q.emplace(eP);
    }
  }
}
}; // namespace

double approximate(VectorDD& state, const double fidelity, Package& dd) {
  Queue q{};
  q.emplace(&state);

  PathFlags f{};
  Parents m{{&state, {}}};
  Contributions c{{&state, 1.}};

  double budget = 1 - fidelity;
  while (!q.empty() && budget > 0) {
    const vEdge* e = q.front();
    const double contribution = c[e];
    q.pop();

    if (contribution <= budget) {
      f[e] = true;
      markParentEdges(e, m, f);
      budget -= contribution;
      continue;
    }

    if (e->isTerminal()) {
      continue;
    }

    const vNode* n = e->p;
    for (const auto& eChildRef : n->e) {
      const vEdge* eChild = &eChildRef;

      if (eChild->w.exactlyZero()) { // Don't add zero terminals.
        continue;
      }

      if (c.find(eChild) == c.cend()) {
        q.emplace(eChild); // Add to queue.
        c[eChild] = 0.;    // Not necessary, but better than implicit.
      }

      // An edge may have multiple parent edges, and hence, add (instead of
      // assign) the full contribution.
      const double childContribution = ComplexNumbers::mag2(eChild->w);
      c[eChild] += contribution * childContribution;

      m[eChild].emplace_front(e); // Map child to parent.
    }
  }

  const vEdge approx = rebuild(state, f, dd);
  dd.incRef(approx);
  dd.decRef(state);
  dd.garbageCollect();
  state = approx;

  return fidelity + budget;
}

} // namespace dd
