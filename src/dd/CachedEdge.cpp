/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/CachedEdge.hpp"

#include "dd/Complex.hpp"
#include "dd/ComplexNumbers.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Edge.hpp"
#include "dd/MemoryManager.hpp"
#include "dd/Node.hpp"
#include "dd/RealNumber.hpp"
#include "ir/Definitions.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <functional>
#include <optional>

namespace dd {

///-----------------------------------------------------------------------------
///                      \n Methods for vector DDs \n
///-----------------------------------------------------------------------------

template <class Node>
template <typename T, isVector<T>>
CachedEdge<Node>
CachedEdge<Node>::normalize(Node* p,
                            const std::array<CachedEdge<Node>, RADIX>& e,
                            MemoryManager& mm, ComplexNumbers& cn) {
  assert(p != nullptr && "Node pointer passed to normalize is null.");
  const auto zero =
      std::array{e[0].w.approximatelyZero(), e[1].w.approximatelyZero()};

  if (zero[0]) {
    if (zero[1]) {
      mm.returnEntry(*p);
      return CachedEdge::zero();
    }
    p->e = {vEdge::zero(), {e[1].p, Complex::one()}};
    return {p, e[1].w};
  }

  if (zero[1]) {
    p->e = {vEdge{e[0].p, Complex::one()}, vEdge::zero()};
    return {p, e[0].w};
  }

  const auto mag2 = std::array{e[0].w.mag2(), e[1].w.mag2()};

  const auto argMax = (mag2[0] + RealNumber::eps >= mag2[1]) ? 0U : 1U;
  const auto& maxMag2 = mag2[argMax];

  const auto argMin = 1U - argMax;
  const auto& minMag2 = mag2[argMin];

  const auto norm = std::sqrt(maxMag2 + minMag2);
  const auto maxMag = std::sqrt(maxMag2);
  const auto commonFactor = norm / maxMag;

  const auto topWeight = e[argMax].w * commonFactor;
  const auto maxWeight = maxMag / norm;
  const auto minWeight = e[argMin].w / topWeight;

  p->e[argMax] = {e[argMax].p, cn.lookup(maxWeight)};
  assert(!p->e[argMax].w.exactlyZero() &&
         "Max edge weight should not be zero.");

  const auto minW = cn.lookup(minWeight);
  if (minW.exactlyZero()) {
    assert(p->e[argMax].w.exactlyOne() &&
           "Edge weight should be one when minWeight is zero.");
    p->e[argMin] = vEdge::zero();
  } else {
    p->e[argMin] = {e[argMin].p, minW};
  }

  return {p, topWeight};
}

///-----------------------------------------------------------------------------
///                      \n Methods for matrix DDs \n
///-----------------------------------------------------------------------------

template <class Node>
template <typename T, isMatrixVariant<T>>
CachedEdge<Node>
CachedEdge<Node>::normalize(Node* p,
                            const std::array<CachedEdge<Node>, NEDGE>& e,
                            MemoryManager& mm, ComplexNumbers& cn) {
  assert(p != nullptr && "Node pointer passed to normalize is null.");
  const auto zero =
      std::array{e[0].w.approximatelyZero(), e[1].w.approximatelyZero(),
                 e[2].w.approximatelyZero(), e[3].w.approximatelyZero()};

  if (std::all_of(zero.begin(), zero.end(), [](auto b) { return b; })) {
    mm.returnEntry(*p);
    return CachedEdge::zero();
  }

  std::optional<std::size_t> argMax = std::nullopt;
  fp maxMag2 = 0.;
  ComplexValue maxVal = 1.;
  // determine max amplitude
  for (auto i = 0U; i < NEDGE; ++i) {
    if (zero[i]) {
      continue;
    }
    const auto& w = e[i].w;
    if (!argMax.has_value()) {
      argMax = i;
      maxMag2 = w.mag2();
      maxVal = w;
    } else {
      if (const auto mag2 = w.mag2(); mag2 - maxMag2 > RealNumber::eps) {
        argMax = i;
        maxMag2 = mag2;
        maxVal = w;
      }
    }
  }
  assert(argMax.has_value() && "argMax should have been set by now");

  const auto argMaxValue = *argMax;
  for (auto i = 0U; i < NEDGE; ++i) {
    // The approximation below is really important for numerical stability.
    // An exactly zero check will lead to numerical instabilities.
    if (zero[i]) {
      p->e[i] = Edge<Node>::zero();
      continue;
    }
    if (i == argMaxValue) {
      p->e[i] = {e[i].p, Complex::one()};
      continue;
    }
    p->e[i] = {e[i].p, cn.lookup(e[i].w / maxVal)};
    if (p->e[i].w.exactlyZero()) {
      p->e[i].p = Node::getTerminal();
    }
  }
  return CachedEdge<Node>{p, maxVal};
}

///-----------------------------------------------------------------------------
///                      \n Explicit instantiations \n
///-----------------------------------------------------------------------------

template struct CachedEdge<vNode>;
template struct CachedEdge<mNode>;
template struct CachedEdge<dNode>;

template CachedEdge<vNode>
CachedEdge<vNode>::normalize(vNode* p,
                             const std::array<CachedEdge<vNode>, RADIX>& e,
                             MemoryManager& mm, ComplexNumbers& cn);

template CachedEdge<mNode>
CachedEdge<mNode>::normalize(mNode* p,
                             const std::array<CachedEdge<mNode>, NEDGE>& e,
                             MemoryManager& mm, ComplexNumbers& cn);

template CachedEdge<dNode>
CachedEdge<dNode>::normalize(dNode* p,
                             const std::array<CachedEdge<dNode>, NEDGE>& e,
                             MemoryManager& mm, ComplexNumbers& cn);

} // namespace dd

namespace std {
template <class Node>
std::size_t hash<dd::CachedEdge<Node>>::operator()(
    const dd::CachedEdge<Node>& e) const noexcept {
  const auto h1 = dd::murmur64(reinterpret_cast<std::size_t>(e.p));
  const auto h2 = std::hash<dd::ComplexValue>{}(e.w);
  return qc::combineHash(h1, h2);
}

template struct hash<dd::CachedEdge<dd::vNode>>;
template struct hash<dd::CachedEdge<dd::mNode>>;
template struct hash<dd::CachedEdge<dd::dNode>>;
} // namespace std
