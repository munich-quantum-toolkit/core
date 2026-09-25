/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
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

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <optional>

namespace dd {

///-----------------------------------------------------------------------------
///                      \n Methods for vector DDs \n
///-----------------------------------------------------------------------------

template <class Node>
auto CachedEdge<Node>::normalize(Node* p,
                                 const std::array<CachedEdge, RADIX>& e,
                                 MemoryManager& mm, ComplexNumbers& cn)
    -> CachedEdge
  requires IsVector<Node>
{
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

  /// Project nearly equal or opposite coefficients before normalization can
  /// amplify their difference. For unit-norm children, the local Euclidean
  /// error is at most eps before roundoff.
  for (const fp sign : {1., -1.}) {
    const auto other = e[1].w * sign;
    if (e[0].w.approximatelyEquals(other)) {
      p->e[0] = {e[0].p, cn.lookup(SQRT2_2)};
      p->e[1] = {e[1].p, cn.lookup(sign * SQRT2_2)};
      return {p, (e[0].w + other) * SQRT2_2};
    }
  }

  const auto mag2 = std::array{e[0].w.mag2(), e[1].w.mag2()};

  /// Keep the dominant phase independent of the incoming scale.
  const auto argMax =
      mag2[1] - mag2[0] > RealNumber::eps * std::max(mag2[0], mag2[1]) ? 1U
                                                                       : 0U;
  const auto& maxMag2 = mag2[argMax];

  const auto argMin = 1U - argMax;
  const auto& minMag2 = mag2[argMin];

  const auto norm = std::sqrt(maxMag2 + minMag2);
  const auto maxMag = std::sqrt(maxMag2);
  const auto maxWeight = maxMag / norm;
  p->e[argMax] = {e[argMax].p, cn.lookup(maxWeight)};
  /// Preserve the dominant coefficient after interning its normalized weight.
  const auto topWeight = e[argMax].w / RealNumber::val(p->e[argMax].w.r);
  const auto minWeight = e[argMin].w / topWeight;
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
auto CachedEdge<Node>::normalize(Node* p,
                                 const std::array<CachedEdge, NEDGE>& e,
                                 MemoryManager& mm, ComplexNumbers& cn)
    -> CachedEdge
  requires IsMatrix<Node>
{
  assert(p != nullptr && "Node pointer passed to normalize is null.");
  const auto zero = std::array{
      e[0].w.approximatelyZero(),
      e[1].w.approximatelyZero(),
      e[2].w.approximatelyZero(),
      e[3].w.approximatelyZero(),
  };

  if (std::all_of(zero.begin(), zero.end(), [](auto b) { return b; })) {
    mm.returnEntry(*p);
    return CachedEdge::zero();
  }

  /// The incoming scale does not affect normalized coefficients. Remove it
  /// before squared magnitudes and complex division can overflow or underflow.
  const auto maxComponent = std::max({
      std::abs(e[0].w.r),
      std::abs(e[0].w.i),
      std::abs(e[1].w.r),
      std::abs(e[1].w.i),
      std::abs(e[2].w.r),
      std::abs(e[2].w.i),
      std::abs(e[3].w.r),
      std::abs(e[3].w.i),
  });
  auto weights = std::array{e[0].w, e[1].w, e[2].w, e[3].w};
  if (maxComponent < 1. || maxComponent >= 2.) {
    const auto scale = std::scalbn(1., std::ilogb(maxComponent));
    for (auto& w : weights) {
      w = w / scale;
    }
  }

  std::optional<std::size_t> argMax = std::nullopt;
  fp maxMag2 = 0.;
  ComplexValue maxVal = 1.;
  // determine max amplitude
  for (auto i = 0U; i < NEDGE; ++i) {
    if (zero[i]) {
      continue;
    }
    const auto& w = weights[i];
    if (!argMax.has_value()) {
      argMax = i;
      maxMag2 = w.mag2();
      maxVal = e[i].w;
    } else {
      if (const auto mag2 = w.mag2();
          mag2 - maxMag2 > RealNumber::eps * std::max(mag2, maxMag2)) {
        argMax = i;
        maxMag2 = mag2;
        maxVal = e[i].w;
      }
    }
  }
  assert(argMax.has_value() && "argMax should have been set by now");

  const auto argMaxValue = *argMax;
  for (auto i = 0U; i < NEDGE; ++i) {
    /// Treat weights within tolerance as zero before normalization amplifies
    /// them.
    if (zero[i]) {
      p->e[i] = Edge<Node>::zero();
      continue;
    }
    if (i == argMaxValue) {
      p->e[i] = {e[i].p, Complex::one()};
      continue;
    }
    p->e[i] = {e[i].p, cn.lookup(weights[i] / weights[argMaxValue])};
    if (p->e[i].w.exactlyZero()) {
      p->e[i].p = Node::getTerminal();
    }
  }
  return CachedEdge{p, maxVal};
}

///-----------------------------------------------------------------------------
///                      \n Explicit instantiations \n
///-----------------------------------------------------------------------------

template struct CachedEdge<vNode>;
template struct CachedEdge<mNode>;

} // namespace dd

template <class Node>
auto std::hash<dd::CachedEdge<Node>>::operator()(
    const dd::CachedEdge<Node>& e) const noexcept -> std::size_t {
  const auto h1 = dd::murmur64(reinterpret_cast<std::size_t>(e.p));
  const auto h2 = std::hash<dd::ComplexValue>{}(e.w);
  return dd::combineHash(h1, h2);
}

// NOLINTNEXTLINE(bugprone-std-namespace-modification)
template struct std::hash<dd::CachedEdge<dd::vNode>>;
// NOLINTNEXTLINE(bugprone-std-namespace-modification)
template struct std::hash<dd::CachedEdge<dd::mNode>>;
