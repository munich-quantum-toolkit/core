/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/Edge.hpp"

#include "dd/Complex.hpp"
#include "dd/ComplexNumbers.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/MemoryManager.hpp"
#include "dd/Node.hpp"
#include "dd/RealNumber.hpp"
#include "ir/Definitions.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>

namespace dd {

///-----------------------------------------------------------------------------
///                      \n General purpose methods \n
///-----------------------------------------------------------------------------

template <class Node>
auto Edge<Node>::getValueByPath(const std::size_t numQubits,
                                const std::string& decisions) const
    -> std::complex<fp> {
  auto c = static_cast<std::complex<fp>>(w);
  if (isTerminal()) {
    return c;
  }

  auto r = *this;
  auto level = numQubits;
  while (level > 0U) {
    const auto tmp = static_cast<std::size_t>(decisions.at(level - 1U) - '0');

    // node is not at the expected level (skipped node)
    if (r.isTerminal() || r.p->v != level - 1U) {
      if (r.isZeroTerminal() || tmp == 1U || tmp == 2U) {
        return 0.;
      }
      --level;
      continue;
    }

    // node is at the expected level
    assert(tmp < r.p->e.size());
    r = r.p->e[tmp];
    c *= static_cast<std::complex<fp>>(r.w);
    --level;
  }
  return c;
}

template <class Node> auto Edge<Node>::size() const -> std::size_t {
  static constexpr std::size_t NODECOUNT_BUCKETS = 200000U;
  static std::unordered_set<const Node*> visited{NODECOUNT_BUCKETS};
  visited.max_load_factor(10);
  visited.clear();
  return size(visited);
}

template <class Node>
auto Edge<Node>::size(std::unordered_set<const Node*>& visited) const
    -> std::size_t {
  visited.emplace(p);
  std::size_t sum = 1U;
  if (!isTerminal()) {
    for (const auto& e : p->e) {
      if (!visited.contains(e.p)) {
        sum += e.size(visited);
      }
    }
  }
  return sum;
}

template <class Node> void Edge<Node>::mark() const noexcept {
  w.mark();
  if (isTerminal() || p->isMarked()) {
    return;
  }
  p->mark();
  for (const Edge<Node>& e : p->e) {
    e.mark();
  }
}

template <class Node> void Edge<Node>::unmark() const noexcept {
  w.unmark();
  if (isTerminal() || !p->isMarked()) {
    return;
  }
  p->unmark();
  for (const Edge<Node>& e : p->e) {
    e.unmark();
  }
}

///-----------------------------------------------------------------------------
///                      \n Methods for vector DDs \n
///-----------------------------------------------------------------------------

template <class Node>
auto Edge<Node>::normalize(Node* p, const std::array<Edge, RADIX>& e,
                           MemoryManager& mm, ComplexNumbers& cn) -> Edge
  requires IsVector<Node>
{
  assert(p != nullptr && "Node pointer passed to normalize is null.");
  const auto zero = std::array{e[0].w.exactlyZero(), e[1].w.exactlyZero()};

  if (zero[0]) {
    if (zero[1]) {
      mm.returnEntry(*p);
      return vEdge::zero();
    }
    p->e = e;
    vEdge r{p, e[1].w};
    p->e[1].w = Complex::one();
    return r;
  }

  p->e = e;
  if (zero[1]) {
    vEdge r{p, e[0].w};
    p->e[0].w = Complex::one();
    return r;
  }

  const auto weights = std::array{static_cast<ComplexValue>(e[0].w),
                                  static_cast<ComplexValue>(e[1].w)};

  const auto mag2 = std::array{weights[0].mag2(), weights[1].mag2()};

  const auto argMax = (mag2[0] + RealNumber::eps >= mag2[1]) ? 0U : 1U;
  const auto& maxMag2 = mag2[argMax];

  const auto argMin = 1U - argMax;
  const auto& minMag2 = mag2[argMin];

  const auto norm = std::sqrt(maxMag2 + minMag2);
  const auto maxMag = std::sqrt(maxMag2);
  const auto commonFactor = norm / maxMag;

  const auto topWeight = weights[argMax] * commonFactor;
  const auto maxWeight = maxMag / norm;
  p->e[argMax].w = cn.lookup(maxWeight);
  assert(!p->e[argMax].w.exactlyZero() &&
         "Max edge weight should not be zero.");

  vEdge r = {p, cn.lookup(topWeight)};
  assert(!r.w.exactlyZero() && "Top edge weight should not be zero.");

  // In theory, the more efficient computation here would be
  //              weights[argMin] / topWeight
  // However, the lookup of the top weight can slightly change its value.
  // Therefore, we use the following computation instead, which accounts for the
  // potential difference (at the cost of a Complex->ComplexValue conversion).
  const auto minWeight = weights[argMin] / r.w;
  auto& min = p->e[argMin];
  min.w = cn.lookup(minWeight);
  if (min.w.exactlyZero()) {
    assert(p->e[argMax].w.exactlyOne() &&
           "Edge weight should be one when minWeight is zero.");
    min.p = vNode::getTerminal();
  }

  return r;
}

template <class Node>
auto Edge<Node>::getValueByIndex(const std::size_t i) const -> std::complex<fp>
  requires IsVector<Node>
{
  auto bitwidth = static_cast<Qubit>(std::log2(i + 1U));

  if (!isTerminal()) {
    bitwidth = std::max(bitwidth, static_cast<Qubit>(p->v + 1U));
  }

  auto decisions = std::string(bitwidth, '0');
  for (auto j = 0U; j < bitwidth; ++j) {
    if ((i & (1ULL << j)) != 0U) {
      decisions[j] = '1';
    }
  }

  return getValueByPath(bitwidth, decisions);
}

template <class Node>
auto Edge<Node>::getVector(const fp threshold) const -> CVec
  requires IsVector<Node>
{
  if (isTerminal()) {
    return {static_cast<std::complex<fp>>(w)};
  }

  const std::size_t dim = 2ULL << p->v;
  auto vec = CVec(dim, 0.);
  traverseVector(
      1., 0,
      [&vec](const std::size_t i, const std::complex<fp>& c) { vec.at(i) = c; },
      threshold);
  return vec;
}

template <class Node>
auto Edge<Node>::getSparseVector(const fp threshold) const -> SparseCVec
  requires IsVector<Node>
{
  if (isTerminal()) {
    return {{0, static_cast<std::complex<fp>>(w)}};
  }

  auto vec = SparseCVec{};
  traverseVector(
      1., 0,
      [&vec](const std::size_t i, const std::complex<fp>& c) { vec[i] = c; },
      threshold);
  return vec;
}

template <class Node>
auto Edge<Node>::printVector() const -> void
  requires IsVector<Node>
{
  constexpr auto precision = 3;
  const auto oldPrecision = std::cout.precision();
  std::cout << std::setprecision(precision);

  if (isTerminal()) {
    std::cout << "0: " << static_cast<std::complex<fp>>(w) << "\n";
    return;
  }
  const std::size_t element = 2ULL << p->v;
  for (auto i = 0ULL; i < element; i++) {
    const auto amplitude = getValueByIndex(i);
    const auto n = static_cast<std::size_t>(p->v) + 1U;
    for (auto j = n; j > 0; --j) {
      std::cout << ((i >> (j - 1)) & 1ULL);
    }
    std::cout << ": " << amplitude << "\n";
  }
  std::cout << std::setprecision(static_cast<int>(oldPrecision));
  std::cout << std::flush;
}

template <class Node>
auto Edge<Node>::addToVector(CVec& amplitudes) const -> void
  requires IsVector<Node>
{
  if (isTerminal()) {
    amplitudes[0] += static_cast<std::complex<fp>>(w);
    return;
  }

  traverseVector(1., 0,
                 [&amplitudes](const std::size_t i, const std::complex<fp>& c) {
                   amplitudes[i] += c;
                 });
}

template <class Node>
void Edge<Node>::traverseVector(const std::complex<fp>& amp,
                                const std::size_t i, AmplitudeFunc f,
                                const fp threshold) const
  requires IsVector<Node>
{
  // calculate new accumulated amplitude
  const auto c = amp * static_cast<std::complex<fp>>(w);

  if (std::abs(c) < threshold) {
    return;
  }

  if (isTerminal()) {
    f(i, c);
    return;
  }

  // recursive case
  if (const auto& e = p->e[0]; !e.w.exactlyZero()) {
    e.traverseVector(c, i, f, threshold);
  }
  if (const auto& e = p->e[1]; !e.w.exactlyZero()) {
    e.traverseVector(c, i | (1ULL << p->v), f, threshold);
  }
}

///-----------------------------------------------------------------------------
///                      \n Methods for matrix DDs \n
///-----------------------------------------------------------------------------
template <class Node>
auto Edge<Node>::normalize(Node* p, const std::array<Edge, NEDGE>& e,
                           MemoryManager& mm, ComplexNumbers& cn) -> Edge
  requires IsMatrix<Node>
{
  assert(p != nullptr && "Node pointer passed to normalize is null.");
  const auto zero = std::array{e[0].w.exactlyZero(), e[1].w.exactlyZero(),
                               e[2].w.exactlyZero(), e[3].w.exactlyZero()};

  if (std::all_of(zero.begin(), zero.end(), [](auto b) { return b; })) {
    mm.returnEntry(*p);
    return Edge::zero();
  }

  const auto weights = std::array{
      static_cast<ComplexValue>(e[0].w), static_cast<ComplexValue>(e[1].w),
      static_cast<ComplexValue>(e[2].w), static_cast<ComplexValue>(e[3].w)};

  std::optional<std::size_t> argMax = std::nullopt;
  fp maxMag2 = 0.;
  auto maxVal = Complex::one();
  // determine max amplitude
  for (auto i = 0U; i < NEDGE; ++i) {
    if (zero[i]) {
      p->e[i] = Edge::zero();
      continue;
    }
    const auto& w = weights[i];
    if (!argMax.has_value()) {
      argMax = i;
      maxMag2 = w.mag2();
      maxVal = e[i].w;
    } else {
      if (const auto mag2 = w.mag2(); mag2 - maxMag2 > RealNumber::eps) {
        argMax = i;
        maxMag2 = mag2;
        maxVal = e[i].w;
      }
    }
  }
  assert(argMax.has_value() && "argMax should have been set by now");

  const auto argMaxValue = *argMax;
  const auto argMaxWeight = weights[argMaxValue];
  for (auto i = 0U; i < NEDGE; ++i) {
    if (zero[i]) {
      continue;
    }
    if (i == argMaxValue) {
      p->e[i] = {e[i].p, Complex::one()};
      continue;
    }
    p->e[i] = {e[i].p, cn.lookup(weights[i] / argMaxWeight)};
    if (p->e[i].w.exactlyZero()) {
      p->e[i].p = Node::getTerminal();
    }
  }
  return Edge{p, maxVal};
}

template <class Node>
auto Edge<Node>::getValueByIndex(const std::size_t numQubits,
                                 const std::size_t i, const std::size_t j) const
    -> std::complex<fp>
  requires IsMatrix<Node>
{
  if (isTerminal()) {
    return static_cast<std::complex<fp>>(w);
  }

  auto decisions = std::string(numQubits, '0');
  for (auto k = 0U; k < numQubits; ++k) {
    if ((i & (1ULL << k)) != 0U) {
      decisions[k] = '2';
    }
  }
  for (auto k = 0U; k < numQubits; ++k) {
    if ((j & (1ULL << k)) != 0U) {
      if (decisions[k] == '2') {
        decisions[k] = '3';
      } else {
        decisions[k] = '1';
      }
    }
  }

  return getValueByPath(numQubits, decisions);
}

template <class Node>
auto Edge<Node>::getMatrix(const std::size_t numQubits,
                           const fp threshold) const -> CMat
  requires IsMatrix<Node>
{
  if (numQubits == 0U) {
    return CMat{1, {static_cast<std::complex<fp>>(w)}};
  }

  const std::size_t dim = 1ULL << numQubits;
  auto mat = CMat(dim, CVec(dim, 0.));
  traverseMatrix(
      1, 0ULL, 0ULL,
      [&mat](const std::size_t i, const std::size_t j,
             const std::complex<fp>& c) { mat.at(i).at(j) = c; },
      numQubits, threshold);
  return mat;
}

template <class Node>
auto Edge<Node>::getSparseMatrix(const std::size_t numQubits,
                                 const fp threshold) const -> SparseCMat
  requires IsMatrix<Node>
{
  if (numQubits == 0U) {
    return {{{0U, 0U}, static_cast<std::complex<fp>>(w)}};
  }

  auto mat = SparseCMat{};
  traverseMatrix(
      1, 0ULL, 0ULL,
      [&mat](const std::size_t i, const std::size_t j,
             const std::complex<fp>& c) { mat[{i, j}] = c; },
      numQubits, threshold);

  return mat;
}

template <class Node>
auto Edge<Node>::printMatrix(const std::size_t numQubits) const -> void
  requires IsMatrix<Node>
{
  constexpr auto precision = 3;
  const auto oldPrecision = std::cout.precision();
  std::cout << std::setprecision(precision);

  if (numQubits == 0U) {
    std::cout << static_cast<std::complex<fp>>(w) << "\n";
    return;
  }
  // total number of qubits should not be lower than the highest qubit index
  assert(isTerminal() || numQubits > p->v);
  const std::size_t element = 1ULL << numQubits;
  for (auto i = 0ULL; i < element; ++i) {
    for (auto j = 0ULL; j < element; ++j) {
      const auto amplitude = getValueByIndex(numQubits, i, j);
      std::cout << amplitude << " ";
    }
    std::cout << "\n";
  }
  std::cout << std::setprecision(static_cast<int>(oldPrecision));
  std::cout << std::flush;
}

template <class Node>
void Edge<Node>::traverseMatrix(const std::complex<fp>& amp,
                                const std::size_t i, const std::size_t j,
                                MatrixEntryFunc f, const std::size_t level,
                                const fp threshold) const
  requires IsMatrix<Node>
{
  // calculate new accumulated amplitude
  const auto c = amp * static_cast<std::complex<fp>>(w);

  if (std::abs(c) < threshold) {
    return;
  }

  if (level == 0) {
    assert(isTerminal());
    f(i, j, c);
    return;
  }

  const auto nextLevel = static_cast<Qubit>(level - 1U);
  const std::size_t x = i | (1ULL << nextLevel);
  const std::size_t y = j | (1ULL << nextLevel);
  if (isTerminal() || p->v < nextLevel) {
    traverseMatrix(amp, i, j, f, nextLevel, threshold);
    traverseMatrix(amp, x, y, f, nextLevel, threshold);
    return;
  }

  const auto coords = {std::pair{i, j}, {i, y}, {x, j}, {x, y}};
  std::size_t k = 0U;
  for (const auto& [a, b] : coords) {
    if (auto& e = p->e[k++]; !e.w.exactlyZero()) {
      e.traverseMatrix(c, a, b, f, nextLevel, threshold);
    }
  }
}

///-----------------------------------------------------------------------------
///                      \n Explicit instantiations \n
///-----------------------------------------------------------------------------

template struct Edge<vNode>;
template struct Edge<mNode>;

} // namespace dd

///-----------------------------------------------------------------------------
///                         \n Hash related code \n
///-----------------------------------------------------------------------------

namespace std {
template <class Node>
auto hash<dd::Edge<Node>>::operator()(const dd::Edge<Node>& e) const noexcept
    -> std::size_t {
  const auto h1 = dd::murmur64(reinterpret_cast<std::size_t>(e.p));
  const auto h2 = std::hash<dd::Complex>{}(e.w);
  return qc::combineHash(h1, h2);
}

template struct hash<dd::Edge<dd::vNode>>;
template struct hash<dd::Edge<dd::mNode>>;
} // namespace std
