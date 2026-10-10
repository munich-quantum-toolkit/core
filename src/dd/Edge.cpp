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

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/ScopeExit.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace dd {
namespace {

void traverseVector(const vEdge& edge, const std::complex<fp>& amp,
                    const size_t i,
                    llvm::function_ref<void(size_t, const std::complex<fp>&)> f,
                    const fp threshold) {
  const auto c = amp * static_cast<std::complex<fp>>(edge.w);

  if (threshold > 0. && std::abs(c) < threshold) {
    return;
  }

  if (edge.isTerminal()) {
    f(i, c);
    return;
  }

  // recursive case
  if (const auto& e = edge.p->e[0]; !e.w.exactlyZero()) {
    traverseVector(e, c, i, f, threshold);
  }
  if (const auto& e = edge.p->e[1]; !e.w.exactlyZero()) {
    traverseVector(e, c, i | (1ULL << edge.p->v), f, threshold);
  }
}

void traverseMatrixImpl(const mEdge& edge, const std::complex<fp>& amp,
                        const size_t i, const size_t j,
                        const MatrixEntryFunc& f, const size_t level,
                        const fp threshold) {
  const auto c = amp * static_cast<std::complex<fp>>(edge.w);

  if (threshold > 0. && std::abs(c) < threshold) {
    return;
  }

  if (level == 0) {
    assert(edge.isTerminal());
    f(i, j, c);
    return;
  }

  const auto nextLevel = static_cast<Qubit>(level - 1U);
  const size_t x = i | (1ULL << nextLevel);
  const size_t y = j | (1ULL << nextLevel);
  if (edge.isTerminal() || edge.p->v < nextLevel) {
    traverseMatrixImpl(edge, amp, i, j, f, nextLevel, threshold);
    traverseMatrixImpl(edge, amp, x, y, f, nextLevel, threshold);
    return;
  }

  const auto coords = {std::pair{i, j}, {i, y}, {x, j}, {x, y}};
  size_t k = 0U;
  for (const auto& [a, b] : coords) {
    if (auto const& e = edge.p->e[k++]; !e.w.exactlyZero()) {
      traverseMatrixImpl(e, c, a, b, f, nextLevel, threshold);
    }
  }
}

} // namespace

//-----------------------------------------------------------------------------
//                      \n General purpose methods \n
//-----------------------------------------------------------------------------

template <class Node>
auto Edge<Node>::getValueByPath(const std::size_t numQubits,
                                const std::string& decisions) const
    -> std::complex<fp> {
  if (decisions.size() < numQubits) {
    throw std::out_of_range(
        "Decision path is shorter than the number of qubits.");
  }
  const auto path = std::string_view(decisions).substr(0, numQubits);
  if (path.find_first_not_of(IsVector<Node> ? "01" : "0123") !=
      std::string_view::npos) {
    throw std::invalid_argument("Decision path contains an invalid digit.");
  }
  auto c = static_cast<std::complex<fp>>(w);
  if constexpr (IsVector<Node>) {
    if (isTerminal()) {
      return c;
    }
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
  if (isTerminal()) {
    return 1U;
  }
  static constexpr std::size_t NODECOUNT_BUCKETS = 200000U;
  static thread_local std::unordered_set<const Node*> visited{
      NODECOUNT_BUCKETS,
  };
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

//-----------------------------------------------------------------------------
//                      \n Methods for vector DDs \n
//-----------------------------------------------------------------------------

auto normalize(vNode* p, const std::array<Edge<vNode>, RADIX>& e,
               MemoryManager& mm, ComplexNumbers& cn) -> Edge<vNode> {
  assert(p != nullptr && "Node pointer passed to normalize is null.");
  const auto zero = std::array{e[0].w.exactlyZero(), e[1].w.exactlyZero()};

  if (zero[0]) {
    if (zero[1]) {
      mm.returnEntry(*p);
      return vEdge::zero();
    }
    p->e = e;
    vEdge r{.p = p, .w = e[1].w};
    p->e[1].w = Complex::one();
    return r;
  }

  p->e = e;
  if (zero[1]) {
    vEdge r{.p = p, .w = e[0].w};
    p->e[0].w = Complex::one();
    return r;
  }

  const auto weights = std::array{
      static_cast<ComplexValue>(e[0].w),
      static_cast<ComplexValue>(e[1].w),
  };

  const auto mag2 = std::array{weights[0].mag2(), weights[1].mag2()};

  // Keep the dominant phase independent of the incoming scale.
  const auto argMax =
      mag2[1] - mag2[0] > RealNumber::eps * std::max(mag2[0], mag2[1]) ? 1U
                                                                       : 0U;
  const auto& maxMag2 = mag2[argMax];

  const auto argMin = 1U - argMax;
  const auto& minMag2 = mag2[argMin];

  const auto norm = std::sqrt(maxMag2 + minMag2);
  const auto maxMag = std::sqrt(maxMag2);
  const auto maxWeight = maxMag / norm;
  p->e[argMax].w = cn.lookup(maxWeight);
  // Preserve the dominant coefficient after interning its normalized weight.
  const auto topWeight = weights[argMax] / RealNumber::val(p->e[argMax].w.r);
  assert(!p->e[argMax].w.exactlyZero() &&
         "Max edge weight should not be zero.");

  vEdge r = {.p = p, .w = cn.lookup(topWeight)};
  assert(!r.w.exactlyZero() && "Top edge weight should not be zero.");

  // Lookup can round the top weight; normalize against the stored value.
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

auto getValueByIndex(const vEdge& edge, const size_t i) -> std::complex<fp> {
  const auto numQubits =
      edge.isTerminal() ? 0U : static_cast<size_t>(edge.p->v) + 1U;
  if (numQubits < std::numeric_limits<size_t>::digits &&
      (i >> numQubits) != 0U) {
    throw std::out_of_range("Vector index is out of range.");
  }
  auto current = edge;
  auto amplitude = static_cast<std::complex<fp>>(current.w);
  while (!current.isTerminal()) {
    const auto q = current.p->v;
    const auto bit =
        q < std::numeric_limits<size_t>::digits ? (i >> q) & 1U : 0U;
    current = current.p->e[bit];
    amplitude *= static_cast<std::complex<fp>>(current.w);
  }
  return amplitude;
}

auto getVector(const vEdge& edge, const fp threshold) -> CVec {
  if (edge.isTerminal()) {
    return {static_cast<std::complex<fp>>(edge.w)};
  }

  const size_t dim = 2ULL << edge.p->v;
  auto vec = CVec(dim, 0.);
  traverseVector(
      edge, 1., 0,
      [&vec](const size_t i, const std::complex<fp>& c) { vec.at(i) = c; },
      threshold);
  return vec;
}

auto getSparseVector(const vEdge& edge, const fp threshold) -> SparseCVec {
  if (edge.isTerminal()) {
    return {{0, static_cast<std::complex<fp>>(edge.w)}};
  }

  auto vec = SparseCVec{};
  traverseVector(
      edge, 1., 0,
      [&vec](const size_t i, const std::complex<fp>& c) { vec[i] = c; },
      threshold);
  return vec;
}

auto printVector(const vEdge& edge) -> void {
  constexpr auto precision = 3;
  const auto oldPrecision = std::cout.precision();
  const auto restorePrecision =
      llvm::scope_exit([oldPrecision] { std::cout.precision(oldPrecision); });
  std::cout << std::setprecision(precision);

  if (edge.isTerminal()) {
    std::cout << "0: " << static_cast<std::complex<fp>>(edge.w) << "\n";
    return;
  }
  const size_t element = 2ULL << edge.p->v;
  for (auto i = 0ULL; i < element; i++) {
    const auto amplitude = getValueByIndex(edge, i);
    const auto n = static_cast<size_t>(edge.p->v) + 1U;
    for (auto j = n; j > 0; --j) {
      std::cout << ((i >> (j - 1)) & 1ULL);
    }
    std::cout << ": " << amplitude << "\n";
  }
  std::cout << std::flush;
}

auto addToVector(const vEdge& edge, CVec& amplitudes) -> void {
  if (edge.isTerminal()) {
    amplitudes[0] += static_cast<std::complex<fp>>(edge.w);
    return;
  }

  traverseVector(
      edge, 1., 0,
      [&amplitudes](const size_t i, const std::complex<fp>& c) {
        amplitudes[i] += c;
      },
      0.);
}

//-----------------------------------------------------------------------------
//                      \n Methods for matrix DDs \n
//-----------------------------------------------------------------------------
auto normalize(mNode* p, const std::array<Edge<mNode>, NEDGE>& e,
               MemoryManager& mm, ComplexNumbers& cn) -> Edge<mNode> {
  assert(p != nullptr && "Node pointer passed to normalize is null.");
  const auto zero = std::array{
      e[0].w.exactlyZero(),
      e[1].w.exactlyZero(),
      e[2].w.exactlyZero(),
      e[3].w.exactlyZero(),
  };

  if (std::ranges::all_of(zero, [](auto b) { return b; })) {
    mm.returnEntry(*p);
    return mEdge::zero();
  }

  auto weights = std::array{
      static_cast<ComplexValue>(e[0].w),
      static_cast<ComplexValue>(e[1].w),
      static_cast<ComplexValue>(e[2].w),
      static_cast<ComplexValue>(e[3].w),
  };

  // The incoming scale does not affect normalized coefficients. Remove it
  // before squared magnitudes and complex division can overflow or underflow.
  fp maxComponent = 0.;
  for (const auto& w : weights) {
    maxComponent = std::max({maxComponent, std::abs(w.r), std::abs(w.i)});
  }
  if (maxComponent < 1. || maxComponent >= 2.) {
    const auto scale = std::scalbn(1., std::ilogb(maxComponent));
    for (auto& w : weights) {
      w = w / scale;
    }
  }

  std::optional<size_t> argMax = std::nullopt;
  fp maxMag2 = 0.;
  auto maxVal = Complex::one();
  // determine max amplitude
  for (auto i = 0U; i < NEDGE; ++i) {
    if (zero[i]) {
      p->e[i] = mEdge::zero();
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
  const auto argMaxWeight = weights[argMaxValue];
  for (auto i = 0U; i < NEDGE; ++i) {
    if (zero[i]) {
      continue;
    }
    if (i == argMaxValue) {
      p->e[i] = {.p = e[i].p, .w = Complex::one()};
      continue;
    }
    p->e[i] = {.p = e[i].p, .w = cn.lookup(weights[i] / argMaxWeight)};
    if (p->e[i].w.exactlyZero()) {
      p->e[i].p = mNode::getTerminal();
    }
  }
  return mEdge{.p = p, .w = maxVal};
}

auto getValueByIndex(const mEdge& edge, const size_t numQubits, const size_t i,
                     const size_t j) -> std::complex<fp> {
  if (numQubits < std::numeric_limits<size_t>::digits &&
      ((i >> numQubits) != 0U || (j >> numQubits) != 0U)) {
    throw std::out_of_range("Matrix index is out of range.");
  }
  if (edge.isTerminal()) {
    return i == j ? static_cast<std::complex<fp>>(edge.w) : 0.;
  }

  auto current = edge;
  auto amplitude = static_cast<std::complex<fp>>(current.w);
  for (auto level = numQubits; level > 0; --level) {
    const auto q = level - 1;
    const auto rowBit =
        q < std::numeric_limits<size_t>::digits ? (i >> q) & 1U : 0U;
    const auto colBit =
        q < std::numeric_limits<size_t>::digits ? (j >> q) & 1U : 0U;
    if (current.isTerminal() || current.p->v != q) {
      if (current.isZeroTerminal() || rowBit != colBit) {
        return 0.;
      }
    } else {
      current = current.p->e[(2 * rowBit) + colBit];
      amplitude *= static_cast<std::complex<fp>>(current.w);
    }
  }
  return amplitude;
}

auto getMatrix(const mEdge& edge, const size_t numQubits, const fp threshold)
    -> CMat {
  if (numQubits == 0U) {
    return CMat{1, {static_cast<std::complex<fp>>(edge.w)}};
  }

  const size_t dim = 1ULL << numQubits;
  auto mat = CMat(dim, CVec(dim, 0.));
  traverseMatrix(
      edge, 1, 0ULL, 0ULL,
      [&mat](const size_t i, const size_t j, const std::complex<fp>& c) {
        mat.at(i).at(j) = c;
      },
      numQubits, threshold);
  return mat;
}

auto getSparseMatrix(const mEdge& edge, const size_t numQubits,
                     const fp threshold) -> SparseCMat {
  if (numQubits == 0U) {
    return {{{0U, 0U}, static_cast<std::complex<fp>>(edge.w)}};
  }

  auto mat = SparseCMat{};
  traverseMatrix(
      edge, 1, 0ULL, 0ULL,
      [&mat](const size_t i, const size_t j, const std::complex<fp>& c) {
        mat[{i, j}] = c;
      },
      numQubits, threshold);

  return mat;
}

auto printMatrix(const mEdge& edge, const size_t numQubits) -> void {
  constexpr auto precision = 3;
  const auto oldPrecision = std::cout.precision();
  const auto restorePrecision =
      llvm::scope_exit([oldPrecision] { std::cout.precision(oldPrecision); });
  std::cout << std::setprecision(precision);

  if (numQubits == 0U) {
    std::cout << static_cast<std::complex<fp>>(edge.w) << "\n";
    return;
  }
  assert(edge.isTerminal() || numQubits > edge.p->v);
  const size_t element = 1ULL << numQubits;
  for (auto i = 0ULL; i < element; ++i) {
    for (auto j = 0ULL; j < element; ++j) {
      const auto amplitude = getValueByIndex(edge, numQubits, i, j);
      std::cout << amplitude << " ";
    }
    std::cout << "\n";
  }
  std::cout << std::flush;
}

void traverseMatrix(const mEdge& edge, const std::complex<fp>& amp,
                    const size_t i, const size_t j,
                    // Keep one callback copy at the public traversal boundary.
                    // NOLINTNEXTLINE(performance-unnecessary-value-param)
                    MatrixEntryFunc f, const size_t level, const fp threshold) {
  traverseMatrixImpl(edge, amp, i, j, f, level, threshold);
}

//-----------------------------------------------------------------------------
//                      \n Explicit instantiations \n
//-----------------------------------------------------------------------------

template struct Edge<vNode>;
template struct Edge<mNode>;

} // namespace dd

//-----------------------------------------------------------------------------
//                         \n Hash related code \n
//-----------------------------------------------------------------------------

template <class Node>
auto std::hash<dd::Edge<Node>>::operator()(
    const dd::Edge<Node>& e) const noexcept -> std::size_t {
  const auto h1 = dd::murmur64(e.p == nullptr ? 0U : e.p->id);
  const auto h2 = std::hash<dd::Complex>{}(e.w);
  return dd::combineHash(h1, h2);
}

// NOLINTNEXTLINE(bugprone-std-namespace-modification)
template struct std::hash<dd::Edge<dd::vNode>>;
// NOLINTNEXTLINE(bugprone-std-namespace-modification)
template struct std::hash<dd::Edge<dd::mNode>>;
