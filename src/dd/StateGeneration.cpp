/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/StateGeneration.hpp"

#include "dd/CachedEdge.hpp"
#include "dd/ComplexNumbers.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Edge.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "dd/RealNumber.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace dd {
namespace {
/**
 * @brief Validate that the package is suitable for the use with up to @p n
 * qubits.
 * @throws `std::invalid_argument`, if `dd.qubits() < n`.
 */
void suitablePackage(const std::size_t n, const Package& dd) {
  const std::size_t nqubits = dd.qubits();
  if (nqubits < n) {
    throw std::invalid_argument{
        "Requested state with " + std::to_string(n) +
        " qubits, but current package configuration only supports up to " +
        std::to_string(nqubits) +
        " qubits. Please allocate a larger package instance."};
  }
}

/**
 * @brief Constructs a decision diagram (DD) from a state vector using a
 * recursive algorithm.
 *
 * @param begin Iterator pointing to the beginning of the state vector.
 * @param end Iterator pointing to the end of the state vector.
 * @param v The current level of recursion. Starts at the highest level of
 * the state vector (log base 2 of the vector size - 1).
 * @param dd The DD package to use.
 * @return A vCachedEdge representing the root node of the created DD.
 *
 * @details This function recursively breaks down the state vector into halves
 * until each half has only one element. At each level of recursion, two new
 * edges are created, one for each half of the state vector. The two resulting
 * decision diagram edges are used to create a new decision diagram node at
 * the current level, and this node is returned as the result of the current
 * recursive call. At the base case of recursion, the state vector has only
 * two elements, which are converted into terminal nodes of the decision
 * diagram.
 *
 * @note This function assumes that the state vector size is a power of two.
 */
vCachedEdge makeStateFromVector(const CVec::const_iterator& begin,
                                const CVec::const_iterator& end, const Qubit v,
                                Package& dd) {
  if (v == 0U) {
    const auto zeroSuccessor = vCachedEdge::terminal(*begin);
    const auto oneSuccessor = vCachedEdge::terminal(*(begin + 1));
    return dd.makeDDNode<vNode, CachedEdge>(0, {zeroSuccessor, oneSuccessor});
  }

  const auto pivot = std::next(begin, std::distance(begin, end) / 2);
  const auto zeroSuccessor = makeStateFromVector(begin, pivot, v - 1, dd);
  const auto oneSuccessor = makeStateFromVector(pivot, end, v - 1, dd);
  return dd.makeDDNode<vNode, CachedEdge>(v, {zeroSuccessor, oneSuccessor});
}

} // namespace

VectorDD makeZeroState(const std::size_t n, Package& dd,
                       const std::size_t start) {
  const std::vector<BasisStates> state(n, BasisStates::zero);
  return makeBasisState(n, state, dd, start);
}

VectorDD makeBasisState(const std::size_t n, const std::vector<bool>& state,
                        Package& dd, const std::size_t start) {
  const auto op = [](bool b) {
    return b ? BasisStates::one : BasisStates::zero;
  };
  std::vector<BasisStates> bState(state.size());
  std::ranges::transform(state, bState.begin(), op);
  return makeBasisState(n, bState, dd, start);
}

VectorDD makeBasisState(const std::size_t n,
                        const std::vector<BasisStates>& state, Package& dd,
                        const std::size_t start) {
  suitablePackage(n + start, dd);

  if (state.size() < n) {
    throw std::invalid_argument(
        "Insufficient qubit states provided. Requested " + std::to_string(n) +
        ", but received " + std::to_string(state.size()));
  }

  vCachedEdge f = vCachedEdge::one();
  for (std::size_t p = 0; p < n; ++p) {
    std::array<vCachedEdge, RADIX> edges{};

    const auto v = static_cast<Qubit>(p + start);
    switch (state[p]) {
    case BasisStates::zero:
      edges = {f, vCachedEdge::zero()};
      break;
    case BasisStates::one:
      edges = {vCachedEdge::zero(), f};
      break;
    case BasisStates::plus:
      edges = {{{f.p, dd::SQRT2_2}, {f.p, dd::SQRT2_2}}};
      break;
    case BasisStates::minus:
      edges = {{{f.p, dd::SQRT2_2}, {f.p, -dd::SQRT2_2}}};
      break;
    case BasisStates::right:
      edges = {{{f.p, dd::SQRT2_2}, {f.p, {0, dd::SQRT2_2}}}};
      break;
    case BasisStates::left:
      edges = {{{f.p, dd::SQRT2_2}, {f.p, {0, -dd::SQRT2_2}}}};
      break;
    }
    f = dd.makeDDNode(v, edges);
  }
  const vEdge e{.p = f.p, .w = dd.cn.lookup(f.w)};
  dd.incRef(e);
  return e;
}

VectorDD makeGHZState(const std::size_t n, Package& dd) {
  suitablePackage(n, dd);

  if (n == 0U) {
    return vEdge::one();
  }

  auto leftSubtree = vEdge::one();
  auto rightSubtree = vEdge::one();

  for (std::size_t p = 0; p < n - 1; ++p) {
    leftSubtree = dd.makeDDNode(static_cast<Qubit>(p),
                                std::array{leftSubtree, vEdge::zero()});
    rightSubtree = dd.makeDDNode(static_cast<Qubit>(p),
                                 std::array{vEdge::zero(), rightSubtree});
  }

  const vEdge e = dd.makeDDNode(
      static_cast<Qubit>(n - 1),
      std::array<vEdge, RADIX>{
          {{.p = leftSubtree.p,
            .w = {.r = &constants::sqrt2over2, .i = &constants::zero}},
           {.p = rightSubtree.p,
            .w = {.r = &constants::sqrt2over2, .i = &constants::zero}}}});
  dd.incRef(e);
  return e;
}

VectorDD makeWState(const std::size_t n, Package& dd) {
  suitablePackage(n, dd);

  if (n == 0U) {
    return vEdge::one();
  }

  if ((1. / sqrt(static_cast<double>(n))) < RealNumber::eps) {
    throw std::invalid_argument(
        "Requested qubit size for generating W-state would lead to an "
        "underflow due to 1 / sqrt(n) being smaller than the currently set "
        "tolerance " +
        std::to_string(RealNumber::eps) +
        ". If you still wanna run the computation, please lower "
        "the tolerance accordingly.");
  }

  vEdge leftSubtree = vEdge::zero();
  vEdge rightSubtree = vEdge::terminal(dd.cn.lookup(1. / std::sqrt(n)));
  for (size_t p = 0; p < n; ++p) {
    leftSubtree = dd.makeDDNode(static_cast<Qubit>(p),
                                std::array{leftSubtree, rightSubtree});
    if (p != n - 1U) {
      rightSubtree = dd.makeDDNode(static_cast<Qubit>(p),
                                   std::array{rightSubtree, vEdge::zero()});
    }
  }
  dd.incRef(leftSubtree);
  return leftSubtree;
}

VectorDD makeStateFromVector(const CVec& vec, Package& dd) {
  const std::size_t sz = vec.size();

  if ((sz & (sz - 1)) != 0) {
    throw std::invalid_argument(
        "State vector must have a length of a power of two.");
  }

  if (sz == 0) {
    return vEdge::one();
  }

  if (sz == 1) {
    return vEdge::terminal(dd.cn.lookup(vec[0]));
  }

  const auto v = static_cast<Qubit>(std::log2(sz) - 1);
  suitablePackage(v, dd);

  const vCachedEdge state = makeStateFromVector(vec.begin(), vec.end(), v, dd);

  const vEdge ret{.p = state.p, .w = dd.cn.lookup(state.w)};
  dd.incRef(ret);
  return ret;
}
} // namespace dd
