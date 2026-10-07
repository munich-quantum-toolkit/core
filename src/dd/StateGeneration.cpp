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

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace dd {
namespace {
/// Validate that @p n qubits starting at @p start fit in the package.
/// @throws std::invalid_argument If the qubit interval exceeds the capacity.
void suitablePackage(const size_t n, const Package& dd,
                     const size_t start = 0) {
  const std::size_t nqubits = dd.qubits();
  if (start > nqubits || n > nqubits - start) {
    throw std::invalid_argument{
        "Requested state with " + std::to_string(n) + " qubits starting at " +
        std::to_string(start) +
        ", but current package configuration only supports up to " +
        std::to_string(nqubits) +
        " qubits. Please allocate a larger package instance."};
  }
}

template <class BasisEntry>
VectorDD buildBasisState(const size_t n, const size_t available,
                         const BasisEntry& entry, Package& dd,
                         const size_t start) {
  suitablePackage(n, dd, start);
  if (available < n) {
    throw std::invalid_argument(
        "Insufficient qubit states provided. Requested " + std::to_string(n) +
        ", but received " + std::to_string(available));
  }

  vCachedEdge f = vCachedEdge::one();
  for (std::size_t p = 0; p < n; ++p) {
    std::array<vCachedEdge, RADIX> edges{};

    const auto v = static_cast<Qubit>(p + start);
    switch (entry(p)) {
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

} // namespace

VectorDD makeZeroState(const size_t n, Package& dd, const size_t start) {
  return buildBasisState(
      n, n, [](size_t) { return BasisStates::zero; }, dd, start);
}

VectorDD makeBasisState(const size_t n, const std::vector<bool>& state,
                        Package& dd, const size_t start) {
  return buildBasisState(
      n, state.size(),
      [&state](const size_t i) {
        return state[i] ? BasisStates::one : BasisStates::zero;
      },
      dd, start);
}

VectorDD makeBasisState(const size_t n, const std::vector<BasisStates>& state,
                        Package& dd, const size_t start) {
  return buildBasisState(
      n, state.size(), [&state](const size_t i) { return state[i]; }, dd,
      start);
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
          {
              {
                  .p = leftSubtree.p,
                  .w = {.r = &constants::sqrt2over2, .i = &constants::zero},
              },
              {
                  .p = rightSubtree.p,
                  .w = {.r = &constants::sqrt2over2, .i = &constants::zero},
              },
          },
      });
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
  return makeStateFromVector(
      vec.size(), [&vec](const size_t index) { return vec[index]; }, dd);
}
} // namespace dd
