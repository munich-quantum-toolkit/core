/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file StateGeneration.hpp
/// Construct common quantum states as decision diagrams.

#pragma once

#include "dd/CachedEdge.hpp"
#include "dd/ComplexNumbers.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Edge.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"

#include <bit>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace dd {
/// Construct the all-zero state \f$|0...0\rangle\f$
/// @param n The number of qubits.
/// @param dd The DD package to use for making the vector DD.
/// @param start The starting qubit index. Default is 0.
/// @throws `std::invalid_argument`, if the qubit interval exceeds package
/// capacity.
/// @return A vector DD for the all-zero state.
VectorDD makeZeroState(std::size_t n, Package& dd, std::size_t start = 0);

/// Construct a computational basis state \f$|b_{n-1}...b_0\rangle\f$
/// @param n The number of qubits.
/// @param state The state to construct.
/// @param dd The DD package to use for making the vector DD.
/// @param start The starting qubit index. Default is 0.
/// @throws std::invalid_argument If the qubit interval exceeds package capacity
/// or `size(state) < n`.
/// @return A vector DD for the computational basis state.
VectorDD makeBasisState(std::size_t n, const std::vector<bool>& state,
                        Package& dd, std::size_t start = 0);

/// Construct a product state out of
///        \f$\{0, 1, +, -, R, L\}^{\otimes n}\f$.
/// @param n The number of qubits
/// @param state The state to construct.
/// @param dd The DD package to use for making the vector DD.
/// @param start The starting qubit index. Default is 0.
/// @throws std::invalid_argument If the qubit interval exceeds package capacity
/// or `size(state) < n`.
/// @return A vector DD for the product state.
VectorDD makeBasisState(std::size_t n, const std::vector<BasisStates>& state,
                        Package& dd, std::size_t start = 0);

/// Construct a GHZ state \f$|0...0\rangle + |1...1\rangle\f$.
/// @param n The number of qubits.
/// @param dd The DD package to use for making the vector DD.
/// @throws `std::invalid_argument`, if `dd.qubits() < n`.
/// @return A vector DD for the GHZ state.
VectorDD makeGHZState(std::size_t n, Package& dd);

/// Construct a W state.
///
/// The W state is defined as
/// \f[
/// |0...01\rangle + |0...10\rangle + |10...0\rangle
/// \f]
/// @param n The number of qubits.
/// @param dd The DD package to use for making the vector DD.
/// @throws `std::invalid_argument`, if `dd.qubits() < n` or the number of
/// qubits and currently set tolerance would lead to an underflow.
/// @return A vector DD for the W state.
VectorDD makeWState(std::size_t n, Package& dd);

/// Construct a decision diagram from an arbitrary state vector.
/// @param vec The state vector to convert to a DD.
/// @param dd The DD package to use for making the vector DD.
/// @throws `std::invalid_argument`, if `vec.size()` is not a power of two or
/// `dd.qubits() < log2(vec.size())`.
/// @return A vector DD representing the state with its reference count
/// increased.
VectorDD makeStateFromVector(const CVec& vec, Package& dd);

namespace detail {
/// Read successive halves of a state vector without copying its storage.
template <class VectorEntry>
vCachedEdge buildStateFromVector(const VectorEntry& entry, const size_t level,
                                 const size_t start, Package& dd) {
  if (level == 0) {
    return dd.makeDDNode<vNode, CachedEdge>(
        0, {vCachedEdge::terminal(entry(start)),
            vCachedEdge::terminal(entry(start + 1))});
  }
  const auto half = start + (size_t{1} << level);
  return dd.makeDDNode<vNode, CachedEdge>(
      static_cast<Qubit>(level),
      {buildStateFromVector(entry, level - 1, start, dd),
       buildStateFromVector(entry, level - 1, half, dd)});
}
} // namespace detail

/// Construct a state DD from an indexed view without copying its storage.
/// @param length Number of amplitudes; zero yields the one-terminal.
/// @param entry Callable returning the complex amplitude at an index.
/// @param dd Package that owns the resulting DD.
/// @pre entry is valid for all indices smaller than length.
/// @return A state DD with its reference count increased.
/// @throws std::invalid_argument If length is not a power of two or exceeds
/// the package qubit capacity.
template <class VectorEntry>
VectorDD makeStateFromVector(const size_t length, const VectorEntry& entry,
                             Package& dd) {
  if (length == 0) {
    return vEdge::one();
  }
  if (!std::has_single_bit(length)) {
    throw std::invalid_argument(
        "State vector must have a length of a power of two.");
  }
  const auto levels = std::bit_width(length) - 1;
  if (levels > dd.qubits()) {
    throw std::invalid_argument(
        "State vector exceeds the package qubit capacity.");
  }
  const auto root =
      levels == 0 ? vCachedEdge::terminal(entry(0))
                  : detail::buildStateFromVector(entry, levels - 1, 0, dd);
  const vEdge state{.p = root.p, .w = dd.cn.lookup(root.w)};
  dd.incRef(state);
  return state;
}

}; // namespace dd
