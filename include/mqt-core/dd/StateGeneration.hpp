/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "dd/Node.hpp"
#include "dd/Package.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dd {
/**
 * @brief Construct the all-zero state \f$|0...0\rangle\f$
 * @param n The number of qubits.
 * @param dd The DD package to use for making the vector DD.
 * @param start The starting qubit index. Default is 0.
 * @throws `std::invalid_argument`, if `dd.qubits() < n`.
 * @return A vector DD for the all-zero state.
 */
VectorDD makeZeroState(std::size_t n, Package& dd, std::size_t start = 0);

/**
 * @brief Construct a computational basis state \f$|b_{n-1}...b_0\rangle\f$
 * @param n The number of qubits.
 * @param state The state to construct.
 * @param dd The DD package to use for making the vector DD.
 * @param start The starting qubit index. Default is 0.
 * @throws `std::invalid_argument`, if `dd.qubits() < n` or `size(state) < n`.
 * @return A vector DD for the computational basis state.
 */
VectorDD makeBasisState(std::size_t n, const std::vector<bool>& state,
                        Package& dd, std::size_t start = 0);

/**
 * @brief Construct a product state out of
 *        \f$\{0, 1, +, -, R, L\}^{\otimes n}\f$.
 * @param n The number of qubits
 * @param state The state to construct.
 * @param dd The DD package to use for making the vector DD.
 * @param start The starting qubit index. Default is 0.
 * @throws `std::invalid_argument`, if `dd.qubits() < n` or `size(state) < n`.
 * @return A vector DD for the product state.
 */
VectorDD makeBasisState(std::size_t n, const std::vector<BasisStates>& state,
                        Package& dd, std::size_t start = 0);

/**
 * @brief Construct a GHZ state \f$|0...0\rangle + |1...1\rangle\f$.
 * @param n The number of qubits.
 * @param dd The DD package to use for making the vector DD.
 * @throws `std::invalid_argument`, if `dd.qubits() < n`.
 * @return A vector DD for the GHZ state.
 */
VectorDD makeGHZState(std::size_t n, Package& dd);

/**
 * @brief Construct a W state.
 * @details The W state is defined as
 * \f[
 * |0...01\rangle + |0...10\rangle + |10...0\rangle
 * \f]
 * @param n The number of qubits.
 * @param dd The DD package to use for making the vector DD.
 * @throws `std::invalid_argument`, if `dd.qubits() < n` or the number of qubits
 * and currently set tolerance would lead to an underflow.
 * @return A vector DD for the W state.
 */
VectorDD makeWState(std::size_t n, Package& dd);

/**
 * @brief Construct a decision diagram from an arbitrary state vector.
 * @param vec The state vector to convert to a DD.
 * @param dd The DD package to use for making the vector DD.
 * @throws `std::invalid_argument`, if `vec.size()` is not a power of two or
 * `dd.qubits() < log2(vec.size()) - 1`.
 * @return A vector DD representing the state.
 */
VectorDD makeStateFromVector(const CVec& vec, Package& dd);

/// @brief The strategy to wire two layers.
enum GenerationWireStrategy : std::uint8_t {
  ROUNDROBIN, // Choose nodes in the next layer in a round-robin fashion.
  RANDOM      // Randomly choose nodes in the next layer.
};

/**
 * @brief Generate exponentially large vector DD.
 * @param levels The number of levels in the vector DD.
 * @param dd The DD package to use for generating the vector DD.
 * @throws `std::invalid_argument`, if `dd.qubits() < levels`.
 * @return The exponentially large vector DD.
 */
VectorDD generateExponentialState(std::size_t levels, Package& dd);

/**
 * @brief Generate exponentially large vector DD. Use @p seed for randomization.
 * @param levels The number of levels in the vector DD.
 * @param dd The DD package to use for generating the vector DD.
 * @param seed The seed used for randomization.
 * @throws `std::invalid_argument`, if `dd.qubits() < levels`.
 * @return The exponentially large vector DD.
 */
VectorDD generateExponentialState(std::size_t levels, Package& dd,
                                  std::size_t seed);

/**
 * @brief Generate random vector DD.
 * @param levels The number of levels in the vector DD.
 * @param nodesPerLevel The number of nodes per level.
 * @param strategy The strategy to wire two layers.
 * @param dd The DD package to use for generating the vector DD.
 * @throws `std::invalid_argument`, if `dd.qubits() < levels`.
 * @return The random vector DD.
 */
VectorDD generateRandomState(std::size_t levels,
                             const std::vector<std::size_t>& nodesPerLevel,
                             GenerationWireStrategy strategy, Package& dd);

/**
 * @brief Generate random vector DD. Use @p seed for randomization.
 * @param levels The number of levels in the vector DD.
 * @param nodesPerLevel The number of nodes per level.
 * @param strategy The strategy to wire two layers.
 * @param dd The DD package to use for generating the vector DD.
 * @param seed The seed used for randomization.
 * @throws `std::invalid_argument`, if `dd.qubits() < levels`, `levels <= 0`, or
 * `nodesPerLevel.size() != levels`.
 * @return The random vector DD.
 */
VectorDD generateRandomState(std::size_t levels,
                             const std::vector<std::size_t>& nodesPerLevel,
                             GenerationWireStrategy strategy, Package& dd,
                             std::size_t seed);
}; // namespace dd
