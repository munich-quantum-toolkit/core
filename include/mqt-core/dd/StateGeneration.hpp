/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/Node.hpp"
#include "dd/Package.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dd {
/// @brief The strategy to use to wire two layers.
enum GenerationLinkStrategy : std::uint8_t {
  ROUNDROBIN, // Choose nodes in the next layer in a round-robin fashion.
  RANDOM      // Randomly choose nodes in the next layer.
};

/**
 * @brief Generate exponentially large vector DD.
 * @param levels The number of levels in the vector DD.
 * @param dd The DD package to use for generating the vector DD.
 * @return The exponentially large vector DD.
 */
VectorDD generateExponentialState(std::size_t levels, Package& dd);

/**
 * @brief Generate exponentially large vector DD. Use @p seed for randomization.
 * @param levels The number of levels in the vector DD.
 * @param seed The seed used for randomization.
 * @param dd The DD package to use for generating the vector DD.
 * @return The exponentially large vector DD.
 */
VectorDD generateExponentialState(std::size_t levels, std::size_t seed,
                                  Package& dd);

/**
 * @brief Generate random large vector DD.
 * @param levels The number of levels in the vector DD.
 * @param nodesPerLevel The number of nodes per level. Implicitly, contains `1`
 * (the root node) as first element.
 * @param strategy The strategy to use to wire two layers.
 * @param dd The DD package to use for generating the vector DD.
 * @return The exponentially large VectorDD.
 */
VectorDD generateRandomState(std::size_t levels,
                             const std::vector<std::size_t>& nodesPerLevel,
                             GenerationLinkStrategy strategy, Package& dd);

/**
 * @brief Generate random vector DD. Use @p seed for randomization.
 * @param levels The number of levels in the vector DD.
 * @param nodesPerLevel The number of nodes per level. Implicitly, contains `1`
 * (the root node) as first element.
 * @param strategy The strategy to use to wire two layers.
 * @param seed The seed used for randomization.
 * @param dd The DD package to use for generating the vector DD.
 * @return The exponentially large VectorDD.
 */
VectorDD generateRandomState(std::size_t levels,
                             const std::vector<std::size_t>& nodesPerLevel,
                             GenerationLinkStrategy strategy, std::size_t seed,
                             Package& dd);
}; // namespace dd
