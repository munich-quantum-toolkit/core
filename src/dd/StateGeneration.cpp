/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/StateGeneration.hpp"

#include "dd/Complex.hpp"
#include "dd/ComplexNumbers.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Edge.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "ir/Definitions.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dd {
namespace {
using Generator = std::mt19937_64;
using AngleDistribution = std::uniform_real_distribution<double>;
using IndexDistribution = std::uniform_int_distribution<std::size_t>;

/**
 * @brief Return random complex number on the unit circle, such that,
 * its squared magnitude is one.
 */
ComplexValue randomComplexOnUnitCircle(Generator& gen,
                                       AngleDistribution& dist) {
  const double angle = dist(gen);
  return {std::cos(angle), std::sin(angle)};
}

/**
 * @brief Generate node with edges pointing at @p left and @p right.
 * Initialize edge weights randomly with the constraint that their norm is one.
 * @note The CachedEdge ensures that the weight of the resulting edge is not
 * stored in the lookup table.
 */
vCachedEdge randomNode(Qubit v, vNode* left, vNode* right, Generator& gen,
                       AngleDistribution& dist, Package& dd) {
  const auto alpha = randomComplexOnUnitCircle(gen, dist) * SQRT2_2;
  const auto beta = randomComplexOnUnitCircle(gen, dist) * SQRT2_2;

  const std::array<vCachedEdge, RADIX> edges{vCachedEdge(left, alpha),
                                             vCachedEdge(right, beta)};

  return dd.makeDDNode(v, edges);
}
} // namespace

VectorDD generateExponentialState(const std::size_t levels, Package& dd) {
  std::random_device rd;
  return generateExponentialState(levels, dd, rd());
}

VectorDD generateExponentialState(const std::size_t levels, Package& dd,
                                  const std::size_t seed) {
  std::vector<std::size_t> nodesPerLevel(levels); // [1, 2, 4, 8, ...]
  std::generate(nodesPerLevel.begin(), nodesPerLevel.end(),
                [exp = 0]() mutable { return 1ULL << exp++; });
  return generateRandomState(levels, nodesPerLevel, ROUNDROBIN, dd, seed);
}

VectorDD generateRandomState(const std::size_t levels,
                             const std::vector<std::size_t>& nodesPerLevel,
                             const GenerationWireStrategy strategy,
                             Package& dd) {
  std::random_device rd;
  return generateRandomState(levels, nodesPerLevel, strategy, dd, rd());
}

VectorDD generateRandomState(const std::size_t levels,
                             const std::vector<std::size_t>& nodesPerLevel,
                             const GenerationWireStrategy strategy, Package& dd,
                             const std::size_t seed) {
  if (levels <= 0U) {
    throw std::invalid_argument("Number of levels must be greater than zero");
  }
  if (nodesPerLevel.size() != levels) {
    throw std::invalid_argument(
        "Number of levels must match nodesPerLevel size");
  }

  Generator gen(seed);
  AngleDistribution dist{0, 2. * qc::PI};

  // Generate terminal nodes.
  constexpr vNode* terminal = vNode::getTerminal();
  std::vector<vCachedEdge> below(nodesPerLevel.back());
  std::generate(below.begin(), below.end(), [&] {
    return randomNode(0, terminal, terminal, gen, dist, dd);
  });

  Qubit v{1};
  auto it = nodesPerLevel.rbegin();
  std::advance(it, 1); // Dealt with terminals above.
  for (; it != nodesPerLevel.rend(); ++it, ++v) {
    const std::size_t n = *it;
    const std::size_t m = below.size();

    if (2UL * n < m) {
      throw std::invalid_argument(
          "Number of nodes per level must not exceed twice the number of "
          "nodes in the level above");
    }

    std::vector<std::size_t> indices(2 * n); // Indices for wireing.
    switch (strategy) {
    case ROUNDROBIN: {
      std::generate(indices.begin(), indices.end(),
                    [&m, r = 0UL]() mutable { return (r++) % m; });
      break;
    }
    case RANDOM: {
      IndexDistribution idxDist{0, m - 1};

      // Ensure that all the nodes below have a connection upwards.
      auto pivot = indices.begin();
      std::advance(pivot, m);
      std::iota(indices.begin(), pivot, 0);

      // Choose the rest randomly.
      std::generate(pivot, indices.end(),
                    [&idxDist, &gen]() { return idxDist(gen); });

      // Shuffle to randomly interleave the resulting indices.
      std::shuffle(indices.begin(), indices.end(), gen);
    }
    }

    std::vector<vCachedEdge> curr(n); // Random nodes on layer v.
    for (std::size_t i = 0; i < n; ++i) {
      vNode* left = below[indices[2 * i]].p;
      vNode* right = below[indices[(2 * i) + 1]].p;
      curr[i] = randomNode(v, left, right, gen, dist, dd);
    }

    below = std::move(curr);
  }

  // Below only contains one element: the root.
  vEdge ret{below.at(0).p, Complex::one()};
  dd.incRef(ret);
  return ret;
}
} // namespace dd
