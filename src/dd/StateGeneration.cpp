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
 *        its squared magnitude is one.
 */
std::complex<double> randomComplexOnUnitCircle(Generator& gen,
                                               AngleDistribution& dist) {
  const double angle = dist(gen);
  return {std::cos(angle), std::sin(angle)};
}

/**
 * @brief Generate node with terminal edges initialized with random weights.
 *        The norm of the node's edge weights is one.
 * @note  Increase the ref counts of the node's outgoing edge weights.
 *        Due to the use of cached edges, the weight of the returned edge is not
 *        stored in the lookup table.
 */
vCachedEdge randomNode(Qubit v, vNode* left, vNode* right, Generator& gen,
                       AngleDistribution& dist, Package& dd) {
  const auto alpha = randomComplexOnUnitCircle(gen, dist) * SQRT2_2;
  const auto beta = randomComplexOnUnitCircle(gen, dist) * SQRT2_2;

  const std::array<vCachedEdge, RADIX> edges{
      vCachedEdge(left, ComplexValue(alpha)),
      vCachedEdge(right, ComplexValue(beta))};

  const vCachedEdge ret = dd.makeDDNode(v, edges);

  // const auto leftSuccessor = ret.p->e[0];
  // const auto rightSuccessor = ret.p->e[1];

  // leftSuccessor.p->ref++;
  // rightSuccessor.p->ref++;

  // dd.cn.incRef(leftSuccessor.w);
  // dd.cn.incRef(rightSuccessor.w);

  return ret;
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

    if (2UL * n < below.size()) {
      throw std::invalid_argument(
          "Number of nodes per level must not exceed twice the number of "
          "nodes in the level above");
    }

    const std::size_t m = below.size();
    std::vector<std::size_t> indices(2 * n); // Indices for wireing.
    switch (strategy) {
    case ROUNDROBIN: {
      std::generate(indices.begin(), indices.end(),
                    [&m, r = 0UL]() mutable { return (r++) % m; });
      break;
    }
    case RANDOM: {
      IndexDistribution idxDist{0, below.size() - 1};

      // Make sure that each node below is connected.
      auto pivot = indices.begin();
      std::advance(pivot, below.size());
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

  vEdge ret{below[0].p, Complex::one()};
  dd.incRef(ret);
  return ret;
}
} // namespace dd
