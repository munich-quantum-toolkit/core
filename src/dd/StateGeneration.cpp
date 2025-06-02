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
vCachedEdge randomNode(Qubit v, Generator& gen, AngleDistribution& dist,
                       Package& dd) {
  const auto alpha = randomComplexOnUnitCircle(gen, dist);
  const auto beta = randomComplexOnUnitCircle(gen, dist);
  const std::array<vCachedEdge, RADIX> edges{
      vCachedEdge::terminal(alpha / SQRT2_2),
      vCachedEdge::terminal(beta / SQRT2_2),
  };
  const vCachedEdge ret = dd.makeDDNode(v, edges);

  dd.cn.incRef(ret.p->e[0].w);
  dd.cn.incRef(ret.p->e[1].w);

  return ret;
}
} // namespace

VectorDD generateExponentialState(const std::size_t levels, Package& dd) {
  std::random_device rd;
  return generateExponentialState(levels, dd, rd());
}

VectorDD generateExponentialState(const std::size_t levels, Package& dd,
                                  const std::size_t seed) {
  std::vector<std::size_t> nodesPerLevel(levels - 1); // [2, 4, 8, ...]
  std::generate(nodesPerLevel.begin(), nodesPerLevel.end(),
                [exp = 1]() mutable { return 1ULL << exp++; });
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
  if (nodesPerLevel.size() != levels - 1) {
    throw std::invalid_argument(
        "Number of levels - 1 must match nodesPerLevel size");
  }

  [[maybe_unused]] bool inc{};

  Generator gen(seed);
  AngleDistribution dist{0, 2. * qc::PI};

  std::size_t v = levels - 1;
  const vCachedEdge root = randomNode(static_cast<Qubit>(v), gen, dist, dd);

  std::vector<vCachedEdge> prev{root};
  for (const std::size_t n : nodesPerLevel) {
    if (n > 2UL * prev.size()) {
      throw std::invalid_argument(
          "Number of nodes per level must not exceed twice the number of "
          "nodes in the level above");
    }

    --v;

    std::vector<vCachedEdge> curr(n); // Random nodes on layer v.
    std::generate(curr.begin(), curr.end(), [&] {
      return randomNode(static_cast<Qubit>(v), gen, dist, dd);
    });

    std::vector<std::size_t> indices(2 * prev.size()); // Indices for wireing.
    switch (strategy) {
    case ROUNDROBIN: {
      std::generate(indices.begin(), indices.end(),
                    [&n, r = 0UL]() mutable { return (r++) % n; });
      break;
    }
    case RANDOM: {
      IndexDistribution idxDist{0, n - 1};

      // Make sure that each successor is connected to the previous layer.
      auto lastOfN = indices.begin();
      std::advance(lastOfN, n);
      std::iota(indices.begin(), lastOfN, 0);

      // Choose the rest randomly.
      std::generate(lastOfN, indices.end(),
                    [&idxDist, &gen]() { return idxDist(gen); });

      // Shuffle to randomly interleave the resulting indices.
      std::shuffle(indices.begin(), indices.end(), gen);
      break;
    }
    }

    // Wire previous layer with the current.
    for (std::size_t i = 0; i < prev.size(); ++i) {
      const vCachedEdge& e = prev[i];

      vEdge& left = e.p->e[0];
      vEdge& right = e.p->e[1];

      // Note that the following assignments ignore the edge weights of the
      // current layer. Since these aren't in the lookup table, we can
      // safely do so.

      left.p = curr[indices[2 * i]].p;
      right.p = curr[indices[(2 * i) + 1]].p;

      inc = dd.getUniqueTable<vNode>().incRef(left.p) &&
            dd.getUniqueTable<vNode>().incRef(right.p);
    }

    prev = std::move(curr);
  }

  vEdge ret{root.p, Complex::one()};

  dd.cn.incRef(ret.w);
  inc = dd.getUniqueTable<vNode>().incRef(ret.p);

  return ret;
}
} // namespace dd
