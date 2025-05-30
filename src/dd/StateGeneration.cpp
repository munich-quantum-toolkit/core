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

#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

namespace dd {
namespace {
using AngleGenerator = std::mt19937_64;
using AngleDistribution = std::uniform_real_distribution<double>;
using IndexDistribution = std::uniform_int_distribution<std::size_t>;

/**
 * @brief Return random double-precision complex number `alpha` on the unit
 * circle. This ensures that `abs(alpha)^2 = 1`.
 */
std::complex<double> randomComplexOnUnitCircle(AngleGenerator& generator,
                                               AngleDistribution& dist) {
  const double angle = dist(generator);
  return {std::cos(angle), std::sin(angle)};
}

/**
 * @brief Generate random node with terminal edges initialized with random
 * weights. The function ensures that the norm of the edge weights is 1.
 */
vEdge randomNode(Qubit v, AngleGenerator& generator, AngleDistribution& dist,
                 Package& dd) {
  const auto alpha = randomComplexOnUnitCircle(generator, dist);
  const auto beta = randomComplexOnUnitCircle(generator, dist);
  const auto norm = std::sqrt(2);

  const std::array<vEdge, RADIX> edges{
      vEdge::terminal(dd.cn.lookup(alpha / norm)),
      vEdge::terminal(dd.cn.lookup(beta / norm)),
  };

  // Check: Properly normalized.
  assert(std::sqrt(ComplexNumbers::mag2(edges[0].w) +
                   ComplexNumbers::mag2(edges[1].w)) < 1 + 1e-6);

  return dd.makeDDNode(v, edges);
}
} // namespace

VectorDD generateExponentialState(const std::size_t levels, Package& dd) {
  std::random_device rd;
  return generateExponentialState(levels, rd(), dd);
}

VectorDD generateExponentialState(const std::size_t levels,
                                  const std::size_t seed, Package& dd) {
  std::vector<std::size_t> nodesPerLevel(levels - 1);
  for (std::size_t i = 1; i < levels; ++i) { // [2, 4, 8, ...]
    nodesPerLevel[i - 1] = static_cast<std::size_t>(std::pow(2, i));
  }
  return generateRandomState(levels, nodesPerLevel, ROUNDROBIN, seed, dd);
}

VectorDD generateRandomState(const std::size_t levels,
                             const std::vector<std::size_t>& nodesPerLevel,
                             const GenerationLinkStrategy strategy,
                             Package& dd) {
  std::random_device rd;
  return generateRandomState(levels, nodesPerLevel, strategy, rd(), dd);
}

VectorDD generateRandomState(const std::size_t levels,
                             const std::vector<std::size_t>& nodesPerLevel,
                             const GenerationLinkStrategy strategy,
                             const std::size_t seed, Package& dd) {
  assert(levels > 0U && "Number of levels must be greater than zero");
  assert(nodesPerLevel.size() == levels - 1 &&
         "Number of levels - 1 must match nodesPerLevel size");

  AngleGenerator generator(seed);
  AngleDistribution dist{0, 2. * qc::PI};

  auto v = static_cast<Qubit>(levels - 1);
  const vEdge root = randomNode(v, generator, dist, dd);

  std::vector<vEdge> prev{root};
  for (const std::size_t n : nodesPerLevel) {
    --v;

    // Generate nodes of layer.
    std::vector<vEdge> curr(n);
    for (std::size_t j = 0; j < n; ++j) {
      curr[j] = randomNode(v, generator, dist, dd);
    }

    // Connect to previous layer based on the given strategy.
    switch (strategy) {
    case ROUNDROBIN: {
      std::size_t r = 0;
      for (auto& ePrev : prev) {
        ePrev.p->e[0].p = curr[r % n].p;
        ePrev.p->e[1].p = curr[(r + 1) % n].p;
        r += 2;
      }
      break;
    }
    case RANDOM: {
      IndexDistribution indexDist{0, curr.size() - 1};
      for (auto& ePrev : prev) {
        ePrev.p->e[0].p = curr[indexDist(generator)].p;
        ePrev.p->e[1].p = curr[indexDist(generator)].p;
      }
      break;
    }
    }

    prev = std::move(curr);
  }

  vEdge ret{root.p, Complex::one()};
  dd.incRef(ret);
  return ret;
}
} // namespace dd
