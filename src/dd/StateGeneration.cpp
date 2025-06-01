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
#include <limits>
#include <random>
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
 * @brief Generate random node with terminal edges initialized with random
 * weights. The function ensures that the norm of the edge weights is 1.
 */
vEdge randomNode(Qubit v, Generator& gen, AngleDistribution& dist,
                 Package& dd) {
  constexpr double eps = std::numeric_limits<double>::epsilon();

  const auto alpha = randomComplexOnUnitCircle(gen, dist);
  const auto beta = randomComplexOnUnitCircle(gen, dist);
  const auto norm = std::sqrt(2);

  const std::array<vEdge, RADIX> edges{
      vEdge::terminal(dd.cn.lookup(alpha / norm)),
      vEdge::terminal(dd.cn.lookup(beta / norm)),
  };

  // Check: Properly normalized.
  assert(std::sqrt(ComplexNumbers::mag2(edges[0].w) +
                   ComplexNumbers::mag2(edges[1].w)) < 1 + eps);

  return dd.makeDDNode(v, edges);
}
} // namespace

VectorDD generateExponentialState(const std::size_t levels, Package& dd) {
  std::random_device rd;
  return generateExponentialState(levels, rd(), dd);
}

VectorDD generateExponentialState(const std::size_t levels,
                                  const std::size_t seed, Package& dd) {
  std::vector<std::size_t> nodesPerLevel(levels - 1); // [2, 4, 8, ...]
  std::generate(nodesPerLevel.begin(), nodesPerLevel.end(),
                [exp = 1]() mutable { return std::pow(2, exp++); });
  return generateRandomState(levels, nodesPerLevel, ROUNDROBIN, seed, dd);
}

VectorDD generateRandomState(const std::size_t levels,
                             const std::vector<std::size_t>& nodesPerLevel,
                             const GenerationWireStrategy strategy,
                             Package& dd) {
  std::random_device rd;
  return generateRandomState(levels, nodesPerLevel, strategy, rd(), dd);
}

VectorDD generateRandomState(const std::size_t levels,
                             const std::vector<std::size_t>& nodesPerLevel,
                             const GenerationWireStrategy strategy,
                             const std::size_t seed, Package& dd) {
  assert(levels > 0U && "Number of levels must be greater than zero");
  assert(nodesPerLevel.size() == levels - 1 &&
         "Number of levels - 1 must match nodesPerLevel size");

  Generator gen(seed);
  AngleDistribution dist{0, 2. * qc::PI};

  auto v = static_cast<Qubit>(levels - 1);
  const vEdge root = randomNode(v, gen, dist, dd);

  std::vector<vEdge> prev{root};
  for (const std::size_t n : nodesPerLevel) {
    --v;

    std::vector<vEdge> curr(n); // Random nodes on layer v.
    std::generate(curr.begin(), curr.end(),
                  [&] { return randomNode(v, gen, dist, dd); });

    std::vector<std::size_t> indices(2 * prev.size()); // Indices for wireing.
    switch (strategy) {
    case ROUNDROBIN: {
      std::generate(indices.begin(), indices.end(),
                    [&n, r = std::size_t{0}]() mutable { return (r++) % n; });
      break;
    }
    case RANDOM: {
      IndexDistribution indexDist{0, curr.size() - 1};
      std::generate(indices.begin(), indices.end(),
                    [&indexDist, &gen]() { return indexDist(gen); });
      break;
    }
    }

    // Wire previous layer with the current.
    for (std::size_t i = 0; i < prev.size(); ++i) {
      vEdge& e = prev[i];
      e.p->e[0].p = curr[indices[2 * i]].p;
      e.p->e[1].p = curr[indices[(2 * i) + 1]].p;
    }

    prev = std::move(curr);
  }

  vEdge ret{root.p, Complex::one()};
  dd.incRef(ret);
  return ret;
}
} // namespace dd
