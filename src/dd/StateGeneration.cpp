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
 * @brief Generate random node with terminal edges initialized with random
 * weights. The function ensures that the norm of the edge weights is 1.
 */
vEdge randomNode(Qubit v, Generator& gen, AngleDistribution& dist,
                 Package& dd) {
  const auto alpha = randomComplexOnUnitCircle(gen, dist);
  const auto beta = randomComplexOnUnitCircle(gen, dist);
  const std::array<vEdge, RADIX> edges{
      vEdge::terminal(dd.cn.lookup(alpha / SQRT2_2)),
      vEdge::terminal(dd.cn.lookup(beta / SQRT2_2)),
  };

  return dd.makeDDNode(v, edges);
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

  Generator gen(seed);
  AngleDistribution dist{0, 2. * qc::PI};

  auto v = static_cast<Qubit>(levels - 1);
  const vEdge root = randomNode(v, gen, dist, dd);
  std::vector<vEdge> prev{root};
  for (const std::size_t n : nodesPerLevel) {
    if (n > 2UL * prev.size()) {
      throw std::invalid_argument(
          "Number of nodes per level must not exceed twice the number of "
          "nodes in the level above");
    }

    --v;

    std::vector<vEdge> curr(n); // Random nodes on layer v.
    std::generate(curr.begin(), curr.end(),
                  [&] { return randomNode(v, gen, dist, dd); });

    std::vector<std::size_t> indices(2 * prev.size()); // Indices for wireing.
    switch (strategy) {
    case ROUNDROBIN: {
      std::generate(indices.begin(), indices.end(),
                    [&n, r = 0UL]() mutable { return (r++) % n; });
      break;
    }
    case RANDOM: {
      // First make sure that each successor node is connected to the previous
      // layer.
      auto lastOfN = indices.begin();
      std::advance(lastOfN, n);
      std::iota(indices.begin(), lastOfN, 0);
      std::shuffle(indices.begin(), lastOfN, gen);

      // Choose the rest randomly.
      IndexDistribution idxDist{0, n};
      std::generate(lastOfN, indices.end(),
                    [&idxDist, &gen]() { return idxDist(gen); });

      // Shuffle one last time to interleave the resulting indices.
      std::shuffle(indices.begin(), indices.end(), gen);

      for (auto i : indices) {
        std::cout << i << ' ';
      }
      std::cout << '\n';
      break;
    }
    }

    // Wire previous layer with the current.
    for (std::size_t i = 0; i < prev.size(); ++i) {
      const vEdge& e = prev[i];
      e.p->e[0].p = curr[indices[2 * i]].p;
      e.p->e[1].p = curr[indices[(2 * i) + 1]].p;
    }

    prev = std::move(curr);
  }

  vEdge e{root.p, Complex::one()};
  dd.incRef(e);
  return e;
}
} // namespace dd
