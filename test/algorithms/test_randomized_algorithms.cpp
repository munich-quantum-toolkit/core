/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "algorithms/BernsteinVazirani.hpp"
#include "algorithms/Grover.hpp"
#include "algorithms/QPE.hpp"
#include "algorithms/RandomCliffordCircuit.hpp"
#include "ir/QuantumComputation.hpp"

#include <gtest/gtest.h>

#include <cstddef>

TEST(RandomizedAlgorithms, ReproducibleWithExplicitSeed) {
  constexpr std::size_t seed = 17;

  EXPECT_EQ(qc::createBernsteinVazirani(64, seed),
            qc::createBernsteinVazirani(64, seed));
  EXPECT_EQ(qc::createIterativeBernsteinVazirani(64, seed),
            qc::createIterativeBernsteinVazirani(64, seed));
  EXPECT_EQ(qc::createGrover(5, seed), qc::createGrover(5, seed));
  EXPECT_EQ(qc::createQPE(8, true, seed), qc::createQPE(8, true, seed));
  EXPECT_EQ(qc::createIterativeQPE(8, false, seed),
            qc::createIterativeQPE(8, false, seed));
  EXPECT_EQ(qc::createRandomCliffordCircuit(4, 8, seed),
            qc::createRandomCliffordCircuit(4, 8, seed));
}

TEST(RandomizedAlgorithms, SeedsAreIndependentBetweenCalls) {
  constexpr std::size_t seed = 23;
  const qc::QuantumComputation expected = qc::createBernsteinVazirani(64, seed);

  static_cast<void>(qc::createGrover(5, seed + 1));
  static_cast<void>(qc::createQPE(8, true, seed + 2));
  static_cast<void>(qc::createRandomCliffordCircuit(4, 8, seed + 3));

  EXPECT_EQ(qc::createBernsteinVazirani(64, seed), expected);

  auto foundDistinctCircuit = false;
  for (auto candidateSeed = seed + 1; candidateSeed <= seed + 3;
       ++candidateSeed) {
    if (qc::createBernsteinVazirani(64, candidateSeed) != expected) {
      foundDistinctCircuit = true;
      break;
    }
  }
  EXPECT_TRUE(foundDistinctCircuit);
}
