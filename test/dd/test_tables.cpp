/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/DDDefinitions.hpp"
#include "dd/DDpackageConfig.hpp"
#include "dd/MemoryManager.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "dd/StateGeneration.hpp"
#include "dd/UniqueTable.hpp"

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using namespace dd;
TEST(DDTableTest, RehashPreservesCanonicalNodesAndOwnedRoots) {
  DDPackageConfig config;
  config.utVecNumBucket = 2;
  config.utMatNumBucket = 2;
  config.utMaxNumBucket = 16;
  Package package(2, config);
  std::vector<vEdge> states;
  std::vector<mEdge> gates;
  for (size_t i = 1; i <= 40; ++i) {
    const auto angle = static_cast<double>(i) / 100.;
    states.push_back(
        makeStateFromVector(CVec{std::cos(angle), std::sin(angle)}, package));
    gates.push_back(package.makeGateDD(
        GateMatrix{
            std::cos(angle),
            -std::sin(angle),
            std::sin(angle),
            std::cos(angle),
        },
        0));
    package.incRef(gates.back());
  }
  package.resize(3);
  for (const auto* table : {&package.vUniqueTable, &package.mUniqueTable}) {
    EXPECT_EQ(table->getStats(2).numBuckets, 2);
    EXPECT_EQ(table->getStats(0).numBuckets, 16);
    EXPECT_EQ(table->getStats(0).numEntries, 40);
    EXPECT_EQ(table->getStats(1).numBuckets, 2);
    EXPECT_EQ(table->getStats(1).numEntries, 0);
  }
  package.garbageCollect(true);
  for (size_t i = 1; i <= 40; ++i) {
    const auto angle = static_cast<double>(i) / 100.;
    const auto duplicate =
        makeStateFromVector(CVec{std::cos(angle), std::sin(angle)}, package);
    EXPECT_EQ(duplicate, states[i - 1]);
    EXPECT_EQ(package.makeGateDD(
                  GateMatrix{
                      std::cos(angle),
                      -std::sin(angle),
                      std::sin(angle),
                      std::cos(angle),
                  },
                  0),
              gates[i - 1]);
    const auto values = states[i - 1].getVector();
    ASSERT_EQ(values.size(), 2);
    EXPECT_NEAR(values[0].real(), std::cos(angle), 1e-12);
    EXPECT_NEAR(values[1].real(), std::sin(angle), 1e-12);
    package.decRef(duplicate);
    package.decRef(states[i - 1]);
    package.decRef(gates[i - 1]);
  }
  package.garbageCollect(true);
  for (const auto* table : {&package.vUniqueTable, &package.mUniqueTable}) {
    EXPECT_EQ(table->getNumEntries(), 0);
  }
  package.reset();
  for (const auto* table : {&package.vUniqueTable, &package.mUniqueTable}) {
    EXPECT_EQ(table->getStats(0).numBuckets, 16);
    EXPECT_EQ(table->getStats(0).numEntries, 0);
  }
  const auto fresh = makeZeroState(2, package);
  EXPECT_EQ(fresh.getVector(), (CVec{1., 0., 0., 0.}));
  package.decRef(fresh);
}

TEST(DDTableTest, InitialLevelsAndCapacityValidation) {
  auto manager = MemoryManager::create<vNode>();
  EXPECT_THROW((UniqueTable(manager, {.nBuckets = 0})), std::invalid_argument);
  EXPECT_THROW((UniqueTable(manager, {.nBuckets = 3})), std::invalid_argument);
  EXPECT_THROW((UniqueTable(manager, {.nBuckets = 4, .maxBuckets = 2})),
               std::invalid_argument);
  EXPECT_THROW((UniqueTable(manager, {.nBuckets = 4, .maxBuckets = 6})),
               std::invalid_argument);
  UniqueTable table(manager, {.nVars = 1, .nBuckets = 1});
  auto* node = manager.get<vNode>();
  node->v = 0;
  node->e = {vEdge::one(), vEdge::zero()};
  EXPECT_EQ(table.lookup(node), node);
  EXPECT_EQ(table.getStats(0).numBuckets, 1);
  EXPECT_EQ(table.lookup(node), node);
  table.clear();
  EXPECT_EQ(table.getNumEntries(), 0);
}
TEST(DDTableTest, FixedHashDoesNotRequireAllocatedLevels) {
  auto manager = MemoryManager::create<vNode>();
  UniqueTable table(manager, {.nBuckets = 4});
  vNode node{};
  node.v = 0;
  node.e = {vEdge::one(), vEdge::zero()};
  const auto key = table.hash(node);
  EXPECT_LT(key, 4);
  table.resize(1);
  EXPECT_EQ(table.hash(node), key);
}
