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
#include "dd/RealNumber.hpp"
#include "dd/StateGeneration.hpp"
#include "dd/UnaryComputeTable.hpp"
#include "dd/UniqueTable.hpp"

#include "support/Diagnostics.hpp"
#include "support/TestSupport.hpp"

#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

using namespace dd;
TEST(DDTableTest, RehashPreservesCanonicalNodesAndOwnedRoots) {
  DDPackageConfig config;
  config.utVecNumBucket = 2;
  config.utMatNumBucket = 2;
  config.utMaxNumBucket = 16;
  auto packageOwner = ::mqt::test::value(Package::create(2, config));
  auto& package = *packageOwner;
  std::vector<vEdge> states;
  std::vector<mEdge> gates;
  for (size_t i = 1; i <= 40; ++i) {
    const auto angle = static_cast<double>(i) / 100.;
    states.push_back(::mqt::test::value(
        makeStateFromVector(CVec{std::cos(angle), std::sin(angle)}, package)));
    gates.push_back(::mqt::test::value(package.makeGateDD(
        GateMatrix{
            std::cos(angle),
            -std::sin(angle),
            std::sin(angle),
            std::cos(angle),
        },
        0)));
    package.incRef(gates.back());
  }
  ::mqt::test::value(package.resize(3));
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
    const auto duplicate = ::mqt::test::value(
        makeStateFromVector(CVec{std::cos(angle), std::sin(angle)}, package));
    EXPECT_EQ(duplicate, states[i - 1]);
    EXPECT_EQ(::mqt::test::value(package.makeGateDD(
                  GateMatrix{
                      std::cos(angle),
                      -std::sin(angle),
                      std::sin(angle),
                      std::cos(angle),
                  },
                  0)),
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
  const auto fresh = ::mqt::test::value(makeZeroState(2, package));
  EXPECT_EQ(fresh.getVector(), (CVec{1., 0., 0., 0.}));
  package.decRef(fresh);
}

TEST(DDTableTest, InitialLevelsAndCapacityValidation) {
  auto manager = MemoryManager::create<vNode>();
  for (const auto& [initial, maximum] : {
           std::pair{0U, 4U},
           {3U, 4U},
           {4U, 0U},
           {4U, 2U},
           {4U, 6U},
       }) {
    EXPECT_EQ(::mqt::test::errorKind([&] {
                return UniqueTable::create(
                    manager, {.nBuckets = initial, .maxBuckets = maximum});
              }),
              ::mqt::ErrorCategory::InvalidArgument);
    DDPackageConfig config;
    config.utVecNumBucket = initial;
    config.utMatNumBucket = 1;
    config.utMaxNumBucket = maximum;
    EXPECT_EQ(
        ::mqt::test::errorKind([&] { return Package::create(1, config); }),
        ::mqt::ErrorCategory::InvalidArgument);
    std::swap(config.utVecNumBucket, config.utMatNumBucket);
    EXPECT_EQ(
        ::mqt::test::errorKind([&] { return Package::create(1, config); }),
        ::mqt::ErrorCategory::InvalidArgument);
  }
  auto table = ::mqt::test::value(
      UniqueTable::create(manager, {.nVars = 1, .nBuckets = 1}));
  auto* node = manager.get<vNode>();
  node->v = 0;
  node->e = {vEdge::one(), vEdge::zero()};
  EXPECT_EQ(table.lookup(node), node);
  EXPECT_EQ(table.getStats(0).numBuckets, 1);
  EXPECT_EQ(table.lookup(node), node);
  table.clear();
  EXPECT_EQ(table.getNumEntries(), 0);
}
TEST(DDTableTest, FixedCapacityUsesMatchingInitialAndMaximum) {
  auto manager = MemoryManager::create<vNode>();
  auto table = ::mqt::test::value(UniqueTable::create(
      manager, {.nVars = 1, .nBuckets = 1, .maxBuckets = 1}));
  for (const auto bit : {false, true}) {
    auto* node = manager.get<vNode>();
    node->v = 0;
    node->e = bit ? std::array{vEdge::zero(), vEdge::one()}
                  : std::array{vEdge::one(), vEdge::zero()};
    EXPECT_EQ(table.lookup(node), node);
    EXPECT_EQ(table.hash(*node), 0);
    EXPECT_EQ(table.getStats(0).numBuckets, 1);
  }
  EXPECT_EQ(table.getNumEntries(), 2);
}

TEST(DDTableTest, DefaultPackageGrowsPopulatedLevels) {
  auto packageOwner = ::mqt::test::value(Package::create(2));
  auto& package = *packageOwner;
  EXPECT_EQ(package.vUniqueTable.getStats(0).numBuckets, 64);
  for (size_t i = 1; i <= 65; ++i) {
    const auto angle = static_cast<double>(i) / 4096.;
    const auto state = ::mqt::test::value(
        makeStateFromVector(CVec{std::cos(angle), std::sin(angle)}, package));
    package.decRef(state);
  }
  EXPECT_GT(package.vUniqueTable.getStats(0).numBuckets, 64);
  EXPECT_EQ(package.vUniqueTable.getStats(1).numBuckets, 64);
  EXPECT_EQ(package.mUniqueTable.getStats(0).numBuckets, 64);
  package.garbageCollect(true);
  EXPECT_EQ(package.vUniqueTable.getNumEntries(), 0);
}

TEST(DDTableTest, UnaryCacheDistributesAlignedPointers) {
  struct alignas(64) Key {};
  std::array<Key, 64> keys{};
  auto table =
      ::mqt::test::value(UnaryComputeTable<const Key*, size_t>::create(64));
  for (size_t i = 0; i < keys.size(); ++i) {
    table.insert(&keys[i], i);
    ASSERT_NE(table.lookup(&keys[i]), nullptr);
    EXPECT_EQ(*table.lookup(&keys[i]), i);
  }
  EXPECT_GT(table.getStats().numEntries, 1);
  table.clear();
  EXPECT_EQ(table.getStats().numEntries, 0);
  for (const auto& key : keys) {
    EXPECT_EQ(table.lookup(&key), nullptr);
  }
}

TEST(DDTableTest, AdaptiveMultiplicationCachePreservesStateAcrossCollection) {
  CVec input(8);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<fp>(i + 1U) / std::sqrt(204.);
  }
  auto expected = input;
  for (size_t i = 0; i < input.size(); i += 2) {
    expected[i] = SQRT2_2 * (input[i] + input[i + 1]);
    expected[i + 1] = SQRT2_2 * (input[i] - input[i + 1]);
  }

  for (const size_t initialBuckets : {4U, 64U}) {
    DDPackageConfig config;
    config.ctMatVecMultNumBucket = initialBuckets;
    auto packageOwner = ::mqt::test::value(Package::create(3, config));
    auto& package = *packageOwner;
    const auto state = ::mqt::test::value(makeStateFromVector(input, package));
    const auto gate = ::mqt::test::value(
        package.makeGateDD(GateMatrix{SQRT2_2, SQRT2_2, SQRT2_2, -SQRT2_2}, 0));
    package.incRef(gate);
    const auto& stats = package.matrixVectorMultiplication.getStats();
    static_cast<void>(package.multiply(gate, state));
    ASSERT_TRUE(package.garbageCollect(true));
    EXPECT_EQ(stats.numBuckets, initialBuckets);
    for (size_t i = 0; i < 2 * initialBuckets; ++i) {
      const auto result = package.multiply(gate, state);
      EXPECT_NEAR(::mqt::test::value(result.getValueByIndex(0)).real(),
                  expected[0].real(), 1e-12);
    }
    ASSERT_GE(stats.hits, initialBuckets);
    ASSERT_TRUE(package.garbageCollect(true));
    EXPECT_EQ(stats.numBuckets, initialBuckets == 4 ? 32 : initialBuckets);
    EXPECT_EQ(stats.numEntries, 0);
    const auto result = package.multiply(gate, state).getVector();
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_NEAR(result[i].real(), expected[i].real(), 1e-12);
      EXPECT_NEAR(result[i].imag(), expected[i].imag(), 1e-12);
    }
    package.decRef(state);
    package.decRef(gate);
  }
}

TEST(DDTableTest, MemoryManagerGrowthReuseAndResetStatistics) {
  const auto check = []<class T> {
    auto manager = MemoryManager::create<T>(2);
    auto* first = manager.template get<T>();
    auto* second = manager.template get<T>();
    if constexpr (std::derived_from<T, NodeBase>) {
      EXPECT_EQ(first->flags, 0);
      EXPECT_EQ(second->flags, 0);
    }
    first->setNext(second);
    auto* third = manager.template get<T>();
    EXPECT_EQ(first->next(), second);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(first) % alignof(T), 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(third) % alignof(T), 0);
    if constexpr (std::derived_from<T, NodeBase>) {
      EXPECT_EQ(third->flags, 0);
    }
    const auto& stats = manager.getStats();
    EXPECT_NE(first, second);
    EXPECT_NE(first, third);
    EXPECT_NE(second, third);
    EXPECT_EQ(stats.numAllocated, 6);
    EXPECT_EQ(stats.numAllocations, 2);
    EXPECT_EQ(stats.numUsed, 3);
    EXPECT_EQ(stats.peakNumUsed, 3);
    manager.returnEntry(*first);
    manager.returnEntry(*second);
    EXPECT_EQ(stats.numAvailableForReuse, 2);
    EXPECT_EQ(stats.peakNumAvailableForReuse, 2);
    EXPECT_EQ(manager.template get<T>(), second);
    EXPECT_EQ(manager.template get<T>(), first);
    EXPECT_EQ(stats.numAvailableForReuse, 0);
    EXPECT_EQ(stats.numUsed, 3);
    manager.reset(true);
    EXPECT_EQ(stats.numAllocated, 6);
    EXPECT_EQ(stats.numAllocations, 3);
    EXPECT_EQ(stats.numUsed, 0);
    EXPECT_EQ(stats.peakNumUsed, 3);
    EXPECT_EQ(stats.peakNumAvailableForReuse, 2);
    manager.reset(true);
    EXPECT_EQ(stats.numAllocations, 3);
    for (size_t i = 0; i < 6; ++i) {
      EXPECT_NE(manager.template get<T>(), nullptr);
    }
    EXPECT_EQ(stats.numAllocations, 3);
    EXPECT_EQ(stats.numUsed, 6);
    EXPECT_EQ(stats.peakNumUsed, 6);
    manager.reset();
    EXPECT_EQ(stats.numAllocated, 6);
    EXPECT_EQ(stats.numUsed, 0);
    EXPECT_EQ(stats.numAvailableForReuse, 0);
  };
  check.operator()<vNode>();
  check.operator()<mNode>();
  check.operator()<RealNumber>();
}
