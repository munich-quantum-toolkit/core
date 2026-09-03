/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Compiler/MappingTarget.h"

#include <gtest/gtest.h>
#include <llvm/Support/Error.h>

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mqt::test::compiler {

using Target = mlir::CompilerTarget;
using Connectivity = Target::Connectivity;
using GateKind = Target::GateKind;
using MappingTarget = mlir::MappingTarget;
using NativeOperations = Target::NativeOperations;
using Operation = Target::Operation;
using SiteId = Target::SiteId;

template <class T>
[[nodiscard]] static T validMappingValue(llvm::Expected<T> value) {
  return llvm::cantFail(std::move(value));
}

[[nodiscard]] static Operation globalUMappingOperation() {
  return validMappingValue(Operation::create("u", 1, 3));
}

[[nodiscard]] static Operation
oneWayMappingGate(std::string name, size_t numParameters,
                  std::vector<std::vector<SiteId>> applicableSiteTuples) {
  return validMappingValue(Operation::create(std::move(name), 2, numParameters,
                                             {}, std::nullopt, std::nullopt,
                                             std::move(applicableSiteTuples)));
}

TEST(MappingTargetTest, CachesDirectionalCostsOnExplicitTopology) {
  const auto target = validMappingValue(
      Target::create(3, Connectivity::fromCouplings({{0, 1}, {1, 2}}),
                     NativeOperations::fromOperations(
                         {globalUMappingOperation(),
                          oneWayMappingGate("cx", 0, {{1, 0}, {1, 2}})})));

  const MappingTarget mappingTarget(target);
  EXPECT_EQ(mappingTarget.compilerTarget().sites().data(),
            target.sites().data());
  EXPECT_EQ(mappingTarget.numSites(), 3);
  EXPECT_EQ(mappingTarget.maxDegree(), 2);
  EXPECT_EQ(mappingTarget.distanceBetween(0, 2), 2);
  std::vector<size_t> neighbours;
  mappingTarget.forEachNeighbour(
      1, [&](size_t neighbour) { neighbours.emplace_back(neighbour); });
  EXPECT_EQ(neighbours, (std::vector<size_t>{0, 2}));
  EXPECT_FLOAT_EQ(mappingTarget.pathCostBetween(0, 1), 1.F);
  EXPECT_FLOAT_EQ(mappingTarget.pathCostBetween(1, 0), 0.F);
  EXPECT_FLOAT_EQ(mappingTarget.pathCostBetween(1, 2), 0.F);
  EXPECT_FLOAT_EQ(mappingTarget.pathCostBetween(2, 1), 1.F);
  EXPECT_FLOAT_EQ(mappingTarget.pathCostBetween(0, 2), 1.F);
  EXPECT_FALSE(mappingTarget.isExecutable(0, 1));
  EXPECT_TRUE(mappingTarget.isExecutable(1, 0));
  EXPECT_FALSE(mappingTarget.isExecutable(1, 1));
}

TEST(MappingTargetTest, TreatsSwapInvariantEntanglersAsBidirectional) {
  struct Gate {
    GateKind kind;
    std::string_view name;
    size_t numParameters;
  };
  constexpr std::array gates{
      Gate{GateKind::CZ, "cz", 0},   Gate{GateKind::ISWAP, "iswap", 0},
      Gate{GateKind::RXX, "rxx", 1}, Gate{GateKind::RYY, "ryy", 1},
      Gate{GateKind::RZZ, "rzz", 1},
  };

  for (const auto& [kind, name, numParameters] : gates) {
    SCOPED_TRACE(name);
    const auto target = validMappingValue(Target::create(
        2, Connectivity::fromCouplings({{0, 1}}),
        NativeOperations::fromOperations(
            {globalUMappingOperation(),
             oneWayMappingGate(std::string{name}, numParameters, {{0, 1}})})));
    ASSERT_TRUE(target.synthesisBasis());
    EXPECT_EQ(target.synthesisBasis()->entangler, kind);

    const MappingTarget mappingTarget(target);
    EXPECT_FLOAT_EQ(mappingTarget.pathCostBetween(0, 1), 0.F);
    EXPECT_FLOAT_EQ(mappingTarget.pathCostBetween(1, 0), 0.F);
    EXPECT_TRUE(mappingTarget.isExecutable(0, 1));
    EXPECT_TRUE(mappingTarget.isExecutable(1, 0));
  }
}

TEST(MappingTargetTest, KeepsTopologyOnlyTargetsUsable) {
  const auto target = validMappingValue(
      Target::create(3, Connectivity::fromCouplings({{0, 1}, {1, 2}}),
                     NativeOperations::fromOperations({})));
  const MappingTarget mappingTarget(target);

  EXPECT_FLOAT_EQ(mappingTarget.pathCostBetween(0, 0), 0.F);
  EXPECT_FLOAT_EQ(mappingTarget.pathCostBetween(0, 1), 0.F);
  EXPECT_FLOAT_EQ(mappingTarget.pathCostBetween(0, 2), 1.F);
  EXPECT_FALSE(mappingTarget.isExecutable(0, 0));
  EXPECT_TRUE(mappingTarget.isExecutable(0, 1));
  EXPECT_FALSE(mappingTarget.isExecutable(0, 2));
}

TEST(MappingTargetTest, SupportsImplicitAllToAllTopology) {
  const auto target = validMappingValue(Target::create(
      3, Connectivity::allToAll(),
      NativeOperations::fromOperations(
          {globalUMappingOperation(),
           oneWayMappingGate("cx", 0, {{0, 1}, {0, 2}, {1, 2}})})));
  const MappingTarget mappingTarget(target);

  EXPECT_FLOAT_EQ(mappingTarget.pathCostBetween(0, 2), 0.F);
  EXPECT_FLOAT_EQ(mappingTarget.pathCostBetween(2, 0), 1.F);
  EXPECT_TRUE(mappingTarget.isExecutable(0, 2));
  EXPECT_FALSE(mappingTarget.isExecutable(2, 0));
}

} // namespace mqt::test::compiler
