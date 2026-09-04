/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Compiler/Target.h"
#include "mlir/Dialect/MQT/IR/MQTAttributes.h"
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"

#include <gtest/gtest.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Error.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace mqt::test::compiler {
template <class T> [[nodiscard]] static T valid(llvm::Expected<T> value) {
  return llvm::cantFail(std::move(value));
}

template <class T>
static void expectInvalid(llvm::Expected<T> value,
                          const std::string_view expectedMessage) {
  ASSERT_FALSE(value);
  EXPECT_EQ(llvm::toString(value.takeError()), expectedMessage);
}

namespace {

using Target = mlir::CompilerTarget;
using Connectivity = Target::Connectivity;
using Coupling = Target::Coupling;
using DurationUnit = Target::DurationUnit;
using GateKind = Target::GateKind;
using Operation = Target::Operation;
using Arity = Operation::Arity;
using NativeOperations = Target::NativeOperations;
using Site = Target::Site;
using SiteId = Target::SiteId;
using SiteTuple = Target::SiteTuple;

TEST(CompilerTargetTest, ConstructsDetailedNamedTargetAndSharesStorage) {
  std::vector<Site> sites;
  sites.emplace_back(valid(Site::create(7, "left", 100, 80)));
  sites.emplace_back(valid(Site::create(2, std::nullopt, 120, std::nullopt)));
  sites.emplace_back(valid(Site::create(11, "right")));

  std::vector<Operation> operations;
  std::vector siteTuples{valid(SiteTuple::create({7}, 0, 0.99)),
                         valid(SiteTuple::create({2}, 5, 0.98)),
                         valid(SiteTuple::create({11}))};
  operations.emplace_back(
      valid(Operation::create(" PRX ", 1, 2, std::move(siteTuples), 0, 0.97)));

  const auto target = valid(
      Target::create("device", std::move(sites),
                     Connectivity::fromCouplings({{11, 2}, {2, 11}, {7, 2}}),
                     NativeOperations::fromOperations(operations),
                     valid(DurationUnit::create("ns", 0.5))));
  // The copy itself is the behavior under test: both objects must share the
  // immutable backing storage.
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  const auto copy = target;

  ASSERT_TRUE(target.name());
  EXPECT_EQ(*target.name(), "device");
  ASSERT_TRUE(target.durationUnit());
  EXPECT_EQ(target.durationUnit()->unit(), "ns");
  EXPECT_DOUBLE_EQ(target.durationUnit()->scaleFactor(), 0.5);
  ASSERT_EQ(target.sites().size(), 3);
  EXPECT_EQ(target.sites()[0].id(), 7);
  ASSERT_TRUE(target.sites()[0].name());
  EXPECT_EQ(*target.sites()[0].name(), "left");
  EXPECT_EQ(target.sites()[0].t1(), 100);
  EXPECT_EQ(target.sites()[0].t2(), 80);
  EXPECT_EQ(target.operations()[0].name(), " PRX ");
  EXPECT_EQ(target.operations()[0].canonicalName(), "r");
  EXPECT_EQ(target.operations()[0].arity(), Arity::fixed(1));
  EXPECT_EQ(target.operations()[0].numParameters(), 2);
  EXPECT_EQ(target.operations()[0].duration(), 0);
  EXPECT_EQ(target.operations()[0].fidelity(), 0.97);
  ASSERT_EQ(target.operations()[0].siteTuples().size(), 3);
  EXPECT_EQ(target.operations()[0].siteTuples()[0].duration(), 0);
  EXPECT_EQ(target.operations()[0].siteTuples()[0].fidelity(), 0.99);

  EXPECT_EQ(copy.sites().data(), target.sites().data());
  EXPECT_EQ(copy.couplings().data(), target.couplings().data());
  EXPECT_EQ(copy.operations().data(), target.operations().data());
}

TEST(CompilerTargetTest, ConstructsDenseUnnamedAllToAllTarget) {
  const auto target = valid(Target::create(3, Connectivity::allToAll(),
                                           NativeOperations::unrestricted()));
  const auto named =
      valid(Target::create("simulator", 2, Connectivity::allToAll(),
                           NativeOperations::unrestricted()));

  EXPECT_FALSE(target.name());
  ASSERT_TRUE(named.name());
  EXPECT_EQ(*named.name(), "simulator");
  EXPECT_EQ(named.siteIds(), (llvm::ArrayRef<SiteId>{0, 1}));
  EXPECT_FALSE(target.durationUnit());
  EXPECT_EQ(target.siteIds(), (llvm::ArrayRef<SiteId>{0, 1, 2}));
  EXPECT_EQ(target.vertexForSite(0), 0);
  EXPECT_EQ(target.vertexForSite(2), 2);
  EXPECT_FALSE(target.vertexForSite(3));
  EXPECT_EQ(target.siteForVertex(1), 1);
  EXPECT_EQ(target.connectivityKind(), Connectivity::Kind::AllToAll);
  EXPECT_TRUE(target.couplings().empty());
  EXPECT_TRUE(target.areAdjacent(0, 2));
  EXPECT_FALSE(target.areAdjacent(1, 1));
  EXPECT_EQ(target.distanceBetween(0, 2), 1);
  EXPECT_EQ(target.distanceBetween(2, 2), 0);
  EXPECT_EQ(target.maxDegree(), 2);

  std::vector<size_t> neighbours;
  target.forEachNeighbour(
      1, [&](const auto neighbour) { neighbours.emplace_back(neighbour); });
  EXPECT_EQ(neighbours, (std::vector<size_t>{0, 2}));
}

TEST(CompilerTargetTest, ModelsFixedAndVariadicOperationArities) {
  const auto zero = Arity::fixed(0);
  EXPECT_EQ(zero.kind(), Arity::Kind::Fixed);
  EXPECT_EQ(zero.value(), 0U);
  EXPECT_TRUE(zero.accepts(0));
  EXPECT_FALSE(zero.accepts(1));

  const auto fixed = Arity::fixed(2);
  EXPECT_FALSE(fixed.accepts(1));
  EXPECT_TRUE(fixed.accepts(2));
  EXPECT_FALSE(fixed.accepts(3));

  const auto variadic = Arity::variadic(2);
  EXPECT_EQ(variadic.kind(), Arity::Kind::Variadic);
  EXPECT_EQ(variadic.value(), 2U);
  EXPECT_FALSE(variadic.accepts(1));
  EXPECT_TRUE(variadic.accepts(2));
  EXPECT_TRUE(variadic.accepts(7));
}

TEST(CompilerTargetTest, PreservesFullNonnegativeSiteIdDomain) {
  constexpr auto maxSite = std::numeric_limits<SiteId>::max();
  constexpr auto nextSite = maxSite - 1;
  auto siteTuple = valid(SiteTuple::create({maxSite, nextSite}));
  std::vector sites{valid(Site::create(maxSite)),
                    valid(Site::create(nextSite))};
  const auto target = valid(Target::create(
      std::move(sites), Connectivity::fromCouplings({{maxSite, nextSite}}),
      NativeOperations::fromOperations(
          {valid(Operation::create("cx", 2, 0, {std::move(siteTuple)}))})));

  EXPECT_EQ(target.siteIds(), (llvm::ArrayRef<SiteId>{maxSite, nextSite}));
  EXPECT_EQ(target.vertexForSite(maxSite), 0);
  EXPECT_EQ(target.vertexForSite(nextSite), 1);
  EXPECT_EQ(target.couplings(),
            (llvm::ArrayRef<Coupling>{{nextSite, maxSite}}));
  EXPECT_EQ(target.operations().front().siteTuples().front().sites(),
            (llvm::ArrayRef<SiteId>{maxSite, nextSite}));
}

TEST(CompilerTargetTest, CanonicalizesConnectedTopologyAndCachesDistances) {
  std::vector sites{valid(Site::create(7)), valid(Site::create(2)),
                    valid(Site::create(11))};
  const auto target = valid(Target::create(
      std::move(sites),
      Connectivity::fromCouplings({{11, 2}, {2, 11}, {7, 2}, {2, 7}}),
      NativeOperations::unrestricted()));

  EXPECT_EQ(target.connectivityKind(), Connectivity::Kind::Explicit);
  EXPECT_EQ(target.couplings(), (llvm::ArrayRef<Coupling>{{2, 7}, {2, 11}}));
  EXPECT_EQ(target.vertexForSite(7), 0);
  EXPECT_EQ(target.vertexForSite(2), 1);
  EXPECT_EQ(target.vertexForSite(11), 2);
  EXPECT_TRUE(target.areAdjacent(0, 1));
  EXPECT_TRUE(target.areAdjacent(1, 2));
  EXPECT_FALSE(target.areAdjacent(0, 2));
  EXPECT_EQ(target.distanceBetween(0, 2), 2);
  EXPECT_EQ(target.distanceBetween(2, 0), 2);
  EXPECT_EQ(target.maxDegree(), 2);

  std::vector<size_t> neighbours;
  target.forEachNeighbour(
      1, [&](const auto neighbour) { neighbours.emplace_back(neighbour); });
  EXPECT_EQ(neighbours, (std::vector<size_t>{0, 2}));
}

TEST(CompilerTargetTest, RejectsInvalidMetadata) {
  expectInvalid(Target::create(0, Connectivity::allToAll(),
                               NativeOperations::unrestricted()),
                "Compiler target must contain at least one site");
  if constexpr (sizeof(size_t) >= sizeof(uint64_t)) {
    expectInvalid(
        Target::create(std::numeric_limits<size_t>::max(),
                       Connectivity::allToAll(),
                       NativeOperations::unrestricted()),
        "Compiler target site count exceeds the nonnegative i64 site domain");
  }
  expectInvalid(Site::create(-1),
                "Compiler target site ID must be nonnegative");
  expectInvalid(Site::create(0, ""),
                "Compiler target site name must not be empty when present");
  expectInvalid(Site::create(0, std::nullopt, 0),
                "Compiler target site T1 must be positive");
  expectInvalid(Site::create(0, std::nullopt, std::nullopt, 0),
                "Compiler target site T2 must be positive");
  expectInvalid(DurationUnit::create("", 1.),
                "Compiler target duration unit must not be empty");
  expectInvalid(
      DurationUnit::create("ns", 0.),
      "Compiler target duration scale factor must be positive and finite");
  expectInvalid(
      DurationUnit::create("ns", std::numeric_limits<double>::infinity()),
      "Compiler target duration scale factor must be positive and finite");
  expectInvalid(SiteTuple::create({0, 0}),
                "Compiler target site tuple contains a duplicate site");
  expectInvalid(SiteTuple::create({-1}),
                "Compiler target site tuple contains a negative site ID");
  expectInvalid(
      SiteTuple::create({0}, std::nullopt, -0.1),
      "Compiler target site-tuple fidelity must be finite and in [0, 1]");
  expectInvalid(Operation::create("", 1, 0),
                "Compiler target operation name must not be empty");
  expectInvalid(Operation::create("x", Arity::variadic(0), 0),
                "Compiler target operation variadic minimum must be positive");
  expectInvalid(
      Operation::create(
          "gphase", Arity::fixed(0), 1,
          std::vector{valid(SiteTuple::create(std::vector<SiteId>{}))}),
      "Compiler target zero-arity operation cannot contain site tuples");
  expectInvalid(
      Operation::create(
          "h", Arity::variadic(1), 0,
          std::vector{valid(SiteTuple::create(std::vector<SiteId>{0}))}),
      "Compiler target variadic operation cannot contain site tuples");
  expectInvalid(
      Operation::create("x", 1, 0,
                        std::vector{valid(SiteTuple::create({0, 1}))}),
      "Compiler target operation site tuple does not match its arity");
  expectInvalid(Operation::create("x", 1, 0,
                                  std::vector{valid(SiteTuple::create({0})),
                                              valid(SiteTuple::create({0}))}),
                "Compiler target operation contains a duplicate site tuple");
  expectInvalid(
      Operation::create("x", 1, 0, {}, std::nullopt,
                        std::numeric_limits<double>::quiet_NaN()),
      "Compiler target operation fidelity must be finite and in [0, 1]");

  expectInvalid(Target::create(std::vector<Site>{}, Connectivity::allToAll(),
                               NativeOperations::unrestricted()),
                "Compiler target must contain at least one site");
  expectInvalid(Target::create("", 1, Connectivity::allToAll(),
                               NativeOperations::unrestricted()),
                "Compiler target name must not be empty when present");
  expectInvalid(Target::create("invalid", 0, Connectivity::allToAll(),
                               NativeOperations::unrestricted()),
                "Compiler target must contain at least one site");
  expectInvalid(Target::create(
                    std::vector{valid(Site::create(1)), valid(Site::create(1))},
                    Connectivity::allToAll(), NativeOperations::unrestricted()),
                "Compiler target contains duplicate site IDs");
  expectInvalid(Target::create(
                    std::vector{valid(Site::create(0, std::nullopt, 1))},
                    Connectivity::allToAll(), NativeOperations::unrestricted()),
                "Compiler target timing metadata requires a duration unit");
  expectInvalid(Target::create(1, Connectivity::allToAll(),
                               NativeOperations::fromOperations({valid(
                                   Operation::create("x", 1, 0, {}, 1))})),
                "Compiler target timing metadata requires a duration unit");
  expectInvalid(
      Target::create(
          1, Connectivity::allToAll(),
          NativeOperations::fromOperations({valid(Operation::create(
              "x", 1, 0, std::vector{valid(SiteTuple::create({0}, 1))}))})),
      "Compiler target timing metadata requires a duration unit");
  expectInvalid(Target::create(2, Connectivity::fromCouplings({{0, 0}}),
                               NativeOperations::unrestricted()),
                "Compiler target topology contains a self-coupling");
  expectInvalid(Target::create(2, Connectivity::fromCouplings({{0, 2}}),
                               NativeOperations::unrestricted()),
                "Compiler target topology references an unknown site");
  expectInvalid(Target::create(3, Connectivity::fromCouplings({{0, 1}}),
                               NativeOperations::unrestricted()),
                "Compiler target topology must be connected");
  expectInvalid(
      Target::create(
          2, Connectivity::allToAll(),
          NativeOperations::fromOperations({valid(Operation::create(
              "x", 1, 0, std::vector{valid(SiteTuple::create({2}))}))})),
      "Compiler target operation site tuple references an unknown site");
  expectInvalid(Target::create(1, Connectivity::allToAll(),
                               NativeOperations::fromOperations(
                                   {valid(Operation::create("cx", 2, 0))})),
                "Compiler target operation arity exceeds its site count");
  expectInvalid(
      Target::create(2, Connectivity::allToAll(),
                     NativeOperations::fromOperations({valid(
                         Operation::create("h", Arity::variadic(3), 0))})),
      "Compiler target operation variadic minimum exceeds its site count");
}

TEST(CompilerTargetTest, DistinguishesOperationSupport) {
  const auto unrestricted = valid(Target::create(
      2, Connectivity::allToAll(), NativeOperations::unrestricted()));
  const auto closed = valid(Target::create(
      2, Connectivity::allToAll(), NativeOperations::fromOperations({})));
  const auto variadic = valid(Target::create(
      4, Connectivity::allToAll(),
      NativeOperations::fromOperations(
          {valid(Operation::create("gphase", Arity::fixed(0), 1)),
           valid(Operation::create("h", Arity::variadic(1), 0)),
           valid(Operation::create("rxx", Arity::variadic(2), 1)),
           valid(Operation::create("I", Arity::fixed(1), 0))})));

  EXPECT_EQ(unrestricted.nativeOperationsKind(),
            NativeOperations::Kind::Unrestricted);
  EXPECT_EQ(unrestricted.supportsOperation("device.operation", 1), true);
  EXPECT_EQ(unrestricted.supports(GateKind::CX), true);
  EXPECT_EQ(unrestricted.supportsOperation("", 1), false);
  EXPECT_EQ(unrestricted.supportsOperation("   ", 1), false);
  EXPECT_EQ(unrestricted.supportsOperation("gphase", 0), true);
  EXPECT_EQ(unrestricted.supportsOperation("x", 3), false);

  EXPECT_EQ(closed.nativeOperationsKind(), NativeOperations::Kind::Explicit);
  EXPECT_TRUE(closed.operations().empty());
  EXPECT_EQ(closed.supportsOperation("x", 1), false);
  EXPECT_EQ(closed.supports(GateKind::CX), false);
  EXPECT_TRUE(closed.supportedGates().empty());
  EXPECT_FALSE(closed.synthesisBasis());

  EXPECT_TRUE(variadic.supportsOperation("gphase", 0, 1));
  EXPECT_FALSE(variadic.supportsOperation("gphase", 1, 1));
  EXPECT_FALSE(variadic.supportsOperation("h", 0, 0));
  EXPECT_TRUE(variadic.supportsOperation("h", 1, 0));
  EXPECT_TRUE(variadic.supportsOperation("h", 4, 0));
  EXPECT_FALSE(variadic.supportsOperation("h", 5, 0));
  EXPECT_FALSE(variadic.supportsOperation("rxx", 1, 1));
  EXPECT_TRUE(variadic.supportsOperation("rxx", 2, 1));
  EXPECT_TRUE(variadic.supportsOperation("rxx", 4, 1));
  EXPECT_FALSE(variadic.supportsOperation("rxx", 4, 0));
  EXPECT_TRUE(variadic.supportsOperation("id", 1, 0));
  EXPECT_TRUE(variadic.supportsOperation("i", 1, 0));
}

TEST(CompilerTargetTest, PreservesCalibrationAndResolvesHomogeneousBasis) {
  const std::vector<Coupling> chain{{0, 1}, {1, 2}};
  const auto globalU = valid(Operation::create("U3", 1, 3));
  const auto cz = valid(
      Operation::create("cz", 2, 0,
                        std::vector{valid(SiteTuple::create({1, 0}, 5, 0.99)),
                                    valid(SiteTuple::create({1, 2}))},
                        7, 0.98));
  const auto target =
      valid(Target::create(3, Connectivity::fromCouplings(chain),
                           NativeOperations::fromOperations({globalU, cz}),
                           valid(DurationUnit::create("ns", 1.))));

  EXPECT_EQ(target.supportsOperation("u", 1, 3), true);
  EXPECT_EQ(target.supportsOperation(" U3 ", 1, 3), true);
  EXPECT_EQ(target.supports(GateKind::CZ), true);
  EXPECT_TRUE(llvm::is_contained(target.supportedGates(), GateKind::CZ));
  ASSERT_EQ(target.operations().size(), 2U);
  ASSERT_EQ(target.operations()[1].siteTuples().size(), 2U);
  EXPECT_EQ(target.operations()[1].siteTuples()[0].sites(),
            (llvm::ArrayRef<SiteId>{1, 0}));
  EXPECT_EQ(target.operations()[1].siteTuples()[0].duration(), 5);
  EXPECT_EQ(target.operations()[1].siteTuples()[0].fidelity(), 0.99);
  EXPECT_FALSE(target.operations()[1].siteTuples()[1].duration());
  EXPECT_FALSE(target.operations()[1].siteTuples()[1].fidelity());
  EXPECT_EQ(target.operations()[1].duration(), 7);
  EXPECT_EQ(target.operations()[1].fidelity(), 0.98);
  EXPECT_TRUE(target.supports(GateKind::CZ, {1, 2}));
  EXPECT_FALSE(target.supports(GateKind::CZ, {2, 1}));
  ASSERT_TRUE(target.synthesisBasis());
  EXPECT_EQ(target.synthesisBasis()->singleQubit, Target::SingleQubitBasis::U);
  EXPECT_EQ(target.synthesisBasis()->entangler, GateKind::CZ);
}

TEST(CompilerTargetTest, RoundTripsTypedCompilationTargetAttribute) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::mqt::MQTDialect>();

  std::vector sites{valid(Site::create(7, "left", 100, 80)),
                    valid(Site::create(2, std::nullopt, 120, std::nullopt)),
                    valid(Site::create(11, "right"))};
  std::vector operations{
      valid(
          Operation::create(" PRX ", 1, 2,
                            std::vector{valid(SiteTuple::create({7}, 0, 0.99)),
                                        valid(SiteTuple::create({2}, 5, 0.98))},
                            0, 0.97)),
      valid(Operation::create("gphase", Arity::fixed(0), 1)),
      valid(Operation::create("h", Arity::variadic(1), 0))};
  const auto target =
      valid(Target::create("device", std::move(sites),
                           Connectivity::fromCouplings({{7, 2}, {2, 11}}),
                           NativeOperations::fromOperations(operations),
                           valid(DurationUnit::create("ns", 0.5))));

  const auto attribute = target.materialize(context);
  const auto reconstructed = valid(Target::create(attribute));

  EXPECT_EQ(reconstructed.materialize(context), attribute);
  EXPECT_EQ(reconstructed.couplings(), target.couplings());
  EXPECT_EQ(reconstructed.supportsOperation("r", 1, 2), true);
  EXPECT_TRUE(reconstructed.supportsOperation("r", 1, 2, {7}));
  EXPECT_FALSE(reconstructed.supportsOperation("r", 1, 2, {11}));
  EXPECT_EQ(reconstructed.supportsOperation("gphase", 0, 1), true);
  EXPECT_EQ(reconstructed.supportsOperation("h", 3, 0), true);
  EXPECT_EQ(reconstructed.operations()[1].arity(), Arity::fixed(0));
  EXPECT_EQ(reconstructed.operations()[2].arity(), Arity::variadic(1));
  EXPECT_EQ(reconstructed.synthesisBasis(), target.synthesisBasis());
}

TEST(CompilerTargetTest, SupportsMaximumSiteIds) {
  constexpr auto maxSite = std::numeric_limits<SiteId>::max();
  constexpr auto nextSite = maxSite - 1;
  std::vector sites{valid(Site::create(nextSite)),
                    valid(Site::create(maxSite))};
  const auto x = valid(
      Operation::create("x", 1, 0,
                        std::vector{valid(SiteTuple::create({nextSite})),
                                    valid(SiteTuple::create({maxSite}))}));
  const auto cx = valid(Operation::create(
      "cx", 2, 0, std::vector{valid(SiteTuple::create({nextSite, maxSite}))}));
  const auto target =
      valid(Target::create(std::move(sites), Connectivity::allToAll(),
                           NativeOperations::fromOperations({x, cx})));

  EXPECT_TRUE(target.supports(GateKind::X, {nextSite}));
  EXPECT_TRUE(target.supports(GateKind::X, {maxSite}));
  EXPECT_TRUE(target.supports(GateKind::CX, {nextSite, maxSite}));

  mlir::MLIRContext context;
  context.loadDialect<mlir::mqt::MQTDialect>();
  const auto attribute = target.materialize(context);
  EXPECT_EQ(valid(Target::create(attribute)).materialize(context), attribute);
}

TEST(CompilerTargetTest, RoundTripsSupportedTargetStates) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::mqt::MQTDialect>();

  const std::array targets{
      valid(Target::create(2, Connectivity::allToAll(),
                           NativeOperations::unrestricted())),
      valid(Target::create(2, Connectivity::fromCouplings({{0, 1}}),
                           NativeOperations::fromOperations({}))),
  };

  for (const auto& target : targets) {
    const auto attribute = target.materialize(context);
    const auto reconstructed = valid(Target::create(attribute));
    EXPECT_EQ(reconstructed.materialize(context), attribute);
  }

  expectInvalid(Target::create(mlir::mqt::CompilationTargetAttr{}),
                "Compiler target attribute must not be null");

  const auto site =
      mlir::mqt::SiteAttr::get(&context, 0, {}, std::nullopt, std::nullopt);
  const auto secondSite =
      mlir::mqt::SiteAttr::get(&context, 1, {}, std::nullopt, std::nullopt);
  expectInvalid(Target::create(mlir::mqt::CompilationTargetAttr::get(
                    &context, {}, {site, secondSite}, {},
                    mlir::mqt::ConnectivityKind::Explicit, {},
                    mlir::mqt::NativeOperationsKind::Unrestricted, {})),
                "Compiler target topology must be connected");
}

TEST(CompilerTargetTest, EnforcesExactOrderedOperationApplicability) {
  std::vector sites{valid(Site::create(10)), valid(Site::create(20)),
                    valid(Site::create(30))};
  const auto globalU = valid(Operation::create("u", 1, 3));
  const auto restrictedX = valid(Operation::create(
      "x", 1, 0, std::vector{valid(SiteTuple::create({10}))}));
  const auto directionalCX =
      valid(Operation::create("cx", 2, 0,
                              std::vector{valid(SiteTuple::create({10, 20})),
                                          valid(SiteTuple::create({20, 30}))}));
  const auto exactCZ = valid(Operation::create(
      "cz", 2, 0, std::vector{valid(SiteTuple::create({10, 20}))}));
  const auto threeQubit = valid(
      Operation::create("device.operation", 3, 0,
                        std::vector{valid(SiteTuple::create({10, 20, 30}))}));
  const auto target = valid(Target::create(
      std::move(sites), Connectivity::fromCouplings({{10, 20}, {20, 30}}),
      NativeOperations::fromOperations(
          {globalU, restrictedX, directionalCX, exactCZ, threeQubit})));

  EXPECT_TRUE(globalU.siteTuples().empty());
  EXPECT_FALSE(restrictedX.siteTuples().empty());
  EXPECT_FALSE(directionalCX.siteTuples().empty());

  EXPECT_TRUE(target.supports(GateKind::U, {30}));
  EXPECT_TRUE(target.supports(GateKind::X, {10}));
  EXPECT_FALSE(target.supports(GateKind::X, {20}));
  EXPECT_TRUE(target.supports(GateKind::CX, {10, 20}));
  EXPECT_FALSE(target.supports(GateKind::CX, {20, 10}));
  EXPECT_TRUE(target.supports(GateKind::CX, {20, 30}));
  EXPECT_FALSE(target.supports(GateKind::CX, {30, 20}));
  EXPECT_FALSE(target.supports(GateKind::CX, {10, 30}));
  EXPECT_TRUE(target.supports(GateKind::CZ, {10, 20}));
  EXPECT_FALSE(target.supports(GateKind::CZ, {20, 10}));
  EXPECT_FALSE(target.supports(GateKind::ECR));
  EXPECT_FALSE(target.supports(GateKind::ECR, {10, 20}));
  EXPECT_FALSE(target.supports(GateKind::CX, {10}));
  EXPECT_FALSE(target.supports(GateKind::CX, {10, 10}));
  EXPECT_FALSE(target.supports(GateKind::CX, {10, 40}));
  EXPECT_TRUE(target.supportsOperation("device.operation", 3, 0, {10, 20, 30}));
  EXPECT_FALSE(
      target.supportsOperation("device.operation", 3, 0, {30, 20, 10}));
}

TEST(CompilerTargetTest, ClassifiesEveryEntangler) {
  using Entangler = std::tuple<GateKind, std::string_view, size_t>;
  const std::array entanglers{Entangler{GateKind::CZ, "cz", 0},
                              Entangler{GateKind::RXX, "rxx", 1},
                              Entangler{GateKind::RYY, "ryy", 1},
                              Entangler{GateKind::RZZ, "rzz", 1},
                              Entangler{GateKind::ISWAP, "iswap", 0},
                              Entangler{GateKind::CX, "cx", 0},
                              Entangler{GateKind::ECR, "ecr", 0},
                              Entangler{GateKind::RZX, "rzx", 1}};
  const std::vector<Coupling> chain{{0, 1}, {1, 2}};
  const auto globalU = valid(Operation::create("u", 1, 3));

  for (const auto& [gate, name, numParameters] : entanglers) {
    SCOPED_TRACE(name);
    const auto operation =
        valid(Operation::create(std::string{name}, 2, numParameters));
    const auto target = valid(
        Target::create(3, Connectivity::fromCouplings(chain),
                       NativeOperations::fromOperations({globalU, operation})));
    EXPECT_TRUE(llvm::is_contained(target.supportedGates(), gate));
    EXPECT_EQ(target.supports(gate), true);
    ASSERT_TRUE(target.synthesisBasis());
    EXPECT_EQ(target.synthesisBasis()->entangler, gate);
  }
}

TEST(CompilerTargetTest, DerivesControlledEntanglersFromVariadicBases) {
  constexpr std::array bases{
      std::pair{std::string_view{"x"}, GateKind::CX},
      std::pair{std::string_view{"z"}, GateKind::CZ},
  };
  const auto globalU = valid(Operation::create("u", 1, 3));

  for (const auto& [base, entangler] : bases) {
    SCOPED_TRACE(base);
    const auto variadic =
        valid(Operation::create(std::string{base}, Arity::variadic(1), 0));
    const auto target = valid(
        Target::create(2, Connectivity::allToAll(),
                       NativeOperations::fromOperations({globalU, variadic})));

    EXPECT_TRUE(target.supports(entangler));
    EXPECT_TRUE(llvm::is_contained(target.supportedGates(), entangler));
    ASSERT_TRUE(target.synthesisBasis());
    EXPECT_EQ(target.synthesisBasis()->singleQubit,
              Target::SingleQubitBasis::U);
    EXPECT_EQ(target.synthesisBasis()->entangler, entangler);
  }
}

TEST(CompilerTargetTest, ResolvesLargeAllToAllVariadicBasis) {
  constexpr size_t numSites = 65'535;
  const auto target = valid(Target::create(
      numSites, Connectivity::allToAll(),
      NativeOperations::fromOperations(
          {valid(Operation::create("u", 1, 3)),
           valid(Operation::create("x", Arity::variadic(1), 0))})));

  ASSERT_TRUE(target.synthesisBasis());
  EXPECT_EQ(target.synthesisBasis()->singleQubit, Target::SingleQubitBasis::U);
  EXPECT_EQ(target.synthesisBasis()->entangler, GateKind::CX);
}

TEST(CompilerTargetTest, SupportsRealQCOOperationsAndStructuralOps) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::qco::QCODialect, mlir::qtensor::QTensorDialect,
                  mlir::arith::ArithDialect, mlir::func::FuncDialect>();
  mlir::MLIRContext context;
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  auto moduleOp = mlir::qco::QCOProgramBuilder::build(
      &context, [](mlir::qco::QCOProgramBuilder& builder) {
        auto q0 = builder.staticQubit(0);
        auto q1 = builder.staticQubit(1);
        q0 = builder.x(q0);
        std::tie(q0, q1) = builder.cx(q0, q1);
        auto barrierResults = builder.barrier({q0, q1});
        q0 = barrierResults[0];
        q1 = barrierResults[1];
        std::tie(q0, q1) = builder.cz(q0, q1);
        builder.gphase(0.25);
        auto [measured, result] = builder.measure(q0);
        static_cast<void>(result);
        q0 = builder.reset(measured);
        static_cast<void>(q0);
        static_cast<void>(q1);
        return builder.intConstant(0);
      });
  ASSERT_TRUE(moduleOp);

  mlir::Operation* x = nullptr;
  mlir::Operation* cx = nullptr;
  mlir::Operation* cz = nullptr;
  mlir::Operation* measure = nullptr;
  mlir::Operation* reset = nullptr;
  mlir::Operation* barrier = nullptr;
  mlir::Operation* gphase = nullptr;
  moduleOp->walk([&](mlir::Operation* operation) {
    if (mlir::isa<mlir::qco::XOp>(operation) && x == nullptr) {
      x = operation;
    } else if (auto controlled = mlir::dyn_cast<mlir::qco::CtrlOp>(operation)) {
      auto* body = controlled.getBodyUnitary(0).getOperation();
      if (mlir::isa<mlir::qco::XOp>(body)) {
        cx = operation;
      } else if (mlir::isa<mlir::qco::ZOp>(body)) {
        cz = operation;
      }
    } else if (mlir::isa<mlir::qco::MeasureOp>(operation)) {
      measure = operation;
    } else if (mlir::isa<mlir::qco::ResetOp>(operation)) {
      reset = operation;
    } else if (mlir::isa<mlir::qco::BarrierOp>(operation)) {
      barrier = operation;
    } else if (mlir::isa<mlir::qco::GPhaseOp>(operation)) {
      gphase = operation;
    }
  });
  ASSERT_NE(x, nullptr);
  ASSERT_NE(cx, nullptr);
  ASSERT_NE(cz, nullptr);
  ASSERT_NE(measure, nullptr);
  ASSERT_NE(reset, nullptr);
  ASSERT_NE(barrier, nullptr);
  ASSERT_NE(gphase, nullptr);

  std::vector sites{valid(Site::create(10)), valid(Site::create(20))};
  std::vector directionalTuples{valid(SiteTuple::create({10, 20}))};
  std::vector operations{
      valid(Operation::create("x", 1, 0)),
      valid(Operation::create("gphase", 0, 1)),
      valid(Operation::create("measure", 1, 0)),
      valid(Operation::create("reset", 1, 0)),
      valid(Operation::create("cnot", 2, 0, std::move(directionalTuples))),
      valid(Operation::create("cz", 2, 0))};
  const auto target =
      valid(Target::create(std::move(sites), Connectivity::allToAll(),
                           NativeOperations::fromOperations(operations)));
  EXPECT_TRUE(target.supports(x));
  EXPECT_TRUE(target.supports(cx));
  EXPECT_TRUE(target.supports(cz));
  EXPECT_TRUE(target.supports(measure));
  EXPECT_TRUE(target.supports(reset));
  EXPECT_TRUE(target.supports(barrier));
  EXPECT_TRUE(target.supports(gphase));
  EXPECT_TRUE(target.supports(cx, {10, 20}));
  EXPECT_FALSE(target.supports(cx, {20, 10}));
  EXPECT_TRUE(target.supports(measure, {10}));
  EXPECT_TRUE(target.supports(reset, {10}));
  EXPECT_FALSE(target.supports(nullptr));
  EXPECT_FALSE(target.supports(nullptr, {10}));
  EXPECT_FALSE(target.supports(moduleOp->getOperation(), {10}));

  const auto closed = valid(Target::create(
      2, Connectivity::allToAll(), NativeOperations::fromOperations({})));
  EXPECT_EQ(closed.supports(barrier), true);
  EXPECT_EQ(closed.supports(gphase), false);
  EXPECT_EQ(closed.supports(x), false);
  EXPECT_EQ(closed.supports(measure), false);
}

TEST(CompilerTargetTest, SupportsArbitrarilyControlledBaseOperations) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::qco::QCODialect, mlir::qtensor::QTensorDialect,
                  mlir::arith::ArithDialect, mlir::func::FuncDialect>();
  mlir::MLIRContext context;
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  auto supportedModule = mlir::qco::QCOProgramBuilder::build(
      &context, [](mlir::qco::QCOProgramBuilder& builder) {
        static_cast<void>(
            builder.mch({builder.staticQubit(0), builder.staticQubit(1)},
                        builder.staticQubit(2)));
        static_cast<void>(
            builder.mcrx(0.25, {builder.staticQubit(3), builder.staticQubit(4)},
                         builder.staticQubit(5)));
        static_cast<void>(
            builder.mcrxx(0.5, {builder.staticQubit(6), builder.staticQubit(7)},
                          builder.staticQubit(8), builder.staticQubit(9)));
        static_cast<void>(
            builder.mcrccx({builder.staticQubit(10), builder.staticQubit(11)},
                           builder.staticQubit(12), builder.staticQubit(13),
                           builder.staticQubit(14)));
        return builder.intConstant(0);
      });
  ASSERT_TRUE(supportedModule);

  std::vector<mlir::Operation*> supportedControls;
  supportedModule->walk([&](mlir::qco::CtrlOp controlled) {
    supportedControls.emplace_back(controlled.getOperation());
  });
  ASSERT_EQ(supportedControls.size(), 4U);

  const auto target = valid(Target::create(
      5, Connectivity::allToAll(),
      NativeOperations::fromOperations(
          {valid(Operation::create("h", Arity::variadic(1), 0)),
           valid(Operation::create("rx", Arity::variadic(1), 1)),
           valid(Operation::create("rxx", Arity::variadic(2), 1)),
           valid(Operation::create("rccx", Arity::variadic(3), 0))})));
  for (auto* controlled : supportedControls) {
    EXPECT_TRUE(target.supports(controlled));
  }

  const auto fixedOnly = valid(
      Target::create(5, Connectivity::allToAll(),
                     NativeOperations::fromOperations(
                         {valid(Operation::create("h", Arity::fixed(3), 0))})));
  EXPECT_FALSE(fixedOnly.supports(supportedControls.front()));

  auto rejectedModule = mlir::qco::QCOProgramBuilder::build(
      &context, [](mlir::qco::QCOProgramBuilder& builder) {
        static_cast<void>(builder.mch({}, builder.staticQubit(0)));
        static_cast<void>(
            builder.ctrl({builder.staticQubit(1)},
                         {builder.staticQubit(2), builder.staticQubit(3)},
                         [&](mlir::ValueRange targets) {
                           return llvm::SmallVector<mlir::Value>{
                               builder.h(targets[0]), builder.x(targets[1])};
                         }));
        static_cast<void>(
            builder.ctrl({builder.staticQubit(4)},
                         {builder.staticQubit(5), builder.staticQubit(6)},
                         [&](mlir::ValueRange targets) {
                           return llvm::SmallVector<mlir::Value>{
                               builder.h(targets[0]), targets[1]};
                         }));
        return builder.intConstant(0);
      });
  ASSERT_TRUE(rejectedModule);

  std::vector<mlir::Operation*> rejectedControls;
  rejectedModule->walk([&](mlir::qco::CtrlOp controlled) {
    rejectedControls.emplace_back(controlled.getOperation());
  });
  ASSERT_EQ(rejectedControls.size(), 3U);
  for (auto* controlled : rejectedControls) {
    EXPECT_FALSE(target.supports(controlled));
  }
}

} // namespace
} // namespace mqt::test::compiler
