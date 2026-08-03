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
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"

#include <gtest/gtest.h>
#include <llvm/ADT/STLExtras.h>
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
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace mqt::test::compiler {
namespace {

using Target = mlir::CompilerTarget;
using Coupling = Target::Coupling;
using DurationUnit = Target::DurationUnit;
using GateKind = Target::GateKind;
using Operation = Target::Operation;
using OperationLocus = Target::OperationLocus;
using Site = Target::Site;
using SiteId = Target::SiteId;

TEST(CompilerTargetTest, ConstructsDetailedNamedTargetAndSharesStorage) {
  std::vector<Site> sites;
  sites.emplace_back(7, "left", 100, 80);
  sites.emplace_back(2, std::nullopt, 120, std::nullopt);
  sites.emplace_back(11, "right");

  std::vector<Operation> operations;
  operations.emplace_back(
      " PRX ", 1, 2,
      std::vector{OperationLocus{{7}, 0, 0.99}, OperationLocus{{2}, 5, 0.98}},
      0, 0.97);

  const Target target{"device", std::move(sites),
                      std::vector<Coupling>{{11, 2}, {2, 11}, {7, 2}},
                      std::move(operations), DurationUnit{"ns", 0.5}};
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
  EXPECT_EQ(target.operations()[0].providerName(), " PRX ");
  EXPECT_EQ(target.operations()[0].canonicalName(), "r");
  EXPECT_EQ(target.operations()[0].numQubits(), 1);
  EXPECT_EQ(target.operations()[0].numParameters(), 2);
  EXPECT_EQ(target.operations()[0].duration(), 0);
  EXPECT_EQ(target.operations()[0].fidelity(), 0.97);
  ASSERT_EQ(target.operations()[0].loci().size(), 2);
  EXPECT_EQ(target.operations()[0].loci()[0].duration(), 0);
  EXPECT_EQ(target.operations()[0].loci()[0].fidelity(), 0.99);

  EXPECT_EQ(copy.sites().data(), target.sites().data());
  EXPECT_EQ(copy.couplings().data(), target.couplings().data());
  EXPECT_EQ(copy.operations().data(), target.operations().data());
}

TEST(CompilerTargetTest, ConstructsDenseUnnamedAllToAllTarget) {
  const Target target{3};
  const Target named{"simulator", 2};

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
  EXPECT_FALSE(target.hasExplicitTopology());
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
  EXPECT_THROW(target.siteForVertex(3), std::out_of_range);
  EXPECT_THROW(target.areAdjacent(0, 3), std::out_of_range);
  EXPECT_THROW(target.distanceBetween(3, 0), std::out_of_range);
  EXPECT_THROW(target.forEachNeighbour(3, [](size_t) {}), std::out_of_range);
}

TEST(CompilerTargetTest, CanonicalizesConnectedTopologyAndCachesDistances) {
  const Target target{std::vector<Site>{Site{7}, Site{2}, Site{11}},
                      std::vector<Coupling>{{11, 2}, {2, 11}, {7, 2}, {2, 7}}};

  EXPECT_TRUE(target.hasExplicitTopology());
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
  const auto expectInvalid = [](const auto& construct) {
    EXPECT_THROW(construct(), std::invalid_argument);
  };

  expectInvalid([] { static_cast<void>(Target{0}); });
  if constexpr (sizeof(size_t) >= sizeof(uint64_t)) {
    expectInvalid(
        [] { static_cast<void>(Target{std::numeric_limits<size_t>::max()}); });
  }
  expectInvalid([] { static_cast<void>(Site{-1}); });
  expectInvalid([] { static_cast<void>(Site{0, ""}); });
  expectInvalid([] { static_cast<void>(Site{0, std::nullopt, 0}); });
  expectInvalid([] { static_cast<void>(DurationUnit{"", 1.}); });
  expectInvalid([] { static_cast<void>(DurationUnit{"ns", 0.}); });
  expectInvalid([] {
    static_cast<void>(
        DurationUnit{"ns", std::numeric_limits<double>::infinity()});
  });
  expectInvalid([] { static_cast<void>(OperationLocus{{0, 0}}); });
  expectInvalid(
      [] { static_cast<void>(OperationLocus{{0}, std::nullopt, -0.1}); });
  expectInvalid([] { static_cast<void>(Operation{"", 1, 0}); });
  expectInvalid([] { static_cast<void>(Operation{"x", 0, 0}); });
  expectInvalid([] {
    static_cast<void>(
        Operation{"x", 1, 0, std::vector{OperationLocus{{0, 1}}}});
  });
  expectInvalid([] {
    static_cast<void>(Operation{
        "x", 1, 0, std::vector{OperationLocus{{0}}, OperationLocus{{0}}}});
  });
  expectInvalid([] {
    static_cast<void>(Operation{"x", 1, 0, std::nullopt, std::nullopt,
                                std::numeric_limits<double>::quiet_NaN()});
  });

  expectInvalid([] { static_cast<void>(Target{std::vector<Site>{}}); });
  expectInvalid(
      [] { static_cast<void>(Target{std::vector<Site>{Site{1}, Site{1}}}); });
  expectInvalid([] {
    static_cast<void>(Target{std::vector<Site>{Site{0, std::nullopt, 1}}});
  });
  expectInvalid([] {
    static_cast<void>(Target{
        1, std::nullopt, std::vector{Operation{"x", 1, 0, std::nullopt, 1}}});
  });
  expectInvalid([] {
    std::vector<Operation> operations;
    operations.emplace_back("x", 1, 0, std::vector{OperationLocus{{0}, 1}});
    static_cast<void>(Target{1, std::nullopt, std::move(operations)});
  });
  expectInvalid(
      [] { static_cast<void>(Target{2, std::vector<Coupling>{{0, 0}}}); });
  expectInvalid(
      [] { static_cast<void>(Target{2, std::vector<Coupling>{{0, 2}}}); });
  expectInvalid(
      [] { static_cast<void>(Target{3, std::vector<Coupling>{{0, 1}}}); });
  expectInvalid([] {
    std::vector<Operation> operations;
    operations.emplace_back("x", 1, 0, std::vector{OperationLocus{{2}}});
    static_cast<void>(Target{2, std::nullopt, std::move(operations)});
  });
}

TEST(CompilerTargetTest, DistinguishesAbsentAndEmptyOperationSets) {
  const Target permissive{2};
  const Target closed{2, std::nullopt, std::vector<Operation>{}};
  const Operation globalX{"x", 1, 0};
  const Operation globalCX{"cx", 2, 0};

  EXPECT_FALSE(permissive.hasExplicitOperations());
  EXPECT_TRUE(permissive.operations().empty());
  EXPECT_TRUE(permissive.supportsOperation("provider.operation", {0}));
  EXPECT_TRUE(permissive.supports(GateKind::CX, {0, 1}));
  EXPECT_FALSE(permissive.supportsOperation("x", {0, 0}));
  EXPECT_FALSE(permissive.supportsOperation("x", {2}));
  EXPECT_FALSE(permissive.supportsOperation("", {0}));
  EXPECT_FALSE(permissive.supportsOperation("   ", {0}));
  EXPECT_FALSE(permissive.supportsOperation("x", {}));
  EXPECT_FALSE(globalX.supports({-1}));
  EXPECT_FALSE(globalCX.supports({0, 0}));

  EXPECT_TRUE(closed.hasExplicitOperations());
  EXPECT_TRUE(closed.operations().empty());
  EXPECT_FALSE(closed.supportsOperation("x", {0}));
  EXPECT_FALSE(closed.supports(GateKind::CX, {0, 1}));
  EXPECT_TRUE(closed.globallySupportedGates().empty());
  EXPECT_FALSE(closed.synthesisBasis());
}

TEST(CompilerTargetTest, PreservesRawLociAndResolvesBidirectionalBasis) {
  const std::vector<Coupling> chain{{0, 1}, {1, 2}};
  const Operation globalU{"U3", 1, 3};
  const Operation symmetricCZ{
      "cz", 2, 0, std::vector{OperationLocus{{1, 0}}, OperationLocus{{1, 2}}}};
  const Target symmetric{3, chain, std::vector{globalU, symmetricCZ}};

  EXPECT_TRUE(symmetric.supportsOperation("u", {0}, 3));
  EXPECT_TRUE(symmetric.supportsOperation(" U3 ", {2}, 3));
  ASSERT_EQ(symmetric.operations().size(), 2U);
  EXPECT_TRUE(symmetric.operations()[1].supports({1, 0}));
  EXPECT_FALSE(symmetric.operations()[1].supports({0, 1}));
  EXPECT_TRUE(symmetric.supports(GateKind::CZ, {1, 0}));
  EXPECT_TRUE(symmetric.supports(GateKind::CZ, {0, 1}));
  EXPECT_TRUE(
      llvm::is_contained(symmetric.globallySupportedGates(), GateKind::CZ));
  ASSERT_TRUE(symmetric.synthesisBasis());
  EXPECT_EQ(symmetric.synthesisBasis()->singleQubit,
            Target::SingleQubitBasis::U);
  EXPECT_EQ(symmetric.synthesisBasis()->entangler, GateKind::CZ);

  const Operation oneWayCX{
      "CNOT", 2, 0,
      std::vector{OperationLocus{{0, 1}}, OperationLocus{{1, 2}}}};
  const Target oneWay{3, chain, std::vector{globalU, oneWayCX}};
  EXPECT_TRUE(oneWay.supports(GateKind::CX, {0, 1}));
  EXPECT_FALSE(oneWay.supports(GateKind::CX, {1, 0}));
  EXPECT_FALSE(
      llvm::is_contained(oneWay.globallySupportedGates(), GateKind::CX));
  EXPECT_FALSE(oneWay.synthesisBasis());

  const Operation twoWayCX{
      "cnot", 2, 0,
      std::vector{OperationLocus{{0, 1}}, OperationLocus{{1, 0}},
                  OperationLocus{{1, 2}}, OperationLocus{{2, 1}}}};
  const Target twoWay{3, chain, std::vector{globalU, twoWayCX}};
  EXPECT_TRUE(
      llvm::is_contained(twoWay.globallySupportedGates(), GateKind::CX));
  ASSERT_TRUE(twoWay.synthesisBasis());
  EXPECT_EQ(twoWay.synthesisBasis()->entangler, GateKind::CX);
}

TEST(CompilerTargetTest, ClassifiesEveryEntanglerOrientation) {
  using Entangler = std::tuple<GateKind, std::string_view, size_t>;
  const std::array symmetricEntanglers{
      Entangler{GateKind::CZ, "cz", 0}, Entangler{GateKind::RXX, "rxx", 1},
      Entangler{GateKind::RYY, "ryy", 1}, Entangler{GateKind::RZZ, "rzz", 1},
      Entangler{GateKind::ISWAP, "iswap", 0}};
  const std::array directionalEntanglers{Entangler{GateKind::CX, "cx", 0},
                                         Entangler{GateKind::ECR, "ecr", 0},
                                         Entangler{GateKind::RZX, "rzx", 1}};
  const std::vector<Coupling> chain{{0, 1}, {1, 2}};
  const Operation globalU{"u", 1, 3};

  for (const auto& [gate, name, numParameters] : symmetricEntanglers) {
    SCOPED_TRACE(name);
    const Operation oneOrientation{
        std::string{name}, 2, numParameters,
        std::vector{OperationLocus{{1, 0}}, OperationLocus{{1, 2}}}};
    const Target target{3, chain, std::vector{globalU, oneOrientation}};
    EXPECT_TRUE(llvm::is_contained(target.globallySupportedGates(), gate));
    EXPECT_TRUE(target.supports(gate, {1, 0}));
    EXPECT_TRUE(target.supports(gate, {0, 1}));
    EXPECT_TRUE(target.supports(gate, {1, 2}));
    EXPECT_TRUE(target.supports(gate, {2, 1}));
    ASSERT_TRUE(target.synthesisBasis());
    EXPECT_EQ(target.synthesisBasis()->entangler, gate);
  }

  for (const auto& [gate, name, numParameters] : directionalEntanglers) {
    SCOPED_TRACE(name);
    const Operation oneOrientation{
        std::string{name}, 2, numParameters,
        std::vector{OperationLocus{{0, 1}}, OperationLocus{{1, 2}}}};
    const Target oneWay{3, chain, std::vector{globalU, oneOrientation}};
    EXPECT_FALSE(llvm::is_contained(oneWay.globallySupportedGates(), gate));
    EXPECT_FALSE(oneWay.synthesisBasis());

    const Operation bothOrientations{
        std::string{name}, 2, numParameters,
        std::vector{OperationLocus{{0, 1}}, OperationLocus{{1, 0}},
                    OperationLocus{{1, 2}}, OperationLocus{{2, 1}}}};
    const Target twoWay{3, chain, std::vector{globalU, bothOrientations}};
    EXPECT_TRUE(llvm::is_contained(twoWay.globallySupportedGates(), gate));
    ASSERT_TRUE(twoWay.synthesisBasis());
    EXPECT_EQ(twoWay.synthesisBasis()->entangler, gate);
  }
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
        const auto barrierResults = builder.barrier({q0, q1});
        q0 = barrierResults[0];
        q1 = barrierResults[1];
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
  mlir::Operation* measure = nullptr;
  mlir::Operation* reset = nullptr;
  mlir::Operation* barrier = nullptr;
  mlir::Operation* gphase = nullptr;
  moduleOp->walk([&](mlir::Operation* operation) {
    if (mlir::isa<mlir::qco::XOp>(operation) && x == nullptr) {
      x = operation;
    } else if (mlir::isa<mlir::qco::CtrlOp>(operation)) {
      cx = operation;
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
  ASSERT_NE(measure, nullptr);
  ASSERT_NE(reset, nullptr);
  ASSERT_NE(barrier, nullptr);
  ASSERT_NE(gphase, nullptr);

  const Target target{
      std::vector<Site>{Site{10}, Site{20}}, std::nullopt,
      std::vector{Operation{"x", 1, 0}, Operation{"measure", 1, 0},
                  Operation{"reset", 1, 0},
                  Operation{"cnot", 2, 0,
                            std::vector{OperationLocus{{10, 20}},
                                        OperationLocus{{20, 10}}}}}};
  EXPECT_TRUE(target.supports(x, {10}));
  EXPECT_FALSE(target.supports(x, {10, 20}));
  EXPECT_TRUE(target.supports(cx, {10, 20}));
  EXPECT_TRUE(target.supports(measure, {20}));
  EXPECT_TRUE(target.supports(reset, {10}));
  EXPECT_TRUE(target.supports(barrier, {10, 20}));
  EXPECT_TRUE(target.supports(gphase, {}));
  EXPECT_FALSE(target.supports(nullptr, {10}));

  const Target closed{2, std::nullopt, std::vector<Operation>{}};
  EXPECT_TRUE(closed.supports(barrier, {0, 1}));
  EXPECT_TRUE(closed.supports(gphase, {}));
  EXPECT_FALSE(closed.supports(x, {0}));
  EXPECT_FALSE(closed.supports(measure, {0}));
}

} // namespace
} // namespace mqt::test::compiler
