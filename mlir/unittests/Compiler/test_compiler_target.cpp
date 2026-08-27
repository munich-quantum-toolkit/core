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
using Coupling = Target::Coupling;
using DurationUnit = Target::DurationUnit;
using GateKind = Target::GateKind;
using Operation = Target::Operation;
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
                         valid(SiteTuple::create({2}, 5, 0.98))};
  operations.emplace_back(
      valid(Operation::create(" PRX ", 1, 2, std::move(siteTuples), 0, 0.97)));

  const auto target = valid(Target::create(
      "device", std::move(sites),
      std::vector<Coupling>{{11, 2}, {2, 11}, {7, 2}}, std::move(operations),
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
  EXPECT_EQ(target.operations()[0].numQubits(), 1);
  EXPECT_EQ(target.operations()[0].numParameters(), 2);
  EXPECT_EQ(target.operations()[0].duration(), 0);
  EXPECT_EQ(target.operations()[0].fidelity(), 0.97);
  ASSERT_EQ(target.operations()[0].siteTuples().size(), 2);
  EXPECT_EQ(target.operations()[0].siteTuples()[0].duration(), 0);
  EXPECT_EQ(target.operations()[0].siteTuples()[0].fidelity(), 0.99);

  EXPECT_EQ(copy.sites().data(), target.sites().data());
  EXPECT_EQ(copy.couplings().data(), target.couplings().data());
  EXPECT_EQ(copy.operations().data(), target.operations().data());
}

TEST(CompilerTargetTest, ConstructsDenseUnnamedAllToAllTarget) {
  const auto target = valid(Target::create(3));
  const auto named = valid(Target::create("simulator", 2));

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
}

TEST(CompilerTargetTest, CanonicalizesConnectedTopologyAndCachesDistances) {
  std::vector sites{valid(Site::create(7)), valid(Site::create(2)),
                    valid(Site::create(11))};
  const auto target = valid(
      Target::create(std::move(sites),
                     std::vector<Coupling>{{11, 2}, {2, 11}, {7, 2}, {2, 7}}));

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
  expectInvalid(Target::create(0),
                "Compiler target must contain at least one site");
  if constexpr (sizeof(size_t) >= sizeof(uint64_t)) {
    expectInvalid(
        Target::create(std::numeric_limits<size_t>::max()),
        "Compiler target qubit count exceeds the nonnegative i64 site domain");
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
  expectInvalid(Operation::create("x", 0, 0),
                "Compiler target operation qubit count must be positive");
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

  expectInvalid(Target::create(std::vector<Site>{}),
                "Compiler target must contain at least one site");
  expectInvalid(Target::create("", 1),
                "Compiler target name must not be empty when present");
  expectInvalid(Target::create("invalid", 0),
                "Compiler target must contain at least one site");
  expectInvalid(Target::create(std::vector{valid(Site::create(1)),
                                           valid(Site::create(1))}),
                "Compiler target contains duplicate site IDs");
  expectInvalid(
      Target::create(std::vector{valid(Site::create(0, std::nullopt, 1))}),
      "Compiler target timing metadata requires a duration unit");
  expectInvalid(
      Target::create(1, std::nullopt,
                     std::vector{valid(Operation::create("x", 1, 0, {}, 1))}),
      "Compiler target timing metadata requires a duration unit");
  expectInvalid(
      Target::create(
          1, std::nullopt,
          std::vector{valid(Operation::create(
              "x", 1, 0, std::vector{valid(SiteTuple::create({0}, 1))}))}),
      "Compiler target timing metadata requires a duration unit");
  expectInvalid(Target::create(2, std::vector<Coupling>{{0, 0}}),
                "Compiler target topology contains a self-coupling");
  expectInvalid(Target::create(2, std::vector<Coupling>{{0, 2}}),
                "Compiler target topology references an unknown site");
  expectInvalid(Target::create(3, std::vector<Coupling>{{0, 1}}),
                "Compiler target topology must be connected");
  expectInvalid(
      Target::create(
          2, std::nullopt,
          std::vector{valid(Operation::create(
              "x", 1, 0, std::vector{valid(SiteTuple::create({2}))}))}),
      "Compiler target operation site tuple references an unknown site");
  expectInvalid(
      Target::create(1, std::nullopt,
                     std::vector{valid(Operation::create("cx", 2, 0))}),
      "Compiler target operation arity exceeds its site count");
}

TEST(CompilerTargetTest, DistinguishesAbsentAndEmptyOperationSets) {
  const auto permissive = valid(Target::create(2));
  const auto closed =
      valid(Target::create(2, std::nullopt, std::vector<Operation>{}));

  EXPECT_FALSE(permissive.hasExplicitOperations());
  EXPECT_TRUE(permissive.operations().empty());
  EXPECT_TRUE(permissive.supportsOperation("device.operation", 1));
  EXPECT_TRUE(permissive.supports(GateKind::CX));
  EXPECT_FALSE(permissive.supportsOperation("", 1));
  EXPECT_FALSE(permissive.supportsOperation("   ", 1));
  EXPECT_FALSE(permissive.supportsOperation("x", 0));
  EXPECT_FALSE(permissive.supportsOperation("x", 3));

  EXPECT_TRUE(closed.hasExplicitOperations());
  EXPECT_TRUE(closed.operations().empty());
  EXPECT_FALSE(closed.supportsOperation("x", 1));
  EXPECT_FALSE(closed.supports(GateKind::CX));
  EXPECT_TRUE(closed.supportedGates().empty());
  EXPECT_FALSE(closed.synthesisBasis());
}

TEST(CompilerTargetTest, PreservesCalibrationAndResolvesHomogeneousBasis) {
  const std::vector<Coupling> chain{{0, 1}, {1, 2}};
  const auto globalU = valid(Operation::create("U3", 1, 3));
  const auto cz = valid(Operation::create(
      "cz", 2, 0, std::vector{valid(SiteTuple::create({1, 0}, 5, 0.99))}));
  const auto target =
      valid(Target::create(3, chain, std::vector{globalU, cz},
                           valid(DurationUnit::create("ns", 1.))));

  EXPECT_TRUE(target.supportsOperation("u", 1, 3));
  EXPECT_TRUE(target.supportsOperation(" U3 ", 1, 3));
  EXPECT_TRUE(target.supports(GateKind::CZ));
  EXPECT_TRUE(llvm::is_contained(target.supportedGates(), GateKind::CZ));
  ASSERT_EQ(target.operations().size(), 2U);
  ASSERT_EQ(target.operations()[1].siteTuples().size(), 1U);
  EXPECT_EQ(target.operations()[1].siteTuples()[0].sites(),
            (llvm::ArrayRef<SiteId>{1, 0}));
  EXPECT_EQ(target.operations()[1].siteTuples()[0].duration(), 5);
  EXPECT_EQ(target.operations()[1].siteTuples()[0].fidelity(), 0.99);
  ASSERT_TRUE(target.synthesisBasis());
  EXPECT_EQ(target.synthesisBasis()->singleQubit, Target::SingleQubitBasis::U);
  EXPECT_EQ(target.synthesisBasis()->entangler, GateKind::CZ);
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
    const auto target =
        valid(Target::create(3, chain, std::vector{globalU, operation}));
    EXPECT_TRUE(llvm::is_contained(target.supportedGates(), gate));
    EXPECT_TRUE(target.supports(gate));
    ASSERT_TRUE(target.synthesisBasis());
    EXPECT_EQ(target.synthesisBasis()->entangler, gate);
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
        auto barrierResults = builder.barrier({q0, q1});
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

  std::vector sites{valid(Site::create(10)), valid(Site::create(20))};
  std::vector directionalTuples{valid(SiteTuple::create({10, 20})),
                                valid(SiteTuple::create({20, 10}))};
  std::vector operations{
      valid(Operation::create("x", 1, 0)),
      valid(Operation::create("measure", 1, 0)),
      valid(Operation::create("reset", 1, 0)),
      valid(Operation::create("cnot", 2, 0, std::move(directionalTuples)))};
  const auto target = valid(
      Target::create(std::move(sites), std::nullopt, std::move(operations)));
  EXPECT_TRUE(target.supports(x));
  EXPECT_TRUE(target.supports(cx));
  EXPECT_TRUE(target.supports(measure));
  EXPECT_TRUE(target.supports(reset));
  EXPECT_TRUE(target.supports(barrier));
  EXPECT_TRUE(target.supports(gphase));
  EXPECT_FALSE(target.supports(nullptr));

  const auto closed =
      valid(Target::create(2, std::nullopt, std::vector<Operation>{}));
  EXPECT_TRUE(closed.supports(barrier));
  EXPECT_TRUE(closed.supports(gphase));
  EXPECT_FALSE(closed.supports(x));
  EXPECT_FALSE(closed.supports(measure));
}

} // namespace
} // namespace mqt::test::compiler
