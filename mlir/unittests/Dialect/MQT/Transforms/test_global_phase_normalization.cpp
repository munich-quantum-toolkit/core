/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ExactUnitaryTest.h"
#include "mlir/Conversion/QCToQCO/QCToQCO.h"
#include "mlir/Dialect/MQT/Transforms/GlobalPhaseNormalization.h"
#include "mlir/Dialect/MQT/Utils/Angles.h"
#include "mlir/Dialect/MQT/Utils/ConstantFolding.h"
#include "mlir/Dialect/MQT/Utils/Parameters.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"

#include <gtest/gtest.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numbers>
#include <string>

using namespace mlir;

namespace {

class GlobalPhaseNormalizationTest : public testing::Test {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    DialectRegistry registry;
    registry.insert<arith::ArithDialect, cf::ControlFlowDialect,
                    func::FuncDialect, memref::MemRefDialect,
                    mlir::qc::QCDialect, qco::QCODialect, scf::SCFDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  [[nodiscard]] OwningOpRef<ModuleOp> parse(const StringRef source) const {
    return parseSourceString<ModuleOp>(source, context.get());
  }

  static void expectFoldableGlobalPhase(Value angle,
                                        const double expectedAngle) {
    const auto value = mlir::mqt::valueToConstantDouble(angle);
    ASSERT_TRUE(value.has_value());
    EXPECT_TRUE(mlir::mqt::isValidGlobalPhaseAngle(*value));
    EXPECT_NEAR(mlir::mqt::normalizeAngle(*value - expectedAngle), 0.0,
                mlir::mqt::PARAMETER_COMPARISON_TOLERANCE);
  }

  static void expectNormalizedUnitary(OwningOpRef<ModuleOp>& moduleOp,
                                      const std::size_t numQubits) {
    auto cloned = cast<ModuleOp>((*moduleOp)->clone());
    OwningOpRef<ModuleOp> expected(cloned);
    ASSERT_TRUE(mlir::mqt::normalizeGlobalPhases(*moduleOp).succeeded());
    ASSERT_TRUE(verify(*moduleOp).succeeded());
    ::mqt::test::expectFullUnitaryEqual(*expected, *moduleOp, numQubits);
  }

  static void expectNormalizedQCUnitary(OwningOpRef<ModuleOp>& moduleOp,
                                        const std::size_t numQubits) {
    auto cloned = cast<ModuleOp>((*moduleOp)->clone());
    OwningOpRef<ModuleOp> expected(cloned);
    ASSERT_TRUE(mlir::mqt::normalizeGlobalPhases(*moduleOp).succeeded());

    for (ModuleOp candidate : {expected.get(), moduleOp.get()}) {
      PassManager pm(candidate.getContext());
      pm.addPass(createQCToQCO());
      ASSERT_TRUE(pm.run(candidate).succeeded());
      ASSERT_TRUE(verify(candidate).succeeded());
    }
    ::mqt::test::expectFullUnitaryEqual(*expected, *moduleOp, numQubits);
  }
};

} // namespace

TEST_F(GlobalPhaseNormalizationTest, CombinesQCOConstantsAtBlockExit) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @test(%q: !qco.qubit) -> !qco.qubit {
        %c0 = arith.constant 0.25 : f64
        %c1 = arith.constant 0.5 : f64
        qco.gphase(%c0)
        %q1 = qco.x %q : !qco.qubit -> !qco.qubit
        qco.gphase(%c1)
        return %q1 : !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(mlir::mqt::normalizeGlobalPhases(*moduleOp).succeeded());

  auto func = cast<func::FuncOp>(moduleOp->getBody()->front());
  auto phases = llvm::to_vector(func.getBody().getOps<qco::GPhaseOp>());
  ASSERT_EQ(phases.size(), 1);
  EXPECT_EQ(phases.front()->getNextNode(),
            func.getBody().front().getTerminator());
  expectFoldableGlobalPhase(phases.front().getTheta(), 0.75);
}

TEST_F(GlobalPhaseNormalizationTest,
       FoldsMulDerivedPhasesWithinPracticalAngleLimit) {
  // Many arith.mulf-derived gphase angles used to be treated as dynamic and
  // merged into an addf chain whose later constant-fold exceeded the 1e4 rad
  // GPhase verifier contract (seen on QASMBench vqe_uccsd_n28 / QV_n100).
  OwningOpRef moduleOp = ModuleOp::create(UnknownLoc::get(context.get()));
  OpBuilder builder(context.get());
  builder.setInsertionPointToStart(moduleOp->getBody());
  const auto loc = moduleOp->getLoc();
  auto function = func::FuncOp::create(builder, loc, "test",
                                       builder.getFunctionType({}, {}));
  auto* entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  constexpr int phaseCount = 4000;
  constexpr double half = 1.5;
  constexpr double two = 2.0;
  const double expected =
      mlir::mqt::normalizeAngle(static_cast<double>(phaseCount) * half * two);
  for (int i = 0; i < phaseCount; ++i) {
    auto lhs = mlir::mqt::constantFromScalar(builder, loc, half);
    auto rhs = mlir::mqt::constantFromScalar(builder, loc, two);
    auto angle = arith::MulFOp::create(builder, loc, lhs, rhs);
    qco::GPhaseOp::create(builder, loc, angle.getResult());
  }
  func::ReturnOp::create(builder, loc);

  ASSERT_TRUE(mlir::mqt::normalizeGlobalPhases(*moduleOp).succeeded());
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  auto phases = llvm::to_vector(function.getBody().getOps<qco::GPhaseOp>());
  ASSERT_EQ(phases.size(), 1);
  auto constantOp =
      phases.front().getTheta().getDefiningOp<arith::ConstantOp>();
  ASSERT_TRUE(constantOp);
  const auto value = dyn_cast<FloatAttr>(constantOp.getValue());
  ASSERT_TRUE(value);
  expectFoldableGlobalPhase(phases.front().getTheta(), expected);
}

TEST_F(GlobalPhaseNormalizationTest,
       FoldsSitofpMulDerivedPhasesWithinPracticalAngleLimit) {
  // QV_n100 lowers many phases as mulf(sitofp(i64), f64), which must fold the
  // same way as pure float mulf trees.
  OwningOpRef moduleOp = ModuleOp::create(UnknownLoc::get(context.get()));
  OpBuilder builder(context.get());
  builder.setInsertionPointToStart(moduleOp->getBody());
  const auto loc = moduleOp->getLoc();
  auto function = func::FuncOp::create(builder, loc, "test",
                                       builder.getFunctionType({}, {}));
  auto* entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  constexpr int phaseCount = 4000;
  constexpr int64_t intFactor = 3;
  constexpr double floatFactor = 2.0;
  const double expected =
      mlir::mqt::normalizeAngle(static_cast<double>(phaseCount) *
                                static_cast<double>(intFactor) * floatFactor);
  for (int i = 0; i < phaseCount; ++i) {
    auto intConst = arith::ConstantOp::create(
        builder, loc, builder.getIntegerAttr(builder.getI64Type(), intFactor));
    auto lhs = arith::SIToFPOp::create(builder, loc, builder.getF64Type(),
                                       intConst.getResult());
    auto rhs = mlir::mqt::constantFromScalar(builder, loc, floatFactor);
    auto angle = arith::MulFOp::create(builder, loc, lhs.getResult(), rhs);
    qco::GPhaseOp::create(builder, loc, angle.getResult());
  }
  func::ReturnOp::create(builder, loc);

  ASSERT_TRUE(mlir::mqt::normalizeGlobalPhases(*moduleOp).succeeded());
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  auto phases = llvm::to_vector(function.getBody().getOps<qco::GPhaseOp>());
  ASSERT_EQ(phases.size(), 1);
  auto constantOp =
      phases.front().getTheta().getDefiningOp<arith::ConstantOp>();
  ASSERT_TRUE(constantOp);
  const auto value = dyn_cast<FloatAttr>(constantOp.getValue());
  ASSERT_TRUE(value);
  expectFoldableGlobalPhase(phases.front().getTheta(), expected);
}

TEST_F(GlobalPhaseNormalizationTest,
       QCControlledExtractionPreservesFullUnitaryUnderOuterControl) {
  auto moduleOp = mlir::qc::QCProgramBuilder::build(
      context.get(), [](mlir::qc::QCProgramBuilder& builder) {
        auto outer = builder.staticQubit(0);
        auto inner = builder.staticQubit(1);
        auto target = builder.staticQubit(2);
        builder.ctrl(outer, {inner, target}, [&](ValueRange outerTargets) {
          builder.ctrl(outerTargets[0], outerTargets[1],
                       [&](Value innerTarget) {
                         builder.x(innerTarget);
                         builder.gphase(0.731);
                       });
        });
        return builder.intConstant(0);
      });
  ASSERT_TRUE(moduleOp);
  expectNormalizedQCUnitary(moduleOp, 3);
}

TEST_F(GlobalPhaseNormalizationTest,
       QCInverseAndIntegralPowerPreserveFullUnitary) {
  auto moduleOp = mlir::qc::QCProgramBuilder::build(
      context.get(), [](mlir::qc::QCProgramBuilder& builder) {
        auto q0 = builder.staticQubit(0);
        auto q1 = builder.staticQubit(1);
        builder.inv(q0, [&](Value target) {
          builder.h(target);
          builder.gphase(0.371);
        });
        builder.pow(3.0, q1, [&](Value target) {
          builder.y(target);
          builder.gphase(-0.417);
        });
        return builder.intConstant(0);
      });
  ASSERT_TRUE(moduleOp);
  expectNormalizedQCUnitary(moduleOp, 2);
}

TEST_F(GlobalPhaseNormalizationTest,
       PreservesDynamicDependenciesAcrossRepeatedRuns) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @test(%q: !qc.qubit, %a: f64, %b: f64) {
        qc.gphase(%a)
        qc.x %q : !qc.qubit
        qc.gphase(%b)
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(mlir::mqt::normalizeGlobalPhases(*moduleOp).succeeded());

  auto func = cast<func::FuncOp>(moduleOp->getBody()->front());
  auto phases = llvm::to_vector(func.getBody().getOps<mlir::qc::GPhaseOp>());
  ASSERT_EQ(phases.size(), 1);
  const auto dependsOn = [](Value value, Value input) {
    llvm::SmallVector<Value> worklist{value};
    while (!worklist.empty()) {
      auto current = worklist.pop_back_val();
      if (current == input) {
        return true;
      }
      if (auto* definingOp = current.getDefiningOp()) {
        llvm::append_range(worklist, definingOp->getOperands());
      }
    }
    return false;
  };
  EXPECT_TRUE(dependsOn(phases.front().getTheta(), func.getArgument(1)));
  EXPECT_TRUE(dependsOn(phases.front().getTheta(), func.getArgument(2)));

  const auto countOperations = [&moduleOp]() {
    size_t count = 0;
    moduleOp->walk([&count](Operation*) { ++count; });
    return count;
  };
  const auto firstRunOperationCount = countOperations();

  ASSERT_TRUE(mlir::mqt::normalizeGlobalPhases(*moduleOp).succeeded());
  ASSERT_TRUE(verify(*moduleOp).succeeded());

  phases = llvm::to_vector(func.getBody().getOps<mlir::qc::GPhaseOp>());
  ASSERT_EQ(phases.size(), 1);
  EXPECT_TRUE(dependsOn(phases.front().getTheta(), func.getArgument(1)));
  EXPECT_TRUE(dependsOn(phases.front().getTheta(), func.getArgument(2)));
  EXPECT_LE(countOperations(), firstRunOperationCount);
}

TEST_F(GlobalPhaseNormalizationTest, KeepsSCFStyleRegionsIndependent) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @test(%q: !qco.qubit, %condition: i1) -> !qco.qubit {
        %result = qco.if %condition args(%arg = %q) -> (!qco.qubit) {
          %c0 = arith.constant 0.25 : f64
          qco.gphase(%c0)
          %c1 = arith.constant 0.5 : f64
          qco.gphase(%c1)
          qco.yield %arg : !qco.qubit
        } else args(%arg = %q) {
          %c2 = arith.constant 1.0 : f64
          qco.gphase(%c2)
          qco.yield %arg : !qco.qubit
        }
        return %result : !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(mlir::mqt::normalizeGlobalPhases(*moduleOp).succeeded());

  auto func = *moduleOp->getOps<func::FuncOp>().begin();
  auto ifOp = *func.getBody().getOps<qco::IfOp>().begin();
  EXPECT_EQ(llvm::range_size(ifOp.getThenRegion().getOps<qco::GPhaseOp>()), 1);
  EXPECT_EQ(llvm::range_size(ifOp.getElseRegion().getOps<qco::GPhaseOp>()), 1);
  EXPECT_TRUE(func.getBody().getOps<qco::GPhaseOp>().empty());
}

TEST_F(GlobalPhaseNormalizationTest, FactorsInverseAndIntegralPower) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit)
          -> (!qco.qubit, !qco.qubit) {
        %c0 = arith.constant 0.25 : f64
        %i = qco.inv (%arg0 = %q0) {
          %x = qco.x %arg0 : !qco.qubit -> !qco.qubit
          qco.gphase(%c0)
          qco.yield %x : !qco.qubit
        } : {!qco.qubit} -> {!qco.qubit}
        %two = arith.constant 2.0 : f64
        %p = qco.pow(%two) (%arg1 = %q1) {
          %x = qco.x %arg1 : !qco.qubit -> !qco.qubit
          qco.gphase(%c0)
          qco.yield %x : !qco.qubit
        } : {!qco.qubit} -> {!qco.qubit}
        return %i, %p : !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(mlir::mqt::normalizeGlobalPhases(*moduleOp).succeeded());

  auto func = *moduleOp->getOps<func::FuncOp>().begin();
  auto inv = *func.getBody().getOps<qco::InvOp>().begin();
  auto pow = *func.getBody().getOps<qco::PowOp>().begin();
  EXPECT_TRUE(inv.getBody()->getOps<qco::GPhaseOp>().empty());
  EXPECT_TRUE(pow.getBody()->getOps<qco::GPhaseOp>().empty());
  EXPECT_EQ(llvm::range_size(func.getBody().getOps<qco::GPhaseOp>()), 1);
}

TEST_F(GlobalPhaseNormalizationTest, FractionalPowerRemainsBoundary) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @test(%q: !qco.qubit) -> !qco.qubit {
        %half = arith.constant 0.5 : f64
        %phase = arith.constant 4.71238898038469 : f64
        %p = qco.pow(%half) (%arg = %q) {
          %x = qco.x %arg : !qco.qubit -> !qco.qubit
          qco.gphase(%phase)
          qco.yield %x : !qco.qubit
        } : {!qco.qubit} -> {!qco.qubit}
        return %p : !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(mlir::mqt::normalizeGlobalPhases(*moduleOp).succeeded());

  auto func = *moduleOp->getOps<func::FuncOp>().begin();
  auto pow = *func.getBody().getOps<qco::PowOp>().begin();
  EXPECT_EQ(llvm::range_size(pow.getBody()->getOps<qco::GPhaseOp>()), 1);
}

TEST_F(GlobalPhaseNormalizationTest, DynamicPowerRemainsBoundary) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @test(%q: !qco.qubit, %exponent: f64) -> !qco.qubit {
        %phase = arith.constant 0.371 : f64
        %p = qco.pow(%exponent) (%arg = %q) {
          %x = qco.x %arg : !qco.qubit -> !qco.qubit
          qco.gphase(%phase)
          qco.yield %x : !qco.qubit
        } : {!qco.qubit} -> {!qco.qubit}
        return %p : !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(mlir::mqt::normalizeGlobalPhases(*moduleOp).succeeded());

  auto func = *moduleOp->getOps<func::FuncOp>().begin();
  auto pow = *func.getBody().getOps<qco::PowOp>().begin();
  EXPECT_EQ(llvm::range_size(pow.getBody()->getOps<qco::GPhaseOp>()), 1);
  EXPECT_TRUE(func.getBody().getOps<qco::GPhaseOp>().empty());
}

TEST_F(GlobalPhaseNormalizationTest, FactorsControlledPhaseOntoControl) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @test(%control: !qco.qubit, %target: !qco.qubit)
          -> (!qco.qubit, !qco.qubit) {
        %phase = arith.constant 0.25 : f64
        %control_out, %target_out = qco.ctrl(%control)
            targets(%arg = %target) {
          %x = qco.x %arg : !qco.qubit -> !qco.qubit
          qco.gphase(%phase)
          qco.yield %x : !qco.qubit
        } : ({!qco.qubit}, {!qco.qubit})
          -> ({!qco.qubit}, {!qco.qubit})
        return %control_out, %target_out : !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(mlir::mqt::normalizeGlobalPhases(*moduleOp).succeeded());

  auto func = *moduleOp->getOps<func::FuncOp>().begin();
  auto ctrl = *func.getBody().getOps<qco::CtrlOp>().begin();
  EXPECT_TRUE(ctrl.getBody()->getOps<qco::GPhaseOp>().empty());
  ASSERT_EQ(llvm::range_size(func.getBody().getOps<qco::POp>()), 1);
  auto returnOp = cast<func::ReturnOp>(func.getBody().front().getTerminator());
  auto p = *func.getBody().getOps<qco::POp>().begin();
  EXPECT_EQ(returnOp.getOperand(0), p.getOutputTarget(0));
  EXPECT_EQ(returnOp.getOperand(1), ctrl.getOutputTarget(0));
}

TEST_F(GlobalPhaseNormalizationTest,
       ControlledExtractionPreservesFullUnitaryUnderOuterControl) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @test(%outer: !qco.qubit, %inner: !qco.qubit,
                      %target: !qco.qubit)
          -> (!qco.qubit, !qco.qubit, !qco.qubit) {
        %phase = arith.constant 0.731 : f64
        %outer_out, %inner_out, %target_out = qco.ctrl(%outer)
            targets(%inner_arg = %inner, %target_arg = %target) {
          %inner_control_out, %inner_target_out = qco.ctrl(%inner_arg)
              targets(%arg = %target_arg) {
            %x = qco.x %arg : !qco.qubit -> !qco.qubit
            qco.gphase(%phase)
            qco.yield %x : !qco.qubit
          } : ({!qco.qubit}, {!qco.qubit})
            -> ({!qco.qubit}, {!qco.qubit})
          qco.yield %inner_control_out, %inner_target_out
              : !qco.qubit, !qco.qubit
        } : ({!qco.qubit}, {!qco.qubit, !qco.qubit})
          -> ({!qco.qubit}, {!qco.qubit, !qco.qubit})
        return %outer_out, %inner_out, %target_out
            : !qco.qubit, !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  expectNormalizedUnitary(moduleOp, 3);
}

TEST_F(GlobalPhaseNormalizationTest, ThreeControlsPreserveFullUnitary) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit,
                      %q2: !qco.qubit, %target: !qco.qubit)
          -> (!qco.qubit, !qco.qubit, !qco.qubit, !qco.qubit) {
        %phase = arith.constant -1.137 : f64
        %q0_out, %q1_out, %q2_out, %target_out =
            qco.ctrl(%q0, %q1, %q2) targets(%arg = %target) {
          %h = qco.h %arg : !qco.qubit -> !qco.qubit
          %x = qco.x %h : !qco.qubit -> !qco.qubit
          qco.gphase(%phase)
          qco.yield %x : !qco.qubit
        } : ({!qco.qubit, !qco.qubit, !qco.qubit}, {!qco.qubit})
          -> ({!qco.qubit, !qco.qubit, !qco.qubit}, {!qco.qubit})
        return %q0_out, %q1_out, %q2_out, %target_out
            : !qco.qubit, !qco.qubit, !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  expectNormalizedUnitary(moduleOp, 4);

  auto func = *moduleOp->getOps<func::FuncOp>().begin();
  auto controls = llvm::to_vector(func.getBody().getOps<qco::CtrlOp>());
  ASSERT_EQ(controls.size(), 2);
  EXPECT_EQ(controls.back().getNumControls(), 2);
  EXPECT_EQ(controls.back().getNumTargets(), 1);
}

TEST_F(GlobalPhaseNormalizationTest, ReorderedQCOControlsThreadCorrectResults) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit,
                      %q2: !qco.qubit, %target: !qco.qubit)
          -> (!qco.qubit, !qco.qubit, !qco.qubit, !qco.qubit) {
        %phase = arith.constant -1.137 : f64
        %q2_out, %q0_out, %q1_out, %target_out =
            qco.ctrl(%q2, %q0, %q1) targets(%arg = %target) {
          %x = qco.x %arg : !qco.qubit -> !qco.qubit
          qco.gphase(%phase)
          qco.yield %x : !qco.qubit
        } : ({!qco.qubit, !qco.qubit, !qco.qubit}, {!qco.qubit})
          -> ({!qco.qubit, !qco.qubit, !qco.qubit}, {!qco.qubit})
        return %q0_out, %q1_out, %q2_out, %target_out
            : !qco.qubit, !qco.qubit, !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto cloned = cast<ModuleOp>((*moduleOp)->clone());
  OwningOpRef<ModuleOp> expected(cloned);
  ASSERT_TRUE(mlir::mqt::normalizeGlobalPhases(*moduleOp).succeeded());
  ASSERT_TRUE(verify(*moduleOp).succeeded());

  auto func = *moduleOp->getOps<func::FuncOp>().begin();
  auto controls = llvm::to_vector(func.getBody().getOps<qco::CtrlOp>());
  ASSERT_EQ(controls.size(), 2);
  auto returnOp = cast<func::ReturnOp>(func.getBody().front().getTerminator());
  ::mqt::test::expectFullUnitaryEqual(*expected, *moduleOp, 4);
  EXPECT_EQ(returnOp.getOperand(3), controls[0].getOutputTarget(0));
}

TEST_F(GlobalPhaseNormalizationTest, MultipleTargetsPreserveFullUnitary) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @test(%c0: !qco.qubit, %c1: !qco.qubit,
                      %t0: !qco.qubit, %t1: !qco.qubit)
          -> (!qco.qubit, !qco.qubit, !qco.qubit, !qco.qubit) {
        %phase = arith.constant 2.173 : f64
        %c0_out, %c1_out, %t0_out, %t1_out =
            qco.ctrl(%c0, %c1) targets(%a = %t0, %b = %t1) {
          %x = qco.x %a : !qco.qubit -> !qco.qubit
          %h = qco.h %b : !qco.qubit -> !qco.qubit
          qco.gphase(%phase)
          qco.yield %x, %h : !qco.qubit, !qco.qubit
        } : ({!qco.qubit, !qco.qubit}, {!qco.qubit, !qco.qubit})
          -> ({!qco.qubit, !qco.qubit}, {!qco.qubit, !qco.qubit})
        return %c0_out, %c1_out, %t0_out, %t1_out
            : !qco.qubit, !qco.qubit, !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  expectNormalizedUnitary(moduleOp, 4);
}

TEST_F(GlobalPhaseNormalizationTest,
       IntegralPowersPreserveFullUnitaryAndReleasePhase) {
  for (const std::string exponent :
       {"-3.0", "-1.0", "0.0", "1.0", "2.0", "3.0"}) {
    const std::string source =
        R"mlir(module {
          func.func @test(%q: !qco.qubit) -> !qco.qubit {
            %exponent = arith.constant )mlir" +
        exponent + R"mlir( : f64
            %phase = arith.constant 0.371 : f64
            %out = qco.pow(%exponent) (%arg = %q) {
              %h = qco.h %arg : !qco.qubit -> !qco.qubit
              qco.gphase(%phase)
              qco.yield %h : !qco.qubit
            } : {!qco.qubit} -> {!qco.qubit}
            return %out : !qco.qubit
          }
        })mlir";
    auto moduleOp = parse(source);
    ASSERT_TRUE(moduleOp) << exponent;
    expectNormalizedUnitary(moduleOp, 1);
    auto func = *moduleOp->getOps<func::FuncOp>().begin();
    auto pow = *func.getBody().getOps<qco::PowOp>().begin();
    EXPECT_TRUE(pow.getBody()->getOps<qco::GPhaseOp>().empty()) << exponent;
  }
}

TEST_F(GlobalPhaseNormalizationTest,
       NestedInverseAndPowerOrdersPreserveFullUnitary) {
  constexpr StringLiteral source = R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit)
          -> (!qco.qubit, !qco.qubit) {
        %phase = arith.constant 0.371 : f64
        %two = arith.constant 2.0 : f64
        %a = qco.inv (%outer_arg = %q0) {
          %inner = qco.pow(%two) (%inner_arg = %outer_arg) {
            %h = qco.h %inner_arg : !qco.qubit -> !qco.qubit
            qco.gphase(%phase)
            qco.yield %h : !qco.qubit
          } : {!qco.qubit} -> {!qco.qubit}
          qco.yield %inner : !qco.qubit
        } : {!qco.qubit} -> {!qco.qubit}
        %b = qco.pow(%two) (%outer_arg = %q1) {
          %inner = qco.inv (%inner_arg = %outer_arg) {
            %x = qco.x %inner_arg : !qco.qubit -> !qco.qubit
            qco.gphase(%phase)
            qco.yield %x : !qco.qubit
          } : {!qco.qubit} -> {!qco.qubit}
          qco.yield %inner : !qco.qubit
        } : {!qco.qubit} -> {!qco.qubit}
        return %a, %b : !qco.qubit, !qco.qubit
      }
    }
  )mlir";
  auto moduleOp = parse(source);
  ASSERT_TRUE(moduleOp);
  expectNormalizedUnitary(moduleOp, 2);

  moduleOp->walk([&](qco::InvOp inv) {
    EXPECT_TRUE(inv.getBody()->getOps<qco::GPhaseOp>().empty());
  });
  moduleOp->walk([&](qco::PowOp pow) {
    EXPECT_TRUE(pow.getBody()->getOps<qco::GPhaseOp>().empty());
  });
}

TEST_F(GlobalPhaseNormalizationTest, ZeroControlsReleaseAnUnchangedPhase) {
  OwningOpRef moduleOp = ModuleOp::create(UnknownLoc::get(context.get()));
  OpBuilder builder(context.get());
  builder.setInsertionPointToStart(moduleOp->getBody());
  const auto loc = moduleOp->getLoc();
  const auto qubitType = qco::QubitType::get(context.get());
  auto function = func::FuncOp::create(
      builder, loc, "test", builder.getFunctionType({qubitType}, {qubitType}));
  auto* entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  auto phase = mlir::mqt::constantFromScalar(builder, loc, 0.417);
  auto ctrl = qco::CtrlOp::create(
      builder, loc, ValueRange{}, entry->getArgument(0), [&](Value target) {
        auto out = qco::XOp::create(builder, loc, target).getQubitOut();
        qco::GPhaseOp::create(builder, loc, phase);
        return out;
      });
  func::ReturnOp::create(builder, loc, ctrl.getOutputTarget(0));
  expectNormalizedUnitary(moduleOp, 1);

  EXPECT_TRUE(ctrl.getBody()->getOps<qco::GPhaseOp>().empty());
  EXPECT_EQ(llvm::range_size(function.getBody().getOps<qco::GPhaseOp>()), 1);
}

TEST_F(GlobalPhaseNormalizationTest,
       MemoryDependentAngleRemainsInsideModifier) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @test(%q: !qco.qubit, %angles: memref<1xf64>)
          -> !qco.qubit {
        %c0 = arith.constant 0 : index
        %out = qco.inv (%arg = %q) {
          %phase = memref.load %angles[%c0] : memref<1xf64>
          %x = qco.x %arg : !qco.qubit -> !qco.qubit
          qco.gphase(%phase)
          qco.yield %x : !qco.qubit
        } : {!qco.qubit} -> {!qco.qubit}
        return %out : !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(mlir::mqt::normalizeGlobalPhases(*moduleOp).succeeded());
  ASSERT_TRUE(verify(*moduleOp).succeeded());

  auto func = *moduleOp->getOps<func::FuncOp>().begin();
  auto inv = *func.getBody().getOps<qco::InvOp>().begin();
  EXPECT_EQ(llvm::range_size(inv.getBody()->getOps<qco::GPhaseOp>()), 1);
  EXPECT_TRUE(func.getBody().getOps<qco::GPhaseOp>().empty());
}

TEST_F(GlobalPhaseNormalizationTest, CFGBlocksRemainIndependentScopes) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @test(%condition: i1) {
        cf.cond_br %condition, ^then, ^else
      ^then:
        %a = arith.constant 0.25 : f64
        qco.gphase(%a)
        %b = arith.constant 0.5 : f64
        qco.gphase(%b)
        cf.br ^exit
      ^else:
        %c = arith.constant 1.0 : f64
        qco.gphase(%c)
        cf.br ^exit
      ^exit:
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(mlir::mqt::normalizeGlobalPhases(*moduleOp).succeeded());
  ASSERT_TRUE(verify(*moduleOp).succeeded());

  auto func = *moduleOp->getOps<func::FuncOp>().begin();
  SmallVector<Block*> blocks;
  for (auto& block : func.getBlocks()) {
    blocks.push_back(&block);
  }
  ASSERT_EQ(blocks.size(), 4);
  EXPECT_EQ(llvm::range_size(blocks[1]->getOps<qco::GPhaseOp>()), 1);
  EXPECT_EQ(llvm::range_size(blocks[2]->getOps<qco::GPhaseOp>()), 1);
  EXPECT_TRUE(blocks[0]->getOps<qco::GPhaseOp>().empty());
  EXPECT_TRUE(blocks[3]->getOps<qco::GPhaseOp>().empty());
}

TEST_F(GlobalPhaseNormalizationTest, FunctionsRemainIndependentScopes) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @first() {
        %a = arith.constant 0.25 : f64
        qco.gphase(%a)
        %b = arith.constant 0.5 : f64
        qco.gphase(%b)
        return
      }
      func.func @second() {
        %a = arith.constant -0.25 : f64
        qco.gphase(%a)
        %b = arith.constant -0.5 : f64
        qco.gphase(%b)
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(mlir::mqt::normalizeGlobalPhases(*moduleOp).succeeded());

  for (auto func : moduleOp->getOps<func::FuncOp>()) {
    EXPECT_EQ(llvm::range_size(func.getBody().getOps<qco::GPhaseOp>()), 1);
  }
  EXPECT_TRUE(moduleOp->getBody()->getOps<qco::GPhaseOp>().empty());
}

TEST_F(GlobalPhaseNormalizationTest,
       IndexSwitchRegionsRemainIndependentScopes) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @test(%index: index, %q: !qco.qubit) -> !qco.qubit {
        %out = qco.index_switch %index -> !qco.qubit
        case 0 args(%arg = %q) {
          %a = arith.constant 0.25 : f64
          qco.gphase(%a)
          %b = arith.constant 0.5 : f64
          qco.gphase(%b)
          qco.yield %arg : !qco.qubit
        }
        default args(%arg = %q) {
          %a = arith.constant -0.75 : f64
          qco.gphase(%a)
          qco.yield %arg : !qco.qubit
        }
        return %out : !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(mlir::mqt::normalizeGlobalPhases(*moduleOp).succeeded());
  ASSERT_TRUE(verify(*moduleOp).succeeded());

  auto func = *moduleOp->getOps<func::FuncOp>().begin();
  auto switchOp = *func.getBody().getOps<qco::IndexSwitchOp>().begin();
  for (auto& region : switchOp->getRegions()) {
    EXPECT_EQ(llvm::range_size(region.getOps<qco::GPhaseOp>()), 1);
  }
  EXPECT_TRUE(func.getBody().getOps<qco::GPhaseOp>().empty());
}

TEST_F(GlobalPhaseNormalizationTest, SCFLoopRegionRemainsAnIndependentScope) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @test() {
        %lb = arith.constant 0 : index
        %ub = arith.constant 4 : index
        %step = arith.constant 1 : index
        scf.for %i = %lb to %ub step %step {
          %a = arith.constant 0.25 : f64
          qco.gphase(%a)
          %b = arith.constant 0.5 : f64
          qco.gphase(%b)
        }
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(mlir::mqt::normalizeGlobalPhases(*moduleOp).succeeded());
  ASSERT_TRUE(verify(*moduleOp).succeeded());

  auto func = *moduleOp->getOps<func::FuncOp>().begin();
  auto loop = *func.getBody().getOps<scf::ForOp>().begin();
  EXPECT_EQ(llvm::range_size(loop.getBody()->getOps<qco::GPhaseOp>()), 1);
  EXPECT_TRUE(func.getBody().getOps<qco::GPhaseOp>().empty());
}

TEST_F(GlobalPhaseNormalizationTest,
       ExactSpecialConstantsCancelWithoutTolerance) {
  OwningOpRef moduleOp = ModuleOp::create(UnknownLoc::get(context.get()));
  OpBuilder builder(context.get());
  builder.setInsertionPointToStart(moduleOp->getBody());
  const auto loc = moduleOp->getLoc();
  auto function = func::FuncOp::create(builder, loc, "test",
                                       builder.getFunctionType({}, {}));
  auto* entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  for (const double angle : {0.0, std::numbers::pi, -std::numbers::pi,
                             2.0 * std::numbers::pi, -2.0 * std::numbers::pi}) {
    qco::GPhaseOp::create(builder, loc,
                          mlir::mqt::constantFromScalar(builder, loc, angle));
  }
  func::ReturnOp::create(builder, loc);

  ASSERT_TRUE(mlir::mqt::normalizeGlobalPhases(*moduleOp).succeeded());
  EXPECT_TRUE(function.getBody().getOps<qco::GPhaseOp>().empty());
}

TEST_F(GlobalPhaseNormalizationTest,
       PracticalAngleBoundaryPreservesFullUnitaryUnderControl) {
  for (const std::string angle : {"10000.0", "-10000.0"}) {
    const std::string source =
        R"mlir(module {
          func.func @test(%control: !qco.qubit, %target: !qco.qubit)
              -> (!qco.qubit, !qco.qubit) {
            %phase = arith.constant )mlir" +
        angle + R"mlir( : f64
            %control_out, %target_out = qco.ctrl(%control)
                targets(%arg = %target) {
              %x = qco.x %arg : !qco.qubit -> !qco.qubit
              qco.gphase(%phase)
              qco.yield %x : !qco.qubit
            } : ({!qco.qubit}, {!qco.qubit})
              -> ({!qco.qubit}, {!qco.qubit})
            return %control_out, %target_out : !qco.qubit, !qco.qubit
          }
        })mlir";
    auto moduleOp = parse(source);
    ASSERT_TRUE(moduleOp) << angle;
    expectNormalizedUnitary(moduleOp, 2);
  }
}

TEST_F(GlobalPhaseNormalizationTest, VerifiesPracticalConstantAngleRange) {
  const auto verifyAngle = [&](const double angle,
                               const bool useQCO) -> LogicalResult {
    OwningOpRef moduleOp = ModuleOp::create(UnknownLoc::get(context.get()));
    OpBuilder builder(context.get());
    builder.setInsertionPointToStart(moduleOp->getBody());
    const auto loc = moduleOp->getLoc();
    auto function = func::FuncOp::create(builder, loc, "test",
                                         builder.getFunctionType({}, {}));
    auto* entry = function.addEntryBlock();
    builder.setInsertionPointToStart(entry);
    auto value = mlir::mqt::constantFromScalar(builder, loc, angle);
    if (useQCO) {
      qco::GPhaseOp::create(builder, loc, value);
    } else {
      mlir::qc::GPhaseOp::create(builder, loc, value);
    }
    func::ReturnOp::create(builder, loc);
    return verify(*moduleOp);
  };

  for (const bool useQCO : {false, true}) {
    SCOPED_TRACE(useQCO ? "QCO" : "QC");
    EXPECT_TRUE(
        succeeded(verifyAngle(mlir::mqt::MAX_GLOBAL_PHASE_ANGLE, useQCO)));
    for (const double angle :
         {std::nextafter(mlir::mqt::MAX_GLOBAL_PHASE_ANGLE,
                         std::numeric_limits<double>::infinity()),
          -std::nextafter(mlir::mqt::MAX_GLOBAL_PHASE_ANGLE,
                          std::numeric_limits<double>::infinity()),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::infinity()}) {
      EXPECT_TRUE(failed(verifyAngle(angle, useQCO)));
    }
  }
}
