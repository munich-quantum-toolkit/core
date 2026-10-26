/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/Edge.hpp"
#include "dd/Package.hpp"
#include "mqt/Conversion/QCToQCO/QCToQCO.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/MQT/Transforms/GlobalPhaseNormalization.h"
#include "mqt/Dialect/MQT/Transforms/Passes.h"
#include "mqt/Dialect/MQT/Utils/Angles.h"
#include "mqt/Dialect/MQT/Utils/ConstantFolding.h"
#include "mqt/Dialect/MQT/Utils/Parameters.h"
#include "mqt/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mqt/Dialect/QC/IR/QCDialect.h"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/Utils/DDFunctionality.h"

#include "ExactUnitaryTest.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
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
    registry
        .insert<arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect,
                memref::MemRefDialect, mlir::mqt::MQTDialect,
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

TEST_F(GlobalPhaseNormalizationTest,
       UnrollsEagerParameterChainsWithoutCrossingClassicalControl) {
  for (bool convertToQCO : {false, true}) {
    for (StringRef modifier :
         {"qc.inv", "qc.ctrl(%c) targets", "qc.pow(%two)"}) {
      SCOPED_TRACE(modifier.str());
      SCOPED_TRACE(convertToQCO);
      const auto source = std::string(R"mlir(
        func.func @test(%cond: i1, %theta: f64, %q: !qc.qubit,
                        %r: !qc.qubit, %c: !qc.qubit) {
          %two = arith.constant 2.0 : f64
          scf.if %cond {
      )mlir") + modifier.str() +
                          R"mlir( (%a = %q, %b = %r) {
            %n = arith.fptosi %theta : f64 to i64
            %one = arith.constant 1 : i64
            %d = arith.divsi %one, %n : i64
            %angle = arith.sitofp %d : i64 to f64
            qc.rx(%angle) %a : !qc.qubit
            qc.ry(%angle) %b : !qc.qubit
            qc.yield
          } : )mlir" +
                          (modifier.starts_with("qc.ctrl")
                               ? "{!qc.qubit}, {!qc.qubit, !qc.qubit}"
                               : "!qc.qubit, !qc.qubit") +
                          R"mlir(
          }
          return
        }
      )mlir";
      auto moduleOp = parse(source);
      ASSERT_TRUE(moduleOp);
      PassManager pm(context.get());
      if (convertToQCO) {
        pm.addPass(createQCToQCO());
      }
      pm.addPass(mlir::mqt::createUnrollModifiers());
      ASSERT_TRUE(succeeded(pm.run(*moduleOp)));
      ASSERT_TRUE(succeeded(verify(*moduleOp)));
      size_t divisions = 0;
      size_t modifiers = 0;
      moduleOp->walk([&](arith::DivSIOp div) {
        ++divisions;
        EXPECT_TRUE((isa<scf::IfOp, qco::IfOp>(div->getParentOp())));
        EXPECT_FALSE(isSpeculatable(div));
      });
      moduleOp->walk([&](Operation* op) {
        if (isa<qc::InvOp, qc::CtrlOp, qc::PowOp, qco::InvOp, qco::CtrlOp,
                qco::PowOp>(op)) {
          ++modifiers;
          EXPECT_EQ(op->getRegion(0).front().getOperations().size(), 2);
        }
      });
      EXPECT_EQ(divisions, 1);
      EXPECT_EQ(modifiers, 2);
    }
  }
}

TEST_F(GlobalPhaseNormalizationTest,
       ExtractsEagerPhaseParameterWithoutCrossingClassicalControl) {
  for (bool convertToQCO : {false, true}) {
    auto moduleOp = parse(R"mlir(
      func.func @test(%cond: i1, %theta: f64, %q: !qc.qubit) {
        scf.if %cond {
          qc.inv (%a = %q) {
            %n = arith.fptosi %theta : f64 to i64
            %one = arith.constant 1 : i64
            qc.inv (%b = %a) {
              %d = arith.divsi %one, %n : i64
              %angle = arith.sitofp %d : i64 to f64
              qc.inv (%c = %b) {
                qc.gphase(%angle)
                qc.x %c : !qc.qubit
                qc.yield
              } : !qc.qubit
              qc.yield
            } : !qc.qubit
            %late = arith.negf %theta : f64
            qc.rx(%late) %a : !qc.qubit
            qc.yield
          } : !qc.qubit
        }
        return
      }
    )mlir");
    ASSERT_TRUE(moduleOp);
    if (convertToQCO) {
      PassManager pm(context.get());
      pm.addPass(createQCToQCO());
      ASSERT_TRUE(succeeded(pm.run(*moduleOp)));
    }
    SmallVector<Operation*> producers;
    moduleOp->walk<WalkOrder::PreOrder>([&](Operation* op) {
      if (isa<arith::FPToSIOp, arith::ConstantOp, arith::DivSIOp,
              arith::SIToFPOp, arith::NegFOp>(op)) {
        producers.push_back(op);
      }
    });
    struct Moves final : RewriterBase::Listener {
      SmallVector<Operation*> operations;
      void notifyOperationInserted(Operation* op,
                                   OpBuilder::InsertPoint previous) override {
        if (previous.isSet()) {
          operations.push_back(op);
        }
      }
    } moves;
    IRRewriter rewriter(context.get());
    rewriter.setListener(&moves);
    ASSERT_TRUE(
        succeeded(mlir::mqt::normalizeGlobalPhases(*moduleOp, rewriter)));
    ASSERT_TRUE(succeeded(verify(*moduleOp)));
    // Each producer moves once, in source order, regardless of modifier depth.
    EXPECT_EQ(moves.operations, producers);
    size_t phases = 0;
    size_t divisions = 0;
    moduleOp->walk([&](arith::DivSIOp div) {
      ++divisions;
      EXPECT_TRUE((isa<scf::IfOp, qco::IfOp>(div->getParentOp())));
    });
    moduleOp->walk([&](Operation* op) {
      if (isa<qc::GPhaseOp, qco::GPhaseOp>(op)) {
        ++phases;
        EXPECT_TRUE((isa<scf::IfOp, qco::IfOp>(op->getParentOp())));
        EXPECT_TRUE(op->getOperand(0).getDefiningOp<arith::NegFOp>());
      }
    });
    EXPECT_EQ(phases, 1);
    EXPECT_EQ(divisions, 1);
  }
}

TEST_F(GlobalPhaseNormalizationTest,
       NestedModifierRewritesPreserveParameterProducers) {
  for (bool convertToQCO : {false, true}) {
    for (int variant : {0, 1, 2}) {
      SCOPED_TRACE(convertToQCO);
      SCOPED_TRACE(variant);
      const auto* const outer = variant == 2 ? "qc.pow(%power)" : "qc.inv";
      const auto* const inner = variant == 0 ? "qc.inv" : "qc.ctrl(%a) targets";
      const auto* const innerTypes =
          variant == 0 ? "!qc.qubit" : "{!qc.qubit}, {!qc.qubit}";
      const auto source = std::string(R"mlir(
        func.func private @rotate(%angle: f64, %q: !qc.qubit)
            attributes {mqt.unitary, no_inline} {
          qc.rx(%angle) %q : !qc.qubit
          return
        }
        func.func @test(%cond: i1, %theta: f64, %power: f64,
                        %q: !qc.qubit, %r: !qc.qubit) {
          scf.if %cond {
      )mlir") + outer + R"mlir( (%a = %q, %b = %r) {
            %n = arith.fptosi %theta : f64 to i64
            %one = arith.constant 1 : i64
            %d = arith.divsi %one, %n : i64
            %angle = arith.sitofp %d : i64 to f64
      )mlir" + inner + R"mlir( (%t = %b) {
              qc.call @rotate(%angle, %t) : f64, !qc.qubit
              qc.yield
            } : )mlir" + innerTypes +
                          R"mlir(
            qc.yield
          } : !qc.qubit, !qc.qubit
          }
          return
        }
      )mlir";
      auto moduleOp = parse(source);
      ASSERT_TRUE(moduleOp);
      PassManager pm(context.get());
      if (convertToQCO) {
        pm.addPass(createQCToQCO());
      }
      pm.addPass(createCanonicalizerPass());
      ASSERT_TRUE(succeeded(pm.run(*moduleOp)));
      ASSERT_TRUE(succeeded(verify(*moduleOp)));
      size_t divisions = 0;
      moduleOp->walk([&](arith::DivSIOp div) {
        ++divisions;
        EXPECT_TRUE((isa<scf::IfOp, qco::IfOp>(div->getParentOp())));
        EXPECT_FALSE(div->use_empty());
      });
      EXPECT_EQ(divisions, 1);
    }
  }
}

TEST_F(GlobalPhaseNormalizationTest, CombinesQCOConstantsAtBlockExit) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @test(%q: !qco.qubit) -> !qco.qubit {
        %c0 = arith.constant 0.25 : f64
        %c1 = arith.constant 0.5 : f64
        qco.gphase(%c0)
        %q1 = qco.x %q : !qco.qubit
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
       FoldsDerivedPhasesWithinPracticalAngleLimit) {
  // QV_n100 also produces mulf(sitofp(i64), f64) phase expressions.
  for (bool integerLeaf : {false, true}) {
    SCOPED_TRACE(integerLeaf);
    OwningOpRef moduleOp = ModuleOp::create(UnknownLoc::get(context.get()));
    OpBuilder builder(context.get());
    builder.setInsertionPointToStart(moduleOp->getBody());
    const auto loc = moduleOp->getLoc();
    auto function = func::FuncOp::create(builder, loc, "test",
                                         builder.getFunctionType({}, {}));
    builder.setInsertionPointToStart(function.addEntryBlock());
    constexpr int phaseCount = 4000;
    for (int i = 0; i < phaseCount; ++i) {
      Value lhs;
      if (integerLeaf) {
        auto integer = arith::ConstantOp::create(builder, loc,
                                                 builder.getI64IntegerAttr(3));
        lhs = arith::SIToFPOp::create(builder, loc, builder.getF64Type(),
                                      integer);
      } else {
        lhs = mlir::mqt::constantFromScalar(builder, loc, 1.5);
      }
      auto rhs = mlir::mqt::constantFromScalar(builder, loc, 2.0);
      auto angle = arith::MulFOp::create(builder, loc, lhs, rhs);
      qco::GPhaseOp::create(builder, loc, angle.getResult());
    }
    func::ReturnOp::create(builder, loc);
    ASSERT_TRUE(succeeded(mlir::mqt::normalizeGlobalPhases(*moduleOp)));
    ASSERT_TRUE(succeeded(verify(*moduleOp)));
    // Fold constant contributions without creating a dead accumulation tree.
    EXPECT_TRUE(function.getBody().getOps<arith::AddFOp>().empty());
    auto phases = llvm::to_vector(function.getBody().getOps<qco::GPhaseOp>());
    ASSERT_EQ(phases.size(), 1);
    EXPECT_TRUE(phases.front().getTheta().getDefiningOp<arith::ConstantOp>());
    expectFoldableGlobalPhase(
        phases.front().getTheta(),
        mlir::mqt::normalizeAngle(phaseCount * (integerLeaf ? 6.0 : 3.0)));
  }
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
        return SmallVector<Value>{};
      });
  ASSERT_TRUE(moduleOp);
  expectNormalizedQCUnitary(moduleOp, 3);
}

TEST_F(GlobalPhaseNormalizationTest, RemovesQCControlEmptiedByPhaseExtraction) {
  auto moduleOp = mlir::qc::QCProgramBuilder::build(
      context.get(), [](mlir::qc::QCProgramBuilder& builder) {
        builder.ctrl(
            builder.staticQubit(0), builder.staticQubit(1), [&](Value target) {
              builder.inv(target, [&](Value) { builder.gphase(0.25); });
            });
        return SmallVector<Value>{};
      });
  ASSERT_TRUE(moduleOp);
  auto cloned = cast<ModuleOp>((*moduleOp)->clone());
  OwningOpRef<ModuleOp> expected(cloned);
  auto function = *moduleOp->getOps<func::FuncOp>().begin();
  ASSERT_EQ(llvm::range_size(function.getBody().getOps<mlir::qc::CtrlOp>()),
            1U);

  ASSERT_TRUE(mlir::mqt::normalizeGlobalPhases(*moduleOp).succeeded());
  ASSERT_TRUE(verify(*moduleOp).succeeded());
  EXPECT_TRUE(function.getBody().getOps<mlir::qc::CtrlOp>().empty());
  EXPECT_EQ(llvm::range_size(function.getBody().getOps<mlir::qc::POp>()), 1U);

  for (ModuleOp candidate : {expected.get(), moduleOp.get()}) {
    PassManager pm(candidate.getContext());
    pm.addPass(createQCToQCO());
    ASSERT_TRUE(pm.run(candidate).succeeded());
    ASSERT_TRUE(verify(candidate).succeeded());
  }
  ::mqt::test::expectFullUnitaryEqual(*expected, *moduleOp, 2);
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
        return SmallVector<Value>{};
      });
  ASSERT_TRUE(moduleOp);
  expectNormalizedQCUnitary(moduleOp, 2);
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

TEST_F(GlobalPhaseNormalizationTest, PowersRemainUnchanged) {
  for (bool useQCO : {false, true}) {
    for (StringRef exponent : {
             "",
             "-3.0",
             "-1.0",
             "0.0",
             "1.0",
             "2.0",
             "3.0",
             "0.5",
             "1.0e16",
             "1.0e308",
         }) {
      SCOPED_TRACE(exponent.str());
      SCOPED_TRACE(useQCO);
      const std::string source =
          R"mlir(func.func @test(%q: !qc.qubit, %dynamic: f64) {
          )mlir" +
          (exponent.empty()
               ? ""
               : "%exponent = arith.constant " + exponent.str() + " : f64\n") +
          "qc.inv (%outer = %q) { qc.pow(" +
          (exponent.empty() ? "%dynamic" : "%exponent") +
          R"mlir() (%arg = %outer) {
            %three = arith.constant 3.0 : f64
            %six = arith.constant 6.0 : f64
            qc.h %arg : !qc.qubit
            qc.gphase(%three)
            qc.gphase(%six)
            qc.yield
          } : !qc.qubit
          qc.yield
          } : !qc.qubit
          return
        })mlir";
      auto moduleOp = parse(source);
      ASSERT_TRUE(moduleOp);
      if (useQCO) {
        PassManager pm(context.get());
        pm.addPass(createQCToQCO());
        ASSERT_TRUE(succeeded(pm.run(*moduleOp)));
      }
      OwningOpRef<ModuleOp> original(cast<ModuleOp>((*moduleOp)->clone()));
      ASSERT_TRUE(succeeded(mlir::mqt::normalizeGlobalPhases(*moduleOp)));
      ASSERT_TRUE(succeeded(verify(*moduleOp)));
      EXPECT_TRUE(OperationEquivalence::isEquivalentTo(
          moduleOp->getOperation(), original->getOperation(),
          OperationEquivalence::Flags::None));
    }
  }
}

TEST_F(GlobalPhaseNormalizationTest, FactorsControlledPhaseOntoControl) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @test(%control: !qco.qubit, %target: !qco.qubit)
          -> (!qco.qubit, !qco.qubit) {
        %phase = arith.constant 0.25 : f64
        %control_out, %target_out = qco.ctrl(%control)
            targets(%arg = %target) {
          %x = qco.x %arg : !qco.qubit
          qco.gphase(%phase)
          qco.yield %x : !qco.qubit
        } : ({!qco.qubit}, {!qco.qubit})
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
       RemovesQCOControlEmptiedByPhaseExtraction) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @test(%control: !qco.qubit, %lhs: !qco.qubit,
                      %rhs: !qco.qubit)
          -> (!qco.qubit, !qco.qubit, !qco.qubit) {
        %control_out, %lhs_out, %rhs_out = qco.ctrl(%control)
            targets(%lhs_arg = %lhs, %rhs_arg = %rhs) {
          %lhs_inv, %rhs_inv = qco.inv (%a = %lhs_arg, %b = %rhs_arg) {
            %phase = arith.constant 0.25 : f64
            qco.gphase(%phase)
            qco.yield %a, %b : !qco.qubit, !qco.qubit
          } : {!qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit}
          qco.yield %lhs_inv, %rhs_inv : !qco.qubit, !qco.qubit
        } : ({!qco.qubit}, {!qco.qubit, !qco.qubit})
        return %control_out, %lhs_out, %rhs_out
            : !qco.qubit, !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(verify(*moduleOp).succeeded());
  auto function = *moduleOp->getOps<func::FuncOp>().begin();
  ASSERT_EQ(llvm::range_size(function.getBody().getOps<qco::CtrlOp>()), 1U);

  ASSERT_TRUE(mlir::mqt::normalizeGlobalPhases(*moduleOp).succeeded());
  ASSERT_TRUE(verify(*moduleOp).succeeded());
  EXPECT_TRUE(function.getBody().getOps<qco::CtrlOp>().empty());
  EXPECT_EQ(llvm::range_size(function.getBody().getOps<qco::POp>()), 1U);
  auto returnOp =
      cast<func::ReturnOp>(function.getBody().front().getTerminator());
  auto p = *function.getBody().getOps<qco::POp>().begin();
  EXPECT_EQ(returnOp.getOperand(0), p.getOutputTarget(0));
  EXPECT_EQ(returnOp.getOperand(1), function.getArgument(1));
  EXPECT_EQ(returnOp.getOperand(2), function.getArgument(2));
  expectFoldableGlobalPhase(p.getTheta(), -0.25);
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
            %x = qco.x %arg : !qco.qubit
            qco.gphase(%phase)
            qco.yield %x : !qco.qubit
          } : ({!qco.qubit}, {!qco.qubit})
          qco.yield %inner_control_out, %inner_target_out
              : !qco.qubit, !qco.qubit
        } : ({!qco.qubit}, {!qco.qubit, !qco.qubit})
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
          %h = qco.h %arg : !qco.qubit
          %x = qco.x %h : !qco.qubit
          qco.gphase(%phase)
          qco.yield %x : !qco.qubit
        } : ({!qco.qubit, !qco.qubit, !qco.qubit}, {!qco.qubit})
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
          %x = qco.x %arg : !qco.qubit
          qco.gphase(%phase)
          qco.yield %x : !qco.qubit
        } : ({!qco.qubit, !qco.qubit, !qco.qubit}, {!qco.qubit})
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
          %x = qco.x %a : !qco.qubit
          %h = qco.h %b : !qco.qubit
          qco.gphase(%phase)
          qco.yield %x, %h : !qco.qubit, !qco.qubit
        } : ({!qco.qubit, !qco.qubit}, {!qco.qubit, !qco.qubit})
        return %c0_out, %c1_out, %t0_out, %t1_out
            : !qco.qubit, !qco.qubit, !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  expectNormalizedUnitary(moduleOp, 4);
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
            %h = qco.h %inner_arg : !qco.qubit
            qco.gphase(%phase)
            qco.yield %h : !qco.qubit
          } : {!qco.qubit} -> {!qco.qubit}
          qco.yield %inner : !qco.qubit
        } : {!qco.qubit} -> {!qco.qubit}
        %b = qco.pow(%two) (%outer_arg = %q1) {
          %inner = qco.inv (%inner_arg = %outer_arg) {
            %x = qco.x %inner_arg : !qco.qubit
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

  size_t phaseCount = 0;
  moduleOp->walk([&](qco::GPhaseOp) { ++phaseCount; });
  EXPECT_EQ(phaseCount, 2U);
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
       MemoryDependentAngleCapturedByModifierMovesOutsideModifier) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @test(%q: !qco.qubit, %angles: memref<1xf64>)
          -> !qco.qubit {
        %c0 = arith.constant 0 : index
        %phase = memref.load %angles[%c0] : memref<1xf64>
        %out = qco.inv (%arg = %q) {
          %x = qco.x %arg : !qco.qubit
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
  EXPECT_TRUE(inv.getBody()->getOps<qco::GPhaseOp>().empty());
  EXPECT_EQ(llvm::range_size(func.getBody().getOps<qco::GPhaseOp>()), 1);
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
  for (const double angle : {
           0.0,
           std::numbers::pi,
           -std::numbers::pi,
           2.0 * std::numbers::pi,
           -2.0 * std::numbers::pi,
       }) {
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
              %x = qco.x %arg : !qco.qubit
              qco.gphase(%phase)
              qco.yield %x : !qco.qubit
            } : ({!qco.qubit}, {!qco.qubit})
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
    for (const double angle : {
             std::nextafter(mlir::mqt::MAX_GLOBAL_PHASE_ANGLE,
                            std::numeric_limits<double>::infinity()),
             -std::nextafter(mlir::mqt::MAX_GLOBAL_PHASE_ANGLE,
                             std::numeric_limits<double>::infinity()),
             std::numeric_limits<double>::quiet_NaN(),
             std::numeric_limits<double>::infinity(),
         }) {
      EXPECT_TRUE(failed(verifyAngle(angle, useQCO)));
    }
  }
}

TEST_F(GlobalPhaseNormalizationTest,
       SymbolicPhaseSumIsBalancedBoundedAndStable) {
  constexpr size_t count = 257;
  for (bool useQCO : {false, true}) {
    for (double angle : {0.37, 6000.0, -6000.0, std::numbers::pi / 65.0}) {
      SCOPED_TRACE(useQCO);
      SCOPED_TRACE(angle);
      auto moduleOp = parse(R"mlir(
        func.func @test(%theta: f64, %other: f64) { return }
      )mlir");
      ASSERT_TRUE(moduleOp);
      auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
      OpBuilder builder(function);
      builder.setInsertionPoint(function.getBody().front().getTerminator());
      for (size_t i = 0; i < count; ++i) {
        auto theta = function.getArgument(i % 2);
        if (useQCO) {
          qco::GPhaseOp::create(builder, function.getLoc(), theta);
        } else {
          qc::GPhaseOp::create(builder, function.getLoc(), theta);
        }
      }
      ASSERT_TRUE(succeeded(mlir::mqt::normalizeGlobalPhases(*moduleOp)));
      ASSERT_TRUE(succeeded(verify(*moduleOp)));
      llvm::DenseMap<Value, size_t> depths;
      size_t maxDepth = 0;
      Value phase;
      size_t phaseCount = 0;
      moduleOp->walk([&](Operation* operation) {
        if (auto add = dyn_cast<arith::AddFOp>(operation)) {
          const auto depth = 1 + std::max(depths.lookup(add.getLhs()),
                                          depths.lookup(add.getRhs()));
          depths[add.getResult()] = depth;
          maxDepth = std::max(maxDepth, depth);
        } else if (isa<qc::GPhaseOp, qco::GPhaseOp>(operation)) {
          phase = operation->getOperand(0);
          ++phaseCount;
        }
      });
      ASSERT_EQ(phaseCount, 1U);
      EXPECT_LE(maxDepth, std::bit_width(count - 1));
      const auto operationCount = llvm::range_size(function.getBody().front());
      ASSERT_TRUE(succeeded(mlir::mqt::normalizeGlobalPhases(*moduleOp)));
      EXPECT_EQ(llvm::range_size(function.getBody().front()), operationCount);
      builder.setInsertionPointToStart(&function.getBody().front());
      function.getArgument(0).replaceAllUsesWith(
          mlir::mqt::constantFromScalar(builder, function.getLoc(), angle));
      function.getArgument(1).replaceAllUsesWith(mlir::mqt::constantFromScalar(
          builder, function.getLoc(), -angle / 2));
      const auto actual = mlir::mqt::valueToConstantDouble(phase);
      ASSERT_TRUE(actual);
      EXPECT_TRUE(mlir::mqt::isValidGlobalPhaseAngle(*actual));
      constexpr double pairs = (count - 1.0) / 2.0;
      const double expected = (pairs + 1) * angle - pairs * angle / 2;
      EXPECT_NEAR(std::cos(*actual), std::cos(expected), 1e-10);
      EXPECT_NEAR(std::sin(*actual), std::sin(expected), 1e-10);
    }
  }
}

TEST_F(GlobalPhaseNormalizationTest, KeepsLastContributionBeforeMeasurements) {
  for (bool convertToQCO : {false, true}) {
    SCOPED_TRACE(convertToQCO);
    auto moduleOp = parse(R"mlir(
      func.func @main(%angle: f64) attributes {mqt.entry_point} {
        %q = qc.alloc : !qc.qubit
        %half = arith.constant 0.5 : f64
        qc.gphase(%half)
        qc.inv (%a = %q) {
          qc.gphase(%angle)
          qc.yield
        } : !qc.qubit
        qc.ctrl(%q) targets () {
          qc.gphase(%half)
          qc.yield
        } : {!qc.qubit}
        %m = qc.measure %q : !qc.qubit -> i1
        qc.dealloc %q : !qc.qubit
        return
      }
    )mlir");
    ASSERT_TRUE(moduleOp);
    if (convertToQCO) {
      PassManager pm(context.get());
      pm.addPass(createQCToQCO());
      ASSERT_TRUE(succeeded(pm.run(*moduleOp)));
    }
    ASSERT_TRUE(succeeded(mlir::mqt::normalizeGlobalPhases(*moduleOp)));
    ASSERT_TRUE(succeeded(verify(*moduleOp)));
    Operation* phase = nullptr;
    moduleOp->walk([&](Operation* op) {
      if (isa<qc::GPhaseOp, qco::GPhaseOp>(op)) {
        ASSERT_EQ(phase, nullptr);
        phase = op;
      }
    });
    ASSERT_NE(phase, nullptr);
    EXPECT_TRUE((isa<qc::MeasureOp, qco::MeasureOp>(phase->getNextNode())));
    ASSERT_TRUE(succeeded(mlir::mqt::normalizeGlobalPhases(*moduleOp)));
    EXPECT_TRUE((isa<qc::MeasureOp, qco::MeasureOp>(phase->getNextNode())));
  }
}

TEST_F(GlobalPhaseNormalizationTest, DropsOnlyFinalProgramPhases) {
  for (bool convertToQCO : {false, true}) {
    SCOPED_TRACE(convertToQCO);
    auto moduleOp = parse(R"mlir(
      func.func @helper(%theta: f64) {
        qc.gphase(%theta)
        return
      }
      func.func @main(%theta: f64, %cond: i1) attributes {mqt.entry_point} {
        %q = qc.alloc : !qc.qubit
        qc.gphase(%theta)
        scf.if %cond {
          qc.gphase(%theta)
        }
        qc.ctrl(%q) targets () {
          qc.gphase(%theta)
          qc.yield
        } : {!qc.qubit}
        qc.inv () {
          qc.gphase(%theta)
          qc.yield
        }
        qc.pow(%theta) () {
          qc.gphase(%theta)
          qc.yield
        }
        qc.dealloc %q : !qc.qubit
        return
      }
    )mlir");
    ASSERT_TRUE(moduleOp);
    PassManager pm(context.get());
    if (convertToQCO) {
      pm.addPass(createQCToQCO());
    }
    pm.addPass(mlir::mqt::createDropGlobalPhases());
    ASSERT_TRUE(succeeded(pm.run(*moduleOp)));
    ASSERT_TRUE(succeeded(verify(*moduleOp)));
    unsigned remaining = 0;
    moduleOp->walk([&](Operation* op) {
      remaining += isa<qc::GPhaseOp, qco::GPhaseOp>(op);
    });
    EXPECT_EQ(remaining, 4);
  }
}

TEST_F(GlobalPhaseNormalizationTest, DropPreservesObservableControlledPhase) {
  auto moduleOp = parse(R"mlir(
    func.func @main() attributes {mqt.entry_point} {
      %q = qc.static 0 : !qc.qubit
      %global = arith.constant 0.4 : f64
      %relative = arith.constant 0.6 : f64
      qc.gphase(%global)
      qc.h %q : !qc.qubit
      qc.ctrl(%q) targets () {
        qc.gphase(%relative)
        qc.yield
      } : {!qc.qubit}
      qc.h %q : !qc.qubit
      qc.dealloc %q : !qc.qubit
      return
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  PassManager pm(context.get());
  pm.addPass(createQCToQCO());
  pm.addPass(mlir::mqt::createDropGlobalPhases());
  pm.addPass(mlir::mqt::createNormalizeGlobalPhases());
  ASSERT_TRUE(succeeded(pm.run(*moduleOp)));
  auto package = llvm::cantFail(dd::Package::create(1));
  auto function = *moduleOp->getOps<func::FuncOp>().begin();
  auto result = qco::buildFunctionality(function, *package);
  ASSERT_TRUE(succeeded(result));
  const auto matrix = llvm::cantFail(dd::getMatrix(*result, 1));
  // H diag(1, exp(0.6i)) H retains the relative phase exactly.
  const auto phase = std::polar(1.0, 0.6);
  for (size_t row = 0; row < 2; ++row) {
    for (size_t column = 0; column < 2; ++column) {
      const auto expected = (1.0 + (row == column ? phase : -phase)) / 2.0;
      EXPECT_NEAR(std::abs(matrix[row][column] - expected), 0.0, 1e-12);
    }
  }
  package->decRef(*result);
}
