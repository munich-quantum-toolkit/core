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
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/Utils/Transforms/GlobalPhaseNormalization.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <gtest/gtest.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>

#include <array>
#include <chrono>
#include <limits>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

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

  static void expectNormalizedUnitary(OwningOpRef<ModuleOp>& module,
                                      const std::size_t numQubits) {
    const auto cloned = cast<ModuleOp>((*module)->clone());
    OwningOpRef<ModuleOp> expected(cloned);
    ASSERT_TRUE(quantum::normalizeGlobalPhases(*module).succeeded());
    ASSERT_TRUE(verify(*module).succeeded());
    mqt::test::expectFullUnitaryEqual(*expected, *module, numQubits);
  }

  static void expectNormalizedQCUnitary(OwningOpRef<ModuleOp>& module,
                                        const std::size_t numQubits) {
    const auto cloned = cast<ModuleOp>((*module)->clone());
    OwningOpRef<ModuleOp> expected(cloned);
    ASSERT_TRUE(quantum::normalizeGlobalPhases(*module).succeeded());

    for (ModuleOp candidate : {expected.get(), module.get()}) {
      PassManager pm(candidate.getContext());
      pm.addPass(createQCToQCO());
      ASSERT_TRUE(pm.run(candidate).succeeded());
      ASSERT_TRUE(verify(candidate).succeeded());
    }
    mqt::test::expectFullUnitaryEqual(*expected, *module, numQubits);
  }
};

} // namespace

TEST_F(GlobalPhaseNormalizationTest, CombinesQCOConstantsAtBlockExit) {
  auto module = parse(R"mlir(
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
  ASSERT_TRUE(module);
  ASSERT_TRUE(quantum::normalizeGlobalPhases(*module).succeeded());

  auto func = cast<func::FuncOp>(module->getBody()->front());
  auto phases = llvm::to_vector(func.getBody().getOps<qco::GPhaseOp>());
  ASSERT_EQ(phases.size(), 1);
  EXPECT_EQ(phases.front()->getNextNode(),
            func.getBody().front().getTerminator());
  const auto value = dyn_cast<FloatAttr>(
      phases.front().getTheta().getDefiningOp<arith::ConstantOp>().getValue());
  ASSERT_TRUE(value);
  EXPECT_DOUBLE_EQ(value.getValueAsDouble(), 0.75);
}

TEST_F(GlobalPhaseNormalizationTest,
       QCControlledExtractionPreservesFullUnitaryUnderOuterControl) {
  auto module = mlir::qc::QCProgramBuilder::build(
      context.get(), [](mlir::qc::QCProgramBuilder& builder) {
        const auto outer = builder.staticQubit(0);
        const auto inner = builder.staticQubit(1);
        const auto target = builder.staticQubit(2);
        builder.ctrl(outer, {inner, target}, [&](ValueRange outerTargets) {
          builder.ctrl(outerTargets[0], outerTargets[1],
                       [&](Value innerTarget) {
                         builder.x(innerTarget);
                         builder.gphase(0.731);
                       });
        });
        return builder.intConstant(0);
      });
  ASSERT_TRUE(module);
  expectNormalizedQCUnitary(module, 3);
}

TEST_F(GlobalPhaseNormalizationTest,
       QCInverseAndIntegralPowerPreserveFullUnitary) {
  auto module = mlir::qc::QCProgramBuilder::build(
      context.get(), [](mlir::qc::QCProgramBuilder& builder) {
        const auto q0 = builder.staticQubit(0);
        const auto q1 = builder.staticQubit(1);
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
  ASSERT_TRUE(module);
  expectNormalizedQCUnitary(module, 2);
}

TEST_F(GlobalPhaseNormalizationTest, PreservesDynamicOrderAndIsIdempotent) {
  auto module = parse(R"mlir(
    module {
      func.func @test(%q: !qc.qubit, %a: f64, %b: f64) {
        qc.gphase(%a)
        qc.x %q : !qc.qubit
        qc.gphase(%b)
        return
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(quantum::normalizeGlobalPhases(*module).succeeded());

  auto func = cast<func::FuncOp>(module->getBody()->front());
  auto phases = llvm::to_vector(func.getBody().getOps<mlir::qc::GPhaseOp>());
  ASSERT_EQ(phases.size(), 1);
  auto add = phases.front().getTheta().getDefiningOp<arith::AddFOp>();
  ASSERT_TRUE(add);
  EXPECT_EQ(add.getLhs(), func.getArgument(1));
  EXPECT_EQ(add.getRhs(), func.getArgument(2));

  std::string once;
  llvm::raw_string_ostream onceStream(once);
  module->print(onceStream);
  ASSERT_TRUE(quantum::normalizeGlobalPhases(*module).succeeded());
  std::string twice;
  llvm::raw_string_ostream twiceStream(twice);
  module->print(twiceStream);
  EXPECT_EQ(once, twice);
}

TEST_F(GlobalPhaseNormalizationTest, KeepsSCFStyleRegionsIndependent) {
  auto module = parse(R"mlir(
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
  ASSERT_TRUE(module);
  ASSERT_TRUE(quantum::normalizeGlobalPhases(*module).succeeded());

  auto func = *module->getOps<func::FuncOp>().begin();
  auto ifOp = *func.getBody().getOps<qco::IfOp>().begin();
  EXPECT_EQ(llvm::range_size(ifOp.getThenRegion().getOps<qco::GPhaseOp>()), 1);
  EXPECT_EQ(llvm::range_size(ifOp.getElseRegion().getOps<qco::GPhaseOp>()), 1);
  EXPECT_TRUE(func.getBody().getOps<qco::GPhaseOp>().empty());
}

TEST_F(GlobalPhaseNormalizationTest, FactorsInverseAndIntegralPower) {
  auto module = parse(R"mlir(
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
  ASSERT_TRUE(module);
  ASSERT_TRUE(quantum::normalizeGlobalPhases(*module).succeeded());

  auto func = *module->getOps<func::FuncOp>().begin();
  auto inv = *func.getBody().getOps<qco::InvOp>().begin();
  auto pow = *func.getBody().getOps<qco::PowOp>().begin();
  EXPECT_TRUE(inv.getBody()->getOps<qco::GPhaseOp>().empty());
  EXPECT_TRUE(pow.getBody()->getOps<qco::GPhaseOp>().empty());
  EXPECT_EQ(llvm::range_size(func.getBody().getOps<qco::GPhaseOp>()), 1);
}

TEST_F(GlobalPhaseNormalizationTest, FractionalPowerRemainsBoundary) {
  auto module = parse(R"mlir(
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
  ASSERT_TRUE(module);
  ASSERT_TRUE(quantum::normalizeGlobalPhases(*module).succeeded());

  auto func = *module->getOps<func::FuncOp>().begin();
  auto pow = *func.getBody().getOps<qco::PowOp>().begin();
  EXPECT_EQ(llvm::range_size(pow.getBody()->getOps<qco::GPhaseOp>()), 1);
}

TEST_F(GlobalPhaseNormalizationTest, DynamicPowerRemainsBoundary) {
  auto module = parse(R"mlir(
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
  ASSERT_TRUE(module);
  ASSERT_TRUE(quantum::normalizeGlobalPhases(*module).succeeded());

  auto func = *module->getOps<func::FuncOp>().begin();
  auto pow = *func.getBody().getOps<qco::PowOp>().begin();
  EXPECT_EQ(llvm::range_size(pow.getBody()->getOps<qco::GPhaseOp>()), 1);
  EXPECT_TRUE(func.getBody().getOps<qco::GPhaseOp>().empty());
}

TEST_F(GlobalPhaseNormalizationTest, NonFinitePowerExponentsRemainBoundaries) {
  for (const double exponent : {std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::infinity()}) {
    OwningOpRef module = ModuleOp::create(UnknownLoc::get(context.get()));
    OpBuilder builder(context.get());
    builder.setInsertionPointToStart(module->getBody());
    const auto loc = module->getLoc();
    const auto qubitType = qco::QubitType::get(context.get());
    auto function =
        func::FuncOp::create(builder, loc, "test",
                             builder.getFunctionType({qubitType}, {qubitType}));
    auto* entry = function.addEntryBlock();
    builder.setInsertionPointToStart(entry);
    auto pow = qco::PowOp::create(
        builder, loc, entry->getArgument(0), exponent, [&](Value target) {
          const auto out = qco::XOp::create(builder, loc, target).getQubitOut();
          qco::GPhaseOp::create(builder, loc,
                                utils::constantFromScalar(builder, loc, 0.371));
          return out;
        });
    func::ReturnOp::create(builder, loc, pow.getOutputTarget(0));

    ASSERT_TRUE(quantum::normalizeGlobalPhases(*module).succeeded());
    ASSERT_TRUE(verify(*module).succeeded());
    EXPECT_EQ(llvm::range_size(pow.getBody()->getOps<qco::GPhaseOp>()), 1);
    EXPECT_TRUE(function.getBody().getOps<qco::GPhaseOp>().empty());
  }
}

TEST_F(GlobalPhaseNormalizationTest, FactorsControlledPhaseOntoControl) {
  auto module = parse(R"mlir(
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
  ASSERT_TRUE(module);
  ASSERT_TRUE(quantum::normalizeGlobalPhases(*module).succeeded());

  auto func = *module->getOps<func::FuncOp>().begin();
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
  auto module = parse(R"mlir(
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
  ASSERT_TRUE(module);
  expectNormalizedUnitary(module, 3);
}

TEST_F(GlobalPhaseNormalizationTest, ThreeControlsPreserveFullUnitary) {
  auto module = parse(R"mlir(
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
  ASSERT_TRUE(module);
  expectNormalizedUnitary(module, 4);

  auto func = *module->getOps<func::FuncOp>().begin();
  auto controls = llvm::to_vector(func.getBody().getOps<qco::CtrlOp>());
  ASSERT_EQ(controls.size(), 2);
  EXPECT_EQ(controls.back().getNumControls(), 2);
  EXPECT_EQ(controls.back().getNumTargets(), 1);
}

TEST_F(GlobalPhaseNormalizationTest, ReorderedQCOControlsThreadCorrectResults) {
  auto module = parse(R"mlir(
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
  ASSERT_TRUE(module);
  ASSERT_TRUE(quantum::normalizeGlobalPhases(*module).succeeded());
  ASSERT_TRUE(verify(*module).succeeded());

  auto func = *module->getOps<func::FuncOp>().begin();
  auto controls = llvm::to_vector(func.getBody().getOps<qco::CtrlOp>());
  ASSERT_EQ(controls.size(), 2);
  auto returnOp = cast<func::ReturnOp>(func.getBody().front().getTerminator());
  EXPECT_EQ(returnOp.getOperand(0), controls[1].getOutputControl(1));
  EXPECT_EQ(returnOp.getOperand(1), controls[1].getOutputTarget(0));
  EXPECT_EQ(returnOp.getOperand(2), controls[1].getOutputControl(0));
  EXPECT_EQ(returnOp.getOperand(3), controls[0].getOutputTarget(0));
}

TEST_F(GlobalPhaseNormalizationTest, MultipleTargetsPreserveFullUnitary) {
  auto module = parse(R"mlir(
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
  ASSERT_TRUE(module);
  expectNormalizedUnitary(module, 4);
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
    auto module = parse(source);
    ASSERT_TRUE(module) << exponent;
    expectNormalizedUnitary(module, 1);
    auto func = *module->getOps<func::FuncOp>().begin();
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
  auto module = parse(source);
  ASSERT_TRUE(module);
  expectNormalizedUnitary(module, 2);

  module->walk([&](qco::InvOp inv) {
    EXPECT_TRUE(inv.getBody()->getOps<qco::GPhaseOp>().empty());
  });
  module->walk([&](qco::PowOp pow) {
    EXPECT_TRUE(pow.getBody()->getOps<qco::GPhaseOp>().empty());
  });
}

TEST_F(GlobalPhaseNormalizationTest, ZeroControlsReleaseAnUnchangedPhase) {
  OwningOpRef module = ModuleOp::create(UnknownLoc::get(context.get()));
  OpBuilder builder(context.get());
  builder.setInsertionPointToStart(module->getBody());
  const auto loc = module->getLoc();
  const auto qubitType = qco::QubitType::get(context.get());
  auto function = func::FuncOp::create(
      builder, loc, "test", builder.getFunctionType({qubitType}, {qubitType}));
  auto* entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  const auto phase = utils::constantFromScalar(builder, loc, 0.417);
  auto ctrl = qco::CtrlOp::create(
      builder, loc, ValueRange{}, entry->getArgument(0), [&](Value target) {
        const auto out = qco::XOp::create(builder, loc, target).getQubitOut();
        qco::GPhaseOp::create(builder, loc, phase);
        return out;
      });
  func::ReturnOp::create(builder, loc, ctrl.getOutputTarget(0));
  expectNormalizedUnitary(module, 1);

  EXPECT_TRUE(ctrl.getBody()->getOps<qco::GPhaseOp>().empty());
  EXPECT_EQ(llvm::range_size(function.getBody().getOps<qco::GPhaseOp>()), 1);
}

TEST_F(GlobalPhaseNormalizationTest,
       MemoryDependentAngleRemainsInsideModifier) {
  auto module = parse(R"mlir(
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
  ASSERT_TRUE(module);
  ASSERT_TRUE(quantum::normalizeGlobalPhases(*module).succeeded());
  ASSERT_TRUE(verify(*module).succeeded());

  auto func = *module->getOps<func::FuncOp>().begin();
  auto inv = *func.getBody().getOps<qco::InvOp>().begin();
  EXPECT_EQ(llvm::range_size(inv.getBody()->getOps<qco::GPhaseOp>()), 1);
  EXPECT_TRUE(func.getBody().getOps<qco::GPhaseOp>().empty());
}

TEST_F(GlobalPhaseNormalizationTest, CFGBlocksRemainIndependentScopes) {
  auto module = parse(R"mlir(
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
  ASSERT_TRUE(module);
  ASSERT_TRUE(quantum::normalizeGlobalPhases(*module).succeeded());
  ASSERT_TRUE(verify(*module).succeeded());

  auto func = *module->getOps<func::FuncOp>().begin();
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
  auto module = parse(R"mlir(
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
  ASSERT_TRUE(module);
  ASSERT_TRUE(quantum::normalizeGlobalPhases(*module).succeeded());

  for (auto func : module->getOps<func::FuncOp>()) {
    EXPECT_EQ(llvm::range_size(func.getBody().getOps<qco::GPhaseOp>()), 1);
  }
  EXPECT_TRUE(module->getBody()->getOps<qco::GPhaseOp>().empty());
}

TEST_F(GlobalPhaseNormalizationTest,
       IndexSwitchRegionsRemainIndependentScopes) {
  auto module = parse(R"mlir(
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
  ASSERT_TRUE(module);
  ASSERT_TRUE(quantum::normalizeGlobalPhases(*module).succeeded());
  ASSERT_TRUE(verify(*module).succeeded());

  auto func = *module->getOps<func::FuncOp>().begin();
  auto switchOp = *func.getBody().getOps<qco::IndexSwitchOp>().begin();
  for (auto& region : switchOp->getRegions()) {
    EXPECT_EQ(llvm::range_size(region.getOps<qco::GPhaseOp>()), 1);
  }
  EXPECT_TRUE(func.getBody().getOps<qco::GPhaseOp>().empty());
}

TEST_F(GlobalPhaseNormalizationTest, SCFLoopRegionRemainsAnIndependentScope) {
  auto module = parse(R"mlir(
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
  ASSERT_TRUE(module);
  ASSERT_TRUE(quantum::normalizeGlobalPhases(*module).succeeded());
  ASSERT_TRUE(verify(*module).succeeded());

  auto func = *module->getOps<func::FuncOp>().begin();
  auto loop = *func.getBody().getOps<scf::ForOp>().begin();
  EXPECT_EQ(llvm::range_size(loop.getBody()->getOps<qco::GPhaseOp>()), 1);
  EXPECT_TRUE(func.getBody().getOps<qco::GPhaseOp>().empty());
}

TEST_F(GlobalPhaseNormalizationTest,
       ExactSpecialConstantsCancelWithoutTolerance) {
  OwningOpRef module = ModuleOp::create(UnknownLoc::get(context.get()));
  OpBuilder builder(context.get());
  builder.setInsertionPointToStart(module->getBody());
  const auto loc = module->getLoc();
  auto function = func::FuncOp::create(builder, loc, "test",
                                       builder.getFunctionType({}, {}));
  auto* entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  for (const double angle : {0.0, std::numbers::pi, -std::numbers::pi,
                             2.0 * std::numbers::pi, -2.0 * std::numbers::pi}) {
    qco::GPhaseOp::create(builder, loc,
                          utils::constantFromScalar(builder, loc, angle));
  }
  func::ReturnOp::create(builder, loc);

  ASSERT_TRUE(quantum::normalizeGlobalPhases(*module).succeeded());
  EXPECT_TRUE(function.getBody().getOps<qco::GPhaseOp>().empty());
}

TEST_F(GlobalPhaseNormalizationTest, NonFiniteConstantsRemainExplicit) {
  OwningOpRef module = ModuleOp::create(UnknownLoc::get(context.get()));
  OpBuilder builder(context.get());
  builder.setInsertionPointToStart(module->getBody());
  const auto loc = module->getLoc();
  auto function = func::FuncOp::create(builder, loc, "test",
                                       builder.getFunctionType({}, {}));
  auto* entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  qco::GPhaseOp::create(
      builder, loc,
      utils::constantFromScalar(builder, loc,
                                std::numeric_limits<double>::quiet_NaN()));
  qco::GPhaseOp::create(
      builder, loc,
      utils::constantFromScalar(builder, loc,
                                std::numeric_limits<double>::infinity()));
  func::ReturnOp::create(builder, loc);

  ASSERT_TRUE(quantum::normalizeGlobalPhases(*module).succeeded());
  ASSERT_TRUE(verify(*module).succeeded());
  auto phases = llvm::to_vector(function.getBody().getOps<qco::GPhaseOp>());
  ASSERT_EQ(phases.size(), 1);
  EXPECT_TRUE(phases.front().getTheta().getDefiningOp<arith::AddFOp>());
  EXPECT_EQ(phases.front()->getNextNode(), entry->getTerminator());
}

TEST_F(GlobalPhaseNormalizationTest, ScalesLinearlyAcrossLargePhaseScopes) {
  constexpr std::array<std::size_t, 3> sizes{1'000, 10'000, 100'000};
  std::vector<std::chrono::nanoseconds> durations;
  durations.reserve(sizes.size());

  for (const auto size : sizes) {
    SCOPED_TRACE(size);
    OwningOpRef module = ModuleOp::create(UnknownLoc::get(context.get()));
    OpBuilder builder(context.get());
    builder.setInsertionPointToStart(module->getBody());
    const auto loc = module->getLoc();
    auto function = func::FuncOp::create(builder, loc, "test",
                                         builder.getFunctionType({}, {}));
    auto* entry = function.addEntryBlock();
    builder.setInsertionPointToStart(entry);
    const auto angle = utils::constantFromScalar(builder, loc, 0.001);
    for (std::size_t i = 0; i < size; ++i) {
      qco::GPhaseOp::create(builder, loc, angle);
    }
    func::ReturnOp::create(builder, loc);

    const auto start = std::chrono::steady_clock::now();
    ASSERT_TRUE(quantum::normalizeGlobalPhases(*module).succeeded());
    durations.emplace_back(std::chrono::steady_clock::now() - start);
    EXPECT_EQ(llvm::range_size(function.getBody().getOps<qco::GPhaseOp>()), 1);
  }

  RecordProperty("normalize_1000_ns", durations[0].count());
  RecordProperty("normalize_10000_ns", durations[1].count());
  RecordProperty("normalize_100000_ns", durations[2].count());
  EXPECT_LT(durations[1].count(), durations[0].count() * 50);
  EXPECT_LT(durations[2].count(), durations[1].count() * 50);
}
