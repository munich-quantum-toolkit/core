/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/MQT/Utils/ConstantFolding.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/QCOUtils.h"

#include <gtest/gtest.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/Passes.h>

using namespace mlir;
using namespace mlir::qco;

namespace {

class QCOModifierYieldCanonicalizationTest : public testing::Test {
protected:
  MLIRContext context_;

  void SetUp() override {
    context_.loadDialect<mlir::mqt::MQTDialect, QCODialect, arith::ArithDialect,
                         func::FuncDialect>();
  }

  OwningOpRef<ModuleOp> canonicalize(StringRef source) {
    auto moduleOp = parseSourceString<ModuleOp>(source, &context_);
    EXPECT_TRUE(moduleOp);
    if (!moduleOp) {
      return {};
    }
    EXPECT_TRUE(succeeded(verify(*moduleOp)));
    EXPECT_TRUE(succeeded(verifyLinearity(*moduleOp)));
    PassManager manager(&context_);
    manager.addPass(createCanonicalizerPass());
    EXPECT_TRUE(succeeded(manager.run(*moduleOp)));
    EXPECT_TRUE(succeeded(verify(*moduleOp)));
    EXPECT_TRUE(succeeded(verifyLinearity(*moduleOp)));
    return moduleOp;
  }

  static void expectModifiers(ModuleOp moduleOp, unsigned controls,
                              unsigned inverses, unsigned powers) {
    unsigned ctrlCount = 0;
    unsigned invCount = 0;
    unsigned powCount = 0;
    moduleOp.walk([&](CtrlOp) { ++ctrlCount; });
    moduleOp.walk([&](InvOp) { ++invCount; });
    moduleOp.walk([&](PowOp) { ++powCount; });
    EXPECT_EQ(ctrlCount, controls);
    EXPECT_EQ(invCount, inverses);
    EXPECT_EQ(powCount, powers);
  }
};

TEST_F(QCOModifierYieldCanonicalizationTest, EmptyInverseRetainsYieldCycle) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit, %q2: !qco.qubit) -> (!qco.qubit, !qco.qubit, !qco.qubit) {
        %o0, %o1, %o2 = qco.inv(%a = %q0, %b = %q1, %c = %q2) {
          qco.yield %b, %c, %a : !qco.qubit, !qco.qubit, !qco.qubit
        } : {!qco.qubit, !qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit, !qco.qubit}
        return %o0, %o1, %o2 : !qco.qubit, !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  expectModifiers(*moduleOp, 0U, 1U, 0U);
}

TEST_F(QCOModifierYieldCanonicalizationTest, EmptyPowerRetainsYieldCycle) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit, %q2: !qco.qubit) -> (!qco.qubit, !qco.qubit, !qco.qubit) {
        %two = arith.constant 2.0 : f64
        %o0, %o1, %o2 = qco.pow(%two) (%a = %q0, %b = %q1, %c = %q2) {
          qco.yield %b, %c, %a : !qco.qubit, !qco.qubit, !qco.qubit
        } : {!qco.qubit, !qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit, !qco.qubit}
        return %o0, %o1, %o2 : !qco.qubit, !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  expectModifiers(*moduleOp, 0U, 0U, 1U);
}

TEST_F(QCOModifierYieldCanonicalizationTest,
       EmptyControlRetainsYieldPermutation) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit, %q2: !qco.qubit) -> (!qco.qubit, !qco.qubit, !qco.qubit) {
        %o0, %o1, %o2 = qco.ctrl(%q0) targets(%a = %q1, %b = %q2) {
          qco.yield %b, %a : !qco.qubit, !qco.qubit
        } : ({!qco.qubit}, {!qco.qubit, !qco.qubit}) -> ({!qco.qubit}, {!qco.qubit, !qco.qubit})
        return %o0, %o1, %o2 : !qco.qubit, !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  expectModifiers(*moduleOp, 1U, 0U, 0U);
}

TEST_F(QCOModifierYieldCanonicalizationTest,
       ControlledBarrierRetainsYieldPermutation) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit, %q2: !qco.qubit) -> (!qco.qubit, !qco.qubit, !qco.qubit) {
        %o0, %o1, %o2 = qco.ctrl(%q0) targets(%a = %q1, %b = %q2) {
          %x, %y = qco.barrier %a, %b : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
          qco.yield %y, %x : !qco.qubit, !qco.qubit
        } : ({!qco.qubit}, {!qco.qubit, !qco.qubit}) -> ({!qco.qubit}, {!qco.qubit, !qco.qubit})
        return %o0, %o1, %o2 : !qco.qubit, !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  expectModifiers(*moduleOp, 1U, 0U, 0U);
}

TEST_F(QCOModifierYieldCanonicalizationTest,
       SelfAdjointInverseRetainsYieldPermutation) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit) -> (!qco.qubit, !qco.qubit) {
        %o0, %o1 = qco.inv(%a = %q0, %b = %q1) {
          %x = qco.x %a : !qco.qubit -> !qco.qubit
          qco.yield %b, %x : !qco.qubit, !qco.qubit
        } : {!qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit}
        return %o0, %o1 : !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  expectModifiers(*moduleOp, 0U, 1U, 0U);
}

TEST_F(QCOModifierYieldCanonicalizationTest,
       KnownInverseRetainsYieldPermutation) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit) -> (!qco.qubit, !qco.qubit) {
        %o0, %o1 = qco.inv(%a = %q0, %b = %q1) {
          %x = qco.s %a : !qco.qubit -> !qco.qubit
          qco.yield %b, %x : !qco.qubit, !qco.qubit
        } : {!qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit}
        return %o0, %o1 : !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  expectModifiers(*moduleOp, 0U, 1U, 0U);
}

TEST_F(QCOModifierYieldCanonicalizationTest,
       NestedInverseRetainsOuterYieldPermutation) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit, %angle: f64) -> (!qco.qubit, !qco.qubit) {
        %o0, %o1 = qco.inv(%a = %q0, %b = %q1) {
          %square = arith.mulf %angle, %angle : f64
          %i0, %i1 = qco.inv(%c = %a, %d = %b) {
            %h = qco.h %c : !qco.qubit -> !qco.qubit
            %r = qco.rx(%square) %h : !qco.qubit -> !qco.qubit
            %x = qco.x %d : !qco.qubit -> !qco.qubit
            qco.yield %r, %x : !qco.qubit, !qco.qubit
          } : {!qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit}
          qco.yield %i1, %i0 : !qco.qubit, !qco.qubit
        } : {!qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit}
        return %o0, %o1 : !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  expectModifiers(*moduleOp, 0U, 2U, 0U);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto inverse = *function.getBody().getOps<InvOp>().begin();
  EXPECT_TRUE(
      llvm::hasSingleElement(inverse.getBody()->getOps<arith::MulFOp>()));
}

TEST_F(QCOModifierYieldCanonicalizationTest,
       NestedControlRetainsOuterYieldPermutation) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit, %q2: !qco.qubit, %q3: !qco.qubit) -> (!qco.qubit, !qco.qubit, !qco.qubit, !qco.qubit) {
        %o0, %o1, %o2, %o3 = qco.ctrl(%q0) targets(%a = %q1, %b = %q2, %c = %q3) {
          %i0, %i1, %i2 = qco.ctrl(%a) targets(%d = %b, %e = %c) {
            %x = qco.x %d : !qco.qubit -> !qco.qubit
            %s = qco.s %e : !qco.qubit -> !qco.qubit
            qco.yield %x, %s : !qco.qubit, !qco.qubit
          } : ({!qco.qubit}, {!qco.qubit, !qco.qubit}) -> ({!qco.qubit}, {!qco.qubit, !qco.qubit})
          qco.yield %i0, %i2, %i1 : !qco.qubit, !qco.qubit, !qco.qubit
        } : ({!qco.qubit}, {!qco.qubit, !qco.qubit, !qco.qubit}) -> ({!qco.qubit}, {!qco.qubit, !qco.qubit, !qco.qubit})
        return %o0, %o1, %o2, %o3 : !qco.qubit, !qco.qubit, !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  expectModifiers(*moduleOp, 2U, 0U, 0U);
}

TEST_F(QCOModifierYieldCanonicalizationTest,
       InverseControlRetainsOuterYieldPermutation) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit) -> (!qco.qubit, !qco.qubit) {
        %o0, %o1 = qco.inv(%a = %q0, %b = %q1) {
          %i0, %i1 = qco.ctrl(%a) targets(%c = %b) {
            %x = qco.x %c : !qco.qubit -> !qco.qubit
            qco.yield %x : !qco.qubit
          } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
          qco.yield %i1, %i0 : !qco.qubit, !qco.qubit
        } : {!qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit}
        return %o0, %o1 : !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  expectModifiers(*moduleOp, 1U, 1U, 0U);
}

TEST_F(QCOModifierYieldCanonicalizationTest,
       PoweredControlRetainsOuterYieldPermutation) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit) -> (!qco.qubit, !qco.qubit) {
        %two = arith.constant 2.0 : f64
        %o0, %o1 = qco.pow(%two) (%a = %q0, %b = %q1) {
          %i0, %i1 = qco.ctrl(%a) targets(%c = %b) {
            %x = qco.x %c : !qco.qubit -> !qco.qubit
            qco.yield %x : !qco.qubit
          } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
          qco.yield %i1, %i0 : !qco.qubit, !qco.qubit
        } : {!qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit}
        return %o0, %o1 : !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  expectModifiers(*moduleOp, 1U, 0U, 1U);
}

TEST_F(QCOModifierYieldCanonicalizationTest,
       InversePowerRetainsOuterYieldPermutation) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit) -> (!qco.qubit, !qco.qubit) {
        %three = arith.constant 3.0 : f64
        %o0, %o1 = qco.inv(%a = %q0, %b = %q1) {
          %i0, %i1 = qco.pow(%three) (%c = %a, %d = %b) {
            %x = qco.x %c : !qco.qubit -> !qco.qubit
            %s = qco.s %d : !qco.qubit -> !qco.qubit
            qco.yield %x, %s : !qco.qubit, !qco.qubit
          } : {!qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit}
          qco.yield %i1, %i0 : !qco.qubit, !qco.qubit
        } : {!qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit}
        return %o0, %o1 : !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  expectModifiers(*moduleOp, 0U, 1U, 1U);
}

TEST_F(QCOModifierYieldCanonicalizationTest,
       NestedPowerRetainsOuterYieldPermutation) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit) -> (!qco.qubit, !qco.qubit) {
        %two = arith.constant 2.0 : f64
        %three = arith.constant 3.0 : f64
        %o0, %o1 = qco.pow(%two) (%a = %q0, %b = %q1) {
          %i0, %i1 = qco.pow(%three) (%c = %a, %d = %b) {
            %x = qco.x %c : !qco.qubit -> !qco.qubit
            %s = qco.s %d : !qco.qubit -> !qco.qubit
            qco.yield %x, %s : !qco.qubit, !qco.qubit
          } : {!qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit}
          qco.yield %i1, %i0 : !qco.qubit, !qco.qubit
        } : {!qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit}
        return %o0, %o1 : !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  expectModifiers(*moduleOp, 0U, 0U, 2U);
}

TEST_F(QCOModifierYieldCanonicalizationTest, GatePowerRetainsYieldPermutation) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit) -> (!qco.qubit, !qco.qubit) {
        %two = arith.constant 2.0 : f64
        %angle = arith.constant 0.3 : f64
        %o0, %o1 = qco.pow(%two) (%a = %q0, %b = %q1) {
          %r0, %r1 = qco.rzx(%angle) %a, %b : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
          qco.yield %r1, %r0 : !qco.qubit, !qco.qubit
        } : {!qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit}
        return %o0, %o1 : !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  expectModifiers(*moduleOp, 0U, 0U, 1U);
}

TEST_F(QCOModifierYieldCanonicalizationTest, PowerOneInlinesYieldCycle) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit, %q2: !qco.qubit) -> (!qco.qubit, !qco.qubit, !qco.qubit) {
        %exponent = arith.constant 1.0 : f64
        %o0, %o1, %o2 = qco.pow(%exponent) (%a = %q0, %b = %q1, %c = %q2) {
          qco.yield %b, %c, %a : !qco.qubit, !qco.qubit, !qco.qubit
        } : {!qco.qubit, !qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit, !qco.qubit}
        return %o0, %o1, %o2 : !qco.qubit, !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  expectModifiers(*moduleOp, 0U, 0U, 0U);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto returned =
      cast<func::ReturnOp>(function.getBody().front().getTerminator());
  EXPECT_EQ(returned.getOperand(0), function.getArgument(1));
  EXPECT_EQ(returned.getOperand(1), function.getArgument(2));
  EXPECT_EQ(returned.getOperand(2), function.getArgument(0));
}

TEST_F(QCOModifierYieldCanonicalizationTest, PowerZeroErasesYieldCycle) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit, %q2: !qco.qubit) -> (!qco.qubit, !qco.qubit, !qco.qubit) {
        %exponent = arith.constant 0.0 : f64
        %o0, %o1, %o2 = qco.pow(%exponent) (%a = %q0, %b = %q1, %c = %q2) {
          qco.yield %b, %c, %a : !qco.qubit, !qco.qubit, !qco.qubit
        } : {!qco.qubit, !qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit, !qco.qubit}
        return %o0, %o1, %o2 : !qco.qubit, !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  expectModifiers(*moduleOp, 0U, 0U, 0U);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto returned =
      cast<func::ReturnOp>(function.getBody().front().getTerminator());
  EXPECT_EQ(returned.getOperand(0), function.getArgument(0));
  EXPECT_EQ(returned.getOperand(1), function.getArgument(1));
  EXPECT_EQ(returned.getOperand(2), function.getArgument(2));
}

TEST_F(QCOModifierYieldCanonicalizationTest, ZeroControlsInlineYieldCycle) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit, %q2: !qco.qubit) -> (!qco.qubit, !qco.qubit, !qco.qubit) {
        %o0, %o1, %o2 = "qco.ctrl"(%q0, %q1, %q2) <{
          operandSegmentSizes = array<i32: 0, 3>,
          resultSegmentSizes = array<i32: 0, 3>
        }> ({
        ^bb0(%a: !qco.qubit, %b: !qco.qubit, %c: !qco.qubit):
          qco.yield %b, %c, %a : !qco.qubit, !qco.qubit, !qco.qubit
        }) : (!qco.qubit, !qco.qubit, !qco.qubit) -> (!qco.qubit, !qco.qubit, !qco.qubit)
        return %o0, %o1, %o2 : !qco.qubit, !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  expectModifiers(*moduleOp, 0U, 0U, 0U);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto returned =
      cast<func::ReturnOp>(function.getBody().front().getTerminator());
  EXPECT_EQ(returned.getOperand(0), function.getArgument(1));
  EXPECT_EQ(returned.getOperand(1), function.getArgument(2));
  EXPECT_EQ(returned.getOperand(2), function.getArgument(0));
}

TEST_F(QCOModifierYieldCanonicalizationTest,
       DoubleInverseInlinesInnerYieldCycle) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit, %q2: !qco.qubit) -> (!qco.qubit, !qco.qubit, !qco.qubit) {
        %o0, %o1, %o2 = qco.inv(%a = %q0, %b = %q1, %c = %q2) {
          %i0, %i1, %i2 = qco.inv(%d = %a, %e = %b, %f = %c) {
            qco.yield %e, %f, %d : !qco.qubit, !qco.qubit, !qco.qubit
          } : {!qco.qubit, !qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit, !qco.qubit}
          qco.yield %i0, %i1, %i2 : !qco.qubit, !qco.qubit, !qco.qubit
        } : {!qco.qubit, !qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit, !qco.qubit}
        return %o0, %o1, %o2 : !qco.qubit, !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  expectModifiers(*moduleOp, 0U, 0U, 0U);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto returned =
      cast<func::ReturnOp>(function.getBody().front().getTerminator());
  EXPECT_EQ(returned.getOperand(0), function.getArgument(1));
  EXPECT_EQ(returned.getOperand(1), function.getArgument(2));
  EXPECT_EQ(returned.getOperand(2), function.getArgument(0));
}

TEST_F(QCOModifierYieldCanonicalizationTest,
       NestedPowerInlinesInnerYieldCycle) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit, %q2: !qco.qubit) -> (!qco.qubit, !qco.qubit, !qco.qubit) {
        %two = arith.constant 2.0 : f64
        %half = arith.constant 0.5 : f64
        %o0, %o1, %o2 = qco.pow(%two) (%a = %q0, %b = %q1, %c = %q2) {
          %i0, %i1, %i2 = qco.pow(%half) (%d = %a, %e = %b, %f = %c) {
            qco.yield %e, %f, %d : !qco.qubit, !qco.qubit, !qco.qubit
          } : {!qco.qubit, !qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit, !qco.qubit}
          qco.yield %i0, %i1, %i2 : !qco.qubit, !qco.qubit, !qco.qubit
        } : {!qco.qubit, !qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit, !qco.qubit}
        return %o0, %o1, %o2 : !qco.qubit, !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  expectModifiers(*moduleOp, 0U, 0U, 0U);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto returned =
      cast<func::ReturnOp>(function.getBody().front().getTerminator());
  EXPECT_EQ(returned.getOperand(0), function.getArgument(1));
  EXPECT_EQ(returned.getOperand(1), function.getArgument(2));
  EXPECT_EQ(returned.getOperand(2), function.getArgument(0));
}

TEST_F(QCOModifierYieldCanonicalizationTest,
       NestedControlMergesInnerYieldPermutation) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit, %q2: !qco.qubit, %q3: !qco.qubit) -> (!qco.qubit, !qco.qubit, !qco.qubit, !qco.qubit) {
        %o0, %o1, %o2, %o3 = qco.ctrl(%q0) targets(%a = %q1, %b = %q2, %c = %q3) {
          %i0, %i1, %i2 = qco.ctrl(%a) targets(%d = %b, %e = %c) {
            qco.yield %e, %d : !qco.qubit, !qco.qubit
          } : ({!qco.qubit}, {!qco.qubit, !qco.qubit}) -> ({!qco.qubit}, {!qco.qubit, !qco.qubit})
          qco.yield %i0, %i1, %i2 : !qco.qubit, !qco.qubit, !qco.qubit
        } : ({!qco.qubit}, {!qco.qubit, !qco.qubit, !qco.qubit}) -> ({!qco.qubit}, {!qco.qubit, !qco.qubit, !qco.qubit})
        return %o0, %o1, %o2, %o3 : !qco.qubit, !qco.qubit, !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  expectModifiers(*moduleOp, 1U, 0U, 0U);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto control = *function.getBody().getOps<CtrlOp>().begin();
  EXPECT_EQ(control.getNumControls(), 2U);
  EXPECT_EQ(control.getNumTargets(), 2U);
  auto yielded = cast<YieldOp>(control.getBody()->getTerminator());
  EXPECT_EQ(yielded.getOperand(0), control.getBody()->getArgument(1));
  EXPECT_EQ(yielded.getOperand(1), control.getBody()->getArgument(0));
}

TEST_F(QCOModifierYieldCanonicalizationTest,
       InverseControlMovesInnerYieldPermutation) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit, %q2: !qco.qubit) -> (!qco.qubit, !qco.qubit, !qco.qubit) {
        %o0, %o1, %o2 = qco.inv(%a = %q0, %b = %q1, %c = %q2) {
          %i0, %i1, %i2 = qco.ctrl(%a) targets(%d = %b, %e = %c) {
            qco.yield %e, %d : !qco.qubit, !qco.qubit
          } : ({!qco.qubit}, {!qco.qubit, !qco.qubit}) -> ({!qco.qubit}, {!qco.qubit, !qco.qubit})
          qco.yield %i0, %i1, %i2 : !qco.qubit, !qco.qubit, !qco.qubit
        } : {!qco.qubit, !qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit, !qco.qubit}
        return %o0, %o1, %o2 : !qco.qubit, !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  expectModifiers(*moduleOp, 1U, 1U, 0U);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto control = *function.getBody().getOps<CtrlOp>().begin();
  EXPECT_TRUE(llvm::hasSingleElement(control.getBody()->getOps<InvOp>()));
}

TEST_F(QCOModifierYieldCanonicalizationTest,
       PoweredControlMovesInnerYieldPermutation) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit, %q2: !qco.qubit) -> (!qco.qubit, !qco.qubit, !qco.qubit) {
        %half = arith.constant 0.5 : f64
        %o0, %o1, %o2 = qco.pow(%half) (%a = %q0, %b = %q1, %c = %q2) {
          %i0, %i1, %i2 = qco.ctrl(%a) targets(%d = %b, %e = %c) {
            qco.yield %e, %d : !qco.qubit, !qco.qubit
          } : ({!qco.qubit}, {!qco.qubit, !qco.qubit}) -> ({!qco.qubit}, {!qco.qubit, !qco.qubit})
          qco.yield %i0, %i1, %i2 : !qco.qubit, !qco.qubit, !qco.qubit
        } : {!qco.qubit, !qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit, !qco.qubit}
        return %o0, %o1, %o2 : !qco.qubit, !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  expectModifiers(*moduleOp, 1U, 0U, 1U);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto control = *function.getBody().getOps<CtrlOp>().begin();
  EXPECT_TRUE(llvm::hasSingleElement(control.getBody()->getOps<PowOp>()));
}

TEST_F(QCOModifierYieldCanonicalizationTest,
       InversePowerPreservesInnerYieldCycle) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit, %q2: !qco.qubit) -> (!qco.qubit, !qco.qubit, !qco.qubit) {
        %half = arith.constant 0.5 : f64
        %o0, %o1, %o2 = qco.inv(%a = %q0, %b = %q1, %c = %q2) {
          %i0, %i1, %i2 = qco.pow(%half) (%d = %a, %e = %b, %f = %c) {
            qco.yield %e, %f, %d : !qco.qubit, !qco.qubit, !qco.qubit
          } : {!qco.qubit, !qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit, !qco.qubit}
          qco.yield %i0, %i1, %i2 : !qco.qubit, !qco.qubit, !qco.qubit
        } : {!qco.qubit, !qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit, !qco.qubit}
        return %o0, %o1, %o2 : !qco.qubit, !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  expectModifiers(*moduleOp, 0U, 0U, 1U);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto power = *function.getBody().getOps<PowOp>().begin();
  const auto exponent = mlir::mqt::valueToDouble(power.getExponent());
  ASSERT_TRUE(exponent);
  EXPECT_DOUBLE_EQ(*exponent, -0.5);
  auto yielded = cast<YieldOp>(power.getBody()->getTerminator());
  EXPECT_EQ(yielded.getOperand(0), power.getBody()->getArgument(1));
  EXPECT_EQ(yielded.getOperand(1), power.getBody()->getArgument(2));
  EXPECT_EQ(yielded.getOperand(2), power.getBody()->getArgument(0));
}

TEST_F(QCOModifierYieldCanonicalizationTest,
       InverseDropsUnusedWireAroundYieldCycle) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit, %q2: !qco.qubit, %q3: !qco.qubit) -> (!qco.qubit, !qco.qubit, !qco.qubit, !qco.qubit) {
        %o0, %o1, %o2, %o3 = qco.inv(%a = %q0, %b = %q1, %c = %q2, %d = %q3) {
          qco.yield %b, %c, %a, %d : !qco.qubit, !qco.qubit, !qco.qubit, !qco.qubit
        } : {!qco.qubit, !qco.qubit, !qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit, !qco.qubit, !qco.qubit}
        return %o0, %o1, %o2, %o3 : !qco.qubit, !qco.qubit, !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  expectModifiers(*moduleOp, 0U, 1U, 0U);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto modifier = *function.getBody().getOps<InvOp>().begin();
  EXPECT_EQ(modifier.getNumTargets(), 3U);
  auto returned =
      cast<func::ReturnOp>(function.getBody().front().getTerminator());
  EXPECT_EQ(returned.getOperand(3), function.getArgument(3));
}

TEST_F(QCOModifierYieldCanonicalizationTest,
       PowerDropsUnusedWireAroundYieldCycle) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit, %q2: !qco.qubit, %q3: !qco.qubit) -> (!qco.qubit, !qco.qubit, !qco.qubit, !qco.qubit) {
        %two = arith.constant 2.0 : f64
        %o0, %o1, %o2, %o3 = qco.pow(%two) (%a = %q0, %b = %q1, %c = %q2, %d = %q3) {
          qco.yield %b, %c, %a, %d : !qco.qubit, !qco.qubit, !qco.qubit, !qco.qubit
        } : {!qco.qubit, !qco.qubit, !qco.qubit, !qco.qubit} -> {!qco.qubit, !qco.qubit, !qco.qubit, !qco.qubit}
        return %o0, %o1, %o2, %o3 : !qco.qubit, !qco.qubit, !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  expectModifiers(*moduleOp, 0U, 0U, 1U);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto modifier = *function.getBody().getOps<PowOp>().begin();
  EXPECT_EQ(modifier.getNumTargets(), 3U);
  auto returned =
      cast<func::ReturnOp>(function.getBody().front().getTerminator());
  EXPECT_EQ(returned.getOperand(3), function.getArgument(3));
}

TEST_F(QCOModifierYieldCanonicalizationTest,
       ControlDropsUnusedWireAroundYieldCycle) {
  auto moduleOp = canonicalize(R"mlir(
    module {
      func.func @test(%q0: !qco.qubit, %q1: !qco.qubit, %q2: !qco.qubit, %q3: !qco.qubit, %q4: !qco.qubit) -> (!qco.qubit, !qco.qubit, !qco.qubit, !qco.qubit, !qco.qubit) {
        %o0, %o1, %o2, %o3, %o4 = qco.ctrl(%q0) targets(%a = %q1, %b = %q2, %c = %q3, %d = %q4) {
          qco.yield %b, %c, %a, %d : !qco.qubit, !qco.qubit, !qco.qubit, !qco.qubit
        } : ({!qco.qubit}, {!qco.qubit, !qco.qubit, !qco.qubit, !qco.qubit}) -> ({!qco.qubit}, {!qco.qubit, !qco.qubit, !qco.qubit, !qco.qubit})
        return %o0, %o1, %o2, %o3, %o4 : !qco.qubit, !qco.qubit, !qco.qubit, !qco.qubit, !qco.qubit
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  expectModifiers(*moduleOp, 1U, 0U, 0U);
  auto function = moduleOp->lookupSymbol<func::FuncOp>("test");
  auto control = *function.getBody().getOps<CtrlOp>().begin();
  EXPECT_EQ(control.getNumTargets(), 3U);
  auto returned =
      cast<func::ReturnOp>(function.getBody().front().getTerminator());
  EXPECT_EQ(returned.getOperand(4), function.getArgument(4));
}

} // namespace
