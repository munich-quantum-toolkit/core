/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"
#include "mqt/Support/Passes.h"

#include "VerificationTest.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"

#include <cstddef>
#include <utility>

using namespace mlir;
using namespace mlir::qco;

namespace {
class CombineControlledGatesTest : public testing::Test {
protected:
  MLIRContext context;

  void SetUp() override {
    context.loadDialect<QCODialect, arith::ArithDialect, func::FuncDialect>();
  }

  static size_t countControls(ModuleOp moduleOp) {
    size_t count = 0;
    moduleOp.walk([&](CtrlOp) { ++count; });
    return count;
  }

  void combine(ModuleOp moduleOp, bool cleanup = true) {
    ASSERT_TRUE(succeeded(verify(moduleOp)));
    ASSERT_TRUE(succeeded(verifyLinearity(moduleOp)));
    PassManager pm(&context);
    pm.addPass(createCombineControlledGates());
    if (cleanup) {
      pm.addPass(createCanonicalizerPass());
    }
    ASSERT_TRUE(succeeded(pm.run(moduleOp)));
    ASSERT_TRUE(succeeded(verify(moduleOp)));
    ASSERT_TRUE(succeeded(verifyLinearity(moduleOp)));
  }
};
} // namespace

TEST_F(CombineControlledGatesTest, DefaultPipelineMergesParameterizedBodies) {
  auto moduleOp =
      QCOProgramBuilder::build(&context, [](QCOProgramBuilder& builder) {
        auto control = builder.staticQubit(0);
        auto target = builder.staticQubit(1);
        auto [c, t] = builder.ctrl(control, target,
                                   [&](Value q) { return builder.rz(0.2, q); });
        builder.ctrl(c, t, [&](Value q) { return builder.rz(0.3, q); });
        return SmallVector<Value>{};
      });
  ASSERT_TRUE(moduleOp);
  OwningOpRef<ModuleOp> original(moduleOp->clone());
  PassManager pm(&context);
  populateDefaultQCOOptimizationPipeline(pm);
  pm.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(pm.run(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(verifyLinearity(*moduleOp)));
  EXPECT_EQ(countControls(*moduleOp), 1U);
  moduleOp->walk([](CtrlOp op) { EXPECT_EQ(op.getNumBodyUnitaries(), 1U); });
  ::mqt::test::expectUnitaryEquivalent(*original, *moduleOp, 2);
}

TEST_F(CombineControlledGatesTest, RespectsWireOrderPartitionAndUses) {
  for (const auto* boundary : {
           "control order",
           "target order",
           "partition",
           "target gate",
           "control gate",
           "barrier",
       }) {
    SCOPED_TRACE(boundary);
    auto moduleOp =
        QCOProgramBuilder::build(&context, [&](QCOProgramBuilder& builder) {
          SmallVector<Value> wires;
          for (size_t i = 0; i < 4; ++i) {
            wires.push_back(builder.staticQubit(i));
          }
          const auto body = [&](ValueRange args) {
            SmallVector<Value> outputs;
            for (auto q : args) {
              outputs.push_back(builder.h(q));
            }
            return outputs;
          };
          auto [controlsOut, targetsOut] =
              builder.ctrl(ValueRange(wires).take_front(2),
                           ValueRange(wires).slice(2, 2), body);
          llvm::copy(controlsOut, wires.begin());
          llvm::copy(targetsOut, wires.begin() + 2);
          const StringRef kind(boundary);
          if (kind == "control order") {
            std::swap(wires[0], wires[1]);
          } else if (kind == "target order") {
            std::swap(wires[2], wires[3]);
          } else if (kind == "target gate") {
            wires[2] = builder.x(wires[2]);
          } else if (kind == "control gate") {
            wires[0] = builder.x(wires[0]);
          } else if (kind == "barrier") {
            wires[2] = builder.barrier(ValueRange{wires[2]}).front();
          }
          const size_t controls = kind == "partition" ? 1 : 2;
          builder.ctrl(ValueRange(wires).take_front(controls),
                       ValueRange(wires).slice(controls, 4 - controls), body);
          return SmallVector<Value>{};
        });
    ASSERT_TRUE(moduleOp);
    OwningOpRef<ModuleOp> original(moduleOp->clone());
    ASSERT_NO_FATAL_FAILURE(combine(*moduleOp, false));
    EXPECT_EQ(countControls(*moduleOp), 2U);
    ::mqt::test::expectUnitaryEquivalent(*original, *moduleOp, 4);
  }
}

TEST_F(CombineControlledGatesTest, PreservesPhaseWiresAndParameterDominance) {
  Value parameter;
  auto moduleOp =
      QCOProgramBuilder::build(&context, [&](QCOProgramBuilder& builder) {
        parameter = builder.floatConstant(0.2);
        auto [controls, targets] = builder.ctrl(
            ValueRange{builder.staticQubit(0)},
            ValueRange{builder.staticQubit(1), builder.staticQubit(2)},
            [&](ValueRange args) {
              builder.gphase(0.125);
              return SmallVector<Value>{
                  builder.rx(parameter, args[0]),
                  builder.h(args[1]),
              };
            });
        builder.barrier(ValueRange{builder.x(builder.staticQubit(3))});
        auto opposite = arith::NegFOp::create(builder, parameter);
        builder.ctrl(controls, targets, [&](ValueRange args) {
          builder.gphase(0.125);
          return SmallVector<Value>{
              builder.ry(opposite, args[0]),
              builder.h(args[1]),
          };
        });
        return SmallVector<Value>{};
      });
  ASSERT_TRUE(moduleOp);
  auto function = *moduleOp->getOps<func::FuncOp>().begin();
  ASSERT_TRUE(succeeded(function.insertArgument(0, Float64Type::get(&context),
                                                {}, function.getLoc())));
  parameter.replaceAllUsesWith(function.getArgument(0));
  OwningOpRef<ModuleOp> original(moduleOp->clone());
  ASSERT_NO_FATAL_FAILURE(combine(*moduleOp));
  EXPECT_EQ(countControls(*moduleOp), 1U);
  moduleOp->walk([](HOp) { ADD_FAILURE() << "uncancelled Hadamard"; });
  for (const double theta : {0.73, -2.9, 6.28}) {
    OwningOpRef<ModuleOp> expected(original->clone());
    OwningOpRef<ModuleOp> actual(moduleOp->clone());
    for (auto mod : {expected.get(), actual.get()}) {
      auto func = *mod.getOps<func::FuncOp>().begin();
      OpBuilder builder(&func.getBody().front(),
                        func.getBody().front().begin());
      auto value = arith::ConstantOp::create(builder, func.getLoc(),
                                             builder.getF64FloatAttr(theta));
      func.getArgument(0).replaceAllUsesWith(value);
      PassManager fold(&context);
      fold.addPass(createCanonicalizerPass());
      ASSERT_TRUE(succeeded(fold.run(mod)));
    }
    ::mqt::test::expectUnitaryEquivalent(*expected, *actual, 4);
  }
}
