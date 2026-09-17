/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/Transforms/UnrollModifiers.h"
#include "mqt/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;

TEST(UnrollModifiersTest, NonCompositeControlsLeaveIRUnchanged) {
  for (const bool withGate : {false, true}) {
    SCOPED_TRACE(withGate);
    MLIRContext context;
    context.loadDialect<arith::ArithDialect, func::FuncDialect>();
    auto moduleOp = qco::QCOProgramBuilder::build(
        &context, [&](qco::QCOProgramBuilder& builder) {
          builder.ctrl(builder.staticQubit(0), builder.staticQubit(1),
                       [&](Value target) -> Value {
                         return withGate ? builder.x(target) : target;
                       });
          return builder.intConstant(0);
        });
    ASSERT_TRUE(moduleOp);
    ASSERT_TRUE(succeeded(verify(*moduleOp)));
    ASSERT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));
    auto original = OwningOpRef<ModuleOp>(moduleOp->clone());
    auto function = *moduleOp->getOps<func::FuncOp>().begin();
    auto control = *function.getOps<qco::CtrlOp>().begin();
    IRRewriter rewriter(&context);

    EXPECT_TRUE(failed(mlir::mqt::unrollControl(control, rewriter)));

    EXPECT_TRUE(OperationEquivalence::isEquivalentTo(
        *moduleOp, *original, OperationEquivalence::Flags::None));
    EXPECT_TRUE(succeeded(verify(*moduleOp)));
    EXPECT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));
  }
}
