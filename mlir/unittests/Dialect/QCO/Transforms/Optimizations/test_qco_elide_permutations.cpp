/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/CBit/IR/CBitDialect.h"
#include "mqt/Dialect/CBit/IR/CBitOps.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"
#include "mqt/Dialect/QTensor/IR/QTensorDialect.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "Support/IRVerification.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::qco;

namespace {

class ElidePermutationsTest : public testing::Test {
protected:
  MLIRContext context_;

  void SetUp() override {
    context_.loadDialect<qtensor::QTensorDialect, cbit::CBitDialect,
                         arith::ArithDialect, func::FuncDialect>();
  }

  void run(ModuleOp moduleOp) {
    PassManager pm(&context_);
    pm.addPass(createElidePermutations());
    ASSERT_TRUE(succeeded(pm.run(moduleOp)));
    EXPECT_TRUE(succeeded(verify(moduleOp)));
    EXPECT_TRUE(succeeded(verifyLinearity(moduleOp)));
  }
};

TEST_F(ElidePermutationsTest,
       PreservesResourceSlotsAndMeasurementDestinations) {
  for (const bool earlyDisposal : {false, true}) {
    SCOPED_TRACE(earlyDisposal);
    auto moduleOp = parseSourceString<ModuleOp>(R"mlir(module {
      func.func @main() -> !cbit.reg<2> attributes {mqt.entry_point} {
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %bits = cbit.alloc(#cbit.init<zero>) : !cbit.reg<2>
        %left = qtensor.alloc(%one) : tensor<1x!qco.qubit>
        %b = qco.alloc : !qco.qubit
        %l, %a = qtensor.extract %left[%zero] : tensor<1x!qco.qubit>
        %sa, %sb = qco.swap %a, %b : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
        %ha = qco.h %sa : !qco.qubit -> !qco.qubit
        %qa, %ba = qco.measure %ha : !qco.qubit
        cbit.store %ba, %bits[%zero] : !cbit.reg<2>
        %lf = qtensor.insert %qa into %l[%zero] : tensor<1x!qco.qubit>
        %hb = qco.h %sb : !qco.qubit -> !qco.qubit
        %qb, %bb = qco.measure %hb : !qco.qubit
        cbit.store %bb, %bits[%one] : !cbit.reg<2>
        qco.sink %qb : !qco.qubit
        qtensor.dealloc %lf : tensor<1x!qco.qubit>
        return %bits : !cbit.reg<2>
      }
    })mlir",
                                                &context_);
    ASSERT_TRUE(moduleOp);
    auto function = moduleOp->lookupSymbol<func::FuncOp>("main");
    auto insert = *function.getOps<qtensor::InsertOp>().begin();
    if (earlyDisposal) {
      auto dealloc = *function.getOps<qtensor::DeallocOp>().begin();
      dealloc->moveAfter(insert);
      OwningOpRef<ModuleOp> original(moduleOp->clone());
      ASSERT_NO_FATAL_FAILURE(run(*moduleOp));
      EXPECT_TRUE(areModulesStructurallyEquivalent(*original, *moduleOp));
      continue;
    }
    auto swap = *function.getOps<SWAPOp>().begin();
    auto roots = llvm::to_vector(swap.getOperands());
    ASSERT_NO_FATAL_FAILURE(run(*moduleOp));
    EXPECT_TRUE(function.getOps<SWAPOp>().empty());
    for (auto [store, root] : llvm::zip_equal(function.getOps<cbit::StoreOp>(),
                                              llvm::reverse(roots))) {
      auto h = store.getValue()
                   .getDefiningOp<MeasureOp>()
                   .getQubitIn()
                   .getDefiningOp<HOp>();
      EXPECT_EQ(h.getQubitIn(), root);
    }
    auto sink = *function.getOps<SinkOp>().begin();
    for (auto [output, root] : llvm::zip_equal(
             SmallVector<Value>{insert.getScalar(), sink.getQubit()}, roots)) {
      auto h =
          output.getDefiningOp<MeasureOp>().getQubitIn().getDefiningOp<HOp>();
      EXPECT_EQ(h.getQubitIn(), root);
    }
  }
}

TEST_F(ElidePermutationsTest, PreservesPhysicalAndReusedTensorWires) {
  for (
      const auto* source : {
          R"mlir(func.func @main() -> !cbit.reg<2> attributes {mqt.entry_point} {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %bits = cbit.alloc(#cbit.init<zero>) : !cbit.reg<2>
      %a = qco.static 0 : !qco.qubit
      %b = qco.static 1 : !qco.qubit
      %sa, %sb = qco.swap %a, %b : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
      %qa, %ba = qco.measure %sa : !qco.qubit
      %qb, %bb = qco.measure %sb : !qco.qubit
      cbit.store %ba, %bits[%zero] : !cbit.reg<2>
      cbit.store %bb, %bits[%one] : !cbit.reg<2>
      qco.sink %qa : !qco.qubit
      qco.sink %qb : !qco.qubit
      return %bits : !cbit.reg<2>
    })mlir",
          R"mlir(func.func @main() -> !cbit.reg<2> attributes {mqt.entry_point} {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %two = arith.constant 2 : index
      %bits = cbit.alloc(#cbit.init<zero>) : !cbit.reg<2>
      %tensor = qtensor.alloc(%two) : tensor<2x!qco.qubit>
      %t0, %a = qtensor.extract %tensor[%zero] : tensor<2x!qco.qubit>
      %t1, %b = qtensor.extract %t0[%one] : tensor<2x!qco.qubit>
      %sa, %sb = qco.swap %a, %b : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
      %measured, %unused = qco.measure %sa : !qco.qubit
      %t2 = qtensor.insert %measured into %t1[%zero] : tensor<2x!qco.qubit>
      %t3, %again = qtensor.extract %t2[%zero] : tensor<2x!qco.qubit>
      %qa, %ba = qco.measure %again : !qco.qubit
      %qb, %bb = qco.measure %sb : !qco.qubit
      cbit.store %ba, %bits[%zero] : !cbit.reg<2>
      cbit.store %bb, %bits[%one] : !cbit.reg<2>
      %t4 = qtensor.insert %qa into %t3[%zero] : tensor<2x!qco.qubit>
      %t5 = qtensor.insert %qb into %t4[%one] : tensor<2x!qco.qubit>
      qtensor.dealloc %t5 : tensor<2x!qco.qubit>
      return %bits : !cbit.reg<2>
    })mlir",
      }) {
    SCOPED_TRACE(source);
    auto moduleOp = parseSourceString<ModuleOp>(source, &context_);
    ASSERT_TRUE(moduleOp);
    OwningOpRef<ModuleOp> original(moduleOp->clone());
    ASSERT_NO_FATAL_FAILURE(run(*moduleOp));
    EXPECT_TRUE(areModulesStructurallyEquivalent(*original, *moduleOp));
  }
}

} // namespace
