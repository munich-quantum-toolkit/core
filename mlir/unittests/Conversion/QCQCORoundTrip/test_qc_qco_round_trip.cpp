/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/QCOToQC/QCOToQC.h"
#include "mlir/Conversion/QCToQCO/QCToQCO.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;

namespace {

class QCQCORoundTripTest : public testing::Test {
protected:
  MLIRContext context;

  QCQCORoundTripTest() {
    DialectRegistry registry;
    registry
        .insert<qc::QCDialect, qco::QCODialect, arith::ArithDialect,
                func::FuncDialect, memref::MemRefDialect, scf::SCFDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  }

  [[nodiscard]] LogicalResult runRoundTrip(ModuleOp module) {
    PassManager pm(&context);
    pm.addPass(createQCToQCO());
    pm.addPass(createQCOToQC());
    return pm.run(module);
  }

  static void expectNoScratchStorage(ModuleOp module) {
    bool containsScratchStorage = false;
    module.walk([&](Operation* operation) {
      containsScratchStorage |=
          isa<memref::AllocaOp, memref::LoadOp, memref::StoreOp>(operation);
    });
    EXPECT_FALSE(containsScratchStorage);
  }
};

} // namespace

TEST_F(QCQCORoundTripTest, PreservesClassicalIfResultWithoutScratch) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main(%condition: i1) -> i64
      attributes {passthrough = ["entry_point"]} {
    %q = qc.alloc : !qc.qubit
    %result = scf.if %condition -> i64 {
      qc.h %q : !qc.qubit
      %then = arith.constant 1 : i64
      scf.yield %then : i64
    } else {
      qc.x %q : !qc.qubit
      %else = arith.constant 2 : i64
      scf.yield %else : i64
    }
    qc.dealloc %q : !qc.qubit
    return %result : i64
  }
}
)mlir";

  auto module = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  ASSERT_TRUE(succeeded(runRoundTrip(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));

  scf::IfOp ifOp;
  module->walk([&](scf::IfOp candidate) { ifOp = candidate; });
  ASSERT_TRUE(ifOp);
  ASSERT_EQ(ifOp.getNumResults(), 1);

  auto main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  auto returnOp = cast<func::ReturnOp>(main.getBody().front().getTerminator());
  EXPECT_EQ(returnOp.getOperand(0), ifOp.getResult(0));
  expectNoScratchStorage(*module);
}

TEST_F(QCQCORoundTripTest, PreservesClassicalIndexSwitchResultWithoutScratch) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main(%index: index) -> i64
      attributes {passthrough = ["entry_point"]} {
    %q = qc.alloc : !qc.qubit
    %result = scf.index_switch %index -> i64
    case 0 {
      qc.h %q : !qc.qubit
      %case = arith.constant 1 : i64
      scf.yield %case : i64
    }
    default {
      qc.x %q : !qc.qubit
      %default = arith.constant 2 : i64
      scf.yield %default : i64
    }
    qc.dealloc %q : !qc.qubit
    return %result : i64
  }
}
)mlir";

  auto module = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  ASSERT_TRUE(succeeded(runRoundTrip(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));

  scf::IndexSwitchOp switchOp;
  module->walk([&](scf::IndexSwitchOp candidate) { switchOp = candidate; });
  ASSERT_TRUE(switchOp);
  ASSERT_EQ(switchOp.getNumResults(), 1);

  auto main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  auto returnOp = cast<func::ReturnOp>(main.getBody().front().getTerminator());
  EXPECT_EQ(returnOp.getOperand(0), switchOp.getResult(0));
  expectNoScratchStorage(*module);
}
