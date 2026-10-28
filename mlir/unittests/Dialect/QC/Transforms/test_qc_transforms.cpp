/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file test_qc_transforms.cpp
/// Unit tests for QC dialect transformations.

#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QC/IR/QCDialect.h"
#include "mqt/Dialect/QC/Transforms/Passes.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>

using namespace mlir;

namespace {
TEST(QCTransformsTest, ShrinkQubitRegistersPreservesWiresAndMetadata) {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                  mqt::MQTDialect, qc::QCDialect, scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main() {
        %c0 = arith.constant 0 : index
        %c99 = arith.constant 99 : index
        %reg = memref.alloc() {mqt.register_name = "q"}
            : memref<100x!qc.qubit>
        %high = memref.load %reg[%c99] : memref<100x!qc.qubit>
        qc.x %high : !qc.qubit
        %low = memref.load %reg[%c0] : memref<100x!qc.qubit>
        qc.h %low : !qc.qubit
        %again = memref.load %reg[%c99] : memref<100x!qc.qubit>
        qc.z %again : !qc.qubit
        scf.execute_region {
          memref.dealloc %reg : memref<100x!qc.qubit>
          scf.yield
        }
        return
      }
    }
  )mlir",
                                              &context);
  ASSERT_TRUE(moduleOp);

  PassManager manager(&context);
  manager.addPass(qc::createShrinkQubitRegistersPass());
  ASSERT_TRUE(succeeded(manager.run(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  memref::AllocOp allocation;
  moduleOp->walk([&](memref::AllocOp op) { allocation = op; });
  ASSERT_TRUE(allocation);
  EXPECT_EQ(allocation.getType().getShape(), ArrayRef<int64_t>{2});
  EXPECT_EQ(allocation->getAttrOfType<StringAttr>(
                mqt::MQTDialect::RegisterNameAttrHelper::getNameStr()),
            StringAttr::get(&context, "q"));
  SmallVector<int64_t> indices;
  moduleOp->walk([&](memref::LoadOp op) {
    const auto index = getConstantIntValue(op.getIndices().front());
    ASSERT_TRUE(index);
    indices.push_back(*index);
  });
  EXPECT_EQ(indices, (SmallVector<int64_t>{1, 0, 1}));
}

TEST(QCTransformsTest, ShrinkRequiresFoldedIndices) {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                  mqt::MQTDialect, qc::QCDialect, scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main() {
        %c1 = arith.constant 1 : index
        %index = arith.addi %c1, %c1 : index
        %reg = memref.alloc() : memref<4x!qc.qubit>
        %qubit = memref.load %reg[%index] : memref<4x!qc.qubit>
        qc.x %qubit : !qc.qubit
        memref.dealloc %reg : memref<4x!qc.qubit>
        return
      }
    }
  )mlir",
                                              &context);
  ASSERT_TRUE(moduleOp);

  PassManager shrink(&context);
  shrink.addPass(qc::createShrinkQubitRegistersPass());
  ASSERT_TRUE(succeeded(shrink.run(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  memref::AllocOp allocation;
  moduleOp->walk([&](memref::AllocOp op) { allocation = op; });
  ASSERT_TRUE(allocation);
  EXPECT_EQ(allocation.getType().getShape(), ArrayRef<int64_t>{4});
  moduleOp->walk([&](memref::LoadOp op) {
    EXPECT_TRUE(op.getIndices().front().getDefiningOp<arith::AddIOp>());
  });

  PassManager prepared(&context);
  prepared.addPass(createCanonicalizerPass());
  prepared.addPass(qc::createShrinkQubitRegistersPass());
  ASSERT_TRUE(succeeded(prepared.run(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  allocation = {};
  moduleOp->walk([&](memref::AllocOp op) { allocation = op; });
  ASSERT_TRUE(allocation);
  EXPECT_EQ(allocation.getType().getShape(), ArrayRef<int64_t>{1});
  moduleOp->walk([&](memref::LoadOp op) {
    EXPECT_EQ(getConstantIntValue(op.getIndices().front()), 0);
  });
}
} // namespace
