/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/**
 * @file test_qc_transforms.cpp
 * @brief Unit tests for QC dialect transformations.
 */

#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/Transforms/Passes.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>

using namespace mlir;

namespace {
TEST(QCTransformsTest, ShrinkQubitRegistersPreservesMetadata) {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                  mqt::MQTDialect, qc::QCDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main() {
        %c1 = arith.constant 1 : index
        %reg = memref.alloc() {mqt.register_name = "q"}
            : memref<3x!qc.qubit>
        %qubit = memref.load %reg[%c1] : memref<3x!qc.qubit>
        qc.x %qubit : !qc.qubit
        memref.dealloc %reg : memref<3x!qc.qubit>
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
  EXPECT_EQ(allocation.getType().getShape(), ArrayRef<int64_t>{1});
  EXPECT_EQ(allocation->getAttrOfType<StringAttr>(
                mqt::MQTDialect::RegisterNameAttrHelper::getNameStr()),
            StringAttr::get(&context, "q"));
}
} // namespace
