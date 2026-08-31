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
 * @file test_qtensor_transforms.cpp
 * @brief Unit tests for QTensor dialect transformations.
 */

#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"
#include "mlir/Dialect/QTensor/Transforms/Passes.h"

#include <gtest/gtest.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
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

#include <cstdint>
#include <string>

using namespace mlir;

namespace {
TEST(QTensorTransformsTest, ShrinkToFitPreservesMetadata) {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect, mqt::MQTDialect,
                  qco::QCODialect, qtensor::QTensorDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main() {
        %c1 = arith.constant 1 : index
        %c3 = arith.constant 3 : index
        %reg = qtensor.alloc(%c3) {mqt.register_name = "q"}
            : tensor<3x!qco.qubit>
        %rest, %qubit = qtensor.extract %reg[%c1] : tensor<3x!qco.qubit>
        %rotated = qco.x %qubit : !qco.qubit -> !qco.qubit
        %updated = qtensor.insert %rotated into %rest[%c1]
            : tensor<3x!qco.qubit>
        qtensor.dealloc %updated : tensor<3x!qco.qubit>
        return
      }
    }
  )mlir",
                                              &context);
  ASSERT_TRUE(moduleOp);

  PassManager manager(&context);
  manager.addPass(qtensor::createShrinkQTensorToFitPass());
  ASSERT_TRUE(succeeded(manager.run(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  qtensor::AllocOp allocation;
  moduleOp->walk([&](qtensor::AllocOp op) { allocation = op; });
  ASSERT_TRUE(allocation);
  EXPECT_EQ(cast<RankedTensorType>(allocation.getType()).getShape(),
            ArrayRef<int64_t>{1});
  EXPECT_EQ(allocation->getAttrOfType<StringAttr>(
                mqt::MQTDialect::RegisterNameAttrHelper::getNameStr()),
            StringAttr::get(&context, "q"));
}

TEST(QTensorTransformsTest, HugeDeclaredTensorUsesSparseShrinkPlan) {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect, mqt::MQTDialect,
                  qco::QCODialect, qtensor::QTensorDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main() {
        %c7 = arith.constant 7 : index
        %huge = arith.constant 1099511627776 : index
        %reg = qtensor.alloc(%huge) : tensor<1099511627776x!qco.qubit>
        %rest, %qubit = qtensor.extract %reg[%c7]
            : tensor<1099511627776x!qco.qubit>
        %flipped = qco.x %qubit : !qco.qubit -> !qco.qubit
        %updated = qtensor.insert %flipped into %rest[%c7]
            : tensor<1099511627776x!qco.qubit>
        qtensor.dealloc %updated : tensor<1099511627776x!qco.qubit>
        return
      }
    }
  )mlir",
                                              &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  PassManager manager(&context);
  manager.addPass(qtensor::createShrinkQTensorToFitPass());
  ASSERT_TRUE(succeeded(manager.run(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  qtensor::AllocOp allocation;
  moduleOp->walk([&](qtensor::AllocOp op) { allocation = op; });
  ASSERT_TRUE(allocation);
  EXPECT_EQ(cast<RankedTensorType>(allocation.getType()).getShape(),
            ArrayRef<int64_t>{1});
}

TEST(QTensorTransformsTest, RejectsNonLinearChainWithoutMutation) {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect, mqt::MQTDialect,
                  qco::QCODialect, qtensor::QTensorDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @main() {
        %c1 = arith.constant 1 : index
        %c3 = arith.constant 3 : index
        %reg = qtensor.alloc(%c3) : tensor<3x!qco.qubit>
        %rest, %qubit = qtensor.extract %reg[%c1]
            : tensor<3x!qco.qubit>
        qtensor.dealloc %rest : tensor<3x!qco.qubit>
        qtensor.dealloc %rest : tensor<3x!qco.qubit>
        return
      }
    }
  )mlir",
                                              &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  std::string before;
  llvm::raw_string_ostream(before) << *moduleOp;

  PassManager manager(&context);
  manager.addPass(qtensor::createShrinkQTensorToFitPass());
  ASSERT_TRUE(succeeded(manager.run(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  std::string after;
  llvm::raw_string_ostream(after) << *moduleOp;
  EXPECT_EQ(after, before);
}
} // namespace
