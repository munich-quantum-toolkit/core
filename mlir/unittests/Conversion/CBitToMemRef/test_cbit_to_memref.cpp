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
 * @file test_cbit_to_memref.cpp
 * @brief Unit tests for the CBit-to-memref conversion.
 */

#include "mlir/Conversion/CBitToMemRef/CBitToMemRef.h"
#include "mlir/Dialect/CBit/IR/CBitDialect.h"
#include "mlir/Dialect/CBit/IR/CBitOps.h"

#include <gtest/gtest.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cstddef>
#include <memory>

using namespace mlir;

namespace {
class CBitToMemRefTest : public ::testing::Test {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    DialectRegistry registry;
    registry.insert<arith::ArithDialect, cbit::CBitDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
    context = std::make_unique<MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  [[nodiscard]] OwningOpRef<ModuleOp> convert(const StringRef source) const {
    auto moduleOp = parseSourceString<ModuleOp>(source, context.get());
    if (!moduleOp) {
      return {};
    }
    PassManager manager(context.get());
    manager.addPass(createConvertCBitToMemRef());
    if (failed(manager.run(*moduleOp))) {
      return {};
    }
    return moduleOp;
  }
};

TEST_F(CBitToMemRefTest, LowersInitializationLoadsAndStores) {
  auto moduleOp = convert(R"mlir(
    module {
      func.func @main() -> (!cbit.reg<2>, !cbit.reg<1>) {
        %c0 = arith.constant 0 : index
        %true = arith.constant true
        %zero = cbit.alloc(#cbit.init<zero>) : !cbit.reg<2>
        %undefined = cbit.alloc(#cbit.init<undefined>) : !cbit.reg<1>
        cbit.store %true, %undefined[%c0] : !cbit.reg<1>
        %bit = cbit.load %undefined[%c0] : !cbit.reg<1>
        return %zero, %undefined : !cbit.reg<2>, !cbit.reg<1>
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  EXPECT_TRUE(succeeded(verify(*moduleOp)));

  bool containsCBit = false;
  moduleOp->walk([&](cbit::AllocOp) { containsCBit = true; });
  EXPECT_FALSE(containsCBit);
  size_t allocations = 0;
  size_t stores = 0;
  size_t loads = 0;
  moduleOp->walk([&](memref::AllocOp) { ++allocations; });
  moduleOp->walk([&](memref::StoreOp) { ++stores; });
  moduleOp->walk([&](memref::LoadOp) { ++loads; });
  EXPECT_EQ(allocations, 2);
  EXPECT_EQ(stores, 3);
  EXPECT_EQ(loads, 1);
}

TEST_F(CBitToMemRefTest, ConvertsFunctionSignaturesCallsAndReturns) {
  auto moduleOp = convert(R"mlir(
    module {
      func.func private @identity(!cbit.reg<3>) -> !cbit.reg<3>
      func.func @main(%arg: !cbit.reg<3>) -> !cbit.reg<3> {
        %result = call @identity(%arg) : (!cbit.reg<3>) -> !cbit.reg<3>
        return %result : !cbit.reg<3>
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  EXPECT_TRUE(succeeded(verify(*moduleOp)));

  moduleOp->walk([&](Operation* op) {
    for (const Type type : op->getOperandTypes()) {
      EXPECT_FALSE(isa<cbit::RegisterType>(type));
    }
    for (const Type type : op->getResultTypes()) {
      EXPECT_FALSE(isa<cbit::RegisterType>(type));
    }
  });
  for (auto function : moduleOp->getOps<func::FuncOp>()) {
    EXPECT_FALSE(function.getFunctionType().getInputs().empty() &&
                 function.getFunctionType().getResults().empty());
    EXPECT_TRUE(llvm::all_of(function.getFunctionType().getInputs(),
                             [](Type type) { return isa<MemRefType>(type); }));
    EXPECT_TRUE(llvm::all_of(function.getFunctionType().getResults(),
                             [](Type type) { return isa<MemRefType>(type); }));
  }
}
} // namespace
