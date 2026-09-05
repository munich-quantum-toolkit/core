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
#include "mlir/Dialect/MQT/IR/MQTDialect.h"

#include <gtest/gtest.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
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
                    memref::MemRefDialect, mqt::MQTDialect, scf::SCFDialect>();
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
        %zero = cbit.alloc(#cbit.init<zero>) {mqt.register_name = "result"}
            : !cbit.reg<2>
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
  StringAttr registerName;
  size_t stores = 0;
  size_t loads = 0;
  moduleOp->walk([&](memref::AllocOp alloc) {
    ++allocations;
    if (const auto name = alloc->getAttrOfType<StringAttr>(
            mqt::MQTDialect::RegisterNameAttrHelper::getNameStr())) {
      registerName = name;
    }
  });
  moduleOp->walk([&](memref::StoreOp) { ++stores; });
  moduleOp->walk([&](memref::LoadOp) { ++loads; });
  EXPECT_EQ(allocations, 2);
  EXPECT_EQ(stores, 2);
  EXPECT_EQ(loads, 1);
  ASSERT_TRUE(registerName);
  EXPECT_EQ(registerName.getValue(), "result");
}

TEST_F(CBitToMemRefTest, LargeZeroInitializationProducesBoundedIR) {
  auto moduleOp = convert(R"mlir(
    module {
      func.func @main() {
        %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1000000000>
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  SmallVector<scf::ForOp> loops;
  moduleOp->walk([&](scf::ForOp loop) { loops.emplace_back(loop); });
  ASSERT_EQ(loops.size(), 1);
  EXPECT_EQ(getConstantIntValue(loops.front().getLowerBound()), 0);
  EXPECT_EQ(getConstantIntValue(loops.front().getUpperBound()), 1000000000);
  EXPECT_EQ(getConstantIntValue(loops.front().getStep()), 1);

  size_t stores = 0;
  loops.front().getBody()->walk([&](memref::StoreOp) { ++stores; });
  EXPECT_EQ(stores, 1);
}

TEST_F(CBitToMemRefTest, LowersRegisterComparisons) {
  auto moduleOp = convert(R"mlir(
    module {
      func.func @main() -> (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1) {
        %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<3>
        %eq_read = cbit.read %reg : !cbit.reg<3> -> i3
        %eq_rhs = arith.constant 5 : i3
        %eq = arith.cmpi eq, %eq_read, %eq_rhs : i3
        %ne_read = cbit.read %reg : !cbit.reg<3> -> i3
        %ne_rhs = arith.constant 5 : i3
        %ne = arith.cmpi ne, %ne_read, %ne_rhs : i3
        %ult_read = cbit.read %reg : !cbit.reg<3> -> i3
        %ult_rhs = arith.constant 5 : i3
        %ult = arith.cmpi ult, %ult_read, %ult_rhs : i3
        %ule_read = cbit.read %reg : !cbit.reg<3> -> i3
        %ule_rhs = arith.constant 5 : i3
        %ule = arith.cmpi ule, %ule_read, %ule_rhs : i3
        %ugt_read = cbit.read %reg : !cbit.reg<3> -> i3
        %ugt_rhs = arith.constant 5 : i3
        %ugt = arith.cmpi ugt, %ugt_read, %ugt_rhs : i3
        %uge_read = cbit.read %reg : !cbit.reg<3> -> i3
        %uge_rhs = arith.constant 5 : i3
        %uge = arith.cmpi uge, %uge_read, %uge_rhs : i3
        %slt_read = cbit.read %reg : !cbit.reg<3> -> i3
        %slt_rhs = arith.constant 5 : i3
        %slt = arith.cmpi slt, %slt_read, %slt_rhs : i3
        %sle_read = cbit.read %reg : !cbit.reg<3> -> i3
        %sle_rhs = arith.constant 5 : i3
        %sle = arith.cmpi sle, %sle_read, %sle_rhs : i3
        %sgt_read = cbit.read %reg : !cbit.reg<3> -> i3
        %sgt_rhs = arith.constant 5 : i3
        %sgt = arith.cmpi sgt, %sgt_read, %sgt_rhs : i3
        %sge_read = cbit.read %reg : !cbit.reg<3> -> i3
        %sge_rhs = arith.constant 5 : i3
        %sge = arith.cmpi sge, %sge_read, %sge_rhs : i3
        return %eq, %ne, %ult, %ule, %ugt, %uge, %slt, %sle, %sgt, %sge
            : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  EXPECT_TRUE(succeeded(verify(*moduleOp)));

  bool containsCBit = false;
  moduleOp->walk([&](Operation* op) {
    containsCBit |= op->getDialect() == context->getLoadedDialect("cbit");
  });
  EXPECT_FALSE(containsCBit);
  size_t loads = 0;
  moduleOp->walk([&](memref::LoadOp) { ++loads; });
  EXPECT_GT(loads, 0);
}

TEST_F(CBitToMemRefTest, LowersWholeRegisterReadsAndWrites) {
  auto moduleOp = convert(R"mlir(
    module {
      func.func @main() -> i3 {
        %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<3>
        %value = cbit.read %reg : !cbit.reg<3> -> i3
        cbit.write %value, %reg : i3, !cbit.reg<3>
        return %value : i3
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  EXPECT_TRUE(succeeded(verify(*moduleOp)));

  size_t loads = 0;
  size_t stores = 0;
  bool containsRead = false;
  bool containsWrite = false;
  moduleOp->walk([&](memref::LoadOp) { ++loads; });
  moduleOp->walk([&](memref::StoreOp) { ++stores; });
  moduleOp->walk([&](cbit::ReadOp) { containsRead = true; });
  moduleOp->walk([&](cbit::WriteOp) { containsWrite = true; });
  EXPECT_GT(loads, 0);
  EXPECT_GT(stores, 0);
  EXPECT_FALSE(containsRead);
  EXPECT_FALSE(containsWrite);
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
