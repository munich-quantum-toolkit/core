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
 * @file test_cbit_ir.cpp
 * @brief Unit tests for the CBit MLIR dialect.
 */

#include "mlir/Dialect/CBit/IR/CBitDialect.h"
#include "mlir/Dialect/CBit/IR/CBitOps.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/Passes.h>

#include <memory>
#include <string>

using namespace mlir;

namespace {
class CBitIRTest : public ::testing::Test {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    DialectRegistry registry;
    registry
        .insert<arith::ArithDialect, cbit::CBitDialect, func::FuncDialect>();
    context = std::make_unique<MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  [[nodiscard]] OwningOpRef<ModuleOp> parse(const StringRef source) const {
    return parseSourceString<ModuleOp>(source, context.get());
  }
};

TEST_F(CBitIRTest, ParsesAndPrintsRegisterOperations) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main() -> !cbit.reg<2> {
        %c0 = arith.constant 0 : index
        %false = arith.constant false
        %reg = cbit.alloc(#cbit.init<zero>) source_name = "c" : !cbit.reg<2>
        cbit.store %false, %reg[%c0] : !cbit.reg<2>
        %bit = cbit.load %reg[%c0] : !cbit.reg<2>
        return %reg : !cbit.reg<2>
      }
    }
  )mlir");

  ASSERT_TRUE(moduleOp);
  EXPECT_TRUE(verify(*moduleOp).succeeded());

  std::string printed;
  llvm::raw_string_ostream stream(printed);
  moduleOp->print(stream);
  EXPECT_NE(printed.find("cbit.alloc(#cbit.init<zero>) source_name = \"c\""),
            std::string::npos)
      << printed;
  EXPECT_NE(printed.find("!cbit.reg<2>"), std::string::npos);
  EXPECT_NE(printed.find("cbit.store"), std::string::npos);
  EXPECT_NE(printed.find("cbit.load"), std::string::npos);
}

TEST_F(CBitIRTest, RejectsNonPositiveRegisterWidth) {
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main() {
        %reg = cbit.alloc(#cbit.init<undefined>) : !cbit.reg<0>
        return
      }
    }
  )mlir"));
}

TEST_F(CBitIRTest, RejectsConstantOutOfBoundsIndex) {
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main() {
        %c2 = arith.constant 2 : index
        %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<2>
        %bit = cbit.load %reg[%c2] : !cbit.reg<2>
        return
      }
    }
  )mlir"));
}

TEST_F(CBitIRTest, RejectsNegativeConstantIndex) {
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main() {
        %neg = arith.constant -1 : index
        %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<2>
        %bit = cbit.load %reg[%neg] : !cbit.reg<2>
        return
      }
    }
  )mlir"));
}

TEST_F(CBitIRTest, RejectsInvalidOperandTypes) {
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main() {
        %false = arith.constant false
        %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>
        %bit = "cbit.load"(%reg, %false)
            : (!cbit.reg<1>, i1) -> i1
        return
      }
    }
  )mlir"));
}

TEST_F(CBitIRTest, ReportsMemoryEffects) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main() {
        %c0 = arith.constant 0 : index
        %false = arith.constant false
        %reg = cbit.alloc(#cbit.init<undefined>) : !cbit.reg<1>
        cbit.store %false, %reg[%c0] : !cbit.reg<1>
        %bit = cbit.load %reg[%c0] : !cbit.reg<1>
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  cbit::AllocOp alloc;
  cbit::LoadOp load;
  cbit::StoreOp store;
  moduleOp->walk([&](cbit::AllocOp op) { alloc = op; });
  moduleOp->walk([&](cbit::LoadOp op) { load = op; });
  moduleOp->walk([&](cbit::StoreOp op) { store = op; });

  ASSERT_NE(alloc.getOperation(), nullptr);
  ASSERT_NE(load.getOperation(), nullptr);
  ASSERT_NE(store.getOperation(), nullptr);

  SmallVector<MemoryEffects::EffectInstance> effects;
  alloc.getEffects(effects);
  ASSERT_EQ(effects.size(), 1);
  EXPECT_TRUE(isa<MemoryEffects::Allocate>(effects.front().getEffect()));

  effects.clear();
  load.getEffects(effects);
  ASSERT_EQ(effects.size(), 1);
  EXPECT_TRUE(isa<MemoryEffects::Read>(effects.front().getEffect()));
  EXPECT_EQ(effects.front().getValue(), load.getReg());

  effects.clear();
  store.getEffects(effects);
  ASSERT_EQ(effects.size(), 1);
  EXPECT_TRUE(isa<MemoryEffects::Write>(effects.front().getEffect()));
  EXPECT_EQ(effects.front().getValue(), store.getReg());
}

TEST_F(CBitIRTest, ForwardsStraightLineStoresAndZeroInitialization) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main() -> (i1, i1) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %true = arith.constant true
        %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<2>
        %zero = cbit.load %reg[%c0] : !cbit.reg<2>
        cbit.store %true, %reg[%c1] : !cbit.reg<2>
        %stored = cbit.load %reg[%c1] : !cbit.reg<2>
        return %zero, %stored : i1, i1
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  PassManager canonicalizer(context.get());
  canonicalizer.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(canonicalizer.run(*moduleOp)));

  auto funcOp = *moduleOp->getOps<func::FuncOp>().begin();
  auto returnOp = *funcOp.getOps<func::ReturnOp>().begin();
  std::string canonicalized;
  llvm::raw_string_ostream canonicalizedStream(canonicalized);
  moduleOp->print(canonicalizedStream);
  APInt zero;
  APInt stored;
  EXPECT_TRUE(matchPattern(returnOp.getOperand(0), m_ConstantInt(&zero)))
      << canonicalized;
  EXPECT_TRUE(matchPattern(returnOp.getOperand(1), m_ConstantInt(&stored)))
      << canonicalized;
  EXPECT_TRUE(zero.isZero());
  EXPECT_TRUE(stored.isOne());
}

TEST_F(CBitIRTest, DoesNotForwardAcrossAnAmbiguousStore) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main(%dynamic: index) -> i1 {
        %c0 = arith.constant 0 : index
        %true = arith.constant true
        %reg = cbit.alloc(#cbit.init<zero>) : !cbit.reg<2>
        cbit.store %true, %reg[%dynamic] : !cbit.reg<2>
        %value = cbit.load %reg[%c0] : !cbit.reg<2>
        return %value : i1
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  PassManager canonicalizer(context.get());
  canonicalizer.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(canonicalizer.run(*moduleOp)));

  auto funcOp = *moduleOp->getOps<func::FuncOp>().begin();
  auto returnOp = *funcOp.getOps<func::ReturnOp>().begin();
  EXPECT_TRUE(returnOp.getOperand(0).getDefiningOp<cbit::LoadOp>());
}
} // namespace
