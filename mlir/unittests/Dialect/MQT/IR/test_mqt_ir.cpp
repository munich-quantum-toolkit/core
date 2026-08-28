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
 * @file test_mqt_ir.cpp
 * @brief Unit tests for the MQT metadata dialect.
 */

#include "mlir/Dialect/CBit/IR/CBitDialect.h"
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LLVM.h>

#include <memory>

using namespace mlir;

namespace {
class MQTIRTest : public ::testing::Test {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    DialectRegistry registry;
    registry.insert<arith::ArithDialect, cbit::CBitDialect, func::FuncDialect,
                    LLVM::LLVMDialect, memref::MemRefDialect, mqt::MQTDialect,
                    qc::QCDialect, qco::QCODialect, qtensor::QTensorDialect>();
    context = std::make_unique<MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  [[nodiscard]] OwningOpRef<ModuleOp> parse(const StringRef source) const {
    return parseSourceString<ModuleOp>(source, context.get());
  }
};

TEST_F(MQTIRTest, AcceptsProgramInputAndRegisterNames) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @qc(%theta: f64 {mqt.input_name = "theta[2]",
          mqt.parameter_group = {identity = "group-id", name = "theta",
                                 index = 2 : i64, size = 4 : i64}},
          %element: f64 {mqt.input_name = "[0]",
          mqt.parameter_group = {identity = "empty-name", name = "",
                                 index = 0 : i64, size = 1 : i64}}) {
        %reg = memref.alloc() {mqt.register_name = "q"}
            : memref<2x!qc.qubit>
        return
      }
      func.func @qco(%enabled: i1 {mqt.input_name = "enabled"}) {
        %c2 = arith.constant 2 : index
        %reg = qtensor.alloc(%c2) {mqt.register_name = "r"}
            : tensor<2x!qco.qubit>
        return
      }
      func.func @cbit() {
        %reg = cbit.alloc(#cbit.init<zero>) {mqt.register_name = "c"}
            : !cbit.reg<2>
        return
      }
      func.func @lowered_cbit() {
        %reg = memref.alloc() {mqt.register_name = "lowered"} : memref<2xi1>
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  EXPECT_TRUE(succeeded(mqt::verifyProgramMetadata(*moduleOp)));
}

TEST_F(MQTIRTest, ManagesAndFindsEntryPoint) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @helper() { return }
      func.func @main() attributes {mqt.entry_point} { return }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto main = mqt::getEntryPoint(*moduleOp);
  ASSERT_TRUE(main);
  EXPECT_EQ(main.getSymName(), "main");
  EXPECT_TRUE(mqt::isEntryPoint(main));

  mqt::removeEntryPoint(main);
  EXPECT_FALSE(mqt::isEntryPoint(main));
  EXPECT_FALSE(mqt::getEntryPoint(*moduleOp));

  auto helper = moduleOp->lookupSymbol<func::FuncOp>("helper");
  ASSERT_TRUE(helper);
  mqt::setEntryPoint(helper);
  EXPECT_TRUE(mqt::isEntryPoint(helper));
  EXPECT_EQ(mqt::getEntryPoint(*moduleOp), helper);
}

TEST_F(MQTIRTest, RejectsInvalidEntryPoints) {
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main() attributes {mqt.entry_point = "yes"} { return }
    }
  )mlir"));
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func private @main() attributes {mqt.entry_point}
    }
  )mlir"));
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main() {
        %c0 = "arith.constant"() {mqt.entry_point, value = 0 : i64}
            : () -> i64
        return
      }
    }
  )mlir"));
}

TEST_F(MQTIRTest, AcceptsDefinedLLVMEntryPoint) {
  auto llvmEntryPoint = parse(R"mlir(
    module {
      llvm.func @main() attributes {mqt.entry_point} {
        llvm.return
      }
    }
  )mlir");
  ASSERT_TRUE(llvmEntryPoint);
  EXPECT_TRUE(succeeded(mqt::verifyProgramMetadata(*llvmEntryPoint)));
}

TEST_F(MQTIRTest, ProgramMetadataRejectsDuplicateEntryPoints) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @first() attributes {mqt.entry_point} { return }
      func.func @second() attributes {mqt.entry_point} { return }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  EXPECT_TRUE(failed(mqt::verifyProgramMetadata(*moduleOp)));
}

TEST_F(MQTIRTest, ProgramMetadataRejectsNonFuncEntryPoint) {
  auto moduleOp = parse(R"mlir(
    module {
      memref.global "private" @storage : memref<1xi8>
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  auto global = moduleOp->lookupSymbol<memref::GlobalOp>("storage");
  ASSERT_TRUE(global);
  mqt::setEntryPoint(global);
  EXPECT_TRUE(failed(mqt::verifyProgramMetadata(*moduleOp)));
}

TEST_F(MQTIRTest, RejectsInvalidInputNames) {
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @empty(%arg: f64 {mqt.input_name = ""}) { return }
    }
  )mlir"));
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @null(%arg: f64 {mqt.input_name = "a\00b"}) { return }
    }
  )mlir"));
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @wrong_type(%arg: f64 {mqt.input_name = 1 : i64}) { return }
    }
  )mlir"));
}

TEST_F(MQTIRTest, ProgramMetadataRejectsDuplicateInputNames) {
  auto module = parse(R"mlir(
    module {
      func.func @main(%lhs: f64 {mqt.input_name = "theta"},
                      %rhs: i1 {mqt.input_name = "theta"}) {
        return
      }
    }
  )mlir");
  ASSERT_TRUE(module);
  EXPECT_TRUE(failed(mqt::verifyProgramMetadata(*module)));
}

TEST_F(MQTIRTest, RejectsInvalidInputGroups) {
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @incomplete(%arg: f64 {mqt.input_name = "theta[0]",
          mqt.parameter_group = {identity = "group"}}) { return }
    }
  )mlir"));
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @empty_identity(%arg: f64 {mqt.input_name = "theta[0]",
          mqt.parameter_group = {identity = "", name = "theta",
                                 index = 0 : i64, size = 1 : i64}}) { return }
    }
  )mlir"));
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @null_name(%arg: f64 {mqt.input_name = "theta[0]",
          mqt.parameter_group = {identity = "group", name = "theta\00",
                                 index = 0 : i64, size = 1 : i64}}) { return }
    }
  )mlir"));
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @wrong_integer_type(%arg: f64 {mqt.input_name = "theta[0]",
          mqt.parameter_group = {identity = "group", name = "theta",
                                 index = 0 : i32, size = 1 : i64}}) { return }
    }
  )mlir"));
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @wrong_name(%arg: f64 {mqt.input_name = "phi[0]",
          mqt.parameter_group = {identity = "group", name = "theta",
                                 index = 0 : i64, size = 1 : i64}}) { return }
    }
  )mlir"));
}

TEST_F(MQTIRTest, AcceptsParameterGroupsOutsideCurrentVectorSize) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main(
          %empty: f64 {mqt.input_name = "theta[0]",
            mqt.parameter_group = {identity = "empty-vector", name = "theta",
                                   index = 0 : i64, size = 0 : i64}},
          %shrunk: f64 {mqt.input_name = "phi[1]",
            mqt.parameter_group = {identity = "shrunk-vector", name = "phi",
                                   index = 1 : i64, size = 1 : i64}}) {
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  EXPECT_TRUE(succeeded(mqt::verifyProgramMetadata(*moduleOp)));
}

TEST_F(MQTIRTest, ProgramMetadataRejectsInconsistentParameterGroups) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main(
          %lhs: f64 {mqt.input_name = "theta[0]",
            mqt.parameter_group = {identity = "group", name = "theta",
                                   index = 0 : i64, size = 2 : i64}},
          %rhs: f64 {mqt.input_name = "phi[1]",
            mqt.parameter_group = {identity = "group", name = "phi",
                                   index = 1 : i64, size = 3 : i64}}) {
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);
  EXPECT_TRUE(failed(mqt::verifyProgramMetadata(*moduleOp)));
}

TEST_F(MQTIRTest, RejectsInputMetadataOnOperations) {
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main() {
        %c0 = "arith.constant"() {mqt.input_name = "theta", value = 0.0 : f64}
            : () -> f64
        return
      }
    }
  )mlir"));
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main() {
        %c0 = "arith.constant"() {
          mqt.parameter_group = {identity = "group", name = "theta",
                                 index = 0 : i64, size = 1 : i64},
          value = 0.0 : f64} : () -> f64
        return
      }
    }
  )mlir"));
}

TEST_F(MQTIRTest, RejectsInvalidRegisterNamesAndOwners) {
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @empty() {
        %reg = cbit.alloc(#cbit.init<zero>) {mqt.register_name = ""}
            : !cbit.reg<2>
        return
      }
    }
  )mlir"));
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main() {
        %reg = memref.alloc() {mqt.register_name = "values"}
            : memref<2xf64>
        return
      }
    }
  )mlir"));
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main(%arg: f64 {mqt.register_name = "q"}) {
        return
      }
    }
  )mlir"));
}

TEST_F(MQTIRTest, RejectsDuplicateProgramNames) {
  auto duplicateRegisters = parse(R"mlir(
    module {
      func.func @main() {
        %lhs = memref.alloc() {mqt.register_name = "state"}
            : memref<1x!qc.qubit>
        %rhs = cbit.alloc(#cbit.init<zero>) {mqt.register_name = "state"}
            : !cbit.reg<2>
        return
      }
    }
  )mlir");
  ASSERT_TRUE(duplicateRegisters);
  EXPECT_TRUE(failed(mqt::verifyProgramMetadata(*duplicateRegisters)));

  auto duplicateInputAndRegister = parse(R"mlir(
    module {
      func.func @main(%arg: f64 {mqt.input_name = "state"}) {
        %reg = cbit.alloc(#cbit.init<zero>) {mqt.register_name = "state"}
            : !cbit.reg<2>
        return
      }
    }
  )mlir");
  ASSERT_TRUE(duplicateInputAndRegister);
  EXPECT_TRUE(failed(mqt::verifyProgramMetadata(*duplicateInputAndRegister)));
}

TEST_F(MQTIRTest, RejectsUnknownMQTAttributes) {
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main() attributes {mqt.unknown} { return }
    }
  )mlir"));
}
} // namespace
