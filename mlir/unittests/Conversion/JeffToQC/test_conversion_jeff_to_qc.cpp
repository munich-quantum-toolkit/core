/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/JeffToQC/JeffToQC.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"

#include <functional>
#include <gtest/gtest.h>
#include <jeff/IR/JeffDialect.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/Passes.h>
#include <string>

using namespace mlir;

class ConversionTest : public ::testing::Test {
protected:
  std::unique_ptr<mlir::MLIRContext> context;
  void SetUp() override {
    // Register all dialects needed for the full compilation pipeline
    DialectRegistry registry;
    registry.insert<jeff::JeffDialect, mlir::qc::QCDialect>();

    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  static std::string
  getOutputString(mlir::OwningOpRef<mlir::ModuleOp>& module_) {
    std::string outputString;
    llvm::raw_string_ostream outputStream(outputString);
    module_->print(outputStream);
    outputStream.flush();
    return outputString;
  }
};

TEST_F(ConversionTest, X) {
  const auto inputString = R"(
    %0 = jeff.qubit_alloc : !jeff.qubit
    %1 = jeff.x {is_adjoint = false, num_ctrls = 0 : i8, power = 1 : i8} %0 : !jeff.qubit
    %2 = jeff.x {is_adjoint = false, num_ctrls = 0 : i8, power = 1 : i8} %1 : !jeff.qubit
    jeff.qubit_free %2 : !jeff.qubit
  )";

  auto input = parseSourceString<ModuleOp>(inputString, context.get());
  if (!input) {
    FAIL() << "Failed to parse Jeff IR";
  }

  PassManager pm(context.get());
  pm.addPass(createJeffToQC());
  if (failed(pm.run(input.get()))) {
    FAIL() << "Error during Jeff-to-QC conversion";
  }

  const auto outputString = getOutputString(input);

  // ASSERT_EQ(outputString, "test");

  ASSERT_NE(outputString.find("qc.x"), std::string::npos);
}
