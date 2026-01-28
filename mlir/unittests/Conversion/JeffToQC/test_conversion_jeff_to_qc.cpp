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

#include <gtest/gtest.h>
#include <jeff/IR/JeffDialect.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <string>

using namespace mlir;

class JeffToQCConversionTest : public ::testing::Test {
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

  static std::string getOutputString(mlir::OwningOpRef<mlir::ModuleOp>& mod) {
    std::string outputString;
    llvm::raw_string_ostream outputStream(outputString);
    mod->print(outputStream);
    outputStream.flush();
    return outputString;
  }
};

TEST_F(JeffToQCConversionTest, Id) {
  const auto* const inputString = R"(
    %q0_0 = jeff.qubit_alloc : !jeff.qubit
    %q0_1 = jeff.i {is_adjoint = false, num_ctrls = 0 : i8, power = 1 : i8} %q0_0 : !jeff.qubit
    jeff.qubit_free %q0_1 : !jeff.qubit
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

  ASSERT_NE(outputString.find("qc.id"), std::string::npos);
}

TEST_F(JeffToQCConversionTest, X) {
  const auto* const inputString = R"(
    %q0_0 = jeff.qubit_alloc : !jeff.qubit
    %q0_1 = jeff.x {is_adjoint = false, num_ctrls = 0 : i8, power = 1 : i8} %q0_0 : !jeff.qubit
    %q0_2 = jeff.x {is_adjoint = false, num_ctrls = 0 : i8, power = 1 : i8} %q0_1 : !jeff.qubit
    jeff.qubit_free %q0_2 : !jeff.qubit
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

TEST_F(JeffToQCConversionTest, Y) {
  const auto* const inputString = R"(
    %q0_0 = jeff.qubit_alloc : !jeff.qubit
    %q0_1 = jeff.y {is_adjoint = false, num_ctrls = 0 : i8, power = 1 : i8} %q0_0 : !jeff.qubit
    jeff.qubit_free %q0_1 : !jeff.qubit
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

  ASSERT_NE(outputString.find("qc.y"), std::string::npos);
}

TEST_F(JeffToQCConversionTest, Z) {
  const auto* const inputString = R"(
    %q0_0 = jeff.qubit_alloc : !jeff.qubit
    %q0_1 = jeff.z {is_adjoint = false, num_ctrls = 0 : i8, power = 1 : i8} %q0_0 : !jeff.qubit
    jeff.qubit_free %q0_1 : !jeff.qubit
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

  ASSERT_NE(outputString.find("qc.z"), std::string::npos);
}

TEST_F(JeffToQCConversionTest, H) {
  const auto* const inputString = R"(
    %q0_0 = jeff.qubit_alloc : !jeff.qubit
    %q0_1 = jeff.h {is_adjoint = false, num_ctrls = 0 : i8, power = 1 : i8} %q0_0 : !jeff.qubit
    jeff.qubit_free %q0_1 : !jeff.qubit
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

  ASSERT_NE(outputString.find("qc.h"), std::string::npos);
}

TEST_F(JeffToQCConversionTest, S) {
  const auto* const inputString = R"(
    %q0_0 = jeff.qubit_alloc : !jeff.qubit
    %q0_1 = jeff.s {is_adjoint = false, num_ctrls = 0 : i8, power = 1 : i8} %q0_0 : !jeff.qubit
    jeff.qubit_free %q0_1 : !jeff.qubit
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

  ASSERT_NE(outputString.find("qc.s"), std::string::npos);
}

TEST_F(JeffToQCConversionTest, Sdg) {
  const auto* const inputString = R"(
    %q0_0 = jeff.qubit_alloc : !jeff.qubit
    %q0_1 = jeff.s {is_adjoint = true, num_ctrls = 0 : i8, power = 1 : i8} %q0_0 : !jeff.qubit
    jeff.qubit_free %q0_1 : !jeff.qubit
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

  ASSERT_NE(outputString.find("qc.sdg"), std::string::npos);
}

TEST_F(JeffToQCConversionTest, T) {
  const auto* const inputString = R"(
    %q0_0 = jeff.qubit_alloc : !jeff.qubit
    %q0_1 = jeff.t {is_adjoint = false, num_ctrls = 0 : i8, power = 1 : i8} %q0_0 : !jeff.qubit
    jeff.qubit_free %q0_1 : !jeff.qubit
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

  ASSERT_NE(outputString.find("qc.t"), std::string::npos);
}

TEST_F(JeffToQCConversionTest, Tdg) {
  const auto* const inputString = R"(
    %q0_0 = jeff.qubit_alloc : !jeff.qubit
    %q0_1 = jeff.t {is_adjoint = true, num_ctrls = 0 : i8, power = 1 : i8} %q0_0 : !jeff.qubit
    jeff.qubit_free %q0_1 : !jeff.qubit
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

  ASSERT_NE(outputString.find("qc.tdg"), std::string::npos);
}

TEST_F(JeffToQCConversionTest, Bell) {
  const auto* const inputString = R"(
    %q0_0 = jeff.qubit_alloc : !jeff.qubit
    %q1_0 = jeff.qubit_alloc : !jeff.qubit
    %q0_1 = jeff.h {is_adjoint = false, num_ctrls = 0 : i8, power = 1 : i8} %q0_0 : !jeff.qubit
    %q1_1, %q0_2 = jeff.x {is_adjoint = false, num_ctrls = 1 : i8, power = 1 : i8} %q1_0 ctrls(%q0_1) : !jeff.qubit ctrls !jeff.qubit
    jeff.qubit_free %q0_2 : !jeff.qubit
    jeff.qubit_free %q1_1 : !jeff.qubit
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

  ASSERT_NE(outputString.find("qc.h"), std::string::npos);
  ASSERT_NE(outputString.find("qc.ctrl"), std::string::npos);
  ASSERT_NE(outputString.find("qc.x"), std::string::npos);
}
