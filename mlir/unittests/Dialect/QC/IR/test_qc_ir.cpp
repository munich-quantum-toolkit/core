/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "test_qc_ir.h"

#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Support/IRVerification.h"
#include "mlir/Support/Passes.h"
#include "mlir/Support/PrettyPrinting.h"

#include <gtest/gtest.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Verifier.h>

using namespace mlir;

void QCTest::SetUp() {
  // Register all necessary dialects
  DialectRegistry registry;
  registry
      .insert<mlir::qc::QCDialect, arith::ArithDialect, func::FuncDialect>();
  context = std::make_unique<MLIRContext>();
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();

  emptyQC = mlir::qc::QCProgramBuilder::build(context.get(), [](auto& b) {});
}

void QCTest::TearDown() {
  const auto name = " (" + GetParam().name + ")";
  printProgram(program.get(), "Original QC IR" + name, llvm::errs());
  printProgram(canonicalizedProgram.get(), "Canonicalized QC IR" + name,
               llvm::errs());
  printProgram(reference.get(), "Reference QC IR" + name, llvm::errs());
  printProgram(canonicalizedReference.get(),
               "Canonicalized Reference QC IR" + name, llvm::errs());
}

std::string printTestName(const testing::TestParamInfo<QCTestCase>& info) {
  return info.param.name;
}

TEST_P(QCTest, ProgramEquivalence) {
  auto& [_, programBuilder, referenceBuilder] = GetParam();

  program = mlir::qc::QCProgramBuilder::build(context.get(), programBuilder);
  ASSERT_TRUE(program);

  canonicalizedProgram = program.get().clone();
  runCanonicalizationPasses(canonicalizedProgram.get());
  ASSERT_TRUE(canonicalizedProgram);
  EXPECT_TRUE(mlir::verify(*canonicalizedProgram).succeeded());

  reference =
      mlir::qc::QCProgramBuilder::build(context.get(), referenceBuilder);
  ASSERT_TRUE(reference);

  canonicalizedReference = reference.get().clone();
  runCanonicalizationPasses(canonicalizedReference.get());
  ASSERT_TRUE(canonicalizedReference);
  EXPECT_TRUE(mlir::verify(*canonicalizedReference).succeeded());

  EXPECT_TRUE(areModulesEquivalentWithPermutations(
      canonicalizedProgram.get(), canonicalizedReference.get()));
}
