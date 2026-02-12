/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "test_qco_to_qc.h"

#include "mlir/Conversion/QCOToQC/QCOToQC.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Support/IRVerification.h"
#include "mlir/Support/Passes.h"
#include "mlir/Support/PrettyPrinting.h"

#include <gtest/gtest.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;

void QCOToQCTest::SetUp() {
  // Register all necessary dialects
  DialectRegistry registry;
  registry.insert<mlir::qc::QCDialect, mlir::qco::QCODialect,
                  arith::ArithDialect, func::FuncDialect>();
  context = std::make_unique<MLIRContext>();
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();
}

std::string printTestName(const testing::TestParamInfo<QCOToQCTestCase>& info) {
  return info.param.name;
}

static LogicalResult runQCOToQCConversion(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(createQCOToQC());
  return pm.run(module);
}

TEST_P(QCOToQCTest, ProgramEquivalence) {
  auto& [_, programBuilder, referenceBuilder] = GetParam();
  const auto name = " (" + GetParam().name + ")";

  program = mlir::qco::QCOProgramBuilder::build(context.get(), programBuilder);
  ASSERT_TRUE(program);
  printProgram(program.get(), "Original QCO IR" + name, llvm::errs());
  EXPECT_TRUE(mlir::verify(*program).succeeded());

  runCanonicalizationPasses(program.get());
  printProgram(program.get(), "Canonicalized QCO IR" + name, llvm::errs());
  EXPECT_TRUE(mlir::verify(*program).succeeded());

  EXPECT_TRUE(succeeded(runQCOToQCConversion(program.get())));
  printProgram(program.get(), "Converted QC IR" + name, llvm::errs());
  EXPECT_TRUE(mlir::verify(*program).succeeded());

  runCanonicalizationPasses(program.get());
  printProgram(program.get(), "Canonicalized Converted QC IR" + name,
               llvm::errs());
  EXPECT_TRUE(mlir::verify(*program).succeeded());

  reference =
      mlir::qc::QCProgramBuilder::build(context.get(), referenceBuilder);
  ASSERT_TRUE(reference);
  printProgram(reference.get(), "Reference QC IR" + name, llvm::errs());
  EXPECT_TRUE(mlir::verify(*reference).succeeded());

  runCanonicalizationPasses(reference.get());
  printProgram(reference.get(), "Canonicalized Reference QC IR" + name,
               llvm::errs());
  EXPECT_TRUE(mlir::verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}
