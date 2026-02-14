/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "test_qc_to_qir.h"

#include "mlir/Conversion/QCToQIR/QCToQIR.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QIR/Builder/QIRProgramBuilder.h"
#include "mlir/Support/IRVerification.h"
#include "mlir/Support/Passes.h"
#include "mlir/Support/PrettyPrinting.h"

#include <gtest/gtest.h>
#include <iosfwd>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <string>

using namespace mlir;

void QCToQIRTest::SetUp() {
  DialectRegistry registry;
  registry.insert<qc::QCDialect, LLVM::LLVMDialect, arith::ArithDialect,
                  func::FuncDialect>();
  context = std::make_unique<MLIRContext>();
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();
}

std::ostream& operator<<(std::ostream& os, const QCToQIRTestCase& info) {
  return os << "QCToQIR{" << info.name
            << ", original=" << mqt::test::displayName(info.programBuilder.name)
            << ", reference="
            << mqt::test::displayName(info.referenceBuilder.name) << "}";
}

static LogicalResult runQCToQIRConversion(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(createQCToQIR());
  return pm.run(module);
}

TEST_P(QCToQIRTest, ProgramEquivalence) {
  const auto& [_, programBuilder, referenceBuilder] = GetParam();
  const auto name = " (" + GetParam().name + ")";

  auto program = qc::QCProgramBuilder::build(context.get(), programBuilder.fn);
  ASSERT_TRUE(program);
  printProgram(program.get(), "Original QC IR" + name, llvm::errs());
  EXPECT_TRUE(mlir::verify(*program).succeeded());

  runCanonicalizationPasses(program.get());
  printProgram(program.get(), "Canonicalized QC IR" + name, llvm::errs());
  EXPECT_TRUE(mlir::verify(*program).succeeded());

  EXPECT_TRUE(succeeded(runQCToQIRConversion(program.get())));
  printProgram(program.get(), "Converted QIR IR" + name, llvm::errs());
  EXPECT_TRUE(mlir::verify(*program).succeeded());

  runCanonicalizationPasses(program.get());
  printProgram(program.get(), "Canonicalized Converted QIR IR" + name,
               llvm::errs());
  EXPECT_TRUE(mlir::verify(*program).succeeded());

  auto reference =
      qir::QIRProgramBuilder::build(context.get(), referenceBuilder.fn);
  ASSERT_TRUE(reference);
  printProgram(reference.get(), "Reference QIR IR" + name, llvm::errs());
  EXPECT_TRUE(mlir::verify(*reference).succeeded());

  runCanonicalizationPasses(reference.get());
  printProgram(reference.get(), "Canonicalized Reference QIR IR" + name,
               llvm::errs());
  EXPECT_TRUE(mlir::verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}
