/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "TestCaseUtils.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/Translation/TranslateQASM3ToQC.h"
#include "mlir/Support/IRVerification.h"
#include "mlir/Support/Passes.h"
#include "qc_programs.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>

#include <filesystem>
#include <memory>
#include <ostream>
#include <string>

using namespace mlir;

namespace {

struct QASM3TranslationTestCase {
  std::string name;
  std::string path;
  mqt::test::NamedBuilder<qc::QCProgramBuilder> referenceBuilder;

  friend std::ostream& operator<<(std::ostream& os,
                                  const QASM3TranslationTestCase& test);
};

// NOLINTNEXTLINE(llvm-prefer-static-over-anonymous-namespace)
std::ostream& operator<<(std::ostream& os,
                         const QASM3TranslationTestCase& test) {
  return os << "QASM3Translation{" << test.name << ", original=" << test.path
            << ", reference="
            << mqt::test::displayName(test.referenceBuilder.name) << "}";
}

class QASM3TranslationTest
    : public testing::TestWithParam<QASM3TranslationTestCase> {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    DialectRegistry registry;
    registry.insert<qc::QCDialect, arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }
};

} // namespace

TEST_P(QASM3TranslationTest, ProgramEquivalence) {
  const auto& [name, path, referenceBuilder] = GetParam();
  const auto testName = " (" + name + ")";
  const auto programPath = (std::filesystem::path(__FILE__).parent_path() /
                            "../../../programs/qasm_programs" / path)
                               .lexically_normal()
                               .string();
  mqt::test::DeferredPrinter printer;

  auto translated = qc::translateQASM3ToQC(context.get(), programPath);
  ASSERT_TRUE(translated);
  printer.record(translated.get(), "Translated QC IR" + testName);
  EXPECT_TRUE(verify(*translated).succeeded());

  EXPECT_TRUE(runQCCleanupPipeline(translated.get()).succeeded());
  printer.record(translated.get(), "Canonicalized Translated QC IR" + testName);
  EXPECT_TRUE(verify(*translated).succeeded());

  auto reference =
      qc::QCProgramBuilder::build(context.get(), referenceBuilder.fn);
  ASSERT_TRUE(reference);
  printer.record(reference.get(), "Reference QC IR" + testName);
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(runQCCleanupPipeline(reference.get()).succeeded());
  printer.record(reference.get(), "Canonicalized Reference QC IR" + testName);
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(translated.get(), reference.get()));
}

INSTANTIATE_TEST_SUITE_P(QASM3TranslationProgramsTest, QASM3TranslationTest,
                         testing::Values(QASM3TranslationTestCase{
                             "X", "x.qasm", MQT_NAMED_BUILDER(qc::x)}));
