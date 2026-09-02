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
 * @file
 * @brief Shared fixture for the interprocedural optimization test suites.
 *
 * @details
 * Each interprocedural pass has its own test file so that a case is scheduled
 * on the pass it is about, rather than on a pipeline where a later pass could
 * mask a regression in an earlier one.
 */

#pragma once

#include "Support/IRVerification.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Support/Passes.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/Passes.h>

#include <memory>
#include <utility>

namespace mqt::test {

/// Every name is spelled out: this is a header, and a `using namespace` in one
/// leaks into every other test file sharing the translation unit under a unity
/// build.
class IPOTestBase : public testing::Test {

protected:
  mlir::MLIRContext context;
  mlir::qco::QCOProgramBuilder programBuilder;
  mlir::qco::QCOProgramBuilder referenceBuilder;
  mlir::OwningOpRef<mlir::ModuleOp> moduleOp;
  mlir::OwningOpRef<mlir::ModuleOp> reference;

  IPOTestBase() : programBuilder(&context), referenceBuilder(&context) {}

  void SetUp() override {
    // Register all necessary dialects
    mlir::DialectRegistry registry;
    registry.insert<mlir::qco::QCODialect, mlir::arith::ArithDialect,
                    mlir::func::FuncDialect, mlir::qtensor::QTensorDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  }

  /**
   * @brief Runs a single interprocedural stage and compares against the
   * reference.
   *
   * @param stage The one stage to schedule.
   */
  void expectSingleStageMatchesReference(std::unique_ptr<mlir::Pass> stage) {
    mlir::PassManager pm(moduleOp->getContext());
    pm.addPass(std::move(stage));
    pm.addPass(mlir::createCanonicalizerPass());
    ASSERT_TRUE(pm.run(moduleOp.get()).succeeded());
    ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

    EXPECT_TRUE(
        areModulesEquivalentWithPermutations(moduleOp.get(), reference.get()));
  }

  /**
   * @brief Runs the whole interprocedural pipeline on a module.
   *
   * @details
   * Only for the cross-stage cases. A case about one pass belongs in that
   * pass's own suite, scheduled on the pass alone.
   *
   * @param moduleOp The module to transform.
   */
  static mlir::LogicalResult runQuantumIPOPipeline(mlir::ModuleOp moduleOp) {
    mlir::PassManager pm(moduleOp.getContext());
    populateQuantumIPOPipeline(pm);
    pm.addPass(mlir::createCanonicalizerPass());
    return pm.run(moduleOp);
  }

  /**
   * @brief Runs the whole pipeline and compares against the reference.
   */
  void expectPipelineMatchesReference() {
    ASSERT_TRUE(runQuantumIPOPipeline(moduleOp.get()).succeeded());
    ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());
    EXPECT_TRUE(
        areModulesEquivalentWithPermutations(moduleOp.get(), reference.get()));
  }

  /**
   * @brief Parses a module from MLIR source.
   *
   * @details
   * Used by the few cases describing IR `QCOProgramBuilder` cannot build.
   *
   * @param source The MLIR source to parse.
   * @return The parsed module.
   */
  mlir::OwningOpRef<mlir::ModuleOp> parseModule(const char* source) {
    return mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  }

  /**
   * @brief Runs one stage on a module without comparing against a reference.
   *
   * @param module The module to transform.
   * @param stage The stage to schedule.
   */
  static mlir::LogicalResult runStage(mlir::ModuleOp module,
                                      std::unique_ptr<mlir::Pass> stage) {
    mlir::PassManager pm(module.getContext());
    pm.addPass(std::move(stage));
    return pm.run(module);
  }

  /**
   * @brief Counts the qubit allocations inside a named function.
   *
   * @param module The module to look in.
   * @param name The name of the function to count in.
   */
  static unsigned countAllocsIn(mlir::ModuleOp module, mlir::StringRef name) {
    unsigned count = 0;
    module.walk([&](mlir::func::FuncOp func) {
      if (func.getName() == name) {
        func.walk([&](mlir::qco::AllocOp) { ++count; });
      }
    });
    return count;
  }

  /**
   * @brief Adds the canonicalizerPass to the current context and runs it.
   */
  static mlir::LogicalResult runCanonicalizerPass(mlir::ModuleOp moduleOp) {
    mlir::PassManager pm(moduleOp.getContext());
    pm.addPass(mlir::createCanonicalizerPass());
    return pm.run(moduleOp);
  }
};
} // namespace mqt::test
