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

#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Support/IRVerification.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/Passes.h>

#include <memory>
#include <utility>

namespace mqt::test {

using namespace mlir;
using namespace mlir::qco;

using namespace mlir;
using namespace mlir::qco;

class IPOTestBase : public testing::Test {

protected:
  MLIRContext context;
  QCOProgramBuilder programBuilder;
  QCOProgramBuilder referenceBuilder;
  OwningOpRef<ModuleOp> moduleOp;
  OwningOpRef<ModuleOp> reference;

  IPOTestBase() : programBuilder(&context), referenceBuilder(&context) {}

  void SetUp() override {
    // Register all necessary dialects
    DialectRegistry registry;
    registry.insert<QCODialect, arith::ArithDialect, func::FuncDialect,
                    qtensor::QTensorDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  }

  /**
   * @brief Runs a single interprocedural stage and compares against the
   * reference.
   *
   * @param stage The one stage to schedule.
   */
  void expectSingleStageMatchesReference(std::unique_ptr<Pass> stage) {
    PassManager pm(moduleOp->getContext());
    pm.addPass(std::move(stage));
    pm.addPass(createCanonicalizerPass());
    ASSERT_TRUE(pm.run(moduleOp.get()).succeeded());
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
  OwningOpRef<ModuleOp> parseModule(const char* source) {
    return parseSourceString<ModuleOp>(source, &context);
  }

  /**
   * @brief Runs one stage on a module without comparing against a reference.
   *
   * @param module The module to transform.
   * @param stage The stage to schedule.
   */
  static LogicalResult runStage(ModuleOp module, std::unique_ptr<Pass> stage) {
    PassManager pm(module.getContext());
    pm.addPass(std::move(stage));
    return pm.run(module);
  }

  /**
   * @brief Counts the qubit allocations inside a named function.
   *
   * @param module The module to look in.
   * @param name The name of the function to count in.
   */
  static unsigned countAllocsIn(ModuleOp module, StringRef name) {
    unsigned count = 0;
    module.walk([&](func::FuncOp func) {
      if (func.getName() == name) {
        func.walk([&](AllocOp) { ++count; });
      }
    });
    return count;
  }

  /**
   * @brief Adds the canonicalizerPass to the current context and runs it.
   */
  static LogicalResult runCanonicalizerPass(ModuleOp moduleOp) {
    PassManager pm(moduleOp.getContext());
    pm.addPass(createCanonicalizerPass());
    return pm.run(moduleOp);
  }
};
} // namespace mqt::test
