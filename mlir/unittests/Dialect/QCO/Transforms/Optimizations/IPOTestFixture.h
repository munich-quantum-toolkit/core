/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file
/// Shared fixture for the interprocedural optimization test suites.
///
/// Each interprocedural pass has its own test file so that a case is scheduled
/// on the pass it is about, rather than on a pipeline where a later pass could
/// mask a regression in an earlier one.

#pragma once

#include "mqt/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QTensor/IR/QTensorDialect.h"
#include "mqt/Support/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"

#include <Support/IRVerification.h>
#include <cstdint>
#include <gtest/gtest.h>
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

  /// Returns the `!qco.qubit` type.
  mlir::Type getQubitType() { return mlir::qco::QubitType::get(&context); }

  /// Returns `tensor<size x !qco.qubit>`.
  mlir::Type getQubitTensorType(int64_t size) {
    return mlir::RankedTensorType::get({size}, getQubitType());
  }

  /// Calls the function of the given name in the builder's module.
  ///
  /// Tests build callees in helpers and call them by name, so look the callee
  /// up instead of threading its handle through.
  static mlir::SmallVector<mlir::Value>
  callFunction(mlir::qco::QCOProgramBuilder& builder, mlir::StringRef name,
               mlir::ValueRange operands) {
    auto moduleOp = builder.getInsertionBlock()
                        ->getParentOp()
                        ->getParentOfType<mlir::ModuleOp>();
    auto callee = moduleOp.lookupSymbol<mlir::func::FuncOp>(name);
    EXPECT_TRUE(callee) << "no function named " << name.str();
    return builder.call(callee, operands);
  }

  /// Runs a single interprocedural stage and compares against the
  /// reference.
  ///
  /// @param stage The one stage to schedule.
  void expectSingleStageMatchesReference(std::unique_ptr<mlir::Pass> stage) {
    mlir::PassManager pm(moduleOp->getContext());
    pm.addPass(std::move(stage));
    pm.addPass(mlir::createCanonicalizerPass());
    ASSERT_TRUE(pm.run(moduleOp.get()).succeeded());
    ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());

    EXPECT_TRUE(
        areModulesEquivalentWithPermutations(moduleOp.get(), reference.get()));
  }

  /// Runs the whole interprocedural pipeline on a module.
  ///
  /// Only for the cross-stage cases. A case about one pass belongs in that
  /// pass's own suite, scheduled on the pass alone.
  ///
  /// @param moduleOp The module to transform.
  static mlir::LogicalResult runQuantumIPOPipeline(mlir::ModuleOp moduleOp) {
    mlir::PassManager pm(moduleOp.getContext());
    populateQuantumIPOPipeline(pm);
    pm.addPass(mlir::createCanonicalizerPass());
    return pm.run(moduleOp);
  }

  /// Runs the whole pipeline and compares against the reference.
  void expectPipelineMatchesReference() {
    ASSERT_TRUE(runQuantumIPOPipeline(moduleOp.get()).succeeded());
    ASSERT_TRUE(runCanonicalizerPass(reference.get()).succeeded());
    EXPECT_TRUE(
        areModulesEquivalentWithPermutations(moduleOp.get(), reference.get()));
  }

  /// Parses a module from MLIR source.
  ///
  /// Used by the few cases describing IR `QCOProgramBuilder` cannot build.
  ///
  /// @param source The MLIR source to parse.
  /// @return The parsed module.
  mlir::OwningOpRef<mlir::ModuleOp> parseModule(const char* source) {
    return mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  }

  /// Runs one stage on a module without comparing against a reference.
  ///
  /// @param module The module to transform.
  /// @param stage The stage to schedule.
  static mlir::LogicalResult runStage(mlir::ModuleOp module,
                                      std::unique_ptr<mlir::Pass> stage) {
    mlir::PassManager pm(module.getContext());
    pm.addPass(std::move(stage));
    return pm.run(module);
  }

  /// Counts the qubit allocations inside a named function.
  ///
  /// @param module The module to look in.
  /// @param name The name of the function to count in.
  static unsigned countAllocsIn(mlir::ModuleOp module, mlir::StringRef name) {
    unsigned count = 0;
    module.walk([&](mlir::func::FuncOp func) {
      if (func.getName() == name) {
        func.walk([&](mlir::qco::AllocOp) { ++count; });
      }
    });
    return count;
  }

  /// Adds the canonicalizerPass to the current context and runs it.
  static mlir::LogicalResult runCanonicalizerPass(mlir::ModuleOp moduleOp) {
    mlir::PassManager pm(moduleOp.getContext());
    pm.addPass(mlir::createCanonicalizerPass());
    return pm.run(moduleOp);
  }
};
} // namespace mqt::test
