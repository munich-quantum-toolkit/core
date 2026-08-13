/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/Package.hpp"
#include "mlir/Conversion/QCToQCO/QCToQCO.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/Translation/TranslateQASM3ToQC.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/Utils/DDFunctionality.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Support/Passes.h"

#include <gtest/gtest.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cstddef>
#include <memory>
#include <random>
#include <string>

using namespace mlir;

namespace {

class QCOPhaseEstimationExampleTest : public testing::Test {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    DialectRegistry registry;
    registry
        .insert<mlir::qc::QCDialect, qco::QCODialect, qtensor::QTensorDialect,
                arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                scf::SCFDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  [[nodiscard]] OwningOpRef<ModuleOp> parseAndRoundTrip(StringRef path) const {
    auto module = parseSourceFile<ModuleOp>(path, context.get());
    EXPECT_TRUE(module);
    if (!module) {
      return {};
    }
    EXPECT_TRUE(succeeded(verify(*module)));

    std::string printed;
    llvm::raw_string_ostream stream(printed);
    module->print(stream);
    auto reparsed = parseSourceString<ModuleOp>(printed, context.get());
    EXPECT_TRUE(reparsed);
    if (reparsed) {
      EXPECT_TRUE(succeeded(verify(*reparsed)));
    }
    return reparsed;
  }

  [[nodiscard]] OwningOpRef<ModuleOp> importOpenQASM(StringRef path) const {
    const auto source = llvm::MemoryBuffer::getFile(path);
    EXPECT_FALSE(source.getError());
    if (source.getError()) {
      return {};
    }
    auto module =
        mlir::qc::translateQASM3ToQC((*source)->getBuffer(), context.get());
    EXPECT_TRUE(module);
    if (module) {
      EXPECT_TRUE(succeeded(verify(*module)));
      EXPECT_TRUE(succeeded(runQCCleanupPipeline(*module)));
    }
    return module;
  }

  static void expectEstimate(ModuleOp module, size_t numQubits,
                             StringRef expected) {
    // Check that the examples traverse the conversion and optimization
    // pipeline. Simulate a separate conversion because cleanup may correctly
    // remove final measurements whose SSA results are otherwise unused.
    OwningOpRef<ModuleOp> optimized = module.clone();
    PassManager optimizationManager(module.getContext());
    optimizationManager.addPass(createQCToQCO());
    populateQCOCleanupPipeline(optimizationManager);
    ASSERT_TRUE(succeeded(optimizationManager.run(*optimized)));
    ASSERT_TRUE(succeeded(verify(*optimized)));

    PassManager conversionManager(module.getContext());
    conversionManager.addPass(createQCToQCO());
    ASSERT_TRUE(succeeded(conversionManager.run(module)));
    ASSERT_TRUE(succeeded(verify(module)));

    const auto main = module.lookupSymbol<func::FuncOp>("main");
    ASSERT_TRUE(main);
    constexpr size_t shots = 32;
    auto package = std::make_unique<dd::Package>(numQubits);
    std::mt19937_64 rng(0);
    const auto result = qco::sampleWithClassics(main, *package, shots, rng);
    ASSERT_TRUE(succeeded(result));

    ASSERT_EQ(result->classical.size(), 1U);
    EXPECT_EQ(result->classical.begin()->first, expected);
    EXPECT_EQ(result->classical.begin()->second, shots);
  }
};

TEST_F(QCOPhaseEstimationExampleTest, IterativePhaseEstimation) {
  auto module = parseAndRoundTrip(QC_ITERATIVE_PHASE_ESTIMATION_EXAMPLE);
  ASSERT_TRUE(module);
  auto main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  EXPECT_EQ(llvm::range_size(main.getBody().getOps<scf::ForOp>()), 1U);
  expectEstimate(*module, 2, "110");
}

TEST_F(QCOPhaseEstimationExampleTest, QuantumPhaseEstimation) {
  auto module = parseAndRoundTrip(QC_QUANTUM_PHASE_ESTIMATION_EXAMPLE);
  ASSERT_TRUE(module);
  expectEstimate(*module, 4, "011");
}

TEST_F(QCOPhaseEstimationExampleTest, ImportsOpenQASMIterativePhaseEstimation) {
  auto module = importOpenQASM(QASM3_ITERATIVE_PHASE_ESTIMATION_EXAMPLE);
  ASSERT_TRUE(module);
  expectEstimate(*module, 2, "110");
}

TEST_F(QCOPhaseEstimationExampleTest, ImportsOpenQASMQuantumPhaseEstimation) {
  auto module = importOpenQASM(QASM3_QUANTUM_PHASE_ESTIMATION_EXAMPLE);
  ASSERT_TRUE(module);
  expectEstimate(*module, 4, "011");
}

} // namespace
