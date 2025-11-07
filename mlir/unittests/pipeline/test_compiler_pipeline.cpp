/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/QuantumComputation.hpp"
#include "mlir/Compiler/CompilerPipeline.h"
#include "mlir/Conversion/FluxToQuartz/FluxToQuartz.h"
#include "mlir/Conversion/QuartzToFlux/QuartzToFlux.h"
#include "mlir/Conversion/QuartzToQIR/QuartzToQIR.h"
#include "mlir/Dialect/Flux/Builder/FluxProgramBuilder.h"
#include "mlir/Dialect/Flux/IR/FluxDialect.h"
#include "mlir/Dialect/QIR/Builder/QIRProgramBuilder.h"
#include "mlir/Dialect/Quartz/Builder/QuartzProgramBuilder.h"
#include "mlir/Dialect/Quartz/IR/QuartzDialect.h"
#include "mlir/Dialect/Quartz/Translation/TranslateQuantumComputationToQuartz.h"
#include "mlir/Support/PrettyPrinting.h"

#include <algorithm>
#include <functional>
#include <gtest/gtest.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Region.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/Passes.h>
#include <sstream>
#include <string>
#include <unordered_map>

namespace {

using namespace mlir;

//===----------------------------------------------------------------------===//
// Stage Verification Utility
//===----------------------------------------------------------------------===//

/// Compute a structural hash for an operation (excluding SSA value identities).
/// This hash is based on operation name, types, and attributes only.
struct OperationStructuralHash {
  size_t operator()(Operation* op) const {
    size_t hash = llvm::hash_value(op->getName().getStringRef());

    // Hash result types
    for (const Type type : op->getResultTypes()) {
      hash = llvm::hash_combine(hash, type.getAsOpaquePointer());
    }

    // Hash operand types (not values)
    for (const Value operand : op->getOperands()) {
      hash = llvm::hash_combine(hash, operand.getType().getAsOpaquePointer());
    }

    // Hash attributes
    // for (const auto& attr : op->getAttrDictionary()) {
    //   hash = llvm::hash_combine(hash, attr.getName().str());
    //   hash = llvm::hash_combine(hash, attr.getValue().getAsOpaquePointer());
    // }

    return hash;
  }
};

/// Check if two operations are structurally equivalent (excluding SSA value
/// identities).
struct OperationStructuralEquality {
  bool operator()(Operation* lhs, Operation* rhs) const {
    // Check operation name
    if (lhs->getName() != rhs->getName()) {
      return false;
    }

    // Check result types
    if (lhs->getResultTypes() != rhs->getResultTypes()) {
      return false;
    }

    // Check operand types (not values)
    auto lhsOperandTypes = lhs->getOperandTypes();
    auto rhsOperandTypes = rhs->getOperandTypes();
    if (!std::equal(lhsOperandTypes.begin(), lhsOperandTypes.end(),
                    rhsOperandTypes.begin(), rhsOperandTypes.end())) {
      return false;
    }

    // Note: Attributes are intentionally not checked here to allow relaxed
    // comparison. Attributes like function names, parameter names, etc. may
    // differ while operations are still structurally equivalent.

    return true;
  }
};

/// Map to track value equivalence between two modules.
using ValueEquivalenceMap = DenseMap<Value, Value>;

/// Compare two operations for structural equivalence.
/// Updates valueMap to track corresponding SSA values.
bool areOperationsEquivalent(Operation* lhs, Operation* rhs,
                             ValueEquivalenceMap& valueMap) {
  // Check operation name
  if (lhs->getName() != rhs->getName()) {
    return false;
  }

  // Check number of operands and results
  if (lhs->getNumOperands() != rhs->getNumOperands() ||
      lhs->getNumResults() != rhs->getNumResults() ||
      lhs->getNumRegions() != rhs->getNumRegions()) {
    return false;
  }

  // Note: Attributes are intentionally not checked to allow relaxed comparison

  // Check result types
  if (lhs->getResultTypes() != rhs->getResultTypes()) {
    return false;
  }

  // Check operands according to value mapping
  for (auto [lhsOperand, rhsOperand] :
       llvm::zip(lhs->getOperands(), rhs->getOperands())) {
    if (auto it = valueMap.find(lhsOperand); it != valueMap.end()) {
      // Value already mapped, must match
      if (it->second != rhsOperand) {
        return false;
      }
    } else {
      // Establish new mapping
      valueMap[lhsOperand] = rhsOperand;
    }
  }

  // Update value mapping for results
  for (auto [lhsResult, rhsResult] :
       llvm::zip(lhs->getResults(), rhs->getResults())) {
    valueMap[lhsResult] = rhsResult;
  }

  return true;
}

/// Forward declaration for mutual recursion.
bool areBlocksEquivalent(Block& lhs, Block& rhs, ValueEquivalenceMap& valueMap);

/// Compare two regions for structural equivalence.
bool areRegionsEquivalent(Region& lhs, Region& rhs,
                          ValueEquivalenceMap& valueMap) {
  if (lhs.getBlocks().size() != rhs.getBlocks().size()) {
    return false;
  }

  for (auto [lhsBlock, rhsBlock] : llvm::zip(lhs, rhs)) {
    if (!areBlocksEquivalent(lhsBlock, rhsBlock, valueMap)) {
      return false;
    }
  }

  return true;
}

/// Check if an operation has memory effects or control flow side effects
/// that would prevent reordering.
bool hasOrderingConstraints(Operation* op) {
  // Terminators must maintain their position
  if (op->hasTrait<OpTrait::IsTerminator>()) {
    return true;
  }

  // Symbol-defining operations (like function declarations) can be reordered
  if (op->hasTrait<OpTrait::SymbolTable>() ||
      isa<LLVM::LLVMFuncOp, func::FuncOp>(op)) {
    return false;
  }

  // Check for memory effects that enforce ordering
  if (auto memInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance> effects;
    memInterface.getEffects(effects);

    bool hasNonAllocFreeEffects = false;
    for (const auto& effect : effects) {
      // Allow operations with no effects or pure allocation/free effects
      if (!isa<MemoryEffects::Allocate, MemoryEffects::Free>(
              effect.getEffect())) {
        hasNonAllocFreeEffects = true;
        break;
      }
    }

    if (hasNonAllocFreeEffects) {
      return true;
    }
  }

  return false;
}

/// Build a dependence graph for operations.
/// Returns a map from each operation to the set of operations it depends on.
DenseMap<Operation*, DenseSet<Operation*>>
buildDependenceGraph(ArrayRef<Operation*> ops) {
  DenseMap<Operation*, DenseSet<Operation*>> dependsOn;
  DenseMap<Value, Operation*> valueProducers;

  // Build value-to-producer map and dependence relationships
  for (Operation* op : ops) {
    dependsOn[op] = DenseSet<Operation*>();

    // This operation depends on the producers of its operands
    for (Value operand : op->getOperands()) {
      if (auto it = valueProducers.find(operand); it != valueProducers.end()) {
        dependsOn[op].insert(it->second);
      }
    }

    // Register this operation as the producer of its results
    for (Value result : op->getResults()) {
      valueProducers[result] = op;
    }
  }

  return dependsOn;
}

/// Partition operations into groups that can be compared as multisets.
/// Operations in the same group are independent and can be reordered.
std::vector<SmallVector<Operation*>>
partitionIndependentGroups(ArrayRef<Operation*> ops) {
  std::vector<SmallVector<Operation*>> groups;
  if (ops.empty()) {
    return groups;
  }

  auto dependsOn = buildDependenceGraph(ops);
  DenseSet<Operation*> processed;
  SmallVector<Operation*> currentGroup;

  for (Operation* op : ops) {
    bool dependsOnCurrent = false;

    // Check if this operation depends on any operation in the current group
    for (Operation* groupOp : currentGroup) {
      if (dependsOn[op].contains(groupOp)) {
        dependsOnCurrent = true;
        break;
      }
    }

    // Check if this operation has ordering constraints
    bool hasConstraints = hasOrderingConstraints(op);

    // If it depends on current group or has ordering constraints,
    // finalize the current group and start a new one
    if (dependsOnCurrent || (hasConstraints && !currentGroup.empty())) {
      if (!currentGroup.empty()) {
        groups.push_back(std::move(currentGroup));
        currentGroup.clear();
      }
    }

    currentGroup.push_back(op);

    // If this operation has ordering constraints, finalize the group
    if (hasConstraints) {
      groups.push_back(std::move(currentGroup));
      currentGroup.clear();
    }
  }

  // Add any remaining operations
  if (!currentGroup.empty()) {
    groups.push_back(std::move(currentGroup));
  }

  return groups;
}

/// Compare two groups of independent operations using multiset equivalence.
bool areIndependentGroupsEquivalent(ArrayRef<Operation*> lhsOps,
                                    ArrayRef<Operation*> rhsOps) {
  if (lhsOps.size() != rhsOps.size()) {
    return false;
  }

  // Build frequency maps for both groups
  std::unordered_map<Operation*, size_t, OperationStructuralHash,
                     OperationStructuralEquality>
      lhsFrequencyMap;
  std::unordered_map<Operation*, size_t, OperationStructuralHash,
                     OperationStructuralEquality>
      rhsFrequencyMap;

  for (Operation* op : lhsOps) {
    lhsFrequencyMap[op]++;
  }

  for (Operation* op : rhsOps) {
    rhsFrequencyMap[op]++;
  }

  // Check structural equivalence
  if (lhsFrequencyMap.size() != rhsFrequencyMap.size()) {
    return false;
  }

  for (const auto& [lhsOp, lhsCount] : lhsFrequencyMap) {
    auto it = rhsFrequencyMap.find(lhsOp);
    if (it == rhsFrequencyMap.end() || it->second != lhsCount) {
      return false;
    }
  }

  return true;
}

/// Compare two blocks for structural equivalence, allowing permutations
/// of independent operations.
bool areBlocksEquivalent(Block& lhs, Block& rhs,
                         ValueEquivalenceMap& valueMap) {
  // Check block arguments
  if (lhs.getNumArguments() != rhs.getNumArguments()) {
    return false;
  }

  for (auto [lhsArg, rhsArg] :
       llvm::zip(lhs.getArguments(), rhs.getArguments())) {
    if (lhsArg.getType() != rhsArg.getType()) {
      return false;
    }
    valueMap[lhsArg] = rhsArg;
  }

  // Collect all operations
  SmallVector<Operation*> lhsOps;
  SmallVector<Operation*> rhsOps;

  for (Operation& op : lhs) {
    lhsOps.push_back(&op);
  }

  for (Operation& op : rhs) {
    rhsOps.push_back(&op);
  }

  if (lhsOps.size() != rhsOps.size()) {
    return false;
  }

  // Partition operations into independent groups
  auto lhsGroups = partitionIndependentGroups(lhsOps);
  auto rhsGroups = partitionIndependentGroups(rhsOps);

  if (lhsGroups.size() != rhsGroups.size()) {
    return false;
  }

  // Compare each group
  for (size_t groupIdx = 0; groupIdx < lhsGroups.size(); ++groupIdx) {
    auto& lhsGroup = lhsGroups[groupIdx];
    auto& rhsGroup = rhsGroups[groupIdx];

    if (!areIndependentGroupsEquivalent(lhsGroup, rhsGroup)) {
      return false;
    }

    // Update value mappings for operations in this group
    // We need to match operations and update the value map
    // Since they are structurally equivalent, we can match them
    // by trying all permutations (for small groups) or use a greedy approach

    // Use a simple greedy matching
    DenseSet<Operation*> matchedRhs;
    for (Operation* lhsOp : lhsGroup) {
      bool matched = false;
      for (Operation* rhsOp : rhsGroup) {
        if (matchedRhs.contains(rhsOp)) {
          continue;
        }

        ValueEquivalenceMap tempMap = valueMap;
        if (areOperationsEquivalent(lhsOp, rhsOp, tempMap)) {
          valueMap = std::move(tempMap);
          matchedRhs.insert(rhsOp);
          matched = true;

          // Recursively compare regions
          for (auto [lhsRegion, rhsRegion] :
               llvm::zip(lhsOp->getRegions(), rhsOp->getRegions())) {
            if (!areRegionsEquivalent(lhsRegion, rhsRegion, valueMap)) {
              return false;
            }
          }
          break;
        }
      }

      if (!matched) {
        return false;
      }
    }
  }

  return true;
}

/// Compare two MLIR modules for structural equivalence, allowing permutations
/// of speculatable operations.
bool areModulesEquivalentWithPermutations(ModuleOp lhs, ModuleOp rhs) {
  ValueEquivalenceMap valueMap;
  return areRegionsEquivalent(lhs.getBodyRegion(), rhs.getBodyRegion(),
                              valueMap);
}

/**
 * @brief Verify a stage matches expected module
 *
 * @param stageName Human-readable stage name for error messages
 * @param actualIR String IR from CompilationRecord
 * @param expectedModule Expected module to compare against
 * @return True if modules match, false otherwise with diagnostic output
 */
[[nodiscard]] testing::AssertionResult verify(const std::string& stageName,
                                              const std::string& actualIR,
                                              ModuleOp expectedModule) {
  // Parse the actual IR string into a ModuleOp
  const auto actualModule =
      parseSourceString<ModuleOp>(actualIR, expectedModule.getContext());
  if (!actualModule) {
    return testing::AssertionFailure()
           << stageName << " failed to parse actual IR";
  }

  if (!areModulesEquivalentWithPermutations(*actualModule, expectedModule)) {
    std::ostringstream msg;
    msg << stageName << " IR does not match expected structure\n\n";

    std::string expectedStr;
    llvm::raw_string_ostream expectedStream(expectedStr);
    expectedModule.print(expectedStream);
    expectedStream.flush();

    msg << "=== EXPECTED IR ===\n" << expectedStr << "\n\n";
    msg << "=== ACTUAL IR ===\n" << actualIR << "\n";

    return testing::AssertionFailure() << msg.str();
  }

  return testing::AssertionSuccess();
}

//===----------------------------------------------------------------------===//
// Stage Expectation Builder
//===----------------------------------------------------------------------===//

/**
 * @brief Helper to build expected IR for multiple stages at once
 *
 * @details
 * Reduces boilerplate by allowing specification of which stages should
 * match which expected IR.
 */
struct StageExpectations {
  ModuleOp quartzImport;
  ModuleOp fluxConversion;
  ModuleOp optimization;
  ModuleOp quartzConversion;
  ModuleOp qirConversion;
};

//===----------------------------------------------------------------------===//
// Base Test Fixture
//===----------------------------------------------------------------------===//

/**
 * @brief Base test fixture for end-to-end compiler pipeline tests
 *
 * @details
 * Provides a configured MLIR context with all necessary dialects loaded
 * and utility methods for creating quantum circuits and running the
 * compilation pipeline.
 */
class CompilerPipelineTest : public testing::Test {
protected:
  std::unique_ptr<MLIRContext> context;
  QuantumCompilerConfig config;
  CompilationRecord record;

  OwningOpRef<ModuleOp> emptyQuartz;
  OwningOpRef<ModuleOp> emptyFlux;
  OwningOpRef<ModuleOp> emptyQIR;

  void SetUp() override {
    // Register all dialects needed for the full compilation pipeline
    DialectRegistry registry;
    registry
        .insert<quartz::QuartzDialect, flux::FluxDialect, arith::ArithDialect,
                cf::ControlFlowDialect, func::FuncDialect,
                memref::MemRefDialect, scf::SCFDialect, LLVM::LLVMDialect>();

    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();

    // Enable QIR conversion and recording by default
    config.convertToQIR = true;
    config.recordIntermediates = true;
    config.printIRAfterAllStages =
        true; /// TODO: Change back after everything is running

    emptyQuartz = buildQuartzIR([](quartz::QuartzProgramBuilder&) {});
    emptyFlux = buildFluxIR([](flux::FluxProgramBuilder&) {});
    emptyQIR = buildQIR([](qir::QIRProgramBuilder&) {});
  }

  //===--------------------------------------------------------------------===//
  // Quantum Circuit Construction and Import
  //===--------------------------------------------------------------------===//

  /**
   * @brief Pretty print quantum computation with ASCII art borders
   *
   * @param qc The quantum computation to print
   */
  static void prettyPrintQuantumComputation(const qc::QuantumComputation& qc) {
    llvm::errs() << "\n";
    printBoxTop();

    // Print header
    printBoxLine("Initial Quantum Computation");

    printBoxMiddle();

    // Print internal representation
    printBoxLine("Internal Representation:");

    // Capture the internal representation
    std::ostringstream internalRepr;
    internalRepr << qc;
    const std::string internalStr = internalRepr.str();

    // Print with line wrapping
    printBoxText(internalStr);

    printBoxMiddle();

    // Print OpenQASM3 representation
    printBoxLine("OpenQASM3 Representation:");
    printBoxLine("");

    const auto qasmStr = qc.toQASM();

    // Print with line wrapping
    printBoxText(qasmStr);

    printBoxBottom();
    llvm::errs().flush();
  }

  /**
   * @brief Import a QuantumComputation into Quartz dialect
   */
  [[nodiscard]] OwningOpRef<ModuleOp>
  importQuantumCircuit(const qc::QuantumComputation& qc) const {
    if (config.printIRAfterAllStages) {
      prettyPrintQuantumComputation(qc);
    }
    return translateQuantumComputationToQuartz(context.get(), qc);
  }

  /**
   * @brief Run the compiler pipeline with the current configuration
   */
  [[nodiscard]] LogicalResult runPipeline(const ModuleOp module) {
    const QuantumCompilerPipeline pipeline(config);
    return pipeline.runPipeline(module, &record);
  }

  //===--------------------------------------------------------------------===//
  // Expected IR Builder Methods
  //===--------------------------------------------------------------------===//

  /**
   * @brief Run canonicalization
   */
  static void runCanonicalizationPasses(ModuleOp module) {
    PassManager pm(module.getContext());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    if (pm.run(module).failed()) {
      llvm::errs() << "Failed to run canonicalization passes\n";
    }
  }

  /**
   * @brief Build expected Quartz IR programmatically and run canonicalization
   */
  [[nodiscard]] OwningOpRef<ModuleOp> buildQuartzIR(
      const std::function<void(quartz::QuartzProgramBuilder&)>& buildFunc)
      const {
    quartz::QuartzProgramBuilder builder(context.get());
    builder.initialize();
    buildFunc(builder);
    auto module = builder.finalize();
    runCanonicalizationPasses(module.get());
    return module;
  }

  /**
   * @brief Build expected Flux IR programmatically and run canonicalization
   */
  [[nodiscard]] OwningOpRef<ModuleOp> buildFluxIR(
      const std::function<void(flux::FluxProgramBuilder&)>& buildFunc) const {
    flux::FluxProgramBuilder builder(context.get());
    builder.initialize();
    buildFunc(builder);
    auto module = builder.finalize();
    runCanonicalizationPasses(module.get());
    return module;
  }

  /**
   * @brief Build expected QIR programmatically and run canonicalization
   */
  [[nodiscard]] OwningOpRef<ModuleOp> buildQIR(
      const std::function<void(qir::QIRProgramBuilder&)>& buildFunc) const {
    qir::QIRProgramBuilder builder(context.get());
    builder.initialize();
    buildFunc(builder);
    auto module = builder.finalize();
    runCanonicalizationPasses(module.get());
    return module;
  }

  //===--------------------------------------------------------------------===//
  // Stage Verification Methods
  //===--------------------------------------------------------------------===//

  /**
   * @brief Verify all stages match their expected IR
   *
   * @details
   * Simplifies test writing by checking all stages with one call.
   * Stages without expectations are skipped.
   */
  void verifyAllStages(const StageExpectations& expectations) const {
    if (expectations.quartzImport != nullptr) {
      EXPECT_TRUE(verify("Quartz Import", record.afterInitialCanon,
                         expectations.quartzImport));
    }

    if (expectations.fluxConversion != nullptr) {
      EXPECT_TRUE(verify("Flux Conversion", record.afterFluxCanon,
                         expectations.fluxConversion));
    }

    if (expectations.optimization != nullptr) {
      EXPECT_TRUE(verify("Optimization", record.afterOptimizationCanon,
                         expectations.optimization));
    }

    if (expectations.quartzConversion != nullptr) {
      EXPECT_TRUE(verify("Quartz Conversion", record.afterQuartzCanon,
                         expectations.quartzConversion));
    }

    if (expectations.qirConversion != nullptr) {
      EXPECT_TRUE(verify("QIR Conversion", record.afterQIRCanon,
                         expectations.qirConversion));
    }
  }

  void TearDown() override {
    // Verify all stages were recorded (basic sanity check)
    EXPECT_FALSE(record.afterQuartzImport.empty())
        << "Quartz import stage was not recorded";
    EXPECT_FALSE(record.afterInitialCanon.empty())
        << "Initial canonicalization stage was not recorded";
    EXPECT_FALSE(record.afterFluxConversion.empty())
        << "Flux conversion stage was not recorded";
    EXPECT_FALSE(record.afterFluxCanon.empty())
        << "Flux canonicalization stage was not recorded";
    EXPECT_FALSE(record.afterOptimization.empty())
        << "Optimization stage was not recorded";
    EXPECT_FALSE(record.afterOptimizationCanon.empty())
        << "Optimization canonicalization stage was not recorded";
    EXPECT_FALSE(record.afterQuartzConversion.empty())
        << "Quartz conversion stage was not recorded";
    EXPECT_FALSE(record.afterQuartzCanon.empty())
        << "Quartz canonicalization stage was not recorded";

    if (config.convertToQIR) {
      EXPECT_FALSE(record.afterQIRConversion.empty())
          << "QIR conversion stage was not recorded";
      EXPECT_FALSE(record.afterQIRCanon.empty())
          << "QIR canonicalization stage was not recorded";
    }
  }
};

//===----------------------------------------------------------------------===//
// Test Cases
//===----------------------------------------------------------------------===//

// ##################################################
// # Empty Circuit Tests
// ##################################################

/**
 * @brief Test: Empty circuit compilation
 *
 * @details
 * Verifies that an empty circuit compiles correctly through all stages,
 * producing empty but valid IR at each stage.
 */
TEST_F(CompilerPipelineTest, EmptyCircuit) {
  // Create empty circuit
  const qc::QuantumComputation qc;

  // Import to Quartz dialect
  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);

  // Run compilation
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  // Verify all stages
  verifyAllStages({
      .quartzImport = emptyQuartz.get(),
      .fluxConversion = emptyFlux.get(),
      .optimization = emptyFlux.get(),
      .quartzConversion = emptyQuartz.get(),
      .qirConversion = emptyQIR.get(),
  });
}

// ##################################################
// # Quantum Register Allocation Tests
// ##################################################

/**
 * @brief Test: Single qubit register allocation
 *
 * @details
 * Since the register is unused, it should be removed during canonicalization
 * in the Flux dialect.
 */
TEST_F(CompilerPipelineTest, SingleQubitRegister) {
  qc::QuantumComputation qc;
  qc.addQubitRegister(1, "q");

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto quartzExpected = buildQuartzIR(
      [](quartz::QuartzProgramBuilder& b) { b.allocQubitRegister(1, "q"); });
  const auto fluxExpected = buildFluxIR(
      [](flux::FluxProgramBuilder& b) { b.allocQubitRegister(1, "q"); });

  verifyAllStages({
      .quartzImport = quartzExpected.get(),
      .fluxConversion = fluxExpected.get(),
      .optimization = emptyFlux.get(),
      .quartzConversion = emptyQuartz.get(),
      .qirConversion = emptyQIR.get(),
  });
}

/**
 * @brief Test: Multi-qubit register allocation
 */
TEST_F(CompilerPipelineTest, MultiQubitRegister) {
  qc::QuantumComputation qc;
  qc.addQubitRegister(3, "q");

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto quartzExpected = buildQuartzIR(
      [](quartz::QuartzProgramBuilder& b) { b.allocQubitRegister(3, "q"); });
  const auto fluxExpected = buildFluxIR(
      [](flux::FluxProgramBuilder& b) { b.allocQubitRegister(3, "q"); });

  verifyAllStages({
      .quartzImport = quartzExpected.get(),
      .fluxConversion = fluxExpected.get(),
      .optimization = emptyFlux.get(),
      .quartzConversion = emptyQuartz.get(),
      .qirConversion = emptyQIR.get(),
  });
}

/**
 * @brief Test: Multiple quantum registers
 */
TEST_F(CompilerPipelineTest, MultipleQuantumRegisters) {
  qc::QuantumComputation qc;
  qc.addQubitRegister(2, "q");
  qc.addQubitRegister(3, "aux");

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto quartzExpected =
      buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
        b.allocQubitRegister(2, "q");
        b.allocQubitRegister(3, "aux");
      });
  const auto fluxExpected = buildFluxIR([](flux::FluxProgramBuilder& b) {
    b.allocQubitRegister(2, "q");
    b.allocQubitRegister(3, "aux");
  });

  verifyAllStages({
      .quartzImport = quartzExpected.get(),
      .fluxConversion = fluxExpected.get(),
      .optimization = emptyFlux.get(),
      .quartzConversion = emptyQuartz.get(),
      .qirConversion = emptyQIR.get(),
  });
}

/**
 * @brief Test: Large qubit register allocation
 */
TEST_F(CompilerPipelineTest, LargeQubitRegister) {
  qc::QuantumComputation qc;
  qc.addQubitRegister(100, "q");

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());
}

// ##################################################
// # Classical Register Allocation Tests
// ##################################################

/**
 * @brief Test: Single classical bit register
 *
 * @details
 * Since the register is unused, it should be removed during canonicalization.
 */
TEST_F(CompilerPipelineTest, SingleClassicalBitRegister) {
  qc::QuantumComputation qc;
  qc.addClassicalRegister(1, "c");

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto expected = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    b.allocClassicalBitRegister(1, "c");
  });

  verifyAllStages({
      .quartzImport = expected.get(),
      .fluxConversion = emptyFlux.get(),
      .optimization = emptyFlux.get(),
      .quartzConversion = emptyQuartz.get(),
      .qirConversion = emptyQIR.get(),
  });
}

/**
 * @brief Test: Multi-bit classical register
 *
 * @details
 * Since the register is unused, it should be removed during canonicalization.
 */
TEST_F(CompilerPipelineTest, MultiBitClassicalRegister) {
  qc::QuantumComputation qc;
  qc.addClassicalRegister(5, "c");

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto expected = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    b.allocClassicalBitRegister(5, "c");
  });

  verifyAllStages({
      .quartzImport = expected.get(),
      .fluxConversion = emptyFlux.get(),
      .optimization = emptyFlux.get(),
      .quartzConversion = emptyQuartz.get(),
      .qirConversion = emptyQIR.get(),
  });
}

/**
 * @brief Test: Multiple classical registers
 *
 * @details
 * Since the registers are unused, they should be removed during
 * canonicalization.
 */
TEST_F(CompilerPipelineTest, MultipleClassicalRegisters) {
  qc::QuantumComputation qc;
  qc.addClassicalRegister(3, "c");
  qc.addClassicalRegister(2, "d");

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto expected = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    b.allocClassicalBitRegister(3, "c");
    b.allocClassicalBitRegister(2, "d");
  });

  verifyAllStages({
      .quartzImport = expected.get(),
      .fluxConversion = emptyFlux.get(),
      .optimization = emptyFlux.get(),
      .quartzConversion = emptyQuartz.get(),
      .qirConversion = emptyQIR.get(),
  });
}

/**
 * @brief Test: Large classical bit register
 */
TEST_F(CompilerPipelineTest, LargeClassicalBitRegister) {
  qc::QuantumComputation qc;
  qc.addClassicalRegister(128, "c");

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());
}

// ##################################################
// # Reset Operation Tests
// ##################################################

/**
 * @brief Test: Single reset in single qubit circuit
 *
 * @details
 * Since the reset directly follows an allocation, it should be removed during
 * canonicalization.
 */
TEST_F(CompilerPipelineTest, SingleResetInSingleQubitCircuit) {
  qc::QuantumComputation qc(1);
  qc.reset(0);

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto expected = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    const auto q = b.allocQubitRegister(1, "q");
    b.reset(q[0]);
  });
  const auto fluxExpected = buildFluxIR([](flux::FluxProgramBuilder& b) {
    const auto q = b.allocQubitRegister(1, "q");
    b.reset(q[0]);
  });

  verifyAllStages({
      .quartzImport = expected.get(),
      .fluxConversion = fluxExpected.get(),
      .optimization = emptyFlux.get(),
      .quartzConversion = emptyQuartz.get(),
      .qirConversion = emptyQIR.get(),
  });
}

/**
 * @brief Test: Consecutive reset operations
 *
 * @details
 * Since reset is idempotent, consecutive resets should be reduced to a single
 * reset during canonicalization. Since that single reset directly follows an
 * allocation, it should be removed as well.
 */
TEST_F(CompilerPipelineTest, ConsecutiveResetOperations) {
  qc::QuantumComputation qc(1);
  qc.reset(0);
  qc.reset(0);
  qc.reset(0);

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto expected = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    const auto q = b.allocQubitRegister(1, "q");
    b.reset(q[0]);
    b.reset(q[0]);
    b.reset(q[0]);
  });

  const auto fluxExpected = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto q = b.allocQubitRegister(1, "q");
    q[0] = b.reset(q[0]);
    q[0] = b.reset(q[0]);
    q[0] = b.reset(q[0]);
  });

  verifyAllStages({
      .quartzImport = expected.get(),
      .fluxConversion = fluxExpected.get(),
      .optimization = emptyFlux.get(),
      .quartzConversion = emptyQuartz.get(),
      .qirConversion = emptyQIR.get(),
  });
}

/**
 * @brief Test: Separate resets in two qubit system
 */
TEST_F(CompilerPipelineTest, SeparateResetsInTwoQubitSystem) {
  qc::QuantumComputation qc(2);
  qc.reset(0);
  qc.reset(1);

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto expected = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    const auto q = b.allocQubitRegister(2, "q");
    b.reset(q[0]);
    b.reset(q[1]);
  });

  const auto fluxExpected = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto q = b.allocQubitRegister(2, "q");
    q[0] = b.reset(q[0]);
    q[1] = b.reset(q[1]);
  });

  verifyAllStages({
      .quartzImport = expected.get(),
      .fluxConversion = fluxExpected.get(),
      .optimization = emptyFlux.get(),
      .quartzConversion = emptyQuartz.get(),
      .qirConversion = emptyQIR.get(),
  });
}

// ##################################################
// # Measure Operation Tests
// ##################################################

/**
 * @brief Test: Single measurement to single bit
 */
TEST_F(CompilerPipelineTest, SingleMeasurementToSingleBit) {
  qc::QuantumComputation qc(1, 1);
  qc.measure(0, 0);

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto expected = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    const auto& c = b.allocClassicalBitRegister(1);
    b.measure(q[0], c[0]);
  });

  const auto fluxExpected = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    const auto& c = b.allocClassicalBitRegister(1);
    b.measure(q[0], c[0]);
  });

  const auto qirExpected = buildQIR([](qir::QIRProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    const auto& c = b.allocClassicalBitRegister(1);
    b.measure(q[0], c[0]);
  });

  verifyAllStages({
      .quartzImport = expected.get(),
      .fluxConversion = fluxExpected.get(),
      .optimization = fluxExpected.get(),
      .quartzConversion = expected.get(),
      .qirConversion = qirExpected.get(),
  });
}

/**
 * @brief Test: Repeated measurement to same bit
 */
TEST_F(CompilerPipelineTest, RepeatedMeasurementToSameBit) {
  qc::QuantumComputation qc(1, 1);
  qc.measure(0, 0);
  qc.measure(0, 0);

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto expected = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    const auto& c = b.allocClassicalBitRegister(1);
    b.measure(q[0], c[0]);
    b.measure(q[0], c[0]);
  });

  const auto fluxExpected = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    const auto& c = b.allocClassicalBitRegister(1);
    q[0] = b.measure(q[0], c[0]);
    q[0] = b.measure(q[0], c[0]);
  });

  const auto qirExpected = buildQIR([](qir::QIRProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    const auto& c = b.allocClassicalBitRegister(1);
    b.measure(q[0], c[0]);
    b.measure(q[0], c[0]);
  });

  verifyAllStages({
      .quartzImport = expected.get(),
      .fluxConversion = fluxExpected.get(),
      .optimization = fluxExpected.get(),
      .quartzConversion = expected.get(),
      .qirConversion = qirExpected.get(),
  });
}

/**
 * @brief Test: Repeated measurement on separate bits
 */
TEST_F(CompilerPipelineTest, RepeatedMeasurementOnSeparateBits) {
  qc::QuantumComputation qc(1);
  qc.addClassicalRegister(3);
  qc.measure(0, 0);
  qc.measure(0, 1);
  qc.measure(0, 2);

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto expected = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    const auto& c = b.allocClassicalBitRegister(3);
    b.measure(q[0], c[0]);
    b.measure(q[0], c[1]);
    b.measure(q[0], c[2]);
  });

  const auto fluxExpected = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    const auto& c = b.allocClassicalBitRegister(3);
    q[0] = b.measure(q[0], c[0]);
    q[0] = b.measure(q[0], c[1]);
    q[0] = b.measure(q[0], c[2]);
  });

  const auto qirExpected = buildQIR([](qir::QIRProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    const auto& c = b.allocClassicalBitRegister(3);
    b.measure(q[0], c[0]);
    b.measure(q[0], c[1]);
    b.measure(q[0], c[2]);
  });

  verifyAllStages({
      .quartzImport = expected.get(),
      .fluxConversion = fluxExpected.get(),
      .optimization = fluxExpected.get(),
      .quartzConversion = expected.get(),
      .qirConversion = qirExpected.get(),
  });
}

/**
 * @brief Test: Multiple classical registers with measurements
 */
TEST_F(CompilerPipelineTest, MultipleClassicalRegistersAndMeasurements) {
  qc::QuantumComputation qc(2);
  const auto& c1 = qc.addClassicalRegister(1, "c1");
  const auto& c2 = qc.addClassicalRegister(1, "c2");
  qc.measure(0, c1[0]);
  qc.measure(1, c2[0]);

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto expected = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    auto q = b.allocQubitRegister(2);
    const auto& creg1 = b.allocClassicalBitRegister(1, "c1");
    const auto& creg2 = b.allocClassicalBitRegister(1, "c2");
    b.measure(q[0], creg1[0]);
    b.measure(q[1], creg2[0]);
  });

  const auto fluxExpected = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto q = b.allocQubitRegister(2);
    const auto& creg1 = b.allocClassicalBitRegister(1, "c1");
    const auto& creg2 = b.allocClassicalBitRegister(1, "c2");
    q[0] = b.measure(q[0], creg1[0]);
    q[1] = b.measure(q[1], creg2[0]);
  });

  const auto qirExpected = buildQIR([](qir::QIRProgramBuilder& b) {
    auto q = b.allocQubitRegister(2);
    const auto& creg1 = b.allocClassicalBitRegister(1, "c1");
    const auto& creg2 = b.allocClassicalBitRegister(1, "c2");
    b.measure(q[0], creg1[0]);
    b.measure(q[1], creg2[0]);
  });

  verifyAllStages({
      .quartzImport = expected.get(),
      .fluxConversion = fluxExpected.get(),
      .optimization = fluxExpected.get(),
      .quartzConversion = expected.get(),
      .qirConversion = qirExpected.get(),
  });
}

// ##################################################
// # Temporary Unitary Operation Tests
// ##################################################

TEST_F(CompilerPipelineTest, X) {
  qc::QuantumComputation qc;
  qc.addQubitRegister(1, "q");
  qc.x(0);
  qc.x(0);
  qc.x(0);

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto quartzInit = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    auto q = reg[0];
    b.x(q);
    b.x(q);
    b.x(q);
  });
  const auto fluxInit = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    auto q0 = reg[0];
    auto q1 = b.x(q0);
    auto q2 = b.x(q1);
    b.x(q2);
  });
  const auto fluxOpt = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    auto q0 = reg[0];
    b.x(q0);
  });
  const auto quartzOpt = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    auto q = reg[0];
    b.x(q);
  });
  const auto qirOpt = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1);
    auto q = reg[0];
    b.x(q);
  });

  verifyAllStages({
      .quartzImport = quartzInit.get(),
      .fluxConversion = fluxInit.get(),
      .optimization = fluxOpt.get(),
      .quartzConversion = quartzOpt.get(),
      .qirConversion = qirOpt.get(),
  });
}

TEST_F(CompilerPipelineTest, CX) {
  qc::QuantumComputation qc;
  qc.addQubitRegister(2, "q");
  qc.cx(0, 1);

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto quartzInit = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    auto q0 = reg[0];
    auto q1 = reg[1];
    b.ctrl(q0, [&](auto& b) { b.x(q1); });
  });
  const auto fluxInit = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    auto q0 = reg[0];
    auto q1 = reg[1];
    b.ctrl({q0}, {q1}, [&](auto& b) {
      auto q1res = b.x(q1);
      return SmallVector<Value>{q1res};
    });
  });
  const auto fluxOpt = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    auto q0 = reg[0];
    auto q1 = reg[1];
    b.ctrl({q0}, {q1}, [&](auto& b) {
      auto q1res = b.x(q1);
      return SmallVector<Value>{q1res};
    });
  });
  const auto quartzOpt = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    auto q0 = reg[0];
    auto q1 = reg[1];
    b.ctrl(q0, [&](auto& b) { b.x(q1); });
  });
  const auto qirOpt = buildQIR([](qir::QIRProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2);
    /// TODO: Replace uncomment this when CX can be converted to QIR
    // auto q1 = reg[1];
    // b.cx(q0, q1);
    b.x(reg[1]);
  });

  verifyAllStages({
      .quartzImport = quartzInit.get(),
      .fluxConversion = fluxInit.get(),
      .optimization = fluxOpt.get(),
      .quartzConversion = quartzOpt.get(),
      .qirConversion = qirOpt.get(),
  });
}

TEST_F(CompilerPipelineTest, RX) {
  qc::QuantumComputation qc;
  qc.addQubitRegister(1, "q");
  qc.rx(1.0, 0);
  qc.rx(0.5, 0);

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto quartzInit = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    auto q = reg[0];
    b.rx(1.0, q);
    b.rx(0.5, q);
  });
  const auto fluxInit = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    auto q0 = reg[0];
    auto q1 = b.rx(1.0, q0);
    b.rx(0.5, q1);
  });
  const auto fluxOpt = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    auto q0 = reg[0];
    b.rx(1.5, q0);
  });
  const auto quartzOpt = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    auto q = reg[0];
    b.rx(1.5, q);
  });
  const auto qirOpt = buildQIR([](qir::QIRProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    b.rx(1.5, q[0]);
  });

  verifyAllStages({
      .quartzImport = quartzInit.get(),
      .fluxConversion = fluxInit.get(),
      .optimization = fluxOpt.get(),
      .quartzConversion = quartzOpt.get(),
      .qirConversion = qirOpt.get(),
  });
}

TEST_F(CompilerPipelineTest, U2) {
  qc::QuantumComputation qc;
  qc.addQubitRegister(1, "q");
  qc.u2(1.0, 0.5, 0);

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto quartzExpected =
      buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
        auto q = b.allocQubitRegister(1, "q");
        b.u2(1.0, 0.5, q[0]);
      });
  const auto fluxExpected = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto q = b.allocQubitRegister(1, "q");
    b.u2(1.0, 0.5, q[0]);
  });
  const auto qirExpected = buildQIR([](qir::QIRProgramBuilder& b) {
    auto q = b.allocQubitRegister(1);
    b.u2(1.0, 0.5, q[0]);
  });

  verifyAllStages({
      .quartzImport = quartzExpected.get(),
      .fluxConversion = fluxExpected.get(),
      .optimization = fluxExpected.get(),
      .quartzConversion = quartzExpected.get(),
      .qirConversion = qirExpected.get(),
  });
}

TEST_F(CompilerPipelineTest, SWAP) {
  qc::QuantumComputation qc;
  qc.addQubitRegister(2, "q");
  qc.swap(0, 1);

  const auto module = importQuantumCircuit(qc);
  ASSERT_TRUE(module);
  ASSERT_TRUE(runPipeline(module.get()).succeeded());

  const auto quartzExpected =
      buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
        auto q = b.allocQubitRegister(2, "q");
        b.swap(q[0], q[1]);
      });
  const auto fluxExpected = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto q = b.allocQubitRegister(2, "q");
    b.swap(q[0], q[1]);
  });
  const auto qirExpected = buildQIR([](qir::QIRProgramBuilder& b) {
    auto q = b.allocQubitRegister(2);
    b.swap(q[0], q[1]);
  });

  verifyAllStages({
      .quartzImport = quartzExpected.get(),
      .fluxConversion = fluxExpected.get(),
      .optimization = fluxExpected.get(),
      .quartzConversion = quartzExpected.get(),
      .qirConversion = qirExpected.get(),
  });
}

// ##################################################
// # Temporary Simple Conversion Tests
// ##################################################

class SimpleConversionTest : public testing::Test {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    // Register all dialects needed for the full compilation pipeline
    DialectRegistry registry;
    registry
        .insert<quartz::QuartzDialect, flux::FluxDialect, arith::ArithDialect,
                cf::ControlFlowDialect, func::FuncDialect,
                memref::MemRefDialect, scf::SCFDialect, LLVM::LLVMDialect>();

    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  /**
   * @brief Run canonicalization
   */
  static void runCanonicalizationPasses(ModuleOp module) {
    PassManager pm(module.getContext());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    if (pm.run(module).failed()) {
      llvm::errs() << "Failed to run canonicalization passes\n";
    }
  }

  /**
   * @brief Build expected Quartz IR programmatically and run canonicalization
   */
  [[nodiscard]] OwningOpRef<ModuleOp> buildQuartzIR(
      const std::function<void(quartz::QuartzProgramBuilder&)>& buildFunc)
      const {
    quartz::QuartzProgramBuilder builder(context.get());
    builder.initialize();
    buildFunc(builder);
    auto module = builder.finalize();
    runCanonicalizationPasses(module.get());
    return module;
  }

  /**
   * @brief Build expected Flux IR programmatically and run canonicalization
   */
  [[nodiscard]] OwningOpRef<ModuleOp> buildFluxIR(
      const std::function<void(flux::FluxProgramBuilder&)>& buildFunc) const {
    flux::FluxProgramBuilder builder(context.get());
    builder.initialize();
    buildFunc(builder);
    auto module = builder.finalize();
    runCanonicalizationPasses(module.get());
    return module;
  }

  /**
   * @brief Build expected QIR programmatically and run canonicalization
   */
  [[nodiscard]] OwningOpRef<ModuleOp> buildQIR(
      const std::function<void(qir::QIRProgramBuilder&)>& buildFunc) const {
    qir::QIRProgramBuilder builder(context.get());
    builder.initialize();
    buildFunc(builder);
    auto module = builder.finalize();
    runCanonicalizationPasses(module.get());
    return module;
  }

  static std::string captureIR(ModuleOp module) {
    std::string result;
    llvm::raw_string_ostream os(result);
    module.print(os);
    os.flush();
    return result;
  }
};

TEST_F(SimpleConversionTest, RXQuartzToFlux) {
  auto module = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    auto thetaAttr = b.builder.getF64FloatAttr(1.0);
    auto thetaOperand = b.builder.create<arith::ConstantOp>(b.loc, thetaAttr);
    auto q = b.allocQubitRegister(1, "q");
    b.rx(thetaOperand, q[0]);
  });

  PassManager pm(module.get().getContext());
  pm.addPass(createQuartzToFlux());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createRemoveDeadValuesPass());
  ASSERT_TRUE(pm.run(module.get()).succeeded());

  const auto fluxResult = captureIR(module.get());
  const auto fluxExpected = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto q = b.allocQubitRegister(1, "q");
    b.rx(1.0, q[0]);
  });

  EXPECT_TRUE(verify("Quartz to Flux", fluxResult, fluxExpected.get()));
}

TEST_F(SimpleConversionTest, RXQuartzToQIR) {
  auto module = buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
    auto thetaAttr = b.builder.getF64FloatAttr(1.0);
    auto thetaOperand = b.builder.create<arith::ConstantOp>(b.loc, thetaAttr);
    auto q = b.allocQubitRegister(1, "q");
    b.rx(thetaOperand, q[0]);
  });

  PassManager pm(module.get().getContext());
  pm.addPass(createQuartzToQIR());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createRemoveDeadValuesPass());
  ASSERT_TRUE(pm.run(module.get()).succeeded());

  // const auto qirResult = captureIR(module.get());
  // const auto qirExpected = buildQIR([](qir::QIRProgramBuilder& b) {
  //   auto thetaAttr = b.builder.getF64FloatAttr(1.0);
  //   auto thetaOperand = b.builder.create<LLVM::ConstantOp>(b.loc, thetaAttr);
  //   auto q = b.allocQubitRegister(1);
  //   b.rx(thetaOperand, q[0]);
  // });

  // EXPECT_TRUE(verify("Quartz to QIR", qirResult, qirExpected.get()));
}

TEST_F(SimpleConversionTest, RXSimplifyAttrValue) {
  const auto fluxInit = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto thetaAttr = b.builder.getF64FloatAttr(1.0);
    auto thetaOperand = b.builder.create<arith::ConstantOp>(b.loc, thetaAttr);
    auto reg = b.allocQubitRegister(1, "q");
    auto q0 = reg[0];
    auto q1 = b.rx(0.5, q0);
    b.rx(thetaOperand, q1);
  });
  const auto fluxInitIR = captureIR(fluxInit.get());

  const auto fluxOpt = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    auto q0 = reg[0];
    b.rx(1.5, q0);
  });

  EXPECT_TRUE(verify("Flux Canonicalization", fluxInitIR, fluxOpt.get()));
}

TEST_F(SimpleConversionTest, RXSimplifyValueAttr) {
  const auto fluxInit = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto thetaAttr = b.builder.getF64FloatAttr(1.0);
    auto thetaOperand = b.builder.create<arith::ConstantOp>(b.loc, thetaAttr);
    auto reg = b.allocQubitRegister(1, "q");
    auto q0 = reg[0];
    auto q1 = b.rx(thetaOperand, q0);
    b.rx(0.5, q1);
  });
  const auto fluxInitIR = captureIR(fluxInit.get());

  const auto fluxOpt = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    auto q0 = reg[0];
    b.rx(1.5, q0);
  });

  EXPECT_TRUE(verify("Flux Canonicalization", fluxInitIR, fluxOpt.get()));
}

TEST_F(SimpleConversionTest, RXSimplifyValueValue) {
  const auto fluxInit = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto thetaAttr10 = b.builder.getF64FloatAttr(1.0);
    auto thetaOperand10 =
        b.builder.create<arith::ConstantOp>(b.loc, thetaAttr10);
    auto thetaAttr05 = b.builder.getF64FloatAttr(0.5);
    auto thetaOperand05 =
        b.builder.create<arith::ConstantOp>(b.loc, thetaAttr05);
    auto reg = b.allocQubitRegister(1, "q");
    auto q0 = reg[0];
    auto q1 = b.rx(thetaOperand10, q0);
    b.rx(thetaOperand05, q1);
  });
  const auto fluxInitIR = captureIR(fluxInit.get());

  const auto fluxOpt = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    auto q0 = reg[0];
    b.rx(1.5, q0);
  });

  EXPECT_TRUE(verify("Flux Canonicalization", fluxInitIR, fluxOpt.get()));
}

TEST_F(SimpleConversionTest, CX) {
  qc::QuantumComputation qc;
  qc.addQubitRegister(2, "q");
  qc.cx(0, 1);

  const auto module = translateQuantumComputationToQuartz(context.get(), qc);
  ASSERT_TRUE(module);
  runCanonicalizationPasses(module.get());
  const auto quartzIRInit = captureIR(module.get());

  const auto quartzExpected =
      buildQuartzIR([](quartz::QuartzProgramBuilder& b) {
        auto reg = b.allocQubitRegister(2, "q");
        auto q0 = reg[0];
        auto q1 = reg[1];
        b.ctrl(q0, [&](auto& b) { b.x(q1); });
      });

  EXPECT_TRUE(verify("Translation", quartzIRInit, quartzExpected.get()));

  PassManager pm(module.get().getContext());
  pm.addPass(createQuartzToFlux());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createRemoveDeadValuesPass());
  ASSERT_TRUE(pm.run(module.get()).succeeded());
  const auto fluxIR = captureIR(module.get());

  const auto fluxExpected = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    auto q0 = reg[0];
    auto q1 = reg[1];
    b.ctrl({q0}, {q1}, [&](auto& b) {
      auto q1res = b.x(q1);
      return SmallVector<Value>{q1res};
    });
  });

  EXPECT_TRUE(verify("Quartz to Flux", fluxIR, fluxExpected.get()));

  pm.clear();
  pm.addPass(createFluxToQuartz());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createRemoveDeadValuesPass());
  ASSERT_TRUE(pm.run(module.get()).succeeded());
  const auto quartzIRConv = captureIR(module.get());

  EXPECT_TRUE(verify("Flux to Quartz", quartzIRConv, quartzExpected.get()));
}

TEST_F(SimpleConversionTest, CXMergeNested) {
  const auto fluxInit = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    auto q0 = reg[0];
    auto q1 = reg[1];
    auto q2 = reg[2];
    b.ctrl({q0}, {q1, q2}, [&](auto& b) {
      auto q12res = b.ctrl({q1}, {q2}, [&](auto& b) {
        auto q2res = b.x(q2);
        return SmallVector<Value>{q2res};
      });
      return SmallVector<Value>{q12res.first[0], q12res.second[0]};
    });
  });
  const auto fluxInitIR = captureIR(fluxInit.get());

  const auto fluxOpt = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto reg = b.allocQubitRegister(3, "q");
    auto q0 = reg[0];
    auto q1 = reg[1];
    auto q2 = reg[2];
    b.ctrl({q0, q1}, {q2}, [&](auto& b) {
      auto q2res = b.x(q2);
      return SmallVector<Value>{q2res};
    });
  });

  EXPECT_TRUE(verify("Flux Canonicalization", fluxInitIR, fluxOpt.get()));
}

TEST_F(SimpleConversionTest, CXRemoveTrivial) {
  const auto fluxInit = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    auto q0 = reg[0];
    b.ctrl({}, {q0}, [&](auto& b) {
      auto q0res = b.x(q0);
      return SmallVector<Value>{q0res};
    });
  });
  const auto fluxInitIR = captureIR(fluxInit.get());

  const auto fluxOpt = buildFluxIR([](flux::FluxProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    auto q0a = reg[0];
    b.x(q0a);
  });

  EXPECT_TRUE(verify("Flux Canonicalization", fluxInitIR, fluxOpt.get()));
}

} // namespace
