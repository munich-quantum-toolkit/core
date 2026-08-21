/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Compiler/TargetCompilation.h"

#include "mlir/Compiler/Target.h"
#include "mlir/Dialect/MQT/Utils/ConstantFolding.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Mapping/Mapping.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"
#include "mlir/Support/Passes.h"

#include <llvm/ADT/APSInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/TypeID.h>
#include <mlir/Transforms/Passes.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>

namespace mlir {

using ClassicalControl = CompilerTarget::ClassicalControl;

[[nodiscard]] static constexpr llvm::StringRef
classicalControlName(const ClassicalControl control) {
  switch (control) {
  case ClassicalControl::Conditional:
    return "conditional";
  case ClassicalControl::Iteration:
    return "iteration";
  case ClassicalControl::ConditionalLoop:
    return "conditional-loop";
  case ClassicalControl::MultiwayBranch:
    return "multiway-branch";
  }
  llvm_unreachable("unknown classical-control capability");
}

[[nodiscard]] static std::optional<ClassicalControl>
requiredClassicalControl(Operation* operation) {
  if (llvm::isa<qco::IfOp, scf::IfOp>(operation)) {
    return ClassicalControl::Conditional;
  }
  if (llvm::isa<scf::ForOp>(operation)) {
    return ClassicalControl::Iteration;
  }
  if (llvm::isa<scf::WhileOp>(operation)) {
    return ClassicalControl::ConditionalLoop;
  }
  if (llvm::isa<qco::IndexSwitchOp, scf::IndexSwitchOp>(operation)) {
    return ClassicalControl::MultiwayBranch;
  }
  return std::nullopt;
}

[[nodiscard]] static bool isQubitTensor(Type type) {
  const auto tensor = llvm::dyn_cast<RankedTensorType>(type);
  return tensor && llvm::isa<qco::QubitType>(tensor.getElementType());
}

[[nodiscard]] static bool hasUnsupportedQubitTensorState(Operation* operation) {
  if (!llvm::isa<qco::IfOp, scf::IfOp, scf::ForOp, scf::WhileOp,
                 qco::IndexSwitchOp, scf::IndexSwitchOp>(operation)) {
    return false;
  }

  return llvm::any_of(operation->getOperandTypes(), isQubitTensor) ||
         llvm::any_of(operation->getResultTypes(), isQubitTensor);
}

namespace {

constexpr unsigned noQuantumDefinition = std::numeric_limits<unsigned>::max();

struct OperationSummary {
  bool containsQuantumState = false;
  bool capturesQuantumState = false;
  bool hasDynamicQubitIndex = false;
  bool hasZeroTripCount = false;
  std::optional<int64_t> staticSelector;
  unsigned minimumQuantumDefinitionDepth = noQuantumDefinition;
};

class TargetControlAnalysis {
public:
  explicit TargetControlAnalysis(Operation* root) { analyzeOperation(root, 0); }

  [[nodiscard]] const OperationSummary& get(Operation* operation) const {
    return summaries.at(operation);
  }

private:
  [[nodiscard]] static bool isQuantumType(const Type type) {
    return llvm::isa<qco::QubitType>(type) || isQubitTensor(type);
  }

  [[nodiscard]] std::optional<Attribute> fold(Value value) {
    return mqt::valueToConstantAttr(value, constantCache);
  }

  [[nodiscard]] std::optional<int64_t> foldInteger(Value value) {
    const auto attr = fold(value);
    if (!attr) {
      return std::nullopt;
    }
    const auto integer = llvm::dyn_cast<IntegerAttr>(*attr);
    if (!integer || integer.getValue().getBitWidth() > 64) {
      return std::nullopt;
    }
    return integer.getValue().getSExtValue();
  }

  [[nodiscard]] bool hasZeroTripCount(scf::ForOp operation) {
    const auto lowerBound = fold(operation.getLowerBound());
    const auto upperBound = fold(operation.getUpperBound());
    const auto step = fold(operation.getStep());
    if (!lowerBound || !upperBound || !step) {
      return false;
    }
    const auto tripCount = constantTripCount(
        *lowerBound, *upperBound, *step, !operation.getUnsignedCmp(),
        [](Value, Value, bool) -> std::optional<llvm::APSInt> {
          return std::nullopt;
        });
    return tripCount && tripCount->isZero();
  }

  static void merge(OperationSummary& summary, const OperationSummary& nested) {
    summary.containsQuantumState |= nested.containsQuantumState;
    summary.minimumQuantumDefinitionDepth =
        std::min(summary.minimumQuantumDefinitionDepth,
                 nested.minimumQuantumDefinitionDepth);
  }

  [[nodiscard]] OperationSummary analyzeRegion(Region& region,
                                               const unsigned depth) {
    regionDepths[&region] = depth;
    OperationSummary summary;
    for (Block& block : region) {
      for (Operation& operation : block) {
        merge(summary, analyzeOperation(&operation, depth));
      }
    }
    return summary;
  }

  [[nodiscard]] OperationSummary analyzeOperation(Operation* operation,
                                                  const unsigned depth) {
    OperationSummary summary;
    OperationSummary nestedSummary;
    if (auto extract = llvm::dyn_cast<qtensor::ExtractOp>(operation)) {
      summary.hasDynamicQubitIndex = !foldInteger(extract.getIndex());
    } else if (auto insert = llvm::dyn_cast<qtensor::InsertOp>(operation)) {
      summary.hasDynamicQubitIndex = !foldInteger(insert.getIndex());
    }
    summary.containsQuantumState =
        llvm::any_of(operation->getOperandTypes(), isQuantumType) ||
        llvm::any_of(operation->getResultTypes(), isQuantumType);
    for (Value operand : operation->getOperands()) {
      if (!isQuantumType(operand.getType())) {
        continue;
      }
      const auto definition = regionDepths.find(operand.getParentRegion());
      assert(definition != regionDepths.end());
      summary.minimumQuantumDefinitionDepth =
          std::min(summary.minimumQuantumDefinitionDepth, definition->second);
    }

    const auto analyzeNestedRegion = [&](Region& region) {
      merge(nestedSummary, analyzeRegion(region, depth + 1));
    };

    if (auto ifOp = llvm::dyn_cast<qco::IfOp>(operation)) {
      summary.staticSelector = foldInteger(ifOp.getCondition());
      if (summary.staticSelector) {
        analyzeNestedRegion(*summary.staticSelector != 0
                                ? ifOp.getThenRegion()
                                : ifOp.getElseRegion());
      } else {
        llvm::for_each(operation->getRegions(), analyzeNestedRegion);
      }
    } else if (auto ifOp = llvm::dyn_cast<scf::IfOp>(operation)) {
      summary.staticSelector = foldInteger(ifOp.getCondition());
      if (summary.staticSelector) {
        analyzeNestedRegion(*summary.staticSelector != 0
                                ? ifOp.getThenRegion()
                                : ifOp.getElseRegion());
      } else {
        llvm::for_each(operation->getRegions(), analyzeNestedRegion);
      }
    } else if (auto switchOp = llvm::dyn_cast<qco::IndexSwitchOp>(operation)) {
      summary.staticSelector = foldInteger(switchOp.getArg());
      if (summary.staticSelector) {
        analyzeSelectedSwitchRegion(switchOp, *summary.staticSelector,
                                    analyzeNestedRegion);
      } else {
        llvm::for_each(operation->getRegions(), analyzeNestedRegion);
      }
    } else if (auto switchOp = llvm::dyn_cast<scf::IndexSwitchOp>(operation)) {
      summary.staticSelector = foldInteger(switchOp.getArg());
      if (summary.staticSelector) {
        analyzeSelectedSwitchRegion(switchOp, *summary.staticSelector,
                                    analyzeNestedRegion);
      } else {
        llvm::for_each(operation->getRegions(), analyzeNestedRegion);
      }
    } else if (auto forOp = llvm::dyn_cast<scf::ForOp>(operation);
               forOp && hasZeroTripCount(forOp)) {
      summary.hasZeroTripCount = true;
    } else {
      llvm::for_each(operation->getRegions(), analyzeNestedRegion);
    }

    summary.capturesQuantumState =
        nestedSummary.minimumQuantumDefinitionDepth <= depth;
    merge(summary, nestedSummary);
    summaries[operation] = summary;
    return summary;
  }

  template <class SwitchOp, class Callback>
  static void analyzeSelectedSwitchRegion(SwitchOp operation,
                                          const int64_t selector,
                                          Callback&& callback) {
    for (const auto [caseIndex, caseValue] :
         llvm::enumerate(operation.getCases())) {
      if (caseValue == selector) {
        callback(operation.getCaseRegions()[caseIndex]);
        return;
      }
    }
    callback(operation.getDefaultRegion());
  }

  llvm::DenseMap<Value, std::optional<Attribute>> constantCache;
  llvm::DenseMap<Region*, unsigned> regionDepths;
  llvm::DenseMap<Operation*, OperationSummary> summaries;
};

struct VerifyTargetClassicalControlPass final
    : PassWrapper<VerifyTargetClassicalControlPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyTargetClassicalControlPass)

  explicit VerifyTargetClassicalControlPass(const CompilerTarget& targetIn)
      : target(targetIn) {}

protected:
  void runOnOperation() override {
    const TargetControlAnalysis analysis(getOperation());
    if (failed(verifyNestedRegions(getOperation(), analysis))) {
      signalPassFailure();
    }
  }

private:
  [[nodiscard]] LogicalResult
  verifyRegion(Region& region, const TargetControlAnalysis& analysis) const {
    for (Block& block : region) {
      for (Operation& operation : block) {
        if (failed(verifyOperation(&operation, analysis))) {
          return failure();
        }
      }
    }
    return success();
  }

  [[nodiscard]] LogicalResult
  verifyNestedRegions(Operation* operation,
                      const TargetControlAnalysis& analysis) const {
    for (Region& region : operation->getRegions()) {
      if (failed(verifyRegion(region, analysis))) {
        return failure();
      }
    }
    return success();
  }

  template <class SwitchOp>
  [[nodiscard]] LogicalResult
  verifySelectedSwitchRegion(SwitchOp operation, const int64_t selector,
                             const TargetControlAnalysis& analysis) const {
    for (const auto [caseIndex, caseValue] :
         llvm::enumerate(operation.getCases())) {
      if (caseValue == selector) {
        return verifyRegion(operation.getCaseRegions()[caseIndex], analysis);
      }
    }
    return verifyRegion(operation.getDefaultRegion(), analysis);
  }

  [[nodiscard]] LogicalResult
  verifyOperation(Operation* operation,
                  const TargetControlAnalysis& analysis) const {
    if (hasUnsupportedQubitTensorState(operation)) {
      operation->emitError()
          << "target compilation cannot lower quantum tensor state carried "
             "through classical-control construct '"
          << operation->getName() << "'";
      return failure();
    }

    const OperationSummary& summary = analysis.get(operation);
    if (auto ifOp = llvm::dyn_cast<qco::IfOp>(operation)) {
      if (summary.staticSelector) {
        return verifyRegion(*summary.staticSelector != 0 ? ifOp.getThenRegion()
                                                         : ifOp.getElseRegion(),
                            analysis);
      }
    } else if (auto ifOp = llvm::dyn_cast<scf::IfOp>(operation)) {
      if (summary.staticSelector) {
        return verifyRegion(*summary.staticSelector != 0 ? ifOp.getThenRegion()
                                                         : ifOp.getElseRegion(),
                            analysis);
      }
    } else if (auto switchOp = llvm::dyn_cast<qco::IndexSwitchOp>(operation)) {
      if (summary.staticSelector) {
        return verifySelectedSwitchRegion(switchOp, *summary.staticSelector,
                                          analysis);
      }
    } else if (auto switchOp = llvm::dyn_cast<scf::IndexSwitchOp>(operation)) {
      if (summary.staticSelector) {
        return verifySelectedSwitchRegion(switchOp, *summary.staticSelector,
                                          analysis);
      }
    } else if (llvm::isa<scf::ForOp>(operation) && summary.hasZeroTripCount) {
      return success();
    }

    if (summary.hasDynamicQubitIndex) {
      operation->emitError()
          << "target compilation cannot lower '" << operation->getName()
          << "' with a dynamic qubit index";
      return failure();
    }

    if (const auto required = requiredClassicalControl(operation)) {
      if (!target.supportsClassicalControl(*required)) {
        operation->emitError()
            << "target compilation does not support classical-control "
               "capability '"
            << classicalControlName(*required) << "' required by '"
            << operation->getName() << "'";
        return failure();
      }
      if (summary.capturesQuantumState) {
        operation->emitError()
            << "target compilation cannot lower quantum state captured by "
               "classical-control construct '"
            << operation->getName() << "'";
        return failure();
      }
      if (llvm::isa<scf::IfOp, scf::IndexSwitchOp>(operation) &&
          summary.containsQuantumState) {
        operation->emitError()
            << "target compilation cannot lower quantum state nested in "
               "generic classical-control construct '"
            << operation->getName() << "'";
        return failure();
      }
    } else if (llvm::isa<BranchOpInterface, RegionBranchOpInterface>(
                   operation)) {
      operation->emitError()
          << "target compilation cannot lower classical-control construct '"
          << operation->getName() << "'";
      return failure();
    }

    return verifyNestedRegions(operation, analysis);
  }

  CompilerTarget target;
};

} // namespace

void populateTargetCompilationPipeline(OpPassManager& pm,
                                       const CompilerTarget& target) {
  pm.addPass(std::make_unique<VerifyTargetClassicalControlPass>(target));
  populateQCOCleanupPipeline(pm);
  populateDecomposeMultiControlledPipeline(pm, 3);
  populateDefaultQCOOptimizationPipeline(pm);
  pm.addPass(qco::createFuseTwoQubitGates());
  pm.addPass(qco::createMappingPass(target, qco::MappingPassOptions{}));
  populateQCOCleanupPipeline(pm);
  pm.addPass(qco::createTargetNativeSynthesis(target));
  pm.addPass(createCSEPass());
  pm.addPass(createRemoveDeadValuesPass());
  pm.addPass(qco::createVerifyTargetConformance(target));
}

} // namespace mlir
