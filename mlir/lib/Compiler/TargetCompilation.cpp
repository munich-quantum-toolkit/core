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
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/QCOUtils.h"
#include "mlir/Dialect/QCO/Transforms/Mapping/Mapping.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"
#include "mlir/Support/Passes.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/TypeID.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/Passes.h>

#include <cstdint>
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

[[nodiscard]] static bool hasDynamicQubitIndex(Operation* operation) {
  if (auto extract = llvm::dyn_cast<qtensor::ExtractOp>(operation)) {
    return !getConstantIntValue(extract.getIndex());
  }
  if (auto insert = llvm::dyn_cast<qtensor::InsertOp>(operation)) {
    return !getConstantIntValue(insert.getIndex());
  }
  return false;
}

[[nodiscard]] static bool isQubitTensor(const Type type) {
  const auto tensor = llvm::dyn_cast<RankedTensorType>(type);
  return tensor && llvm::isa<qco::QubitType>(tensor.getElementType());
}

[[nodiscard]] static bool capturesQuantumState(Operation* operation) {
  bool captured = false;
  operation->walk([&](Operation* nested) {
    if (nested == operation) {
      return WalkResult::advance();
    }
    for (Value operand : nested->getOperands()) {
      if (!llvm::isa<qco::QubitType>(operand.getType()) &&
          !isQubitTensor(operand.getType())) {
        continue;
      }
      const bool definedInside =
          llvm::any_of(operation->getRegions(), [&](Region& region) {
            return region.isAncestor(operand.getParentRegion());
          });
      if (!definedInside) {
        captured = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return captured;
}

[[nodiscard]] static bool containsQuantumState(Operation* operation) {
  bool contains = false;
  operation->walk([&](Operation* nested) {
    const auto isQuantum = [](const Type type) {
      return llvm::isa<qco::QubitType>(type) || isQubitTensor(type);
    };
    if (llvm::any_of(nested->getOperandTypes(), isQuantum) ||
        llvm::any_of(nested->getResultTypes(), isQuantum)) {
      contains = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return contains;
}

[[nodiscard]] static bool hasUnsupportedQubitTensorState(Operation* operation) {
  if (const auto ifOp = llvm::dyn_cast<qco::IfOp>(operation)) {
    return !qco::hasOnlyScalarizableTensorInputs(ifOp);
  }
  if (!llvm::isa<scf::IfOp, scf::ForOp, scf::WhileOp, qco::IndexSwitchOp,
                 scf::IndexSwitchOp>(operation)) {
    return false;
  }

  return llvm::any_of(operation->getOperandTypes(), isQubitTensor) ||
         llvm::any_of(operation->getResultTypes(), isQubitTensor);
}

namespace {

struct VerifyTargetClassicalControlPass final
    : PassWrapper<VerifyTargetClassicalControlPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyTargetClassicalControlPass)

  explicit VerifyTargetClassicalControlPass(const CompilerTarget& targetIn)
      : target(targetIn) {}

protected:
  void runOnOperation() override {
    if (failed(verifyNestedRegions(getOperation()))) {
      signalPassFailure();
    }
  }

private:
  [[nodiscard]] LogicalResult verifyRegion(Region& region) const {
    for (Block& block : region) {
      for (Operation& operation : block) {
        if (failed(verifyOperation(&operation))) {
          return failure();
        }
      }
    }
    return success();
  }

  [[nodiscard]] LogicalResult verifyNestedRegions(Operation* operation) const {
    for (Region& region : operation->getRegions()) {
      if (failed(verifyRegion(region))) {
        return failure();
      }
    }
    return success();
  }

  template <class SwitchOp>
  [[nodiscard]] LogicalResult
  verifySelectedSwitchRegion(SwitchOp operation, const int64_t selector) const {
    for (const auto [caseIndex, caseValue] :
         llvm::enumerate(operation.getCases())) {
      if (caseValue == selector) {
        return verifyRegion(operation.getCaseRegions()[caseIndex]);
      }
    }
    return verifyRegion(operation.getDefaultRegion());
  }

  [[nodiscard]] LogicalResult verifyOperation(Operation* operation) const {
    if (hasUnsupportedQubitTensorState(operation)) {
      operation->emitError()
          << "target compilation cannot lower quantum tensor state carried "
             "through classical-control construct '"
          << operation->getName() << "'";
      return failure();
    }

    if (auto ifOp = llvm::dyn_cast<qco::IfOp>(operation)) {
      if (const auto condition = getConstantIntValue(ifOp.getCondition())) {
        return verifyRegion(*condition != 0 ? ifOp.getThenRegion()
                                            : ifOp.getElseRegion());
      }
    } else if (auto ifOp = llvm::dyn_cast<scf::IfOp>(operation)) {
      if (const auto condition = getConstantIntValue(ifOp.getCondition())) {
        return verifyRegion(*condition != 0 ? ifOp.getThenRegion()
                                            : ifOp.getElseRegion());
      }
    } else if (auto switchOp = llvm::dyn_cast<qco::IndexSwitchOp>(operation)) {
      if (const auto selector = getConstantIntValue(switchOp.getArg())) {
        return verifySelectedSwitchRegion(switchOp, *selector);
      }
    } else if (auto switchOp = llvm::dyn_cast<scf::IndexSwitchOp>(operation)) {
      if (const auto selector = getConstantIntValue(switchOp.getArg())) {
        return verifySelectedSwitchRegion(switchOp, *selector);
      }
    }

    if (hasDynamicQubitIndex(operation)) {
      operation->emitError()
          << "target compilation cannot lower classical-control construct '"
          << operation->getName() << "' with a dynamic qubit index";
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
      if (capturesQuantumState(operation)) {
        operation->emitError()
            << "target compilation cannot lower quantum state captured by "
               "classical-control construct '"
            << operation->getName() << "'";
        return failure();
      }
      if (llvm::isa<scf::IfOp, scf::IndexSwitchOp>(operation) &&
          containsQuantumState(operation)) {
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

    return verifyNestedRegions(operation);
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
