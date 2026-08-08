/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/QCOUtils.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/QTensor/IR/QTensorOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>

using namespace mlir;
using namespace mlir::qco;

/**
 * @brief Materialize a global phase controlled by multiple @p controls.
 * @return The updated control qubits in their original order.
 */
static SmallVector<Value> createMultiControlledPhase(PatternRewriter& rewriter,
                                                     Location controlledLoc,
                                                     Location phaseLoc,
                                                     ValueRange controls,
                                                     Value theta) {
  assert(controls.size() > 1);
  auto controlledPhase = CtrlOp::create(
      rewriter, controlledLoc, controls.drop_back(), controls.back(),
      [&](Value target) -> Value {
        return POp::create(rewriter, phaseLoc, target, theta).getOutputQubit(0);
      });
  return SmallVector<Value>(controlledPhase.getOutputQubits());
}

namespace {

/**
 * @brief Merge nested control modifiers into a single one.
 */
struct MergeNestedCtrl final : OpRewritePattern<CtrlOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    // Require at least one control
    // Trivial case is handled by ReduceCtrl
    if (op.getNumControls() == 0) {
      return failure();
    }

    // Only proceed if body contains only one operation besides terminator
    if (op.getBody()->getOperations().size() != 2) {
      return failure();
    }

    auto inner = utils::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (!inner) {
      return failure();
    }
    auto innerCtrlOp = dyn_cast<CtrlOp>(inner.getOperation());
    if (!innerCtrlOp) {
      return failure();
    }

    // The inner control's controls and targets are block arguments of the outer
    // body that alias outer targets. Re-resolve them to the outer qubits: inner
    // controls join the outer controls, inner targets become the merged
    // targets. Inner-target order is kept so the inner body's block arguments
    // line up with the merged targets and the body can be reused verbatim.
    auto outerTargets = op.getTargetsIn();
    auto innerControls = innerCtrlOp.getControlsIn();
    auto innerTargets = innerCtrlOp.getTargetsIn();

    SmallVector<Value> controls(op.getControlsIn());
    for (auto control : innerControls) {
      controls.push_back(
          utils::getValueFromBlockArgument(control, outerTargets));
    }
    const auto targets = llvm::map_to_vector(innerTargets, [&](Value t) {
      return utils::getValueFromBlockArgument(t, outerTargets);
    });

    auto merged =
        CtrlOp::create(rewriter, op.getLoc(), controls, targets,
                       [&](ValueRange mergedTargets) -> SmallVector<Value> {
                         return utils::inlineBodyReturningYields(
                             *innerCtrlOp.getBody(), mergedTargets, rewriter);
                       });

    // Every qubit output of the original control follows its input qubit to the
    // corresponding output of the merged control.
    rewriter.replaceOp(op,
                       llvm::map_to_vector(op.getInputQubits(), [&](Value in) {
                         return merged.getOutputForInput(in);
                       }));
    return success();
  }
};

/**
 * @brief Pull global phases out of multi-operation control modifiers.
 */
struct PullGPhaseOutOfCtrl final : OpRewritePattern<CtrlOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getNumControls() == 0 || op.getNumBodyUnitaries() < 2) {
      return failure();
    }

    SmallVector<GPhaseOp> globalPhases;
    for (auto gphase : op.getBody()->getOps<GPhaseOp>()) {
      // Moving the phase out must not leave its angle defined in the body.
      if (gphase.getTheta().getParentBlock() == op.getBody()) {
        return failure();
      }
      globalPhases.push_back(gphase);
    }
    if (globalPhases.empty()) {
      return failure();
    }

    SmallVector<Value> controls(op.getControlsIn());
    const OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    Value theta = globalPhases.front().getTheta();
    for (auto gphase : llvm::drop_begin(globalPhases)) {
      theta = rewriter.createOrFold<arith::AddFOp>(gphase.getLoc(), theta,
                                                   gphase.getTheta());
    }
    const auto phaseLoc = globalPhases.front().getLoc();
    if (controls.size() == 1) {
      controls.front() =
          POp::create(rewriter, phaseLoc, controls.front(), theta)
              .getOutputQubit(0);
    } else {
      controls = createMultiControlledPhase(rewriter, phaseLoc, phaseLoc,
                                            controls, theta);
    }
    for (auto gphase : globalPhases) {
      rewriter.eraseOp(gphase);
    }

    rewriter.modifyOpInPlace(op, [&]() {
      for (auto [index, control] : llvm::enumerate(controls)) {
        op->setOperand(index, control);
      }
    });
    return success();
  }
};

/**
 * @brief Reduce controls for well-known gates.
 * @details Removes empty control ops and handles controlled IdOp, GPhaseOp and
 * BarrierOp.
 */
struct ReduceCtrl final : OpRewritePattern<CtrlOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    auto inner = utils::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (!inner) {
      return failure();
    }
    auto* innerOp = inner.getOperation();

    // Inline ops from empty control modifiers, IdOp and BarrierOp
    if (op.getNumControls() == 0 || isa<IdOp, BarrierOp>(innerOp)) {
      auto* body = op.getBody();
      auto* terminator = body->getTerminator();
      // Controls are pass-through results outside the body yield, so the
      // generic inlineModifierBody result mapping does not apply here.
      SmallVector<Value> outputs(op.getControlsIn());
      llvm::append_range(outputs, terminator->getOperands());
      rewriter.inlineBlockBefore(body, op, op.getTargetsIn());
      rewriter.eraseOp(terminator);
      rewriter.replaceOp(op, outputs);
      return success();
    }

    // The remaining code explicitly handles GPhaseOp and nothing else
    auto gPhaseOp = dyn_cast<GPhaseOp>(innerOp);
    if (!gPhaseOp) {
      return failure();
    }

    // Only proceed if the GPhaseOp is the only operation besides the terminator
    if (op.getBody()->getOperations().size() != 2) {
      return failure();
    }

    const OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    if (op.getNumControls() == 1) {
      auto output = POp::create(rewriter, op.getLoc(),
                                op.getControlsIn().front(), gPhaseOp.getTheta())
                        .getOutputQubit(0);
      rewriter.replaceOp(op, output);
    } else {
      auto outputs =
          createMultiControlledPhase(rewriter, op.getLoc(), gPhaseOp.getLoc(),
                                     op.getControlsIn(), gPhaseOp.getTheta());
      rewriter.replaceOp(op, outputs);
    }
    return success();
  }
};

/**
 * @brief Erase control modifiers that do not have any body unitaries.
 */
struct EraseEmptyCtrl final : OpRewritePattern<CtrlOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getNumBodyUnitaries() != 0) {
      return failure();
    }

    rewriter.replaceOp(op, op.getOperands());
    return success();
  }
};

} // namespace

static void
buildModifierBody(OpBuilder& odsBuilder, OperationState& odsState,
                  const size_t numBlockArgs,
                  const function_ref<void(OpBuilder&, Block&)>& emitBody) {
  auto& block = odsState.regions.front()->emplaceBlock();
  const auto qubitType = QubitType::get(odsBuilder.getContext());
  for (size_t i = 0; i < numBlockArgs; ++i) {
    block.addArgument(qubitType, odsState.location);
  }

  const OpBuilder::InsertionGuard guard(odsBuilder);
  odsBuilder.setInsertionPointToStart(&block);
  emitBody(odsBuilder, block);
}

size_t CtrlOp::getNumBodyUnitaries() {
  return utils::getNumBodyUnitaries<UnitaryOpInterface>(*getBody());
}

UnitaryOpInterface CtrlOp::getBodyUnitary(const size_t i) {
  return utils::getBodyUnitary<UnitaryOpInterface>(*getBody(), i);
}

Value CtrlOp::getInputForOutput(Value output) {
  if (const auto result = dyn_cast<OpResult>(output);
      result && result.getOwner() == getOperation()) {
    return getInputQubit(result.getResultNumber());
  }
  llvm::reportFatalUsageError("Given qubit is not an output of the operation");
}

Value CtrlOp::getOutputForInput(Value input) {
  for (auto [in, out] : llvm::zip_equal(getInputQubits(), getOutputQubits())) {
    if (in == input) {
      return out;
    }
  }
  llvm::reportFatalUsageError("Given qubit is not an input of the operation");
}

void CtrlOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                   ValueRange controls, ValueRange targets,
                   function_ref<SmallVector<Value>(ValueRange)> bodyBuilder) {
  build(odsBuilder, odsState, controls, targets);
  buildModifierBody(odsBuilder, odsState, targets.size(),
                    [&](OpBuilder& builder, Block& block) {
                      YieldOp::create(builder, odsState.location,
                                      bodyBuilder(block.getArguments()));
                    });
}

void CtrlOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                   ValueRange controls, Value target,
                   function_ref<Value(Value)> bodyBuilder) {
  build(odsBuilder, odsState, controls.getTypes(), target.getType(), controls,
        target);
  buildModifierBody(odsBuilder, odsState, 1,
                    [&](OpBuilder& builder, Block& block) {
                      YieldOp::create(builder, odsState.location,
                                      bodyBuilder(block.getArgument(0)));
                    });
}

void CtrlOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                   Value control, Value target,
                   function_ref<Value(Value)> bodyBuilder) {
  build(odsBuilder, odsState, ValueRange{control}, target, bodyBuilder);
}

LogicalResult CtrlOp::verify() {
  auto& block = *getBody();
  if (llvm::any_of(block, [](Operation& op) {
        return isa<AllocOp, SinkOp, MeasureOp, ResetOp, qtensor::ExtractOp,
                   qtensor::InsertOp>(op);
      })) {
    return emitOpError("body must not contain non-unitary quantum operations "
                       "or modify a quantum register");
  }

  const auto numTargets = getNumTargets();
  if (block.getArguments().size() != numTargets) {
    return emitOpError(
        "number of block arguments must match the number of targets");
  }
  auto qubitType = QubitType::get(getContext());
  for (size_t i = 0; i < numTargets; ++i) {
    if (block.getArgument(i).getType() != qubitType) {
      return emitOpError("block argument type at index ")
             << i << " does not match target type";
    }
  }
  auto* blockTerminator = block.getTerminator();
  if (const auto numYieldOperands = blockTerminator->getNumOperands();
      numYieldOperands != numTargets) {
    return emitOpError("yield operation must yield ")
           << numTargets << " values, but found " << numYieldOperands;
  }

  SmallPtrSet<Value, 4> uniqueQubitsIn;
  for (const auto& control : getInputQubits()) {
    if (!uniqueQubitsIn.insert(control).second) {
      return emitOpError("duplicate qubit found");
    }
  }

  SmallPtrSet<Value, 4> uniqueQubitsOut;
  for (const auto& control : getControlsOut()) {
    if (!uniqueQubitsOut.insert(control).second) {
      return emitOpError("duplicate control qubit found");
    }
  }

  for (size_t i = 0; i < numTargets; i++) {
    if (!uniqueQubitsOut.insert(blockTerminator->getOperand(i)).second) {
      return emitOpError("duplicate qubit found");
    }
  }

  return success();
}

void CtrlOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* context) {
  results.add<MergeNestedCtrl, PullGPhaseOutOfCtrl, ReduceCtrl, EraseEmptyCtrl>(
      context);
}

bool CtrlOp::hasCompileTimeKnownUnitaryMatrix() {
  return all_of(getBody()->getOps<UnitaryOpInterface>(),
                [](UnitaryOpInterface op) {
                  return op.hasCompileTimeKnownUnitaryMatrix();
                });
}

std::optional<DynamicMatrix> CtrlOp::getUnitaryMatrix() {
  if (getNumControls() >= 32) {
    llvm::reportFatalUsageError(
        "Creating the unitary matrix for a CtrlOp with more than 31 controls "
        "is not supported due to memory constraints.");
  }

  const auto numControls = getNumControls();

  // Build `I_{2^controls} ⊗ U` by placing the target block in the bottom-right
  // corner of a `2^controls * targetDim` identity.
  const auto controlledMatrix =
      [numControls](const int64_t targetDim,
                    const auto& targetBlock) -> DynamicMatrix {
    auto matrix = DynamicMatrix::identity(static_cast<int64_t>(
        (1ULL << numControls) * static_cast<size_t>(targetDim)));
    matrix.setBottomRightCorner(targetBlock);
    return matrix;
  };

  // Single inner unitary (e.g. `ctrl { h }`, `ctrl { cx }`).
  if (auto bodyUnitary =
          utils::getSoleBodyUnitary<UnitaryOpInterface>(*getBody())) {
    if (const auto targetMatrix =
            bodyUnitary.getUnitaryMatrix<DynamicMatrix>()) {
      assert(targetMatrix->cols() == targetMatrix->rows());
      return controlledMatrix(targetMatrix->cols(), *targetMatrix);
    }
    return std::nullopt;
  }

  // Composed body (e.g., `ctrl { h; x }` or `ctrl { swap; ry }`)
  if (const auto composed = composeBodyMatrix(*getBody(), getNumTargets())) {
    return controlledMatrix(composed->rows(), *composed);
  }

  return std::nullopt;
}
