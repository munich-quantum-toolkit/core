/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/Flux/IR/FluxDialect.h" // IWYU pragma: associated
#include "mlir/Dialect/Utils/MatrixUtils.h"

// The following headers are needed for some template instantiations.
// IWYU pragma: begin_keep
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/PatternMatch.h>
// IWYU pragma: end_keep

#include <cstddef>
#include <functional>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <variant>

using namespace mlir;
using namespace mlir::flux;
using namespace mlir::utils;

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Flux/IR/FluxOpsDialect.cpp.inc"

void FluxDialect::initialize() {
  // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Flux/IR/FluxOpsTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Flux/IR/FluxOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Flux/IR/FluxOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Flux/IR/FluxInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Flux/IR/FluxOps.cpp.inc"

LogicalResult AllocOp::verify() {
  const auto registerName = getRegisterName();
  const auto registerSize = getRegisterSize();
  const auto registerIndex = getRegisterIndex();

  const auto hasSize = registerSize.has_value();
  const auto hasIndex = registerIndex.has_value();
  const auto hasName = registerName.has_value();

  if (hasName != hasSize || hasName != hasIndex) {
    return emitOpError("register attributes must all be present or all absent");
  }

  if (hasName) {
    if (*registerIndex >= *registerSize) {
      return emitOpError("register_index (")
             << *registerIndex << ") must be less than register_size ("
             << *registerSize << ")";
    }
  }
  return success();
}

LogicalResult MeasureOp::verify() {
  const auto registerName = getRegisterName();
  const auto registerSize = getRegisterSize();
  const auto registerIndex = getRegisterIndex();

  const auto hasSize = registerSize.has_value();
  const auto hasIndex = registerIndex.has_value();
  const auto hasName = registerName.has_value();

  if (hasName != hasSize || hasName != hasIndex) {
    return emitOpError("register attributes must all be present or all absent");
  }

  if (hasName) {
    if (*registerIndex >= *registerSize) {
      return emitOpError("register_index (")
             << *registerIndex << ") must be less than register_size ("
             << *registerSize << ")";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Unitary Operations
//===----------------------------------------------------------------------===//

// IdOp

DenseElementsAttr IdOp::tryGetStaticMatrix() {
  return getMatrixId(getContext());
}

// XOp

DenseElementsAttr XOp::tryGetStaticMatrix() { return getMatrixX(getContext()); }

// SOp

DenseElementsAttr SOp::tryGetStaticMatrix() { return getMatrixS(getContext()); }

// SdgOp

DenseElementsAttr SdgOp::tryGetStaticMatrix() {
  return getMatrixSdg(getContext());
}

// RXOp

DenseElementsAttr RXOp::tryGetStaticMatrix() {
  const auto& theta = getStaticParameter(getTheta());
  if (!theta) {
    return nullptr;
  }
  const auto thetaValue = theta.getValueAsDouble();
  return getMatrixRX(getContext(), thetaValue);
}

void RXOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                 const Value qubit_in, // NOLINT(*-identifier-naming)
                 const std::variant<double, Value>& theta) {
  Value thetaOperand = nullptr;
  if (std::holds_alternative<double>(theta)) {
    thetaOperand = odsBuilder.create<arith::ConstantOp>(
        odsState.location, odsBuilder.getF64FloatAttr(std::get<double>(theta)));
  } else {
    thetaOperand = std::get<Value>(theta);
  }
  build(odsBuilder, odsState, qubit_in, thetaOperand);
}

// U2Op

DenseElementsAttr U2Op::tryGetStaticMatrix() {
  const auto phi = getStaticParameter(getPhi());
  const auto lambda = getStaticParameter(getLambda());
  if (!phi || !lambda) {
    return nullptr;
  }
  const auto phiValue = phi.getValueAsDouble();
  const auto lambdaValue = lambda.getValueAsDouble();
  return getMatrixU2(getContext(), phiValue, lambdaValue);
}

void U2Op::build(OpBuilder& odsBuilder, OperationState& odsState,
                 const Value qubit_in, // NOLINT(*-identifier-naming)
                 const std::variant<double, Value>& phi,
                 const std::variant<double, Value>& lambda) {
  Value phiOperand = nullptr;
  if (std::holds_alternative<double>(phi)) {
    phiOperand = odsBuilder.create<arith::ConstantOp>(
        odsState.location, odsBuilder.getF64FloatAttr(std::get<double>(phi)));
  } else {
    phiOperand = std::get<Value>(phi);
  }

  Value lambdaOperand = nullptr;
  if (std::holds_alternative<double>(lambda)) {
    lambdaOperand = odsBuilder.create<arith::ConstantOp>(
        odsState.location,
        odsBuilder.getF64FloatAttr(std::get<double>(lambda)));
  } else {
    lambdaOperand = std::get<Value>(lambda);
  }

  build(odsBuilder, odsState, qubit_in, phiOperand, lambdaOperand);
}

// SWAPOp

DenseElementsAttr SWAPOp::tryGetStaticMatrix() {
  return getMatrixSWAP(getContext());
}

//===----------------------------------------------------------------------===//
// Modifiers
//===----------------------------------------------------------------------===//

void CtrlOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                   const ValueRange controls, const ValueRange targets,
                   UnitaryOpInterface bodyUnitary) {
  build(odsBuilder, odsState, controls, targets);
  auto& block = odsState.regions.front()->emplaceBlock();

  // Move the unitary op into the block
  const OpBuilder::InsertionGuard guard(odsBuilder);
  odsBuilder.setInsertionPointToStart(&block);
  auto* op = odsBuilder.clone(*bodyUnitary.getOperation());
  odsBuilder.create<YieldOp>(odsState.location, op->getResults());
}

void CtrlOp::build(
    OpBuilder& odsBuilder, OperationState& odsState, const ValueRange controls,
    const ValueRange targets,
    const std::function<ValueRange(OpBuilder&, ValueRange)>& bodyBuilder) {
  build(odsBuilder, odsState, controls, targets);
  auto& block = odsState.regions.front()->emplaceBlock();

  // Move the unitary op into the block
  const OpBuilder::InsertionGuard guard(odsBuilder);
  odsBuilder.setInsertionPointToStart(&block);
  auto targetsOut = bodyBuilder(odsBuilder, targets);
  odsBuilder.create<YieldOp>(odsState.location, targetsOut);
}

UnitaryOpInterface CtrlOp::getBodyUnitary() {
  return llvm::dyn_cast<UnitaryOpInterface>(&getBody().front().front());
}

size_t CtrlOp::getNumQubits() { return getNumTargets() + getNumControls(); }

size_t CtrlOp::getNumTargets() { return getTargetsIn().size(); }

size_t CtrlOp::getNumControls() {
  return getNumPosControls() + getNumNegControls();
}

size_t CtrlOp::getNumPosControls() { return getControlsIn().size(); }

size_t CtrlOp::getNumNegControls() {
  return getBodyUnitary().getNumNegControls();
}

Value CtrlOp::getInputQubit(const size_t i) {
  const auto numPosControls = getNumPosControls();
  if (i < numPosControls) {
    return getControlsIn()[i];
  }
  if (numPosControls <= i && i < getNumQubits()) {
    return getBodyUnitary().getInputQubit(i - numPosControls);
  }
  llvm::report_fatal_error("Invalid qubit index");
}

Value CtrlOp::getOutputQubit(const size_t i) {
  const auto numPosControls = getNumPosControls();
  if (i < numPosControls) {
    return getControlsOut()[i];
  }
  if (numPosControls <= i && i < getNumQubits()) {
    return getBodyUnitary().getOutputQubit(i - numPosControls);
  }
  llvm::report_fatal_error("Invalid qubit index");
}

Value CtrlOp::getInputTarget(const size_t i) { return getTargetsIn()[i]; }

Value CtrlOp::getOutputTarget(const size_t i) { return getTargetsOut()[i]; }

Value CtrlOp::getInputPosControl(const size_t i) { return getControlsIn()[i]; }

Value CtrlOp::getOutputPosControl(const size_t i) {
  return getControlsOut()[i];
}

Value CtrlOp::getInputNegControl(const size_t i) {
  return getBodyUnitary().getInputNegControl(i);
}

Value CtrlOp::getOutputNegControl(const size_t i) {
  return getBodyUnitary().getOutputNegControl(i);
}

Value CtrlOp::getInputForOutput(const Value output) {
  for (size_t i = 0; i < getNumPosControls(); ++i) {
    if (output == getControlsOut()[i]) {
      return getControlsIn()[i];
    }
  }
  for (size_t i = 0; i < getNumTargets(); ++i) {
    if (output == getTargetsOut()[i]) {
      return getTargetsIn()[i];
    }
  }
  llvm::report_fatal_error("Given qubit is not an output of the operation");
}

Value CtrlOp::getOutputForInput(const Value input) {
  for (size_t i = 0; i < getNumPosControls(); ++i) {
    if (input == getControlsIn()[i]) {
      return getControlsOut()[i];
    }
  }
  for (size_t i = 0; i < getNumTargets(); ++i) {
    if (input == getTargetsIn()[i]) {
      return getTargetsOut()[i];
    }
  }
  llvm::report_fatal_error("Given qubit is not an input of the operation");
}

size_t CtrlOp::getNumParams() { return getBodyUnitary().getNumParams(); }

bool CtrlOp::hasStaticUnitary() { return getBodyUnitary().hasStaticUnitary(); }

Value CtrlOp::getParameter(const size_t i) {
  return getBodyUnitary().getParameter(i);
}

DenseElementsAttr CtrlOp::tryGetStaticMatrix() {
  return getMatrixCtrl(getContext(), getNumPosControls(),
                       getBodyUnitary().tryGetStaticMatrix());
}

LogicalResult CtrlOp::verify() {
  auto& block = getBody().front();
  if (block.getOperations().size() != 2) {
    return emitOpError("body region must have exactly two operations");
  }
  if (!llvm::isa<UnitaryOpInterface>(block.front())) {
    return emitOpError(
        "first operation in body region must be a unitary operation");
  }
  if (!llvm::isa<YieldOp>(block.back())) {
    return emitOpError(
        "second operation in body region must be a yield operation");
  }
  // The yield operation must yield as many values as there are targets
  if (block.back().getNumOperands() != getNumTargets()) {
    return emitOpError("yield operation must yield ")
           << getNumTargets() << " values, but found "
           << block.back().getNumOperands();
  }

  SmallPtrSet<Value, 4> uniqueQubitsIn;
  for (const auto& control : getControlsIn()) {
    if (!uniqueQubitsIn.insert(control).second) {
      return emitOpError("duplicate control qubit found");
    }
  }
  auto bodyUnitary = getBodyUnitary();
  const auto numQubits = bodyUnitary.getNumQubits();
  for (size_t i = 0; i < numQubits; i++) {
    if (!uniqueQubitsIn.insert(bodyUnitary.getInputQubit(i)).second) {
      return emitOpError("duplicate qubit found");
    }
  }
  SmallPtrSet<Value, 4> uniqueQubitsOut;
  for (const auto& control : getControlsOut()) {
    if (!uniqueQubitsOut.insert(control).second) {
      return emitOpError("duplicate control qubit found");
    }
  }
  for (size_t i = 0; i < numQubits; i++) {
    if (!uniqueQubitsOut.insert(bodyUnitary.getOutputQubit(i)).second) {
      return emitOpError("duplicate qubit found");
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Canonicalization Patterns
//===----------------------------------------------------------------------===//

namespace {

/**
 * @brief Remove matching allocation and deallocation pairs without operations
 * between them.
 */
struct RemoveAllocDeallocPair final : OpRewritePattern<DeallocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DeallocOp deallocOp,
                                PatternRewriter& rewriter) const override {
    // Check if the predecessor is an AllocOp
    auto allocOp = deallocOp.getQubit().getDefiningOp<AllocOp>();
    if (!allocOp) {
      return failure();
    }

    // Remove the AllocOp and the DeallocOp
    rewriter.eraseOp(deallocOp);
    rewriter.eraseOp(allocOp);
    return success();
  }
};

/**
 * @brief Remove reset operations that immediately follow an allocation.
 */
struct RemoveResetAfterAlloc final : OpRewritePattern<ResetOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ResetOp resetOp,
                                PatternRewriter& rewriter) const override {
    // Check if the predecessor is an AllocOp
    if (auto allocOp = resetOp.getQubitIn().getDefiningOp<AllocOp>();
        !allocOp) {
      return failure();
    }

    // Remove the ResetOp
    rewriter.replaceOp(resetOp, resetOp.getQubitIn());
    return success();
  }
};

/**
 * @brief Remove identity operations.
 */
struct RemoveId final : OpRewritePattern<IdOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IdOp idOp,
                                PatternRewriter& rewriter) const override {
    rewriter.replaceOp(idOp, idOp.getQubitIn());
    return success();
  }
};

/**
 * @brief Remove subsequent X operations on the same qubit.
 */
struct RemoveSubsequentX final : OpRewritePattern<XOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(XOp xOp,
                                PatternRewriter& rewriter) const override {
    // Check if the predecessor is an XOp
    auto prevOp = xOp.getQubitIn().getDefiningOp<XOp>();
    if (!prevOp) {
      return failure();
    }

    // Remove both XOps
    rewriter.replaceOp(prevOp, prevOp.getQubitIn());
    rewriter.replaceOp(xOp, xOp.getQubitIn());

    return success();
  }
};

/**
 * @brief Remove S operations that immediately follow Sdg operations.
 */
struct RemoveSAfterSdg final : OpRewritePattern<SOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SOp sOp,
                                PatternRewriter& rewriter) const override {
    // Check if the predecessor is an SdgOp
    auto prevOp = sOp.getQubitIn().getDefiningOp<SdgOp>();
    if (!prevOp) {
      return failure();
    }

    // Remove both Sdg and S Ops
    rewriter.replaceOp(prevOp, prevOp.getQubitIn());
    rewriter.replaceOp(sOp, sOp.getQubitIn());

    return success();
  }
};

/**
 * @brief Remove Sdg operations that immediately follow S operations.
 */
struct RemoveSdgAfterS final : OpRewritePattern<SdgOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SdgOp sdgOp,
                                PatternRewriter& rewriter) const override {
    // Check if the predecessor is an SOp
    auto prevOp = sdgOp.getQubitIn().getDefiningOp<SOp>();
    if (!prevOp) {
      return failure();
    }

    // Remove both S and Sdg Ops
    rewriter.replaceOp(prevOp, prevOp.getQubitIn());
    rewriter.replaceOp(sdgOp, sdgOp.getQubitIn());

    return success();
  }
};

/**
 * @brief Merge subsequent RX operations on the same qubit by adding their
 * angles.
 */
struct MergeSubsequentRX final : OpRewritePattern<RXOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RXOp rxOp,
                                PatternRewriter& rewriter) const override {
    // Check if the predecessor is an RXOp
    auto prevOp = rxOp.getQubitIn().getDefiningOp<RXOp>();
    if (!prevOp) {
      return failure();
    }

    // Compute and set new theta
    auto newTheta = rewriter.create<arith::AddFOp>(
        rxOp.getLoc(), rxOp.getTheta(), prevOp.getTheta());
    rxOp->setOperand(1, newTheta.getResult());

    // Trivialize previous RXOp
    rewriter.replaceOp(prevOp, prevOp.getQubitIn());

    return success();
  }
};

/**
 * @brief Merge nested control modifiers into a single one.
 */
struct MergeNestedCtrl final : OpRewritePattern<CtrlOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CtrlOp ctrlOp,
                                PatternRewriter& rewriter) const override {
    auto bodyUnitary = ctrlOp.getBodyUnitary();
    auto bodyCtrlOp = llvm::dyn_cast<CtrlOp>(bodyUnitary.getOperation());
    if (!bodyCtrlOp) {
      return failure();
    }

    // Merge controls
    SmallVector<Value> newControls(ctrlOp.getControlsIn());
    for (const auto control : bodyCtrlOp.getControlsIn()) {
      newControls.push_back(control);
    }

    rewriter.replaceOpWithNewOp<CtrlOp>(ctrlOp, newControls,
                                        ctrlOp.getTargetsIn(),
                                        bodyCtrlOp.getBodyUnitary());
    rewriter.eraseOp(bodyCtrlOp);

    return success();
  }
};

/**
 * @brief Remove control modifiers without controls.
 */
struct RemoveTrivialCtrl final : OpRewritePattern<CtrlOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CtrlOp ctrlOp,
                                PatternRewriter& rewriter) const override {
    if (ctrlOp.getNumControls() > 0) {
      return failure();
    }
    rewriter.replaceOp(ctrlOp, ctrlOp.getBodyUnitary());
    return success();
  }
};

struct CtrlInlineId final : OpRewritePattern<CtrlOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CtrlOp ctrlOp,
                                PatternRewriter& rewriter) const override {
    if (!llvm::isa<IdOp>(ctrlOp.getBodyUnitary().getOperation())) {
      return failure();
    }

    auto idOp =
        rewriter.create<IdOp>(ctrlOp.getLoc(), ctrlOp.getTargetsIn().front());

    SmallVector<Value> newOperands;
    newOperands.reserve(ctrlOp.getNumControls() + 1);
    newOperands.append(ctrlOp.getControlsIn().begin(),
                       ctrlOp.getControlsIn().end());
    newOperands.push_back(idOp.getQubitOut());
    rewriter.replaceOp(ctrlOp, newOperands);

    return success();
  }
};

} // namespace

void DeallocOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                            MLIRContext* context) {
  results.add<RemoveAllocDeallocPair>(context);
}

void ResetOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                          MLIRContext* context) {
  results.add<RemoveResetAfterAlloc>(context);
}

void IdOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                       MLIRContext* context) {
  results.add<RemoveId>(context);
}

void XOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                      MLIRContext* context) {
  results.add<RemoveSubsequentX>(context);
}

void SOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                      MLIRContext* context) {
  results.add<RemoveSAfterSdg>(context);
}

void SdgOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<RemoveSdgAfterS>(context);
}

void RXOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                       MLIRContext* context) {
  results.add<MergeSubsequentRX>(context);
}

void CtrlOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* context) {
  results.add<MergeNestedCtrl, RemoveTrivialCtrl, CtrlInlineId>(context);
}
