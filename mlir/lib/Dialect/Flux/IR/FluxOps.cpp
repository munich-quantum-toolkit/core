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

// YOp

DenseElementsAttr YOp::tryGetStaticMatrix() { return getMatrixY(getContext()); }

// ZOp

DenseElementsAttr ZOp::tryGetStaticMatrix() { return getMatrixZ(getContext()); }

// HOp

DenseElementsAttr HOp::tryGetStaticMatrix() { return getMatrixH(getContext()); }

// SOp

DenseElementsAttr SOp::tryGetStaticMatrix() { return getMatrixS(getContext()); }

// SdgOp

DenseElementsAttr SdgOp::tryGetStaticMatrix() {
  return getMatrixSdg(getContext());
}

// TOp

DenseElementsAttr TOp::tryGetStaticMatrix() { return getMatrixT(getContext()); }

// TdgOp

DenseElementsAttr TdgOp::tryGetStaticMatrix() {
  return getMatrixTdg(getContext());
}

// SXOp

DenseElementsAttr SXOp::tryGetStaticMatrix() {
  return getMatrixSX(getContext());
}

// SXdgOp

DenseElementsAttr SXdgOp::tryGetStaticMatrix() {
  return getMatrixSXdg(getContext());
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

  LogicalResult matchAndRewrite(DeallocOp op,
                                PatternRewriter& rewriter) const override {
    // Check if the predecessor is an AllocOp
    auto allocOp = op.getQubit().getDefiningOp<AllocOp>();
    if (!allocOp) {
      return failure();
    }

    // Remove the AllocOp and the DeallocOp
    rewriter.eraseOp(op);
    rewriter.eraseOp(allocOp);
    return success();
  }
};

/**
 * @brief Remove reset operations that immediately follow an allocation.
 */
struct RemoveResetAfterAlloc final : OpRewritePattern<ResetOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ResetOp op,
                                PatternRewriter& rewriter) const override {
    // Check if the predecessor is an AllocOp
    if (auto allocOp = op.getQubitIn().getDefiningOp<AllocOp>(); !allocOp) {
      return failure();
    }

    // Remove the ResetOp
    rewriter.replaceOp(op, op.getQubitIn());
    return success();
  }
};

/**
 * @brief Remove identity operations.
 */
struct RemoveId final : OpRewritePattern<IdOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IdOp op,
                                PatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, op.getQubitIn());
    return success();
  }
};

/**
 * @brief Remove a pair of inverse operations.
 *
 * @tparam InverseOpType The type of the inverse operation.
 * @tparam OpType The type of the operation to be checked.
 * @param op The operation instance to be checked.
 * @param rewriter The pattern rewriter.
 * @return LogicalResult Success or failure of the removal.
 */
template <typename InverseOpType, typename OpType>
LogicalResult removeInversePair(OpType op, PatternRewriter& rewriter) {
  // Check if the predecessor is the inverse operation
  auto prevOp = op.getQubitIn().template getDefiningOp<InverseOpType>();
  if (!prevOp) {
    return failure();
  }

  // Remove both XOps
  rewriter.replaceOp(prevOp, prevOp.getQubitIn());
  rewriter.replaceOp(op, op.getQubitIn());

  return success();
}

/**
 * @brief Remove subsequent X operations on the same qubit.
 */
struct RemoveSubsequentX final : OpRewritePattern<XOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(XOp op,
                                PatternRewriter& rewriter) const override {
    return removeInversePair<XOp>(op, rewriter);
  }
};

/**
 * @brief Remove subsequent Y operations on the same qubit.
 */
struct RemoveSubsequentY final : OpRewritePattern<YOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(YOp op,
                                PatternRewriter& rewriter) const override {
    return removeInversePair<YOp>(op, rewriter);
  }
};

/**
 * @brief Remove subsequent Z operations on the same qubit.
 */
struct RemoveSubsequentZ final : OpRewritePattern<ZOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ZOp op,
                                PatternRewriter& rewriter) const override {
    return removeInversePair<ZOp>(op, rewriter);
  }
};

/**
 * @brief Remove subsequent H operations on the same qubit.
 */
struct RemoveSubsequentH final : OpRewritePattern<HOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(HOp op,
                                PatternRewriter& rewriter) const override {
    return removeInversePair<HOp>(op, rewriter);
  }
};

/**
 * @brief Remove S operations that immediately follow Sdg operations.
 */
struct RemoveSAfterSdg final : OpRewritePattern<SOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SOp op,
                                PatternRewriter& rewriter) const override {
    return removeInversePair<SdgOp>(op, rewriter);
  }
};

/**
 * @brief Remove Sdg operations that immediately follow S operations.
 */
struct RemoveSdgAfterS final : OpRewritePattern<SdgOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SdgOp op,
                                PatternRewriter& rewriter) const override {
    return removeInversePair<SOp>(op, rewriter);
  }
};

/**
 * @brief Remove T operations that immediately follow Tdg operations.
 */
struct RemoveTAfterTdg final : OpRewritePattern<TOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TOp op,
                                PatternRewriter& rewriter) const override {
    return removeInversePair<TdgOp>(op, rewriter);
  }
};

/**
 * @brief Remove Tdg operations that immediately follow T operations.
 */
struct RemoveTdgAfterT final : OpRewritePattern<TdgOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TdgOp op,
                                PatternRewriter& rewriter) const override {
    return removeInversePair<TOp>(op, rewriter);
  }
};

/**
 * @brief Remove SX operations that immediately follow SXdg operations.
 */
struct RemoveSXAfterSXdg final : OpRewritePattern<SXOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SXOp op,
                                PatternRewriter& rewriter) const override {
    return removeInversePair<SXdgOp>(op, rewriter);
  }
};

/**
 * @brief Remove SXdg operations that immediately follow SX operations.
 */
struct RemoveSXdgAfterSX final : OpRewritePattern<SXdgOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SXdgOp op,
                                PatternRewriter& rewriter) const override {
    return removeInversePair<SXOp>(op, rewriter);
  }
};

/**
 * @brief Merge subsequent RX operations on the same qubit by adding their
 * angles.
 */
struct MergeSubsequentRX final : OpRewritePattern<RXOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RXOp op,
                                PatternRewriter& rewriter) const override {
    // Check if the predecessor is an RXOp
    auto prevOp = op.getQubitIn().getDefiningOp<RXOp>();
    if (!prevOp) {
      return failure();
    }

    // Compute and set new theta
    auto newTheta = rewriter.create<arith::AddFOp>(op.getLoc(), op.getTheta(),
                                                   prevOp.getTheta());
    op->setOperand(1, newTheta.getResult());

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

  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    auto bodyUnitary = op.getBodyUnitary();
    auto bodyCtrlOp = llvm::dyn_cast<CtrlOp>(bodyUnitary.getOperation());
    if (!bodyCtrlOp) {
      return failure();
    }

    // Merge controls
    SmallVector<Value> newControls(op.getControlsIn());
    for (const auto control : bodyCtrlOp.getControlsIn()) {
      newControls.push_back(control);
    }

    rewriter.replaceOpWithNewOp<CtrlOp>(op, newControls, op.getTargetsIn(),
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

  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getNumControls() > 0) {
      return failure();
    }
    rewriter.replaceOp(op, op.getBodyUnitary());
    return success();
  }
};

/**
 * @brief Inline controlled identity operations.
 */
struct CtrlInlineId final : OpRewritePattern<CtrlOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    if (!llvm::isa<IdOp>(op.getBodyUnitary().getOperation())) {
      return failure();
    }

    auto idOp = rewriter.create<IdOp>(op.getLoc(), op.getTargetsIn().front());

    SmallVector<Value> newOperands;
    newOperands.reserve(op.getNumControls() + 1);
    newOperands.append(op.getControlsIn().begin(), op.getControlsIn().end());
    newOperands.push_back(idOp.getQubitOut());
    rewriter.replaceOp(op, newOperands);

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

void YOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                      MLIRContext* context) {
  results.add<RemoveSubsequentY>(context);
}

void ZOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                      MLIRContext* context) {
  results.add<RemoveSubsequentZ>(context);
}

void HOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                      MLIRContext* context) {
  results.add<RemoveSubsequentH>(context);
}

void SOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                      MLIRContext* context) {
  results.add<RemoveSAfterSdg>(context);
}

void SdgOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<RemoveSdgAfterS>(context);
}

void TOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                      MLIRContext* context) {
  results.add<RemoveTAfterTdg>(context);
}

void TdgOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<RemoveTdgAfterT>(context);
}

void SXOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                       MLIRContext* context) {
  results.add<RemoveSXAfterSXdg>(context);
}

void SXdgOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* context) {
  results.add<RemoveSXdgAfterSX>(context);
}

void RXOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                       MLIRContext* context) {
  results.add<MergeSubsequentRX>(context);
}

void CtrlOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* context) {
  results.add<MergeNestedCtrl, RemoveTrivialCtrl, CtrlInlineId>(context);
}
