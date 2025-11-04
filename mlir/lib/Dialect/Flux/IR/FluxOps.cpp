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
#include "mlir/Dialect/Utils/ParameterDescriptor.h"

// The following headers are needed for some template instantiations.
// IWYU pragma: begin_keep
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/PatternMatch.h>
// IWYU pragma: end_keep

#include <cstddef>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
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
// Traits
//===----------------------------------------------------------------------===//

namespace mlir::flux {

LogicalResult foldParameterArityTrait(Operation* op) {
  auto concreteOp = llvm::dyn_cast<UnitaryOpInterface>(op);
  if (!concreteOp) {
    return failure();
  }

  LogicalResult succeeded = failure();
  for (size_t i = 0; i < concreteOp.getNumParams(); ++i) {
    const auto& parameter = concreteOp.getParameter(i);

    if (parameter.isStatic()) {
      continue;
    }

    auto constantOp =
        parameter.getValueOperand().getDefiningOp<arith::ConstantOp>();
    if (!constantOp) {
      continue;
    }

    const auto& thetaAttr = llvm::dyn_cast<FloatAttr>(constantOp.getValue());
    if (!thetaAttr) {
      continue;
    }

    concreteOp.replaceOperandWithAttr(i, thetaAttr);
    succeeded = success();
  }

  return succeeded;
}

} // namespace mlir::flux

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

// XOp

DenseElementsAttr XOp::tryGetStaticMatrix() {
  auto* ctx = getContext();
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({2, 2}, complexType);
  return DenseElementsAttr::get(type, getMatrixX());
}

// RXOp

ParameterDescriptor RXOp::getParameter(size_t i) {
  if (i == 0) {
    return {getThetaAttr(), getThetaDyn()};
  }
  llvm::report_fatal_error("RXOp has one parameter");
}

void RXOp::replaceOperandWithAttr(size_t i, FloatAttr attr) {
  if (i == 0) {
    getOperation()->setAttr("theta", attr);
    getOperation()->eraseOperand(1);
    return;
  }
  llvm::report_fatal_error("RXOp has one parameter");
}

bool RXOp::hasStaticUnitary() { return getParameter(0).isStatic(); }

DenseElementsAttr RXOp::tryGetStaticMatrix() {
  if (!hasStaticUnitary()) {
    return nullptr;
  }
  auto* ctx = getContext();
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({2, 2}, complexType);
  const auto& theta = getTheta().value().convertToDouble();
  return DenseElementsAttr::get(type, getMatrixRX(theta));
}

void RXOp::build(OpBuilder& builder, OperationState& state, Value qubitIn,
                 std::variant<double, FloatAttr, Value> theta) {
  FloatAttr thetaAttr = nullptr;
  Value thetaOperand = nullptr;
  if (std::holds_alternative<double>(theta)) {
    thetaAttr = builder.getF64FloatAttr(std::get<double>(theta));
  } else if (std::holds_alternative<FloatAttr>(theta)) {
    thetaAttr = std::get<FloatAttr>(theta);
  } else {
    thetaOperand = std::get<Value>(theta);
  }
  build(builder, state, QubitType::get(builder.getContext()), qubitIn,
        thetaAttr, thetaOperand);
}

LogicalResult RXOp::verify() {
  if (getTheta().has_value() == (getThetaDyn() != nullptr)) {
    return emitOpError("must specify exactly one of static or dynamic theta");
  }
  return success();
}

// U2Op

ParameterDescriptor U2Op::getParameter(size_t i) {
  if (i == 0) {
    return {getPhiAttr(), getPhiDyn()};
  }
  if (i == 1) {
    return {getLambdaAttr(), getLambdaDyn()};
  }
  llvm::report_fatal_error("U2Op has two parameters");
}

void U2Op::replaceOperandWithAttr(size_t i, FloatAttr attr) {
  if (i == 0) {
    getOperation()->setAttr("phi", attr);
    getOperation()->eraseOperand(1);
    return;
  }
  if (i == 1) {
    getOperation()->setAttr("lambda", attr);
    getOperation()->eraseOperand(3);
    return;
  }
  llvm::report_fatal_error("U2Op has two parameter");
}

bool U2Op::hasStaticUnitary() {
  return (getParameter(0).isStatic() && getParameter(1).isStatic());
}

DenseElementsAttr U2Op::tryGetStaticMatrix() {
  if (!hasStaticUnitary()) {
    return nullptr;
  }
  auto* ctx = getContext();
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({2, 2}, complexType);
  const auto& phi = getPhi().value().convertToDouble();
  const auto& lambda = getLambda().value().convertToDouble();
  return DenseElementsAttr::get(type, getMatrixU2(phi, lambda));
}

void U2Op::build(OpBuilder& builder, OperationState& state, Value qubitIn,
                 std::variant<double, FloatAttr, Value> phi,
                 std::variant<double, FloatAttr, Value> lambda) {
  FloatAttr phiAttr = nullptr;
  Value phiOperand = nullptr;
  if (std::holds_alternative<double>(phi)) {
    phiAttr = builder.getF64FloatAttr(std::get<double>(phi));
  } else if (std::holds_alternative<FloatAttr>(phi)) {
    phiAttr = std::get<FloatAttr>(phi);
  } else {
    phiOperand = std::get<Value>(phi);
  }

  FloatAttr lambdaAttr = nullptr;
  Value lambdaOperand = nullptr;
  if (std::holds_alternative<double>(lambda)) {
    lambdaAttr = builder.getF64FloatAttr(std::get<double>(lambda));
  } else if (std::holds_alternative<FloatAttr>(lambda)) {
    lambdaAttr = std::get<FloatAttr>(lambda);
  } else {
    lambdaOperand = std::get<Value>(lambda);
  }

  build(builder, state, QubitType::get(builder.getContext()), qubitIn, phiAttr,
        phiOperand, lambdaAttr, lambdaOperand);
}

LogicalResult U2Op::verify() {
  if (getPhi().has_value() == (getPhiDyn() != nullptr)) {
    return emitOpError("must specify exactly one of static or dynamic phi");
  }
  if (getLambda().has_value() == (getLambdaDyn() != nullptr)) {
    return emitOpError("must specify exactly one of static or dynamic lambda");
  }
  return success();
}

// SWAPOp

DenseElementsAttr SWAPOp::tryGetStaticMatrix() {
  auto* ctx = getContext();
  const auto& complexType = ComplexType::get(Float64Type::get(ctx));
  const auto& type = RankedTensorType::get({4, 4}, complexType);
  return DenseElementsAttr::get(type, getMatrixSWAP());
}

//===----------------------------------------------------------------------===//
// Modifiers
//===----------------------------------------------------------------------===//

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
  auto unitaryOp = getBodyUnitary();
  return unitaryOp.getNumNegControls();
}

Value CtrlOp::getInputQubit(size_t i) {
  if (i < getNumTargets()) {
    return getInputTarget(i);
  }
  if (getNumTargets() <= i && i < getNumPosControls()) {
    return getInputPosControl(i - getNumTargets());
  }
  if (getNumTargets() + getNumPosControls() <= i && i < getNumQubits()) {
    return getInputNegControl(i - getNumTargets() - getNumPosControls());
  }
  llvm::report_fatal_error("Invalid qubit index");
}

Value CtrlOp::getOutputQubit(size_t i) {
  if (i < getNumTargets()) {
    return getOutputTarget(i);
  }
  if (getNumTargets() <= i && i < getNumPosControls()) {
    return getOutputPosControl(i - getNumTargets());
  }
  if (getNumTargets() + getNumPosControls() <= i && i < getNumQubits()) {
    return getOutputNegControl(i - getNumTargets() - getNumPosControls());
  }
  llvm::report_fatal_error("Invalid qubit index");
}

Value CtrlOp::getInputTarget(size_t i) { return getTargetsIn()[i]; }

Value CtrlOp::getOutputTarget(size_t i) { return getTargetsOut()[i]; }

Value CtrlOp::getInputPosControl(size_t i) { return getControlsIn()[i]; }

Value CtrlOp::getOutputPosControl(size_t i) { return getControlsOut()[i]; }

Value CtrlOp::getInputNegControl(size_t i) {
  auto unitaryOp = getBodyUnitary();
  return unitaryOp.getInputNegControl(i);
}

Value CtrlOp::getOutputNegControl(size_t i) {
  auto unitaryOp = getBodyUnitary();
  return unitaryOp.getOutputNegControl(i);
}

Value CtrlOp::getInputForOutput(Value output) {
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

Value CtrlOp::getOutputForInput(Value input) {
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

size_t CtrlOp::getNumParams() {
  auto unitaryOp = getBodyUnitary();
  return unitaryOp.getNumParams();
}

ParameterDescriptor CtrlOp::getParameter(size_t i) {
  auto unitaryOp = getBodyUnitary();
  return unitaryOp.getParameter(i);
}

void CtrlOp::replaceOperandWithAttr(size_t i, FloatAttr attr) {
  auto unitaryOp = getBodyUnitary();
  unitaryOp.replaceOperandWithAttr(i, attr);
}

bool CtrlOp::hasStaticUnitary() {
  auto unitaryOp = getBodyUnitary();
  return unitaryOp.hasStaticUnitary();
}

DenseElementsAttr CtrlOp::tryGetStaticMatrix() {
  llvm::report_fatal_error("Not implemented yet"); // TODO
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
 * @brief Remove subsequent X operations on the same qubit.
 */
struct RemoveSubsequentX final : OpRewritePattern<XOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(XOp xOp,
                                PatternRewriter& rewriter) const override {
    auto prevOp = xOp.getQubitIn().getDefiningOp<XOp>();

    // Check if the predecessor is an XOp
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
 * @brief Merge subsequent RX operations on the same qubit by adding their
 * angles.
 */
struct MergeSubsequentRX final : OpRewritePattern<RXOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RXOp rxOp,
                                PatternRewriter& rewriter) const override {
    auto prevOp = rxOp.getQubitIn().getDefiningOp<RXOp>();

    // Check if the predecessor is an RXOp
    if (!prevOp) {
      return failure();
    }

    // Compute and set new theta
    const auto& theta = prevOp.getParameter(0);
    const auto& prevTheta = rxOp.getParameter(0);
    if (theta.isStatic() && prevTheta.isStatic()) {
      const auto& newTheta =
          theta.getValueDouble() + prevTheta.getValueDouble();
      rxOp.setThetaAttr(rewriter.getF64FloatAttr(newTheta));
    } else if (theta.isStatic() && prevTheta.isDynamic()) {
      auto constantOp = rewriter.create<arith::ConstantOp>(
          rxOp.getLoc(), rewriter.getF64FloatAttr(theta.getValueDouble()));
      auto newTheta = rewriter.create<arith::AddFOp>(
          rxOp.getLoc(), constantOp.getResult(), prevTheta.getValueOperand());
      rxOp->setOperand(1, newTheta.getResult());
    } else if (theta.isDynamic() && prevTheta.isStatic()) {
      auto constantOp = rewriter.create<arith::ConstantOp>(
          rxOp.getLoc(), rewriter.getF64FloatAttr(prevTheta.getValueDouble()));
      auto newTheta = rewriter.create<arith::AddFOp>(
          rxOp.getLoc(), theta.getValueOperand(), constantOp.getResult());
      rxOp.removeThetaAttr();
      rxOp->setOperand(1, newTheta.getResult());
    } else if (theta.isDynamic() && prevTheta.isDynamic()) {
      auto newTheta = rewriter.create<arith::AddFOp>(
          rxOp.getLoc(), theta.getValueOperand(), prevTheta.getValueOperand());
      rxOp->setOperand(1, newTheta.getResult());
    } else {
      return failure();
    }

    // Trivialize previous RXOp
    rewriter.replaceOp(prevOp, prevOp.getQubitIn());

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

void XOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                      MLIRContext* context) {
  results.add<RemoveSubsequentX>(context);
}

void RXOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                       MLIRContext* context) {
  results.add<MergeSubsequentRX>(context);
}
