/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/Quartz/IR/QuartzDialect.h" // IWYU pragma: associated
#include "mlir/Dialect/Utils/MatrixUtils.h"

// The following headers are needed for some template instantiations.
// IWYU pragma: begin_keep
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
// IWYU pragma: end_keep

#include <cstddef>
#include <functional>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>
#include <variant>

using namespace mlir;
using namespace mlir::quartz;
using namespace mlir::utils;

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Quartz/IR/QuartzOpsDialect.cpp.inc"

void QuartzDialect::initialize() {
  // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Quartz/IR/QuartzOpsTypes.cpp.inc"

      >();

  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Quartz/IR/QuartzOps.cpp.inc"

      >();
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Quartz/IR/QuartzOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Quartz/IR/QuartzInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Quartz/IR/QuartzOps.cpp.inc"

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
                   const ValueRange controls, UnitaryOpInterface bodyUnitary) {
  const OpBuilder::InsertionGuard guard(odsBuilder);
  odsState.addOperands(controls);
  auto* region = odsState.addRegion();
  auto& block = region->emplaceBlock();

  // Move the unitary op into the block
  odsBuilder.setInsertionPointToStart(&block);
  odsBuilder.clone(*bodyUnitary.getOperation());
  odsBuilder.create<YieldOp>(odsState.location);
}

void CtrlOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                   const ValueRange controls,
                   const std::function<void(OpBuilder&)>& bodyBuilder) {
  const OpBuilder::InsertionGuard guard(odsBuilder);
  odsState.addOperands(controls);
  auto* region = odsState.addRegion();
  auto& block = region->emplaceBlock();
  odsBuilder.setInsertionPointToStart(&block);
  bodyBuilder(odsBuilder);
  odsBuilder.create<YieldOp>(odsState.location);
}

UnitaryOpInterface CtrlOp::getBodyUnitary() {
  return llvm::dyn_cast<UnitaryOpInterface>(&getBody().front().front());
}

size_t CtrlOp::getNumQubits() { return getNumTargets() + getNumControls(); }

size_t CtrlOp::getNumTargets() { return getBodyUnitary().getNumTargets(); }

size_t CtrlOp::getNumControls() {
  return getNumPosControls() + getNumNegControls();
}

size_t CtrlOp::getNumPosControls() { return getControls().size(); }

size_t CtrlOp::getNumNegControls() {
  return getBodyUnitary().getNumNegControls();
}

Value CtrlOp::getQubit(const size_t i) {
  const auto numPosControls = getNumPosControls();
  if (i < numPosControls) {
    return getControls()[i];
  }
  if (numPosControls <= i && i < getNumQubits()) {
    return getBodyUnitary().getQubit(i - numPosControls);
  }
  llvm::reportFatalUsageError("Invalid qubit index");
}

Value CtrlOp::getTarget(const size_t i) {
  return getBodyUnitary().getTarget(i);
}

Value CtrlOp::getPosControl(const size_t i) { return getControls()[i]; }

Value CtrlOp::getNegControl(const size_t i) {
  return getBodyUnitary().getNegControl(i);
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
  llvm::SmallPtrSet<Value, 4> uniqueQubits;
  for (const auto& control : getControls()) {
    if (!uniqueQubits.insert(control).second) {
      return emitOpError("duplicate control qubit found");
    }
  }
  auto bodyUnitary = getBodyUnitary();
  const auto numQubits = bodyUnitary.getNumQubits();
  for (size_t i = 0; i < numQubits; i++) {
    if (!uniqueQubits.insert(bodyUnitary.getQubit(i)).second) {
      return emitOpError("duplicate qubit found");
    }
  }
  return success();
}

/**
 * @brief A rewrite pattern for merging nested control modifiers.
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

    llvm::SmallVector<Value> newControls(ctrlOp.getControls());
    for (const auto control : bodyCtrlOp.getControls()) {
      newControls.push_back(control);
    }

    rewriter.replaceOpWithNewOp<CtrlOp>(ctrlOp, newControls,
                                        bodyCtrlOp.getBodyUnitary());
    rewriter.eraseOp(bodyCtrlOp);

    return success();
  }
};

/**
 * @brief A rewrite pattern for removing control modifiers without controls.
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

void CtrlOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* context) {
  results.add<MergeNestedCtrl, RemoveTrivialCtrl>(context);
}
