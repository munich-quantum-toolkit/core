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

// The following headers are needed for some template instantiations.
// IWYU pragma: begin_keep
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/PatternMatch.h>
// IWYU pragma: end_keep

#include <cmath>
#include <complex>
#include <cstddef>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::flux;

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

// XOp

Value XOp::getQubit(size_t i) {
  if (i != 0) {
    llvm_unreachable("XOp has only one qubit");
  }
  return getQubitIn();
}

Value XOp::getTarget(size_t i) {
  if (i != 0) {
    llvm_unreachable("XOp has only one target qubit");
  }
  return getQubitIn();
}

Value XOp::getPosControl(size_t /*i*/) {
  llvm_unreachable("XOp does not have controls");
}

Value XOp::getNegControl(size_t /*i*/) {
  llvm_unreachable("XOp does not have controls");
}

Value XOp::getInput(size_t i) {
  if (i != 0) {
    llvm_unreachable("XOp has only one input qubit");
  }
  return getQubitIn();
}

Value XOp::getOutput(size_t i) {
  if (i != 0) {
    llvm_unreachable("XOp has only one output qubit");
  }
  return getQubitOut();
}

Value XOp::getInputForOutput(Value output) {
  if (output != getQubitOut()) {
    llvm_unreachable("Given output is not the XOp's output");
  }
  return getQubitIn();
}

Value XOp::getOutputForInput(Value input) {
  if (input != getQubitIn()) {
    llvm_unreachable("Given input is not the XOp's input");
  }
  return getQubitOut();
}

ParameterDescriptor XOp::getParameter(size_t /*i*/) {
  llvm_unreachable("XOp does not have parameters");
}

DenseElementsAttr XOp::tryGetStaticMatrix() {
  auto* ctx = getContext();
  auto type = RankedTensorType::get({2, 2}, Float64Type::get(ctx));
  return DenseElementsAttr::get(type, llvm::ArrayRef({0.0, 1.0, 1.0, 0.0}));
}

// RXOp

Value RXOp::getQubit(size_t i) {
  if (i != 0) {
    llvm_unreachable("RXOp has only one qubit");
  }
  return getQubitIn();
}

Value RXOp::getTarget(size_t i) {
  if (i != 0) {
    llvm_unreachable("RXOp has only one target qubit");
  }
  return getQubitIn();
}

Value RXOp::getPosControl(size_t /*i*/) {
  llvm_unreachable("RXOp does not have controls");
}

Value RXOp::getNegControl(size_t /*i*/) {
  llvm_unreachable("RXOp does not have controls");
}

Value RXOp::getInput(size_t i) {
  if (i != 0) {
    llvm_unreachable("RXOp has only one input qubit");
  }
  return getQubitIn();
}

Value RXOp::getOutput(size_t i) {
  if (i != 0) {
    llvm_unreachable("RXOp has only one output qubit");
  }
  return getQubitOut();
}

Value RXOp::getInputForOutput(Value output) {
  if (output != getQubitOut()) {
    llvm_unreachable("Given output is not the RXOp's output");
  }
  return getQubitIn();
}

Value RXOp::getOutputForInput(Value input) {
  if (input != getQubitIn()) {
    llvm_unreachable("Given input is not the RXOp's input");
  }
  return getQubitOut();
}

ParameterDescriptor RXOp::getParameter(size_t i) {
  if (i != 0) {
    llvm_unreachable("RXOp has only one parameter");
  }
  return {getThetaAttrAttr(), getThetaOperand()};
}

bool RXOp::hasStaticUnitary() { return getParameter(0).isStatic(); }

DenseElementsAttr RXOp::tryGetStaticMatrix() {
  if (!hasStaticUnitary()) {
    return nullptr;
  }
  auto* ctx = getContext();
  auto type = RankedTensorType::get({2, 2}, Float64Type::get(ctx));
  const auto& theta = getThetaAttr().value().convertToDouble();
  const std::complex<double> m0(std::cos(theta / 2), 0);
  const std::complex<double> m1(0, -std::sin(theta / 2));
  return DenseElementsAttr::get(type, llvm::ArrayRef({m0, m1, m1, m0}));
}

LogicalResult RXOp::verify() {
  if (getThetaAttr().has_value() && getThetaOperand())
    return emitOpError("cannot specify both static and dynamic theta");
  if (!getThetaAttr().has_value() && !getThetaOperand())
    return emitOpError("must specify either static or dynamic theta");
  return success();
}

// U2Op

Value U2Op::getQubit(size_t i) {
  if (i != 0) {
    llvm_unreachable("U2Op has only one qubit");
  }
  return getQubitIn();
}

Value U2Op::getTarget(size_t i) {
  if (i != 0) {
    llvm_unreachable("U2Op has only one target qubit");
  }
  return getQubitIn();
}

Value U2Op::getPosControl(size_t /*i*/) {
  llvm_unreachable("U2Op does not have controls");
}

Value U2Op::getNegControl(size_t /*i*/) {
  llvm_unreachable("U2Op does not have controls");
}

Value U2Op::getInput(size_t i) {
  if (i != 0) {
    llvm_unreachable("U2Op has only one input qubit");
  }
  return getQubitIn();
}

Value U2Op::getOutput(size_t i) {
  if (i != 0) {
    llvm_unreachable("U2Op has only one output qubit");
  }
  return getQubitOut();
}

Value U2Op::getInputForOutput(Value output) {
  if (output != getQubitOut()) {
    llvm_unreachable("Given output is not the U2Op's output");
  }
  return getQubitIn();
}

Value U2Op::getOutputForInput(Value input) {
  if (input != getQubitIn()) {
    llvm_unreachable("Given input is not the U2Op's input");
  }
  return getQubitOut();
}

ParameterDescriptor U2Op::getParameter(size_t i) {
  if (i == 0) {
    return {getPhiAttrAttr(), getPhiOperand()};
  }
  if (i == 1) {
    return {getLambdaAttrAttr(), getLambdaOperand()};
  }
  llvm_unreachable("U2Op has only two parameters");
}

bool U2Op::hasStaticUnitary() {
  return (getParameter(0).isStatic() && getParameter(1).isStatic());
}

DenseElementsAttr U2Op::tryGetStaticMatrix() {
  if (!hasStaticUnitary()) {
    return nullptr;
  }
  auto* ctx = getContext();
  auto type = RankedTensorType::get({2, 2}, Float64Type::get(ctx));
  const auto& phi = getPhiAttr().value().convertToDouble();
  const auto& lambda = getLambdaAttr().value().convertToDouble();
  const std::complex<double> i(0.0, 1.0);
  const std::complex<double> m00(1.0, 0.0);
  const std::complex<double> m01 = -std::exp(i * lambda);
  const std::complex<double> m10 = -std::exp(i * phi);
  const std::complex<double> m11 = std::exp(i * (phi + lambda));
  return DenseElementsAttr::get(type, llvm::ArrayRef({m00, m01, m10, m11}));
}

LogicalResult U2Op::verify() {
  if (getPhiAttr().has_value() && getPhiOperand())
    return emitOpError("cannot specify both static and dynamic phi");
  if (!getPhiAttr().has_value() && !getPhiOperand())
    return emitOpError("must specify either static or dynamic phi");
  if (getLambdaAttr().has_value() && getLambdaOperand())
    return emitOpError("cannot specify both static and dynamic lambda");
  if (!getLambdaAttr().has_value() && !getLambdaOperand())
    return emitOpError("must specify either static or dynamic lambda");
  return success();
}

// SWAPOp

Value SWAPOp::getQubit(size_t i) {
  if (i == 0) {
    return getQubit0In();
  }
  if (i == 1) {
    return getQubit1In();
  }
  llvm_unreachable("SWAPOp only has two input qubits");
}

Value SWAPOp::getTarget(size_t i) { return getQubit(i); }

Value SWAPOp::getPosControl(size_t /*i*/) {
  llvm_unreachable("SWAPOp does not have controls");
}

Value SWAPOp::getNegControl(size_t /*i*/) {
  llvm_unreachable("SWAPOp does not have controls");
}

Value SWAPOp::getInput(size_t i) { return getQubit(i); }

Value SWAPOp::getOutput(size_t i) {
  if (i == 0) {
    return getQubit0Out();
  }
  if (i == 1) {
    return getQubit1Out();
  }
  llvm_unreachable("SWAPOp only has two output qubits");
}

Value SWAPOp::getInputForOutput(Value output) {
  if (output == getQubit0Out()) {
    return getQubit0In();
  }
  if (output == getQubit1Out()) {
    return getQubit1In();
  }
  llvm_unreachable("Given qubit is not an output of the SWAPOp");
}

Value SWAPOp::getOutputForInput(Value input) {
  if (input == getQubit0In()) {
    return getQubit0Out();
  }
  if (input == getQubit1In()) {
    return getQubit1Out();
  }
  llvm_unreachable("Given qubit is not an input of the SWAPOp");
}

ParameterDescriptor SWAPOp::getParameter(size_t /*i*/) {
  llvm_unreachable("SWAPOp does not have parameters");
}

DenseElementsAttr SWAPOp::tryGetStaticMatrix() {
  auto* ctx = getContext();
  const auto& type = RankedTensorType::get({4, 4}, Float64Type::get(ctx));
  return DenseElementsAttr::get(
      type, llvm::ArrayRef({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 1.0}));
}

//===----------------------------------------------------------------------===//
// Canonicalization Patterns
//===----------------------------------------------------------------------===//

namespace {
/**
 * @class RemoveAllocDeallocPair
 * @brief A class designed to identify and remove matching allocation and
 * deallocation pairs without operations between them.
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
    rewriter.eraseOp(allocOp);
    rewriter.eraseOp(deallocOp);
    return success();
  }
};

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
} // namespace

void DeallocOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                            MLIRContext* context) {
  results.add<RemoveAllocDeallocPair>(context);
}

void ResetOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                          MLIRContext* context) {
  results.add<RemoveResetAfterAlloc>(context);
}
