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
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

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
  llvm_unreachable("RXOp has one parameter");
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
  llvm_unreachable("U2Op has two parameters");
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
