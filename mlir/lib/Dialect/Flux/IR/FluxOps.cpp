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

#include <cstddef>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/ErrorHandling.h>
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

size_t XOp::getNumPosControls() const {
  llvm_unreachable("Not implemented yet");
}

size_t XOp::getNumNegControls() const {
  llvm_unreachable("Not implemented yet");
}

Value XOp::getQubit(size_t /*i*/) const {
  llvm_unreachable("Not implemented yet");
}

Value XOp::getTarget(size_t i) {
  if (i != 0) {
    llvm_unreachable("XOp has only one target qubit");
  }
  return getQubitIn();
}

Value XOp::getPosControl(size_t /*i*/) const {
  llvm_unreachable("Not implemented yet");
}

Value XOp::getNegControl(size_t /*i*/) const {
  llvm_unreachable("Not implemented yet");
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

ParameterDescriptor XOp::getParameter(size_t /*i*/) const {
  llvm_unreachable("XOp has no parameters");
}

DenseElementsAttr XOp::tryGetStaticMatrix() {
  auto* ctx = getContext();
  auto type = RankedTensorType::get({2, 2}, Float64Type::get(ctx));
  return DenseElementsAttr::get(type, llvm::ArrayRef({0.0, 1.0, 1.0, 0.0}));
}

CanonicalDescriptor XOp::getCanonicalDescriptor() const {
  return CanonicalDescriptor{
      .baseSymbol = "x",
      .orderedParams = {},
  };
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
