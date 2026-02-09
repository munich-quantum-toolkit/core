/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "Eigen/Core"
#include "mlir/Dialect/QCO/IR/QCODialect.h"

#include <cstddef>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <numbers>
#include <optional>

using namespace mlir;
using namespace mlir::qco;

namespace {

/**
 * @brief Remove inverse modifiers around self-adjoint gates.
 *
 * For self-adjoint gates U (i.e., U = U†), inv(U) = U holds.
 */
struct InlineSelfAdjoint final : OpRewritePattern<InvOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InvOp op,
                                PatternRewriter& rewriter) const override {
    auto* innerOp = op.getBodyUnitary().getOperation();

    if (!llvm::isa<IdOp, HOp, XOp, YOp, ZOp, SWAPOp>(innerOp)) {
      return failure();
    }

    // Map block arguments to operation inputs
    IRMapping mapping;
    auto& block = *op.getBody();
    for (size_t i = 0; i < op.getNumTargets(); ++i) {
      mapping.map(block.getArgument(i), op.getInputTarget(i));
    }

    // Clone the inner operation using the mapping
    auto* cloned = rewriter.clone(*innerOp, mapping);
    rewriter.replaceOp(op, cloned->getResults());
    return success();
  }
};

/**
 * @brief Replace inverse modifiers around gates where the inverse is a known
 * gate by their known inverse.
 *
 * For example, for the T gate, inv(T) = Tdg holds.
 */
struct ReplaceWithKnownGates final : OpRewritePattern<InvOp> {
  using OpRewritePattern::OpRewritePattern;

  /**
   * @brief Computes the negated value, i.e. f(x) = -x.
   */
  static Value negatedAngle(Value theta, PatternRewriter& rewriter,
                            Location loc) {
    return rewriter.create<arith::NegFOp>(loc, theta);
  }

  /**
   * @brief Computes the negated value shifted by minus pi, i.e. f(x) = -x - pi.
   */
  static Value negatedPiShiftedAngle(Value theta, PatternRewriter& rewriter,
                                     Location loc) {
    auto negated = negatedAngle(theta, rewriter, loc);
    auto pi = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64FloatAttr(std::numbers::pi));
    return rewriter.create<arith::SubFOp>(loc, negated, pi);
  }

  LogicalResult matchAndRewrite(InvOp op,
                                PatternRewriter& rewriter) const override {
    auto* innerOp = op.getBodyUnitary().getOperation();

    return llvm::TypeSwitch<Operation*, LogicalResult>(innerOp)
        .Case<GPhaseOp>([&](auto g) {
          auto negTheta = negatedAngle(g.getTheta(), rewriter, op.getLoc());
          rewriter.replaceOpWithNewOp<GPhaseOp>(op, negTheta);
          return success();
        })
        .Case<TOp>([&](auto) {
          rewriter.replaceOpWithNewOp<TdgOp>(op, op.getInputTarget(0));
          return success();
        })
        .Case<TdgOp>([&](auto) {
          rewriter.replaceOpWithNewOp<TOp>(op, op.getInputTarget(0));
          return success();
        })
        .Case<SOp>([&](auto) {
          rewriter.replaceOpWithNewOp<SdgOp>(op, op.getInputTarget(0));
          return success();
        })
        .Case<SdgOp>([&](auto) {
          rewriter.replaceOpWithNewOp<SOp>(op, op.getInputTarget(0));
          return success();
        })
        .Case<SXOp>([&](auto) {
          rewriter.replaceOpWithNewOp<SXdgOp>(op, op.getInputTarget(0));
          return success();
        })
        .Case<SXdgOp>([&](auto) {
          rewriter.replaceOpWithNewOp<SXOp>(op, op.getInputTarget(0));
          return success();
        })
        .Case<POp>([&](auto p) {
          auto negTheta = negatedAngle(p.getTheta(), rewriter, op.getLoc());
          rewriter.replaceOpWithNewOp<POp>(op, op.getInputTarget(0), negTheta);
          return success();
        })
        .Case<ROp>([&](auto r) {
          auto negTheta = negatedAngle(r.getTheta(), rewriter, op.getLoc());
          rewriter.replaceOpWithNewOp<ROp>(op, op.getInputTarget(0), negTheta,
                                           r.getPhi());
          return success();
        })
        .Case<RXOp>([&](auto rx) {
          auto negTheta = negatedAngle(rx.getTheta(), rewriter, op.getLoc());
          rewriter.replaceOpWithNewOp<RXOp>(op, op.getInputTarget(0), negTheta);
          return success();
        })
        .Case<UOp>([&](auto u) {
          auto newPhi =
              negatedPiShiftedAngle(u.getLambda(), rewriter, op.getLoc());
          auto newLambda =
              negatedPiShiftedAngle(u.getPhi(), rewriter, op.getLoc());
          rewriter.replaceOpWithNewOp<UOp>(op, op.getInputTarget(0),
                                           u.getTheta(), newPhi, newLambda);
          return success();
        })
        .Case<U2Op>([&](auto u) {
          auto newPhi =
              negatedPiShiftedAngle(u.getLambda(), rewriter, op.getLoc());
          auto newLambda =
              negatedPiShiftedAngle(u.getPhi(), rewriter, op.getLoc());
          rewriter.replaceOpWithNewOp<U2Op>(op, op.getInputTarget(0), newPhi,
                                            newLambda);
          return success();
        })
        .Case<DCXOp>([&](auto) {
          rewriter.replaceOpWithNewOp<DCXOp>(op, op.getInputTarget(1),
                                             op.getInputTarget(0));
          return success();
        })
        .Case<RXXOp>([&](auto rxx) {
          auto negTheta = negatedAngle(rxx.getTheta(), rewriter, op.getLoc());
          rewriter.replaceOpWithNewOp<RXXOp>(op, op.getInputTarget(0),
                                             op.getInputTarget(1), negTheta);
          return success();
        })
        .Case<RYOp>([&](auto ry) {
          auto negTheta = negatedAngle(ry.getTheta(), rewriter, op.getLoc());
          rewriter.replaceOpWithNewOp<RYOp>(op, op.getInputTarget(0), negTheta);
          return success();
        })
        .Case<RYYOp>([&](auto ryy) {
          auto negTheta = negatedAngle(ryy.getTheta(), rewriter, op.getLoc());
          rewriter.replaceOpWithNewOp<RYYOp>(op, op.getInputTarget(0),
                                             op.getInputTarget(1), negTheta);
          return success();
        })
        .Case<RZOp>([&](auto rz) {
          auto negTheta = negatedAngle(rz.getTheta(), rewriter, op.getLoc());
          rewriter.replaceOpWithNewOp<RZOp>(op, op.getInputTarget(0), negTheta);
          return success();
        })
        .Case<RZXOp>([&](auto rzx) {
          auto negTheta = negatedAngle(rzx.getTheta(), rewriter, op.getLoc());
          rewriter.replaceOpWithNewOp<RZXOp>(op, op.getInputTarget(0),
                                             op.getInputTarget(1), negTheta);
          return success();
        })
        .Case<RZZOp>([&](auto rzz) {
          auto negTheta = negatedAngle(rzz.getTheta(), rewriter, op.getLoc());
          rewriter.replaceOpWithNewOp<RZZOp>(op, op.getInputTarget(0),
                                             op.getInputTarget(1), negTheta);
          return success();
        })
        .Case<XXMinusYYOp>([&](auto xxminusyy) {
          auto negTheta =
              negatedAngle(xxminusyy.getTheta(), rewriter, op.getLoc());
          rewriter.replaceOpWithNewOp<XXMinusYYOp>(
              op, op.getInputTarget(0), op.getInputTarget(1), negTheta,
              xxminusyy.getBeta());
          return success();
        })
        .Case<XXPlusYYOp>([&](auto xxplusyy) {
          auto negTheta =
              negatedAngle(xxplusyy.getTheta(), rewriter, op.getLoc());
          rewriter.replaceOpWithNewOp<XXPlusYYOp>(op, op.getInputTarget(0),
                                                  op.getInputTarget(1),
                                                  negTheta, xxplusyy.getBeta());
          return success();
        })
        .Default([&](auto) { return failure(); });
  }
};

/**
 * @brief Cancel nested inverse modifiers, i.e., `inv(inv(x)) => x`.
 */
struct CancelNestedInv final : OpRewritePattern<InvOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InvOp op,
                                PatternRewriter& rewriter) const override {
    auto innerUnitary = op.getBodyUnitary();
    auto innerInvOp = llvm::dyn_cast<InvOp>(innerUnitary.getOperation());
    if (!innerInvOp) {
      return failure();
    }

    // Remove both inverse operations
    auto innerInnerUnitary = innerInvOp.getBodyUnitary();
    auto* clonedOp = rewriter.clone(*innerInnerUnitary.getOperation());
    rewriter.replaceOp(op, clonedOp->getResults());

    return success();
  }
};

} // namespace

UnitaryOpInterface InvOp::getBodyUnitary() {
  return llvm::dyn_cast<UnitaryOpInterface>(&getBody()->front());
}

size_t InvOp::getNumQubits() { return getNumTargets(); }

size_t InvOp::getNumTargets() { return getQubitsIn().size(); }

size_t InvOp::getNumControls() { return 0; }

Value InvOp::getInputQubit(const size_t i) {
  if (i >= getNumTargets()) {
    llvm::reportFatalUsageError("Qubit index out of bounds");
  }
  return getQubitsIn()[i];
}

OperandRange InvOp::getInputQubits() { return this->getOperands(); }

Value InvOp::getOutputQubit(const size_t i) {
  if (i >= getNumTargets()) {
    llvm::reportFatalUsageError("Qubit index out of bounds");
  }
  return getQubitsOut()[i];
}

ResultRange InvOp::getOutputQubits() { return this->getResults(); }

Value InvOp::getInputTarget(const size_t i) { return getInputQubit(i); }

Value InvOp::getOutputTarget(const size_t i) { return getOutputQubit(i); }

Value InvOp::getInputControl([[maybe_unused]] const size_t i) {
  llvm::reportFatalUsageError("Operation does not have controls");
}

Value InvOp::getOutputControl([[maybe_unused]] const size_t i) {
  llvm::reportFatalUsageError("Operation does not have controls");
}

Value InvOp::getInputForOutput(Value output) {
  for (size_t i = 0; i < getNumTargets(); ++i) {
    if (output == getTargetsOut()[i]) {
      return getTargetsIn()[i];
    }
  }
  llvm::reportFatalUsageError("Given qubit is not an output of the operation");
}

Value InvOp::getOutputForInput(Value input) {
  for (size_t i = 0; i < getNumTargets(); ++i) {
    if (input == getTargetsIn()[i]) {
      return getTargetsOut()[i];
    }
  }
  llvm::reportFatalUsageError("Given qubit is not an input of the operation");
}

size_t InvOp::getNumParams() { return getBodyUnitary().getNumParams(); }

Value InvOp::getParameter(const size_t i) {
  return getBodyUnitary().getParameter(i);
}

void InvOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                  ValueRange targets, UnitaryOpInterface bodyUnitary) {
  build(odsBuilder, odsState, targets);
  auto& block = odsState.regions.front()->emplaceBlock();

  // Move the unitary op into the block
  const OpBuilder::InsertionGuard guard(odsBuilder);
  odsBuilder.setInsertionPointToStart(&block);
  auto* op = odsBuilder.clone(*bodyUnitary.getOperation());
  YieldOp::create(odsBuilder, odsState.location, op->getResults());
}

void InvOp::build(
    OpBuilder& odsBuilder, OperationState& odsState, ValueRange targets,
    llvm::function_ref<llvm::SmallVector<Value>(ValueRange)> bodyBuilder) {
  build(odsBuilder, odsState, targets);
  auto& block = odsState.regions.front()->emplaceBlock();

  const auto qubitType = QubitType::get(odsBuilder.getContext());
  for (size_t i = 0; i < targets.size(); ++i) {
    block.addArgument(qubitType, odsState.location);
  }

  const OpBuilder::InsertionGuard guard(odsBuilder);
  odsBuilder.setInsertionPointToStart(&block);
  YieldOp::create(odsBuilder, odsState.location,
                  bodyBuilder(block.getArguments()));
}

LogicalResult InvOp::verify() {
  auto& block = *getBody();
  if (block.getOperations().size() != 2) {
    return emitOpError("body region must have exactly two operations");
  }
  const auto numTargets = getNumTargets();
  if (block.getArguments().size() != numTargets) {
    return emitOpError(
        "number of block arguments must match the number of targets");
  }
  const auto qubitType = QubitType::get(getContext());
  for (size_t i = 0; i < numTargets; ++i) {
    if (block.getArgument(i).getType() != qubitType) {
      return emitOpError("block argument type at index ")
             << i << " does not match target type";
    }
  }
  if (!llvm::isa<UnitaryOpInterface>(block.front())) {
    return emitOpError(
        "first operation in body region must be a unitary operation");
  }
  if (!llvm::isa<YieldOp>(block.back())) {
    return emitOpError(
        "second operation in body region must be a yield operation");
  }
  if (const auto numYieldOperands = block.back().getNumOperands();
      numYieldOperands != numTargets) {
    return emitOpError("yield operation must yield ")
           << numTargets << " values, but found " << numYieldOperands;
  }

  SmallPtrSet<Value, 4> uniqueQubitsIn;
  for (const auto& target : getTargetsIn()) {
    if (!uniqueQubitsIn.insert(target).second) {
      return emitOpError("duplicate target qubit found");
    }
  }

  auto bodyUnitary = getBodyUnitary();
  if (bodyUnitary.getNumQubits() != numTargets) {
    return emitOpError("body unitary must operate on exactly ")
           << numTargets << " target qubits, but found "
           << bodyUnitary.getNumQubits();
  }
  const auto numQubits = bodyUnitary.getNumQubits();
  for (size_t i = 0; i < numQubits; i++) {
    if (bodyUnitary.getInputQubit(i) != block.getArgument(i)) {
      return emitOpError("body unitary must use target alias block argument ")
             << i << " (and not the original target operand)";
    }
  }

  // Also require yield to forward the unitary's outputs in-order.
  for (size_t i = 0; i < numTargets; ++i) {
    if (block.back().getOperand(i) != bodyUnitary.getOutputQubit(i)) {
      return emitOpError("yield operand ")
             << i << " must be the body unitary output qubit " << i;
    }
  }

  SmallPtrSet<Value, 4> uniqueQubitsOut;
  for (size_t i = 0; i < numQubits; i++) {
    if (!uniqueQubitsOut.insert(bodyUnitary.getOutputQubit(i)).second) {
      return emitOpError("duplicate qubit found");
    }
  }

  if (llvm::isa<BarrierOp>(bodyUnitary.getOperation())) {
    return emitOpError("BarrierOp cannot be inverted");
  }

  return success();
}

void InvOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<InlineSelfAdjoint, ReplaceWithKnownGates>(context);
}

std::optional<Eigen::MatrixXcd> InvOp::getUnitaryMatrix() {
  auto&& bodyUnitary = getBodyUnitary();
  if (!bodyUnitary) {
    return std::nullopt;
  }
  auto&& targetMatrix = bodyUnitary.getUnitaryMatrix<Eigen::MatrixXcd>();
  if (!targetMatrix) {
    return std::nullopt;
  }

  targetMatrix->adjointInPlace();

  return targetMatrix;
}
