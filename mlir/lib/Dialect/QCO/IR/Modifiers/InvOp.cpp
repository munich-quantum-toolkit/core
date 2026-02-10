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
#include <llvm/ADT/SmallVector.h>
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
 * @brief Move nested control modifiers outside, i.e., `inv(ctrl(x)) =>
 * ctrl(inv(x))`.
 */
struct MoveCtrlOutside final : OpRewritePattern<InvOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InvOp invOp,
                                PatternRewriter& rewriter) const override {
    auto bodyUnitary = invOp.getBodyUnitary();
    auto innerCtrlOp = llvm::dyn_cast<CtrlOp>(bodyUnitary.getOperation());
    if (!innerCtrlOp) {
      return failure();
    }

    const auto numControls = innerCtrlOp.getNumControls();
    const auto numTargets = innerCtrlOp.getNumTargets();
    if (invOp.getNumQubits() != numControls + numTargets) {
      return failure();
    }

    llvm::SmallVector<Value> controls;
    controls.reserve(numControls);
    for (size_t i = 0; i < numControls; ++i) {
      controls.push_back(invOp.getInputQubit(i));
    }

    llvm::SmallVector<Value> targets;
    targets.reserve(numTargets);
    for (size_t i = 0; i < numTargets; ++i) {
      targets.push_back(invOp.getInputQubit(numControls + i));
    }

    auto newCtrl = CtrlOp::create(
        rewriter, invOp.getLoc(), controls, targets,
        [&](ValueRange newTargetArgs) -> llvm::SmallVector<Value> {
          auto newInv = InvOp::create(
              rewriter, invOp.getLoc(), newTargetArgs,
              [&](ValueRange invArgs) -> llvm::SmallVector<Value> {
                IRMapping mapping;
                auto* innerBody = innerCtrlOp.getBody();
                for (size_t i = 0; i < innerCtrlOp.getNumTargets(); ++i) {
                  mapping.map(innerBody->getArgument(i), invArgs[i]);
                }
                auto* cloned = rewriter.clone(
                    *innerCtrlOp.getBodyUnitary().getOperation(), mapping);
                return cloned->getResults();
              });
          return newInv.getResults();
        });

    rewriter.replaceOp(invOp, newCtrl.getResults());
    return success();
  }
};

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

    if (!llvm::isa<IdOp, HOp, XOp, YOp, ZOp, SWAPOp, BarrierOp>(innerOp)) {
      return failure();
    }

    rewriter.moveOpBefore(innerOp, op);
    innerOp->setOperands(0, op.getNumQubits(), op.getInputQubits());
    rewriter.replaceOp(op, innerOp->getResults());
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

  LogicalResult matchAndRewrite(InvOp op,
                                PatternRewriter& rewriter) const override {
    auto* innerOp = op.getBodyUnitary().getOperation();

    return llvm::TypeSwitch<Operation*, LogicalResult>(innerOp)
        .Case<GPhaseOp>([&](auto g) {
          Value negTheta =
              arith::NegFOp::create(rewriter, op.getLoc(), g.getTheta());
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
          Value negTheta =
              arith::NegFOp::create(rewriter, op.getLoc(), p.getTheta());
          rewriter.replaceOpWithNewOp<POp>(op, op.getInputTarget(0), negTheta);
          return success();
        })
        .Case<ROp>([&](auto r) {
          Value negTheta =
              arith::NegFOp::create(rewriter, op.getLoc(), r.getTheta());
          rewriter.replaceOpWithNewOp<ROp>(op, op.getInputTarget(0), negTheta,
                                           r.getPhi());
          return success();
        })
        .Case<RXOp>([&](auto rx) {
          Value negTheta =
              arith::NegFOp::create(rewriter, op.getLoc(), rx.getTheta());
          rewriter.replaceOpWithNewOp<RXOp>(op, op.getInputTarget(0), negTheta);
          return success();
        })
        .Case<UOp>([&](auto u) {
          Value newPhi =
              arith::NegFOp::create(rewriter, op.getLoc(), u.getLambda());
          Value newLambda =
              arith::NegFOp::create(rewriter, op.getLoc(), u.getPhi());
          Value newTheta =
              arith::NegFOp::create(rewriter, op.getLoc(), u.getTheta());
          rewriter.replaceOpWithNewOp<UOp>(op, op.getInputTarget(0), newTheta,
                                           newPhi, newLambda);
          return success();
        })
        .Case<U2Op>([&](auto u) {
          auto pi = arith::ConstantOp::create(
              rewriter, op.getLoc(),
              rewriter.getF64FloatAttr(std::numbers::pi));
          Value newPhi =
              arith::NegFOp::create(rewriter, op.getLoc(), u.getLambda());
          newPhi = arith::SubFOp::create(rewriter, op.getLoc(), newPhi, pi);
          Value newLambda =
              arith::NegFOp::create(rewriter, op.getLoc(), u.getPhi());
          newLambda =
              arith::AddFOp::create(rewriter, op.getLoc(), newLambda, pi);
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
          Value negTheta =
              arith::NegFOp::create(rewriter, op.getLoc(), rxx.getTheta());
          rewriter.replaceOpWithNewOp<RXXOp>(op, op.getInputTarget(0),
                                             op.getInputTarget(1), negTheta);
          return success();
        })
        .Case<RYOp>([&](auto ry) {
          Value negTheta =
              arith::NegFOp::create(rewriter, op.getLoc(), ry.getTheta());
          rewriter.replaceOpWithNewOp<RYOp>(op, op.getInputTarget(0), negTheta);
          return success();
        })
        .Case<RYYOp>([&](auto ryy) {
          Value negTheta =
              arith::NegFOp::create(rewriter, op.getLoc(), ryy.getTheta());
          rewriter.replaceOpWithNewOp<RYYOp>(op, op.getInputTarget(0),
                                             op.getInputTarget(1), negTheta);
          return success();
        })
        .Case<RZOp>([&](auto rz) {
          Value negTheta =
              arith::NegFOp::create(rewriter, op.getLoc(), rz.getTheta());
          rewriter.replaceOpWithNewOp<RZOp>(op, op.getInputTarget(0), negTheta);
          return success();
        })
        .Case<RZXOp>([&](auto rzx) {
          Value negTheta =
              arith::NegFOp::create(rewriter, op.getLoc(), rzx.getTheta());
          rewriter.replaceOpWithNewOp<RZXOp>(op, op.getInputTarget(0),
                                             op.getInputTarget(1), negTheta);
          return success();
        })
        .Case<RZZOp>([&](auto rzz) {
          Value negTheta =
              arith::NegFOp::create(rewriter, op.getLoc(), rzz.getTheta());
          rewriter.replaceOpWithNewOp<RZZOp>(op, op.getInputTarget(0),
                                             op.getInputTarget(1), negTheta);
          return success();
        })
        .Case<XXMinusYYOp>([&](auto xxminusyy) {
          Value negTheta = arith::NegFOp::create(rewriter, op.getLoc(),
                                                 xxminusyy.getTheta());
          rewriter.replaceOpWithNewOp<XXMinusYYOp>(
              op, op.getInputTarget(0), op.getInputTarget(1), negTheta,
              xxminusyy.getBeta());
          return success();
        })
        .Case<XXPlusYYOp>([&](auto xxplusyy) {
          Value negTheta =
              arith::NegFOp::create(rewriter, op.getLoc(), xxplusyy.getTheta());
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
    auto innerUnitary = op.getBodyUnitary().getOperation();
    auto innerInvOp = llvm::dyn_cast<InvOp>(innerUnitary);
    if (!innerInvOp) {
      return failure();
    }

    auto innerInnerUnitary = innerInvOp.getBodyUnitary().getOperation();
    rewriter.moveOpBefore(innerInnerUnitary, op);
    innerInnerUnitary->setOperands(0, op.getNumQubits(), op.getInputQubits());
    rewriter.replaceOp(op, innerInnerUnitary->getResults());

    return success();
  }
};

} // namespace

UnitaryOpInterface InvOp::getBodyUnitary() {
  // In principle, the body region should only contain exactly two operations,
  // the actual unitary operation and a yield operation. However, the region may
  // also contain constants and arithmetic operations, e.g., created as part of
  // canonicalization. Thus, the only safe way to access the unitary operation
  // is to get the second operation from the back of the region.
  return llvm::dyn_cast<UnitaryOpInterface>(*(++getBody()->rbegin()));
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
    if (output == getQubitsOut()[i]) {
      return getQubitsIn()[i];
    }
  }
  llvm::reportFatalUsageError("Given qubit is not an output of the operation");
}

Value InvOp::getOutputForInput(Value input) {
  for (size_t i = 0; i < getNumTargets(); ++i) {
    if (input == getQubitsIn()[i]) {
      return getQubitsOut()[i];
    }
  }
  llvm::reportFatalUsageError("Given qubit is not an input of the operation");
}

size_t InvOp::getNumParams() { return getBodyUnitary().getNumParams(); }

Value InvOp::getParameter(const size_t i) {
  return getBodyUnitary().getParameter(i);
}

void InvOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                  ValueRange qubits, UnitaryOpInterface bodyUnitary) {
  build(odsBuilder, odsState, qubits);
  auto& block = odsState.regions.front()->emplaceBlock();

  // Create block arguments and map targets to them
  IRMapping mapping;
  const auto qubitType = QubitType::get(odsBuilder.getContext());
  for (const auto target : qubits) {
    mapping.map(target, block.addArgument(qubitType, odsState.location));
  }

  // Move the unitary op into the block
  const OpBuilder::InsertionGuard guard(odsBuilder);
  odsBuilder.setInsertionPointToStart(&block);
  auto* op = odsBuilder.clone(*bodyUnitary.getOperation(), mapping);
  YieldOp::create(odsBuilder, odsState.location, op->getResults());
}

void InvOp::build(
    OpBuilder& odsBuilder, OperationState& odsState, ValueRange qubits,
    llvm::function_ref<llvm::SmallVector<Value>(ValueRange)> bodyBuilder) {
  build(odsBuilder, odsState, qubits);
  auto& block = odsState.regions.front()->emplaceBlock();

  const auto qubitType = QubitType::get(odsBuilder.getContext());
  for (size_t i = 0; i < qubits.size(); ++i) {
    block.addArgument(qubitType, odsState.location);
  }

  const OpBuilder::InsertionGuard guard(odsBuilder);
  odsBuilder.setInsertionPointToStart(&block);
  YieldOp::create(odsBuilder, odsState.location,
                  bodyBuilder(block.getArguments()));
}

LogicalResult InvOp::verify() {
  auto& block = *getBody();
  if (block.getOperations().size() < 2) {
    return emitOpError("body region must have at least two operations");
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
  if (!llvm::isa<YieldOp>(block.back())) {
    return emitOpError(
        "last operation in body region must be a yield operation");
  }
  if (const auto numYieldOperands = block.back().getNumOperands();
      numYieldOperands != numTargets) {
    return emitOpError("yield operation must yield ")
           << numTargets << " values, but found " << numYieldOperands;
  }
  auto iter = ++block.rbegin();
  if (!llvm::isa<UnitaryOpInterface>(*(iter))) {
    return emitOpError(
        "second to last operation in body region must be a unitary operation");
  }
  for (auto it = ++iter; it != block.rend(); ++it) {
    if (llvm::isa<UnitaryOpInterface>(*it)) {
      return emitOpError("body region may only contain a single unitary op");
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

  return success();
}

void InvOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<MoveCtrlOutside, InlineSelfAdjoint, ReplaceWithKnownGates,
              CancelNestedInv>(context);
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
