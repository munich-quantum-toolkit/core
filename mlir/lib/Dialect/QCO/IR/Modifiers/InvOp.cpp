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
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <Eigen/Core>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
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

  LogicalResult matchAndRewrite(InvOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getNumBodyUnitaries() != 1) {
      return failure();
    }
    auto innerCtrlOp = dyn_cast<CtrlOp>(op.getBodyUnitary(0).getOperation());
    if (!innerCtrlOp) {
      return failure();
    }

    const auto numControls = innerCtrlOp.getNumControls();
    const auto numTargets = innerCtrlOp.getNumTargets();
    auto outerQubits = op.getQubitsIn();
    auto controls = outerQubits.take_front(numControls);
    auto targets = outerQubits.take_back(numTargets);

    rewriter.replaceOpWithNewOp<CtrlOp>(
        op, controls, targets,
        [&](ValueRange targetArgs) -> SmallVector<Value> {
          return InvOp::create(
                     rewriter, op.getLoc(), targetArgs,
                     [&](ValueRange qubitArgs) -> SmallVector<Value> {
                       auto* innerCtrlBody = innerCtrlOp.getBody();
                       IRMapping mapping;
                       utils::populateMapping(mapping, *innerCtrlBody,
                                              innerCtrlOp.getTargetsIn(),
                                              outerQubits, targets, qubitArgs);
                       for (auto& op : innerCtrlBody->without_terminator()) {
                         rewriter.clone(op, mapping);
                       }
                       SmallVector<Value> yields;
                       for (auto value :
                            innerCtrlBody->getTerminator()->getOperands()) {
                         yields.push_back(mapping.lookup(value));
                       }
                       return yields;
                     })
              .getResults();
        });

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
    if (op.getNumBodyUnitaries() != 1) {
      return failure();
    }
    auto* innerOp = op.getBodyUnitary(0).getOperation();

    if (!isa<IdOp, HOp, XOp, YOp, ZOp, ECROp, SWAPOp, BarrierOp>(innerOp)) {
      return failure();
    }

    const auto numQubits = op.getNumQubits();
    auto outerQubits = op.getInputQubits();
    SmallVector<Value> qubits;
    for (auto qubit : innerOp->getOperands().take_front(numQubits)) {
      qubits.push_back(utils::getValueFromBlockArgument(qubit, outerQubits));
    }

    rewriter.moveOpBefore(innerOp, op);
    innerOp->setOperands(0, numQubits, qubits);
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
    if (op.getNumBodyUnitaries() != 1) {
      return failure();
    }
    auto* innerOp = op.getBodyUnitary(0).getOperation();

    auto loc = op.getLoc();
    auto outerQubits = op.getInputQubits();

    return TypeSwitch<Operation*, LogicalResult>(innerOp)
        .Case<GPhaseOp>([&](auto g) {
          Value negTheta = arith::NegFOp::create(rewriter, loc, g.getTheta());
          rewriter.replaceOpWithNewOp<GPhaseOp>(op, negTheta);
          return success();
        })
        .Case<TOp>([&](auto t) {
          rewriter.replaceOpWithNewOp<TdgOp>(
              op, utils::getValueFromBlockArgument(t.getInputTarget(0),
                                                   outerQubits));
          return success();
        })
        .Case<TdgOp>([&](auto tdg) {
          rewriter.replaceOpWithNewOp<TOp>(
              op, utils::getValueFromBlockArgument(tdg.getInputTarget(0),
                                                   outerQubits));
          return success();
        })
        .Case<SOp>([&](auto s) {
          rewriter.replaceOpWithNewOp<SdgOp>(
              op, utils::getValueFromBlockArgument(s.getInputTarget(0),
                                                   outerQubits));
          return success();
        })
        .Case<SdgOp>([&](auto sdg) {
          rewriter.replaceOpWithNewOp<SOp>(
              op, utils::getValueFromBlockArgument(sdg.getInputTarget(0),
                                                   outerQubits));
          return success();
        })
        .Case<SXOp>([&](auto sx) {
          rewriter.replaceOpWithNewOp<SXdgOp>(
              op, utils::getValueFromBlockArgument(sx.getInputTarget(0),
                                                   outerQubits));
          return success();
        })
        .Case<SXdgOp>([&](auto sxdg) {
          rewriter.replaceOpWithNewOp<SXOp>(
              op, utils::getValueFromBlockArgument(sxdg.getInputTarget(0),
                                                   outerQubits));
          return success();
        })
        .Case<POp>([&](auto p) {
          Value negTheta = arith::NegFOp::create(rewriter, loc, p.getTheta());
          rewriter.replaceOpWithNewOp<POp>(
              op,
              utils::getValueFromBlockArgument(p.getInputTarget(0),
                                               outerQubits),
              negTheta);
          return success();
        })
        .Case<ROp>([&](auto r) {
          Value negTheta = arith::NegFOp::create(rewriter, loc, r.getTheta());
          rewriter.replaceOpWithNewOp<ROp>(
              op,
              utils::getValueFromBlockArgument(r.getInputTarget(0),
                                               outerQubits),
              negTheta, r.getPhi());
          return success();
        })
        .Case<RXOp>([&](auto rx) {
          Value negTheta = arith::NegFOp::create(rewriter, loc, rx.getTheta());
          rewriter.replaceOpWithNewOp<RXOp>(
              op,
              utils::getValueFromBlockArgument(rx.getInputTarget(0),
                                               outerQubits),
              negTheta);
          return success();
        })
        .Case<UOp>([&](auto u) {
          Value newPhi = arith::NegFOp::create(rewriter, loc, u.getLambda());
          Value newLambda = arith::NegFOp::create(rewriter, loc, u.getPhi());
          Value newTheta = arith::NegFOp::create(rewriter, loc, u.getTheta());
          rewriter.replaceOpWithNewOp<UOp>(
              op,
              utils::getValueFromBlockArgument(u.getInputTarget(0),
                                               outerQubits),
              newTheta, newPhi, newLambda);
          return success();
        })
        .Case<U2Op>([&](auto u2) {
          auto pi = arith::ConstantOp::create(
              rewriter, loc, rewriter.getF64FloatAttr(std::numbers::pi));
          Value newPhi = arith::NegFOp::create(rewriter, loc, u2.getLambda());
          newPhi = arith::SubFOp::create(rewriter, loc, newPhi, pi);
          Value newLambda = arith::NegFOp::create(rewriter, loc, u2.getPhi());
          newLambda = arith::AddFOp::create(rewriter, loc, newLambda, pi);
          rewriter.replaceOpWithNewOp<U2Op>(
              op,
              utils::getValueFromBlockArgument(u2.getInputTarget(0),
                                               outerQubits),
              newPhi, newLambda);
          return success();
        })
        .Case<RXXOp>([&](auto rxx) {
          Value negTheta = arith::NegFOp::create(rewriter, loc, rxx.getTheta());
          rewriter.replaceOpWithNewOp<RXXOp>(
              op,
              utils::getValueFromBlockArgument(rxx.getInputTarget(0),
                                               outerQubits),
              utils::getValueFromBlockArgument(rxx.getInputTarget(1),
                                               outerQubits),
              negTheta);
          return success();
        })
        .Case<RYOp>([&](auto ry) {
          Value negTheta = arith::NegFOp::create(rewriter, loc, ry.getTheta());
          rewriter.replaceOpWithNewOp<RYOp>(
              op,
              utils::getValueFromBlockArgument(ry.getInputTarget(0),
                                               outerQubits),
              negTheta);
          return success();
        })
        .Case<RYYOp>([&](auto ryy) {
          Value negTheta = arith::NegFOp::create(rewriter, loc, ryy.getTheta());
          rewriter.replaceOpWithNewOp<RYYOp>(
              op,
              utils::getValueFromBlockArgument(ryy.getInputTarget(0),
                                               outerQubits),
              utils::getValueFromBlockArgument(ryy.getInputTarget(1),
                                               outerQubits),
              negTheta);
          return success();
        })
        .Case<RZOp>([&](auto rz) {
          Value negTheta = arith::NegFOp::create(rewriter, loc, rz.getTheta());
          rewriter.replaceOpWithNewOp<RZOp>(
              op,
              utils::getValueFromBlockArgument(rz.getInputTarget(0),
                                               outerQubits),
              negTheta);
          return success();
        })
        .Case<RZXOp>([&](auto rzx) {
          Value negTheta = arith::NegFOp::create(rewriter, loc, rzx.getTheta());
          rewriter.replaceOpWithNewOp<RZXOp>(
              op,
              utils::getValueFromBlockArgument(rzx.getInputTarget(0),
                                               outerQubits),
              utils::getValueFromBlockArgument(rzx.getInputTarget(1),
                                               outerQubits),
              negTheta);
          return success();
        })
        .Case<RZZOp>([&](auto rzz) {
          Value negTheta = arith::NegFOp::create(rewriter, loc, rzz.getTheta());
          rewriter.replaceOpWithNewOp<RZZOp>(
              op,
              utils::getValueFromBlockArgument(rzz.getInputTarget(0),
                                               outerQubits),
              utils::getValueFromBlockArgument(rzz.getInputTarget(1),
                                               outerQubits),
              negTheta);
          return success();
        })
        .Case<XXMinusYYOp>([&](auto xxminusyy) {
          Value negTheta =
              arith::NegFOp::create(rewriter, loc, xxminusyy.getTheta());
          rewriter.replaceOpWithNewOp<XXMinusYYOp>(
              op,
              utils::getValueFromBlockArgument(xxminusyy.getInputTarget(0),
                                               outerQubits),
              utils::getValueFromBlockArgument(xxminusyy.getInputTarget(1),
                                               outerQubits),
              negTheta, xxminusyy.getBeta());
          return success();
        })
        .Case<XXPlusYYOp>([&](auto xxplusyy) {
          Value negTheta =
              arith::NegFOp::create(rewriter, loc, xxplusyy.getTheta());
          rewriter.replaceOpWithNewOp<XXPlusYYOp>(
              op,
              utils::getValueFromBlockArgument(xxplusyy.getInputTarget(0),
                                               outerQubits),
              utils::getValueFromBlockArgument(xxplusyy.getInputTarget(1),
                                               outerQubits),
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
    if (op.getNumBodyUnitaries() != 1) {
      return failure();
    }
    auto innerInvOp = dyn_cast<InvOp>(op.getBodyUnitary(0).getOperation());
    if (!innerInvOp) {
      return failure();
    }

    if (innerInvOp.getNumBodyUnitaries() != 1) {
      return failure();
    }
    auto* innerInnerOp = innerInvOp.getBodyUnitary(0).getOperation();

    const auto numQubits = op.getNumQubits();
    auto outerQubits = op.getInputQubits();
    auto innerQubits = innerInvOp.getInputQubits();
    SmallVector<Value> qubits;
    for (auto qubit : innerInnerOp->getOperands().take_front(numQubits)) {
      auto innerQubit = utils::getValueFromBlockArgument(qubit, innerQubits);
      qubits.push_back(
          utils::getValueFromBlockArgument(innerQubit, outerQubits));
    }

    rewriter.moveOpBefore(innerInnerOp, op);
    innerInnerOp->setOperands(0, numQubits, qubits);
    rewriter.replaceOp(op, innerInnerOp->getResults());
    return success();
  }
};

/**
 * @brief Erase inverse modifiers that do not have any body unitaries.
 */
struct EraseEmptyInv final : OpRewritePattern<InvOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(InvOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getNumBodyUnitaries() != 0) {
      return failure();
    }

    rewriter.replaceOp(op, op.getOperands());
    return success();
  }
};

} // namespace

size_t InvOp::getNumBodyUnitaries() {
  return llvm::count_if(
      *getBody(), [](Operation& op) { return isa<UnitaryOpInterface>(op); });
}

UnitaryOpInterface InvOp::getBodyUnitary(const size_t i) {
  size_t count = 0;
  for (auto& op : *getBody()) {
    if (isa<UnitaryOpInterface>(op)) {
      if (count == i) {
        return cast<UnitaryOpInterface>(op);
      }
      count++;
    }
  }
  llvm::reportFatalUsageError("Unitary index out of bounds");
}

Value InvOp::getInputQubit(const size_t i) {
  if (i >= getNumTargets()) {
    llvm::reportFatalUsageError("Qubit index out of bounds");
  }
  return getQubitsIn()[i];
}

Value InvOp::getOutputQubit(const size_t i) {
  if (i >= getNumTargets()) {
    llvm::reportFatalUsageError("Qubit index out of bounds");
  }
  return getQubitsOut()[i];
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

void InvOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                  ValueRange qubits,
                  function_ref<SmallVector<Value>(ValueRange)> bodyBuilder) {
  build(odsBuilder, odsState, qubits);
  auto& block = odsState.regions.front()->emplaceBlock();

  auto qubitType = QubitType::get(odsBuilder.getContext());
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
  if (llvm::any_of(*getBody(), [](Operation& op) {
        return isa<AllocOp, SinkOp, MeasureOp, ResetOp>(op);
      })) {
    return emitOpError("body must not contain non-unitary quantum operations");
  }
  if (!isa<YieldOp>(block.back())) {
    return emitOpError(
        "last operation in body region must be a yield operation");
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
  if (const auto numYieldOperands = block.back().getNumOperands();
      numYieldOperands != numTargets) {
    return emitOpError("yield operation must yield ")
           << numTargets << " values, but found " << numYieldOperands;
  }

  SmallPtrSet<Value, 4> uniqueQubitsIn;
  for (const auto& target : getQubitsIn()) {
    if (!uniqueQubitsIn.insert(target).second) {
      return emitOpError("duplicate qubit found");
    }
  }

  return success();
}

void InvOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<MoveCtrlOutside, InlineSelfAdjoint, ReplaceWithKnownGates,
              CancelNestedInv, EraseEmptyInv>(context);
}

std::optional<Eigen::MatrixXcd> InvOp::getUnitaryMatrix() {
  if (getNumBodyUnitaries() != 1) {
    return std::nullopt;
  }

  auto bodyUnitary = getBodyUnitary(0);
  if (!bodyUnitary) {
    return std::nullopt;
  }
  auto targetMatrix = bodyUnitary.getUnitaryMatrix<Eigen::MatrixXcd>();
  if (!targetMatrix) {
    return std::nullopt;
  }

  targetMatrix->adjointInPlace();

  return targetMatrix;
}
