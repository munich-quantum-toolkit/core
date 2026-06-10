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
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/QTensor/IR/QTensorOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
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
    auto inner = utils::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (!inner) {
      return failure();
    }
    auto innerCtrlOp = dyn_cast<CtrlOp>(inner.getOperation());
    if (!innerCtrlOp) {
      return failure();
    }

    // inv(ctrl(x)) == ctrl(inv(x)). The inner control's controls and targets
    // are block arguments aliasing the inverse modifier's qubits. Pull the
    // controls out to a new control modifier and wrap the inner body in an
    // inverse modifier whose block arguments match the inner targets, so the
    // inner body is reused verbatim.
    auto outerQubits = op.getQubitsIn();
    const auto controls =
        llvm::map_to_vector(innerCtrlOp.getControlsIn(), [&](Value c) {
          return utils::getValueFromBlockArgument(c, outerQubits);
        });
    const auto targets =
        llvm::map_to_vector(innerCtrlOp.getTargetsIn(), [&](Value t) {
          return utils::getValueFromBlockArgument(t, outerQubits);
        });

    auto newCtrl = CtrlOp::create(
        rewriter, op.getLoc(), controls, targets,
        [&](ValueRange targetArgs) -> SmallVector<Value> {
          auto innerInv = InvOp::create(rewriter, op.getLoc(), targetArgs);
          rewriter.inlineRegionBefore(innerCtrlOp.getRegion(),
                                      innerInv.getRegion(),
                                      innerInv.getRegion().end());
          return innerInv.getResults();
        });

    // Each qubit output of the inverse modifier follows its input qubit to the
    // corresponding output of the new control modifier.
    rewriter.replaceOp(op,
                       llvm::map_to_vector(op.getInputQubits(), [&](Value in) {
                         return newCtrl.getOutputForInput(in);
                       }));
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
    auto inner = utils::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (!inner) {
      return failure();
    }

    if (!isa<IdOp, HOp, XOp, YOp, ZOp, ECROp, SWAPOp, BarrierOp>(
            inner.getOperation())) {
      return failure();
    }

    // A self-adjoint gate is its own inverse, so the modifier can be dropped
    // and its body applied directly to the input qubits.
    utils::inlineModifierBody(op, *op.getBody(), op.getInputQubits(), rewriter);
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
    auto inner = utils::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (!inner) {
      return failure();
    }
    auto* innerOp = inner.getOperation();

    // Replace the body gate in place with its inverse, operating on the same
    // (block-argument) operands; inlining the body afterwards substitutes those
    // block arguments with the modifier's input qubits.
    const auto loc = innerOp->getLoc();
    rewriter.setInsertionPoint(innerOp);
    const auto negTheta = [&](auto g) {
      return arith::NegFOp::create(rewriter, loc, g.getTheta()).getResult();
    };
    const auto replaced =
        TypeSwitch<Operation*, LogicalResult>(innerOp)
            .Case<GPhaseOp>([&](auto g) {
              rewriter.replaceOpWithNewOp<GPhaseOp>(g, negTheta(g));
              return success();
            })
            .Case<TOp>([&](auto g) {
              rewriter.replaceOpWithNewOp<TdgOp>(g, g.getInputTarget(0));
              return success();
            })
            .Case<TdgOp>([&](auto g) {
              rewriter.replaceOpWithNewOp<TOp>(g, g.getInputTarget(0));
              return success();
            })
            .Case<SOp>([&](auto g) {
              rewriter.replaceOpWithNewOp<SdgOp>(g, g.getInputTarget(0));
              return success();
            })
            .Case<SdgOp>([&](auto g) {
              rewriter.replaceOpWithNewOp<SOp>(g, g.getInputTarget(0));
              return success();
            })
            .Case<SXOp>([&](auto g) {
              rewriter.replaceOpWithNewOp<SXdgOp>(g, g.getInputTarget(0));
              return success();
            })
            .Case<SXdgOp>([&](auto g) {
              rewriter.replaceOpWithNewOp<SXOp>(g, g.getInputTarget(0));
              return success();
            })
            .Case<POp>([&](auto g) {
              rewriter.replaceOpWithNewOp<POp>(g, g.getInputTarget(0),
                                               negTheta(g));
              return success();
            })
            .Case<ROp>([&](auto g) {
              rewriter.replaceOpWithNewOp<ROp>(g, g.getInputTarget(0),
                                               negTheta(g), g.getPhi());
              return success();
            })
            .Case<RXOp>([&](auto g) {
              rewriter.replaceOpWithNewOp<RXOp>(g, g.getInputTarget(0),
                                                negTheta(g));
              return success();
            })
            .Case<RYOp>([&](auto g) {
              rewriter.replaceOpWithNewOp<RYOp>(g, g.getInputTarget(0),
                                                negTheta(g));
              return success();
            })
            .Case<RZOp>([&](auto g) {
              rewriter.replaceOpWithNewOp<RZOp>(g, g.getInputTarget(0),
                                                negTheta(g));
              return success();
            })
            .Case<UOp>([&](auto g) {
              Value newPhi =
                  arith::NegFOp::create(rewriter, loc, g.getLambda());
              Value newLambda =
                  arith::NegFOp::create(rewriter, loc, g.getPhi());
              Value newTheta =
                  arith::NegFOp::create(rewriter, loc, g.getTheta());
              rewriter.replaceOpWithNewOp<UOp>(g, g.getInputTarget(0), newTheta,
                                               newPhi, newLambda);
              return success();
            })
            .Case<U2Op>([&](auto g) {
              Value pi = arith::ConstantOp::create(
                  rewriter, loc, rewriter.getF64FloatAttr(std::numbers::pi));
              Value newPhi =
                  arith::NegFOp::create(rewriter, loc, g.getLambda());
              newPhi = arith::SubFOp::create(rewriter, loc, newPhi, pi);
              Value newLambda =
                  arith::NegFOp::create(rewriter, loc, g.getPhi());
              newLambda = arith::AddFOp::create(rewriter, loc, newLambda, pi);
              rewriter.replaceOpWithNewOp<U2Op>(g, g.getInputTarget(0), newPhi,
                                                newLambda);
              return success();
            })
            .Case<RXXOp>([&](auto g) {
              rewriter.replaceOpWithNewOp<RXXOp>(
                  g, g.getInputTarget(0), g.getInputTarget(1), negTheta(g));
              return success();
            })
            .Case<RYYOp>([&](auto g) {
              rewriter.replaceOpWithNewOp<RYYOp>(
                  g, g.getInputTarget(0), g.getInputTarget(1), negTheta(g));
              return success();
            })
            .Case<RZXOp>([&](auto g) {
              rewriter.replaceOpWithNewOp<RZXOp>(
                  g, g.getInputTarget(0), g.getInputTarget(1), negTheta(g));
              return success();
            })
            .Case<RZZOp>([&](auto g) {
              rewriter.replaceOpWithNewOp<RZZOp>(
                  g, g.getInputTarget(0), g.getInputTarget(1), negTheta(g));
              return success();
            })
            .Case<XXMinusYYOp>([&](auto g) {
              rewriter.replaceOpWithNewOp<XXMinusYYOp>(
                  g, g.getInputTarget(0), g.getInputTarget(1), negTheta(g),
                  g.getBeta());
              return success();
            })
            .Case<XXPlusYYOp>([&](auto g) {
              rewriter.replaceOpWithNewOp<XXPlusYYOp>(g, g.getInputTarget(0),
                                                      g.getInputTarget(1),
                                                      negTheta(g), g.getBeta());
              return success();
            })
            .Default([&](auto) { return failure(); });

    if (failed(replaced)) {
      return failure();
    }

    utils::inlineModifierBody(op, *op.getBody(), op.getInputQubits(), rewriter);
    return success();
  }
};

/**
 * @brief Cancel nested inverse modifiers, i.e., `inv(inv(x)) => x`.
 */
struct CancelNestedInv final : OpRewritePattern<InvOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InvOp op,
                                PatternRewriter& rewriter) const override {
    auto inner = utils::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (!inner) {
      return failure();
    }
    auto innerInvOp = dyn_cast<InvOp>(inner.getOperation());
    if (!innerInvOp) {
      return failure();
    }
    if (!utils::getSoleBodyUnitary<UnitaryOpInterface>(*innerInvOp.getBody())) {
      return failure();
    }

    // inv(inv(x)) == x: inline the doubly-nested body directly onto the outer
    // input qubits. The inner body's block arguments alias the inner modifier's
    // inputs, which in turn alias the outer input qubits.
    const auto replacements =
        llvm::map_to_vector(innerInvOp.getInputQubits(), [&](Value q) {
          return utils::getValueFromBlockArgument(q, op.getInputQubits());
        });
    utils::inlineModifierBody(op, *innerInvOp.getBody(), replacements,
                              rewriter);
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
  return utils::getNumBodyUnitaries<UnitaryOpInterface>(*getBody());
}

UnitaryOpInterface InvOp::getBodyUnitary(const size_t i) {
  return utils::getBodyUnitary<UnitaryOpInterface>(*getBody(), i);
}

Value InvOp::getInputForOutput(Value output) {
  if (const auto result = dyn_cast<OpResult>(output);
      result && result.getOwner() == getOperation()) {
    return getInputQubit(result.getResultNumber());
  }
  llvm::reportFatalUsageError("Given qubit is not an output of the operation");
}

Value InvOp::getOutputForInput(Value input) {
  for (auto [in, out] : llvm::zip_equal(getInputQubits(), getOutputQubits())) {
    if (in == input) {
      return out;
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
  if (llvm::any_of(block, [](Operation& op) {
        return isa<AllocOp, SinkOp, MeasureOp, ResetOp, qtensor::ExtractOp,
                   qtensor::InsertOp>(op);
      })) {
    return emitOpError("body must not contain non-unitary quantum operations "
                       "or modify a quantum register");
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
  auto* blockTerminator = block.getTerminator();
  if (const auto numYieldOperands = blockTerminator->getNumOperands();
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

std::optional<DynamicMatrix> InvOp::getUnitaryMatrix() {
  auto bodyUnitary = utils::getSoleBodyUnitary<UnitaryOpInterface>(*getBody());
  if (!bodyUnitary) {
    return std::nullopt;
  }
  const auto targetMatrix = bodyUnitary.getUnitaryMatrix<DynamicMatrix>();
  if (!targetMatrix) {
    return std::nullopt;
  }

  return targetMatrix->adjoint();
}
