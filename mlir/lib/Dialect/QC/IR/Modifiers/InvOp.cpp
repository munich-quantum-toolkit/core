/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCInterfaces.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <numbers>

using namespace mlir;
using namespace mlir::qc;

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

    // The inner control's controls and targets are block arguments aliasing the
    // inverse modifier's qubits. Pull the controls out to a new control
    // modifier and wrap the inner body in an inverse modifier whose block
    // arguments match the inner targets, so the inner body is reused verbatim.
    auto outerQubits = op.getQubits();
    const auto controls =
        llvm::map_to_vector(innerCtrlOp.getControls(), [&](Value c) {
          return utils::getValueFromBlockArgument(c, outerQubits);
        });
    const auto targets =
        llvm::map_to_vector(innerCtrlOp.getTargets(), [&](Value t) {
          return utils::getValueFromBlockArgument(t, outerQubits);
        });

    rewriter.replaceOpWithNewOp<CtrlOp>(
        op, controls, targets, [&](ValueRange targetArgs) {
          auto innerInv = InvOp::create(rewriter, op.getLoc(), targetArgs);
          rewriter.inlineRegionBefore(innerCtrlOp.getRegion(),
                                      innerInv.getRegion(),
                                      innerInv.getRegion().end());
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
    auto inner = utils::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (!inner) {
      return failure();
    }

    if (!isa<IdOp, HOp, XOp, YOp, ZOp, ECROp, SWAPOp, BarrierOp>(
            inner.getOperation())) {
      return failure();
    }

    // A self-adjoint gate is its own inverse, so the modifier can be dropped
    // and its body applied directly to the involved qubits.
    utils::inlineModifierBody(op, *op.getBody(), op.getQubits(), rewriter);
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
    // block arguments with the modifier's qubits.
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
              rewriter.replaceOpWithNewOp<TdgOp>(g, g.getTarget(0));
              return success();
            })
            .Case<TdgOp>([&](auto g) {
              rewriter.replaceOpWithNewOp<TOp>(g, g.getTarget(0));
              return success();
            })
            .Case<SOp>([&](auto g) {
              rewriter.replaceOpWithNewOp<SdgOp>(g, g.getTarget(0));
              return success();
            })
            .Case<SdgOp>([&](auto g) {
              rewriter.replaceOpWithNewOp<SOp>(g, g.getTarget(0));
              return success();
            })
            .Case<SXOp>([&](auto g) {
              rewriter.replaceOpWithNewOp<SXdgOp>(g, g.getTarget(0));
              return success();
            })
            .Case<SXdgOp>([&](auto g) {
              rewriter.replaceOpWithNewOp<SXOp>(g, g.getTarget(0));
              return success();
            })
            .Case<POp>([&](auto g) {
              rewriter.replaceOpWithNewOp<POp>(g, g.getTarget(0), negTheta(g));
              return success();
            })
            .Case<ROp>([&](auto g) {
              rewriter.replaceOpWithNewOp<ROp>(g, g.getTarget(0), negTheta(g),
                                               g.getPhi());
              return success();
            })
            .Case<RXOp>([&](auto g) {
              rewriter.replaceOpWithNewOp<RXOp>(g, g.getTarget(0), negTheta(g));
              return success();
            })
            .Case<RYOp>([&](auto g) {
              rewriter.replaceOpWithNewOp<RYOp>(g, g.getTarget(0), negTheta(g));
              return success();
            })
            .Case<RZOp>([&](auto g) {
              rewriter.replaceOpWithNewOp<RZOp>(g, g.getTarget(0), negTheta(g));
              return success();
            })
            .Case<UOp>([&](auto g) {
              Value newPhi =
                  arith::NegFOp::create(rewriter, loc, g.getLambda());
              Value newLambda =
                  arith::NegFOp::create(rewriter, loc, g.getPhi());
              Value newTheta =
                  arith::NegFOp::create(rewriter, loc, g.getTheta());
              rewriter.replaceOpWithNewOp<UOp>(g, g.getTarget(0), newTheta,
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
              rewriter.replaceOpWithNewOp<U2Op>(g, g.getTarget(0), newPhi,
                                                newLambda);
              return success();
            })
            .Case<DCXOp>([&](auto g) {
              rewriter.replaceOpWithNewOp<DCXOp>(g, g.getTarget(1),
                                                 g.getTarget(0));
              return success();
            })
            .Case<RXXOp>([&](auto g) {
              rewriter.replaceOpWithNewOp<RXXOp>(g, g.getTarget(0),
                                                 g.getTarget(1), negTheta(g));
              return success();
            })
            .Case<RYYOp>([&](auto g) {
              rewriter.replaceOpWithNewOp<RYYOp>(g, g.getTarget(0),
                                                 g.getTarget(1), negTheta(g));
              return success();
            })
            .Case<RZXOp>([&](auto g) {
              rewriter.replaceOpWithNewOp<RZXOp>(g, g.getTarget(0),
                                                 g.getTarget(1), negTheta(g));
              return success();
            })
            .Case<RZZOp>([&](auto g) {
              rewriter.replaceOpWithNewOp<RZZOp>(g, g.getTarget(0),
                                                 g.getTarget(1), negTheta(g));
              return success();
            })
            .Case<XXMinusYYOp>([&](auto g) {
              rewriter.replaceOpWithNewOp<XXMinusYYOp>(
                  g, g.getTarget(0), g.getTarget(1), negTheta(g), g.getBeta());
              return success();
            })
            .Case<XXPlusYYOp>([&](auto g) {
              rewriter.replaceOpWithNewOp<XXPlusYYOp>(
                  g, g.getTarget(0), g.getTarget(1), negTheta(g), g.getBeta());
              return success();
            })
            .Default([&](auto) { return failure(); });

    if (failed(replaced)) {
      return failure();
    }

    utils::inlineModifierBody(op, *op.getBody(), op.getQubits(), rewriter);
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
    // qubits. The inner body's block arguments alias the inner modifier's
    // inputs, which in turn alias the outer qubits.
    const auto replacements =
        llvm::map_to_vector(innerInvOp.getQubits(), [&](Value q) {
          return utils::getValueFromBlockArgument(q, op.getQubits());
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

    rewriter.eraseOp(op);
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

void InvOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                  ValueRange qubits,
                  const function_ref<void(ValueRange)>& body) {
  build(odsBuilder, odsState, qubits);
  auto& block = odsState.regions.front()->emplaceBlock();

  auto qubitType = QubitType::get(odsBuilder.getContext());
  for (size_t i = 0; i < qubits.size(); ++i) {
    block.addArgument(qubitType, odsState.location);
  }

  const OpBuilder::InsertionGuard guard(odsBuilder);
  odsBuilder.setInsertionPointToStart(&block);
  body(block.getArguments());
  YieldOp::create(odsBuilder, odsState.location);
}

void InvOp::build(OpBuilder& odsBuilder, OperationState& odsState, Value qubit,
                  const function_ref<void(Value)>& bodyBuilder) {
  build(odsBuilder, odsState, ValueRange{qubit}, [&](ValueRange qubits) {
    assert(qubits.size() == 1 &&
           "single-qubit inv body expects exactly one qubit");
    bodyBuilder(qubits.front());
  });
}

InvOp InvOp::create(OpBuilder& builder, Location location, Value qubit,
                    const function_ref<void(Value)>& bodyBuilder) {
  OperationState state(location, getOperationName());
  build(builder, state, qubit, bodyBuilder);
  auto op = dyn_cast<InvOp>(builder.create(state));
  assert(op && "builder didn't return the right type");
  return op;
}

InvOp InvOp::create(ImplicitLocOpBuilder& builder, Value qubit,
                    const function_ref<void(Value)>& bodyBuilder) {
  return create(builder, builder.getLoc(), qubit, bodyBuilder);
}

LogicalResult InvOp::verify() {
  if (llvm::any_of(*getBody(), [](Operation& op) {
        return isa<AllocOp, DeallocOp, MeasureOp, ResetOp, memref::LoadOp,
                   memref::StoreOp>(op);
      })) {
    return emitOpError("body must not contain non-unitary quantum operations "
                       "or modify a quantum register");
  }
  return success();
}

void InvOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<CancelNestedInv, MoveCtrlOutside, InlineSelfAdjoint,
              ReplaceWithKnownGates, EraseEmptyInv>(context);
}
