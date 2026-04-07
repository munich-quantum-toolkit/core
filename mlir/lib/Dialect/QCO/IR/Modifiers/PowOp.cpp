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

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
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

#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <optional>

using namespace mlir;
using namespace mlir::qco;

namespace {

/// pow(1.0) { g }  =>  inline g
struct InlinePow1 final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getExponentValue() != 1.0) {
      return failure();
    }

    auto* innerOp = op.getBodyUnitary().getOperation();
    rewriter.moveOpBefore(innerOp, op);
    innerOp->setOperands(op.getInputQubits());
    rewriter.replaceOp(op, innerOp->getResults());
    return success();
  }
};

/// pow(0.0) { g }  =>  identity (pass-through)
struct ErasePow0 final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getExponentValue() != 0.0) {
      return failure();
    }

    rewriter.replaceOp(op, op.getQubitsIn());
    return success();
  }
};

/// pow(p) where p < 0  =>  pow(-p) { inv { g } }
struct NegPowToInvPow final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getExponentValue() >= 0.0) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<PowOp>(
        op, op.getQubitsIn(), -op.getExponentValue(),
        [&](ValueRange powArgs) -> llvm::SmallVector<Value> {
          return InvOp::create(
                     rewriter, op.getLoc(), powArgs,
                     [&](ValueRange invArgs) -> llvm::SmallVector<Value> {
                       IRMapping mapping;
                       auto* innerBody = op.getBody();
                       for (size_t i = 0; i < op.getNumTargets(); ++i) {
                         mapping.map(innerBody->getArgument(i), invArgs[i]);
                       }
                       return rewriter
                           .clone(*op.getBodyUnitary().getOperation(), mapping)
                           ->getResults();
                     })
              .getResults();
        });

    return success();
  }
};

/// pow(a) { pow(b) { g } }  =>  pow(a*b) { g }
struct MergeNestedPow final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    auto innerPow = llvm::dyn_cast<PowOp>(op.getBodyUnitary().getOperation());
    if (!innerPow) {
      return failure();
    }

    const double merged = op.getExponentValue() * innerPow.getExponentValue();

    rewriter.moveOpBefore(innerPow, op);
    innerPow->setOperands(op.getInputQubits());
    innerPow.setExponentAttr(rewriter.getF64FloatAttr(merged));
    rewriter.replaceOp(op, innerPow->getResults());
    return success();
  }
};

/// pow(p) { ctrl(q, g) }  =>  ctrl(q, pow(p, g))
struct MoveCtrlOutside final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    auto bodyUnitary = op.getBodyUnitary();
    auto innerCtrlOp = llvm::dyn_cast<CtrlOp>(bodyUnitary.getOperation());
    if (!innerCtrlOp) {
      return failure();
    }

    const auto numControls = innerCtrlOp.getNumControls();
    const auto numTargets = innerCtrlOp.getNumTargets();
    auto inputQubits = op.getInputQubits();
    auto controls = inputQubits.take_front(numControls);
    auto targets = inputQubits.take_back(numTargets);
    const double exponent = op.getExponentValue();

    rewriter.replaceOpWithNewOp<CtrlOp>(
        op, controls, targets,
        [&](ValueRange ctrlTargetArgs) -> llvm::SmallVector<Value> {
          return PowOp::create(
                     rewriter, op.getLoc(), ctrlTargetArgs, exponent,
                     [&](ValueRange powArgs) -> llvm::SmallVector<Value> {
                       IRMapping mapping;
                       auto* innerBody = innerCtrlOp.getBody();
                       for (size_t i = 0; i < numTargets; ++i) {
                         mapping.map(innerBody->getArgument(i), powArgs[i]);
                       }
                       auto* cloned = rewriter.clone(
                           *innerCtrlOp.getBodyUnitary().getOperation(),
                           mapping);
                       return cloned->getResults();
                     })
              .getResults();
        });

    return success();
  }
};

/// Check if a floating-point value is an integer
bool isIntegerExponent(double r) {
  return r == std::floor(r) && std::isfinite(r);
}

/// Materialize r * constant as an arith.constant
Value mulConst(double r, double c, PowOp op, PatternRewriter& rewriter) {
  return arith::ConstantOp::create(rewriter, op.getLoc(),
                                   rewriter.getF64FloatAttr(r * c));
}

/// Materialize exponent * param as arith ops
Value scaleByExponent(auto param, PowOp op, PatternRewriter& rewriter) {
  auto loc = op.getLoc();
  auto exponent =
      arith::ConstantOp::create(rewriter, loc, op.getExponentAttr());
  return arith::MulFOp::create(rewriter, loc, exponent, param);
}

template <typename GateOp>
static LogicalResult replaceOneTargetOneParam(auto theta, PowOp op,
                                              PatternRewriter& rewriter) {
  rewriter.replaceOpWithNewOp<GateOp>(op, op.getInputTarget(0), theta);
  return success();
}

template <typename GateOp>
LogicalResult replaceTwoTargetsOneParam(auto theta, PowOp op,
                                        PatternRewriter& rewriter) {
  rewriter.replaceOpWithNewOp<GateOp>(op, op.getInputTarget(0),
                                      op.getInputTarget(1), theta);
  return success();
}

template <typename GateOp>
LogicalResult replaceOneTargetTwoParams(auto theta, auto phi, PowOp op,
                                        PatternRewriter& rewriter) {
  rewriter.replaceOpWithNewOp<GateOp>(op, op.getInputTarget(0), theta, phi);
  return success();
}

template <typename GateOp>
LogicalResult replaceTwoTargetsTwoParams(auto theta, auto beta, PowOp op,
                                         PatternRewriter& rewriter) {
  rewriter.replaceOpWithNewOp<GateOp>(op, op.getInputTarget(0),
                                      op.getInputTarget(1), theta, beta);
  return success();
}

/**
 * @brief Fold pow(r) around gates into simpler operations.
 *
 * Rotation gates: multiply angle by exponent, e.g., pow(r) { rx(θ) } → rx(r*θ)
 * Phase/diagonal gates: convert to P gate, e.g., pow(r) { s } → p(r*π/2)
 * Hermitian gates (integer exponent): even → erase, odd → gate
 * Identity/barrier: pass through unchanged
 */
struct FoldPowIntoGate final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    auto* innerOp = op.getBodyUnitary().getOperation();
    const double r = op.getExponentValue();
    auto loc = op.getLoc();

    // Move supporting ops (constants, arithmetic) out of the body so their
    // Values are accessible from outside and survive PowOp erasure.
    for (auto& bodyOp : llvm::make_early_inc_range(*op.getBody())) {
      if (&bodyOp != innerOp && !llvm::isa<YieldOp>(&bodyOp)) {
        rewriter.moveOpBefore(&bodyOp, op);
      }
    }

    return llvm::TypeSwitch<Operation*, LogicalResult>(innerOp)
        // --- Rotation gates: multiply angle by exponent ---
        // pow(r) { gphase(θ) } → gphase(r*θ)
        .Case<GPhaseOp>([&](auto gate) {
          auto newParam = scaleByExponent(gate.getTheta(), op, rewriter);
          rewriter.replaceOpWithNewOp<GPhaseOp>(op, newParam);
          return success();
        })
        // pow(r) { rx/ry/rz/p(θ) } → rx/ry/rz/p(r*θ)
        .Case<RXOp, RYOp, RZOp, POp>([&](auto gate) {
          auto newParam = scaleByExponent(gate.getTheta(), op, rewriter);
          return replaceOneTargetOneParam<decltype(gate)>(newParam, op,
                                                          rewriter);
        })
        // pow(r) { rxx/ryy/rzx/rzz(θ) } → rxx/ryy/rzx/rzz(r*θ)
        .Case<RXXOp, RYYOp, RZXOp, RZZOp>([&](auto gate) {
          auto newParam = scaleByExponent(gate.getTheta(), op, rewriter);
          return replaceTwoTargetsOneParam<decltype(gate)>(newParam, op,
                                                           rewriter);
        })
        // pow(r) { r(θ, φ) } → r(r*θ, φ)
        .Case<ROp>([&](auto gate) {
          auto mul = scaleByExponent(gate.getTheta(), op, rewriter);
          return replaceOneTargetTwoParams<ROp>(mul, gate.getPhi(), op,
                                                rewriter);
        })
        // pow(r) { xx±yy(θ, β) } → xx±yy(r*θ, β)
        .Case<XXPlusYYOp, XXMinusYYOp>([&](auto gate) {
          auto mul = scaleByExponent(gate.getTheta(), op, rewriter);
          return replaceTwoTargetsTwoParams<decltype(gate)>(mul, gate.getBeta(),
                                                            op, rewriter);
        })
        // --- Pauli gates: decompose to rotation + global phase ---
        // pow(r) { z } → p(r*π)
        .Case<ZOp>([&](auto) {
          rewriter.replaceOpWithNewOp<POp>(
              op, op.getInputTarget(0),
              mulConst(r, std::numbers::pi, op, rewriter));
          return success();
        })
        // pow(r) { x } → gphase(-r*π/2); rx(r*π)
        .Case<XOp>([&](auto) {
          if (llvm::isa<CtrlOp, InvOp, PowOp>(op->getParentOp())) {
            return failure();
          }
          GPhaseOp::create(rewriter, loc,
                           mulConst(r, -std::numbers::pi / 2.0, op, rewriter));
          rewriter.replaceOpWithNewOp<RXOp>(
              op, op.getInputTarget(0),
              mulConst(r, std::numbers::pi, op, rewriter));
          return success();
        })
        // pow(r) { y } → gphase(-r*π/2); ry(r*π)
        .Case<YOp>([&](auto) {
          if (llvm::isa<CtrlOp, InvOp, PowOp>(op->getParentOp())) {
            return failure();
          }
          GPhaseOp::create(rewriter, loc,
                           mulConst(r, -std::numbers::pi / 2.0, op, rewriter));
          rewriter.replaceOpWithNewOp<RYOp>(
              op, op.getInputTarget(0),
              mulConst(r, std::numbers::pi, op, rewriter));
          return success();
        })
        // --- Phase/diagonal gates: convert to P gate ---
        // pow(r) { s } → p(r*π/2)
        .Case<SOp>([&](auto) {
          rewriter.replaceOpWithNewOp<POp>(
              op, op.getInputTarget(0),
              mulConst(r, std::numbers::pi / 2.0, op, rewriter));
          return success();
        })
        // pow(r) { sdg } → p(-r*π/2)
        .Case<SdgOp>([&](auto) {
          rewriter.replaceOpWithNewOp<POp>(
              op, op.getInputTarget(0),
              mulConst(r, -std::numbers::pi / 2.0, op, rewriter));
          return success();
        })
        // pow(r) { t } → p(r*π/4)
        .Case<TOp>([&](auto) {
          rewriter.replaceOpWithNewOp<POp>(
              op, op.getInputTarget(0),
              mulConst(r, std::numbers::pi / 4.0, op, rewriter));
          return success();
        })
        // pow(r) { tdg } → p(-r*π/4)
        .Case<TdgOp>([&](auto) {
          rewriter.replaceOpWithNewOp<POp>(
              op, op.getInputTarget(0),
              mulConst(r, -std::numbers::pi / 4.0, op, rewriter));
          return success();
        })
        // --- SX/SXdg gates: decompose to rotation + global phase ---
        // pow(r) { sx } → gphase(-r*π/4); rx(r*π/2)
        .Case<SXOp>([&](auto) {
          if (llvm::isa<CtrlOp, InvOp, PowOp>(op->getParentOp())) {
            return failure();
          }
          GPhaseOp::create(rewriter, loc,
                           mulConst(r, -std::numbers::pi / 4.0, op, rewriter));
          rewriter.replaceOpWithNewOp<RXOp>(
              op, op.getInputTarget(0),
              mulConst(r, std::numbers::pi / 2.0, op, rewriter));
          return success();
        })
        // pow(r) { sxdg } → gphase(r*π/4); rx(-r*π/2)
        .Case<SXdgOp>([&](auto) {
          if (llvm::isa<CtrlOp, InvOp, PowOp>(op->getParentOp())) {
            return failure();
          }
          GPhaseOp::create(rewriter, loc,
                           mulConst(r, std::numbers::pi / 4.0, op, rewriter));
          rewriter.replaceOpWithNewOp<RXOp>(
              op, op.getInputTarget(0),
              mulConst(r, -std::numbers::pi / 2.0, op, rewriter));
          return success();
        })
        // --- Hermitian gates (integer exponent): even → erase, odd → gate ---
        // pow(n) { h/ecr } → id (n even) | h/ecr (n odd)
        .Case<HOp, ECROp>([&](auto gate) {
          if (!isIntegerExponent(r)) {
            return failure();
          }
          const auto n = static_cast<int64_t>(r);
          if (n % 2 == 0) {
            rewriter.replaceOp(op, op.getQubitsIn());
          } else {
            rewriter.moveOpBefore(gate, op);
            gate->setOperands(op.getQubitsIn());
            rewriter.replaceOp(op, gate->getResults());
          }
          return success();
        })
        // pow(n) { swap } → id (n even) | swap (n odd)
        .Case<SWAPOp>([&](auto gate) {
          if (!isIntegerExponent(r)) {
            return failure();
          }
          const auto n = static_cast<int64_t>(r);
          if (n % 2 == 0) {
            rewriter.replaceOp(op, op.getQubitsIn());
          } else {
            rewriter.moveOpBefore(gate, op);
            gate->setOperands(op.getQubitsIn());
            rewriter.replaceOp(op, gate->getResults());
          }
          return success();
        })
        // --- iSWAP: decompose to parametric gate ---
        // pow(r) { iswap } → xx_plus_yy(-r*π, 0)
        .Case<iSWAPOp>([&](auto) {
          rewriter.replaceOpWithNewOp<XXPlusYYOp>(
              op, op.getInputTarget(0), op.getInputTarget(1),
              mulConst(r, -std::numbers::pi, op, rewriter),
              mulConst(r, 0.0, op, rewriter));
          return success();
        })
        // --- Identity and barrier: pass through unchanged ---
        // pow(r) { id } → id
        .Case<IdOp>([&](auto) {
          rewriter.replaceOpWithNewOp<IdOp>(op, op.getInputTarget(0));
          return success();
        })
        // pow(r) { barrier } → barrier
        .Case<BarrierOp>([&](auto) {
          rewriter.replaceOpWithNewOp<BarrierOp>(op, op.getQubitsIn());
          return success();
        })
        .Default([&](auto) { return failure(); });
  }
};

} // namespace

UnitaryOpInterface PowOp::getBodyUnitary() {
  // In principle, the body region should only contain exactly two operations,
  // the actual unitary operation and a yield operation. However, the region may
  // also contain constants and arithmetic operations, e.g., created as part of
  // canonicalization. Thus, the only safe way to access the unitary operation
  // is to get the second operation from the back of the region.
  return llvm::cast<UnitaryOpInterface>(*(++getBody()->rbegin()));
}

Value PowOp::getInputQubit(const size_t i) {
  if (i >= getNumTargets()) {
    llvm::reportFatalUsageError("Qubit index out of bounds");
  }
  return getQubitsIn()[i];
}

Value PowOp::getOutputQubit(const size_t i) {
  if (i >= getNumTargets()) {
    llvm::reportFatalUsageError("Qubit index out of bounds");
  }
  return getQubitsOut()[i];
}

Value PowOp::getInputForOutput(Value output) {
  for (size_t i = 0; i < getNumTargets(); ++i) {
    if (output == getQubitsOut()[i]) {
      return getQubitsIn()[i];
    }
  }
  llvm::reportFatalUsageError("Given qubit is not an output of the operation");
}

Value PowOp::getOutputForInput(Value input) {
  for (size_t i = 0; i < getNumTargets(); ++i) {
    if (input == getQubitsIn()[i]) {
      return getQubitsOut()[i];
    }
  }
  llvm::reportFatalUsageError("Given qubit is not an input of the operation");
}

void PowOp::build(
    OpBuilder& odsBuilder, OperationState& odsState, ValueRange qubits,
    double exponent,
    llvm::function_ref<llvm::SmallVector<Value>(ValueRange)> bodyBuilder) {
  build(odsBuilder, odsState, qubits, exponent);
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

LogicalResult PowOp::verify() {
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
  if (!llvm::isa<UnitaryOpInterface>(*iter)) {
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

void PowOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<InlinePow1, ErasePow0, NegPowToInvPow, MergeNestedPow,
              MoveCtrlOutside>(context);
  // Prefer Known Gate optimizations over everything else
  results.add<FoldPowIntoGate>(context, /*benefit=*/2);
}

std::optional<Eigen::MatrixXcd> PowOp::getUnitaryMatrix() {
  auto&& bodyUnitary = getBodyUnitary();
  if (!bodyUnitary) {
    return std::nullopt;
  }
  auto&& targetMatrix = bodyUnitary.getUnitaryMatrix<Eigen::MatrixXcd>();
  if (!targetMatrix) {
    return std::nullopt;
  }

  const double p = getExponentValue();

  // U^1 = U (no computation needed)
  if (p == 1.0) {
    return targetMatrix;
  }

  // U^0 = I
  if (p == 0.0) {
    const auto dim = targetMatrix->cols();
    return Eigen::MatrixXcd::Identity(dim, dim);
  }

  // General case: eigendecomposition U = V D V† => U^p = V D^p V†
  const Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(*targetMatrix);
  if (solver.info() != Eigen::Success) {
    return std::nullopt;
  }

  const auto& eigenvalues = solver.eigenvalues();
  const auto& V = solver.eigenvectors();

  // Compute D^p: raise each eigenvalue to the power p
  Eigen::VectorXcd powEigenvalues(eigenvalues.size());
  for (Eigen::Index i = 0; i < eigenvalues.size(); ++i) {
    powEigenvalues[i] = std::pow(eigenvalues[i], p);
  }

  return V * powEigenvalues.asDiagonal() * V.inverse();
}
