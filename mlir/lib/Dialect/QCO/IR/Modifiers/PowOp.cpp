/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/Utils/Angles.h"
#include "mqt/Dialect/MQT/Utils/ConstantFolding.h"
#include "mqt/Dialect/MQT/Utils/GatePowering.h"
#include "mqt/Dialect/MQT/Utils/Modifiers.h"
#include "mqt/Dialect/MQT/Utils/Parameters.h"
#include "mqt/Dialect/MQT/Utils/PowerFolding.h"
#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QCO/Utils/Matrix.h"

#include "ModifierUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>

using namespace mlir;
using namespace mlir::qco;
using namespace mlir::mqt;

namespace {

/// The QCO operations that power folding recognizes or creates.
struct PowerFoldingOps {
  using PowOp = qco::PowOp;
  using UnitaryOpInterface = qco::UnitaryOpInterface;
  using GPhaseOp = qco::GPhaseOp;
  using IdOp = qco::IdOp;
  using BarrierOp = qco::BarrierOp;
  using XOp = qco::XOp;
  using YOp = qco::YOp;
  using ZOp = qco::ZOp;
  using HOp = qco::HOp;
  using SOp = qco::SOp;
  using SdgOp = qco::SdgOp;
  using TOp = qco::TOp;
  using TdgOp = qco::TdgOp;
  using SXOp = qco::SXOp;
  using SXdgOp = qco::SXdgOp;
  using RXOp = qco::RXOp;
  using RYOp = qco::RYOp;
  using RZOp = qco::RZOp;
  using POp = qco::POp;
  using ROp = qco::ROp;
  using UOp = qco::UOp;
  using RXXOp = qco::RXXOp;
  using RYYOp = qco::RYYOp;
  using RZXOp = qco::RZXOp;
  using RZZOp = qco::RZZOp;
  using XXPlusYYOp = qco::XXPlusYYOp;
  using XXMinusYYOp = qco::XXMinusYYOp;
  using iSWAPOp = qco::iSWAPOp;
};

/// pow(p) with p < 0 → pow(-p) { inv { U } }
struct NegPowToInvPow final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    const auto exponent = op.getExponentValue();
    // U^{-r} = (U^{-1})^r only when r is an integer: for fractional r,
    // eigenvalue -1 yields (-1)^{-r} ≠ (-1)^r (conjugated phase factors).
    if (!exponent || *exponent >= 0.0 || !mqt::isIntegerExponent(-*exponent)) {
      return failure();
    }
    const double exp = *exponent;

    rewriter.replaceOpWithNewOp<PowOp>(
        op, op.getQubitsIn(), -exp,
        [&](ValueRange powArgs) -> SmallVector<Value> {
          return InvOp::create(rewriter, op.getLoc(), powArgs,
                               [&](ValueRange invArgs) -> SmallVector<Value> {
                                 return mqt::inlineBodyReturningYields(
                                     *op.getBody(), invArgs, rewriter);
                               })
              .getResults();
        });

    return success();
  }
};

/// pow(a) { pow(b) { U } } → pow(a*b) { U }
struct MergeNestedPow final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    const auto outerExponent = op.getExponentValue();
    // Principal matrix powers do not generally satisfy (U^b)^a = U^(a*b)
    // across branch cuts. The rewrite is valid for integral outer powers,
    // where any branch phase is raised to an integer and cancels.
    if (!outerExponent || !mqt::isIntegerExponent(*outerExponent)) {
      return failure();
    }
    auto bodyUnitary =
        mqt::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (!bodyUnitary) {
      return failure();
    }
    auto innerPow = dyn_cast<PowOp>(bodyUnitary.getOperation());
    if (!innerPow) {
      return failure();
    }
    const auto innerExponent = innerPow.getExponentValue();
    if (!innerExponent) {
      return failure();
    }
    const auto mergedExponent =
        scaleConstantAngle(*innerExponent, *outerExponent);
    if (!mergedExponent) {
      return failure();
    }

    // The rewrite hands the qubits of the modifier to the inner operation, so
    // it must act on all of them.
    if (innerPow.getNumQubits() != op.getNumQubits()) {
      return failure();
    }

    // The inner pow's operands alias the outer pow's block args, possibly in a
    // different order / subset. Translate them back to the outer pow's operands
    // so the merged pow's footprint matches the inner pow positionally.
    auto outerQubits = op.getQubitsIn();
    const auto qubits =
        llvm::map_to_vector(innerPow.getInputQubits(), [&](Value v) {
          return mqt::getValueFromBlockArgument(v, outerQubits);
        });
    // Move supporting ops (constants, arithmetic) out of the body so their
    // Values are accessible from outside and survive PowOp erasure.
    mqt::hoistSupportingOpsBefore(*op.getBody(), innerPow.getOperation(), op,
                                  rewriter);
    auto newPow =
        PowOp::create(rewriter, op.getLoc(), qubits, *mergedExponent,
                      [&](ValueRange powArgs) -> llvm::SmallVector<Value> {
                        // Inner pow body args now match the new pow's args
                        // positionally.
                        return mqt::inlineBodyReturningYields(
                            *innerPow.getBody(), powArgs, rewriter);
                      });

    // The merged pow's operands may be a permutation of the outer pow's, so map
    // each original qubit output to the merged pow's output for the same input
    // rather than replacing positionally.
    qco::detail::replacePermutedModifier(op, newPow, rewriter);
    return success();
  }
};

/// pow(p) { ctrl(q) { U } } → ctrl(q) { pow(p) { U } }
struct MoveCtrlOutsidePow final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    auto inner = mqt::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (!inner) {
      return failure();
    }
    auto innerCtrlOp = dyn_cast<CtrlOp>(inner.getOperation());
    if (!innerCtrlOp) {
      return failure();
    }

    // The rewrite hands the qubits of the modifier to the inner operation, so
    // it must act on all of them.
    if (innerCtrlOp.getNumQubits() != op.getNumQubits()) {
      return failure();
    }

    // pow(p) { ctrl(x) } == ctrl(pow(p) { x }). The inner control's controls
    // and targets are block arguments aliasing the power modifier's qubits.
    // Pull the controls out to a new control modifier and wrap the inner body
    // in a power modifier whose block arguments match the inner targets, so the
    // inner body is reused verbatim.
    auto outerQubits = op.getQubitsIn();
    const auto controls =
        llvm::map_to_vector(innerCtrlOp.getControlsIn(), [&](Value c) {
          return mqt::getValueFromBlockArgument(c, outerQubits);
        });
    const auto targets =
        llvm::map_to_vector(innerCtrlOp.getTargetsIn(), [&](Value t) {
          return mqt::getValueFromBlockArgument(t, outerQubits);
        });

    mqt::hoistSupportingOpsBefore(*op.getBody(), innerCtrlOp, op, rewriter);

    auto newCtrl = CtrlOp::create(
        rewriter, op.getLoc(), controls, targets,
        [&](ValueRange targetArgs) -> SmallVector<Value> {
          auto innerPow =
              PowOp::create(rewriter, op.getLoc(), targetArgs, op.getExponent(),
                            [&](ValueRange powArgs) -> SmallVector<Value> {
                              return mqt::inlineBodyReturningYields(
                                  *innerCtrlOp.getBody(), powArgs, rewriter);
                            });
          return innerPow.getResults();
        });

    // Each qubit output of the power modifier follows its input qubit to the
    // corresponding output of the new control modifier.
    qco::detail::replacePermutedModifier(op, newCtrl, rewriter);
    return success();
  }
};

/// Erase power modifiers that do not have any body unitaries.
struct EraseEmptyPow final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    if (!op.getBody()->getOps<UnitaryOpInterface>().empty()) {
      return failure();
    }

    rewriter.replaceOp(op, op.getInputQubits());
    return success();
  }
};

/// Drop the qubits that the body does not use.
struct DropUnusedPowQubits final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    auto* body = op.getBody();
    auto qubits = op.getQubitsIn();
    return qco::detail::dropUnusedQubits(
        op, *body, qubits,
        [&](ValueRange narrowedQubits, ArrayRef<size_t> used) -> Operation* {
          auto newOp = PowOp::create(
              rewriter, op.getLoc(), narrowedQubits, op.getExponent(),
              [&](ValueRange args) -> SmallVector<Value> {
                return qco::detail::inlineNarrowedBody(*body, qubits, used,
                                                       args, rewriter);
              });
          return newOp;
        },
        rewriter);
  }
};

} // namespace

std::optional<double> PowOp::getExponentValue() {
  return mlir::mqt::valueToDouble(getExponent());
}

size_t PowOp::getNumBodyUnitaries() {
  return mqt::getNumBodyUnitaries<UnitaryOpInterface>(*getBody());
}

UnitaryOpInterface PowOp::getBodyUnitary(const size_t i) {
  return mqt::getBodyUnitary<UnitaryOpInterface>(*getBody(), i);
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
  if (auto result = dyn_cast<OpResult>(output);
      result && result.getOwner() == getOperation()) {
    return getInputQubit(result.getResultNumber());
  }
  llvm::reportFatalUsageError("Given qubit is not an output of the operation");
}

Value PowOp::getOutputForInput(Value input) {
  for (auto [in, out] : llvm::zip_equal(getInputQubits(), getOutputQubits())) {
    if (in == input) {
      return out;
    }
  }
  llvm::reportFatalUsageError("Given qubit is not an input of the operation");
}

void PowOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                  ValueRange qubits,
                  const std::variant<double, Value>& exponent) {
  auto expValue = variantToValue(odsBuilder, odsState.location, exponent);
  build(odsBuilder, odsState, qubits.getTypes(), expValue, qubits);
}

void PowOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                  ValueRange qubits,
                  const std::variant<double, Value>& exponent,
                  function_ref<SmallVector<Value>(ValueRange)> bodyBuilder) {
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

void PowOp::build(OpBuilder& odsBuilder, OperationState& odsState, Value qubit,
                  const std::variant<double, Value>& exponent,
                  function_ref<Value(Value)> bodyBuilder) {
  auto expValue = variantToValue(odsBuilder, odsState.location, exponent);
  build(odsBuilder, odsState, qubit.getType(), expValue, qubit);
  auto& block = odsState.regions.front()->emplaceBlock();
  block.addArgument(QubitType::get(odsBuilder.getContext()), odsState.location);

  const OpBuilder::InsertionGuard guard(odsBuilder);
  odsBuilder.setInsertionPointToStart(&block);
  YieldOp::create(odsBuilder, odsState.location,
                  bodyBuilder(block.getArgument(0)));
}

LogicalResult PowOp::verifyRegions() {
  return detail::verifyModifierBody(getOperation(), *getBody());
}

void PowOp::getLoweringPatterns(RewritePatternSet& results,
                                MLIRContext* context) {
  results.add(mqt::foldTrivialPower<UnitaryOpInterface, PowOp>);
  results.add<FoldPowIntoGate<PowerFoldingOps>, MergeNestedPow,
              MoveCtrlOutsidePow, NegPowToInvPow>(context);
  getCanonicalizationPatterns(results, context);
}

// This structural query deliberately avoids constructing the body matrix or
// running the eigensolver. A true result means all inputs needed to attempt the
// computation are known; getUnitaryMatrix() can still fail for unsupported
// bodies or numerical reasons.
bool PowOp::hasCompileTimeKnownUnitaryMatrix() {
  return mqt::valueToConstantDouble(getExponent()).has_value() &&
         all_of(getBody()->getOps<UnitaryOpInterface>(),
                [](UnitaryOpInterface op) {
                  return op.hasCompileTimeKnownUnitaryMatrix();
                });
}

std::optional<DynamicMatrix> PowOp::getUnitaryMatrix() {
  // Matrix queries evaluate computed exponents like gate parameters.
  const auto exponent = mqt::valueToConstantDouble(getExponent());
  return exponent ? getUnitaryMatrix(*exponent) : std::nullopt;
}

std::optional<DynamicMatrix> PowOp::getUnitaryMatrix(double p) {
  if (!std::isfinite(p)) {
    return std::nullopt;
  }
  if (getNumTargets() == 1) {
    if (auto gate = mqt::getSoleBodyUnitary<UnitaryOpInterface>(*getBody())) {
      if (const auto period = getFixedGatePowerPeriod(gate.getBaseSymbol())) {
        p = std::remainder(p, static_cast<double>(period));
      }
    }
  }

  // Raise a fully compile-time-known body matrix U to the power p via the
  // eigendecomposition U = V D V^{-1} → U^p = V D^p V^{-1}. PowOp bodies are
  // unitary, so U is normal and its eigenvectors form a unitary V, giving
  // V^{-1} = V†.
  const auto raiseToPow =
      [p](const DynamicMatrix& u) -> std::optional<DynamicMatrix> {
    // U^1 = U (no computation needed)
    if (p == 1.) {
      return u;
    }

    // U^0 = I
    if (p == 0.) {
      return DynamicMatrix::identity(u.cols());
    }

    const auto eigen = u.eigenDecomposition();
    if (!eigen) {
      return std::nullopt;
    }
    const auto& eigenvalues = eigen->eigenvalues;
    const auto& v = eigen->eigenvectors;
    const std::int64_t dim = v.cols();

    // Reject non-orthonormal eigenvectors (e.g. unresolved degenerate
    // subspaces).
    constexpr double eigenSolverTol = 1e-10;
    if (!(v * v.adjoint())
             .isApprox(DynamicMatrix::identity(dim), eigenSolverTol)) {
      return std::nullopt;
    }

    // Build D^p by raising each eigenvalue to the power p (principal branch).
    DynamicMatrix powDiagonal(dim);
    for (std::int64_t i = 0; i < dim; ++i) {
      powDiagonal(i, i) = std::pow(eigenvalues[static_cast<size_t>(i)], p);
    }

    return v * powDiagonal * v.adjoint();
  };

  if (const auto composed = composeBodyMatrix(*getBody(), getNumTargets())) {
    return raiseToPow(*composed);
  }

  return std::nullopt;
}

void PowOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<EraseEmptyPow, DropUnusedPowQubits>(context);
}
