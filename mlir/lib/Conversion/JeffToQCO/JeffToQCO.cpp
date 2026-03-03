/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/JeffToQCO/JeffToQCO.h"

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"

#include <jeff/IR/JeffDialect.h>
#include <jeff/IR/JeffOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <utility>

namespace mlir {
using namespace qco;

#define GEN_PASS_DEF_JEFFTOQCO
#include "mlir/Conversion/JeffToQCO/JeffToQCO.h.inc"

/**
 * @brief Converts a one-target, zero-parameter Jeff operation to QCO
 *
 * @tparam QCOOpType The operation type of the QCO operation
 * @tparam JeffOpType The operation type of the Jeff operation
 * @tparam JeffOpAdaptorType The OpAdaptor type of the Jeff operation
 * @param op The Jeff operation instance to convert
 * @param adaptor The OpAdaptor of the Jeff operation
 * @param rewriter The pattern rewriter
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QCOOpType, typename JeffOpType, typename JeffOpAdaptorType>
static LogicalResult
convertOneTargetZeroParameter(JeffOpType& op, JeffOpAdaptorType& adaptor,
                              ConversionPatternRewriter& rewriter) {
  if (op.getPower() != 1) {
    return rewriter.notifyMatchFailure(
        op, "Operations with power != 1 are not yet supported");
  }

  auto loc = op.getLoc();

  if (op.getNumCtrls() != 0 && op.getIsAdjoint()) {
    auto ctrlOp = rewriter.create<qco::CtrlOp>(
        loc, adaptor.getInCtrlQubits(), adaptor.getInQubit(),
        [&](ValueRange ctrlTargets) -> llvm::SmallVector<Value> {
          auto invOp = rewriter.create<qco::InvOp>(
              loc, ctrlTargets,
              [&](ValueRange invTargets) -> llvm::SmallVector<Value> {
                auto qcoOp = rewriter.create<QCOOpType>(loc, invTargets[0]);
                return {qcoOp.getQubitOut()};
              });
          return invOp.getQubitsOut();
        });
    llvm::SmallVector<Value> results;
    results.append(ctrlOp.getTargetsOut().begin(),
                   ctrlOp.getTargetsOut().end());
    results.append(ctrlOp.getControlsOut().begin(),
                   ctrlOp.getControlsOut().end());
    rewriter.replaceOp(op, results);
  } else if (op.getNumCtrls() != 0 && !op.getIsAdjoint()) {
    auto ctrlOp = rewriter.create<qco::CtrlOp>(
        loc, adaptor.getInCtrlQubits(), adaptor.getInQubit(),
        [&](ValueRange targets) -> llvm::SmallVector<Value> {
          auto qcoOp = rewriter.create<QCOOpType>(loc, targets[0]);
          return {qcoOp.getQubitOut()};
        });
    llvm::SmallVector<Value> results;
    results.append(ctrlOp.getTargetsOut().begin(),
                   ctrlOp.getTargetsOut().end());
    results.append(ctrlOp.getControlsOut().begin(),
                   ctrlOp.getControlsOut().end());
    rewriter.replaceOp(op, results);
  } else if (op.getNumCtrls() == 0 && op.getIsAdjoint()) {
    auto invOp = rewriter.create<qco::InvOp>(
        loc, adaptor.getInQubit(),
        [&](ValueRange targets) -> llvm::SmallVector<Value> {
          auto qcoOp = rewriter.create<QCOOpType>(loc, targets[0]);
          return {qcoOp.getQubitOut()};
        });
    rewriter.replaceOp(op, invOp.getQubitsOut());
  } else {
    rewriter.replaceOpWithNewOp<QCOOpType>(op, adaptor.getInQubit());
  }

  return success();
}

/**
 * @brief Converts a one-target, one-parameter Jeff operation to QCO
 *
 * @tparam QCOOpType The operation type of the QCO operation
 * @tparam JeffOpType The operation type of the Jeff operation
 * @tparam JeffOpAdaptorType The OpAdaptor type of the Jeff operation
 * @param op The Jeff operation instance to convert
 * @param adaptor The OpAdaptor of the Jeff operation
 * @param rewriter The pattern rewriter
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QCOOpType, typename JeffOpType, typename JeffOpAdaptorType>
static LogicalResult
convertOneTargetOneParameter(JeffOpType& op, JeffOpAdaptorType& adaptor,
                             ConversionPatternRewriter& rewriter) {
  if (op.getPower() != 1) {
    return rewriter.notifyMatchFailure(
        op, "Operations with power != 1 are not yet supported");
  }

  auto loc = op.getLoc();

  if (op.getNumCtrls() != 0 && op.getIsAdjoint()) {
    auto ctrlOp = rewriter.create<qco::CtrlOp>(
        loc, adaptor.getInCtrlQubits(), adaptor.getInQubit(),
        [&](ValueRange ctrlTargets) -> llvm::SmallVector<Value> {
          auto invOp = rewriter.create<qco::InvOp>(
              loc, ctrlTargets,
              [&](ValueRange invTargets) -> llvm::SmallVector<Value> {
                auto qcoOp = rewriter.create<QCOOpType>(loc, invTargets[0],
                                                        op.getRotation());
                return {qcoOp.getQubitOut()};
              });
          return invOp.getQubitsOut();
        });
    llvm::SmallVector<Value> results;
    results.append(ctrlOp.getTargetsOut().begin(),
                   ctrlOp.getTargetsOut().end());
    results.append(ctrlOp.getControlsOut().begin(),
                   ctrlOp.getControlsOut().end());
    rewriter.replaceOp(op, results);
  } else if (op.getNumCtrls() != 0 && !op.getIsAdjoint()) {
    auto ctrlOp = rewriter.create<qco::CtrlOp>(
        loc, adaptor.getInCtrlQubits(), adaptor.getInQubit(),
        [&](ValueRange targets) -> llvm::SmallVector<Value> {
          auto qcoOp =
              rewriter.create<QCOOpType>(loc, targets[0], op.getRotation());
          return {qcoOp.getQubitOut()};
        });
    llvm::SmallVector<Value> results;
    results.append(ctrlOp.getTargetsOut().begin(),
                   ctrlOp.getTargetsOut().end());
    results.append(ctrlOp.getControlsOut().begin(),
                   ctrlOp.getControlsOut().end());
    rewriter.replaceOp(op, results);
  } else if (op.getNumCtrls() == 0 && op.getIsAdjoint()) {
    auto invOp = rewriter.create<qco::InvOp>(
        loc, adaptor.getInQubit(),
        [&](ValueRange targets) -> llvm::SmallVector<Value> {
          auto qcoOp =
              rewriter.create<QCOOpType>(loc, targets[0], op.getRotation());
          return {qcoOp.getQubitOut()};
        });
    rewriter.replaceOp(op, invOp.getQubitsOut());
  } else {
    rewriter.replaceOpWithNewOp<QCOOpType>(op, adaptor.getInQubit(),
                                           op.getRotation());
  }

  return success();
}

/**
 * @brief Converts jeff.qubit_alloc to qco.alloc
 *
 * @par Example:
 * ```mlir
 * %q = jeff.qubit_alloc : !jeff.qubit
 * ```
 * is converted to
 * ```mlir
 * %q = qco.alloc : !qco.qubit
 * ```
 */
struct ConvertJeffQubitAllocOpToQCO final
    : OpConversionPattern<jeff::QubitAllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::QubitAllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<qco::AllocOp>(op);
    return success();
  }
};

/**
 * @brief Converts jeff.qubit_free to qco.reset + qco.dealloc
 *
 * @par Example:
 * ```mlir
 * jeff.qubit_free %q : !jeff.qubit
 * ```
 * is converted to
 * ```mlir
 * %q_out = qco.reset %q_in : !qco.qubit
 * qco.dealloc %q_out : !qco.qubit
 * ```
 */
struct ConvertJeffQubitFreeOpToQCO final
    : OpConversionPattern<jeff::QubitFreeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::QubitFreeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto resetOp =
        rewriter.create<qco::ResetOp>(op.getLoc(), adaptor.getInQubit());
    rewriter.replaceOpWithNewOp<qco::DeallocOp>(op, resetOp.getQubitOut());
    return success();
  }
};

/**
 * @brief Converts jeff.qubit_free_zero to qco.dealloc
 *
 * @par Example:
 * ```mlir
 * jeff.qubit_free_zero %q : !jeff.qubit
 * ```
 * is converted to
 * ```mlir
 * qco.dealloc %q : !qco.qubit
 * ```
 */
struct ConvertJeffQubitFreeZeroOpToQCO final
    : OpConversionPattern<jeff::QubitFreeZeroOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::QubitFreeZeroOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<qco::DeallocOp>(op, adaptor.getInQubit());
    return success();
  }
};

/**
 * @brief Converts jeff.qubit_measure to qco.measure + qco.dealloc
 *
 * @par Example:
 * ```mlir
 * %result = jeff.qubit_measure %q_in : !i1
 * ```
 * is converted to
 * ```mlir
 * %q_out, %result = qco.measure %q_in : !qco.qubit
 * qco.dealloc %q_out : !qco.qubit
 * ```
 */
struct ConvertJeffQubitMeasureOpToQCO final
    : OpConversionPattern<jeff::QubitMeasureOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::QubitMeasureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto measureOp =
        rewriter.replaceOpWithNewOp<qco::MeasureOp>(op, adaptor.getInQubit());
    rewriter.create<qco::DeallocOp>(measureOp.getLoc(),
                                    measureOp.getQubitOut());
    return success();
  }
};

/**
 * @brief Converts jeff.qubit_measure_nd to qco.measure
 *
 * @par Example:
 * ```mlir
 * %q_out, %result = jeff.qubit_measure_nd %q_in : !jeff.qubit, i1
 * ```
 * is converted to
 * ```mlir
 * %q_out, %result = qco.measure %q_in : !qco.qubit
 * ```
 */
struct ConvertJeffQubitMeasureNDOpToQCO final
    : OpConversionPattern<jeff::QubitMeasureNDOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::QubitMeasureNDOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<qco::MeasureOp>(op, adaptor.getInQubit());
    return success();
  }
};

/**
 * @brief Converts jeff.reset to qco.qubit_reset
 *
 * @par Example:
 * ```mlir
 * %q_out = jeff.qubit_reset %q_in : !jeff.qubit
 * ```
 * is converted to
 * ```mlir
 * %q_out = qco.reset %q_in : !qco.qubit
 * ```
 */
struct ConvertJeffQubitResetOpToQCO final
    : OpConversionPattern<jeff::QubitResetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::QubitResetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<qco::ResetOp>(op, adaptor.getInQubit());
    return success();
  }
};

/**
 * @brief Converts jeff.gphase to qco.gphase
 *
 * @par Example:
 * ```mlir
 * jeff.gphase(%theta) {is_adjoint = false, num_ctrls = 0 : i8, power = 1 : i8}
 * ```
 * is converted to
 * ```mlir
 * qco.gphase(%theta)
 * ```
 */
struct ConvertJeffGPhaseOpToQCO final : OpConversionPattern<jeff::GPhaseOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::GPhaseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    if (op.getPower() != 1) {
      return rewriter.notifyMatchFailure(
          op, "Operations with power != 1 are not yet supported");
    }

    auto loc = op.getLoc();

    if (op.getNumCtrls() != 0 && op.getIsAdjoint()) {
      auto ctrlOp = rewriter.create<qco::CtrlOp>(
          loc, adaptor.getInCtrlQubits(), ValueRange{},
          [&](ValueRange /*targets*/) -> llvm::SmallVector<Value> {
            rewriter.create<qco::InvOp>(
                loc, ValueRange{},
                [&](ValueRange /*targets*/) -> llvm::SmallVector<Value> {
                  rewriter.create<qco::GPhaseOp>(loc, op.getRotation());
                  return {};
                });
            return {};
          });
      llvm::SmallVector<Value> results;
      results.append(ctrlOp.getControlsOut().begin(),
                     ctrlOp.getControlsOut().end());
      rewriter.replaceOp(op, results);
    } else if (op.getNumCtrls() != 0 && !op.getIsAdjoint()) {
      auto ctrlOp = rewriter.create<qco::CtrlOp>(
          loc, adaptor.getInCtrlQubits(), ValueRange{},
          [&](ValueRange /*targets*/) -> llvm::SmallVector<Value> {
            rewriter.create<qco::GPhaseOp>(loc, op.getRotation());
            return {};
          });
      llvm::SmallVector<Value> results;
      results.append(ctrlOp.getControlsOut().begin(),
                     ctrlOp.getControlsOut().end());
      rewriter.replaceOp(op, results);
    } else if (op.getNumCtrls() == 0 && op.getIsAdjoint()) {
      rewriter.create<qco::InvOp>(
          loc, ValueRange{},
          [&](ValueRange /*targets*/) -> llvm::SmallVector<Value> {
            rewriter.create<qco::GPhaseOp>(loc, op.getRotation());
            return {};
          });
    } else {
      rewriter.replaceOpWithNewOp<qco::GPhaseOp>(op, op.getRotation());
    }

    return success();
  }
};

// OneTargetZeroParameter

#define DEFINE_ONE_TARGET_ZERO_PARAMETER(OP_CLASS_JEFF, OP_CLASS_QCO)          \
  struct ConvertJeff##OP_CLASS_JEFF##ToQCO final                               \
      : OpConversionPattern<jeff::OP_CLASS_JEFF> {                             \
    using OpConversionPattern::OpConversionPattern;                            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(jeff::OP_CLASS_JEFF op, OpAdaptor adaptor,                 \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertOneTargetZeroParameter<qco::OP_CLASS_QCO>(op, adaptor,     \
                                                              rewriter);       \
    }                                                                          \
  };

DEFINE_ONE_TARGET_ZERO_PARAMETER(IOp, IdOp)
DEFINE_ONE_TARGET_ZERO_PARAMETER(XOp, XOp)
DEFINE_ONE_TARGET_ZERO_PARAMETER(YOp, YOp)
DEFINE_ONE_TARGET_ZERO_PARAMETER(ZOp, ZOp)
DEFINE_ONE_TARGET_ZERO_PARAMETER(HOp, HOp)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SOp, SOp)
DEFINE_ONE_TARGET_ZERO_PARAMETER(TOp, TOp)

#undef DEFINE_ONE_TARGET_ZERO_PARAMETER

// OneTargetOneParameter

#define DEFINE_ONE_TARGET_ONE_PARAMETER(OP_CLASS_JEFF, OP_CLASS_QCO)           \
  struct ConvertJeff##OP_CLASS_JEFF##ToQCO final                               \
      : OpConversionPattern<jeff::OP_CLASS_JEFF> {                             \
    using OpConversionPattern::OpConversionPattern;                            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(jeff::OP_CLASS_JEFF op, OpAdaptor adaptor,                 \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertOneTargetOneParameter<qco::OP_CLASS_QCO>(op, adaptor,      \
                                                             rewriter);        \
    }                                                                          \
  };

DEFINE_ONE_TARGET_ONE_PARAMETER(RxOp, RXOp)
DEFINE_ONE_TARGET_ONE_PARAMETER(RyOp, RYOp)
DEFINE_ONE_TARGET_ONE_PARAMETER(RzOp, RZOp)
DEFINE_ONE_TARGET_ONE_PARAMETER(R1Op, POp)

#undef DEFINE_ONE_TARGET_ONE_PARAMETER

/**
 * @brief Converts jeff.u to qco.u
 *
 * @par Example:
 * ```mlir
 * %q_out = jeff.u(%theta, %phi, %lambda) {is_adjoint = false, num_ctrls = 0 :
 * i8, power = 1 : i8} %q_in : !jeff.qubit
 * ```
 * is converted to
 * ```mlir
 * %q_out = qco.u(%theta, %phi, %lambda) %q_in : !qco.qubit
 * ```
 */
struct ConvertJeffUOpToQCO final : OpConversionPattern<jeff::UOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::UOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    if (op.getPower() != 1) {
      return rewriter.notifyMatchFailure(
          op, "Operations with power != 1 are not yet supported");
    }

    auto loc = op.getLoc();

    if (op.getNumCtrls() != 0 && op.getIsAdjoint()) {
      auto ctrlOp = rewriter.create<qco::CtrlOp>(
          loc, adaptor.getInCtrlQubits(), adaptor.getInQubit(),
          [&](ValueRange ctrlTargets) -> llvm::SmallVector<Value> {
            auto invOp = rewriter.create<qco::InvOp>(
                loc, ctrlTargets,
                [&](ValueRange invTargets) -> llvm::SmallVector<Value> {
                  auto qcoOp = rewriter.create<qco::UOp>(
                      loc, invTargets[0], op.getTheta(), op.getPhi(),
                      op.getLambda());
                  return {qcoOp.getQubitOut()};
                });
            return invOp.getQubitsOut();
          });
      llvm::SmallVector<Value> results;
      results.append(ctrlOp.getTargetsOut().begin(),
                     ctrlOp.getTargetsOut().end());
      results.append(ctrlOp.getControlsOut().begin(),
                     ctrlOp.getControlsOut().end());
      rewriter.replaceOp(op, results);
    } else if (op.getNumCtrls() != 0 && !op.getIsAdjoint()) {
      auto ctrlOp = rewriter.create<qco::CtrlOp>(
          loc, adaptor.getInCtrlQubits(), adaptor.getInQubit(),
          [&](ValueRange targets) -> llvm::SmallVector<Value> {
            auto qcoOp = rewriter.create<qco::UOp>(
                loc, targets[0], op.getTheta(), op.getPhi(), op.getLambda());
            return {qcoOp.getQubitOut()};
          });
      llvm::SmallVector<Value> results;
      results.append(ctrlOp.getTargetsOut().begin(),
                     ctrlOp.getTargetsOut().end());
      results.append(ctrlOp.getControlsOut().begin(),
                     ctrlOp.getControlsOut().end());
      rewriter.replaceOp(op, results);
    } else if (op.getNumCtrls() == 0 && op.getIsAdjoint()) {
      auto invOp = rewriter.create<qco::InvOp>(
          loc, adaptor.getInQubit(),
          [&](ValueRange targets) -> llvm::SmallVector<Value> {
            auto qcoOp = rewriter.create<qco::UOp>(
                loc, targets[0], op.getTheta(), op.getPhi(), op.getLambda());
            return {qcoOp.getQubitOut()};
          });
      rewriter.replaceOp(op, invOp.getQubitsOut());
    } else {
      rewriter.replaceOpWithNewOp<qco::UOp>(
          op, adaptor.getInQubit(), op.getTheta(), op.getPhi(), op.getLambda());
    }

    return success();
  }
};

/**
 * @brief Converts jeff.swap to qco.swap
 *
 * @par Example:
 * ```mlir
 * %q0_out, %q1_out = jeff.swap {is_adjoint = false, num_ctrls = 0 : i8, power =
 * 1 : i8} %q0_in %q1_in : !jeff.qubit !jeff.qubit
 * ```
 * is converted to
 * ```mlir
 * %q0_out, %q1_out = qco.swap %q0_in, %q1_in : !qco.qubit, !qco.qubit
 * ```
 */
struct ConvertJeffSwapOpToQCO final : OpConversionPattern<jeff::SwapOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::SwapOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    if (op.getPower() != 1) {
      return rewriter.notifyMatchFailure(
          op, "Operations with power != 1 are not yet supported");
    }

    auto loc = op.getLoc();

    if (op.getNumCtrls() != 0 && op.getIsAdjoint()) {
      auto ctrlOp = rewriter.create<qco::CtrlOp>(
          loc, adaptor.getInCtrlQubits(),
          ValueRange{adaptor.getInQubitOne(), adaptor.getInQubitTwo()},
          [&](ValueRange ctrlTargets) -> llvm::SmallVector<Value> {
            auto invOp = rewriter.create<qco::InvOp>(
                loc, ctrlTargets,
                [&](ValueRange invTargets) -> llvm::SmallVector<Value> {
                  auto qcoOp = rewriter.create<qco::SWAPOp>(loc, invTargets[0],
                                                            invTargets[1]);
                  return {qcoOp.getQubit0Out(), qcoOp.getQubit1Out()};
                });
            return invOp.getQubitsOut();
          });
      llvm::SmallVector<Value> results;
      results.append(ctrlOp.getTargetsOut().begin(),
                     ctrlOp.getTargetsOut().end());
      results.append(ctrlOp.getControlsOut().begin(),
                     ctrlOp.getControlsOut().end());
      rewriter.replaceOp(op, results);
    } else if (op.getNumCtrls() != 0 && !op.getIsAdjoint()) {
      auto ctrlOp = rewriter.create<qco::CtrlOp>(
          loc, adaptor.getInCtrlQubits(),
          ValueRange{adaptor.getInQubitOne(), adaptor.getInQubitTwo()},
          [&](ValueRange targets) -> llvm::SmallVector<Value> {
            auto qcoOp =
                rewriter.create<qco::SWAPOp>(loc, targets[0], targets[1]);
            return {qcoOp.getQubit0Out(), qcoOp.getQubit1Out()};
          });
      llvm::SmallVector<Value> results;
      results.append(ctrlOp.getTargetsOut().begin(),
                     ctrlOp.getTargetsOut().end());
      results.append(ctrlOp.getControlsOut().begin(),
                     ctrlOp.getControlsOut().end());
      rewriter.replaceOp(op, results);
    } else if (op.getNumCtrls() == 0 && op.getIsAdjoint()) {
      auto invOp = rewriter.create<qco::InvOp>(
          loc, adaptor.getInQubitOne(),
          [&](ValueRange targets) -> llvm::SmallVector<Value> {
            auto qcoOp =
                rewriter.create<qco::SWAPOp>(loc, targets[0], targets[1]);
            return {qcoOp.getQubit0Out(), qcoOp.getQubit1Out()};
          });
      rewriter.replaceOp(op, invOp.getQubitsOut());
    } else {
      rewriter.replaceOpWithNewOp<qco::SWAPOp>(op, adaptor.getInQubitOne(),
                                               adaptor.getInQubitTwo());
    }

    return success();
  }
};

template <typename QCOOpType>
static void createOneTargetZeroParameter(jeff::CustomOp& op,
                                         jeff::CustomOpAdaptor& adaptor,
                                         ConversionPatternRewriter& rewriter) {
  auto loc = op.getLoc();
  if (op.getNumCtrls() != 0 && op.getIsAdjoint()) {
    auto ctrlOp = rewriter.create<qco::CtrlOp>(
        loc, adaptor.getInCtrlQubits(), adaptor.getInTargetQubits()[0],
        [&](ValueRange ctrlTargets) -> llvm::SmallVector<Value> {
          auto invOp = rewriter.create<qco::InvOp>(
              loc, ctrlTargets,
              [&](ValueRange invTargets) -> llvm::SmallVector<Value> {
                auto qcoOp = rewriter.create<QCOOpType>(loc, invTargets[0]);
                return {qcoOp.getQubitOut()};
              });
          return invOp.getQubitsOut();
        });
    llvm::SmallVector<Value> results;
    results.append(ctrlOp.getTargetsOut().begin(),
                   ctrlOp.getTargetsOut().end());
    results.append(ctrlOp.getControlsOut().begin(),
                   ctrlOp.getControlsOut().end());
    rewriter.replaceOp(op, results);
  } else if (op.getNumCtrls() != 0 && !op.getIsAdjoint()) {
    auto ctrlOp = rewriter.create<qco::CtrlOp>(
        op.getLoc(), adaptor.getInCtrlQubits(), adaptor.getInTargetQubits()[0],
        [&](ValueRange targets) -> llvm::SmallVector<Value> {
          auto qcoOp = rewriter.create<QCOOpType>(op.getLoc(), targets[0]);
          return {qcoOp.getQubitOut()};
        });
    llvm::SmallVector<Value> results;
    results.append(ctrlOp.getTargetsOut().begin(),
                   ctrlOp.getTargetsOut().end());
    results.append(ctrlOp.getControlsOut().begin(),
                   ctrlOp.getControlsOut().end());
    rewriter.replaceOp(op, results);
  } else if (op.getNumCtrls() == 0 && op.getIsAdjoint()) {
    auto invOp = rewriter.create<qco::InvOp>(
        op.getLoc(), adaptor.getInTargetQubits()[0],
        [&](ValueRange targets) -> llvm::SmallVector<Value> {
          auto qcoOp = rewriter.create<QCOOpType>(op.getLoc(), targets[0]);
          return {qcoOp.getQubitOut()};
        });
    rewriter.replaceOp(op, invOp.getQubitsOut());
  } else {
    rewriter.replaceOpWithNewOp<QCOOpType>(op, adaptor.getInTargetQubits()[0]);
  }
}

template <typename QCOOpType>
static void createOneTargetTwoParameter(jeff::CustomOp& op,
                                        jeff::CustomOpAdaptor& adaptor,
                                        ConversionPatternRewriter& rewriter) {
  auto loc = op.getLoc();
  if (op.getNumCtrls() != 0 && op.getIsAdjoint()) {
    auto ctrlOp = rewriter.create<qco::CtrlOp>(
        loc, adaptor.getInCtrlQubits(), adaptor.getInTargetQubits()[0],
        [&](ValueRange ctrlTargets) -> llvm::SmallVector<Value> {
          auto invOp = rewriter.create<qco::InvOp>(
              loc, ctrlTargets,
              [&](ValueRange invTargets) -> llvm::SmallVector<Value> {
                auto qcoOp = rewriter.create<QCOOpType>(
                    loc, invTargets[0], op.getParams()[0], op.getParams()[1]);
                return {qcoOp.getQubitOut()};
              });
          return invOp.getQubitsOut();
        });
    llvm::SmallVector<Value> results;
    results.append(ctrlOp.getTargetsOut().begin(),
                   ctrlOp.getTargetsOut().end());
    results.append(ctrlOp.getControlsOut().begin(),
                   ctrlOp.getControlsOut().end());
    rewriter.replaceOp(op, results);
  } else if (op.getNumCtrls() != 0 && !op.getIsAdjoint()) {
    auto ctrlOp = rewriter.create<qco::CtrlOp>(
        op.getLoc(), adaptor.getInCtrlQubits(), adaptor.getInTargetQubits()[0],
        [&](ValueRange targets) -> llvm::SmallVector<Value> {
          auto qcoOp = rewriter.create<QCOOpType>(
              op.getLoc(), targets[0], op.getParams()[0], op.getParams()[1]);
          return {qcoOp.getQubitOut()};
        });
    llvm::SmallVector<Value> results;
    results.append(ctrlOp.getTargetsOut().begin(),
                   ctrlOp.getTargetsOut().end());
    results.append(ctrlOp.getControlsOut().begin(),
                   ctrlOp.getControlsOut().end());
    rewriter.replaceOp(op, results);
  } else if (op.getNumCtrls() == 0 && op.getIsAdjoint()) {
    auto invOp = rewriter.create<qco::InvOp>(
        op.getLoc(), adaptor.getInTargetQubits()[0],
        [&](ValueRange targets) -> llvm::SmallVector<Value> {
          auto qcoOp = rewriter.create<QCOOpType>(
              op.getLoc(), targets[0], op.getParams()[0], op.getParams()[1]);
          return {qcoOp.getQubitOut()};
        });
    rewriter.replaceOp(op, invOp.getQubitsOut());
  } else {
    rewriter.replaceOpWithNewOp<QCOOpType>(op, adaptor.getInTargetQubits()[0],
                                           op.getParams()[0],
                                           op.getParams()[1]);
  }
}

template <typename QCOOpType>
static void createTwoTargetZeroParameter(jeff::CustomOp& op,
                                         jeff::CustomOpAdaptor& adaptor,
                                         ConversionPatternRewriter& rewriter) {
  auto loc = op.getLoc();
  if (op.getNumCtrls() != 0 && op.getIsAdjoint()) {
    auto ctrlOp = rewriter.create<qco::CtrlOp>(
        loc, adaptor.getInCtrlQubits(), adaptor.getInTargetQubits(),
        [&](ValueRange ctrlTargets) -> llvm::SmallVector<Value> {
          auto invOp = rewriter.create<qco::InvOp>(
              loc, ctrlTargets,
              [&](ValueRange invTargets) -> llvm::SmallVector<Value> {
                auto qcoOp = rewriter.create<QCOOpType>(loc, invTargets[0],
                                                        invTargets[1]);
                return {qcoOp.getQubit0Out(), qcoOp.getQubit1Out()};
              });
          return invOp.getQubitsOut();
        });
    llvm::SmallVector<Value> results;
    results.append(ctrlOp.getTargetsOut().begin(),
                   ctrlOp.getTargetsOut().end());
    results.append(ctrlOp.getControlsOut().begin(),
                   ctrlOp.getControlsOut().end());
    rewriter.replaceOp(op, results);
  } else if (op.getNumCtrls() != 0 && !op.getIsAdjoint()) {
    auto ctrlOp = rewriter.create<qco::CtrlOp>(
        op.getLoc(), adaptor.getInCtrlQubits(), adaptor.getInTargetQubits(),
        [&](ValueRange targets) -> llvm::SmallVector<Value> {
          auto qcoOp =
              rewriter.create<QCOOpType>(op.getLoc(), targets[0], targets[1]);
          return {qcoOp.getQubit0Out(), qcoOp.getQubit1Out()};
        });
    llvm::SmallVector<Value> results;
    results.append(ctrlOp.getTargetsOut().begin(),
                   ctrlOp.getTargetsOut().end());
    results.append(ctrlOp.getControlsOut().begin(),
                   ctrlOp.getControlsOut().end());
    rewriter.replaceOp(op, results);
  } else if (op.getNumCtrls() == 0 && op.getIsAdjoint()) {
    auto invOp = rewriter.create<qco::InvOp>(
        op.getLoc(), adaptor.getInTargetQubits(),
        [&](ValueRange targets) -> llvm::SmallVector<Value> {
          auto qcoOp =
              rewriter.create<QCOOpType>(op.getLoc(), targets[0], targets[1]);
          return {qcoOp.getQubit0Out(), qcoOp.getQubit1Out()};
        });
    rewriter.replaceOp(op, invOp.getQubitsOut());
  } else {
    rewriter.replaceOpWithNewOp<QCOOpType>(op, adaptor.getInTargetQubits()[0],
                                           adaptor.getInTargetQubits()[1]);
  }
}

template <typename QCOOpType>
static void createTwoTargetOneParameter(jeff::PPROp& op,
                                        jeff::PPROpAdaptor& adaptor,
                                        ConversionPatternRewriter& rewriter) {
  auto loc = op.getLoc();
  if (op.getNumCtrls() != 0 && op.getIsAdjoint()) {
    auto ctrlOp = rewriter.create<qco::CtrlOp>(
        loc, adaptor.getInCtrlQubits(), adaptor.getInQubits(),
        [&](ValueRange ctrlTargets) -> llvm::SmallVector<Value> {
          auto invOp = rewriter.create<qco::InvOp>(
              loc, ctrlTargets,
              [&](ValueRange invTargets) -> llvm::SmallVector<Value> {
                auto qcoOp = rewriter.create<QCOOpType>(
                    loc, invTargets[0], invTargets[1], op.getRotation());
                return {qcoOp.getQubit0Out(), qcoOp.getQubit1Out()};
              });
          return invOp.getQubitsOut();
        });
    llvm::SmallVector<Value> results;
    results.append(ctrlOp.getTargetsOut().begin(),
                   ctrlOp.getTargetsOut().end());
    results.append(ctrlOp.getControlsOut().begin(),
                   ctrlOp.getControlsOut().end());
    rewriter.replaceOp(op, results);
  } else if (op.getNumCtrls() != 0 && !op.getIsAdjoint()) {
    auto ctrlOp = rewriter.create<qco::CtrlOp>(
        op.getLoc(), adaptor.getInCtrlQubits(), adaptor.getInQubits(),
        [&](ValueRange targets) -> llvm::SmallVector<Value> {
          auto qcoOp = rewriter.create<QCOOpType>(op.getLoc(), targets[0],
                                                  targets[1], op.getRotation());
          return {qcoOp.getQubit0Out(), qcoOp.getQubit1Out()};
        });
    llvm::SmallVector<Value> results;
    results.append(ctrlOp.getTargetsOut().begin(),
                   ctrlOp.getTargetsOut().end());
    results.append(ctrlOp.getControlsOut().begin(),
                   ctrlOp.getControlsOut().end());
    rewriter.replaceOp(op, results);
  } else if (op.getNumCtrls() == 0 && op.getIsAdjoint()) {
    auto invOp = rewriter.create<qco::InvOp>(
        op.getLoc(), adaptor.getInQubits(),
        [&](ValueRange targets) -> llvm::SmallVector<Value> {
          auto qcoOp = rewriter.create<QCOOpType>(op.getLoc(), targets[0],
                                                  targets[1], op.getRotation());
          return {qcoOp.getQubit0Out(), qcoOp.getQubit1Out()};
        });
    rewriter.replaceOp(op, invOp.getQubitsOut());
  } else {
    rewriter.replaceOpWithNewOp<QCOOpType>(op, adaptor.getInQubits()[0],
                                           adaptor.getInQubits()[1],
                                           op.getRotation());
  }
}

template <typename QCOOpType>
static void createTwoTargetTwoParameter(jeff::CustomOp& op,
                                        jeff::CustomOpAdaptor& adaptor,
                                        ConversionPatternRewriter& rewriter) {
  auto loc = op.getLoc();
  if (op.getNumCtrls() != 0 && op.getIsAdjoint()) {
    auto ctrlOp = rewriter.create<qco::CtrlOp>(
        loc, adaptor.getInCtrlQubits(), adaptor.getInTargetQubits(),
        [&](ValueRange ctrlTargets) -> llvm::SmallVector<Value> {
          auto invOp = rewriter.create<qco::InvOp>(
              loc, ctrlTargets,
              [&](ValueRange invTargets) -> llvm::SmallVector<Value> {
                auto qcoOp = rewriter.create<QCOOpType>(
                    loc, invTargets[0], invTargets[1], op.getParams()[0],
                    op.getParams()[1]);
                return {qcoOp.getQubit0Out(), qcoOp.getQubit1Out()};
              });
          return invOp.getQubitsOut();
        });
    llvm::SmallVector<Value> results;
    results.append(ctrlOp.getTargetsOut().begin(),
                   ctrlOp.getTargetsOut().end());
    results.append(ctrlOp.getControlsOut().begin(),
                   ctrlOp.getControlsOut().end());
    rewriter.replaceOp(op, results);
  } else if (op.getNumCtrls() != 0 && !op.getIsAdjoint()) {
    auto ctrlOp = rewriter.create<qco::CtrlOp>(
        op.getLoc(), adaptor.getInCtrlQubits(), adaptor.getInTargetQubits(),
        [&](ValueRange targets) -> llvm::SmallVector<Value> {
          auto qcoOp =
              rewriter.create<QCOOpType>(op.getLoc(), targets[0], targets[1],
                                         op.getParams()[0], op.getParams()[1]);
          return {qcoOp.getQubit0Out(), qcoOp.getQubit1Out()};
        });
    llvm::SmallVector<Value> results;
    results.append(ctrlOp.getTargetsOut().begin(),
                   ctrlOp.getTargetsOut().end());
    results.append(ctrlOp.getControlsOut().begin(),
                   ctrlOp.getControlsOut().end());
    rewriter.replaceOp(op, results);
  } else if (op.getNumCtrls() == 0 && op.getIsAdjoint()) {
    auto invOp = rewriter.create<qco::InvOp>(
        op.getLoc(), adaptor.getInTargetQubits(),
        [&](ValueRange targets) -> llvm::SmallVector<Value> {
          auto qcoOp =
              rewriter.create<QCOOpType>(op.getLoc(), targets[0], targets[1],
                                         op.getParams()[0], op.getParams()[1]);
          return {qcoOp.getQubit0Out(), qcoOp.getQubit1Out()};
        });
    rewriter.replaceOp(op, invOp.getQubitsOut());
  } else {
    rewriter.replaceOpWithNewOp<QCOOpType>(
        op, adaptor.getInTargetQubits()[0], adaptor.getInTargetQubits()[1],
        op.getParams()[0], op.getParams()[1]);
  }
}

static void createBarrierOp(jeff::CustomOp& op, jeff::CustomOpAdaptor& adaptor,
                            ConversionPatternRewriter& rewriter) {
  rewriter.replaceOpWithNewOp<qco::BarrierOp>(op, adaptor.getInTargetQubits());
}

struct ConvertJeffCustomOpToQCO final : OpConversionPattern<jeff::CustomOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::CustomOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    if (op.getPower() != 1) {
      return rewriter.notifyMatchFailure(
          op, "Operations with power != 1 are not yet supported");
    }

    auto name = op.getName();
    if (name == "sx") {
      createOneTargetZeroParameter<qco::SXOp>(op, adaptor, rewriter);
    } else if (name == "r") {
      createOneTargetTwoParameter<qco::ROp>(op, adaptor, rewriter);
    } else if (name == "iswap") {
      createTwoTargetZeroParameter<qco::iSWAPOp>(op, adaptor, rewriter);
    } else if (name == "dcx") {
      createTwoTargetZeroParameter<qco::DCXOp>(op, adaptor, rewriter);
    } else if (name == "ecr") {
      createTwoTargetZeroParameter<qco::ECROp>(op, adaptor, rewriter);
    } else if (name == "xx_minus_yy") {
      createTwoTargetTwoParameter<qco::XXMinusYYOp>(op, adaptor, rewriter);
    } else if (name == "xx_plus_yy") {
      createTwoTargetTwoParameter<qco::XXPlusYYOp>(op, adaptor, rewriter);
    } else if (name == "barrier") {
      if (op.getNumCtrls() != 0) {
        return rewriter.notifyMatchFailure(
            op, "Barrier operations with controls are not supported");
      }
      createBarrierOp(op, adaptor, rewriter);
    } else {
      return rewriter.notifyMatchFailure(op, "Unsupported custom operation: " +
                                                 name);
    }

    return success();
  }
};

struct ConvertJeffPPROpToQCO final : OpConversionPattern<jeff::PPROp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::PPROp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    if (op.getPower() != 1) {
      return rewriter.notifyMatchFailure(
          op, "Operations with power != 1 are not yet supported");
    }

    auto pauliGates = op.getPauliGates();
    if (pauliGates.size() != 2) {
      return rewriter.notifyMatchFailure(
          op, "Only PPR operations with exactly 2 Pauli gates are supported");
    } else if (pauliGates[0] == 1 && pauliGates[1] == 1) {
      createTwoTargetOneParameter<qco::RXXOp>(op, adaptor, rewriter);
    } else if (pauliGates[0] == 2 && pauliGates[1] == 2) {
      createTwoTargetOneParameter<qco::RYYOp>(op, adaptor, rewriter);
    } else if (pauliGates[0] == 3 && pauliGates[1] == 1) {
      createTwoTargetOneParameter<qco::RZXOp>(op, adaptor, rewriter);
    } else if (pauliGates[0] == 3 && pauliGates[1] == 3) {
      createTwoTargetOneParameter<qco::RZZOp>(op, adaptor, rewriter);
    } else {
      return rewriter.notifyMatchFailure(op, "Unsupported PPR operation");
    }

    return success();
  }
};

/**
 * @brief Type converter for Jeff-to-QCO conversion
 *
 * @details
 * Converts `!jeff.qubit` to `!qco.qubit`.
 */
class JeffToQCOTypeConverter final : public TypeConverter {
public:
  explicit JeffToQCOTypeConverter(MLIRContext* ctx) {
    // Identity conversion for all types by default
    addConversion([](Type type) { return type; });

    addConversion([ctx](jeff::QubitType /*type*/) -> Type {
      return qco::QubitType::get(ctx);
    });
  }
};

struct ConvertJeffFloatConst64OpToArith final
    : OpConversionPattern<jeff::FloatConst64Op> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::FloatConst64Op op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto floatType = rewriter.getF64Type();
    auto floatAttr = rewriter.getF64FloatAttr(op.getVal().convertToDouble());
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, floatType, floatAttr);
    return success();
  }
};

/**
 * @brief Pass for converting Jeff operations to QCO operations
 *
 * @details
 * TODO
 */
struct JeffToQCO final : impl::JeffToQCOBase<JeffToQCO> {
  using JeffToQCOBase::JeffToQCOBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    const JeffToQCOTypeConverter typeConverter(context);

    // Configure conversion target: Jeff illegal, QCO legal
    target.addIllegalDialect<jeff::JeffDialect>();
    target.addLegalDialect<QCODialect>();
    target.addLegalDialect<arith::ArithDialect>();

    // Register operation conversion patterns
    patterns.add<
        ConvertJeffQubitAllocOpToQCO, ConvertJeffQubitFreeOpToQCO,
        ConvertJeffQubitFreeZeroOpToQCO, ConvertJeffQubitMeasureOpToQCO,
        ConvertJeffQubitMeasureNDOpToQCO, ConvertJeffQubitResetOpToQCO,
        ConvertJeffGPhaseOpToQCO, ConvertJeffIOpToQCO, ConvertJeffXOpToQCO,
        ConvertJeffYOpToQCO, ConvertJeffZOpToQCO, ConvertJeffHOpToQCO,
        ConvertJeffSOpToQCO, ConvertJeffTOpToQCO, ConvertJeffRxOpToQCO,
        ConvertJeffRyOpToQCO, ConvertJeffRzOpToQCO, ConvertJeffR1OpToQCO,
        ConvertJeffUOpToQCO, ConvertJeffSwapOpToQCO, ConvertJeffCustomOpToQCO,
        ConvertJeffPPROpToQCO, ConvertJeffFloatConst64OpToArith>(typeConverter,
                                                                 context);

    // Apply the conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir
