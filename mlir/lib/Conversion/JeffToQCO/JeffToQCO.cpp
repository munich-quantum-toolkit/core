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

  if (op.getNumCtrls() != 0) {
    auto controls = adaptor.getInCtrlQubits();
    auto ctrlOp = rewriter.create<qco::CtrlOp>(
        op.getLoc(), adaptor.getInCtrlQubits(), adaptor.getInQubit(),
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
  if (op.getIsAdjoint()) {
    return rewriter.notifyMatchFailure(
        op, "Adjoint operations are not yet supported");
  }

  if (op.getPower() != 1) {
    return rewriter.notifyMatchFailure(
        op, "Operations with power != 1 are not yet supported");
  }

  if (op.getNumCtrls() != 0) {
    auto ctrlOp = rewriter.create<qco::CtrlOp>(
        op.getLoc(), adaptor.getInCtrlQubits(), adaptor.getInQubit(),
        [&](ValueRange targets) -> llvm::SmallVector<Value> {
          auto qcoOp = rewriter.create<QCOOpType>(op.getLoc(), targets[0],
                                                  op.getRotation());
          return {qcoOp.getQubitOut()};
        });
    llvm::SmallVector<Value> results;
    results.append(ctrlOp.getTargetsOut().begin(),
                   ctrlOp.getTargetsOut().end());
    results.append(ctrlOp.getControlsOut().begin(),
                   ctrlOp.getControlsOut().end());
    rewriter.replaceOp(op, results);
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
    if (op.getIsAdjoint()) {
      return rewriter.notifyMatchFailure(
          op, "Adjoint operations are not yet supported");
    }

    if (op.getPower() != 1) {
      return rewriter.notifyMatchFailure(
          op, "Operations with power != 1 are not yet supported");
    }

    if (op.getNumCtrls() != 0) {
      auto ctrlOp = rewriter.create<qco::CtrlOp>(
          op.getLoc(), adaptor.getInCtrlQubits(), ValueRange{},
          [&](ValueRange targets) -> llvm::SmallVector<Value> {
            rewriter.create<qco::GPhaseOp>(op.getLoc(), op.getRotation());
            return {};
          });
      llvm::SmallVector<Value> results;
      results.append(ctrlOp.getControlsOut().begin(),
                     ctrlOp.getControlsOut().end());
      rewriter.replaceOp(op, results);
    } else {
      rewriter.replaceOpWithNewOp<qco::GPhaseOp>(op, op.getRotation());
    }

    return success();
  }
};

// OneTargetZeroParameter

#define DEFINE_ONE_TARGET_ZERO_PARAMETER(OP_CLASS_JEFF, OP_CLASS_QCO,          \
                                         OP_CLASS_QCO_ADJOINT)                 \
  struct ConvertJeff##OP_CLASS_JEFF##ToQCO final                               \
      : OpConversionPattern<jeff::OP_CLASS_JEFF> {                             \
    using OpConversionPattern::OpConversionPattern;                            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(jeff::OP_CLASS_JEFF op, OpAdaptor adaptor,                 \
                    ConversionPatternRewriter& rewriter) const override {      \
      if (op.getIsAdjoint()) {                                                 \
        return convertOneTargetZeroParameter<qco::OP_CLASS_QCO_ADJOINT>(       \
            op, adaptor, rewriter);                                            \
      }                                                                        \
      return convertOneTargetZeroParameter<qco::OP_CLASS_QCO>(op, adaptor,     \
                                                              rewriter);       \
    }                                                                          \
  };

DEFINE_ONE_TARGET_ZERO_PARAMETER(IOp, IdOp, IdOp)
DEFINE_ONE_TARGET_ZERO_PARAMETER(XOp, XOp, XOp)
DEFINE_ONE_TARGET_ZERO_PARAMETER(YOp, YOp, YOp)
DEFINE_ONE_TARGET_ZERO_PARAMETER(ZOp, ZOp, ZOp)
DEFINE_ONE_TARGET_ZERO_PARAMETER(HOp, HOp, HOp)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SOp, SOp, SdgOp)
DEFINE_ONE_TARGET_ZERO_PARAMETER(TOp, TOp, TdgOp)

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
    if (op.getIsAdjoint()) {
      return rewriter.notifyMatchFailure(
          op, "Adjoint operations are not yet supported");
    }

    if (op.getPower() != 1) {
      return rewriter.notifyMatchFailure(
          op, "Operations with power != 1 are not yet supported");
    }

    if (op.getNumCtrls() != 0) {
      auto ctrlOp = rewriter.create<qco::CtrlOp>(
          op.getLoc(), adaptor.getInCtrlQubits(), adaptor.getInQubit(),
          [&](ValueRange targets) -> llvm::SmallVector<Value> {
            auto qcoOp = rewriter.create<qco::UOp>(op.getLoc(), targets[0],
                                                   op.getTheta(), op.getPhi(),
                                                   op.getLambda());
            return {qcoOp.getQubitOut()};
          });
      llvm::SmallVector<Value> results;
      results.append(ctrlOp.getTargetsOut().begin(),
                     ctrlOp.getTargetsOut().end());
      results.append(ctrlOp.getControlsOut().begin(),
                     ctrlOp.getControlsOut().end());
      rewriter.replaceOp(op, results);
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

    if (op.getNumCtrls() != 0) {
      auto ctrlOp = rewriter.create<qco::CtrlOp>(
          op.getLoc(), adaptor.getInCtrlQubits(),
          ValueRange{adaptor.getInQubitOne(), adaptor.getInQubitTwo()},
          [&](ValueRange targets) -> llvm::SmallVector<Value> {
            auto qcoOp = rewriter.create<qco::SWAPOp>(op.getLoc(), targets[0],
                                                      targets[1]);
            return {qcoOp.getQubit0Out(), qcoOp.getQubit1Out()};
          });
      llvm::SmallVector<Value> results;
      results.append(ctrlOp.getTargetsOut().begin(),
                     ctrlOp.getTargetsOut().end());
      results.append(ctrlOp.getControlsOut().begin(),
                     ctrlOp.getControlsOut().end());
      rewriter.replaceOp(op, results);
    } else {
      rewriter.replaceOpWithNewOp<qco::SWAPOp>(op, adaptor.getInQubitOne(),
                                               adaptor.getInQubitTwo());
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

    // Register operation conversion patterns
    patterns
        .add<ConvertJeffQubitAllocOpToQCO, ConvertJeffQubitFreeOpToQCO,
             ConvertJeffQubitFreeZeroOpToQCO, ConvertJeffQubitMeasureOpToQCO,
             ConvertJeffQubitMeasureNDOpToQCO, ConvertJeffQubitResetOpToQCO,
             ConvertJeffGPhaseOpToQCO, ConvertJeffIOpToQCO, ConvertJeffXOpToQCO,
             ConvertJeffYOpToQCO, ConvertJeffZOpToQCO, ConvertJeffHOpToQCO,
             ConvertJeffSOpToQCO, ConvertJeffTOpToQCO, ConvertJeffRxOpToQCO,
             ConvertJeffRyOpToQCO, ConvertJeffRzOpToQCO, ConvertJeffR1OpToQCO,
             ConvertJeffUOpToQCO, ConvertJeffSwapOpToQCO>(typeConverter,
                                                          context);

    // Apply the conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir
