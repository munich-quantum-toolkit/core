/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/JeffToQC/JeffToQC.h"

#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCOps.h"

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
using namespace qc;

#define GEN_PASS_DEF_JEFFTOQC
#include "mlir/Conversion/JeffToQC/JeffToQC.h.inc"

/**
 * @brief Converts a one-target, zero-parameter Jeff operation to QC
 *
 * @tparam QCOpType The operation type of the QC operation
 * @tparam JeffOpType The operation type of the Jeff operation
 * @tparam JeffOpAdaptorType The OpAdaptor type of the Jeff operation
 * @param op The Jeff operation instance to convert
 * @param adaptor The OpAdaptor of the Jeff operation
 * @param rewriter The pattern rewriter
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QCOpType, typename JeffOpType, typename JeffOpAdaptorType>
static LogicalResult
convertOneTargetZeroParameter(JeffOpType& op, JeffOpAdaptorType& adaptor,
                              ConversionPatternRewriter& rewriter) {
  if (op.getPower() != 1) {
    return rewriter.notifyMatchFailure(
        op, "Operations with power != 1 are not yet supported");
  }

  auto target = adaptor.getInQubit();

  if (op.getNumCtrls() != 0) {
    auto controls = adaptor.getInCtrlQubits();
    rewriter.create<qc::CtrlOp>(op.getLoc(), controls, [&] {
      rewriter.create<QCOpType>(op.getLoc(), target);
    });
    SmallVector<Value> operands;
    operands.reserve(1 + controls.size());
    operands.push_back(target);
    operands.append(controls.begin(), controls.end());
    rewriter.replaceOp(op, operands);
  } else {
    rewriter.create<QCOpType>(op.getLoc(), target);
    rewriter.replaceOp(op, target);
  }

  return success();
}

/**
 * @brief Converts a one-target, one-parameter Jeff operation to QC
 *
 * @tparam QCOpType The operation type of the QC operation
 * @tparam JeffOpType The operation type of the Jeff operation
 * @tparam JeffOpAdaptorType The OpAdaptor type of the Jeff operation
 * @param op The Jeff operation instance to convert
 * @param adaptor The OpAdaptor of the Jeff operation
 * @param rewriter The pattern rewriter
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QCOpType, typename JeffOpType, typename JeffOpAdaptorType>
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

  auto target = adaptor.getInQubit();

  if (op.getNumCtrls() != 0) {
    auto controls = adaptor.getInCtrlQubits();
    rewriter.create<qc::CtrlOp>(op.getLoc(), controls, [&] {
      rewriter.create<QCOpType>(op.getLoc(), target, op.getRotation());
    });
    SmallVector<Value> operands;
    operands.reserve(1 + controls.size());
    operands.push_back(target);
    operands.append(controls.begin(), controls.end());
    rewriter.replaceOp(op, operands);
  } else {
    rewriter.create<QCOpType>(op.getLoc(), target, op.getRotation());
    rewriter.replaceOp(op, target);
  }

  return success();
}

/**
 * @brief Converts jeff.qubit_alloc to qc.alloc
 *
 * @par Example:
 * ```mlir
 * %q = jeff.qubit_alloc : !jeff.qubit
 * ```
 * is converted to
 * ```mlir
 * %q = qc.alloc : !qc.qubit
 * ```
 */
struct ConvertJeffQubitAllocOpToQC final
    : OpConversionPattern<jeff::QubitAllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::QubitAllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<qc::AllocOp>(op);
    return success();
  }
};

/**
 * @brief Converts jeff.qubit_free to qc.dealloc
 *
 * @par Example:
 * ```mlir
 * jeff.qubit_free %q : !jeff.qubit
 * ```
 * is converted to
 * ```mlir
 * qc.dealloc %q : !qc.qubit
 * ```
 */
struct ConvertJeffQubitFreeOpToQC final
    : OpConversionPattern<jeff::QubitFreeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::QubitFreeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto qcQubit = adaptor.getInQubit();
    rewriter.replaceOpWithNewOp<qc::DeallocOp>(op, qcQubit);
    return success();
  }
};

/**
 * @brief Converts jeff.qubit_measure_nd to qc.measure
 *
 * @par Example:
 * ```mlir
 * %result, %q_out = jeff.qubit_measure_nd %q_in : i1, !jeff.qubit
 * ```
 * is converted to
 * ```mlir
 * %result = qc.measure %q : i1
 * ```
 */
struct ConvertJeffQubitMeasureNDOpToQC final
    : OpConversionPattern<jeff::QubitMeasureNDOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::QubitMeasureNDOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto qcQubit = adaptor.getInQubit();
    auto qcOp = rewriter.create<qc::MeasureOp>(op.getLoc(), qcQubit);
    rewriter.replaceOp(op, {qcOp.getResult(), qcQubit});
    return success();
  }
};

/**
 * @brief Converts jeff.reset to qc.qubit_reset
 *
 * @par Example:
 * ```mlir
 * %q_out = jeff.qubit_reset %q_in : !jeff.qubit
 * ```
 * is converted to
 * ```mlir
 * %result = qc.reset %q : !qc.qubit
 * ```
 */
struct ConvertJeffQubitResetOpToQC final
    : OpConversionPattern<jeff::QubitResetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::QubitResetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto qcQubit = adaptor.getInQubit();
    rewriter.create<qc::ResetOp>(op.getLoc(), qcQubit);
    rewriter.replaceOp(op, qcQubit);
    return success();
  }
};

/**
 * @brief Converts jeff.gphase to qc.gphase
 *
 * @par Example:
 * ```mlir
 * jeff.gphase(%theta) {is_adjoint = false, num_ctrls = 0 : i8, power = 1 : i8}
 * ```
 * is converted to
 * ```mlir
 * qc.gphase(%theta)
 * ```
 */
struct ConvertJeffGPhaseOpToQC final : OpConversionPattern<jeff::GPhaseOp> {
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
      auto controls = adaptor.getInCtrlQubits();
      rewriter.create<qc::CtrlOp>(op.getLoc(), controls, [&] {
        rewriter.create<qc::GPhaseOp>(op.getLoc(), op.getRotation());
      });
      rewriter.replaceOp(op, controls);
    } else {
      rewriter.create<qc::GPhaseOp>(op.getLoc(), op.getRotation());
      rewriter.eraseOp(op);
    }

    return success();
  }
};

// OneTargetZeroParameter

#define DEFINE_ONE_TARGET_ZERO_PARAMETER(OP_CLASS_JEFF, OP_CLASS_QC,           \
                                         OP_CLASS_QC_ADJOINT)                  \
  struct ConvertJeff##OP_CLASS_JEFF##ToQC final                                \
      : OpConversionPattern<jeff::OP_CLASS_JEFF> {                             \
    using OpConversionPattern::OpConversionPattern;                            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(jeff::OP_CLASS_JEFF op, OpAdaptor adaptor,                 \
                    ConversionPatternRewriter& rewriter) const override {      \
      if (op.getIsAdjoint()) {                                                 \
        return convertOneTargetZeroParameter<qc::OP_CLASS_QC_ADJOINT>(         \
            op, adaptor, rewriter);                                            \
      }                                                                        \
      return convertOneTargetZeroParameter<qc::OP_CLASS_QC>(op, adaptor,       \
                                                            rewriter);         \
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

#define DEFINE_ONE_TARGET_ONE_PARAMETER(OP_CLASS_JEFF, OP_CLASS_QC)            \
  struct ConvertJeff##OP_CLASS_JEFF##ToQC final                                \
      : OpConversionPattern<jeff::OP_CLASS_JEFF> {                             \
    using OpConversionPattern::OpConversionPattern;                            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(jeff::OP_CLASS_JEFF op, OpAdaptor adaptor,                 \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertOneTargetOneParameter<qc::OP_CLASS_QC>(op, adaptor,        \
                                                           rewriter);          \
    }                                                                          \
  };

DEFINE_ONE_TARGET_ONE_PARAMETER(RxOp, RXOp)
DEFINE_ONE_TARGET_ONE_PARAMETER(RyOp, RYOp)
DEFINE_ONE_TARGET_ONE_PARAMETER(RzOp, RZOp)
DEFINE_ONE_TARGET_ONE_PARAMETER(R1Op, POp)

#undef DEFINE_ONE_TARGET_ONE_PARAMETER

/**
 * @brief Converts jeff.u to qc.u
 *
 * @par Example:
 * ```mlir
 * %q_out = jeff.u(%theta, %phi, %lambda) {is_adjoint = false, num_ctrls = 0 :
 * i8, power = 1 : i8} %q_in : !jeff.qubit
 * ```
 * is converted to
 * ```mlir
 * qc.u(%theta, %phi, %lambda) %q : !qc.qubit
 * ```
 */
struct ConvertJeffUOpToQC final : OpConversionPattern<jeff::UOp> {
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

    auto target = adaptor.getInQubit();

    if (op.getNumCtrls() != 0) {
      auto controls = adaptor.getInCtrlQubits();
      rewriter.create<qc::CtrlOp>(op.getLoc(), controls, [&] {
        rewriter.create<qc::UOp>(op.getLoc(), target, op.getTheta(),
                                 op.getPhi(), op.getLambda());
      });
      SmallVector<Value> operands;
      operands.reserve(1 + controls.size());
      operands.push_back(target);
      operands.append(controls.begin(), controls.end());
      rewriter.replaceOp(op, operands);
    } else {
      rewriter.create<qc::UOp>(op.getLoc(), target, op.getTheta(), op.getPhi(),
                               op.getLambda());
      rewriter.replaceOp(op, target);
    }

    return success();
  }
};

/**
 * @brief Converts jeff.swap to qc.swap
 *
 * @par Example:
 * ```mlir
 * %q0_out, %q1_out = jeff.swap {is_adjoint = false, num_ctrls = 0 : i8, power =
 * 1 : i8} %q0_in %q1_in : !jeff.qubit !jeff.qubit
 * ```
 * is converted to
 * ```mlir
 * qc.swap %q0, %q1 : !qc.qubit, !qc.qubit
 * ```
 */
struct ConvertJeffSwapOpToQC final : OpConversionPattern<jeff::SwapOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::SwapOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    if (op.getPower() != 1) {
      return rewriter.notifyMatchFailure(
          op, "Operations with power != 1 are not yet supported");
    }

    auto target1 = adaptor.getInQubitOne();
    auto target2 = adaptor.getInQubitTwo();

    if (op.getNumCtrls() != 0) {
      auto controls = adaptor.getInCtrlQubits();
      rewriter.create<qc::CtrlOp>(op.getLoc(), controls, [&] {
        rewriter.create<qc::SWAPOp>(op.getLoc(), target1, target2);
      });
      SmallVector<Value> operands;
      operands.reserve(2 + controls.size());
      operands.push_back(target1);
      operands.push_back(target2);
      operands.append(controls.begin(), controls.end());
      rewriter.replaceOp(op, operands);
    } else {
      rewriter.create<qc::SWAPOp>(op.getLoc(), target1, target2);
      rewriter.replaceOp(op, {target1, target2});
    }

    return success();
  }
};

/**
 * @brief Type converter for Jeff-to-QC conversion
 *
 * @details
 * Converts `!jeff.qubit` to `!qc.qubit`.
 */
class JeffToQCTypeConverter final : public TypeConverter {
public:
  explicit JeffToQCTypeConverter(MLIRContext* ctx) {
    // Identity conversion for all types by default
    addConversion([](Type type) { return type; });

    addConversion([ctx](jeff::QubitType /*type*/) -> Type {
      return qc::QubitType::get(ctx);
    });
  }
};

/**
 * @brief Pass for converting Jeff operations to QC operations
 *
 * @details
 * TODO
 */
struct JeffToQC final : impl::JeffToQCBase<JeffToQC> {
  using JeffToQCBase::JeffToQCBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    const JeffToQCTypeConverter typeConverter(context);

    // Configure conversion target: Jeff illegal, QC legal
    target.addIllegalDialect<jeff::JeffDialect>();
    target.addLegalDialect<QCDialect>();

    // Register operation conversion patterns
    patterns
        .add<ConvertJeffQubitAllocOpToQC, ConvertJeffQubitFreeOpToQC,
             ConvertJeffQubitMeasureNDOpToQC, ConvertJeffQubitResetOpToQC,
             ConvertJeffGPhaseOpToQC, ConvertJeffIOpToQC, ConvertJeffXOpToQC,
             ConvertJeffYOpToQC, ConvertJeffZOpToQC, ConvertJeffHOpToQC,
             ConvertJeffSOpToQC, ConvertJeffTOpToQC, ConvertJeffRxOpToQC,
             ConvertJeffRyOpToQC, ConvertJeffRzOpToQC, ConvertJeffR1OpToQC,
             ConvertJeffUOpToQC, ConvertJeffSwapOpToQC>(typeConverter, context);

    // Apply the conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir
