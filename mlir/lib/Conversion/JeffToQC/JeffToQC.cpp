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

template <typename QCOpType, typename JeffOpType, typename JeffOpAdaptorType>
static LogicalResult
convertOneTargetOneParameter(JeffOpType& op, JeffOpAdaptorType& adaptor,
                             ConversionPatternRewriter& rewriter) {
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

struct ConvertJeffQubitMeasureNDOpToQC final
    : OpConversionPattern<jeff::QubitMeasureNDOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::QubitMeasureNDOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto qcQubit = adaptor.getInQubit();
    auto qcOp = rewriter.create<qc::MeasureOp>(op.getLoc(), qcQubit);
    SmallVector<Value> operands;
    operands.reserve(2);
    operands.push_back(qcOp.getResult());
    operands.push_back(qcQubit);
    rewriter.replaceOp(op, operands);
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

#define DEFINE_ONE_TARGET_ZERO_PARAMETER(OP_CLASS_JEFF, OP_CLASS_QC)           \
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

DEFINE_ONE_TARGET_ZERO_PARAMETER(RxOp, RXOp)
DEFINE_ONE_TARGET_ZERO_PARAMETER(RyOp, RYOp)
DEFINE_ONE_TARGET_ZERO_PARAMETER(RzOp, RZOp)
DEFINE_ONE_TARGET_ZERO_PARAMETER(R1Op, POp)

#undef DEFINE_ONE_TARGET_ZERO_PARAMETER

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
    patterns.add<ConvertJeffQubitAllocOpToQC, ConvertJeffQubitFreeOpToQC,
                 ConvertJeffQubitMeasureNDOpToQC, ConvertJeffIOpToQC,
                 ConvertJeffXOpToQC, ConvertJeffYOpToQC, ConvertJeffZOpToQC,
                 ConvertJeffHOpToQC, ConvertJeffSOpToQC, ConvertJeffTOpToQC,
                 ConvertJeffRxOpToQC, ConvertJeffRyOpToQC, ConvertJeffRzOpToQC,
                 ConvertJeffR1OpToQC>(typeConverter, context);

    // Apply the conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir
