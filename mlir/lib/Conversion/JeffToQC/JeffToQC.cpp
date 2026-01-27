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

struct ConvertJeffXOpToQC final : OpConversionPattern<jeff::XOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(jeff::XOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto qcQubit = adaptor.getInQubit();
    rewriter.create<qc::XOp>(op.getLoc(), qcQubit);
    rewriter.replaceOp(op, qcQubit);
    return success();
  }
};

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
    JeffToQCTypeConverter typeConverter(context);

    // Configure conversion target: Jeff illegal, QC legal
    target.addIllegalDialect<jeff::JeffDialect>();
    target.addLegalDialect<QCDialect>();

    // Register operation conversion patterns
    patterns.add<ConvertJeffQubitAllocOpToQC, ConvertJeffQubitFreeOpToQC,
                 ConvertJeffXOpToQC>(typeConverter, context);

    // Apply the conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir
