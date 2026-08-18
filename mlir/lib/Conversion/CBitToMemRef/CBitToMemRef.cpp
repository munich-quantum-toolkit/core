/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/CBitToMemRef/CBitToMemRef.h"

#include "mlir/Dialect/CBit/IR/CBitOps.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/Transforms/Patterns.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir {

#define GEN_PASS_DEF_CONVERTCBITTOMEMREF
#include "mlir/Conversion/CBitToMemRef/CBitToMemRef.h.inc"

namespace {
class CBitTypeConverter final : public TypeConverter {
public:
  CBitTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](const cbit::RegisterType type) -> Type {
      return MemRefType::get({type.getWidth()},
                             IntegerType::get(type.getContext(), 1));
    });
  }
};

struct ConvertAllocOp final : OpConversionPattern<cbit::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cbit::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    const auto type = cast<MemRefType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    auto allocation = memref::AllocOp::create(rewriter, op.getLoc(), type);

    if (op.getInitialization() == cbit::Initialization::Zero) {
      auto zero = arith::ConstantOp::create(rewriter, op.getLoc(),
                                            rewriter.getBoolAttr(false));
      for (int64_t index = 0; index < type.getDimSize(0); ++index) {
        auto indexValue =
            arith::ConstantIndexOp::create(rewriter, op.getLoc(), index);
        memref::StoreOp::create(rewriter, op.getLoc(), zero.getResult(),
                                allocation.getResult(),
                                ValueRange{indexValue.getResult()});
      }
    }

    rewriter.replaceOp(op, allocation.getResult());
    return success();
  }
};

struct ConvertLoadOp final : OpConversionPattern<cbit::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cbit::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, adaptor.getReg(),
                                                adaptor.getIndex());
    return success();
  }
};

struct ConvertStoreOp final : OpConversionPattern<cbit::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cbit::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    memref::StoreOp::create(rewriter, op.getLoc(), adaptor.getValue(),
                            adaptor.getReg(), adaptor.getIndex());
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertCBitToMemRef final
    : impl::ConvertCBitToMemRefBase<ConvertCBitToMemRef> {
  using ConvertCBitToMemRefBase::ConvertCBitToMemRefBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* moduleOp = getOperation();
    CBitTypeConverter typeConverter;
    ConversionTarget target(*context);
    RewritePatternSet patterns(context);

    target.addIllegalDialect<cbit::CBitDialect>();
    target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect>();
    target.markUnknownOpDynamicallyLegal(
        [&](Operation* op) { return typeConverter.isLegal(op); });
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    target.addDynamicallyLegalOp<func::ReturnOp, func::CallOp>(
        [&](Operation* op) { return typeConverter.isLegal(op); });

    patterns.add<ConvertAllocOp, ConvertLoadOp, ConvertStoreOp>(typeConverter,
                                                                context);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
    scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter,
                                                         patterns, target);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir
