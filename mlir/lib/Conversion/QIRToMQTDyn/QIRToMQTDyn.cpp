/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/QIRToMQTDyn/QIRToMQTDyn.h"

#include "mlir/Dialect/MQTDyn/IR/MQTDynDialect.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/LLVMIR/FunctionCallUtils.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mqt::ir {

using namespace mlir;

#define GEN_PASS_DEF_QIRTOMQTDYN
#include "mlir/Conversion/QIRToMQTDyn/QIRToMQTDyn.h.inc"

class QIRToMQTDynTypeConverter final : public TypeConverter {
public:
  explicit QIRToMQTDynTypeConverter(MLIRContext* ctx) {
    // Identity conversion
    addConversion([](Type type) { return type; });
    // Pointer conversion
    addConversion([ctx](LLVM::LLVMPointerType type) -> Type {
      if (type.getAddressSpace() == 1) {
        return dyn::QubitType::get(ctx);
      }
      return dyn::QubitRegisterType::get(ctx);
    });
  }
};

struct ConvertQIRLoad final : OpConversionPattern<LLVM::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {

    // erase the operation and use the operands as results
    rewriter.replaceOp(op, adaptor.getOperands());
    return success();
  }
};
struct ConvertQIRCall final : OpConversionPattern<LLVM::CallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto fnName = op.getCallee();
    auto qubitType = dyn::QubitType::get(rewriter.getContext());
    auto qregType = dyn::QubitRegisterType::get(rewriter.getContext());
    auto operands = adaptor.getOperands();
    if (fnName == "__quantum__rt__qubit_allocate_array") {
      auto alloc =
          rewriter.create<dyn::AllocOp>(op.getLoc(), qregType, operands);
      rewriter.replaceOp(op, alloc->getResults());
    } else if (fnName == "__catalyst__rt__array_get_element_ptr_1d") {
      rewriter.replaceOpWithNewOp<dyn::ExtractOp>(op, qubitType,
                                                  adaptor.getOperands());

    } else if (fnName == "__quantum__qis__h__body") {
      rewriter.replaceOpWithNewOp<dyn::HOp>(
          op, DenseF64ArrayAttr{}, DenseBoolArrayAttr{}, ValueRange{},
          adaptor.getOperands(), ValueRange{}, ValueRange{});
    } else if (fnName == "__quantum__qis__cnot__body") {
      rewriter.replaceOpWithNewOp<dyn::HOp>(
          op, DenseF64ArrayAttr{}, DenseBoolArrayAttr{}, ValueRange{},
          adaptor.getOperands().front(), adaptor.getOperands().back(),
          ValueRange{});
    } else if (fnName == "__quantum__qis__m__body") {
      SmallVector<Type> newBits(adaptor.getOperands().size(),
                                IntegerType::get(rewriter.getContext(), 1));
      rewriter.replaceOpWithNewOp<dyn::MeasureOp>(op, newBits,
                                                  adaptor.getOperands());
    } else if (fnName == "__quantum__rt__qubit_release_array") {
      rewriter.replaceOpWithNewOp<dyn::DeallocOp>(
          op, adaptor.getOperands().front());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

struct ConvertQIRFunc final : OpConversionPattern<LLVM::LLVMFuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::LLVMFuncOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    // erase the operation
    rewriter.eraseOp(op);
    return success();
  }
};

struct QIRToMQTDyn final : impl::QIRToMQTDynBase<QIRToMQTDyn> {
  using QIRToMQTDynBase::QIRToMQTDynBase;
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    QIRToMQTDynTypeConverter typeConverter(context);
    target.addLegalDialect<dyn::MQTDynDialect>();
    target.addIllegalDialect<LLVM::LLVMDialect>();
    target.addLegalOp<LLVM::ConstantOp>();
    patterns.add<ConvertQIRFunc>(typeConverter, context);
    patterns.add<ConvertQIRLoad>(typeConverter, context);
    patterns.add<ConvertQIRCall>(typeConverter, context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  };
};

} // namespace mqt::ir
