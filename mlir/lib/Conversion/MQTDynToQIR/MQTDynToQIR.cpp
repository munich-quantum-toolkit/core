/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/MQTDynToQIR/MQTDynToQIR.h"

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

#define GEN_PASS_DEF_MQTDYNTOQIR
#include "mlir/Conversion/MQTDynToQIR/MQTDynToQIR.h.inc"

class MQTDynToQIRTypeConverter final : public TypeConverter {
public:
  explicit MQTDynToQIRTypeConverter(MLIRContext* ctx) {
    // Identity conversion
    addConversion([](Type type) { return type; });
    // Qubit conversion
    addConversion([ctx](dyn::QubitType /*type*/) -> Type {
      assert(ctx->getLoadedDialect<mlir::LLVM::LLVMDialect>() &&
             "LLVM dialect must be loaded in context!");

      return LLVM::LLVMPointerType::get(ctx);
    });

    // QregType conversion
    addConversion([ctx](dyn::QubitRegisterType /*type*/) -> Type {
      assert(ctx->getLoadedDialect<mlir::LLVM::LLVMDialect>() &&
             "LLVM dialect must be loaded in context!");

      return LLVM::LLVMPointerType::get(ctx);
      ;
    });
  }
};
static LLVM::LLVMFuncOp ensureFunctionDeclaration(PatternRewriter& rewriter,
                                                  Operation* op,
                                                  StringRef fnSymbol,
                                                  Type fnType) {
  Operation* fnDecl = SymbolTable::lookupNearestSymbolFrom(
      op, rewriter.getStringAttr(fnSymbol));

  if (fnDecl == nullptr) {
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    auto mod = op->getParentOfType<ModuleOp>();
    rewriter.setInsertionPointToStart(mod.getBody());

    fnDecl = rewriter.create<LLVM::LLVMFuncOp>(op->getLoc(), fnSymbol, fnType);
  }

  return cast<LLVM::LLVMFuncOp>(fnDecl);
}

struct ConvertMQTDynAllocQIR final : OpConversionPattern<dyn::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(dyn::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    StringRef fnName = "__quantum__rt__qubit_allocate_array";
    const TypeConverter* conv = getTypeConverter();
    MLIRContext* ctx = getContext();
    Type qirSignature = LLVM::LLVMFunctionType::get(
        conv->convertType(dyn::QubitRegisterType::get(ctx)),
        IntegerType::get(ctx, 64));
    LLVM::LLVMFuncOp fnDecl =
        ensureFunctionDeclaration(rewriter, op, fnName, qirSignature);

    auto size = adaptor.getSize();
    if (!size) {
      size =
          rewriter.create<LLVM::ConstantOp>(op->getLoc(), op.getSizeAttrAttr());
    }
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, size);
    return success();
  }
};
struct ConvertMQTDynDeallocQIR final : OpConversionPattern<dyn::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(dyn::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {

    StringRef fnName = "__quantum__rt__qubit_release_array";
    const TypeConverter* conv = getTypeConverter();
    MLIRContext* ctx = getContext();
    Type qirSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx),
        conv->convertType(dyn::QubitRegisterType::get(ctx)));
    LLVM::LLVMFuncOp fnDecl =
        ensureFunctionDeclaration(rewriter, op, fnName, qirSignature);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl,
                                              adaptor.getOperands());
    return success();
  }
};

struct ConvertMQTDynExtractQIR final : OpConversionPattern<dyn::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(dyn::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {

    StringRef fnName = "__catalyst__rt__array_get_element_ptr_1d";
    const TypeConverter* conv = getTypeConverter();
    MLIRContext* ctx = getContext();
    Type qirSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMPointerType::get(ctx),
        {conv->convertType(dyn::QubitRegisterType::get(ctx)),
         IntegerType::get(ctx, 64)});

    LLVM::LLVMFuncOp fnDecl =
        ensureFunctionDeclaration(rewriter, op, fnName, qirSignature);

    auto index = adaptor.getIndex();
    if (!index) {
      index = rewriter.create<LLVM::ConstantOp>(op->getLoc(),
                                                op.getIndexAttrAttr());
    }

    SmallVector<Value> operands = {adaptor.getInQreg(), index};
    Value elemPtr = rewriter.create<LLVM::CallOp>(op.getLoc(), fnDecl, operands)
                        .getResult();
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
        op, conv->convertType(dyn::QubitType::get(ctx)), elemPtr);

    return success();
  }
};

template <typename MQTGateDynOp>
struct ConvertMQTDynGateOpQIR final : OpConversionPattern<MQTGateDynOp> {
  using OpConversionPattern<MQTGateDynOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MQTGateDynOp op, typename MQTGateDynOp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext* ctx = rewriter.getContext();
    const auto& dynInQubitsValues = adaptor.getInQubits();
    const auto& dynPosCtrlQubitsValues = adaptor.getPosCtrlInQubits();
    const auto& dynNegCtrlQubitsValues = adaptor.getNegCtrlInQubits();
    SmallVector<Type> argTypes;
    argTypes.insert(argTypes.end(), dynInQubitsValues.getTypes().begin(),
                    dynInQubitsValues.getTypes().end());
    argTypes.insert(argTypes.end(), dynPosCtrlQubitsValues.getTypes().begin(),
                    dynPosCtrlQubitsValues.getTypes().end());
    argTypes.insert(argTypes.end(), dynNegCtrlQubitsValues.getTypes().begin(),
                    dynNegCtrlQubitsValues.getTypes().end());

    SmallVector<Value> dynQubitsValues;
    dynQubitsValues.reserve(dynInQubitsValues.size() +
                            dynPosCtrlQubitsValues.size() +
                            dynNegCtrlQubitsValues.size());
    dynQubitsValues.append(dynInQubitsValues.begin(), dynInQubitsValues.end());
    dynQubitsValues.append(dynPosCtrlQubitsValues.begin(),
                           dynPosCtrlQubitsValues.end());
    dynQubitsValues.append(dynNegCtrlQubitsValues.begin(),
                           dynNegCtrlQubitsValues.end());

    StringRef name = op->getName().getStringRef().split('.').second;
    std::string fnName;
    if (dynPosCtrlQubitsValues.size() == 0 &&
        dynNegCtrlQubitsValues.size() == 0) {
      fnName = ("__quantum__qis__" + name + "__body").str();
    } else {
      fnName = "__quantum__qis__cnot__body";
    }
    Type qirSignature =
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), argTypes);

    LLVM::LLVMFuncOp fnDecl =
        ensureFunctionDeclaration(rewriter, op, fnName, qirSignature);
    rewriter.create<LLVM::CallOp>(loc, fnDecl, dynQubitsValues);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertMQTDynMeasureQIR final : OpConversionPattern<dyn::MeasureOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(dyn::MeasureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {

    MLIRContext* ctx = rewriter.getContext();
    StringRef fnName = "__quantum__qis__m__body";
    Type qirSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMPointerType::get(ctx), LLVM::LLVMPointerType::get(ctx));
    LLVM::LLVMFuncOp fnDecl =
        ensureFunctionDeclaration(rewriter, op, fnName, qirSignature);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl,
                                              adaptor.getInQubits());

    return success();
  }
};

struct MQTDynToQIR final : impl::MQTDynToQIRBase<MQTDynToQIR> {
  using MQTDynToQIRBase::MQTDynToQIRBase;
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();
    assert(context->getLoadedDialect<mlir::LLVM::LLVMDialect>() &&
           "LLVM dialect smust be loaded in context!");
    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    MQTDynToQIRTypeConverter typeConverter(context);

    target.addIllegalDialect<dyn::MQTDynDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();
    patterns.add<ConvertMQTDynAllocQIR>(typeConverter, context);
    patterns.add<ConvertMQTDynDeallocQIR>(typeConverter, context);
    patterns.add<ConvertMQTDynExtractQIR>(typeConverter, context);
    patterns.add<ConvertMQTDynMeasureQIR>(typeConverter, context);
    patterns.add<ConvertMQTDynGateOpQIR<dyn::HOp>>(typeConverter, context);
    patterns.add<ConvertMQTDynGateOpQIR<dyn::XOp>>(typeConverter, context);
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  };
};

} // namespace mqt::ir
