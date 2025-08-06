/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

// macro to add the conversion pattern from any dyn gate operation to a llvm
// call operation that adheres to the qir specification

#define ADD_CONVERT_PATTERN(gate)                                              \
  mqtPatterns.add<ConvertMQTDynGateOpQIR<dyn::gate>>(typeConverter, context);

#include "mlir/Conversion/MQTDynToQIR/MQTDynToQIR.h"

#include "mlir/Dialect/MQTDyn/IR/MQTDynDialect.h"

#include <iterator>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/LLVMIR/FunctionCallUtils.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <string>
#include <utility>
namespace mqt::ir {

using namespace mlir;

#define GEN_PASS_DEF_MQTDYNTOQIR
#include "mlir/Conversion/MQTDynToQIR/MQTDynToQIR.h.inc"

namespace {
// add function declaration at the end if it does not exist already and
// return the function declaration
LLVM::LLVMFuncOp getFunctionDeclaration(PatternRewriter& rewriter,
                                        Operation* op, StringRef fnName,
                                        Type fnType) {
  auto* fnDecl =
      SymbolTable::lookupNearestSymbolFrom(op, rewriter.getStringAttr(fnName));

  if (fnDecl == nullptr) {
    const PatternRewriter::InsertionGuard insertGuard(rewriter);
    auto module = op->getParentOfType<ModuleOp>();
    rewriter.setInsertionPointToEnd(module.getBody());

    fnDecl = rewriter.create<LLVM::LLVMFuncOp>(op->getLoc(), fnName, fnType);
  }

  return cast<LLVM::LLVMFuncOp>(fnDecl);
}

} // namespace

struct MQTDynToQIRTypeConverter final : public LLVMTypeConverter {
  explicit MQTDynToQIRTypeConverter(MLIRContext* ctx) : LLVMTypeConverter(ctx) {
    // QubitType conversion
    addConversion([ctx](dyn::QubitType /*type*/) {
      return LLVM::LLVMPointerType::get(ctx);
    });
    // QregType Conversion
    addConversion([ctx](dyn::QubitRegisterType /*type*/) {
      return LLVM::LLVMPointerType::get(ctx);
    });
  }
};

struct ConvertMQTDynAllocQIR final : OpConversionPattern<dyn::AllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(dyn::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = getContext();

    // create name and signature of the new function
    const StringRef fnName = "__quantum__rt__qubit_allocate_array";
    const auto qirSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMPointerType::get(ctx), IntegerType::get(ctx, 64));

    // get the function declaration
    const auto fnDecl =
        getFunctionDeclaration(rewriter, op, fnName, qirSignature);

    // create a constantOp if the size is an attribute
    auto size = adaptor.getSize();
    if (!size) {
      size =
          rewriter.create<LLVM::ConstantOp>(op.getLoc(), op.getSizeAttrAttr());
    }

    // replace the old operation with new callOp
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, size);
    return success();
  }
};
struct ConvertMQTDynDeallocQIR final : OpConversionPattern<dyn::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(dyn::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = getContext();

    // create name and signature of the new function
    const StringRef fnName = "__quantum__rt__qubit_release_array";
    const auto qirSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx), LLVM::LLVMPointerType::get(ctx));

    // get the function declaration
    const auto fnDecl =
        getFunctionDeclaration(rewriter, op, fnName, qirSignature);

    // replace the old operation with new callOp
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
    auto* ctx = getContext();

    // create name and signature of the new function
    const StringRef fnName = "__quantum__rt__array_get_element_ptr_1d";
    const auto qirSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMPointerType::get(ctx),
        {LLVM::LLVMPointerType::get(ctx), IntegerType::get(ctx, 64)});

    // get the function declaration
    const auto fnDecl =
        getFunctionDeclaration(rewriter, op, fnName, qirSignature);

    // create a constantOp if the index is an attribute
    auto index = adaptor.getIndex();
    if (!index) {
      index =
          rewriter.create<LLVM::ConstantOp>(op.getLoc(), op.getIndexAttrAttr());
    }

    // create the new callOp
    const auto elemPtr =
        rewriter
            .create<LLVM::CallOp>(op.getLoc(), fnDecl,
                                  ValueRange{adaptor.getInQreg(), index})
            .getResult();

    // replace the old operation with a loadOp
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
        op, LLVM::LLVMPointerType::get(ctx), elemPtr);

    return success();
  }
};

template <typename MQTGateDynOp>
struct ConvertMQTDynGateOpQIR final : OpConversionPattern<MQTGateDynOp> {
  using OpConversionPattern<MQTGateDynOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MQTGateDynOp op, typename MQTGateDynOp::Adaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = rewriter.getContext();

    // get all the values
    const auto& params = adaptor.getParams();
    const auto& inQubits = adaptor.getInQubits();
    const auto& posCtrlQubits = adaptor.getPosCtrlInQubits();
    const auto& negCtrlQubits = adaptor.getNegCtrlInQubits();

    // concatenate all the types
    SmallVector<Type> types;
    types.reserve(params.size() + inQubits.size() + posCtrlQubits.size() +
                  negCtrlQubits.size());
    types.append(params.getTypes().begin(), params.getTypes().end());
    types.append(inQubits.getTypes().begin(), inQubits.getTypes().end());
    types.append(posCtrlQubits.getTypes().begin(),
                 posCtrlQubits.getTypes().end());
    types.append(negCtrlQubits.getTypes().begin(),
                 negCtrlQubits.getTypes().end());

    // concatenate all the values
    SmallVector<Value> operands;
    operands.reserve(params.size() + inQubits.size() + posCtrlQubits.size() +
                     negCtrlQubits.size());
    operands.append(params.begin(), params.end());
    operands.append(inQubits.begin(), inQubits.end());
    operands.append(posCtrlQubits.begin(), posCtrlQubits.end());
    operands.append(negCtrlQubits.begin(), negCtrlQubits.end());

    // get the name of the gate
    const StringRef name = op->getName().getStringRef().split('.').second;
    std::string fnName;

    // add leading c's depending on the number of control qubits
    const auto ctrQubitsCount = posCtrlQubits.size() + negCtrlQubits.size();
    fnName.insert(0, ctrQubitsCount, 'c');

    // check if it is a cnot gate
    if (name == "x" && ctrQubitsCount == 1) {
      fnName = ("__quantum__qis__" + fnName + "not__body");
    } else {
      fnName = ("__quantum__qis__" + fnName + name + "__body").str();
    }

    // create the signature of the function
    const auto qirSignature =
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), types);

    // get the function declaration
    const auto fnDecl =
        getFunctionDeclaration(rewriter, op, fnName, qirSignature);

    // replace the old operation with a callOp
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl, operands);

    return success();
  }
};

struct ConvertMQTDynMeasureQIR final : OpConversionPattern<dyn::MeasureOp> {
  using OpConversionPattern::OpConversionPattern;

  SmallVector<Operation*>* measureConstants;
  Operation* constantOp;
  explicit ConvertMQTDynMeasureQIR(TypeConverter& typeConverter,
                                   MLIRContext* context,
                                   SmallVector<Operation*>& measureConstants,
                                   Operation* constantOp)
      : OpConversionPattern(typeConverter, context),
        measureConstants(&measureConstants), constantOp(constantOp) {}
  LogicalResult
  matchAndRewrite(dyn::MeasureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto* ctx = rewriter.getContext();

    // create name and signature of the new function
    StringRef fnName = "__quantum__qis__m__body";
    auto qirSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMPointerType::get(ctx), LLVM::LLVMPointerType::get(ctx));

    // get the function declaration
    auto fnDecl = getFunctionDeclaration(rewriter, op, fnName, qirSignature);

    // replace the old operation with new callOp
    auto newOp = rewriter.create<LLVM::CallOp>(op->getLoc(), fnDecl,
                                               adaptor.getInQubits());

    // create record result output
    fnName = "__quantum__rt__result_record_output";
    qirSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx),
        {LLVM::LLVMPointerType::get(ctx), LLVM::LLVMPointerType::get(ctx)});
    fnDecl = getFunctionDeclaration(rewriter, op, fnName, qirSignature);

    rewriter.create<LLVM::CallOp>(
        op.getLoc(), fnDecl,
        ValueRange{newOp->getResult(0),
                   measureConstants->front()->getResult(0)});
    measureConstants->erase(&measureConstants->front());

    // create record update reference count
    fnName = "__quantum__rt__result_update_reference_count";
    qirSignature = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx),
        {LLVM::LLVMPointerType::get(ctx), IntegerType::get(ctx, 32)});
    fnDecl = getFunctionDeclaration(rewriter, op, fnName, qirSignature);
    rewriter.create<LLVM::CallOp>(
        op.getLoc(), fnDecl,
        ValueRange{newOp->getResult(0), constantOp->getResult(0)});

    // create read result op
    fnName = "__quantum__rt__read_result";
    qirSignature = LLVM::LLVMFunctionType::get(
        IntegerType::get(ctx, 1), {LLVM::LLVMPointerType::get(ctx)});
    fnDecl = getFunctionDeclaration(rewriter, op, fnName, qirSignature);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fnDecl,
                                              ValueRange{newOp->getResult(0)});
    return success();
  }
};

struct MQTDynToQIR final : impl::MQTDynToQIRBase<MQTDynToQIR> {
  using MQTDynToQIRBase::MQTDynToQIRBase;

  static LLVM::LLVMFuncOp getMainFunction(Operation* op) {
    auto module = llvm::dyn_cast<ModuleOp>(op);
    // find the main function
    for (auto funcOp : module.getOps<LLVM::LLVMFuncOp>()) {
      auto passthrough = funcOp->getAttrOfType<ArrayAttr>("passthrough");
      if (!passthrough) {
        continue;
      }
      if (llvm::any_of(passthrough, [](Attribute attr) {
            auto strAttr = dyn_cast<StringAttr>(attr);
            return strAttr && strAttr.getValue() == "entry_point";
          })) {
        return funcOp;
      }
    }
    return nullptr;
  }
  // collect all the necessary operations for the measure operation conversion
  static Operation* collectMeasureConstants(Operation* op,
                                            SmallVector<Operation*>& operations,
                                            MLIRContext* ctx) {

    LLVM::LLVMFuncOp main = getMainFunction(op);

    // get the first block in the main function
    auto& firstBlock = *(main.getBlocks().begin());
    Operation* result = nullptr;
    // walk through the block and collect all addressOfOp and get the -1
    // constant value
    firstBlock.walk([&](Operation* op) {
      if (auto addressOfOp = llvm::dyn_cast<LLVM::AddressOfOp>(op)) {
        operations.emplace_back(addressOfOp);
      }

      if (auto constantOp = llvm::dyn_cast<LLVM::ConstantOp>(op)) {
        if (auto intAttr = dyn_cast<IntegerAttr>(constantOp.getValue())) {
          // check if the value is -1
          if (intAttr.getInt() == -1) {
            result = op;
          }
        }
      }
    });
    // create constant -1 value if there is none
    if (result == nullptr) {
      OpBuilder builder(main.getBody());
      builder.setInsertionPointToStart(&firstBlock);
      result = builder.create<LLVM::ConstantOp>(main->getLoc(),
                                                IntegerType::get(ctx, 32),
                                                builder.getI32IntegerAttr(-1));
    }
    return result;
  }

  // create the quantum initialize op
  static void addInitialize(Operation* op, MLIRContext* ctx) {
    auto module = llvm::dyn_cast<ModuleOp>(op);
    LLVM::LLVMFuncOp main = getMainFunction(op);

    auto& firstBlock = *(main.getBlocks().begin());
    OpBuilder builder(main.getBody());
    Operation* zeroOperation = nullptr;
    // find the zeroOp or create one
    firstBlock.walk([&](Operation* op) {
      if (auto zeroOp = llvm::dyn_cast<LLVM::ZeroOp>(op)) {
        zeroOperation = zeroOp;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (zeroOperation == nullptr) {
      builder.setInsertionPointToStart(&firstBlock);
      zeroOperation = builder.create<LLVM::ZeroOp>(
          main->getLoc(), LLVM::LLVMPointerType::get(ctx));
    }

    // set the builder to the 2nd last operation in the first block
    auto& ops = firstBlock.getOperations();
    const auto insertPoint = std::prev(ops.end(), 1);
    builder.setInsertionPoint(&*insertPoint);

    // get the function declaration of initialize otherwise create one
    const StringRef fnName = "__quantum__rt__initialize";
    auto* fnDecl =
        SymbolTable::lookupNearestSymbolFrom(op, builder.getStringAttr(fnName));
    if (fnDecl == nullptr) {
      const PatternRewriter::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToEnd(module.getBody());
      auto fnSignature = LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(ctx), LLVM::LLVMPointerType::get(ctx));
      fnDecl =
          builder.create<LLVM::LLVMFuncOp>(op->getLoc(), fnName, fnSignature);
    }
    // create and insert the initialize operation
    builder.create<LLVM::CallOp>(op->getLoc(),
                                 static_cast<LLVM::LLVMFuncOp>(fnDecl),
                                 ValueRange{zeroOperation->getResult(0)});
  }

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();
    ConversionTarget target(*context);
    RewritePatternSet stdPatterns(context);
    RewritePatternSet mqtPatterns(context);
    MQTDynToQIRTypeConverter typeConverter(context);

    // transform the default dialects first
    // maybe need to add more?
    target.addLegalDialect<LLVM::LLVMDialect>();
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, stdPatterns);
    populateFuncToLLVMConversionPatterns(typeConverter, stdPatterns);
    arith::populateArithToLLVMConversionPatterns(typeConverter, stdPatterns);
    target.addIllegalDialect<arith::ArithDialect>();
    target.addIllegalDialect<func::FuncDialect>();
    target.addIllegalDialect<cf::ControlFlowDialect>();
    if (failed(
            applyPartialConversion(module, target, std::move(stdPatterns)))) {
      signalPassFailure();
    }
    target.addIllegalDialect<dyn::MQTDynDialect>();

    SmallVector<Operation*> addressOfOps;
    auto* constantOp = collectMeasureConstants(module, addressOfOps, context);
    addInitialize(module, context);

    mqtPatterns.add<ConvertMQTDynAllocQIR>(typeConverter, context);
    mqtPatterns.add<ConvertMQTDynDeallocQIR>(typeConverter, context);
    mqtPatterns.add<ConvertMQTDynExtractQIR>(typeConverter, context);
    mqtPatterns.add<ConvertMQTDynMeasureQIR>(typeConverter, context,
                                             addressOfOps, constantOp);

    ADD_CONVERT_PATTERN(GPhaseOp)
    ADD_CONVERT_PATTERN(IOp)
    ADD_CONVERT_PATTERN(BarrierOp)
    ADD_CONVERT_PATTERN(HOp)
    ADD_CONVERT_PATTERN(XOp)
    ADD_CONVERT_PATTERN(YOp)
    ADD_CONVERT_PATTERN(ZOp)
    ADD_CONVERT_PATTERN(SOp)
    ADD_CONVERT_PATTERN(SdgOp)
    ADD_CONVERT_PATTERN(TOp)
    ADD_CONVERT_PATTERN(TdgOp)
    ADD_CONVERT_PATTERN(VOp)
    ADD_CONVERT_PATTERN(VdgOp)
    ADD_CONVERT_PATTERN(UOp)
    ADD_CONVERT_PATTERN(U2Op)
    ADD_CONVERT_PATTERN(POp)
    ADD_CONVERT_PATTERN(SXOp)
    ADD_CONVERT_PATTERN(SXdgOp)
    ADD_CONVERT_PATTERN(RXOp)
    ADD_CONVERT_PATTERN(RYOp)
    ADD_CONVERT_PATTERN(RZOp)
    ADD_CONVERT_PATTERN(SWAPOp)
    ADD_CONVERT_PATTERN(iSWAPOp)
    ADD_CONVERT_PATTERN(iSWAPdgOp)
    ADD_CONVERT_PATTERN(PeresOp)
    ADD_CONVERT_PATTERN(PeresdgOp)
    ADD_CONVERT_PATTERN(DCXOp)
    ADD_CONVERT_PATTERN(ECROp)
    ADD_CONVERT_PATTERN(RXXOp)
    ADD_CONVERT_PATTERN(RYYOp)
    ADD_CONVERT_PATTERN(RZZOp)
    ADD_CONVERT_PATTERN(RZXOp)
    ADD_CONVERT_PATTERN(XXminusYY)
    ADD_CONVERT_PATTERN(XXplusYY)

    if (failed(
            applyPartialConversion(module, target, std::move(mqtPatterns)))) {
      signalPassFailure();
    }
  };
};

} // namespace mqt::ir
